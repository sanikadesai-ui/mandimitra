#!/usr/bin/env python3
"""
MANDIMITRA - Historical Mandi Data Import (Local File)

Imports historical mandi data from a local CSV/ZIP file provided by the user.
Processes with chunked reading for memory safety and filters to Maharashtra-only.

This is the "bring your own data" option for users who:
- Have downloaded historical data manually
- Have data from another source
- Cannot use Kaggle API

Usage:
    python scripts/import_mandi_history.py --input-file /path/to/data.csv
    python scripts/import_mandi_history.py --input-file /path/to/data.zip
    python scripts/import_mandi_history.py --input-file /path/to/data.csv.gz
    python scripts/import_mandi_history.py --help

Output:
    data/processed/mandi/history_maharashtra.parquet  # Cleaned Maharashtra-only data

⚠️  HARD CONSTRAINT: Maharashtra-only data.
"""

import argparse
import gzip
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import ensure_directory, load_config, save_receipt
from src.utils.logging_utils import setup_logger, get_utc_timestamp_safe
from src.utils.maharashtra import MAHARASHTRA_STATE_NAME, is_maharashtra_state
from src.utils.audit import AuditLogger


# Lazy import schemas
def get_mandi_schema():
    from src.schemas.mandi import validate_mandi_dataframe, summarize_mandi_data, normalize_columns, parse_dates
    return validate_mandi_dataframe, summarize_mandi_data, normalize_columns, parse_dates


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import historical mandi data from local file (Maharashtra-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SUPPORTED FORMATS:
  - .csv      Plain CSV file
  - .csv.gz   Gzip-compressed CSV
  - .zip      ZIP archive containing CSV(s)
  - .parquet  Apache Parquet file

EXPECTED COLUMNS (flexible naming):
  - state, district, market, commodity, variety, grade
  - arrival_date (date of price observation)
  - min_price, max_price, modal_price

The script will auto-detect column name variations (e.g., "State", "STATE", 
"state_name" all map to "state").

Examples:
    # Import from CSV
    python scripts/import_mandi_history.py --input-file data/external/agmarknet_2010_2023.csv
    
    # Import from compressed file
    python scripts/import_mandi_history.py --input-file data/external/historical.csv.gz
    
    # Import from ZIP (extracts and processes all CSVs)
    python scripts/import_mandi_history.py --input-file data/external/agmarknet_data.zip
    
    # Specify output format
    python scripts/import_mandi_history.py --input-file data.csv --output-format csv
    
    # Custom chunk size for low-memory systems
    python scripts/import_mandi_history.py --input-file data.csv --chunk-size 25000
        """,
    )
    
    # Required input
    parser.add_argument(
        "--input-file", "-i",
        type=str,
        required=True,
        help="Path to input file (CSV, CSV.GZ, ZIP, or Parquet)",
    )
    
    # Processing settings
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows per chunk for memory-safe processing (default: 100000)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output file path (default: data/processed/mandi/history_maharashtra.parquet)",
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources configuration",
    )
    
    # Validation options
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Strict mode: fail if non-Maharashtra data found (default: True)",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Lenient mode: drop non-Maharashtra data silently",
    )
    
    # Options
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview first 1000 rows without full processing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


# =============================================================================
# FILE HANDLING
# =============================================================================

def get_csv_files_from_path(
    input_path: Path,
    logger: logging.Logger,
) -> List[Path]:
    """
    Extract/identify CSV files from input path.
    
    Handles:
    - Plain CSV files
    - Gzip-compressed CSV (.csv.gz)
    - ZIP archives
    - Parquet files
    
    Args:
        input_path: Path to input file
        logger: Logger instance
        
    Returns:
        List of CSV file paths (may be in temp directory for extracted files)
    """
    suffix = input_path.suffix.lower()
    suffixes = "".join(input_path.suffixes).lower()
    
    if suffix == ".parquet":
        # Parquet handled separately
        return [input_path]
    
    if suffix == ".csv" or suffixes == ".csv.gz":
        return [input_path]
    
    if suffix == ".zip":
        # Extract to temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="mandimitra_import_"))
        logger.info(f"Extracting ZIP to: {temp_dir}")
        
        with zipfile.ZipFile(input_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Find all CSVs in extracted content
        csv_files = list(temp_dir.glob("**/*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in ZIP: {input_path}")
        
        logger.info(f"Extracted {len(csv_files)} CSV files")
        return csv_files
    
    if suffix == ".gz":
        # Assume gzipped CSV
        return [input_path]
    
    # Try as CSV anyway
    return [input_path]


def stream_csv_chunks(
    filepath: Path,
    chunk_size: int,
    logger: logging.Logger,
) -> Iterator[pd.DataFrame]:
    """
    Stream CSV file in chunks for memory-safe processing.
    
    Args:
        filepath: Path to CSV file
        chunk_size: Rows per chunk
        logger: Logger instance
        
    Yields:
        DataFrame chunks
    """
    suffix = filepath.suffix.lower()
    suffixes = "".join(filepath.suffixes).lower()
    
    logger.info(f"Streaming: {filepath.name} (chunk_size={chunk_size:,})")
    
    # Determine compression
    compression = None
    if suffix == ".gz" or suffixes == ".csv.gz":
        compression = "gzip"
    elif suffix == ".bz2":
        compression = "bz2"
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(
            filepath,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8",
            compression=compression,
            on_bad_lines="skip",
        )):
            yield chunk
            
    except UnicodeDecodeError:
        # Try alternate encoding
        logger.warning(f"UTF-8 failed, trying latin-1: {filepath.name}")
        for chunk in pd.read_csv(
            filepath,
            chunksize=chunk_size,
            low_memory=False,
            encoding="latin-1",
            compression=compression,
            on_bad_lines="skip",
        ):
            yield chunk


def read_parquet_file(filepath: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Read Parquet file.
    
    Args:
        filepath: Path to Parquet file
        logger: Logger instance
        
    Returns:
        DataFrame
    """
    logger.info(f"Reading Parquet: {filepath.name}")
    return pd.read_parquet(filepath)


# =============================================================================
# PROCESSING
# =============================================================================

def process_chunk(
    chunk: pd.DataFrame,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Process a single chunk: normalize, filter to Maharashtra.
    
    Args:
        chunk: Raw DataFrame chunk
        logger: Logger instance
        
    Returns:
        (filtered_chunk, original_rows, non_mh_dropped)
    """
    validate_mandi_dataframe, summarize_mandi_data, normalize_columns, parse_dates = get_mandi_schema()
    
    original_rows = len(chunk)
    
    # Normalize columns
    chunk = normalize_columns(chunk)
    
    # Parse dates
    chunk = parse_dates(chunk)
    
    # Check for state column
    if "state" not in chunk.columns:
        return pd.DataFrame(), original_rows, original_rows
    
    # Filter to Maharashtra only
    mh_mask = chunk["state"].apply(is_maharashtra_state)
    non_mh_count = (~mh_mask).sum()
    
    filtered = chunk[mh_mask].copy()
    
    return filtered, original_rows, non_mh_count


def process_files(
    csv_files: List[Path],
    chunk_size: int,
    output_path: Path,
    output_format: str,
    strict: bool,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Process input files with chunked reading.
    
    Args:
        csv_files: List of file paths
        chunk_size: Rows per chunk
        output_path: Output file path
        output_format: "parquet" or "csv"
        strict: Strict Maharashtra validation
        logger: Logger instance
        
    Returns:
        Processing statistics
    """
    validate_mandi_dataframe, summarize_mandi_data, normalize_columns, parse_dates = get_mandi_schema()
    
    stats = {
        "files_processed": 0,
        "total_rows_read": 0,
        "maharashtra_rows": 0,
        "non_mh_dropped": 0,
        "non_mh_samples": {},
        "chunks_processed": 0,
        "errors": [],
    }
    
    # Ensure output directory exists
    ensure_directory(output_path.parent)
    
    # Process files and accumulate Maharashtra data
    all_mh_chunks: List[pd.DataFrame] = []
    
    for filepath in csv_files:
        logger.info(f"Processing: {filepath.name}")
        file_rows = 0
        file_mh = 0
        
        try:
            # Handle Parquet differently
            if filepath.suffix.lower() == ".parquet":
                chunk = read_parquet_file(filepath, logger)
                filtered, original, non_mh = process_chunk(chunk, logger)
                
                stats["total_rows_read"] += original
                stats["non_mh_dropped"] += non_mh
                stats["chunks_processed"] += 1
                
                if not filtered.empty:
                    all_mh_chunks.append(filtered)
                    file_mh = len(filtered)
                file_rows = original
                
            else:
                # Stream CSV
                for chunk in stream_csv_chunks(filepath, chunk_size, logger):
                    filtered, original, non_mh = process_chunk(chunk, logger)
                    
                    stats["total_rows_read"] += original
                    stats["non_mh_dropped"] += non_mh
                    stats["chunks_processed"] += 1
                    file_rows += original
                    
                    if not filtered.empty:
                        all_mh_chunks.append(filtered)
                        file_mh += len(filtered)
                    
                    # Progress update
                    if stats["chunks_processed"] % 10 == 0:
                        logger.info(
                            f"  Progress: {stats['chunks_processed']} chunks, "
                            f"{stats['total_rows_read']:,} rows read"
                        )
            
            stats["files_processed"] += 1
            logger.info(f"  {filepath.name}: {file_rows:,} rows → {file_mh:,} Maharashtra")
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            stats["errors"].append({"file": str(filepath), "error": str(e)})
    
    # Check strict mode
    if strict and stats["non_mh_dropped"] > 0:
        logger.warning(
            f"Non-Maharashtra records found: {stats['non_mh_dropped']:,}. "
            f"Strict mode is ON - these will be dropped."
        )
    
    # Concatenate and save
    if all_mh_chunks:
        logger.info(f"Concatenating {len(all_mh_chunks)} chunks...")
        df = pd.concat(all_mh_chunks, ignore_index=True)
        
        # Validate final dataset
        logger.info("Validating final dataset...")
        try:
            df, validation_report = validate_mandi_dataframe(df, strict=False, normalize=True)
            stats["validation_report"] = validation_report
        except Exception as e:
            logger.warning(f"Validation warning: {e}")
        
        # Deduplicate
        dedup_cols = ["state", "district", "market", "commodity", "variety", "grade", "arrival_date"]
        dedup_cols = [c for c in dedup_cols if c in df.columns]
        rows_before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
        stats["duplicates_removed"] = rows_before - len(df)
        
        stats["maharashtra_rows"] = len(df)
        
        # Save
        logger.info(f"Saving {len(df):,} rows to {output_path}")
        
        if output_format == "parquet":
            df.to_parquet(output_path, index=False, compression="snappy")
        else:
            df.to_csv(output_path, index=False)
        
        # Generate summary
        stats["summary"] = summarize_mandi_data(df)
        
    else:
        logger.warning("No Maharashtra data found!")
        stats["maharashtra_rows"] = 0
    
    return stats


def preview_file(
    input_path: Path,
    logger: logging.Logger,
    n_rows: int = 1000,
) -> None:
    """
    Preview first N rows of input file.
    
    Args:
        input_path: Path to input file
        logger: Logger instance
        n_rows: Number of rows to preview
    """
    validate_mandi_dataframe, summarize_mandi_data, normalize_columns, parse_dates = get_mandi_schema()
    
    logger.info(f"Previewing first {n_rows} rows of: {input_path}")
    
    csv_files = get_csv_files_from_path(input_path, logger)
    
    for filepath in csv_files[:1]:  # Preview first file only
        if filepath.suffix.lower() == ".parquet":
            df = pd.read_parquet(filepath).head(n_rows)
        else:
            df = pd.read_csv(filepath, nrows=n_rows)
        
        print("\n" + "=" * 70)
        print(f"FILE: {filepath.name}")
        print("=" * 70)
        
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}: {df[col].dtype}")
        
        # Normalize and check
        df_norm = normalize_columns(df)
        print(f"\nAfter normalization:")
        for col in df_norm.columns:
            print(f"  - {col}: {df_norm[col].dtype}")
        
        # Check state distribution
        if "state" in df_norm.columns:
            print(f"\nState distribution:")
            print(df_norm["state"].value_counts().head(10))
            
            mh_count = df_norm["state"].apply(is_maharashtra_state).sum()
            print(f"\nMaharashtra rows: {mh_count} / {len(df_norm)} ({100*mh_count/len(df_norm):.1f}%)")
        
        # Sample data
        print(f"\nSample data:")
        print(df_norm.head(5).to_string())


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "import_mandi.log"
    logger = setup_logger("import_mandi", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("mandi_history_import", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Local Mandi Data Import")
    logger.info("=" * 70)
    
    try:
        # Validate input file
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        
        logger.info(f"Input file: {input_path}")
        logger.info(f"File size: {input_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Preview mode
        if args.preview:
            preview_file(input_path, logger)
            return 0
        
        # Load config for defaults
        config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            config = load_config(config_path)
        else:
            config = {}
        
        # Determine output path
        output_path = args.output_path
        if not output_path:
            output_path = config.get("mandi", {}).get("historical", {}).get("processing", {}).get("output_path")
        if not output_path:
            ext = ".parquet" if args.output_format == "parquet" else ".csv"
            output_path = f"data/processed/mandi/history_maharashtra{ext}"
        output_path = PROJECT_ROOT / output_path
        
        audit.add_section("Configuration", {
            "input_file": str(input_path),
            "output_path": str(output_path),
            "chunk_size": args.chunk_size,
            "output_format": args.output_format,
            "strict": args.strict,
        })
        
        # Get CSV files
        csv_files = get_csv_files_from_path(input_path, logger)
        logger.info(f"Files to process: {len(csv_files)}")
        
        # Process
        stats = process_files(
            csv_files,
            args.chunk_size,
            output_path,
            args.output_format,
            args.strict,
            logger,
        )
        
        audit.add_section("Processing Results", stats)
        
        # Save receipt
        receipt = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_file": str(input_path),
            "files_processed": stats["files_processed"],
            "total_rows_read": stats["total_rows_read"],
            "maharashtra_rows": stats["maharashtra_rows"],
            "non_mh_dropped": stats["non_mh_dropped"],
            "duplicates_removed": stats.get("duplicates_removed", 0),
            "output_path": str(output_path),
            "summary": stats.get("summary", {}),
        }
        
        receipt_path = output_path.parent / f"import_receipt_{timestamp}.json"
        save_receipt(receipt_path, receipt)
        
        # Summary
        print(f"\n✅ Mandi Data Import Complete!")
        print(f"   📁 Input: {input_path}")
        print(f"   📁 Output: {output_path}")
        print(f"   📊 Rows: {stats['maharashtra_rows']:,}")
        print(f"   ⚠️  Non-MH dropped: {stats['non_mh_dropped']:,}")
        if stats.get("duplicates_removed"):
            print(f"   🔄 Duplicates removed: {stats['duplicates_removed']:,}")
        if stats.get("summary", {}).get("date_range"):
            dr = stats["summary"]["date_range"]
            print(f"   📅 Date range: {dr['min']} to {dr['max']}")
        
        audit_path = audit.save()
        print(f"   📋 Audit: {audit_path}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
