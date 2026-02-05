#!/usr/bin/env python3
"""
MANDIMITRA - Historical Mandi Data Downloader (Kaggle)

Downloads historical AGMARKNET mandi price data from Kaggle datasets.
Processes with chunked reading for memory safety and filters to Maharashtra-only.

Prerequisites:
    - Kaggle API credentials (KAGGLE_USERNAME + KAGGLE_KEY in .env, or kaggle.json)
    - pip install kaggle

Usage:
    python scripts/download_mandi_history_kaggle.py --download
    python scripts/download_mandi_history_kaggle.py --download --dataset "owner/dataset-slug"
    python scripts/download_mandi_history_kaggle.py --process-only  # Process existing downloads
    python scripts/download_mandi_history_kaggle.py --help

Output:
    data/raw/mandi/kaggle_download/           # Raw downloaded files
    data/processed/mandi/history_maharashtra.parquet  # Cleaned Maharashtra-only data

⚠️  HARD CONSTRAINT: Maharashtra-only data.
"""

import argparse
import glob
import json
import logging
import os
import sys
import tempfile
import time
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

# Lazy import schemas (may not be needed if just downloading)
def get_mandi_schema():
    from src.schemas.mandi import validate_mandi_dataframe, summarize_mandi_data, normalize_columns, parse_dates
    return validate_mandi_dataframe, summarize_mandi_data, normalize_columns, parse_dates


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download historical mandi data from Kaggle (Maharashtra-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PREREQUISITES:
  1. Kaggle account with API access enabled
  2. Set credentials via:
     - Environment variables: KAGGLE_USERNAME, KAGGLE_KEY
     - OR: Place kaggle.json in ~/.kaggle/
  3. pip install kaggle

WORKFLOW:
  1. --download: Download dataset from Kaggle
  2. Automatic: Extract ZIP files
  3. Automatic: Process CSVs with chunked reading
  4. Automatic: Filter to Maharashtra-only
  5. Output: Cleaned parquet file

Examples:
    # Download default dataset
    python scripts/download_mandi_history_kaggle.py --download
    
    # Download specific dataset
    python scripts/download_mandi_history_kaggle.py --download --dataset "ramjasmaurya/daily-commodity-prices-india-2003-2021"
    
    # Process already-downloaded files
    python scripts/download_mandi_history_kaggle.py --process-only
    
    # Specify chunk size for memory-constrained systems
    python scripts/download_mandi_history_kaggle.py --download --chunk-size 50000
        """,
    )
    
    # Actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Kaggle and process",
    )
    action_group.add_argument(
        "--process-only",
        action="store_true",
        help="Process existing downloaded files (skip download)",
    )
    
    # Kaggle settings
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Kaggle dataset slug (owner/dataset-name). Default from config.",
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
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources configuration",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="data/raw/mandi/kaggle_download",
        help="Directory for Kaggle downloads",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output file path (default from config)",
    )
    
    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        default=True,
        help="Keep raw downloaded files after processing (default: True)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


# =============================================================================
# KAGGLE OPERATIONS
# =============================================================================

def setup_kaggle_credentials(logger: logging.Logger) -> bool:
    """
    Setup Kaggle API credentials from environment.
    
    Credentials can be provided via:
    1. KAGGLE_USERNAME + KAGGLE_KEY environment variables
    2. ~/.kaggle/kaggle.json file
    
    Returns:
        True if credentials are configured
    """
    load_dotenv(PROJECT_ROOT / ".env")
    
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        logger.info("Kaggle credentials loaded from environment")
        return True
    
    # Check for kaggle.json
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        PROJECT_ROOT / "kaggle.json",
    ]
    
    for path in kaggle_json_paths:
        if path.exists():
            logger.info(f"Kaggle credentials found: {path}")
            return True
    
    return False


def download_kaggle_dataset(
    dataset_slug: str,
    output_dir: Path,
    logger: logging.Logger,
    force: bool = False,
) -> List[Path]:
    """
    Download a dataset from Kaggle.
    
    Args:
        dataset_slug: Kaggle dataset identifier (owner/dataset-name)
        output_dir: Directory to save files
        logger: Logger instance
        force: Force re-download
        
    Returns:
        List of downloaded file paths
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle package not installed. Run: pip install kaggle"
        )
    
    ensure_directory(output_dir)
    
    # Check for existing files
    existing_files = list(output_dir.glob("*"))
    if existing_files and not force:
        logger.info(f"Files already exist in {output_dir}. Use --force to re-download.")
        return existing_files
    
    logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        api.dataset_download_files(
            dataset_slug,
            path=str(output_dir),
            unzip=True,
        )
        
        # List downloaded files
        downloaded = list(output_dir.glob("**/*"))
        csv_files = [f for f in downloaded if f.suffix.lower() == ".csv"]
        zip_files = [f for f in downloaded if f.suffix.lower() == ".zip"]
        
        logger.info(f"Downloaded: {len(csv_files)} CSV files, {len(zip_files)} ZIP files")
        
        # Extract any remaining ZIP files
        for zip_path in zip_files:
            logger.info(f"Extracting: {zip_path.name}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(output_dir)
        
        # Return all CSV files
        all_csvs = list(output_dir.glob("**/*.csv"))
        return all_csvs
        
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        raise


def find_csv_files(directory: Path, logger: logging.Logger) -> List[Path]:
    """
    Find all CSV files in a directory (including extracted from ZIPs).
    
    Args:
        directory: Directory to search
        logger: Logger instance
        
    Returns:
        List of CSV file paths
    """
    csv_files = []
    
    # Direct CSV files
    csv_files.extend(directory.glob("*.csv"))
    csv_files.extend(directory.glob("**/*.csv"))
    
    # Extract ZIP files first
    zip_files = list(directory.glob("*.zip")) + list(directory.glob("**/*.zip"))
    for zip_path in zip_files:
        logger.info(f"Extracting ZIP: {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(directory)
        except Exception as e:
            logger.warning(f"Failed to extract {zip_path}: {e}")
    
    # Re-scan for CSVs after extraction
    csv_files = list(set(directory.glob("**/*.csv")))
    
    # Sort by size (process larger files first for better parallelism)
    csv_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    
    return csv_files


# =============================================================================
# CHUNKED PROCESSING (MEMORY SAFE)
# =============================================================================

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
    logger.info(f"Streaming: {filepath.name} (chunk_size={chunk_size:,})")
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(
            filepath,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8",
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
            on_bad_lines="skip",
        ):
            yield chunk


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


def process_csv_files(
    csv_files: List[Path],
    chunk_size: int,
    output_path: Path,
    output_format: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Process multiple CSV files with chunked reading.
    
    Args:
        csv_files: List of CSV file paths
        chunk_size: Rows per chunk
        output_path: Output file path
        output_format: "parquet" or "csv"
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
        "chunks_processed": 0,
        "errors": [],
    }
    
    # Ensure output directory exists
    ensure_directory(output_path.parent)
    
    # Process files and accumulate Maharashtra data
    all_mh_chunks: List[pd.DataFrame] = []
    
    for csv_file in csv_files:
        logger.info(f"Processing: {csv_file.name}")
        file_rows = 0
        file_mh = 0
        
        try:
            for chunk in stream_csv_chunks(csv_file, chunk_size, logger):
                filtered, original, non_mh = process_chunk(chunk, logger)
                
                stats["total_rows_read"] += original
                stats["non_mh_dropped"] += non_mh
                stats["chunks_processed"] += 1
                file_rows += original
                
                if not filtered.empty:
                    all_mh_chunks.append(filtered)
                    file_mh += len(filtered)
                
                # Progress update every 10 chunks
                if stats["chunks_processed"] % 10 == 0:
                    logger.info(
                        f"  Progress: {stats['chunks_processed']} chunks, "
                        f"{stats['total_rows_read']:,} rows read, "
                        f"{len(all_mh_chunks)} MH chunks accumulated"
                    )
            
            stats["files_processed"] += 1
            logger.info(f"  {csv_file.name}: {file_rows:,} rows → {file_mh:,} Maharashtra")
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")
            stats["errors"].append({"file": str(csv_file), "error": str(e)})
    
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "kaggle_download.log"
    logger = setup_logger("kaggle_mandi", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("mandi_history_kaggle", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Kaggle Historical Mandi Downloader")
    logger.info("=" * 70)
    
    try:
        # Load config
        config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            config = load_config(config_path)
            mandi_config = config.get("mandi", {}).get("historical", {}).get("kaggle", {})
        else:
            mandi_config = {}
            logger.warning(f"Config not found: {config_path}")
        
        # Determine dataset slug
        dataset_slug = args.dataset or mandi_config.get("dataset_slug")
        if not dataset_slug and args.download:
            logger.error("No dataset slug provided. Use --dataset or configure in data_sources.yaml")
            return 1
        
        # Paths
        download_dir = PROJECT_ROOT / args.download_dir
        
        output_path = args.output_path
        if not output_path:
            output_path = mandi_config.get("output_path") or config.get("mandi", {}).get("historical", {}).get("processing", {}).get("output_path")
        if not output_path:
            output_path = "data/processed/mandi/history_maharashtra.parquet"
        output_path = PROJECT_ROOT / output_path
        
        # Chunk size
        chunk_size = args.chunk_size or mandi_config.get("processing", {}).get("chunk_size", 100000)
        
        audit.add_section("Configuration", {
            "dataset_slug": dataset_slug,
            "download_dir": str(download_dir),
            "output_path": str(output_path),
            "chunk_size": chunk_size,
            "output_format": args.output_format,
        })
        
        csv_files = []
        
        # Download from Kaggle
        if args.download:
            # Check credentials
            if not setup_kaggle_credentials(logger):
                logger.error(
                    "Kaggle credentials not found. Either:\n"
                    "  1. Set KAGGLE_USERNAME and KAGGLE_KEY in .env\n"
                    "  2. Place kaggle.json in ~/.kaggle/"
                )
                return 1
            
            csv_files = download_kaggle_dataset(
                dataset_slug,
                download_dir,
                logger,
                force=args.force,
            )
        
        # Process existing files
        if args.process_only or args.download:
            if not csv_files:
                csv_files = find_csv_files(download_dir, logger)
            
            if not csv_files:
                logger.error(f"No CSV files found in {download_dir}")
                return 1
            
            logger.info(f"Found {len(csv_files)} CSV files to process")
            
            # Process
            stats = process_csv_files(
                csv_files,
                chunk_size,
                output_path,
                args.output_format,
                logger,
            )
            
            audit.add_section("Processing Results", stats)
            
            # Save receipt
            receipt = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_slug": dataset_slug,
                "files_processed": stats["files_processed"],
                "total_rows_read": stats["total_rows_read"],
                "maharashtra_rows": stats["maharashtra_rows"],
                "non_mh_dropped": stats["non_mh_dropped"],
                "duplicates_removed": stats.get("duplicates_removed", 0),
                "output_path": str(output_path),
                "summary": stats.get("summary", {}),
            }
            
            receipt_path = output_path.parent / f"history_receipt_{timestamp}.json"
            save_receipt(receipt_path, receipt)
            
            # Summary
            print(f"\n✅ Historical Mandi Data Processing Complete!")
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
