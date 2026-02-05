#!/usr/bin/env python3
"""
MANDIMITRA - Maharashtra Mandi Data Downloader

Downloads mandi price data EXCLUSIVELY for Maharashtra state.
Supports chunked downloads with resumability for large datasets.

⚠️  HARD CONSTRAINT: This script ONLY downloads Maharashtra data.
    Non-Maharashtra records are dropped and logged as violations.

Usage:
    python scripts/download_mandi_maharashtra.py --download-all
    python scripts/download_mandi_maharashtra.py --resume
    python scripts/download_mandi_maharashtra.py --force
    python scripts/download_mandi_maharashtra.py --help

Output:
    data/raw/mandi/maharashtra/<district>/<market>/<commodity>/<timestamp>/mandi.csv
    data/raw/mandi/maharashtra/merged/<timestamp>/mandi_maharashtra_merged.csv

Author: MANDIMITRA Team
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import (
    ensure_directory,
    load_config,
    save_dataframe,
    save_receipt,
    sanitize_filename,
)
from src.utils.http_utils import (
    APIError,
    APIKeyMissingError,
    EmptyResponseError,
    create_session,
    make_request,
)
from src.utils.logging_utils import (
    ProgressLogger,
    get_utc_timestamp,
    get_utc_timestamp_safe,
    setup_logger,
)
from src.utils.maharashtra import (
    MAHARASHTRA_STATE_NAME,
    is_maharashtra_state,
    validate_maharashtra_only,
)
from src.utils.progress import ProgressTracker, ChunkStatus
from src.utils.audit import AuditLogger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Maharashtra mandi price data (MAHARASHTRA ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  HARD CONSTRAINT: This script ONLY downloads Maharashtra data.
    The state filter is hardcoded and cannot be changed.
    Any non-Maharashtra records are dropped and flagged as violations.

Download Modes:
    --download-all    Download all Maharashtra mandi data
    --resume          Resume interrupted download (default behavior)
    --force           Restart download from scratch (ignore progress)

Strategy:
    The script automatically chooses between bulk and chunked download:
    - BULK: Single paginated download for smaller datasets (<500K rows)
    - CHUNKED: Download by district for larger datasets (with resumability)

Examples:
    # Full Maharashtra download (auto-selects strategy)
    python scripts/download_mandi_maharashtra.py --download-all
    
    # Resume interrupted download
    python scripts/download_mandi_maharashtra.py --resume
    
    # Force restart (discard previous progress)
    python scripts/download_mandi_maharashtra.py --force --download-all
    
    # Download specific district only
    python scripts/download_mandi_maharashtra.py --district "Pune"
        """,
    )
    
    # Download mode (NO state argument - Maharashtra is hardcoded!)
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all Maharashtra mandi data",
    )
    parser.add_argument(
        "--district",
        type=str,
        help="Download specific district only (must be in Maharashtra)",
    )
    parser.add_argument(
        "--commodity",
        type=str,
        help="Filter by commodity (optional)",
    )
    
    # Resumability
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume interrupted download (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart download from scratch (ignore previous progress)",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/mandi/maharashtra",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Records per API page (default: 1000)",
    )
    
    # Options
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip creating merged file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug output",
    )
    
    return parser.parse_args()


def get_api_key() -> str:
    """Get Data.gov.in API key from environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    
    api_key = os.getenv("DATAGOV_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        raise APIKeyMissingError(
            "DATAGOV_API_KEY not found or is placeholder.\n"
            "Set it in your .env file with a key from https://data.gov.in"
        )
    
    return api_key


def build_maharashtra_params(
    api_key: str,
    page_size: int,
    offset: int = 0,
    district: Optional[str] = None,
    commodity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build query parameters with HARDCODED Maharashtra filter.
    
    Args:
        api_key: API key
        page_size: Records per page
        offset: Pagination offset
        district: Optional district filter
        commodity: Optional commodity filter
        
    Returns:
        Query parameters dict
    """
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": page_size,
        "offset": offset,
        # HARDCODED: Maharashtra ONLY - this is a non-negotiable constraint
        "filters[state]": MAHARASHTRA_STATE_NAME,
    }
    
    if district:
        params["filters[district]"] = district
    if commodity:
        params["filters[commodity]"] = commodity
    
    return params


def fetch_total_count(
    session,
    api_url: str,
    api_key: str,
    district: Optional[str],
    commodity: Optional[str],
    logger,
) -> int:
    """Fetch total record count for Maharashtra with optional filters."""
    params = build_maharashtra_params(api_key, page_size=1, offset=0, district=district, commodity=commodity)
    
    logger.info(f"Fetching total count for Maharashtra" + 
                (f" / {district}" if district else "") +
                (f" / {commodity}" if commodity else "") + "...")
    
    data, _ = make_request(session, api_url, params=params, logger=logger)
    total = data.get("total", 0)
    
    logger.info(f"Total records: {total:,}")
    return total


def download_chunk(
    session,
    api_url: str,
    api_key: str,
    page_size: int,
    district: Optional[str],
    commodity: Optional[str],
    logger,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Download a chunk of Maharashtra data with pagination.
    
    Returns:
        Tuple of (DataFrame, total_pages, non_mh_count)
    """
    all_records = []
    non_mh_count = 0
    offset = 0
    page = 1
    
    # Get total first
    total = fetch_total_count(session, api_url, api_key, district, commodity, logger)
    
    if total == 0:
        return pd.DataFrame(), 0, 0
    
    chunk_name = f"Maharashtra" + (f"/{district}" if district else "") + (f"/{commodity}" if commodity else "")
    
    with ProgressLogger(logger, f"Downloading {chunk_name}") as progress:
        while True:
            params = build_maharashtra_params(api_key, page_size, offset, district, commodity)
            
            try:
                data, _ = make_request(session, api_url, params=params, logger=logger)
            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                raise
            
            records = data.get("records", [])
            
            if not records:
                break
            
            # HARD CONSTRAINT: Filter and count non-Maharashtra records
            mh_records = []
            for record in records:
                state = record.get("state", "")
                if is_maharashtra_state(state):
                    mh_records.append(record)
                else:
                    non_mh_count += 1
                    logger.warning(f"VIOLATION: Non-MH record dropped (state='{state}')")
            
            all_records.extend(mh_records)
            
            progress.update(f"Page {page}: {len(mh_records)}/{len(records)} MH records | Total: {len(all_records):,}")
            
            if len(records) < page_size or (offset + page_size) >= total:
                break
            
            offset += page_size
            page += 1
            
            # Rate limiting
            time.sleep(0.5)
    
    if not all_records:
        return pd.DataFrame(), page, non_mh_count
    
    df = pd.DataFrame(all_records)
    return df, page, non_mh_count


def get_districts_from_metadata(metadata_dir: Path) -> List[str]:
    """Load discovered districts from metadata."""
    districts_file = metadata_dir / "districts.csv"
    
    if not districts_file.exists():
        raise FileNotFoundError(
            f"Districts metadata not found at {districts_file}.\n"
            "Run discovery first: python scripts/discover_maharashtra_mandi_metadata.py"
        )
    
    df = pd.read_csv(districts_file)
    return df["district"].tolist()


def save_chunk_data(
    df: pd.DataFrame,
    output_dir: Path,
    district: Optional[str],
    commodity: Optional[str],
    timestamp: str,
    receipt_data: Dict[str, Any],
    logger,
) -> Path:
    """Save chunk data and receipt."""
    # Build path
    path_parts = [output_dir]
    
    if district:
        path_parts.append(sanitize_filename(district))
    else:
        path_parts.append("all_districts")
    
    if commodity:
        path_parts.append(sanitize_filename(commodity))
    else:
        path_parts.append("all_commodities")
    
    path_parts.append(timestamp)
    
    chunk_dir = Path(*[str(p) for p in path_parts])
    ensure_directory(chunk_dir)
    
    # Save CSV
    csv_path = chunk_dir / "mandi.csv"
    save_dataframe(df, csv_path)
    logger.info(f"Saved {len(df):,} rows to {csv_path}")
    
    # Save receipt
    receipt_path = chunk_dir / "receipt.json"
    save_receipt(receipt_path, receipt_data)
    
    return chunk_dir


def create_merged_file(
    output_dir: Path,
    timestamp: str,
    logger,
) -> Optional[Path]:
    """
    Create merged CSV from all downloaded chunks.
    
    Args:
        output_dir: Base output directory
        timestamp: Timestamp for merged file
        logger: Logger instance
        
    Returns:
        Path to merged file, or None if no data
    """
    logger.info("Creating merged Maharashtra dataset...")
    
    # Find all mandi.csv files
    csv_files = list(output_dir.glob("**/mandi.csv"))
    
    # Exclude any files in 'merged' directory
    csv_files = [f for f in csv_files if "merged" not in str(f)]
    
    if not csv_files:
        logger.warning("No CSV files found to merge")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files to merge")
    
    # Read and concatenate
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
    
    if not dfs:
        return None
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates if any
    initial_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    duplicates_removed = initial_rows - len(merged_df)
    
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed:,} duplicate rows")
    
    # HARD CONSTRAINT: Final verification - no non-MH records
    non_mh_mask = ~merged_df["state"].apply(is_maharashtra_state)
    non_mh_count = non_mh_mask.sum()
    
    if non_mh_count > 0:
        logger.error(f"CRITICAL: {non_mh_count} non-Maharashtra records in merged data - REMOVING")
        merged_df = merged_df[~non_mh_mask]
    
    # Save merged file
    merged_dir = output_dir / "merged" / timestamp
    ensure_directory(merged_dir)
    
    merged_path = merged_dir / "mandi_maharashtra_merged.csv"
    save_dataframe(merged_df, merged_path)
    
    logger.info(f"Saved merged file: {merged_path}")
    logger.info(f"Total rows: {len(merged_df):,}")
    
    # Save merged receipt
    receipt = {
        "merged_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": MAHARASHTRA_STATE_NAME,
        "source_files": len(csv_files),
        "total_rows": len(merged_df),
        "duplicates_removed": duplicates_removed,
        "non_mh_removed": non_mh_count,
        "columns": list(merged_df.columns),
        "unique_districts": merged_df["district"].nunique() if "district" in merged_df else 0,
        "unique_markets": merged_df["market"].nunique() if "market" in merged_df else 0,
        "unique_commodities": merged_df["commodity"].nunique() if "commodity" in merged_df else 0,
    }
    
    save_receipt(merged_dir / "receipt.json", receipt)
    
    return merged_path


def main():
    """Main entry point for Maharashtra mandi download."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "download.log"
    logger = setup_logger("mh_mandi_download", log_file, level=log_level)
    
    # Initialize audit logger
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("mandi_download", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Maharashtra Mandi Data Downloader")
    logger.info("⚠️  HARD CONSTRAINT: MAHARASHTRA ONLY")
    logger.info("=" * 70)
    
    # Track statistics
    stats = {
        "total_rows": 0,
        "total_pages": 0,
        "non_mh_dropped": 0,
        "chunks_completed": 0,
        "chunks_failed": 0,
    }
    
    try:
        # Load configuration
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Verify Maharashtra constraint in config
        if config["mandi"].get("state_filter") != MAHARASHTRA_STATE_NAME:
            logger.warning("Config state_filter mismatch - enforcing Maharashtra")
        
        # Get API key
        api_key = get_api_key()
        logger.info("API key loaded successfully")
        
        # Build API URL
        mandi_config = config["mandi"]
        resource_id = mandi_config["resource_id"]
        api_url = f"{mandi_config['api_base']}/{resource_id}"
        
        audit.add_section("API Configuration", {
            "endpoint": api_url,
            "resource_id": resource_id,
            "state_filter": MAHARASHTRA_STATE_NAME,
            "page_size": args.page_size,
        })
        
        # Create HTTP session
        http_config = config["http"]
        session = create_session(
            max_retries=http_config["max_retries"],
            backoff_factor=http_config["backoff_factor"],
            retry_status_codes=http_config["retry_status_codes"],
            timeout=http_config["timeout"],
        )
        
        # Setup progress tracker
        progress_file = PROJECT_ROOT / config["mandi"]["progress_file"]
        tracker = ProgressTracker(progress_file)
        
        output_dir = PROJECT_ROOT / args.output_dir
        
        # Determine download strategy
        if args.district:
            # Single district download
            logger.info(f"Downloading single district: {args.district}")
            
            df, pages, non_mh = download_chunk(
                session=session,
                api_url=api_url,
                api_key=api_key,
                page_size=args.page_size,
                district=args.district,
                commodity=args.commodity,
                logger=logger,
            )
            
            if df.empty:
                logger.warning("No data downloaded")
                return 1
            
            receipt = {
                "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "state": MAHARASHTRA_STATE_NAME,
                "district": args.district,
                "commodity": args.commodity,
                "total_rows": len(df),
                "total_pages": pages,
                "non_mh_dropped": non_mh,
            }
            
            save_chunk_data(df, output_dir, args.district, args.commodity, timestamp, receipt, logger)
            
            stats["total_rows"] = len(df)
            stats["total_pages"] = pages
            stats["non_mh_dropped"] = non_mh
            stats["chunks_completed"] = 1
            
        elif args.download_all or args.resume:
            # Full Maharashtra download
            logger.info("Starting full Maharashtra download...")
            
            # Get total count to decide strategy
            total = fetch_total_count(session, api_url, api_key, None, args.commodity, logger)
            
            max_bulk = config["mandi"]["strategy"].get("max_rows_for_bulk", 500000)
            
            if total <= max_bulk:
                # BULK strategy
                logger.info(f"Using BULK strategy (total {total:,} <= {max_bulk:,})")
                
                df, pages, non_mh = download_chunk(
                    session=session,
                    api_url=api_url,
                    api_key=api_key,
                    page_size=args.page_size,
                    district=None,
                    commodity=args.commodity,
                    logger=logger,
                )
                
                if not df.empty:
                    receipt = {
                        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "state": MAHARASHTRA_STATE_NAME,
                        "strategy": "bulk",
                        "total_rows": len(df),
                        "total_pages": pages,
                        "non_mh_dropped": non_mh,
                    }
                    
                    save_chunk_data(df, output_dir, None, args.commodity, timestamp, receipt, logger)
                    
                    stats["total_rows"] = len(df)
                    stats["total_pages"] = pages
                    stats["non_mh_dropped"] = non_mh
                    stats["chunks_completed"] = 1
            else:
                # CHUNKED strategy by district
                logger.info(f"Using CHUNKED strategy (total {total:,} > {max_bulk:,})")
                
                # Get districts from metadata
                metadata_dir = PROJECT_ROOT / config["paths"]["maharashtra"]["metadata"]
                try:
                    districts = get_districts_from_metadata(metadata_dir)
                except FileNotFoundError as e:
                    logger.error(str(e))
                    print(f"\n❌ {e}")
                    return 1
                
                logger.info(f"Found {len(districts)} districts to download")
                
                # Initialize progress tracking
                tracker.start_session(
                    "mandi_download",
                    chunks=districts,
                    metadata={"commodity": args.commodity},
                    force_restart=args.force,
                )
                
                # Get pending chunks
                if args.force:
                    pending = districts
                else:
                    pending = tracker.get_pending_chunks("mandi_download")
                
                logger.info(f"Pending districts: {len(pending)}")
                
                audit.add_section("Chunk Strategy", {
                    "total_districts": len(districts),
                    "pending_districts": len(pending),
                    "already_completed": len(districts) - len(pending),
                })
                
                for district in pending:
                    logger.info(f"\n--- Processing district: {district} ---")
                    tracker.mark_in_progress("mandi_download", district)
                    
                    try:
                        df, pages, non_mh = download_chunk(
                            session=session,
                            api_url=api_url,
                            api_key=api_key,
                            page_size=args.page_size,
                            district=district,
                            commodity=args.commodity,
                            logger=logger,
                        )
                        
                        if df.empty:
                            logger.warning(f"No data for district: {district}")
                            tracker.mark_completed("mandi_download", district, rows=0)
                            continue
                        
                        receipt = {
                            "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "state": MAHARASHTRA_STATE_NAME,
                            "district": district,
                            "commodity": args.commodity,
                            "total_rows": len(df),
                            "total_pages": pages,
                            "non_mh_dropped": non_mh,
                        }
                        
                        chunk_dir = save_chunk_data(df, output_dir, district, args.commodity, timestamp, receipt, logger)
                        
                        tracker.mark_completed(
                            "mandi_download",
                            district,
                            rows=len(df),
                            output_file=str(chunk_dir / "mandi.csv"),
                        )
                        
                        stats["total_rows"] += len(df)
                        stats["total_pages"] += pages
                        stats["non_mh_dropped"] += non_mh
                        stats["chunks_completed"] += 1
                        
                        # Rate limiting between districts
                        time.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Failed to download district {district}: {e}")
                        tracker.mark_failed("mandi_download", district, str(e))
                        stats["chunks_failed"] += 1
                        audit.add_error(f"District {district}: {e}")
        
        # Create merged file
        if not args.no_merge and stats["total_rows"] > 0:
            merged_path = create_merged_file(output_dir, timestamp, logger)
            if merged_path:
                audit.add_section("Merged Output", {
                    "path": str(merged_path),
                })
        
        # Update audit with final stats
        audit.add_metric("Total Rows Downloaded", stats["total_rows"])
        audit.add_metric("Total Pages Fetched", stats["total_pages"])
        audit.add_metric("Non-MH Records Dropped", stats["non_mh_dropped"])
        audit.add_metric("Chunks Completed", stats["chunks_completed"])
        audit.add_metric("Chunks Failed", stats["chunks_failed"])
        
        if stats["non_mh_dropped"] > 0:
            audit.add_warning(f"{stats['non_mh_dropped']} non-Maharashtra records were dropped")
        
        # Save audit report
        audit_path = audit.save()
        logger.info(f"Saved audit report to {audit_path}")
        
        # Summary
        logger.info("=" * 70)
        logger.info("✅ Maharashtra Mandi Download Complete!")
        logger.info(f"   State: {MAHARASHTRA_STATE_NAME}")
        logger.info(f"   Total Rows: {stats['total_rows']:,}")
        logger.info(f"   Non-MH Dropped: {stats['non_mh_dropped']}")
        logger.info(f"   Output: {output_dir}")
        logger.info("=" * 70)
        
        print(f"\n✅ Maharashtra Mandi Download Complete!")
        print(f"   📊 Total Rows: {stats['total_rows']:,}")
        print(f"   ⚠️  Non-MH Dropped: {stats['non_mh_dropped']}")
        print(f"   📁 Output: {output_dir}")
        print(f"   📋 Audit: {audit_path}")
        
        # Exit code based on failures
        if stats["chunks_failed"] > 0:
            return 1
        if stats["non_mh_dropped"] > 0:
            logger.warning("Non-zero non-MH records were dropped - review data source")
        
        return 0
        
    except APIKeyMissingError as e:
        logger.error(f"API Key Error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ API Key Error: {e}")
        return 1
        
    except APIError as e:
        logger.error(f"API Error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ API Error: {e}")
        return 2
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ Unexpected error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
