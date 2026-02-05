#!/usr/bin/env python3
"""
MANDIMITRA - Maharashtra Mandi Data Downloader (Production-Grade)

Downloads mandi price data EXCLUSIVELY for Maharashtra with:
- Single count fetch per chunk (no duplicate API calls)
- Parallel downloads with ThreadPoolExecutor
- Adaptive rate limiting with 429 handling
- Streaming CSV writes for large datasets
- Batched non-MH logging (summary, not per-record)
- Progress tracking with atomic saves

⚠️  HARD CONSTRAINT: Maharashtra-only data.

Usage:
    python scripts/download_mandi_maharashtra.py --download-all
    python scripts/download_mandi_maharashtra.py --download-all --max-workers 4
    python scripts/download_mandi_maharashtra.py --district "Pune"
    python scripts/download_mandi_maharashtra.py --resume
    python scripts/download_mandi_maharashtra.py --help

Output:
    data/raw/mandi/maharashtra/<district>/<timestamp>/mandi.csv
    data/raw/mandi/maharashtra/merged/<timestamp>/mandi_merged.csv

Author: MANDIMITRA Team
Version: 2.0.0 (Production Refactor)
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import (
    ensure_directory,
    load_config,
    save_dataframe,
    save_receipt,
    sanitize_filename,
    redact_sensitive_params,
)
from src.utils.http import (
    APIError,
    APIKeyMissingError,
    RateLimitMode,
    AdaptiveRateLimiter,
    create_session,
    make_request,
    fetch_total_count,
    redact_params,
    health_check_maharashtra,
    save_health_check_result,
)
from src.utils.logging_utils import setup_logger, get_utc_timestamp_safe
from src.utils.maharashtra import (
    MAHARASHTRA_STATE_NAME,
    is_maharashtra_state,
    build_maharashtra_api_filters,
    build_maharashtra_request_params,
)
from src.utils.progress import ProgressTracker, ChunkStatus
from src.utils.audit import AuditLogger


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Maharashtra mandi data (production-grade)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  HARD CONSTRAINT: Maharashtra-only data (no --state argument).

OPTIMIZATIONS:
  - Single count fetch per chunk (no duplicate API calls)
  - Parallel district downloads with --max-workers
  - Adaptive rate limiting (handles 429 automatically)
  - Streaming CSV writes for memory safety
  - Batched progress saves (every 10 chunks)

Examples:
    # Full download (auto-selects bulk/chunked strategy)
    python scripts/download_mandi_maharashtra.py --download-all
    
    # Parallel download (4 workers)
    python scripts/download_mandi_maharashtra.py --download-all --max-workers 4
    
    # Single district
    python scripts/download_mandi_maharashtra.py --district "Pune"
    
    # Resume interrupted download
    python scripts/download_mandi_maharashtra.py --resume
    
    # Force restart (discard progress)
    python scripts/download_mandi_maharashtra.py --download-all --no-resume
        """,
    )
    
    # Download mode
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all Maharashtra mandi data",
    )
    parser.add_argument(
        "--district",
        type=str,
        help="Download specific district only",
    )
    parser.add_argument(
        "--commodity",
        type=str,
        help="Filter by commodity (optional)",
    )
    
    # Resumability (FIXED: proper --resume/--no-resume handling)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from progress file if exists (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start fresh, ignore progress file",
    )
    parser.set_defaults(resume=True)
    
    # Concurrency
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers for district downloads (default: 1, max: 8)",
    )
    
    # Rate limiting
    parser.add_argument(
        "--rate-limit",
        type=str,
        choices=["auto", "fixed", "disabled"],
        default="auto",
        help="Rate limiting mode (default: auto)",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=0.5,
        help="Base delay between requests (default: 0.5s)",
    )
    
    # Maharashtra validation
    parser.add_argument(
        "--strict-maharashtra",
        action="store_true",
        default=True,
        help="Drop non-MH rows and fail if any found (default: True)",
    )
    parser.add_argument(
        "--trust-api-filter",
        action="store_true",
        help="Skip per-record filtering, only validate at end",
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/mandi/maharashtra",
        help="Output directory",
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================

def get_api_key() -> str:
    """Get API key from environment."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DATAGOV_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        raise APIKeyMissingError(
            "DATAGOV_API_KEY not found or is placeholder.\n"
            "Set it in .env (see .env.example)"
        )
    return api_key


def build_maharashtra_params(
    api_key: str,
    page_size: int,
    offset: int = 0,
    district: Optional[str] = None,
    commodity: Optional[str] = None,
) -> Dict[str, Any]:
    """Build params with HARDCODED Maharashtra filter.
    
    Uses `filters[state.keyword]` for EXACT matching (not fuzzy).
    This is critical - using `filters[state]` does fuzzy matching
    and can return non-Maharashtra records.
    
    Args:
        api_key: Data.gov.in API key
        page_size: Records per page
        offset: Pagination offset
        district: Optional district filter
        commodity: Optional commodity filter
        
    Returns:
        Dict of query parameters with Maharashtra hardcoded
    """
    # Use centralized filter builder with correct state.keyword syntax
    return build_maharashtra_request_params(
        api_key=api_key,
        limit=page_size,
        offset=offset,
        district=district,
        commodity=commodity,
    )


def get_districts_from_metadata(metadata_dir: Path) -> List[str]:
    """Load discovered districts from metadata CSV."""
    districts_file = metadata_dir / "districts.csv"
    
    if not districts_file.exists():
        raise FileNotFoundError(
            f"Districts metadata not found: {districts_file}\n"
            "Run discovery first: python scripts/discover_maharashtra_mandi_metadata.py --discover-fast"
        )
    
    df = pd.read_csv(districts_file, comment="#", skip_blank_lines=True)
    return df["district"].tolist()


# =============================================================================
# DOWNLOAD FUNCTIONS (Optimized)
# =============================================================================

def download_chunk_streaming(
    session: requests.Session,
    api_url: str,
    api_key: str,
    page_size: int,
    total_count: int,
    district: Optional[str],
    commodity: Optional[str],
    rate_limiter: AdaptiveRateLimiter,
    logger: logging.Logger,
    output_path: Path,
    trust_api_filter: bool = False,
) -> Tuple[int, int, int, float]:
    """
    Download chunk with STREAMING CSV writes (constant memory).
    
    Writes records to CSV incrementally instead of accumulating in memory.
    Ideal for large districts with millions of records.
    
    Args:
        session: HTTP session
        api_url: API endpoint
        api_key: API key
        page_size: Records per page
        total_count: Total records expected (pre-fetched)
        district: District filter
        commodity: Commodity filter
        rate_limiter: Rate limiter instance
        logger: Logger instance
        output_path: Path to write CSV
        trust_api_filter: Skip per-record MH validation
        
    Returns:
        (total_rows, pages_fetched, non_mh_dropped, duration_seconds)
    """
    import csv
    
    start_time = time.time()
    total_rows = 0
    non_mh_count = 0
    non_mh_examples: List[str] = []
    offset = 0
    page = 1
    fieldnames: Optional[List[str]] = None
    
    chunk_name = f"Maharashtra/{district or 'all'}/{commodity or 'all'}"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer: Optional[csv.DictWriter] = None
        
        while True:
            params = build_maharashtra_params(api_key, page_size, offset, district, commodity)
            
            data, response = make_request(
                session, api_url, params=params,
                logger=logger, rate_limiter=rate_limiter
            )
            
            records = data.get("records", [])
            if not records:
                break
            
            # Initialize CSV writer on first page (get fieldnames from data)
            if writer is None and records:
                fieldnames = list(records[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            # Filter and write records
            for record in records:
                if not trust_api_filter:
                    state = record.get("state", "")
                    if not is_maharashtra_state(state):
                        non_mh_count += 1
                        if len(non_mh_examples) < 5:
                            non_mh_examples.append(f"state='{state}'")
                        continue
                
                if writer:
                    writer.writerow(record)
                    total_rows += 1
            
            # Progress (log every 10 pages)
            if page % 10 == 0:
                logger.debug(f"  {chunk_name} page {page}: {total_rows:,} records written")
            
            # Check if done
            if len(records) < page_size or (offset + len(records)) >= total_count:
                break
            
            offset += page_size
            page += 1
    
    duration = time.time() - start_time
    
    # BATCHED non-MH logging (single summary, not per-record)
    if non_mh_count > 0:
        logger.warning(
            f"Chunk '{chunk_name}': Dropped {non_mh_count} non-MH records. "
            f"Examples: {non_mh_examples[:3]}"
        )
    
    return total_rows, page, non_mh_count, duration


def download_chunk(
    session: requests.Session,
    api_url: str,
    api_key: str,
    page_size: int,
    total_count: int,
    district: Optional[str],
    commodity: Optional[str],
    rate_limiter: AdaptiveRateLimiter,
    logger: logging.Logger,
    trust_api_filter: bool = False,
) -> Tuple[pd.DataFrame, int, int, float]:
    """
    Download a chunk of Maharashtra data with pagination.
    
    OPTIMIZATION: total_count is passed in (fetched once by caller).
    For very large chunks, use download_chunk_streaming() instead.
    
    Args:
        session: HTTP session
        api_url: API endpoint
        api_key: API key
        page_size: Records per page
        total_count: Total records expected (pre-fetched)
        district: District filter
        commodity: Commodity filter
        rate_limiter: Rate limiter instance
        logger: Logger instance
        trust_api_filter: Skip per-record MH validation
        
    Returns:
        (DataFrame, pages_fetched, non_mh_dropped, duration_seconds)
    """
    start_time = time.time()
    all_records: List[Dict[str, Any]] = []
    non_mh_count = 0
    non_mh_examples: List[str] = []
    offset = 0
    page = 1
    
    chunk_name = f"Maharashtra/{district or 'all'}/{commodity or 'all'}"
    
    while True:
        params = build_maharashtra_params(api_key, page_size, offset, district, commodity)
        
        data, response = make_request(
            session, api_url, params=params,
            logger=logger, rate_limiter=rate_limiter
        )
        
        records = data.get("records", [])
        if not records:
            break
        
        # Filter non-MH records (unless trusting API filter)
        if trust_api_filter:
            all_records.extend(records)
        else:
            for record in records:
                state = record.get("state", "")
                if is_maharashtra_state(state):
                    all_records.append(record)
                else:
                    non_mh_count += 1
                    if len(non_mh_examples) < 5:
                        non_mh_examples.append(f"state='{state}'")
        
        # Progress (log every 10 pages)
        if page % 10 == 0:
            logger.debug(f"  {chunk_name} page {page}: {len(all_records):,} records")
        
        # Check if done
        if len(records) < page_size or (offset + len(records)) >= total_count:
            break
        
        offset += page_size
        page += 1
    
    duration = time.time() - start_time
    
    # BATCHED non-MH logging (single summary, not per-record)
    if non_mh_count > 0:
        logger.warning(
            f"Chunk '{chunk_name}': Dropped {non_mh_count} non-MH records. "
            f"Examples: {non_mh_examples[:3]}"
        )
    
    if not all_records:
        return pd.DataFrame(), page, non_mh_count, duration
    
    df = pd.DataFrame(all_records)
    return df, page, non_mh_count, duration


def download_district_worker(
    args_tuple: Tuple,
) -> Dict[str, Any]:
    """
    Worker function for parallel district downloads.
    
    Thread-safe: creates own session per worker.
    """
    (
        district, api_url, api_key, page_size, total_count,
        commodity, output_dir, timestamp, rate_limiter, 
        trust_api_filter, http_config, logger_name
    ) = args_tuple
    
    # Create thread-local session (thread-safe)
    session = create_session(
        max_retries=http_config["max_retries"],
        backoff_factor=http_config["backoff_factor"],
        retry_status_codes=http_config["retry_status_codes"],
        timeout=http_config["timeout"],
        pool_connections=2,
        pool_maxsize=5,
    )
    
    # Thread-local logger
    logger = setup_logger(
        f"{logger_name}_{district}",
        PROJECT_ROOT / "logs" / "download.log",
        level="INFO"
    )
    
    result = {
        "district": district,
        "success": False,
        "rows": 0,
        "pages": 0,
        "non_mh_dropped": 0,
        "duration_seconds": 0,
        "error": None,
        "output_file": None,
    }
    
    try:
        # Download
        df, pages, non_mh, duration = download_chunk(
            session=session,
            api_url=api_url,
            api_key=api_key,
            page_size=page_size,
            total_count=total_count,
            district=district,
            commodity=commodity,
            rate_limiter=rate_limiter,
            logger=logger,
            trust_api_filter=trust_api_filter,
        )
        
        result["pages"] = pages
        result["non_mh_dropped"] = non_mh
        result["duration_seconds"] = duration
        
        if df.empty:
            logger.info(f"No data for district: {district}")
            result["success"] = True
            return result
        
        # Save CSV
        district_dir = output_dir / sanitize_filename(district) / timestamp
        ensure_directory(district_dir)
        
        csv_path = district_dir / "mandi.csv"
        save_dataframe(df, csv_path)
        
        # Save receipt (API key redacted)
        receipt = {
            "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "state": MAHARASHTRA_STATE_NAME,
            "district": district,
            "commodity": commodity,
            "total_rows": len(df),
            "total_pages": pages,
            "non_mh_dropped": non_mh,
            "duration_seconds": round(duration, 2),
        }
        save_receipt(district_dir / "receipt.json", receipt)
        
        result["success"] = True
        result["rows"] = len(df)
        result["output_file"] = str(csv_path)
        
        logger.info(f"✓ {district}: {len(df):,} rows in {duration:.1f}s")
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"✗ {district}: {e}")
    
    return result


def create_merged_file(output_dir: Path, timestamp: str, logger) -> Optional[Path]:
    """Create merged CSV from all downloaded chunks."""
    logger.info("Creating merged Maharashtra dataset...")
    
    # Find all mandi.csv files (exclude merged/)
    csv_files = [f for f in output_dir.glob("**/mandi.csv") if "merged" not in str(f)]
    
    if not csv_files:
        logger.warning("No CSV files found to merge")
        return None
    
    logger.info(f"Merging {len(csv_files)} files...")
    
    # Read and concatenate (chunked for memory safety)
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, comment="#")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
    
    if not dfs:
        return None
    
    merged = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate
    initial_rows = len(merged)
    merged = merged.drop_duplicates()
    dups_removed = initial_rows - len(merged)
    
    if dups_removed > 0:
        logger.info(f"Removed {dups_removed:,} duplicates")
    
    # Final MH validation
    if "state" in merged.columns:
        non_mh = ~merged["state"].apply(is_maharashtra_state)
        non_mh_count = non_mh.sum()
        if non_mh_count > 0:
            logger.error(f"CRITICAL: {non_mh_count} non-MH rows in merged data - removing")
            merged = merged[~non_mh]
    
    # Save
    merged_dir = output_dir / "merged" / timestamp
    ensure_directory(merged_dir)
    merged_path = merged_dir / "mandi_merged.csv"
    save_dataframe(merged, merged_path)
    
    # Receipt
    receipt = {
        "merged_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": MAHARASHTRA_STATE_NAME,
        "source_files": len(csv_files),
        "total_rows": len(merged),
        "duplicates_removed": dups_removed,
        "unique_districts": merged["district"].nunique() if "district" in merged.columns else 0,
        "unique_markets": merged["market"].nunique() if "market" in merged.columns else 0,
        "unique_commodities": merged["commodity"].nunique() if "commodity" in merged.columns else 0,
    }
    save_receipt(merged_dir / "receipt.json", receipt)
    
    logger.info(f"Merged file: {merged_path} ({len(merged):,} rows)")
    return merged_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate workers
    max_workers = min(args.max_workers, 8)
    
    # Setup
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "download.log"
    logger = setup_logger("mh_mandi", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("mandi_download", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Maharashtra Mandi Downloader")
    logger.info(f"Workers: {max_workers} | Rate limit: {args.rate_limit}")
    logger.info("=" * 70)
    
    stats = {
        "total_rows": 0,
        "total_pages": 0,
        "non_mh_dropped": 0,
        "chunks_completed": 0,
        "chunks_failed": 0,
        "total_duration": 0,
    }
    
    try:
        config = load_config(PROJECT_ROOT / args.config)
        api_key = get_api_key()
        logger.info("Config loaded ✓")
        
        # API URL
        mandi_config = config["mandi"]
        resource_id = mandi_config["resource_id"]
        api_url = f"{mandi_config['api_base']}/{resource_id}"
        http_config = config["http"]
        
        # Rate limiter (shared across threads)
        rate_limiter = AdaptiveRateLimiter(
            mode=RateLimitMode(args.rate_limit),
            base_delay=args.base_delay,
        )
        
        output_dir = PROJECT_ROOT / args.output_dir
        
        audit.add_section("Configuration", {
            "endpoint": api_url,
            "state_filter": MAHARASHTRA_STATE_NAME,
            "max_workers": max_workers,
            "rate_limit_mode": args.rate_limit,
            "page_size": args.page_size,
        })
        
        # Create main session for count fetches
        session = create_session(
            max_retries=http_config["max_retries"],
            backoff_factor=http_config["backoff_factor"],
            retry_status_codes=http_config["retry_status_codes"],
            timeout=http_config["timeout"],
            pool_connections=5,
            pool_maxsize=10,
        )
        
        # ==================================================================
        # HEALTH CHECK: Verify Maharashtra data is available before download
        # ==================================================================
        logger.info("Running health check...")
        health_result = health_check_maharashtra(session, api_url, api_key, timeout=30, logger=logger)
        
        # Save health check result
        metadata_dir = PROJECT_ROOT / config["paths"]["maharashtra"]["metadata"]
        healthcheck_path = save_health_check_result(health_result, metadata_dir)
        logger.info(f"Health check saved: {healthcheck_path}")
        
        if not health_result.success:
            logger.error(f"Health check FAILED: {health_result.error_message}")
            audit.add_error(f"Health check failed: {health_result.error_message}")
            
            # Try to fall back to cached data
            merged_dir = output_dir / "merged"
            cached_files = list(merged_dir.glob("*/mandi_merged.csv")) if merged_dir.exists() else []
            if cached_files:
                latest_cache = max(cached_files, key=lambda p: p.stat().st_mtime)
                logger.warning(f"API unavailable. Using cached data: {latest_cache}")
                print(f"\n⚠️  API health check failed. Using cached data:")
                print(f"   {latest_cache}")
                print(f"   Error: {health_result.error_message}")
                return 0
            else:
                print(f"\n❌ Health check failed and no cached data available")
                print(f"   Error: {health_result.error_message}")
                return 1
        
        if health_result.total_records == 0:
            logger.warning("Health check OK but 0 Maharashtra records found")
            audit.add_warning("API returned 0 Maharashtra records")
            
            # Try cache
            merged_dir = output_dir / "merged"
            cached_files = list(merged_dir.glob("*/mandi_merged.csv")) if merged_dir.exists() else []
            if cached_files:
                latest_cache = max(cached_files, key=lambda p: p.stat().st_mtime)
                logger.warning(f"Using cached data: {latest_cache}")
                print(f"\n⚠️  API returned 0 Maharashtra records. Using cached data:")
                print(f"   {latest_cache}")
                return 0
            else:
                print(f"\n⚠️  API returned 0 Maharashtra records and no cache available")
                return 1
        
        logger.info(
            f"Health check PASSED: {health_result.total_records:,} records, "
            f"first_state={health_result.first_record_state}, "
            f"districts={health_result.sample_districts}"
        )
        
        audit.add_section("Health Check", {
            "success": health_result.success,
            "total_records": health_result.total_records,
            "first_record_state": health_result.first_record_state,
            "sample_districts": health_result.sample_districts,
            "latency_ms": round(health_result.latency_ms, 1),
        })
        
        # Single district mode
        if args.district:
            logger.info(f"Downloading single district: {args.district}")
            
            # Fetch total ONCE
            base_params = build_maharashtra_params(api_key, 1, 0, args.district, args.commodity)
            total = fetch_total_count(session, api_url, base_params, logger=logger, rate_limiter=rate_limiter)
            logger.info(f"Total records: {total:,}")
            
            if total == 0:
                logger.warning("No records found")
                return 0
            
            df, pages, non_mh, duration = download_chunk(
                session, api_url, api_key, args.page_size, total,
                args.district, args.commodity, rate_limiter, logger,
                args.trust_api_filter
            )
            
            if not df.empty:
                district_dir = output_dir / sanitize_filename(args.district) / timestamp
                ensure_directory(district_dir)
                save_dataframe(df, district_dir / "mandi.csv")
                
                receipt = {
                    "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "state": MAHARASHTRA_STATE_NAME,
                    "district": args.district,
                    "total_rows": len(df),
                    "total_pages": pages,
                    "non_mh_dropped": non_mh,
                    "duration_seconds": round(duration, 2),
                }
                save_receipt(district_dir / "receipt.json", receipt)
            
            stats["total_rows"] = len(df) if not df.empty else 0
            stats["total_pages"] = pages
            stats["non_mh_dropped"] = non_mh
            stats["chunks_completed"] = 1
            stats["total_duration"] = duration
            
        elif args.download_all or args.resume:
            # Full download with chunked strategy
            logger.info("Starting full Maharashtra download...")
            
            # Get districts
            metadata_dir = PROJECT_ROOT / config["paths"]["maharashtra"]["metadata"]
            districts = get_districts_from_metadata(metadata_dir)
            logger.info(f"Found {len(districts)} districts")
            
            # Progress tracker
            progress_file = PROJECT_ROOT / config["mandi"]["progress_file"]
            tracker = ProgressTracker(progress_file, batch_size=10)
            
            # Get pending chunks
            if args.resume and tracker.has_session("mandi_download"):
                pending = tracker.get_pending_chunks("mandi_download")
                completed = tracker.get_completed_chunks("mandi_download")
                logger.info(f"Resuming: {len(completed)} done, {len(pending)} pending")
            else:
                tracker.start_session("mandi_download", districts, force_restart=not args.resume)
                pending = districts
            
            if not pending:
                logger.info("All districts already downloaded")
                return 0
            
            # Fetch counts for all pending districts
            logger.info("Fetching record counts...")
            district_counts = {}
            for district in pending:
                base_params = build_maharashtra_params(api_key, 1, 0, district, args.commodity)
                count = fetch_total_count(session, api_url, base_params, logger=None, rate_limiter=rate_limiter)
                district_counts[district] = count
            
            total_records = sum(district_counts.values())
            logger.info(f"Total records across {len(pending)} districts: {total_records:,}")
            
            audit.add_section("Download Plan", {
                "pending_districts": len(pending),
                "total_records": total_records,
                "max_workers": max_workers,
            })
            
            # Parallel download
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Build worker args
                worker_args = [
                    (
                        district, api_url, api_key, args.page_size,
                        district_counts[district], args.commodity,
                        output_dir, timestamp, rate_limiter,
                        args.trust_api_filter, http_config, "mh_worker"
                    )
                    for district in pending
                ]
                
                futures = {
                    executor.submit(download_district_worker, arg): arg[0]
                    for arg in worker_args
                }
                
                for future in as_completed(futures):
                    district = futures[future]
                    try:
                        result = future.result()
                        
                        if result["success"]:
                            tracker.mark_completed(
                                "mandi_download", district,
                                rows=result["rows"],
                                output_file=result["output_file"],
                                duration_seconds=result["duration_seconds"],
                            )
                            stats["total_rows"] += result["rows"]
                            stats["total_pages"] += result["pages"]
                            stats["non_mh_dropped"] += result["non_mh_dropped"]
                            stats["chunks_completed"] += 1
                        else:
                            tracker.mark_failed("mandi_download", district, result["error"] or "Unknown")
                            stats["chunks_failed"] += 1
                            audit.add_error(f"{district}: {result['error']}")
                            
                    except Exception as e:
                        tracker.mark_failed("mandi_download", district, str(e))
                        stats["chunks_failed"] += 1
                        audit.add_error(f"{district}: {e}")
            
            stats["total_duration"] = time.time() - start_time
            
            # Flush progress
            tracker.flush()
        
        # Create merged file
        if not args.no_merge and stats["total_rows"] > 0:
            merged_path = create_merged_file(output_dir, timestamp, logger)
            if merged_path:
                audit.add_section("Merged Output", {"path": str(merged_path)})
        
        # Audit metrics
        audit.add_metric("Total Rows", stats["total_rows"])
        audit.add_metric("Total Pages", stats["total_pages"])
        audit.add_metric("Non-MH Dropped", stats["non_mh_dropped"])
        audit.add_metric("Chunks Completed", stats["chunks_completed"])
        audit.add_metric("Chunks Failed", stats["chunks_failed"])
        audit.add_metric("Duration (seconds)", round(stats["total_duration"], 2))
        
        if stats["non_mh_dropped"] > 0:
            audit.add_warning(f"{stats['non_mh_dropped']} non-MH records dropped")
        
        audit_path = audit.save()
        
        # Summary
        print(f"\n✅ Maharashtra Mandi Download Complete!")
        print(f"   📊 Rows: {stats['total_rows']:,}")
        print(f"   📄 Pages: {stats['total_pages']}")
        print(f"   ⚠️  Non-MH dropped: {stats['non_mh_dropped']}")
        print(f"   ✓ Completed: {stats['chunks_completed']}")
        print(f"   ✗ Failed: {stats['chunks_failed']}")
        print(f"   ⏱️  Duration: {stats['total_duration']:.1f}s")
        print(f"\n   📁 Output: {output_dir}")
        print(f"   📋 Audit: {audit_path}")
        
        return 1 if stats["chunks_failed"] > 0 else 0
        
    except APIKeyMissingError as e:
        logger.error(str(e))
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ {e}")
        return 1
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ Error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
