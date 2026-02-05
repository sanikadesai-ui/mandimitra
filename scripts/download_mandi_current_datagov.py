#!/usr/bin/env python3
"""
MANDIMITRA - Current Daily Mandi Data Downloader (Data.gov.in)

Downloads current/live mandi data from Data.gov.in API.
Saves data partitioned by date with health check and fallback support.

Features:
- Health check before download (verify Maharashtra data availability)
- Automatic fallback to cached data if API unavailable
- Date-partitioned output (data/raw/mandi/current/YYYY-MM-DD/)
- Adaptive rate limiting with 429 handling
- Strict Maharashtra-only enforcement

Usage:
    python scripts/download_mandi_current_datagov.py --download
    python scripts/download_mandi_current_datagov.py --download --date 2026-02-05
    python scripts/download_mandi_current_datagov.py --help

Output:
    data/raw/mandi/current/YYYY-MM-DD/mandi_current.csv
    data/raw/mandi/current/YYYY-MM-DD/receipt.json

⚠️  HARD CONSTRAINT: Maharashtra-only data.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
)
from src.utils.http import (
    APIError,
    APIKeyMissingError,
    RateLimitMode,
    AdaptiveRateLimiter,
    create_session,
    make_request,
    fetch_total_count,
    health_check_maharashtra,
    save_health_check_result,
    redact_params,
)
from src.utils.logging_utils import setup_logger, get_utc_timestamp_safe
from src.utils.maharashtra import (
    MAHARASHTRA_STATE_NAME,
    is_maharashtra_state,
    build_maharashtra_api_filters,
    build_maharashtra_request_params,
)
from src.utils.audit import AuditLogger


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download current mandi data from Data.gov.in (Maharashtra-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HEALTH CHECK:
  Before downloading, the script performs a health check:
  1. Verifies API is reachable
  2. Confirms Maharashtra filter returns data
  3. If health check fails, uses cached data (if available)

OUTPUT STRUCTURE:
  data/raw/mandi/current/
  └── YYYY-MM-DD/
      ├── mandi_current.csv
      └── receipt.json

Examples:
    # Download today's data
    python scripts/download_mandi_current_datagov.py --download
    
    # Download with specific date partition
    python scripts/download_mandi_current_datagov.py --download --date 2026-02-05
    
    # Skip health check (not recommended)
    python scripts/download_mandi_current_datagov.py --download --no-health-check
    
    # Force download even if cache exists
    python scripts/download_mandi_current_datagov.py --download --force
        """,
    )
    
    # Actions
    parser.add_argument(
        "--download",
        action="store_true",
        required=True,
        help="Download current mandi data",
    )
    
    # Date settings
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Partition date (YYYY-MM-DD). Default: today",
    )
    
    # Health check
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip health check (not recommended)",
    )
    
    # Download settings
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Records per API page (default: 500)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to fetch (default: unlimited)",
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/mandi/current",
        help="Output directory for current data",
    )
    
    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if data exists for date",
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


def find_cached_data(
    output_dir: Path,
    max_age_days: int = 7,
    logger: logging.Logger = None,
) -> Optional[Path]:
    """
    Find most recent cached data within age limit.
    
    Args:
        output_dir: Base output directory
        max_age_days: Maximum age of cache to use
        logger: Logger instance
        
    Returns:
        Path to cached CSV file, or None
    """
    cutoff = datetime.now() - timedelta(days=max_age_days)
    
    # Find date-partitioned directories
    date_dirs = sorted(output_dir.glob("????-??-??"), reverse=True)
    
    for date_dir in date_dirs:
        try:
            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
            if dir_date < cutoff:
                break
            
            csv_file = date_dir / "mandi_current.csv"
            if csv_file.exists():
                if logger:
                    logger.info(f"Found cached data: {csv_file}")
                return csv_file
                
        except ValueError:
            continue
    
    return None


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_current_data(
    session: requests.Session,
    api_url: str,
    api_key: str,
    page_size: int,
    max_pages: Optional[int],
    rate_limiter: AdaptiveRateLimiter,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Download all current Maharashtra mandi data.
    
    Args:
        session: HTTP session
        api_url: API endpoint URL
        api_key: Data.gov.in API key
        page_size: Records per page
        max_pages: Maximum pages (None = unlimited)
        rate_limiter: Rate limiter instance
        logger: Logger instance
        
    Returns:
        (DataFrame, stats_dict)
    """
    stats = {
        "total_records": 0,
        "pages_fetched": 0,
        "non_mh_dropped": 0,
        "non_mh_samples": [],
        "errors": [],
    }
    
    all_records: List[Dict[str, Any]] = []
    offset = 0
    page = 1
    
    # Get total count first
    base_params = build_maharashtra_request_params(api_key, limit=1, offset=0)
    total_available = fetch_total_count(session, api_url, base_params, logger=logger, rate_limiter=rate_limiter)
    
    logger.info(f"Total Maharashtra records available: {total_available:,}")
    stats["total_available"] = total_available
    
    if total_available == 0:
        logger.warning("No Maharashtra records available from API")
        return pd.DataFrame(), stats
    
    # Paginate through all data
    while True:
        if max_pages and page > max_pages:
            logger.info(f"Reached max pages limit: {max_pages}")
            break
        
        params = build_maharashtra_request_params(
            api_key=api_key,
            limit=page_size,
            offset=offset,
        )
        
        try:
            data, response = make_request(
                session, api_url, params=params,
                logger=logger, rate_limiter=rate_limiter
            )
            
            records = data.get("records", [])
            if not records:
                break
            
            # Filter and validate Maharashtra
            for record in records:
                state = record.get("state", "")
                if is_maharashtra_state(state):
                    all_records.append(record)
                else:
                    stats["non_mh_dropped"] += 1
                    if len(stats["non_mh_samples"]) < 5:
                        stats["non_mh_samples"].append(f"state='{state}'")
            
            stats["pages_fetched"] = page
            
            # Progress
            if page % 5 == 0:
                logger.info(f"  Page {page}: {len(all_records):,} MH records collected")
            
            # Check if done
            if len(records) < page_size:
                break
            if offset + len(records) >= total_available:
                break
            
            offset += page_size
            page += 1
            
        except Exception as e:
            logger.error(f"Error on page {page}: {e}")
            stats["errors"].append({"page": page, "error": str(e)})
            break
    
    # Create DataFrame
    if all_records:
        df = pd.DataFrame(all_records)
        stats["total_records"] = len(df)
        
        # Log non-MH summary
        if stats["non_mh_dropped"] > 0:
            logger.warning(
                f"Dropped {stats['non_mh_dropped']} non-MH records. "
                f"Samples: {stats['non_mh_samples'][:3]}"
            )
    else:
        df = pd.DataFrame()
    
    return df, stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "current_download.log"
    logger = setup_logger("current_mandi", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("mandi_current_download", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Current Mandi Data Downloader")
    logger.info("=" * 70)
    
    try:
        # Load config
        config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            config = load_config(config_path)
            current_config = config.get("mandi", {}).get("current", {})
            http_config = config.get("http", {})
        else:
            current_config = {}
            http_config = {}
            logger.warning(f"Config not found: {config_path}")
        
        # Get API key
        api_key = get_api_key()
        logger.info("API key loaded ✓")
        
        # API URL
        api_base = current_config.get("api_base", "https://api.data.gov.in/resource")
        resource_id = current_config.get("resource_id", "9ef84268-d588-465a-a308-a864a43d0070")
        api_url = f"{api_base}/{resource_id}"
        
        # Determine partition date
        partition_date = args.date or datetime.now().strftime("%Y-%m-%d")
        
        # Output paths
        output_dir = PROJECT_ROOT / args.output_dir
        date_dir = output_dir / partition_date
        output_csv = date_dir / "mandi_current.csv"
        
        # Check if already exists
        if output_csv.exists() and not args.force:
            logger.info(f"Data already exists for {partition_date}: {output_csv}")
            print(f"\n✅ Data already exists for {partition_date}")
            print(f"   Use --force to re-download")
            return 0
        
        audit.add_section("Configuration", {
            "api_url": api_url,
            "partition_date": partition_date,
            "output_dir": str(date_dir),
            "page_size": args.page_size,
        })
        
        # Create session
        session = create_session(
            max_retries=http_config.get("max_retries", 5),
            backoff_factor=http_config.get("backoff_factor", 2.0),
            timeout=http_config.get("timeout", 60),
            pool_connections=http_config.get("pool_connections", 10),
            pool_maxsize=http_config.get("pool_maxsize", 20),
        )
        
        # Rate limiter
        rate_limiter = AdaptiveRateLimiter(
            mode=RateLimitMode.AUTO,
            base_delay=http_config.get("rate_limit", {}).get("base_delay", 0.5),
        )
        
        # Health check
        if not args.no_health_check:
            logger.info("Running health check...")
            health_result = health_check_maharashtra(
                session, api_url, api_key, timeout=30, logger=logger
            )
            
            # Save health check
            metadata_dir = PROJECT_ROOT / "data" / "metadata" / "maharashtra"
            save_health_check_result(health_result, metadata_dir)
            
            audit.add_section("Health Check", health_result.to_dict())
            
            if not health_result.success or health_result.total_records == 0:
                logger.warning(f"Health check issue: {health_result.error_message or 'No records'}")
                
                # Try fallback to cache
                max_cache_age = current_config.get("fallback", {}).get("max_cache_age_days", 7)
                cached = find_cached_data(output_dir, max_cache_age, logger)
                
                if cached:
                    logger.info(f"Using cached data: {cached}")
                    print(f"\n⚠️  API health check failed. Using cached data:")
                    print(f"   {cached}")
                    audit.add_warning(f"Used cached data due to health check failure")
                    audit.save()
                    return 0
                else:
                    logger.error("No cached data available")
                    print(f"\n❌ Health check failed and no cache available")
                    print(f"   Error: {health_result.error_message}")
                    return 1
        
        # Download
        logger.info(f"Downloading current Maharashtra mandi data...")
        start_time = time.time()
        
        df, stats = download_current_data(
            session=session,
            api_url=api_url,
            api_key=api_key,
            page_size=args.page_size,
            max_pages=args.max_pages,
            rate_limiter=rate_limiter,
            logger=logger,
        )
        
        duration = time.time() - start_time
        stats["duration_seconds"] = round(duration, 2)
        
        audit.add_section("Download Results", stats)
        
        # Save output
        if not df.empty:
            ensure_directory(date_dir)
            save_dataframe(df, output_csv)
            
            # Receipt
            receipt = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "partition_date": partition_date,
                "state": MAHARASHTRA_STATE_NAME,
                "total_records": len(df),
                "pages_fetched": stats["pages_fetched"],
                "non_mh_dropped": stats["non_mh_dropped"],
                "duration_seconds": stats["duration_seconds"],
                "unique_districts": df["district"].nunique() if "district" in df.columns else 0,
                "unique_markets": df["market"].nunique() if "market" in df.columns else 0,
                "unique_commodities": df["commodity"].nunique() if "commodity" in df.columns else 0,
            }
            save_receipt(date_dir / "receipt.json", receipt)
            
            logger.info(f"Saved {len(df):,} rows to {output_csv}")
            
            # Summary
            print(f"\n✅ Current Mandi Data Download Complete!")
            print(f"   📅 Date: {partition_date}")
            print(f"   📊 Rows: {len(df):,}")
            print(f"   📄 Pages: {stats['pages_fetched']}")
            print(f"   ⚠️  Non-MH dropped: {stats['non_mh_dropped']}")
            print(f"   ⏱️  Duration: {duration:.1f}s")
            print(f"\n   📁 Output: {output_csv}")
            
        else:
            logger.warning("No data downloaded")
            print(f"\n⚠️  No data available for {partition_date}")
        
        audit_path = audit.save()
        print(f"   📋 Audit: {audit_path}")
        
        return 0
        
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
        return 1


if __name__ == "__main__":
    sys.exit(main())
