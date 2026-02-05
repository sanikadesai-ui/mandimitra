#!/usr/bin/env python3
"""
MANDIMITRA - Maharashtra Mandi Metadata Discovery Script (Production-Grade)

Discovers unique districts, markets, and commodities for Maharashtra with:
- CONSTANT MEMORY: Streams records, never stores all in memory
- Fast mode (--discover-fast): Limited pages for quick discovery
- Full mode (--discover-full): All pages, still constant memory
- Field filtering: Only requests needed fields to reduce payload

⚠️  HARD CONSTRAINT: Maharashtra-only data.

Usage:
    python scripts/discover_maharashtra_mandi_metadata.py --discover-fast
    python scripts/discover_maharashtra_mandi_metadata.py --discover-full
    python scripts/discover_maharashtra_mandi_metadata.py --help

Output:
    data/metadata/maharashtra/districts.csv
    data/metadata/maharashtra/markets.csv  
    data/metadata/maharashtra/commodities.csv
    data/metadata/maharashtra/discovery_receipt.json

Author: MANDIMITRA Team
Version: 2.0.0 (Production Refactor)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import (
    ensure_directory,
    load_config,
    save_receipt,
    redact_sensitive_params,
)
from src.utils.http import (
    APIError,
    APIKeyMissingError,
    RateLimitMode,
    AdaptiveRateLimiter,
    create_session,
    fetch_total_count,
    stream_paginated_records,
    redact_params,
)
from src.utils.logging_utils import setup_logger, get_utc_timestamp_safe
from src.utils.maharashtra import (
    MAHARASHTRA_STATE_NAME,
    is_maharashtra_state,
    build_maharashtra_api_filters,
)
from src.utils.audit import AuditLogger


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Discover Maharashtra mandi metadata (memory-safe streaming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DISCOVERY MODES:
  --discover-fast    Quick discovery (max 50 pages, ~50K records)
                     Good for development and testing
  --discover-full    Full discovery (all pages, constant memory)
                     Required for production completeness

MEMORY SAFETY:
  This script uses STREAMING to maintain constant memory regardless of
  dataset size. Records are processed one-at-a-time; only unique values
  are stored (districts, markets, commodities as sets).

Examples:
    # Quick discovery (dev/testing)
    python scripts/discover_maharashtra_mandi_metadata.py --discover-fast
    
    # Full discovery (production)
    python scripts/discover_maharashtra_mandi_metadata.py --discover-full
    
    # Custom limits
    python scripts/discover_maharashtra_mandi_metadata.py --max-pages 100 --max-records 100000
        """,
    )
    
    # Discovery mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--discover-fast",
        action="store_true",
        help="Quick discovery with limited pages (for dev/testing)",
    )
    mode_group.add_argument(
        "--discover-full",
        action="store_true", 
        help="Full discovery of all pages (constant memory)",
    )
    mode_group.add_argument(
        "--max-pages",
        type=int,
        metavar="N",
        help="Custom max pages to fetch",
    )
    
    # Optional limits
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum records to process (early stop)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Records per API page (default: 1000)",
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
        default="data/metadata/maharashtra",
        help="Output directory for metadata",
    )
    
    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if metadata exists",
    )
    parser.add_argument(
        "--rate-limit",
        type=str,
        choices=["auto", "fixed", "disabled"],
        default="auto",
        help="Rate limiting mode (default: auto)",
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
    """Get API key from environment (never log/store the actual key)."""
    load_dotenv(PROJECT_ROOT / ".env")
    
    api_key = os.getenv("DATAGOV_API_KEY")
    
    if not api_key:
        raise APIKeyMissingError(
            "DATAGOV_API_KEY not found.\n"
            "Set it in your .env file (see .env.example)"
        )
    
    if api_key == "your_api_key_here":
        raise APIKeyMissingError(
            "DATAGOV_API_KEY is still placeholder.\n"
            "Update .env with your actual key from https://data.gov.in"
        )
    
    return api_key


def build_maharashtra_params(api_key: str, page_size: int) -> Dict[str, Any]:
    """Build base params with HARDCODED Maharashtra filter.
    
    Uses `filters[state.keyword]` for EXACT matching (not fuzzy).
    This is critical - using `filters[state]` does fuzzy matching
    and can return non-Maharashtra records.
    """
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": page_size,
    }
    params.update(build_maharashtra_api_filters())
    return params


# =============================================================================
# STREAMING DISCOVERY (Constant Memory)
# =============================================================================

def discover_metadata_streaming(
    session,
    api_url: str,
    api_key: str,
    page_size: int,
    max_pages: Optional[int],
    max_records: Optional[int],
    rate_limiter: AdaptiveRateLimiter,
    logger,
) -> Dict[str, Any]:
    """
    Discover unique values using STREAMING (constant memory).
    
    This function NEVER stores all records in memory. It processes
    records one-at-a-time, updating only the unique value sets.
    
    Memory usage: O(unique_districts + unique_markets + unique_commodities)
    NOT: O(total_records)
    """
    start_time = time.time()
    
    # Only store unique values (sets have constant-ish memory for unique values)
    districts: Set[str] = set()
    markets: Set[str] = set()
    commodities: Set[str] = set()
    
    # Counters (constant memory)
    total_processed = 0
    non_mh_dropped = 0
    pages_fetched = 0
    
    # Build params
    base_params = build_maharashtra_params(api_key, page_size)
    
    # Get total count first (single lightweight request)
    total_available = fetch_total_count(
        session, api_url, base_params, 
        logger=logger, rate_limiter=rate_limiter
    )
    logger.info(f"Total Maharashtra records available: {total_available:,}")
    
    if total_available == 0:
        raise APIError("No Maharashtra records found. Check API/filters.")
    
    # Progress callback
    def on_page(records, page_num, fetched_so_far):
        nonlocal pages_fetched
        pages_fetched = page_num
        if page_num % 10 == 0 or page_num == 1:
            logger.info(
                f"  Page {page_num}: {fetched_so_far:,} records | "
                f"Districts: {len(districts)} | Markets: {len(markets)} | "
                f"Commodities: {len(commodities)}"
            )
    
    logger.info("Streaming records (constant memory)...")
    
    # Stream records one-at-a-time
    for record in stream_paginated_records(
        session=session,
        url=api_url,
        base_params=base_params,
        page_size=page_size,
        max_pages=max_pages,
        max_records=max_records,
        logger=None,  # Use callback instead
        rate_limiter=rate_limiter,
        on_page_callback=on_page,
    ):
        # Verify Maharashtra (API filter should handle this, but verify)
        state = record.get("state", "")
        if not is_maharashtra_state(state):
            non_mh_dropped += 1
            continue
        
        # Extract unique values (constant memory per unique value)
        if record.get("district"):
            districts.add(record["district"].strip())
        if record.get("market"):
            markets.add(record["market"].strip())
        if record.get("commodity"):
            commodities.add(record["commodity"].strip())
        
        total_processed += 1
    
    duration = time.time() - start_time
    
    # Log summary (not per-record)
    if non_mh_dropped > 0:
        logger.warning(
            f"Non-MH records dropped: {non_mh_dropped} "
            "(API filter may have returned cross-state data)"
        )
    
    logger.info(f"Discovery complete in {duration:.1f}s:")
    logger.info(f"  Records processed: {total_processed:,}")
    logger.info(f"  Pages fetched: {pages_fetched}")
    logger.info(f"  Unique districts: {len(districts)}")
    logger.info(f"  Unique markets: {len(markets)}")
    logger.info(f"  Unique commodities: {len(commodities)}")
    
    return {
        "districts": districts,
        "markets": markets,
        "commodities": commodities,
        "total_processed": total_processed,
        "total_available": total_available,
        "non_mh_dropped": non_mh_dropped,
        "pages_fetched": pages_fetched,
        "duration_seconds": duration,
    }


def save_metadata_csv(values: Set[str], output_path: Path, column_name: str, logger) -> int:
    """Save unique values as sorted CSV."""
    sorted_values = sorted(list(values))
    df = pd.DataFrame({column_name: sorted_values})
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved {len(sorted_values)} {column_name} to {output_path}")
    return len(sorted_values)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Determine max_pages based on mode
    if args.discover_fast:
        max_pages = 50  # ~50K records, enough for quick lists
    elif args.discover_full:
        max_pages = None  # No limit
    else:
        max_pages = args.max_pages
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "discovery.log"
    logger = setup_logger("mh_discovery", log_file, level=log_level)
    
    # Audit
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("discovery", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Maharashtra Metadata Discovery")
    logger.info("Mode: STREAMING (constant memory)")
    logger.info(f"Max pages: {max_pages or 'unlimited'}")
    logger.info("=" * 70)
    
    try:
        # Load config
        config = load_config(PROJECT_ROOT / args.config)
        
        # Check if metadata exists
        output_dir = PROJECT_ROOT / args.output_dir
        districts_file = output_dir / "districts.csv"
        markets_file = output_dir / "markets.csv"
        commodities_file = output_dir / "commodities.csv"
        
        if not args.force and all(f.exists() for f in [districts_file, markets_file, commodities_file]):
            logger.info("Metadata already exists. Use --force to refresh.")
            
            # Show existing counts
            d_count = len(pd.read_csv(districts_file, comment="#"))
            m_count = len(pd.read_csv(markets_file, comment="#"))
            c_count = len(pd.read_csv(commodities_file, comment="#"))
            
            print(f"\n📊 Existing Maharashtra Metadata:")
            print(f"   Districts: {d_count}")
            print(f"   Markets: {m_count}")
            print(f"   Commodities: {c_count}")
            print(f"\nUse --force to refresh.")
            return 0
        
        # Get API key
        api_key = get_api_key()
        logger.info("API key loaded ✓")
        
        # Build API URL
        mandi_config = config["mandi"]
        resource_id = mandi_config["resource_id"]
        api_url = f"{mandi_config['api_base']}/{resource_id}"
        
        audit.add_section("Configuration", {
            "endpoint": api_url,
            "resource_id": resource_id,
            "state_filter": MAHARASHTRA_STATE_NAME,
            "max_pages": max_pages or "unlimited",
            "page_size": args.page_size,
        })
        
        # Create session with pooling
        http_config = config["http"]
        session = create_session(
            max_retries=http_config["max_retries"],
            backoff_factor=http_config["backoff_factor"],
            retry_status_codes=http_config["retry_status_codes"],
            timeout=http_config["timeout"],
            pool_connections=5,
            pool_maxsize=10,
        )
        
        # Setup rate limiter
        rate_mode = RateLimitMode(args.rate_limit)
        rate_limiter = AdaptiveRateLimiter(mode=rate_mode, base_delay=0.5)
        
        # Run streaming discovery
        results = discover_metadata_streaming(
            session=session,
            api_url=api_url,
            api_key=api_key,
            page_size=args.page_size,
            max_pages=max_pages,
            max_records=args.max_records,
            rate_limiter=rate_limiter,
            logger=logger,
        )
        
        # Save CSVs
        ensure_directory(output_dir)
        
        d_count = save_metadata_csv(results["districts"], districts_file, "district", logger)
        m_count = save_metadata_csv(results["markets"], markets_file, "market", logger)
        c_count = save_metadata_csv(results["commodities"], commodities_file, "commodity", logger)
        
        # Create receipt (API key redacted)
        receipt = {
            "discovery_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "state": MAHARASHTRA_STATE_NAME,
            "api_endpoint": api_url,
            "resource_id": resource_id,
            "mode": "fast" if args.discover_fast else "full",
            "statistics": {
                "total_available": results["total_available"],
                "total_processed": results["total_processed"],
                "pages_fetched": results["pages_fetched"],
                "non_mh_dropped": results["non_mh_dropped"],
                "unique_districts": d_count,
                "unique_markets": m_count,
                "unique_commodities": c_count,
                "duration_seconds": round(results["duration_seconds"], 2),
            },
            "output_files": {
                "districts": str(districts_file.relative_to(PROJECT_ROOT)),
                "markets": str(markets_file.relative_to(PROJECT_ROOT)),
                "commodities": str(commodities_file.relative_to(PROJECT_ROOT)),
            },
        }
        
        receipt_path = output_dir / "discovery_receipt.json"
        save_receipt(receipt_path, receipt)
        
        # Audit metrics
        audit.add_metric("Records Processed", results["total_processed"])
        audit.add_metric("Pages Fetched", results["pages_fetched"])
        audit.add_metric("Non-MH Dropped", results["non_mh_dropped"])
        audit.add_metric("Unique Districts", d_count)
        audit.add_metric("Unique Markets", m_count)
        audit.add_metric("Unique Commodities", c_count)
        audit.add_metric("Duration (seconds)", round(results["duration_seconds"], 2))
        
        if results["non_mh_dropped"] > 0:
            audit.add_warning(f"{results['non_mh_dropped']} non-MH records dropped")
        
        audit_path = audit.save()
        
        # Summary
        print(f"\n✅ Maharashtra Metadata Discovery Complete!")
        print(f"   📍 Districts: {d_count}")
        print(f"   🏪 Markets: {m_count}")
        print(f"   🌾 Commodities: {c_count}")
        print(f"   ⏱️  Duration: {results['duration_seconds']:.1f}s")
        print(f"\n   📁 Output: {output_dir}")
        print(f"   📋 Audit: {audit_path}")
        
        return 0
        
    except APIKeyMissingError as e:
        logger.error(f"API Key Error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ {e}")
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
        print(f"\n❌ Error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
