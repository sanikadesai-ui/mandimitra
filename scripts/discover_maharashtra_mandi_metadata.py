#!/usr/bin/env python3
"""
MANDIMITRA - Maharashtra Mandi Metadata Discovery Script

Discovers and saves unique values for Maharashtra mandi data:
- Districts
- Markets
- Commodities

This MUST be run before full data download to understand coverage.

Usage:
    python scripts/discover_maharashtra_mandi_metadata.py
    python scripts/discover_maharashtra_mandi_metadata.py --verbose
    python scripts/discover_maharashtra_mandi_metadata.py --help

Output:
    data/metadata/maharashtra/districts.csv
    data/metadata/maharashtra/markets.csv
    data/metadata/maharashtra/commodities.csv
    data/metadata/maharashtra/discovery_receipt.json

Author: MANDIMITRA Team
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
from dotenv import load_dotenv

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import (
    ensure_directory,
    load_config,
    save_receipt,
)
from src.utils.http_utils import (
    APIError,
    APIKeyMissingError,
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
)
from src.utils.audit import AuditLogger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Discover Maharashtra mandi metadata (districts, markets, commodities)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script queries the Data.gov.in API to discover all unique values for:
  - Districts in Maharashtra
  - Markets (mandis) in Maharashtra  
  - Commodities available in Maharashtra

The discovered metadata is saved to data/metadata/maharashtra/ and used
by the main download script to plan efficient data downloads.

IMPORTANT: This script ONLY queries Maharashtra data. It is a HARD CONSTRAINT
that no other state's data will ever be downloaded or processed.

Examples:
    # Run discovery
    python scripts/discover_maharashtra_mandi_metadata.py
    
    # Run with verbose output
    python scripts/discover_maharashtra_mandi_metadata.py --verbose
    
    # Force refresh even if metadata exists
    python scripts/discover_maharashtra_mandi_metadata.py --force
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/metadata/maharashtra",
        help="Output directory for metadata files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even if metadata already exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug output",
    )
    
    return parser.parse_args()


def get_api_key() -> str:
    """
    Get Data.gov.in API key from environment.
    
    Returns:
        API key string
        
    Raises:
        APIKeyMissingError: If API key not found
    """
    load_dotenv(PROJECT_ROOT / ".env")
    
    api_key = os.getenv("DATAGOV_API_KEY")
    
    if not api_key:
        raise APIKeyMissingError(
            "DATAGOV_API_KEY not found in environment.\n"
            "Please set it in your .env file:\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your API key from https://data.gov.in/user/register"
        )
    
    if api_key == "your_api_key_here":
        raise APIKeyMissingError(
            "DATAGOV_API_KEY is set to placeholder value.\n"
            "Please update your .env file with a valid API key."
        )
    
    return api_key


def fetch_maharashtra_total_count(
    session,
    api_url: str,
    api_key: str,
    logger,
) -> int:
    """
    Fetch total count of Maharashtra records.
    
    Args:
        session: HTTP session
        api_url: API endpoint
        api_key: API key
        logger: Logger instance
        
    Returns:
        Total record count
    """
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 1,
        "offset": 0,
        f"filters[state]": MAHARASHTRA_STATE_NAME,
    }
    
    logger.info(f"Fetching total count for {MAHARASHTRA_STATE_NAME}...")
    data, _ = make_request(session, api_url, params=params, logger=logger)
    
    total = data.get("total", 0)
    logger.info(f"Total Maharashtra records available: {total:,}")
    
    return total


def discover_unique_values(
    session,
    api_url: str,
    api_key: str,
    page_size: int,
    logger,
) -> Dict[str, Set[str]]:
    """
    Discover unique districts, markets, and commodities for Maharashtra.
    
    This fetches all Maharashtra records and extracts unique values.
    For large datasets, this uses sampling to be efficient.
    
    Args:
        session: HTTP session
        api_url: API endpoint
        api_key: API key
        page_size: Records per page
        logger: Logger instance
        
    Returns:
        Dictionary with sets of unique values
    """
    districts: Set[str] = set()
    markets: Set[str] = set()
    commodities: Set[str] = set()
    non_mh_count = 0
    total_processed = 0
    
    offset = 0
    page = 1
    
    # First, get total count
    total = fetch_maharashtra_total_count(session, api_url, api_key, logger)
    
    if total == 0:
        raise APIError("No Maharashtra records found. Check API filters.")
    
    logger.info("Fetching all Maharashtra records to discover unique values...")
    logger.info(f"This may take a while for {total:,} records...")
    
    with ProgressLogger(logger, "Discovering Maharashtra metadata") as progress:
        while True:
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": page_size,
                "offset": offset,
                f"filters[state]": MAHARASHTRA_STATE_NAME,
            }
            
            try:
                data, _ = make_request(session, api_url, params=params, logger=logger)
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break
            
            records = data.get("records", [])
            
            if not records:
                break
            
            for record in records:
                # HARD CONSTRAINT: Verify Maharashtra
                state = record.get("state", "")
                if not is_maharashtra_state(state):
                    non_mh_count += 1
                    logger.warning(f"Non-Maharashtra record found: state='{state}' - DROPPING")
                    continue
                
                # Extract unique values
                if record.get("district"):
                    districts.add(record["district"].strip())
                if record.get("market"):
                    markets.add(record["market"].strip())
                if record.get("commodity"):
                    commodities.add(record["commodity"].strip())
                
                total_processed += 1
            
            progress.update(f"Page {page}: {len(records)} records | Districts: {len(districts)} | Markets: {len(markets)} | Commodities: {len(commodities)}")
            
            # Check if we've fetched all
            if len(records) < page_size or (offset + page_size) >= total:
                break
            
            offset += page_size
            page += 1
            
            # Rate limiting
            import time
            time.sleep(0.5)
    
    if non_mh_count > 0:
        logger.error(f"HARD CONSTRAINT CHECK: {non_mh_count} non-Maharashtra records were dropped!")
    
    logger.info(f"Discovery complete:")
    logger.info(f"  Total records processed: {total_processed:,}")
    logger.info(f"  Unique districts: {len(districts)}")
    logger.info(f"  Unique markets: {len(markets)}")
    logger.info(f"  Unique commodities: {len(commodities)}")
    
    return {
        "districts": districts,
        "markets": markets,
        "commodities": commodities,
        "total_processed": total_processed,
        "non_mh_dropped": non_mh_count,
    }


def save_metadata_csv(
    values: Set[str],
    output_path: Path,
    column_name: str,
    logger,
) -> int:
    """Save unique values as CSV."""
    sorted_values = sorted(list(values))
    df = pd.DataFrame({column_name: sorted_values})
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Saved {len(sorted_values)} {column_name} to {output_path}")
    return len(sorted_values)


def main():
    """Main entry point for metadata discovery."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "download.log"
    logger = setup_logger("mh_discovery", log_file, level=log_level)
    
    # Initialize audit logger
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("discovery", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Maharashtra Mandi Metadata Discovery")
    logger.info("⚠️  HARD CONSTRAINT: Maharashtra-only data")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Check if metadata already exists
        output_dir = PROJECT_ROOT / args.output_dir
        districts_file = output_dir / "districts.csv"
        markets_file = output_dir / "markets.csv"
        commodities_file = output_dir / "commodities.csv"
        
        if not args.force and all(f.exists() for f in [districts_file, markets_file, commodities_file]):
            logger.info("Metadata files already exist. Use --force to refresh.")
            
            # Load and display existing metadata
            districts_df = pd.read_csv(districts_file)
            markets_df = pd.read_csv(markets_file)
            commodities_df = pd.read_csv(commodities_file)
            
            print(f"\n📊 Existing Maharashtra Metadata:")
            print(f"   Districts: {len(districts_df)}")
            print(f"   Markets: {len(markets_df)}")
            print(f"   Commodities: {len(commodities_df)}")
            print(f"\nUse --force to refresh metadata from API.")
            return 0
        
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
            "page_size": mandi_config["page_size"],
        })
        
        # Create HTTP session
        http_config = config["http"]
        session = create_session(
            max_retries=http_config["max_retries"],
            backoff_factor=http_config["backoff_factor"],
            retry_status_codes=http_config["retry_status_codes"],
            timeout=http_config["timeout"],
        )
        
        # Discover unique values
        results = discover_unique_values(
            session=session,
            api_url=api_url,
            api_key=api_key,
            page_size=mandi_config["page_size"],
            logger=logger,
        )
        
        # Save metadata CSVs
        ensure_directory(output_dir)
        
        district_count = save_metadata_csv(
            results["districts"], districts_file, "district", logger
        )
        market_count = save_metadata_csv(
            results["markets"], markets_file, "market", logger
        )
        commodity_count = save_metadata_csv(
            results["commodities"], commodities_file, "commodity", logger
        )
        
        # Create receipt
        receipt = {
            "discovery_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "state": MAHARASHTRA_STATE_NAME,
            "api_endpoint": api_url,
            "resource_id": resource_id,
            "statistics": {
                "total_records_processed": results["total_processed"],
                "non_maharashtra_dropped": results["non_mh_dropped"],
                "unique_districts": district_count,
                "unique_markets": market_count,
                "unique_commodities": commodity_count,
            },
            "output_files": {
                "districts": str(districts_file.relative_to(PROJECT_ROOT)),
                "markets": str(markets_file.relative_to(PROJECT_ROOT)),
                "commodities": str(commodities_file.relative_to(PROJECT_ROOT)),
            },
        }
        
        receipt_path = output_dir / "discovery_receipt.json"
        save_receipt(receipt_path, receipt)
        logger.info(f"Saved receipt to {receipt_path}")
        
        # Update audit
        audit.add_metric("Total Records Processed", results["total_processed"])
        audit.add_metric("Non-MH Records Dropped", results["non_mh_dropped"])
        audit.add_metric("Unique Districts", district_count)
        audit.add_metric("Unique Markets", market_count)
        audit.add_metric("Unique Commodities", commodity_count)
        
        audit.add_section("Discovered Districts", {
            "count": district_count,
            "list": sorted(list(results["districts"]))[:20],
        })
        
        audit.add_section("Output Files", {
            "districts": str(districts_file),
            "markets": str(markets_file),
            "commodities": str(commodities_file),
            "receipt": str(receipt_path),
        })
        
        if results["non_mh_dropped"] > 0:
            audit.add_warning(
                f"{results['non_mh_dropped']} non-Maharashtra records were found and dropped"
            )
        
        # Save audit
        audit_path = audit.save()
        logger.info(f"Saved audit report to {audit_path}")
        
        # Summary
        logger.info("=" * 70)
        logger.info("✅ Discovery Complete!")
        logger.info(f"   State: {MAHARASHTRA_STATE_NAME}")
        logger.info(f"   Districts: {district_count}")
        logger.info(f"   Markets: {market_count}")
        logger.info(f"   Commodities: {commodity_count}")
        logger.info(f"   Output: {output_dir}")
        logger.info("=" * 70)
        
        print(f"\n✅ Maharashtra Metadata Discovery Complete!")
        print(f"   📍 Districts: {district_count}")
        print(f"   🏪 Markets: {market_count}")
        print(f"   🌾 Commodities: {commodity_count}")
        print(f"\n   📁 Output: {output_dir}")
        print(f"   📋 Audit: {audit_path}")
        
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
