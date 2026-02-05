#!/usr/bin/env python3
"""
MANDIMITRA - Mandi Price Data Downloader

Downloads daily commodity prices from AGMARKNET via Data.gov.in API.
Supports filtering by state, district, commodity, market with full pagination.

Usage:
    python scripts/download_mandi.py --state "Maharashtra" --district "Pune" --commodity "Wheat"
    python scripts/download_mandi.py --state "Karnataka" --all-commodities
    python scripts/download_mandi.py --help

Author: MANDIMITRA Team
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import (
    build_mandi_path,
    create_download_receipt,
    ensure_directory,
    load_config,
    save_dataframe,
    save_receipt,
)
from src.utils.http import (
    APIError,
    APIKeyMissingError,
    create_session,
    make_request,
    fetch_all_records,
)
from src.utils.logging_utils import (
    ProgressLogger,
    get_utc_timestamp,
    get_utc_timestamp_safe,
    setup_logger,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download mandi price data from Data.gov.in AGMARKNET API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all data for a specific state
    python scripts/download_mandi.py --state "Maharashtra"
    
    # Download data with multiple filters
    python scripts/download_mandi.py --state "Punjab" --commodity "Wheat" --district "Ludhiana"
    
    # Download with custom page size
    python scripts/download_mandi.py --state "Karnataka" --page-size 5000
    
    # Dry run to check filters without downloading
    python scripts/download_mandi.py --state "Gujarat" --dry-run
        """,
    )
    
    # Filter arguments
    parser.add_argument(
        "--state",
        type=str,
        help="Filter by state name (e.g., 'Maharashtra', 'Punjab')",
    )
    parser.add_argument(
        "--district",
        type=str,
        help="Filter by district name (e.g., 'Pune', 'Ludhiana')",
    )
    parser.add_argument(
        "--commodity",
        type=str,
        help="Filter by commodity name (e.g., 'Wheat', 'Rice')",
    )
    parser.add_argument(
        "--market",
        type=str,
        help="Filter by market/mandi name",
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
        default="data/raw",
        help="Base output directory for downloaded data",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Number of records per API page (default: 1000)",
    )
    
    # Behavior flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check filters and show expected row count without downloading",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only fetch and display dataset schema/metadata",
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
    # Load from .env file
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


def build_api_params(
    api_key: str,
    filters: Dict[str, Optional[str]],
    page_size: int,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    Build query parameters for Data.gov.in API.
    
    Args:
        api_key: Data.gov.in API key
        filters: Dictionary of filter values (state, district, commodity, market)
        page_size: Number of records per page
        offset: Record offset for pagination
        
    Returns:
        Dictionary of query parameters
    """
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": page_size,
        "offset": offset,
    }
    
    # Add filters if provided
    filter_mapping = {
        "state": "state",
        "district": "district", 
        "commodity": "commodity",
        "market": "market",
    }
    
    for key, api_key_name in filter_mapping.items():
        if filters.get(key):
            params[f"filters[{api_key_name}]"] = filters[key]
    
    return params


def fetch_schema(
    session,
    api_url: str,
    api_key: str,
    logger,
) -> Dict[str, Any]:
    """
    Fetch dataset schema and metadata with a limit=1 sanity check.
    
    Args:
        session: HTTP session
        api_url: API endpoint URL
        api_key: API key
        logger: Logger instance
        
    Returns:
        Schema and metadata dictionary
    """
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 1,
        "offset": 0,
    }
    
    logger.info("Fetching dataset schema (limit=1 sanity check)...")
    data, status = make_request(session, api_url, params=params, logger=logger)
    
    # Extract metadata
    schema = {
        "status": data.get("status"),
        "message": data.get("message"),
        "total_records": data.get("total", 0),
        "count": data.get("count", 0),
        "limit": data.get("limit", 0),
        "offset": data.get("offset", 0),
        "index_name": data.get("index_name"),
        "title": data.get("title"),
        "desc": data.get("desc"),
        "org": data.get("org"),
        "org_type": data.get("org_type"),
        "sector": data.get("sector"),
        "source": data.get("source"),
        "catalog_uuid": data.get("catalog_uuid"),
        "version": data.get("version"),
        "created": data.get("created"),
        "updated": data.get("updated"),
    }
    
    # Extract field information from first record
    records = data.get("records", [])
    if records:
        schema["fields"] = list(records[0].keys())
        schema["sample_record"] = records[0]
    
    # Clean None values
    schema = {k: v for k, v in schema.items() if v is not None}
    
    return schema


def download_mandi_data(
    session,
    api_url: str,
    api_key: str,
    filters: Dict[str, Optional[str]],
    page_size: int,
    logger,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Download all mandi price data with pagination.
    
    Args:
        session: HTTP session
        api_url: API endpoint URL
        api_key: API key
        filters: Filter parameters
        page_size: Records per page
        logger: Logger instance
        
    Returns:
        Tuple of (DataFrame, metadata)
    """
    base_params = build_api_params(api_key, filters, page_size)
    
    # Remove pagination params (handled by paginated_fetch)
    base_params.pop("limit", None)
    base_params.pop("offset", None)
    
    with ProgressLogger(logger, "Downloading mandi data") as progress:
        records, metadata = paginated_fetch(
            session=session,
            url=api_url,
            base_params=base_params,
            page_size=page_size,
            total_key="total",
            records_key="records",
            logger=logger,
        )
    
    if not records:
        raise EmptyResponseError("No records downloaded. Check your filters.")
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    logger.info(f"Created DataFrame with {len(df):,} rows and {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df, metadata


def main():
    """Main entry point for mandi data download."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "download.log"
    logger = setup_logger("mandi_download", log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("MANDIMITRA - Mandi Price Data Downloader")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Get API key
        api_key = get_api_key()
        logger.info("API key loaded successfully")
        
        # Build API URL
        mandi_config = config["mandi"]
        resource_id = mandi_config["resource_id"]
        api_url = f"{mandi_config['api_base']}/{resource_id}"
        logger.info(f"API endpoint: {api_url}")
        
        # Collect filters
        filters = {
            "state": args.state,
            "district": args.district,
            "commodity": args.commodity,
            "market": args.market,
        }
        active_filters = {k: v for k, v in filters.items() if v}
        
        if active_filters:
            logger.info(f"Active filters: {active_filters}")
        else:
            logger.warning("No filters specified - downloading ALL available data")
        
        # Create HTTP session
        http_config = config["http"]
        session = create_session(
            max_retries=http_config["max_retries"],
            backoff_factor=http_config["backoff_factor"],
            retry_status_codes=http_config["retry_status_codes"],
            timeout=http_config["timeout"],
        )
        
        # Fetch schema first (sanity check)
        schema = fetch_schema(session, api_url, api_key, logger)
        
        logger.info("-" * 40)
        logger.info("Dataset Metadata:")
        logger.info(f"  Title: {schema.get('title', 'N/A')}")
        logger.info(f"  Organization: {schema.get('org', 'N/A')}")
        logger.info(f"  Total Records: {schema.get('total_records', 0):,}")
        logger.info(f"  Fields: {schema.get('fields', [])}")
        logger.info(f"  Last Updated: {schema.get('updated', 'N/A')}")
        logger.info("-" * 40)
        
        if args.schema_only:
            logger.info("Schema-only mode - exiting without download")
            print("\n📋 Dataset Schema:")
            import json
            print(json.dumps(schema, indent=2, ensure_ascii=False))
            return 0
        
        if args.dry_run:
            logger.info("Dry-run mode - checking filters without full download")
            
            # Make a request with filters to see filtered count
            test_params = build_api_params(api_key, filters, page_size=1, offset=0)
            data, _ = make_request(session, api_url, params=test_params, logger=logger)
            filtered_total = data.get("total", 0)
            
            logger.info(f"Filtered record count: {filtered_total:,}")
            print(f"\n📊 Dry Run Results:")
            print(f"   Total records matching filters: {filtered_total:,}")
            print(f"   Estimated pages: {(filtered_total // args.page_size) + 1}")
            return 0
        
        # Download data
        timestamp = get_utc_timestamp_safe()
        df, metadata = download_mandi_data(
            session=session,
            api_url=api_url,
            api_key=api_key,
            filters=filters,
            page_size=args.page_size,
            logger=logger,
        )
        
        # Build output path
        output_dir = build_mandi_path(
            base_dir=PROJECT_ROOT / args.output_dir,
            state=args.state,
            district=args.district,
            commodity=args.commodity,
            timestamp=timestamp,
        )
        ensure_directory(output_dir)
        
        # Save CSV
        csv_path = output_dir / "mandi.csv"
        save_dataframe(df, csv_path)
        logger.info(f"Saved data to {csv_path}")
        
        # Calculate pages
        total_rows = len(df)
        total_pages = (total_rows // args.page_size) + (1 if total_rows % args.page_size else 0)
        
        # Create and save receipt
        receipt = create_download_receipt(
            dataset_id=resource_id,
            source="data.gov.in",
            filters=active_filters,
            url_params=build_api_params(api_key, filters, args.page_size),
            total_rows=total_rows,
            total_pages=total_pages,
            output_file=str(csv_path.relative_to(PROJECT_ROOT)),
            metadata={
                "title": schema.get("title"),
                "organization": schema.get("org"),
                "fields": schema.get("fields"),
                "api_total_records": schema.get("total_records"),
                "last_updated": schema.get("updated"),
            },
        )
        
        receipt_path = output_dir / "receipt.json"
        save_receipt(receipt_path, receipt)
        logger.info(f"Saved receipt to {receipt_path}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("✅ Download Complete!")
        logger.info(f"   Rows: {total_rows:,}")
        logger.info(f"   Pages: {total_pages}")
        logger.info(f"   Output: {output_dir}")
        logger.info("=" * 60)
        
        print(f"\n✅ Successfully downloaded {total_rows:,} records")
        print(f"   📁 Data: {csv_path}")
        print(f"   📄 Receipt: {receipt_path}")
        
        return 0
        
    except APIKeyMissingError as e:
        logger.error(f"API Key Error: {e}")
        print(f"\n❌ API Key Error: {e}")
        return 1
        
    except EmptyResponseError as e:
        logger.error(f"Empty Response: {e}")
        print(f"\n❌ No data found: {e}")
        return 1
        
    except APIError as e:
        logger.error(f"API Error: {e}")
        print(f"\n❌ API Error: {e}")
        return 2
        
    except FileNotFoundError as e:
        logger.error(f"File Not Found: {e}")
        print(f"\n❌ File Not Found: {e}")
        return 3
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
