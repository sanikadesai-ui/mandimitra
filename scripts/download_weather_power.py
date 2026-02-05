#!/usr/bin/env python3
"""
MANDIMITRA - NASA POWER Historical Rainfall Downloader

Downloads historical precipitation data (PRECTOTCORR) from NASA POWER Daily API.
Supports multiple locations via configs/locations.csv.

Usage:
    python scripts/download_weather_power.py --lat 28.7041 --lon 77.1025 --start 20230101 --end 20231231
    python scripts/download_weather_power.py --location LOC001 --days-back 365
    python scripts/download_weather_power.py --all-locations --days-back 180
    python scripts/download_weather_power.py --help

Author: MANDIMITRA Team
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import (
    build_weather_path,
    create_download_receipt,
    ensure_directory,
    load_config,
    load_locations,
    save_dataframe,
    save_receipt,
    sanitize_filename,
)
from src.utils.http import (
    APIError,
    create_session,
    make_request,
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
        description="Download historical rainfall data from NASA POWER API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download for specific coordinates and date range
    python scripts/download_weather_power.py --lat 28.7041 --lon 77.1025 --start 20230101 --end 20231231
    
    # Download for a location from locations.csv
    python scripts/download_weather_power.py --location LOC001 --days-back 365
    
    # Download last 180 days for all configured locations
    python scripts/download_weather_power.py --all-locations --days-back 180
    
    # Download with extended parameters (temp, humidity)
    python scripts/download_weather_power.py --lat 19.0760 --lon 72.8777 --days-back 90 --extended
        """,
    )
    
    # Location specification (mutually exclusive)
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument(
        "--lat",
        type=float,
        help="Latitude (-90 to 90)",
    )
    location_group.add_argument(
        "--location",
        type=str,
        help="Location ID from configs/locations.csv",
    )
    location_group.add_argument(
        "--all-locations",
        action="store_true",
        help="Download for all locations in configs/locations.csv",
    )
    
    parser.add_argument(
        "--lon",
        type=float,
        help="Longitude (-180 to 180). Required with --lat",
    )
    
    # Date range specification
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--start",
        type=str,
        help="Start date in YYYYMMDD format",
    )
    date_group.add_argument(
        "--days-back",
        type=int,
        default=365,
        help="Number of days back from today (default: 365)",
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="End date in YYYYMMDD format (default: yesterday)",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project configuration file",
    )
    parser.add_argument(
        "--locations-file",
        type=str,
        default="configs/locations.csv",
        help="Path to locations CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Base output directory for downloaded data",
    )
    
    # Options
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include extended parameters (T2M, RH2M) in addition to precipitation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug output",
    )
    
    return parser.parse_args()


def validate_coordinates(lat: float, lon: float) -> None:
    """
    Validate latitude and longitude values.
    
    Raises:
        ValueError: If coordinates are invalid
    """
    if not -90 <= lat <= 90:
        raise ValueError(f"Invalid latitude: {lat}. Must be between -90 and 90.")
    if not -180 <= lon <= 180:
        raise ValueError(f"Invalid longitude: {lon}. Must be between -180 and 180.")


def get_date_range(
    start: Optional[str],
    end: Optional[str],
    days_back: int,
) -> tuple[str, str]:
    """
    Calculate date range for API request.
    
    Args:
        start: Start date string (YYYYMMDD) or None
        end: End date string (YYYYMMDD) or None
        days_back: Days back from today if start not specified
        
    Returns:
        Tuple of (start_date, end_date) in YYYYMMDD format
    """
    # End date defaults to yesterday (NASA POWER has ~2 day lag)
    if end is None:
        end_date = datetime.now(timezone.utc).date() - timedelta(days=2)
    else:
        end_date = datetime.strptime(end, "%Y%m%d").date()
    
    # Start date from explicit value or days_back
    if start is not None:
        start_date = datetime.strptime(start, "%Y%m%d").date()
    else:
        start_date = end_date - timedelta(days=days_back)
    
    return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")


def build_power_api_url(
    api_base: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    parameters: List[str],
    community: str = "AG",
) -> str:
    """
    Build NASA POWER API URL.
    
    Args:
        api_base: Base API URL
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        parameters: List of parameters to fetch
        community: NASA POWER community (AG, RE, SB)
        
    Returns:
        Complete API URL
    """
    params_str = ",".join(parameters)
    return (
        f"{api_base}?"
        f"parameters={params_str}&"
        f"community={community}&"
        f"longitude={lon}&"
        f"latitude={lat}&"
        f"start={start_date}&"
        f"end={end_date}&"
        f"format=JSON"
    )


def parse_power_response(response: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse NASA POWER API JSON response into DataFrame.
    
    Args:
        response: Raw API response dictionary
        
    Returns:
        DataFrame with date index and parameter columns
        
    Raises:
        EmptyResponseError: If response contains no data
    """
    # Check for errors
    if "messages" in response and response.get("messages"):
        messages = response["messages"]
        if isinstance(messages, list) and messages:
            raise APIError(f"NASA POWER API error: {messages}")
    
    # Extract data
    properties = response.get("properties", {})
    parameter_data = properties.get("parameter", {})
    
    if not parameter_data:
        raise EmptyResponseError("NASA POWER returned no parameter data")
    
    # Convert to DataFrame
    df = pd.DataFrame(parameter_data)
    
    # Parse date index
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    
    # Reset index to make date a column
    df = df.reset_index()
    
    # Handle missing data (-999 is NASA POWER's missing value indicator)
    for col in df.columns:
        if col != "date":
            df[col] = df[col].replace(-999, pd.NA)
    
    return df


def download_power_data(
    session,
    api_url: str,
    location_id: str,
    logger,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Download data from NASA POWER API.
    
    Args:
        session: HTTP session
        api_url: Complete API URL
        location_id: Location identifier for logging
        logger: Logger instance
        
    Returns:
        Tuple of (DataFrame, metadata)
    """
    with ProgressLogger(logger, f"Downloading NASA POWER data for {location_id}"):
        response, status = make_request(session, api_url, logger=logger)
    
    # Parse response
    df = parse_power_response(response)
    
    # Extract metadata
    metadata = {
        "header": response.get("header", {}),
        "geometry": response.get("geometry", {}),
        "parameters_info": response.get("parameters", {}),
    }
    
    logger.info(f"Downloaded {len(df)} days of data")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df, metadata


def process_single_location(
    session,
    config: Dict[str, Any],
    location_id: str,
    location_name: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    parameters: List[str],
    output_dir: Path,
    logger,
) -> Optional[Path]:
    """
    Process download for a single location.
    
    Returns:
        Output directory path if successful, None if failed
    """
    logger.info("-" * 40)
    logger.info(f"Processing: {location_name} ({location_id})")
    logger.info(f"  Coordinates: {lat}, {lon}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    
    try:
        validate_coordinates(lat, lon)
        
        # Build API URL
        power_config = config["nasa_power"]
        api_url = build_power_api_url(
            api_base=power_config["api_base"],
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
            community=power_config["community"],
        )
        
        # Download data
        df, metadata = download_power_data(session, api_url, location_id, logger)
        
        if df.empty:
            logger.warning(f"No data returned for {location_id}")
            return None
        
        # Build output path
        filename = f"power_daily_{start_date}_{end_date}.csv"
        output_path = build_weather_path(
            base_dir=output_dir,
            source="power_daily",
            location_id=location_id,
            filename=filename,
        )
        ensure_directory(output_path.parent)
        
        # Save CSV
        save_dataframe(df, output_path)
        logger.info(f"Saved data to {output_path}")
        
        # Create and save receipt
        receipt = create_download_receipt(
            dataset_id="NASA_POWER_DAILY",
            source="NASA POWER",
            filters={
                "location_id": location_id,
                "location_name": location_name,
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
            },
            url_params={
                "parameters": parameters,
                "community": power_config["community"],
            },
            total_rows=len(df),
            total_pages=1,
            output_file=str(output_path),
            metadata=metadata,
        )
        
        receipt_filename = f"receipt_{start_date}_{end_date}.json"
        receipt_path = output_path.parent / receipt_filename
        save_receipt(receipt_path, receipt)
        logger.info(f"Saved receipt to {receipt_path}")
        
        return output_path.parent
        
    except Exception as e:
        logger.error(f"Failed to download for {location_id}: {e}")
        return None


def main():
    """Main entry point for NASA POWER data download."""
    args = parse_arguments()
    
    # Validate lat/lon pair
    if args.lat is not None and args.lon is None:
        print("❌ Error: --lon is required when using --lat")
        return 1
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "download.log"
    logger = setup_logger("power_download", log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("MANDIMITRA - NASA POWER Historical Rainfall Downloader")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Determine parameters to fetch
        power_config = config["nasa_power"]
        if args.extended:
            parameters = power_config["parameters"]
        else:
            parameters = ["PRECTOTCORR"]  # Just precipitation
        
        logger.info(f"Parameters: {parameters}")
        
        # Calculate date range
        start_date, end_date = get_date_range(args.start, args.end, args.days_back)
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Create HTTP session
        http_config = config["http"]
        session = create_session(
            max_retries=http_config["max_retries"],
            backoff_factor=http_config["backoff_factor"],
            retry_status_codes=http_config["retry_status_codes"],
            timeout=http_config["timeout"],
        )
        
        # Collect locations to process
        locations = []
        
        if args.lat is not None:
            # Single coordinate pair
            locations.append({
                "location_id": f"custom_{args.lat}_{args.lon}",
                "location_name": f"Custom ({args.lat}, {args.lon})",
                "latitude": args.lat,
                "longitude": args.lon,
            })
        elif args.location:
            # Single location from file
            locations_df = load_locations(PROJECT_ROOT / args.locations_file)
            loc_row = locations_df[locations_df["location_id"] == args.location]
            
            if loc_row.empty:
                logger.error(f"Location ID '{args.location}' not found in {args.locations_file}")
                return 1
            
            loc = loc_row.iloc[0]
            locations.append({
                "location_id": loc["location_id"],
                "location_name": loc["location_name"],
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
            })
        else:
            # All locations from file
            locations_df = load_locations(PROJECT_ROOT / args.locations_file)
            for _, loc in locations_df.iterrows():
                locations.append({
                    "location_id": loc["location_id"],
                    "location_name": loc["location_name"],
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                })
        
        logger.info(f"Processing {len(locations)} location(s)")
        
        # Process each location
        output_dir = PROJECT_ROOT / args.output_dir
        successful = 0
        failed = 0
        
        for loc in locations:
            result = process_single_location(
                session=session,
                config=config,
                location_id=loc["location_id"],
                location_name=loc["location_name"],
                lat=loc["latitude"],
                lon=loc["longitude"],
                start_date=start_date,
                end_date=end_date,
                parameters=parameters,
                output_dir=output_dir,
                logger=logger,
            )
            
            if result:
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info("=" * 60)
        logger.info("✅ NASA POWER Download Complete!")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info("=" * 60)
        
        print(f"\n✅ NASA POWER download complete")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        
        return 0 if failed == 0 else 1
        
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
