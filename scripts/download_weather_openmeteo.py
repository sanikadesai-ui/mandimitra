#!/usr/bin/env python3
"""
MANDIMITRA - Open-Meteo 16-Day Rainfall Forecast Downloader

Downloads rainfall forecast data from Open-Meteo API.
Supports multiple locations via configs/locations.csv.

Usage:
    python scripts/download_weather_openmeteo.py --lat 28.7041 --lon 77.1025
    python scripts/download_weather_openmeteo.py --location LOC001
    python scripts/download_weather_openmeteo.py --all-locations
    python scripts/download_weather_openmeteo.py --help

Author: MANDIMITRA Team
"""

import argparse
import sys
from datetime import datetime, timezone
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
        description="Download 16-day rainfall forecast from Open-Meteo API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download forecast for specific coordinates
    python scripts/download_weather_openmeteo.py --lat 28.7041 --lon 77.1025
    
    # Download forecast for a location from locations.csv
    python scripts/download_weather_openmeteo.py --location LOC001
    
    # Download forecast for all configured locations
    python scripts/download_weather_openmeteo.py --all-locations
    
    # Download with extended variables (temp, humidity, precipitation probability)
    python scripts/download_weather_openmeteo.py --lat 19.0760 --lon 72.8777 --extended
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
        "--forecast-days",
        type=int,
        default=16,
        help="Number of forecast days (default: 16, max: 16)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include extended variables (temperature, humidity, wind)",
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


def build_openmeteo_params(
    lat: float,
    lon: float,
    daily_variables: List[str],
    forecast_days: int = 16,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """
    Build query parameters for Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        daily_variables: List of daily variables to fetch
        forecast_days: Number of forecast days
        timezone: Timezone setting
        
    Returns:
        Dictionary of query parameters
    """
    return {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(daily_variables),
        "forecast_days": min(forecast_days, 16),  # Max is 16
        "timezone": timezone,
    }


def parse_openmeteo_response(response: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse Open-Meteo API JSON response into DataFrame.
    
    Args:
        response: Raw API response dictionary
        
    Returns:
        DataFrame with date and weather variables
        
    Raises:
        EmptyResponseError: If response contains no data
    """
    # Check for errors
    if "error" in response and response["error"]:
        reason = response.get("reason", "Unknown error")
        raise APIError(f"Open-Meteo API error: {reason}")
    
    daily_data = response.get("daily", {})
    
    if not daily_data:
        raise EmptyResponseError("Open-Meteo returned no daily data")
    
    # Convert to DataFrame
    df = pd.DataFrame(daily_data)
    
    # Rename time column to date
    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
    
    return df


def download_openmeteo_data(
    session,
    api_base: str,
    params: Dict[str, Any],
    location_id: str,
    logger,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Download forecast data from Open-Meteo API.
    
    Args:
        session: HTTP session
        api_base: API base URL
        params: Query parameters
        location_id: Location identifier for logging
        logger: Logger instance
        
    Returns:
        Tuple of (DataFrame, metadata)
    """
    with ProgressLogger(logger, f"Downloading Open-Meteo forecast for {location_id}"):
        response, status = make_request(session, api_base, params=params, logger=logger)
    
    # Parse response
    df = parse_openmeteo_response(response)
    
    # Extract metadata
    metadata = {
        "latitude": response.get("latitude"),
        "longitude": response.get("longitude"),
        "elevation": response.get("elevation"),
        "timezone": response.get("timezone"),
        "timezone_abbreviation": response.get("timezone_abbreviation"),
        "utc_offset_seconds": response.get("utc_offset_seconds"),
        "generationtime_ms": response.get("generationtime_ms"),
        "daily_units": response.get("daily_units", {}),
    }
    
    logger.info(f"Downloaded {len(df)} days of forecast")
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
    daily_variables: List[str],
    forecast_days: int,
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
    logger.info(f"  Forecast days: {forecast_days}")
    
    try:
        validate_coordinates(lat, lon)
        
        # Build API parameters
        openmeteo_config = config["openmeteo"]
        params = build_openmeteo_params(
            lat=lat,
            lon=lon,
            daily_variables=daily_variables,
            forecast_days=forecast_days,
            timezone=openmeteo_config["timezone"],
        )
        
        # Download data
        df, metadata = download_openmeteo_data(
            session=session,
            api_base=openmeteo_config["api_base"],
            params=params,
            location_id=location_id,
            logger=logger,
        )
        
        if df.empty:
            logger.warning(f"No data returned for {location_id}")
            return None
        
        # Build output path with timestamp
        timestamp = get_utc_timestamp_safe()
        filename = f"forecast_{timestamp}.csv"
        output_path = build_weather_path(
            base_dir=output_dir,
            source="openmeteo_forecast",
            location_id=location_id,
            filename=filename,
        )
        ensure_directory(output_path.parent)
        
        # Save CSV
        save_dataframe(df, output_path)
        logger.info(f"Saved data to {output_path}")
        
        # Create and save receipt
        receipt = create_download_receipt(
            dataset_id="OPENMETEO_FORECAST",
            source="Open-Meteo",
            filters={
                "location_id": location_id,
                "location_name": location_name,
                "latitude": lat,
                "longitude": lon,
            },
            url_params={
                "daily_variables": daily_variables,
                "forecast_days": forecast_days,
                "timezone": openmeteo_config["timezone"],
            },
            total_rows=len(df),
            total_pages=1,
            output_file=str(output_path),
            metadata=metadata,
        )
        
        receipt_filename = f"receipt_{timestamp}.json"
        receipt_path = output_path.parent / receipt_filename
        save_receipt(receipt_path, receipt)
        logger.info(f"Saved receipt to {receipt_path}")
        
        return output_path.parent
        
    except Exception as e:
        logger.error(f"Failed to download for {location_id}: {e}")
        return None


def main():
    """Main entry point for Open-Meteo forecast download."""
    args = parse_arguments()
    
    # Validate lat/lon pair
    if args.lat is not None and args.lon is None:
        print("❌ Error: --lon is required when using --lat")
        return 1
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "download.log"
    logger = setup_logger("openmeteo_download", log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("MANDIMITRA - Open-Meteo Rainfall Forecast Downloader")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Determine variables to fetch
        openmeteo_config = config["openmeteo"]
        if args.extended:
            daily_variables = openmeteo_config["daily_variables"] + [
                "temperature_2m_max",
                "temperature_2m_min",
                "relative_humidity_2m_max",
                "relative_humidity_2m_min",
                "wind_speed_10m_max",
            ]
            # Remove duplicates while preserving order
            daily_variables = list(dict.fromkeys(daily_variables))
        else:
            daily_variables = ["precipitation_sum", "precipitation_probability_max"]
        
        logger.info(f"Daily variables: {daily_variables}")
        
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
                daily_variables=daily_variables,
                forecast_days=args.forecast_days,
                output_dir=output_dir,
                logger=logger,
            )
            
            if result:
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info("=" * 60)
        logger.info("✅ Open-Meteo Download Complete!")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info("=" * 60)
        
        print(f"\n✅ Open-Meteo forecast download complete")
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
