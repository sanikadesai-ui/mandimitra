#!/usr/bin/env python3
"""
MANDIMITRA - Maharashtra Weather Data Downloader

Downloads weather data EXCLUSIVELY for Maharashtra district headquarters:
- Historical rainfall from NASA POWER (PRECTOTCORR)
- 16-day forecast from Open-Meteo

⚠️  HARD CONSTRAINT: This script ONLY downloads data for Maharashtra locations.

Usage:
    python scripts/download_weather_maharashtra.py --power --all-districts
    python scripts/download_weather_maharashtra.py --openmeteo --all-districts
    python scripts/download_weather_maharashtra.py --all --all-districts
    python scripts/download_weather_maharashtra.py --power --district "Pune"
    python scripts/download_weather_maharashtra.py --help

Output:
    data/raw/weather/power_daily/maharashtra/<district>/power_daily_*.csv
    data/raw/weather/openmeteo_forecast/maharashtra/<district>/forecast_*.csv

Author: MANDIMITRA Team
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
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
    EmptyResponseError,
    create_session,
    make_request,
)
from src.utils.logging_utils import (
    ProgressLogger,
    get_utc_timestamp_safe,
    setup_logger,
)
from src.utils.maharashtra import MAHARASHTRA_STATE_NAME
from src.utils.progress import ProgressTracker
from src.utils.audit import AuditLogger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download weather data for Maharashtra districts (MAHARASHTRA ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  HARD CONSTRAINT: This script ONLY downloads data for Maharashtra locations.
    Location coordinates are taken from configs/maharashtra_locations.csv.

Data Sources:
    --power       NASA POWER historical rainfall (default: 365 days)
    --openmeteo   Open-Meteo 16-day forecast
    --all         Download from both sources

Coverage:
    --all-districts   Download for all 36 Maharashtra districts
    --district NAME   Download for specific district only

Examples:
    # Download NASA POWER data for all Maharashtra districts (1 year)
    python scripts/download_weather_maharashtra.py --power --all-districts
    
    # Download Open-Meteo forecast for all districts
    python scripts/download_weather_maharashtra.py --openmeteo --all-districts
    
    # Download both for specific district
    python scripts/download_weather_maharashtra.py --all --district "Pune"
    
    # Download NASA POWER with custom date range
    python scripts/download_weather_maharashtra.py --power --all-districts --days-back 730
        """,
    )
    
    # Data source
    source_group = parser.add_argument_group("Data Source")
    source_group.add_argument(
        "--power",
        action="store_true",
        help="Download NASA POWER historical data",
    )
    source_group.add_argument(
        "--openmeteo",
        action="store_true",
        help="Download Open-Meteo forecast data",
    )
    source_group.add_argument(
        "--all",
        action="store_true",
        help="Download from both sources",
    )
    
    # Coverage (NO state argument - Maharashtra only!)
    coverage_group = parser.add_argument_group("Coverage")
    coverage_group.add_argument(
        "--all-districts",
        action="store_true",
        help="Download for all 36 Maharashtra districts",
    )
    coverage_group.add_argument(
        "--district",
        type=str,
        help="Download for specific Maharashtra district",
    )
    
    # NASA POWER options
    power_group = parser.add_argument_group("NASA POWER Options")
    power_group.add_argument(
        "--days-back",
        type=int,
        default=365,
        help="Days of historical data (default: 365)",
    )
    power_group.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYYMMDD format)",
    )
    power_group.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYYMMDD format)",
    )
    
    # Open-Meteo options
    meteo_group = parser.add_argument_group("Open-Meteo Options")
    meteo_group.add_argument(
        "--forecast-days",
        type=int,
        default=16,
        help="Forecast days (default: 16, max: 16)",
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
        help="Restart from scratch (ignore progress)",
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
        default="configs/maharashtra_locations.csv",
        help="Path to Maharashtra locations CSV",
    )
    
    # Options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug output",
    )
    
    return parser.parse_args()


def load_maharashtra_locations(locations_file: Path) -> pd.DataFrame:
    """
    Load Maharashtra district locations.
    
    Args:
        locations_file: Path to locations CSV
        
    Returns:
        DataFrame with location data
    """
    if not locations_file.exists():
        raise FileNotFoundError(
            f"Maharashtra locations file not found: {locations_file}\n"
            "This file should contain coordinates for all 36 Maharashtra district HQs."
        )
    
    df = pd.read_csv(locations_file)
    
    # Validate required columns
    required = ["location_id", "district", "latitude", "longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in locations file: {missing}")
    
    return df


def get_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    days_back: int,
) -> Tuple[str, str]:
    """Calculate date range for NASA POWER API."""
    # NASA POWER has ~2-3 day lag
    if end_date is None:
        end_dt = datetime.now(timezone.utc).date() - timedelta(days=3)
    else:
        end_dt = datetime.strptime(end_date, "%Y%m%d").date()
    
    if start_date is not None:
        start_dt = datetime.strptime(start_date, "%Y%m%d").date()
    else:
        start_dt = end_dt - timedelta(days=days_back)
    
    return start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")


def download_power_data(
    session,
    api_base: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    parameters: List[str],
    logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Download historical data from NASA POWER API."""
    params_str = ",".join(parameters)
    url = (
        f"{api_base}?"
        f"parameters={params_str}&"
        f"community=AG&"
        f"longitude={lon}&"
        f"latitude={lat}&"
        f"start={start_date}&"
        f"end={end_date}&"
        f"format=JSON"
    )
    
    response, _ = make_request(session, url, logger=logger)
    
    # Check for errors
    if "messages" in response and response.get("messages"):
        raise APIError(f"NASA POWER error: {response['messages']}")
    
    # Extract data
    properties = response.get("properties", {})
    parameter_data = properties.get("parameter", {})
    
    if not parameter_data:
        raise EmptyResponseError("NASA POWER returned no data")
    
    # Convert to DataFrame
    df = pd.DataFrame(parameter_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.reset_index()
    
    # Handle missing values (-999)
    for col in df.columns:
        if col != "date":
            df[col] = df[col].replace(-999, pd.NA)
    
    metadata = {
        "header": response.get("header", {}),
        "geometry": response.get("geometry", {}),
    }
    
    return df, metadata


def download_openmeteo_data(
    session,
    api_base: str,
    lat: float,
    lon: float,
    daily_variables: List[str],
    forecast_days: int,
    timezone_str: str,
    logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Download forecast from Open-Meteo API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(daily_variables),
        "forecast_days": min(forecast_days, 16),
        "timezone": timezone_str,
    }
    
    response, _ = make_request(session, api_base, params=params, logger=logger)
    
    if "error" in response and response["error"]:
        raise APIError(f"Open-Meteo error: {response.get('reason', 'Unknown')}")
    
    daily_data = response.get("daily", {})
    if not daily_data:
        raise EmptyResponseError("Open-Meteo returned no data")
    
    df = pd.DataFrame(daily_data)
    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
    
    metadata = {
        "latitude": response.get("latitude"),
        "longitude": response.get("longitude"),
        "elevation": response.get("elevation"),
        "timezone": response.get("timezone"),
        "daily_units": response.get("daily_units", {}),
    }
    
    return df, metadata


def process_power_location(
    session,
    config: Dict[str, Any],
    location: Dict[str, Any],
    start_date: str,
    end_date: str,
    output_dir: Path,
    logger,
) -> Optional[Path]:
    """Process NASA POWER download for one location."""
    district = location["district"]
    lat = location["latitude"]
    lon = location["longitude"]
    
    logger.info(f"  Downloading NASA POWER for {district} ({lat}, {lon})")
    
    try:
        power_config = config["nasa_power"]
        
        df, metadata = download_power_data(
            session=session,
            api_base=power_config["api_base"],
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            parameters=power_config["parameters"],
            logger=logger,
        )
        
        if df.empty:
            logger.warning(f"No data for {district}")
            return None
        
        # Save
        district_dir = output_dir / sanitize_filename(district)
        ensure_directory(district_dir)
        
        filename = f"power_daily_{start_date}_{end_date}.csv"
        output_path = district_dir / filename
        save_dataframe(df, output_path)
        
        # Receipt
        receipt = {
            "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "state": MAHARASHTRA_STATE_NAME,
            "district": district,
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "parameters": power_config["parameters"],
            "rows": len(df),
            "source": "NASA POWER",
        }
        
        receipt_path = district_dir / f"receipt_{start_date}_{end_date}.json"
        save_receipt(receipt_path, receipt)
        
        logger.info(f"  ✓ Saved {len(df)} days to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"  ✗ Failed for {district}: {e}")
        return None


def process_openmeteo_location(
    session,
    config: Dict[str, Any],
    location: Dict[str, Any],
    forecast_days: int,
    output_dir: Path,
    timestamp: str,
    logger,
) -> Optional[Path]:
    """Process Open-Meteo download for one location."""
    district = location["district"]
    lat = location["latitude"]
    lon = location["longitude"]
    
    logger.info(f"  Downloading Open-Meteo for {district} ({lat}, {lon})")
    
    try:
        meteo_config = config["openmeteo"]
        
        df, metadata = download_openmeteo_data(
            session=session,
            api_base=meteo_config["api_base"],
            lat=lat,
            lon=lon,
            daily_variables=meteo_config["daily_variables"],
            forecast_days=forecast_days,
            timezone_str=meteo_config["timezone"],
            logger=logger,
        )
        
        if df.empty:
            logger.warning(f"No data for {district}")
            return None
        
        # Save
        district_dir = output_dir / sanitize_filename(district)
        ensure_directory(district_dir)
        
        filename = f"forecast_{timestamp}.csv"
        output_path = district_dir / filename
        save_dataframe(df, output_path)
        
        # Receipt
        receipt = {
            "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "state": MAHARASHTRA_STATE_NAME,
            "district": district,
            "latitude": lat,
            "longitude": lon,
            "forecast_days": forecast_days,
            "daily_variables": meteo_config["daily_variables"],
            "rows": len(df),
            "source": "Open-Meteo",
        }
        
        receipt_path = district_dir / f"receipt_{timestamp}.json"
        save_receipt(receipt_path, receipt)
        
        logger.info(f"  ✓ Saved {len(df)} days to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"  ✗ Failed for {district}: {e}")
        return None


def main():
    """Main entry point for Maharashtra weather download."""
    args = parse_arguments()
    
    # Validate arguments
    if not args.power and not args.openmeteo and not args.all:
        print("❌ Error: Specify data source: --power, --openmeteo, or --all")
        return 1
    
    if not args.all_districts and not args.district:
        print("❌ Error: Specify coverage: --all-districts or --district NAME")
        return 1
    
    # Setup
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "download.log"
    logger = setup_logger("mh_weather_download", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("weather_download", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Maharashtra Weather Data Downloader")
    logger.info("⚠️  HARD CONSTRAINT: MAHARASHTRA ONLY")
    logger.info("=" * 70)
    
    stats = {
        "power_success": 0,
        "power_failed": 0,
        "openmeteo_success": 0,
        "openmeteo_failed": 0,
    }
    
    try:
        # Load configuration
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Load Maharashtra locations
        locations_file = PROJECT_ROOT / args.locations_file
        locations_df = load_maharashtra_locations(locations_file)
        logger.info(f"Loaded {len(locations_df)} Maharashtra locations")
        
        # Filter locations if specific district requested
        if args.district:
            locations_df = locations_df[locations_df["district"].str.lower() == args.district.lower()]
            if locations_df.empty:
                logger.error(f"District '{args.district}' not found in locations file")
                return 1
            logger.info(f"Filtered to district: {args.district}")
        
        audit.add_section("Configuration", {
            "state": MAHARASHTRA_STATE_NAME,
            "locations_file": str(locations_file),
            "total_locations": len(locations_df),
            "download_power": args.power or args.all,
            "download_openmeteo": args.openmeteo or args.all,
        })
        
        # Create HTTP session
        http_config = config["http"]
        session = create_session(
            max_retries=http_config["max_retries"],
            backoff_factor=http_config["backoff_factor"],
            retry_status_codes=http_config["retry_status_codes"],
            timeout=http_config["timeout"],
        )
        
        # Progress tracker
        progress_file = PROJECT_ROOT / "data" / "metadata" / "maharashtra" / "weather_progress.json"
        tracker = ProgressTracker(progress_file)
        
        # === NASA POWER Downloads ===
        if args.power or args.all:
            logger.info("\n" + "=" * 50)
            logger.info("NASA POWER Historical Rainfall")
            logger.info("=" * 50)
            
            start_date, end_date = get_date_range(args.start_date, args.end_date, args.days_back)
            logger.info(f"Date range: {start_date} to {end_date}")
            
            power_output = PROJECT_ROOT / config["paths"]["maharashtra"]["weather_power"]
            
            # Track progress
            district_ids = locations_df["location_id"].tolist()
            tracker.start_session(
                "power_download",
                chunks=district_ids,
                metadata={"start_date": start_date, "end_date": end_date},
                force_restart=args.force,
            )
            
            pending = district_ids if args.force else tracker.get_pending_chunks("power_download")
            
            with ProgressLogger(logger, "NASA POWER downloads") as progress:
                for _, loc in locations_df.iterrows():
                    if loc["location_id"] not in pending:
                        logger.debug(f"Skipping {loc['district']} (already completed)")
                        continue
                    
                    tracker.mark_in_progress("power_download", loc["location_id"])
                    
                    result = process_power_location(
                        session=session,
                        config=config,
                        location=loc.to_dict(),
                        start_date=start_date,
                        end_date=end_date,
                        output_dir=power_output,
                        logger=logger,
                    )
                    
                    if result:
                        tracker.mark_completed("power_download", loc["location_id"], output_file=str(result))
                        stats["power_success"] += 1
                    else:
                        tracker.mark_failed("power_download", loc["location_id"], "No data")
                        stats["power_failed"] += 1
                    
                    progress.update(f"Completed {stats['power_success']}/{len(locations_df)}")
                    time.sleep(1)  # Rate limiting
        
        # === Open-Meteo Downloads ===
        if args.openmeteo or args.all:
            logger.info("\n" + "=" * 50)
            logger.info("Open-Meteo 16-Day Forecast")
            logger.info("=" * 50)
            
            meteo_output = PROJECT_ROOT / config["paths"]["maharashtra"]["weather_openmeteo"]
            
            # Track progress
            district_ids = locations_df["location_id"].tolist()
            tracker.start_session(
                "openmeteo_download",
                chunks=district_ids,
                metadata={"forecast_days": args.forecast_days},
                force_restart=args.force,
            )
            
            pending = district_ids if args.force else tracker.get_pending_chunks("openmeteo_download")
            
            with ProgressLogger(logger, "Open-Meteo downloads") as progress:
                for _, loc in locations_df.iterrows():
                    if loc["location_id"] not in pending:
                        logger.debug(f"Skipping {loc['district']} (already completed)")
                        continue
                    
                    tracker.mark_in_progress("openmeteo_download", loc["location_id"])
                    
                    result = process_openmeteo_location(
                        session=session,
                        config=config,
                        location=loc.to_dict(),
                        forecast_days=args.forecast_days,
                        output_dir=meteo_output,
                        timestamp=timestamp,
                        logger=logger,
                    )
                    
                    if result:
                        tracker.mark_completed("openmeteo_download", loc["location_id"], output_file=str(result))
                        stats["openmeteo_success"] += 1
                    else:
                        tracker.mark_failed("openmeteo_download", loc["location_id"], "No data")
                        stats["openmeteo_failed"] += 1
                    
                    progress.update(f"Completed {stats['openmeteo_success']}/{len(locations_df)}")
                    time.sleep(0.5)  # Rate limiting
        
        # Audit summary
        audit.add_metric("NASA POWER Success", stats["power_success"])
        audit.add_metric("NASA POWER Failed", stats["power_failed"])
        audit.add_metric("Open-Meteo Success", stats["openmeteo_success"])
        audit.add_metric("Open-Meteo Failed", stats["openmeteo_failed"])
        
        if stats["power_failed"] > 0:
            audit.add_warning(f"{stats['power_failed']} NASA POWER downloads failed")
        if stats["openmeteo_failed"] > 0:
            audit.add_warning(f"{stats['openmeteo_failed']} Open-Meteo downloads failed")
        
        audit_path = audit.save()
        
        # Summary
        logger.info("=" * 70)
        logger.info("✅ Maharashtra Weather Download Complete!")
        logger.info(f"   State: {MAHARASHTRA_STATE_NAME}")
        logger.info(f"   NASA POWER: {stats['power_success']} success, {stats['power_failed']} failed")
        logger.info(f"   Open-Meteo: {stats['openmeteo_success']} success, {stats['openmeteo_failed']} failed")
        logger.info("=" * 70)
        
        print(f"\n✅ Maharashtra Weather Download Complete!")
        print(f"   🌧️ NASA POWER: {stats['power_success']} success, {stats['power_failed']} failed")
        print(f"   ☁️ Open-Meteo: {stats['openmeteo_success']} success, {stats['openmeteo_failed']} failed")
        print(f"   📋 Audit: {audit_path}")
        
        total_failed = stats["power_failed"] + stats["openmeteo_failed"]
        return 1 if total_failed > 0 else 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ File not found: {e}")
        return 3
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ Unexpected error: {e}")
        return 99


if __name__ == "__main__":
    sys.exit(main())
