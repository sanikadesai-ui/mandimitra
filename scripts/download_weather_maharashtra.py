#!/usr/bin/env python3
"""
MANDIMITRA - Maharashtra Weather Data Downloader (Production-Grade)

Downloads weather data EXCLUSIVELY for Maharashtra district headquarters:
- Historical rainfall from NASA POWER (PRECTOTCORR)
- 16-day forecast from Open-Meteo

⚠️  HARD CONSTRAINT: This script ONLY downloads data for Maharashtra locations.

OPTIMIZATIONS:
- Parallel downloads with ThreadPoolExecutor (--max-workers)
- Adaptive rate limiting with 429 handling
- Batched progress saves (atomic writes)
- Connection pooling per worker
- Summary logging instead of per-record

Usage:
    python scripts/download_weather_maharashtra.py --power --all-districts
    python scripts/download_weather_maharashtra.py --openmeteo --all-districts --max-workers 4
    python scripts/download_weather_maharashtra.py --all --all-districts
    python scripts/download_weather_maharashtra.py --power --district "Pune"
    python scripts/download_weather_maharashtra.py --help

Output:
    data/raw/weather/power_daily/maharashtra/<district>/power_daily_*.csv
    data/raw/weather/openmeteo_forecast/maharashtra/<district>/forecast_*.csv

Author: MANDIMITRA Team
Version: 2.0.0 (Production Refactor)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.utils.http import (
    APIError,
    RateLimitMode,
    AdaptiveRateLimiter,
    create_session,
    make_request,
)
from src.utils.logging_utils import (
    get_utc_timestamp_safe,
    setup_logger,
)
from src.utils.maharashtra import MAHARASHTRA_STATE_NAME
from src.utils.progress import ProgressTracker
from src.utils.audit import AuditLogger


# =============================================================================
# CLI
# =============================================================================

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

OPTIMIZATIONS:
    - Parallel downloads with --max-workers (default: 1, max: 4)
    - Adaptive rate limiting handles API throttling automatically
    - Batched progress saves prevent data corruption

Examples:
    # Download NASA POWER for all districts (parallel, 2 workers)
    python scripts/download_weather_maharashtra.py --power --all-districts --max-workers 2
    
    # Download Open-Meteo forecast for all districts
    python scripts/download_weather_maharashtra.py --openmeteo --all-districts
    
    # Download both for specific district
    python scripts/download_weather_maharashtra.py --all --district "Pune"
    
    # Resume interrupted download
    python scripts/download_weather_maharashtra.py --power --all-districts --resume
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
    
    # Concurrency
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers for downloads (default: 1, max: 4)",
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
        default=1.0,
        help="Base delay between requests (default: 1.0s for weather APIs)",
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


# =============================================================================
# HELPERS
# =============================================================================

def load_maharashtra_locations(locations_file: Path) -> pd.DataFrame:
    """Load Maharashtra district locations with CSV comment handling."""
    if not locations_file.exists():
        raise FileNotFoundError(
            f"Maharashtra locations file not found: {locations_file}\n"
            "This file should contain coordinates for all 36 Maharashtra district HQs."
        )
    
    df = pd.read_csv(locations_file, comment="#", skip_blank_lines=True)
    
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


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_power_data(
    session,
    api_base: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    parameters: List[str],
    rate_limiter: AdaptiveRateLimiter,
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
    
    response, _ = make_request(session, url, logger=logger, rate_limiter=rate_limiter)
    
    # Check for errors
    if "messages" in response and response.get("messages"):
        raise APIError(f"NASA POWER error: {response['messages']}")
    
    # Extract data
    properties = response.get("properties", {})
    parameter_data = properties.get("parameter", {})
    
    if not parameter_data:
        return pd.DataFrame(), {}
    
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
    rate_limiter: AdaptiveRateLimiter,
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
    
    response, _ = make_request(session, api_base, params=params, logger=logger, rate_limiter=rate_limiter)
    
    if "error" in response and response["error"]:
        raise APIError(f"Open-Meteo error: {response.get('reason', 'Unknown')}")
    
    daily_data = response.get("daily", {})
    if not daily_data:
        return pd.DataFrame(), {}
    
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


# =============================================================================
# WORKER FUNCTIONS (Thread-Safe)
# =============================================================================

def power_worker(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel NASA POWER downloads.
    Thread-safe: creates own session per worker.
    """
    (
        location, api_base, parameters, start_date, end_date,
        output_dir, rate_limiter, http_config, logger_name
    ) = args_tuple
    
    district = location["district"]
    lat = location["latitude"]
    lon = location["longitude"]
    location_id = location["location_id"]
    
    result = {
        "location_id": location_id,
        "district": district,
        "success": False,
        "rows": 0,
        "error": None,
        "output_file": None,
        "duration_seconds": 0,
    }
    
    # Create thread-local session
    session = create_session(
        max_retries=http_config["max_retries"],
        backoff_factor=http_config["backoff_factor"],
        retry_status_codes=http_config["retry_status_codes"],
        timeout=http_config.get("timeout", 60),
        pool_connections=2,
        pool_maxsize=5,
    )
    
    logger = setup_logger(
        f"{logger_name}_power_{district}",
        PROJECT_ROOT / "logs" / "download.log",
        level="INFO"
    )
    
    start_time = time.time()
    
    try:
        df, metadata = download_power_data(
            session=session,
            api_base=api_base,
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
            rate_limiter=rate_limiter,
            logger=logger,
        )
        
        result["duration_seconds"] = time.time() - start_time
        
        if df.empty:
            result["success"] = True
            return result
        
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
            "parameters": parameters,
            "rows": len(df),
            "duration_seconds": round(result["duration_seconds"], 2),
            "source": "NASA POWER",
        }
        
        receipt_path = district_dir / f"receipt_{start_date}_{end_date}.json"
        save_receipt(receipt_path, receipt)
        
        result["success"] = True
        result["rows"] = len(df)
        result["output_file"] = str(output_path)
        logger.info(f"✓ {district}: {len(df)} days")
        
    except Exception as e:
        result["error"] = str(e)
        result["duration_seconds"] = time.time() - start_time
        logger.error(f"✗ {district}: {e}")
    
    return result


def openmeteo_worker(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel Open-Meteo downloads.
    Thread-safe: creates own session per worker.
    """
    (
        location, api_base, daily_variables, forecast_days, timezone_str,
        output_dir, timestamp, rate_limiter, http_config, logger_name
    ) = args_tuple
    
    district = location["district"]
    lat = location["latitude"]
    lon = location["longitude"]
    location_id = location["location_id"]
    
    result = {
        "location_id": location_id,
        "district": district,
        "success": False,
        "rows": 0,
        "error": None,
        "output_file": None,
        "duration_seconds": 0,
    }
    
    # Create thread-local session
    session = create_session(
        max_retries=http_config["max_retries"],
        backoff_factor=http_config["backoff_factor"],
        retry_status_codes=http_config["retry_status_codes"],
        timeout=http_config.get("timeout", 60),
        pool_connections=2,
        pool_maxsize=5,
    )
    
    logger = setup_logger(
        f"{logger_name}_meteo_{district}",
        PROJECT_ROOT / "logs" / "download.log",
        level="INFO"
    )
    
    start_time = time.time()
    
    try:
        df, metadata = download_openmeteo_data(
            session=session,
            api_base=api_base,
            lat=lat,
            lon=lon,
            daily_variables=daily_variables,
            forecast_days=forecast_days,
            timezone_str=timezone_str,
            rate_limiter=rate_limiter,
            logger=logger,
        )
        
        result["duration_seconds"] = time.time() - start_time
        
        if df.empty:
            result["success"] = True
            return result
        
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
            "daily_variables": daily_variables,
            "rows": len(df),
            "duration_seconds": round(result["duration_seconds"], 2),
            "source": "Open-Meteo",
        }
        
        receipt_path = district_dir / f"receipt_{timestamp}.json"
        save_receipt(receipt_path, receipt)
        
        result["success"] = True
        result["rows"] = len(df)
        result["output_file"] = str(output_path)
        logger.info(f"✓ {district}: {len(df)} days")
        
    except Exception as e:
        result["error"] = str(e)
        result["duration_seconds"] = time.time() - start_time
        logger.error(f"✗ {district}: {e}")
    
    return result


# =============================================================================
# MAIN
# =============================================================================

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
    
    # Validate workers (weather APIs are more rate-limited)
    max_workers = min(args.max_workers, 4)
    
    # Setup
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "download.log"
    logger = setup_logger("mh_weather", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("weather_download", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Maharashtra Weather Data Downloader")
    logger.info(f"Workers: {max_workers} | Rate limit: {args.rate_limit}")
    logger.info("⚠️  HARD CONSTRAINT: MAHARASHTRA ONLY")
    logger.info("=" * 70)
    
    stats = {
        "power_success": 0,
        "power_failed": 0,
        "power_rows": 0,
        "openmeteo_success": 0,
        "openmeteo_failed": 0,
        "openmeteo_rows": 0,
        "total_duration": 0,
    }
    
    try:
        # Load configuration
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        logger.info(f"Config loaded ✓")
        
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
        
        http_config = config["http"]
        
        audit.add_section("Configuration", {
            "state": MAHARASHTRA_STATE_NAME,
            "locations_file": str(locations_file),
            "total_locations": len(locations_df),
            "max_workers": max_workers,
            "download_power": args.power or args.all,
            "download_openmeteo": args.openmeteo or args.all,
        })
        
        # Shared rate limiter (thread-safe)
        rate_limiter = AdaptiveRateLimiter(
            mode=RateLimitMode(args.rate_limit),
            base_delay=args.base_delay,
        )
        
        # Progress tracker
        progress_file = PROJECT_ROOT / "data" / "metadata" / "maharashtra" / "weather_progress.json"
        tracker = ProgressTracker(progress_file, batch_size=10)
        
        start_time = time.time()
        
        # === NASA POWER Downloads ===
        if args.power or args.all:
            logger.info("\n" + "=" * 50)
            logger.info("NASA POWER Historical Rainfall")
            logger.info("=" * 50)
            
            start_date, end_date = get_date_range(args.start_date, args.end_date, args.days_back)
            logger.info(f"Date range: {start_date} to {end_date}")
            
            power_config = config["nasa_power"]
            power_output = PROJECT_ROOT / config["paths"]["maharashtra"]["weather_power"]
            
            # Track progress
            district_ids = locations_df["location_id"].tolist()
            tracker.start_session(
                "power_download",
                chunks=district_ids,
                metadata={"start_date": start_date, "end_date": end_date},
                force_restart=not args.resume,
            )
            
            pending_ids = tracker.get_pending_chunks("power_download")
            pending_locations = locations_df[locations_df["location_id"].isin(pending_ids)]
            
            if pending_locations.empty:
                logger.info("All NASA POWER downloads completed")
            else:
                logger.info(f"Downloading {len(pending_locations)} locations...")
                
                # Build worker args
                worker_args = [
                    (
                        loc.to_dict(), power_config["api_base"], power_config["parameters"],
                        start_date, end_date, power_output, rate_limiter, http_config, "mh_worker"
                    )
                    for _, loc in pending_locations.iterrows()
                ]
                
                # Parallel download
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(power_worker, arg): arg[0]["district"]
                        for arg in worker_args
                    }
                    
                    for future in as_completed(futures):
                        district = futures[future]
                        try:
                            result = future.result()
                            
                            if result["success"]:
                                tracker.mark_completed(
                                    "power_download", result["location_id"],
                                    rows=result["rows"],
                                    output_file=result["output_file"],
                                )
                                stats["power_success"] += 1
                                stats["power_rows"] += result["rows"]
                            else:
                                tracker.mark_failed("power_download", result["location_id"], result["error"] or "Unknown")
                                stats["power_failed"] += 1
                                audit.add_error(f"NASA POWER {district}: {result['error']}")
                                
                        except Exception as e:
                            tracker.mark_failed("power_download", result["location_id"], str(e))
                            stats["power_failed"] += 1
                            audit.add_error(f"NASA POWER {district}: {e}")
                
                tracker.flush()
        
        # === Open-Meteo Downloads ===
        if args.openmeteo or args.all:
            logger.info("\n" + "=" * 50)
            logger.info("Open-Meteo 16-Day Forecast")
            logger.info("=" * 50)
            
            meteo_config = config["openmeteo"]
            meteo_output = PROJECT_ROOT / config["paths"]["maharashtra"]["weather_openmeteo"]
            
            # Track progress
            district_ids = locations_df["location_id"].tolist()
            tracker.start_session(
                "openmeteo_download",
                chunks=district_ids,
                metadata={"forecast_days": args.forecast_days},
                force_restart=not args.resume,
            )
            
            pending_ids = tracker.get_pending_chunks("openmeteo_download")
            pending_locations = locations_df[locations_df["location_id"].isin(pending_ids)]
            
            if pending_locations.empty:
                logger.info("All Open-Meteo downloads completed")
            else:
                logger.info(f"Downloading {len(pending_locations)} locations...")
                
                # Build worker args
                worker_args = [
                    (
                        loc.to_dict(), meteo_config["api_base"], meteo_config["daily_variables"],
                        args.forecast_days, meteo_config["timezone"], meteo_output,
                        timestamp, rate_limiter, http_config, "mh_worker"
                    )
                    for _, loc in pending_locations.iterrows()
                ]
                
                # Parallel download
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(openmeteo_worker, arg): arg[0]["district"]
                        for arg in worker_args
                    }
                    
                    for future in as_completed(futures):
                        district = futures[future]
                        try:
                            result = future.result()
                            
                            if result["success"]:
                                tracker.mark_completed(
                                    "openmeteo_download", result["location_id"],
                                    rows=result["rows"],
                                    output_file=result["output_file"],
                                )
                                stats["openmeteo_success"] += 1
                                stats["openmeteo_rows"] += result["rows"]
                            else:
                                tracker.mark_failed("openmeteo_download", result["location_id"], result["error"] or "Unknown")
                                stats["openmeteo_failed"] += 1
                                audit.add_error(f"Open-Meteo {district}: {result['error']}")
                                
                        except Exception as e:
                            tracker.mark_failed("openmeteo_download", result["location_id"], str(e))
                            stats["openmeteo_failed"] += 1
                            audit.add_error(f"Open-Meteo {district}: {e}")
                
                tracker.flush()
        
        stats["total_duration"] = time.time() - start_time
        
        # Audit summary
        audit.add_metric("NASA POWER Success", stats["power_success"])
        audit.add_metric("NASA POWER Failed", stats["power_failed"])
        audit.add_metric("NASA POWER Rows", stats["power_rows"])
        audit.add_metric("Open-Meteo Success", stats["openmeteo_success"])
        audit.add_metric("Open-Meteo Failed", stats["openmeteo_failed"])
        audit.add_metric("Open-Meteo Rows", stats["openmeteo_rows"])
        audit.add_metric("Duration (seconds)", round(stats["total_duration"], 2))
        
        if stats["power_failed"] > 0:
            audit.add_warning(f"{stats['power_failed']} NASA POWER downloads failed")
        if stats["openmeteo_failed"] > 0:
            audit.add_warning(f"{stats['openmeteo_failed']} Open-Meteo downloads failed")
        
        audit_path = audit.save()
        
        # Summary
        print(f"\n✅ Maharashtra Weather Download Complete!")
        print(f"   🌧️ NASA POWER: {stats['power_success']} success, {stats['power_failed']} failed ({stats['power_rows']:,} days)")
        print(f"   ☁️ Open-Meteo: {stats['openmeteo_success']} success, {stats['openmeteo_failed']} failed ({stats['openmeteo_rows']:,} days)")
        print(f"   ⏱️  Duration: {stats['total_duration']:.1f}s")
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
