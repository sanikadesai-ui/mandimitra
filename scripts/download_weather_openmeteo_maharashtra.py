#!/usr/bin/env python3
"""
MANDIMITRA - Open-Meteo 16-Day Forecast Weather Downloader

Downloads 16-day weather forecasts from Open-Meteo API for all 36
Maharashtra district headquarters.

Parameters downloaded:
- precipitation_sum: Daily precipitation sum (mm)
- precipitation_probability_max: Max precipitation probability (%)
- temperature_2m_max: Max temperature (°C)
- temperature_2m_min: Min temperature (°C)
- relative_humidity_2m_max: Max relative humidity (%)
- relative_humidity_2m_min: Min relative humidity (%)

Usage:
    python scripts/download_weather_openmeteo_maharashtra.py --download
    python scripts/download_weather_openmeteo_maharashtra.py --download --district Pune
    python scripts/download_weather_openmeteo_maharashtra.py --help

Output:
    data/raw/weather/openmeteo_forecast/maharashtra/<district>/forecast_<timestamp>.csv
    data/raw/weather/openmeteo_forecast/maharashtra/<district>/receipt.json

⚠️  HARD CONSTRAINT: Maharashtra-only (36 district HQs).
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
)
from src.utils.http import (
    AdaptiveRateLimiter,
    RateLimitMode,
    create_session,
)
from src.utils.logging_utils import setup_logger, get_utc_timestamp_safe
from src.utils.audit import AuditLogger


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Open-Meteo 16-day forecast for Maharashtra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OPEN-METEO API:
  Free API for weather forecasts. No API key required.
  Returns 16-day forecast by default.

PARAMETERS DOWNLOADED:
  - precipitation_sum: Daily precipitation (mm)
  - precipitation_probability_max: Max probability (%)
  - temperature_2m_max/min: Temperature range (°C)
  - relative_humidity_2m_max/min: Humidity range (%)

Examples:
    # Download forecasts for all 36 districts
    python scripts/download_weather_openmeteo_maharashtra.py --download
    
    # Download specific district
    python scripts/download_weather_openmeteo_maharashtra.py --download --district Pune
    
    # Parallel download (4 workers)
    python scripts/download_weather_openmeteo_maharashtra.py --download --max-workers 4
        """,
    )
    
    # Actions
    parser.add_argument(
        "--download",
        action="store_true",
        required=True,
        help="Download Open-Meteo forecasts",
    )
    
    # Location filter
    parser.add_argument(
        "--district",
        type=str,
        default=None,
        help="Download specific district only",
    )
    
    # Forecast settings
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=16,
        help="Number of forecast days (default: 16)",
    )
    
    # Parallel settings
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4)",
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources configuration",
    )
    parser.add_argument(
        "--locations-file",
        type=str,
        default="configs/maharashtra_locations.csv",
        help="Path to Maharashtra locations CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/weather/openmeteo_forecast/maharashtra",
        help="Output directory",
    )
    
    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if today's forecast exists",
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

def load_locations(filepath: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load Maharashtra district HQ locations.
    
    Args:
        filepath: Path to locations CSV
        logger: Logger instance
        
    Returns:
        DataFrame with columns: district, latitude, longitude
    """
    df = pd.read_csv(filepath, comment="#", skip_blank_lines=True)
    logger.info(f"Loaded {len(df)} Maharashtra locations")
    return df


# =============================================================================
# OPEN-METEO API
# =============================================================================

OPENMETEO_BASE = "https://api.open-meteo.com/v1/forecast"

DEFAULT_DAILY_VARIABLES = [
    "precipitation_sum",
    "precipitation_probability_max",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
]


def download_openmeteo_forecast(
    session: requests.Session,
    latitude: float,
    longitude: float,
    daily_variables: List[str],
    forecast_days: int,
    timezone_str: str,
    rate_limiter: AdaptiveRateLimiter,
    logger: logging.Logger,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Download Open-Meteo forecast for a single location.
    
    Args:
        session: HTTP session
        latitude: Location latitude
        longitude: Location longitude
        daily_variables: List of daily variable names
        forecast_days: Number of forecast days
        timezone_str: Timezone string
        rate_limiter: Rate limiter instance
        logger: Logger instance
        
    Returns:
        (DataFrame, stats_dict)
    """
    stats = {
        "success": False,
        "rows": 0,
        "error": None,
    }
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(daily_variables),
        "timezone": timezone_str,
        "forecast_days": forecast_days,
    }
    
    # Rate limit
    rate_limiter.acquire()
    
    try:
        response = session.get(OPENMETEO_BASE, params=params, timeout=30)
        
        if response.status_code == 429:
            rate_limiter.record_429()
            logger.warning("Rate limited by Open-Meteo, backing off...")
            time.sleep(10)
            response = session.get(OPENMETEO_BASE, params=params, timeout=30)
        
        if response.status_code != 200:
            stats["error"] = f"HTTP {response.status_code}"
            logger.error(f"Open-Meteo error: {response.status_code}")
            return None, stats
        
        data = response.json()
        
        # Extract daily data
        if "daily" in data:
            daily = data["daily"]
            
            # Build DataFrame
            df = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "latitude": latitude,
                "longitude": longitude,
            })
            
            # Add each variable
            for var in daily_variables:
                if var in daily:
                    df[var] = daily[var]
            
            stats["success"] = True
            stats["rows"] = len(df)
            
            return df, stats
        
        stats["error"] = "No daily data in response"
        return None, stats
        
    except requests.exceptions.Timeout:
        stats["error"] = "Request timeout"
        logger.error("Open-Meteo request timeout")
        return None, stats
        
    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"Open-Meteo error: {e}")
        return None, stats


def download_district_openmeteo(
    district: str,
    latitude: float,
    longitude: float,
    output_dir: Path,
    daily_variables: List[str],
    forecast_days: int,
    timezone_str: str,
    rate_limiter: AdaptiveRateLimiter,
    http_config: Dict[str, Any],
    logger_name: str,
    timestamp: str,
) -> Dict[str, Any]:
    """
    Worker function to download Open-Meteo forecast for one district.
    
    Args:
        district: District name
        latitude: Location latitude
        longitude: Location longitude
        output_dir: Output directory
        daily_variables: Variables to download
        forecast_days: Number of forecast days
        timezone_str: Timezone
        rate_limiter: Shared rate limiter
        http_config: HTTP configuration
        logger_name: Logger name
        timestamp: Timestamp for filename
        
    Returns:
        Result dict
    """
    logger = logging.getLogger(logger_name)
    
    result = {
        "district": district,
        "success": False,
        "rows": 0,
        "output_file": None,
        "error": None,
    }
    
    try:
        # Create session
        session = create_session(
            max_retries=http_config.get("max_retries", 3),
            backoff_factor=http_config.get("backoff_factor", 1.0),
            timeout=http_config.get("timeout", 30),
        )
        
        # Download
        df, stats = download_openmeteo_forecast(
            session=session,
            latitude=latitude,
            longitude=longitude,
            daily_variables=daily_variables,
            forecast_days=forecast_days,
            timezone_str=timezone_str,
            rate_limiter=rate_limiter,
            logger=logger,
        )
        
        if df is not None and not df.empty:
            # Add district column
            df["district"] = district
            
            # Save
            district_dir = output_dir / sanitize_filename(district)
            ensure_directory(district_dir)
            
            output_file = district_dir / f"forecast_{timestamp}.csv"
            save_dataframe(df, output_file)
            
            # Receipt
            receipt = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "forecast_days": forecast_days,
                "daily_variables": daily_variables,
                "rows": len(df),
                "date_range": {
                    "min": str(df["date"].min()),
                    "max": str(df["date"].max()),
                },
            }
            save_receipt(district_dir / f"receipt_{timestamp}.json", receipt)
            
            result["success"] = True
            result["rows"] = len(df)
            result["output_file"] = str(output_file)
            
            logger.info(f"✓ {district}: {len(df)} days forecast")
            
        else:
            result["error"] = stats.get("error", "No data")
            logger.warning(f"✗ {district}: {result['error']}")
        
        session.close()
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"✗ {district}: {e}")
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "openmeteo_download.log"
    logger = setup_logger("openmeteo_mh", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("weather_openmeteo_download", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Open-Meteo Forecast Downloader")
    logger.info("=" * 70)
    
    try:
        # Load config
        config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            config = load_config(config_path)
            openmeteo_config = config.get("weather", {}).get("openmeteo", {})
            http_config = config.get("http", {})
        else:
            openmeteo_config = {}
            http_config = {}
        
        # Load locations
        locations_path = PROJECT_ROOT / args.locations_file
        locations = load_locations(locations_path, logger)
        
        # Filter to specific district if requested
        if args.district:
            locations = locations[locations["district"].str.lower() == args.district.lower()]
            if locations.empty:
                logger.error(f"District not found: {args.district}")
                return 1
        
        logger.info(f"Districts to download: {len(locations)}")
        
        # Settings
        forecast_days = args.forecast_days or openmeteo_config.get("forecast_days", 16)
        daily_variables = openmeteo_config.get("daily_variables", DEFAULT_DAILY_VARIABLES)
        timezone_str = openmeteo_config.get("timezone", "Asia/Kolkata")
        
        logger.info(f"Forecast days: {forecast_days}")
        logger.info(f"Variables: {daily_variables}")
        
        # Output directory
        output_dir = PROJECT_ROOT / args.output_dir
        
        audit.add_section("Configuration", {
            "districts": len(locations),
            "forecast_days": forecast_days,
            "daily_variables": daily_variables,
            "max_workers": args.max_workers,
        })
        
        # Rate limiter (shared across workers)
        rate_limiter = AdaptiveRateLimiter(
            mode=RateLimitMode.AUTO,
            base_delay=openmeteo_config.get("delay_between_requests", 0.5),
        )
        
        # Download in parallel
        max_workers = min(args.max_workers, 8)
        results = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for _, row in locations.iterrows():
                future = executor.submit(
                    download_district_openmeteo,
                    district=row["district"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    output_dir=output_dir,
                    daily_variables=daily_variables,
                    forecast_days=forecast_days,
                    timezone_str=timezone_str,
                    rate_limiter=rate_limiter,
                    http_config=http_config,
                    logger_name=f"openmeteo_{sanitize_filename(row['district'])}",
                    timestamp=timestamp,
                )
                futures[future] = row["district"]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        duration = time.time() - start_time
        
        # Summarize results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        total_rows = sum(r["rows"] for r in successful)
        
        audit.add_section("Results", {
            "successful": len(successful),
            "failed": len(failed),
            "total_rows": total_rows,
            "duration_seconds": round(duration, 1),
            "failed_districts": [r["district"] for r in failed],
        })
        
        # Summary
        print(f"\n✅ Open-Meteo Forecast Download Complete!")
        print(f"   ✓ Successful: {len(successful)}/{len(locations)} districts")
        print(f"   ✗ Failed: {len(failed)}")
        print(f"   📊 Total forecast days: {total_rows}")
        print(f"   📅 Forecast horizon: {forecast_days} days")
        print(f"   ⏱️  Duration: {duration:.1f}s")
        print(f"\n   📁 Output: {output_dir}")
        
        if failed:
            print(f"\n   ⚠️  Failed districts: {[r['district'] for r in failed]}")
        
        audit_path = audit.save()
        print(f"   📋 Audit: {audit_path}")
        
        return 0 if not failed else 1
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
