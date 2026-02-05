#!/usr/bin/env python3
"""
MANDIMITRA - NASA POWER Historical Weather Data Downloader

Downloads historical daily weather data from NASA POWER API for all 36
Maharashtra district headquarters.

Parameters downloaded:
- PRECTOTCORR: Precipitation corrected (mm/day)
- T2M: Temperature at 2m (°C)
- T2M_MAX: Max daily temperature (°C)
- T2M_MIN: Min daily temperature (°C)
- RH2M: Relative humidity at 2m (%)

Usage:
    python scripts/download_weather_power_maharashtra.py --download
    python scripts/download_weather_power_maharashtra.py --download --years-back 5
    python scripts/download_weather_power_maharashtra.py --download --district Pune
    python scripts/download_weather_power_maharashtra.py --help

Output:
    data/raw/weather/power_daily/maharashtra/<district>/power_daily_<start>_<end>.csv
    data/raw/weather/power_daily/maharashtra/<district>/receipt.json

⚠️  HARD CONSTRAINT: Maharashtra-only (36 district HQs).
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        description="Download NASA POWER historical weather data for Maharashtra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NASA POWER API:
  Free API for agricultural weather data. No API key required.
  Rate limit: ~30 requests/minute (be polite).

PARAMETERS DOWNLOADED:
  - PRECTOTCORR: Precipitation corrected (mm/day)
  - T2M: Temperature at 2m (°C)
  - T2M_MAX, T2M_MIN: Temperature range
  - RH2M: Relative humidity (%)

Examples:
    # Download 10 years for all 36 districts
    python scripts/download_weather_power_maharashtra.py --download
    
    # Download 5 years only
    python scripts/download_weather_power_maharashtra.py --download --years-back 5
    
    # Download specific district
    python scripts/download_weather_power_maharashtra.py --download --district Pune
    
    # Parallel download (4 workers)
    python scripts/download_weather_power_maharashtra.py --download --max-workers 4
    
    # Custom date range
    python scripts/download_weather_power_maharashtra.py --download --start-date 2020-01-01 --end-date 2025-12-31
        """,
    )
    
    # Actions
    parser.add_argument(
        "--download",
        action="store_true",
        required=True,
        help="Download NASA POWER data",
    )
    
    # Date range
    parser.add_argument(
        "--years-back",
        type=int,
        default=10,
        help="Years of historical data to download (default: 10)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides --years-back",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: yesterday",
    )
    
    # Location filter
    parser.add_argument(
        "--district",
        type=str,
        default=None,
        help="Download specific district only",
    )
    
    # Parallel settings
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel workers (default: 2, NASA POWER is rate-limited)",
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
        default="data/raw/weather/power_daily/maharashtra",
        help="Output directory",
    )
    
    # Options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
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
# NASA POWER API
# =============================================================================

NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

DEFAULT_PARAMETERS = [
    "PRECTOTCORR",  # Precipitation corrected
    "T2M",          # Temperature at 2m
    "T2M_MAX",      # Max temperature
    "T2M_MIN",      # Min temperature
    "RH2M",         # Relative humidity
]


def download_power_data(
    session: requests.Session,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    parameters: List[str],
    rate_limiter: AdaptiveRateLimiter,
    logger: logging.Logger,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Download NASA POWER data for a single location.
    
    Args:
        session: HTTP session
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        parameters: List of parameter codes
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
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }
    
    # Rate limit
    rate_limiter.acquire()
    
    try:
        response = session.get(NASA_POWER_BASE, params=params, timeout=60)
        
        if response.status_code == 429:
            rate_limiter.record_429()
            logger.warning("Rate limited by NASA POWER, backing off...")
            time.sleep(30)
            response = session.get(NASA_POWER_BASE, params=params, timeout=60)
        
        if response.status_code != 200:
            stats["error"] = f"HTTP {response.status_code}"
            logger.error(f"NASA POWER error: {response.status_code}")
            return None, stats
        
        data = response.json()
        
        # Extract daily data
        if "properties" in data and "parameter" in data["properties"]:
            param_data = data["properties"]["parameter"]
            
            # Build DataFrame
            records = []
            for date_str in list(param_data.get(parameters[0], {}).keys()):
                record = {"date": date_str}
                for param in parameters:
                    value = param_data.get(param, {}).get(date_str)
                    # NASA POWER uses -999 for missing
                    if value == -999:
                        value = None
                    record[param] = value
                records.append(record)
            
            if records:
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
                df["latitude"] = latitude
                df["longitude"] = longitude
                
                stats["success"] = True
                stats["rows"] = len(df)
                
                return df, stats
        
        stats["error"] = "No data in response"
        return None, stats
        
    except requests.exceptions.Timeout:
        stats["error"] = "Request timeout"
        logger.error("NASA POWER request timeout")
        return None, stats
        
    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"NASA POWER error: {e}")
        return None, stats


def download_district_power(
    district: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    output_dir: Path,
    parameters: List[str],
    rate_limiter: AdaptiveRateLimiter,
    http_config: Dict[str, Any],
    logger_name: str,
) -> Dict[str, Any]:
    """
    Worker function to download NASA POWER data for one district.
    
    Args:
        district: District name
        latitude: Location latitude
        longitude: Location longitude
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD)
        output_dir: Output directory
        parameters: Parameters to download
        rate_limiter: Shared rate limiter
        http_config: HTTP configuration
        logger_name: Logger name
        
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
            max_retries=http_config.get("max_retries", 5),
            backoff_factor=http_config.get("backoff_factor", 2.0),
            timeout=http_config.get("timeout", 60),
        )
        
        # Download
        df, stats = download_power_data(
            session=session,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters,
            rate_limiter=rate_limiter,
            logger=logger,
        )
        
        if df is not None and not df.empty:
            # Add district column
            df["district"] = district
            
            # Save
            district_dir = output_dir / sanitize_filename(district)
            ensure_directory(district_dir)
            
            start_fmt = datetime.strptime(start_date, "%Y%m%d").strftime("%Y%m%d")
            end_fmt = datetime.strptime(end_date, "%Y%m%d").strftime("%Y%m%d")
            
            output_file = district_dir / f"power_daily_{start_fmt}_{end_fmt}.csv"
            save_dataframe(df, output_file)
            
            # Receipt
            receipt = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "parameters": parameters,
                "rows": len(df),
                "date_range": {
                    "min": str(df["date"].min()),
                    "max": str(df["date"].max()),
                },
            }
            save_receipt(district_dir / f"receipt_{start_fmt}_{end_fmt}.json", receipt)
            
            result["success"] = True
            result["rows"] = len(df)
            result["output_file"] = str(output_file)
            
            logger.info(f"✓ {district}: {len(df):,} rows")
            
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
    log_file = PROJECT_ROOT / "logs" / "power_download.log"
    logger = setup_logger("power_mh", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("weather_power_download", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - NASA POWER Weather Downloader")
    logger.info("=" * 70)
    
    try:
        # Load config
        config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            config = load_config(config_path)
            power_config = config.get("weather", {}).get("nasa_power", {})
            http_config = config.get("http", {})
        else:
            power_config = {}
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
        
        # Determine date range
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d").strftime("%Y%m%d")
        else:
            years_back = args.years_back or power_config.get("years_back", 10)
            start = datetime.now() - timedelta(days=years_back * 365)
            start_date = start.strftime("%Y%m%d")
        
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").strftime("%Y%m%d")
        else:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Parameters
        parameters = power_config.get("parameters", DEFAULT_PARAMETERS)
        logger.info(f"Parameters: {parameters}")
        
        # Output directory
        output_dir = PROJECT_ROOT / args.output_dir
        
        audit.add_section("Configuration", {
            "districts": len(locations),
            "date_range": f"{start_date} to {end_date}",
            "parameters": parameters,
            "max_workers": args.max_workers,
        })
        
        # Rate limiter (shared across workers)
        rate_limiter = AdaptiveRateLimiter(
            mode=RateLimitMode.AUTO,
            base_delay=power_config.get("delay_between_requests", 2.0),
        )
        
        # Download in parallel
        max_workers = min(args.max_workers, 4)  # Cap at 4 for NASA POWER
        results = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for _, row in locations.iterrows():
                future = executor.submit(
                    download_district_power,
                    district=row["district"],
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=output_dir,
                    parameters=parameters,
                    rate_limiter=rate_limiter,
                    http_config=http_config,
                    logger_name=f"power_{sanitize_filename(row['district'])}",
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
        print(f"\n✅ NASA POWER Download Complete!")
        print(f"   ✓ Successful: {len(successful)}/{len(locations)} districts")
        print(f"   ✗ Failed: {len(failed)}")
        print(f"   📊 Total rows: {total_rows:,}")
        print(f"   📅 Date range: {start_date} to {end_date}")
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
