#!/usr/bin/env python3
"""
MANDIMITRA - Process Weather Data

This script standardizes and validates weather datasets:
- NASA POWER historical daily data
- Open-Meteo forecast data

Features:
- Standardizes column names to canonical schema
- Validates district names against 36 canonical districts
- Deduplicates by (date, district) key
- Generates QC reports

Output:
- data/processed/weather/power_daily_maharashtra.parquet
- data/processed/weather/forecast_maharashtra.parquet
- logs/weather_qc_report_<timestamp>.md

Usage:
    python scripts/process_weather.py
    python scripts/process_weather.py --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import duckdb
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.district_normalize import CANONICAL_DISTRICTS, get_normalizer
from src.schemas.weather_canonical import (
    POWER_CANONICAL_COLUMNS,
    FORECAST_CANONICAL_COLUMNS,
    POWER_NATURAL_KEY,
    FORECAST_NATURAL_KEY,
    validate_power_weather,
    validate_forecast_weather,
    summarize_power_weather,
    summarize_forecast_weather,
    generate_weather_qc_report,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATA_RAW_WEATHER = PROJECT_ROOT / "data" / "raw" / "weather"
DATA_PROCESSED_WEATHER = PROJECT_ROOT / "data" / "processed" / "weather"
LOGS_DIR = PROJECT_ROOT / "logs"

# Input files - Check multiple possible locations
POWER_INPUT = DATA_RAW_WEATHER / "nasa_power_maharashtra_10yr.csv"
POWER_INPUT_DIR = DATA_RAW_WEATHER / "power_daily" / "maharashtra"  # Per-district CSVs

FORECAST_INPUT = DATA_RAW_WEATHER / "openmeteo_forecast_maharashtra.csv"
FORECAST_INPUT_DIR = DATA_RAW_WEATHER / "openmeteo_forecast" / "maharashtra"  # Per-district CSVs

# Output files
POWER_OUTPUT = DATA_PROCESSED_WEATHER / "power_daily_maharashtra.parquet"
FORECAST_OUTPUT = DATA_PROCESSED_WEATHER / "forecast_maharashtra.parquet"

# Column mappings for NASA POWER
POWER_COLUMN_MAP = {
    # Standard variations
    "DATE": "date",
    "Date": "date",
    "DISTRICT": "district",
    "District": "district",
    "LAT": "latitude",
    "LON": "longitude",
    "lat": "latitude",
    "lon": "longitude",
    "latitude": "latitude",
    "longitude": "longitude",
    
    # Temperature columns
    "T2M_MAX": "t2m_max",
    "T2M_MIN": "t2m_min",
    "T2M": "t2m_mean",
    "t2m_max": "t2m_max",
    "t2m_min": "t2m_min",
    "t2m": "t2m_mean",
    "temperature_max": "t2m_max",
    "temperature_min": "t2m_min",
    "temperature_mean": "t2m_mean",
    
    # Humidity
    "RH2M": "rh2m",
    "rh2m": "rh2m",
    "relative_humidity": "rh2m",
    
    # Precipitation
    "PRECTOTCORR": "precipitation",
    "PRECTOT": "precipitation",
    "precipitation": "precipitation",
    "precip": "precipitation",
    
    # Wind
    "WS2M": "ws2m",
    "ws2m": "ws2m",
    "wind_speed": "ws2m",
    "windspeed": "ws2m",
    
    # Solar radiation
    "ALLSKY_SFC_SW_DWN": "srad",
    "srad": "srad",
    "solar_radiation": "srad",
}

# Column mappings for Open-Meteo forecast
FORECAST_COLUMN_MAP = {
    "date": "date",
    "Date": "date",
    "district": "district",
    "District": "district",
    "latitude": "latitude",
    "longitude": "longitude",
    "lat": "latitude",
    "lon": "longitude",
    
    # Temperature
    "temperature_2m_max": "temperature_max",
    "temperature_max": "temperature_max",
    "temp_max": "temperature_max",
    "temperature_2m_min": "temperature_min",
    "temperature_min": "temperature_min",
    "temp_min": "temperature_min",
    
    # Precipitation
    "precipitation_sum": "precipitation_sum",
    "precip_sum": "precipitation_sum",
    "total_precipitation": "precipitation_sum",
    
    # Wind
    "windspeed_10m_max": "windspeed_max",
    "windspeed_max": "windspeed_max",
    "wind_speed_max": "windspeed_max",
    
    # Probability
    "precipitation_probability_max": "precipitation_probability_max",
    "precip_prob_max": "precipitation_probability_max",
    
    # Weather code
    "weathercode": "weathercode",
    "weather_code": "weathercode",
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def standardize_columns(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    """
    Standardize column names using mapping.
    
    Args:
        df: Input DataFrame
        column_map: Mapping from source to canonical names
        
    Returns:
        DataFrame with standardized columns
    """
    # Create rename dict for columns that exist
    rename_dict = {}
    for old_name, new_name in column_map.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    
    df = df.rename(columns=rename_dict)
    
    return df


def normalize_weather_districts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Normalize district names in weather data.
    
    Args:
        df: DataFrame with district column
        
    Returns:
        Tuple of (normalized DataFrame, set of unmapped districts)
    """
    if "district" not in df.columns:
        return df, set()
    
    normalizer = get_normalizer()
    normalizer.reset_stats()  # Reset to track this batch
    
    # Normalize
    original_districts = df["district"].tolist()
    df["district"] = normalizer.normalize_batch(original_districts)
    
    # Get unmapped set
    unmapped = normalizer.unmapped.copy()
    
    return df, unmapped


def parse_weather_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date column in weather data.
    
    Args:
        df: DataFrame with date column
        
    Returns:
        DataFrame with parsed dates
    """
    if "date" not in df.columns:
        return df
    
    # Try multiple date formats
    date_formats = [
        "%Y-%m-%d",
        "%Y%m%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    ]
    
    parsed = pd.to_datetime(df["date"], errors="coerce")
    
    # If many failed, try other formats
    if parsed.isna().sum() > len(df) * 0.1:
        for fmt in date_formats:
            try:
                parsed = pd.to_datetime(df["date"], format=fmt, errors="coerce")
                if parsed.isna().sum() < len(df) * 0.1:
                    break
            except:
                continue
    
    df["date"] = parsed
    
    return df


def deduplicate_weather(df: pd.DataFrame, key_cols: Tuple[str, ...]) -> pd.DataFrame:
    """
    Deduplicate weather data using DuckDB.
    
    Args:
        df: Weather DataFrame
        key_cols: Columns forming the natural key
        
    Returns:
        Deduplicated DataFrame
    """
    before_count = len(df)
    
    con = duckdb.connect()
    con.register("weather_raw", df)
    
    key_str = ", ".join(key_cols)
    
    # Simple dedup: keep first occurrence (or most complete row)
    query = f"""
    WITH ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY {key_str}
                ORDER BY date
            ) AS rn
        FROM weather_raw
        WHERE district IS NOT NULL
    )
    SELECT * EXCLUDE (rn)
    FROM ranked
    WHERE rn = 1
    """
    
    result_df = con.execute(query).fetchdf()
    con.close()
    
    after_count = len(result_df)
    removed = before_count - after_count
    
    if removed > 0:
        logger.info(f"  Dedup: {before_count:,} → {after_count:,} (removed {removed:,})")
    
    return result_df


def process_power_data(dry_run: bool = False) -> Optional[pd.DataFrame]:
    """
    Process NASA POWER weather data.
    
    Args:
        dry_run: If True, don't save
        
    Returns:
        Processed DataFrame or None
    """
    df = None
    
    # Try single combined file first
    if POWER_INPUT.exists():
        logger.info(f"Processing NASA POWER data from: {POWER_INPUT}")
        df = pd.read_csv(POWER_INPUT)
        logger.info(f"  Loaded {len(df):,} rows")
    
    # Try per-district directory structure
    elif POWER_INPUT_DIR.exists():
        logger.info(f"Processing NASA POWER data from district folders: {POWER_INPUT_DIR}")
        dfs = []
        for district_dir in POWER_INPUT_DIR.iterdir():
            if district_dir.is_dir():
                csv_files = list(district_dir.glob("*.csv"))
                for csv_file in csv_files:
                    if "receipt" not in csv_file.name.lower():
                        try:
                            district_df = pd.read_csv(csv_file)
                            # Add district name from folder if not present
                            if "district" not in district_df.columns and "District" not in district_df.columns:
                                district_df["district"] = district_dir.name.replace("_", " ")
                            dfs.append(district_df)
                        except Exception as e:
                            logger.warning(f"  Failed to read {csv_file}: {e}")
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"  Loaded {len(df):,} rows from {len(dfs)} district files")
    
    if df is None or len(df) == 0:
        logger.warning(f"NASA POWER file not found or empty")
        return None
    
    # Standardize columns
    df = standardize_columns(df, POWER_COLUMN_MAP)
    
    # Parse dates
    df = parse_weather_dates(df)
    
    # Normalize districts
    df, unmapped = normalize_weather_districts(df)
    
    if unmapped:
        logger.warning(f"  Unmapped districts: {list(unmapped)}")
    
    # Filter to valid districts only
    before_filter = len(df)
    df = df[df["district"].notna()].copy()
    logger.info(f"  Filtered to valid districts: {before_filter:,} → {len(df):,}")
    
    # Add source
    df["source"] = "nasa_power"
    
    # Deduplicate
    df = deduplicate_weather(df, POWER_NATURAL_KEY)
    
    # Ensure canonical columns exist
    for col in POWER_CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[[c for c in POWER_CANONICAL_COLUMNS if c in df.columns]]
    
    # Validate
    logger.info("  Validating POWER data...")
    try:
        df, errors = validate_power_weather(df, strict=False, raise_on_error=False)
        if errors:
            logger.warning(f"  Validation warnings: {len(errors)}")
    except Exception as e:
        logger.warning(f"  Validation error: {e}")
    
    # Save
    if not dry_run:
        DATA_PROCESSED_WEATHER.mkdir(parents=True, exist_ok=True)
        df.to_parquet(POWER_OUTPUT, index=False)
        logger.info(f"  Saved to: {POWER_OUTPUT}")
    
    return df


def process_forecast_data(dry_run: bool = False) -> Optional[pd.DataFrame]:
    """
    Process Open-Meteo forecast data.
    
    Args:
        dry_run: If True, don't save
        
    Returns:
        Processed DataFrame or None
    """
    df = None
    
    # Try single combined file first
    if FORECAST_INPUT.exists():
        logger.info(f"Processing Open-Meteo forecast from: {FORECAST_INPUT}")
        df = pd.read_csv(FORECAST_INPUT)
        logger.info(f"  Loaded {len(df):,} rows")
    
    # Try per-district directory structure
    elif FORECAST_INPUT_DIR.exists():
        logger.info(f"Processing Open-Meteo forecast from district folders: {FORECAST_INPUT_DIR}")
        dfs = []
        for district_dir in FORECAST_INPUT_DIR.iterdir():
            if district_dir.is_dir():
                csv_files = list(district_dir.glob("*.csv"))
                for csv_file in csv_files:
                    if "receipt" not in csv_file.name.lower():
                        try:
                            district_df = pd.read_csv(csv_file)
                            # Add district name from folder if not present
                            if "district" not in district_df.columns and "District" not in district_df.columns:
                                district_df["district"] = district_dir.name.replace("_", " ")
                            dfs.append(district_df)
                        except Exception as e:
                            logger.warning(f"  Failed to read {csv_file}: {e}")
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"  Loaded {len(df):,} rows from {len(dfs)} district files")
    
    if df is None or len(df) == 0:
        logger.warning(f"Forecast file not found or empty")
        return None
    
    # Standardize columns
    df = standardize_columns(df, FORECAST_COLUMN_MAP)
    
    # Parse dates
    df = parse_weather_dates(df)
    
    # Normalize districts
    df, unmapped = normalize_weather_districts(df)
    
    if unmapped:
        logger.warning(f"  Unmapped districts: {list(unmapped)}")
    
    # Filter to valid districts only
    before_filter = len(df)
    df = df[df["district"].notna()].copy()
    logger.info(f"  Filtered to valid districts: {before_filter:,} → {len(df):,}")
    
    # Add source
    df["source"] = "open_meteo"
    
    # Deduplicate
    df = deduplicate_weather(df, FORECAST_NATURAL_KEY)
    
    # Ensure canonical columns exist
    for col in FORECAST_CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[[c for c in FORECAST_CANONICAL_COLUMNS if c in df.columns]]
    
    # Validate
    logger.info("  Validating forecast data...")
    try:
        df, errors = validate_forecast_weather(df, strict=False, raise_on_error=False)
        if errors:
            logger.warning(f"  Validation warnings: {len(errors)}")
    except Exception as e:
        logger.warning(f"  Validation error: {e}")
    
    # Save
    if not dry_run:
        DATA_PROCESSED_WEATHER.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FORECAST_OUTPUT, index=False)
        logger.info(f"  Saved to: {FORECAST_OUTPUT}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main(dry_run: bool = False):
    """
    Main entry point for weather processing.
    
    Args:
        dry_run: If True, don't save outputs
    """
    logger.info("=" * 60)
    logger.info("MANDIMITRA - Process Weather Data")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("DRY RUN MODE - no files will be saved")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process NASA POWER
    power_df = process_power_data(dry_run=dry_run)
    
    # Process Open-Meteo forecast
    forecast_df = process_forecast_data(dry_run=dry_run)
    
    # Generate QC report
    if power_df is not None or forecast_df is not None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        qc_path = LOGS_DIR / f"weather_qc_report_{timestamp}.md"
        
        if not dry_run:
            generate_weather_qc_report(power_df, forecast_df, output_path=qc_path)
            logger.info(f"QC report saved to: {qc_path}")
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    
    if power_df is not None:
        summary = summarize_power_weather(power_df)
        logger.info(f"NASA POWER:")
        logger.info(f"  Rows: {summary['total_rows']:,}")
        logger.info(f"  Districts: {summary.get('unique_districts', 'N/A')}")
        logger.info(f"  Date range: {summary.get('date_min', 'N/A')} to {summary.get('date_max', 'N/A')}")
    
    if forecast_df is not None:
        summary = summarize_forecast_weather(forecast_df)
        logger.info(f"Open-Meteo Forecast:")
        logger.info(f"  Rows: {summary['total_rows']:,}")
        logger.info(f"  Districts: {summary.get('unique_districts', 'N/A')}")
        logger.info(f"  Date range: {summary.get('date_min', 'N/A')} to {summary.get('date_max', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and standardize weather data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save outputs",
    )
    
    args = parser.parse_args()
    main(dry_run=args.dry_run)
