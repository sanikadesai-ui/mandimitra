#!/usr/bin/env python3
"""
MANDIMITRA - Pandera Schemas for Canonical Weather Data

Defines validation schemas for weather datasets:
- NASA POWER historical daily data (36 districts × ~10 years)
- Open-Meteo forecast data (36 districts × 16 days)

Canonical Weather Columns (NASA POWER):
    date, district, latitude, longitude, t2m_max, t2m_min, t2m_mean,
    rh2m, precipitation, ws2m, srad, source

Canonical Forecast Columns (Open-Meteo):
    date, district, latitude, longitude, temperature_max, temperature_min,
    precipitation_sum, windspeed_max, precipitation_probability_max,
    weathercode, source

Usage:
    from src.schemas.weather_canonical import (
        CanonicalPowerSchema,
        CanonicalForecastSchema,
        validate_power_weather,
        validate_forecast_weather,
    )
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.district_normalize import CANONICAL_DISTRICTS


# =============================================================================
# CONSTANTS
# =============================================================================

# NASA POWER canonical columns
POWER_CANONICAL_COLUMNS: List[str] = [
    "date",
    "district",
    "latitude",
    "longitude",
    "t2m_max",       # Temperature at 2m max (°C)
    "t2m_min",       # Temperature at 2m min (°C)
    "t2m_mean",      # Temperature at 2m mean (°C)
    "rh2m",          # Relative humidity at 2m (%)
    "precipitation", # Precipitation (mm)
    "ws2m",          # Wind speed at 2m (m/s)
    "srad",          # Solar radiation (MJ/m²/day)
    "source",        # Data source identifier
]

# Open-Meteo forecast canonical columns
FORECAST_CANONICAL_COLUMNS: List[str] = [
    "date",
    "district",
    "latitude",
    "longitude",
    "temperature_max",              # Max temperature (°C)
    "temperature_min",              # Min temperature (°C)
    "precipitation_sum",            # Total precipitation (mm)
    "windspeed_max",                # Max wind speed (km/h)
    "precipitation_probability_max", # Max precipitation probability (%)
    "weathercode",                   # WMO weather code
    "source",                        # Data source identifier
]

# Natural keys
POWER_NATURAL_KEY: Tuple[str, ...] = ("date", "district")
FORECAST_NATURAL_KEY: Tuple[str, ...] = ("date", "district")

# Valid data sources
VALID_POWER_SOURCES: Set[str] = {"nasa_power", "NASA POWER"}
VALID_FORECAST_SOURCES: Set[str] = {"open_meteo", "Open-Meteo"}


# =============================================================================
# CUSTOM CHECKS
# =============================================================================

def check_canonical_district(district: str) -> bool:
    """Check if district is in the canonical list of 36."""
    if not district or not isinstance(district, str):
        return False
    return district in CANONICAL_DISTRICTS


def check_temperature_reasonable(temp: float) -> bool:
    """Check if temperature is within reasonable range for India."""
    if pd.isna(temp):
        return True
    return -10 <= temp <= 55  # °C


def check_humidity_range(humidity: float) -> bool:
    """Check if humidity is within valid range."""
    if pd.isna(humidity):
        return True
    return 0 <= humidity <= 100


def check_precipitation_non_negative(precip: float) -> bool:
    """Check if precipitation is non-negative."""
    if pd.isna(precip):
        return True
    return precip >= 0


def check_wind_speed_reasonable(ws: float) -> bool:
    """Check if wind speed is within reasonable range."""
    if pd.isna(ws):
        return True
    return 0 <= ws <= 200  # m/s or km/h


def check_solar_radiation_reasonable(srad: float) -> bool:
    """Check if solar radiation is within reasonable range."""
    if pd.isna(srad):
        return True
    return 0 <= srad <= 50  # MJ/m²/day


def check_latitude_maharashtra(lat: float) -> bool:
    """Check if latitude is within Maharashtra bounds."""
    if pd.isna(lat):
        return True
    return 15.5 <= lat <= 22.5  # Maharashtra range


def check_longitude_maharashtra(lon: float) -> bool:
    """Check if longitude is within Maharashtra bounds."""
    if pd.isna(lon):
        return True
    return 72.5 <= lon <= 81.0  # Maharashtra range


def check_wmo_weather_code(code: float) -> bool:
    """Check if WMO weather code is valid."""
    if pd.isna(code):
        return True
    # WMO codes range 0-99
    return 0 <= code <= 99


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

def create_power_schema(
    strict_district: bool = True,
    strict_ranges: bool = False,
) -> DataFrameSchema:
    """
    Create Pandera schema for NASA POWER weather data.
    
    Args:
        strict_district: Enforce canonical district names
        strict_ranges: Enforce value range checks
        
    Returns:
        DataFrameSchema for validation
    """
    # District checks
    district_checks = [Check.str_length(min_value=1)]
    if strict_district:
        district_checks.append(
            Check(check_canonical_district, element_wise=True, error="District must be canonical")
        )
    
    schema = DataFrameSchema(
        columns={
            "date": Column(
                "datetime64[ns]",
                nullable=False,
                coerce=True,
            ),
            "district": Column(
                str,
                checks=district_checks,
                nullable=False,
                coerce=True,
            ),
            "latitude": Column(
                float,
                checks=[Check(check_latitude_maharashtra, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "longitude": Column(
                float,
                checks=[Check(check_longitude_maharashtra, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "t2m_max": Column(
                float,
                checks=[Check(check_temperature_reasonable, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "t2m_min": Column(
                float,
                checks=[Check(check_temperature_reasonable, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "t2m_mean": Column(
                float,
                checks=[Check(check_temperature_reasonable, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "rh2m": Column(
                float,
                checks=[Check(check_humidity_range, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "precipitation": Column(
                float,
                checks=[Check(check_precipitation_non_negative, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "ws2m": Column(
                float,
                checks=[Check(check_wind_speed_reasonable, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "srad": Column(
                float,
                checks=[Check(check_solar_radiation_reasonable, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "source": Column(
                str,
                nullable=True,
                coerce=True,
            ),
        },
        strict=False,
        coerce=True,
    )
    
    return schema


def create_forecast_schema(
    strict_district: bool = True,
    strict_ranges: bool = False,
) -> DataFrameSchema:
    """
    Create Pandera schema for Open-Meteo forecast data.
    
    Args:
        strict_district: Enforce canonical district names
        strict_ranges: Enforce value range checks
        
    Returns:
        DataFrameSchema for validation
    """
    # District checks
    district_checks = [Check.str_length(min_value=1)]
    if strict_district:
        district_checks.append(
            Check(check_canonical_district, element_wise=True, error="District must be canonical")
        )
    
    schema = DataFrameSchema(
        columns={
            "date": Column(
                "datetime64[ns]",
                nullable=False,
                coerce=True,
            ),
            "district": Column(
                str,
                checks=district_checks,
                nullable=False,
                coerce=True,
            ),
            "latitude": Column(
                float,
                checks=[Check(check_latitude_maharashtra, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "longitude": Column(
                float,
                checks=[Check(check_longitude_maharashtra, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "temperature_max": Column(
                float,
                checks=[Check(check_temperature_reasonable, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "temperature_min": Column(
                float,
                checks=[Check(check_temperature_reasonable, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "precipitation_sum": Column(
                float,
                checks=[Check(check_precipitation_non_negative, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "windspeed_max": Column(
                float,
                checks=[Check(check_wind_speed_reasonable, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "precipitation_probability_max": Column(
                float,
                checks=[Check(check_humidity_range, element_wise=True)] if strict_ranges else [],  # 0-100
                nullable=True,
                coerce=True,
            ),
            "weathercode": Column(
                float,
                checks=[Check(check_wmo_weather_code, element_wise=True)] if strict_ranges else [],
                nullable=True,
                coerce=True,
            ),
            "source": Column(
                str,
                nullable=True,
                coerce=True,
            ),
        },
        strict=False,
        coerce=True,
    )
    
    return schema


# Pre-built schema instances
CanonicalPowerSchema = create_power_schema(strict_district=True)
CanonicalPowerSchemaLenient = create_power_schema(strict_district=False)
CanonicalForecastSchema = create_forecast_schema(strict_district=True)
CanonicalForecastSchemaLenient = create_forecast_schema(strict_district=False)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_power_weather(
    df: pd.DataFrame,
    strict: bool = True,
    raise_on_error: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate NASA POWER weather data.
    
    Args:
        df: DataFrame to validate
        strict: Use strict validation
        raise_on_error: Raise on validation failure
        
    Returns:
        Tuple of (validated DataFrame, list of errors)
    """
    errors = []
    schema = CanonicalPowerSchema if strict else CanonicalPowerSchemaLenient
    
    try:
        validated = schema.validate(df, lazy=True)
        return validated, errors
    except pa.errors.SchemaErrors as e:
        errors = [str(err) for err in e.failure_cases["failure_case"].tolist()]
        if raise_on_error:
            raise
        return df, errors


def validate_forecast_weather(
    df: pd.DataFrame,
    strict: bool = True,
    raise_on_error: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate Open-Meteo forecast data.
    
    Args:
        df: DataFrame to validate
        strict: Use strict validation
        raise_on_error: Raise on validation failure
        
    Returns:
        Tuple of (validated DataFrame, list of errors)
    """
    errors = []
    schema = CanonicalForecastSchema if strict else CanonicalForecastSchemaLenient
    
    try:
        validated = schema.validate(df, lazy=True)
        return validated, errors
    except pa.errors.SchemaErrors as e:
        errors = [str(err) for err in e.failure_cases["failure_case"].tolist()]
        if raise_on_error:
            raise
        return df, errors


def check_weather_key_uniqueness(
    df: pd.DataFrame,
    key_columns: Tuple[str, ...] = ("date", "district"),
) -> pd.DataFrame:
    """
    Check for duplicate keys in weather data.
    
    Args:
        df: DataFrame to check
        key_columns: Columns forming the natural key
        
    Returns:
        DataFrame of duplicate rows (empty if no duplicates)
    """
    duplicated_mask = df.duplicated(subset=list(key_columns), keep=False)
    return df[duplicated_mask].copy()


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def summarize_power_weather(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for NASA POWER data.
    
    Args:
        df: POWER weather DataFrame
        
    Returns:
        Dict with summary statistics
    """
    summary = {
        "total_rows": len(df),
        "columns": list(df.columns),
    }
    
    if "date" in df.columns:
        summary["date_min"] = str(df["date"].min())
        summary["date_max"] = str(df["date"].max())
        summary["unique_dates"] = df["date"].nunique()
    
    if "district" in df.columns:
        summary["unique_districts"] = df["district"].nunique()
        summary["districts"] = sorted(df["district"].unique().tolist())
    
    # Weather variable stats
    weather_vars = ["t2m_max", "t2m_min", "t2m_mean", "rh2m", "precipitation", "ws2m", "srad"]
    for var in weather_vars:
        if var in df.columns:
            summary[f"{var}_mean"] = df[var].mean()
            summary[f"{var}_null_pct"] = df[var].isna().sum() / len(df) * 100 if len(df) > 0 else 0
    
    # Key uniqueness
    duplicates = check_weather_key_uniqueness(df, POWER_NATURAL_KEY)
    summary["duplicate_rows"] = len(duplicates)
    summary["has_duplicates"] = len(duplicates) > 0
    
    return summary


def summarize_forecast_weather(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for Open-Meteo forecast data.
    
    Args:
        df: Forecast weather DataFrame
        
    Returns:
        Dict with summary statistics
    """
    summary = {
        "total_rows": len(df),
        "columns": list(df.columns),
    }
    
    if "date" in df.columns:
        summary["date_min"] = str(df["date"].min())
        summary["date_max"] = str(df["date"].max())
        summary["unique_dates"] = df["date"].nunique()
    
    if "district" in df.columns:
        summary["unique_districts"] = df["district"].nunique()
        summary["districts"] = sorted(df["district"].unique().tolist())
    
    # Weather variable stats
    weather_vars = ["temperature_max", "temperature_min", "precipitation_sum", "windspeed_max"]
    for var in weather_vars:
        if var in df.columns:
            summary[f"{var}_mean"] = df[var].mean()
            summary[f"{var}_null_pct"] = df[var].isna().sum() / len(df) * 100 if len(df) > 0 else 0
    
    # Key uniqueness
    duplicates = check_weather_key_uniqueness(df, FORECAST_NATURAL_KEY)
    summary["duplicate_rows"] = len(duplicates)
    summary["has_duplicates"] = len(duplicates) > 0
    
    return summary


def generate_weather_qc_report(
    power_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a Markdown QC report for weather datasets.
    
    Args:
        power_df: NASA POWER DataFrame
        forecast_df: Open-Meteo forecast DataFrame
        output_path: Optional path to save report
        
    Returns:
        Markdown report string
    """
    lines = [
        "# Weather Data Quality Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
    ]
    
    if power_df is not None:
        power_summary = summarize_power_weather(power_df)
        lines.extend([
            "## NASA POWER Historical Data",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Rows | {power_summary['total_rows']:,} |",
            f"| Unique Districts | {power_summary.get('unique_districts', 'N/A')} |",
            f"| Date Range | {power_summary.get('date_min', 'N/A')} to {power_summary.get('date_max', 'N/A')} |",
            f"| Unique Dates | {power_summary.get('unique_dates', 'N/A')} |",
            f"| Duplicate Rows | {power_summary.get('duplicate_rows', 0)} |",
            "",
        ])
    
    if forecast_df is not None:
        forecast_summary = summarize_forecast_weather(forecast_df)
        lines.extend([
            "## Open-Meteo Forecast Data",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Rows | {forecast_summary['total_rows']:,} |",
            f"| Unique Districts | {forecast_summary.get('unique_districts', 'N/A')} |",
            f"| Date Range | {forecast_summary.get('date_min', 'N/A')} to {forecast_summary.get('date_max', 'N/A')} |",
            f"| Unique Dates | {forecast_summary.get('unique_dates', 'N/A')} |",
            f"| Duplicate Rows | {forecast_summary.get('duplicate_rows', 0)} |",
            "",
        ])
    
    report = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    
    return report
