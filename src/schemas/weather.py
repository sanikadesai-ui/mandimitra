"""
MANDIMITRA - Pandera Schemas for Weather Data

Defines validation schemas for:
- NASA POWER historical daily data
- Open-Meteo 16-day forecast data

All data is Maharashtra-only (36 district headquarters).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema


# =============================================================================
# NASA POWER SCHEMA
# =============================================================================

def create_power_schema() -> DataFrameSchema:
    """
    Create Pandera schema for NASA POWER daily data.
    
    Columns:
        - date: Date of observation
        - district: Maharashtra district name
        - latitude: Location latitude
        - longitude: Location longitude
        - PRECTOTCORR: Precipitation corrected (mm/day)
        - T2M: Temperature at 2m (°C)
        - T2M_MAX: Max temperature (°C)
        - T2M_MIN: Min temperature (°C)
        - RH2M: Relative humidity at 2m (%)
    """
    schema = DataFrameSchema(
        columns={
            "date": Column(
                "datetime64[ns]",
                nullable=False,
                coerce=True,
            ),
            "district": Column(
                str,
                checks=[Check.str_length(min_value=1)],
                nullable=False,
                coerce=True,
            ),
            "latitude": Column(
                float,
                checks=[
                    Check.ge(-90),
                    Check.le(90),
                ],
                nullable=False,
                coerce=True,
            ),
            "longitude": Column(
                float,
                checks=[
                    Check.ge(-180),
                    Check.le(180),
                ],
                nullable=False,
                coerce=True,
            ),
            "PRECTOTCORR": Column(
                float,
                checks=[Check.ge(0)],  # Precipitation can't be negative
                nullable=True,
                coerce=True,
            ),
            "T2M": Column(
                float,
                checks=[
                    Check.ge(-50),  # Reasonable temperature range
                    Check.le(60),
                ],
                nullable=True,
                coerce=True,
            ),
            "T2M_MAX": Column(
                float,
                checks=[
                    Check.ge(-50),
                    Check.le(60),
                ],
                nullable=True,
                coerce=True,
            ),
            "T2M_MIN": Column(
                float,
                checks=[
                    Check.ge(-50),
                    Check.le(60),
                ],
                nullable=True,
                coerce=True,
            ),
            "RH2M": Column(
                float,
                checks=[
                    Check.ge(0),
                    Check.le(100),
                ],
                nullable=True,
                coerce=True,
            ),
        },
        strict=False,  # Allow extra columns
        coerce=True,
    )
    return schema


PowerSchema = create_power_schema()


# =============================================================================
# OPEN-METEO FORECAST SCHEMA
# =============================================================================

def create_openmeteo_schema() -> DataFrameSchema:
    """
    Create Pandera schema for Open-Meteo forecast data.
    
    Columns:
        - date: Forecast date
        - district: Maharashtra district name
        - latitude: Location latitude
        - longitude: Location longitude
        - precipitation_sum: Daily precipitation sum (mm)
        - precipitation_probability_max: Max precipitation probability (%)
        - temperature_2m_max: Max temperature (°C)
        - temperature_2m_min: Min temperature (°C)
        - relative_humidity_2m_max: Max relative humidity (%)
        - relative_humidity_2m_min: Min relative humidity (%)
    """
    schema = DataFrameSchema(
        columns={
            "date": Column(
                "datetime64[ns]",
                nullable=False,
                coerce=True,
            ),
            "district": Column(
                str,
                checks=[Check.str_length(min_value=1)],
                nullable=False,
                coerce=True,
            ),
            "latitude": Column(
                float,
                checks=[Check.ge(-90), Check.le(90)],
                nullable=False,
                coerce=True,
            ),
            "longitude": Column(
                float,
                checks=[Check.ge(-180), Check.le(180)],
                nullable=False,
                coerce=True,
            ),
            "precipitation_sum": Column(
                float,
                checks=[Check.ge(0)],
                nullable=True,
                coerce=True,
            ),
            "precipitation_probability_max": Column(
                float,
                checks=[Check.ge(0), Check.le(100)],
                nullable=True,
                coerce=True,
            ),
            "temperature_2m_max": Column(
                float,
                checks=[Check.ge(-50), Check.le(60)],
                nullable=True,
                coerce=True,
            ),
            "temperature_2m_min": Column(
                float,
                checks=[Check.ge(-50), Check.le(60)],
                nullable=True,
                coerce=True,
            ),
            "relative_humidity_2m_max": Column(
                float,
                checks=[Check.ge(0), Check.le(100)],
                nullable=True,
                coerce=True,
            ),
            "relative_humidity_2m_min": Column(
                float,
                checks=[Check.ge(0), Check.le(100)],
                nullable=True,
                coerce=True,
            ),
        },
        strict=False,
        coerce=True,
    )
    return schema


OpenMeteoSchema = create_openmeteo_schema()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_power_dataframe(
    df: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Validate NASA POWER DataFrame.
    
    Args:
        df: Raw DataFrame
        strict: If True, raise on validation errors
        
    Returns:
        Validated DataFrame
    """
    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    try:
        return PowerSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        if strict:
            raise
        return df


def validate_openmeteo_dataframe(
    df: pd.DataFrame,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Validate Open-Meteo forecast DataFrame.
    
    Args:
        df: Raw DataFrame
        strict: If True, raise on validation errors
        
    Returns:
        Validated DataFrame
    """
    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    try:
        return OpenMeteoSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        if strict:
            raise
        return df


def summarize_weather_data(df: pd.DataFrame, source: str = "power") -> Dict[str, Any]:
    """
    Generate summary statistics for weather data.
    
    Args:
        df: Validated weather DataFrame
        source: "power" or "openmeteo"
        
    Returns:
        Summary dict
    """
    summary = {
        "source": source,
        "total_rows": len(df),
    }
    
    if "district" in df.columns:
        summary["unique_districts"] = df["district"].nunique()
        summary["districts"] = sorted(df["district"].unique().tolist())
    
    if "date" in df.columns:
        summary["date_range"] = {
            "min": str(df["date"].min()),
            "max": str(df["date"].max()),
        }
        summary["unique_dates"] = df["date"].nunique()
    
    if source == "power":
        if "PRECTOTCORR" in df.columns:
            summary["precipitation_stats"] = {
                "mean": float(df["PRECTOTCORR"].mean()) if df["PRECTOTCORR"].notna().any() else None,
                "max": float(df["PRECTOTCORR"].max()) if df["PRECTOTCORR"].notna().any() else None,
            }
        if "T2M" in df.columns:
            summary["temperature_stats"] = {
                "mean": float(df["T2M"].mean()) if df["T2M"].notna().any() else None,
                "min": float(df["T2M_MIN"].min()) if "T2M_MIN" in df.columns and df["T2M_MIN"].notna().any() else None,
                "max": float(df["T2M_MAX"].max()) if "T2M_MAX" in df.columns and df["T2M_MAX"].notna().any() else None,
            }
    
    elif source == "openmeteo":
        if "precipitation_sum" in df.columns:
            summary["precipitation_stats"] = {
                "mean": float(df["precipitation_sum"].mean()) if df["precipitation_sum"].notna().any() else None,
                "max": float(df["precipitation_sum"].max()) if df["precipitation_sum"].notna().any() else None,
            }
    
    return summary


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PowerSchema",
    "OpenMeteoSchema",
    "validate_power_dataframe",
    "validate_openmeteo_dataframe",
    "summarize_weather_data",
]
