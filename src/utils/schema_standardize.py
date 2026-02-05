#!/usr/bin/env python3
"""
MANDIMITRA - Schema Standardization Utilities

Helpers for column renaming, data type conversion, and date parsing.
Ensures consistent schema across historical and current mandi data sources.

Canonical Mandi Schema (snake_case):
    state, district, district_raw, market, commodity, variety, grade,
    arrival_date, min_price, max_price, modal_price, commodity_code,
    source, ingested_at_utc

Usage:
    from src.utils.schema_standardize import (
        standardize_mandi_columns,
        parse_arrival_date,
        CANONICAL_MANDI_COLUMNS,
    )
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# CANONICAL SCHEMA DEFINITIONS
# =============================================================================

# Canonical mandi column names (in order)
CANONICAL_MANDI_COLUMNS: List[str] = [
    "state",
    "district",
    "district_raw",
    "market",
    "commodity",
    "variety",
    "grade",
    "arrival_date",
    "min_price",
    "max_price",
    "modal_price",
    "commodity_code",
    "source",
    "ingested_at_utc",
]

# Required columns (must be present after standardization)
REQUIRED_MANDI_COLUMNS: List[str] = [
    "state",
    "district",
    "market",
    "commodity",
    "arrival_date",
]

# Natural key for deduplication
MANDI_NATURAL_KEY: Tuple[str, ...] = (
    "state",
    "district",
    "market",
    "commodity",
    "variety",
    "grade",
    "arrival_date",
)

# =============================================================================
# COLUMN NAME MAPPINGS
# =============================================================================

# Map from various source column names to canonical names
# Key: lowercase source column name, Value: canonical column name
COLUMN_NAME_MAP: Dict[str, str] = {
    # State
    "state": "state",
    "state_name": "state",
    "statename": "state",
    
    # District - keep as district, we'll copy to district_raw separately
    "district": "district",
    "district_name": "district",
    "districtname": "district",
    "dist": "district",
    
    # Market
    "market": "market",
    "market_name": "market",
    "marketname": "market",
    "mandi": "market",
    "mandi_name": "market",
    
    # Commodity
    "commodity": "commodity",
    "commodity_name": "commodity",
    "commodityname": "commodity",
    "crop": "commodity",
    
    # Variety
    "variety": "variety",
    "variety_name": "variety",
    "varietyname": "variety",
    
    # Grade
    "grade": "grade",
    "grade_name": "grade",
    "gradename": "grade",
    "quality": "grade",
    
    # Arrival Date
    "arrival_date": "arrival_date",
    "arrivaldate": "arrival_date",
    "date": "arrival_date",
    "price_date": "arrival_date",
    "pricedate": "arrival_date",
    "reported_date": "arrival_date",
    
    # Prices
    "min_price": "min_price",
    "minprice": "min_price",
    "minimum_price": "min_price",
    "min": "min_price",
    
    "max_price": "max_price",
    "maxprice": "max_price",
    "maximum_price": "max_price",
    "max": "max_price",
    
    "modal_price": "modal_price",
    "modalprice": "modal_price",
    "mode_price": "modal_price",
    "modal": "modal_price",
    "mod_price": "modal_price",
    
    # Commodity Code
    "commodity_code": "commodity_code",
    "commoditycode": "commodity_code",
    "comm_code": "commodity_code",
    "code": "commodity_code",
}

# =============================================================================
# DATE PARSING
# =============================================================================

# Common date formats in mandi data
DATE_FORMATS: List[str] = [
    "%Y-%m-%d",           # 2024-01-15
    "%d-%m-%Y",           # 15-01-2024
    "%d/%m/%Y",           # 15/01/2024
    "%Y/%m/%d",           # 2024/01/15
    "%d-%b-%Y",           # 15-Jan-2024
    "%d %b %Y",           # 15 Jan 2024
    "%Y-%m-%d %H:%M:%S",  # 2024-01-15 00:00:00
    "%d-%m-%Y %H:%M:%S",  # 15-01-2024 00:00:00
]


def parse_arrival_date(
    date_value: Any,
    formats: Optional[List[str]] = None,
) -> Optional[datetime]:
    """
    Parse a date value to datetime.
    
    Args:
        date_value: Date value (string, datetime, or pd.Timestamp)
        formats: List of date formats to try
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    if date_value is None or pd.isna(date_value):
        return None
    
    # Already a datetime
    if isinstance(date_value, datetime):
        return date_value
    
    # Pandas Timestamp
    if isinstance(date_value, pd.Timestamp):
        return date_value.to_pydatetime()
    
    # String parsing
    if isinstance(date_value, str):
        date_str = date_value.strip()
        
        if not date_str:
            return None
        
        formats = formats or DATE_FORMATS
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try pandas parser as fallback
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except Exception:
            logger.warning(f"Could not parse date: '{date_str}'")
            return None
    
    # Numeric (Excel serial date or timestamp)
    if isinstance(date_value, (int, float)):
        try:
            # Assume pandas Timestamp from numeric
            return pd.Timestamp(date_value, unit="D").to_pydatetime()
        except Exception:
            return None
    
    return None


def parse_dates_column(
    series: pd.Series,
    formats: Optional[List[str]] = None,
) -> pd.Series:
    """
    Parse a pandas Series of dates to datetime.
    
    Args:
        series: Series of date values
        formats: List of date formats to try
        
    Returns:
        Series of datetime values
    """
    # Try pandas to_datetime first (vectorized, fast)
    try:
        result = pd.to_datetime(series, errors="coerce", dayfirst=True)
        null_count = result.isna().sum()
        if null_count < len(series) * 0.1:  # Less than 10% nulls
            return result
    except Exception:
        pass
    
    # Fall back to row-by-row parsing
    return series.apply(lambda x: parse_arrival_date(x, formats))


# =============================================================================
# COLUMN STANDARDIZATION
# =============================================================================

def standardize_column_name(col_name: str) -> str:
    """
    Standardize a single column name.
    
    Args:
        col_name: Original column name
        
    Returns:
        Standardized column name (snake_case)
    """
    # Convert to lowercase
    cleaned = col_name.strip().lower()
    
    # Replace spaces and special chars with underscore
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip("_")
    
    # Map to canonical name if known
    return COLUMN_NAME_MAP.get(cleaned, cleaned)


def standardize_mandi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize mandi DataFrame columns to canonical schema.
    
    Args:
        df: DataFrame with raw column names
        
    Returns:
        DataFrame with standardized column names
    """
    # Create column mapping
    rename_map = {}
    for col in df.columns:
        new_name = standardize_column_name(col)
        rename_map[col] = new_name
    
    # Rename columns
    df_std = df.rename(columns=rename_map)
    
    # If district_raw doesn't exist but district does, copy it
    if "district_raw" not in df_std.columns and "district" in df_std.columns:
        df_std["district_raw"] = df_std["district"]
    
    # Add missing columns with None
    for col in CANONICAL_MANDI_COLUMNS:
        if col not in df_std.columns:
            df_std[col] = None
    
    return df_std


def enforce_mandi_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce correct data types for mandi DataFrame.
    
    Args:
        df: DataFrame with standardized column names
        
    Returns:
        DataFrame with correct data types
    """
    df = df.copy()
    
    # String columns
    string_cols = ["state", "district", "district_raw", "market", "commodity", "variety", "grade", "source"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "").replace("None", "")
    
    # Float columns (prices)
    float_cols = ["min_price", "max_price", "modal_price"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Nullable int/float for commodity_code
    if "commodity_code" in df.columns:
        df["commodity_code"] = pd.to_numeric(df["commodity_code"], errors="coerce")
    
    # Date column
    if "arrival_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["arrival_date"]):
            df["arrival_date"] = parse_dates_column(df["arrival_date"])
    
    # Timestamp column
    if "ingested_at_utc" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["ingested_at_utc"]):
            df["ingested_at_utc"] = pd.to_datetime(df["ingested_at_utc"], errors="coerce")
    
    return df


# =============================================================================
# PRICE VALIDATION
# =============================================================================

def validate_prices(
    df: pd.DataFrame,
    fix_violations: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Validate and optionally fix price relationships.
    
    Rules:
        - All prices >= 0
        - min_price <= max_price (when both present)
        - min_price <= modal_price <= max_price (when all present)
    
    Args:
        df: DataFrame with price columns
        fix_violations: If True, set violating prices to NaN
        
    Returns:
        Tuple of (DataFrame, stats dict with violation counts)
    """
    df = df.copy()
    stats = {
        "negative_prices": 0,
        "min_gt_max": 0,
        "modal_out_of_range": 0,
        "rows_fixed": 0,
    }
    
    # Check for negative prices
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            neg_mask = df[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                stats["negative_prices"] += neg_count
                if fix_violations:
                    df.loc[neg_mask, col] = None
                    stats["rows_fixed"] += neg_count
    
    # Check min <= max
    if "min_price" in df.columns and "max_price" in df.columns:
        both_present = df["min_price"].notna() & df["max_price"].notna()
        violation_mask = both_present & (df["min_price"] > df["max_price"])
        violation_count = violation_mask.sum()
        if violation_count > 0:
            stats["min_gt_max"] = violation_count
            if fix_violations:
                # Swap min and max where violated
                df.loc[violation_mask, ["min_price", "max_price"]] = df.loc[
                    violation_mask, ["max_price", "min_price"]
                ].values
                stats["rows_fixed"] += violation_count
    
    # Check modal within range
    if all(col in df.columns for col in ["min_price", "max_price", "modal_price"]):
        all_present = (
            df["min_price"].notna() & 
            df["max_price"].notna() & 
            df["modal_price"].notna()
        )
        modal_low = all_present & (df["modal_price"] < df["min_price"])
        modal_high = all_present & (df["modal_price"] > df["max_price"])
        violation_count = (modal_low | modal_high).sum()
        if violation_count > 0:
            stats["modal_out_of_range"] = violation_count
            if fix_violations:
                # Clamp modal to [min, max]
                df.loc[modal_low, "modal_price"] = df.loc[modal_low, "min_price"]
                df.loc[modal_high, "modal_price"] = df.loc[modal_high, "max_price"]
                stats["rows_fixed"] += violation_count
    
    return df, stats


# =============================================================================
# DATA QUALITY HELPERS
# =============================================================================

def compute_completeness_score(row: pd.Series) -> int:
    """
    Compute completeness score for a row (used in dedup tiebreaker).
    
    Score = count of non-null values in key price columns.
    
    Args:
        row: DataFrame row
        
    Returns:
        Completeness score (0-3)
    """
    score = 0
    for col in ["min_price", "max_price", "modal_price"]:
        if col in row.index and pd.notna(row[col]):
            score += 1
    return score


def get_missingness_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get percentage of missing values for each column.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dict of column -> missing percentage
    """
    total_rows = len(df)
    if total_rows == 0:
        return {}
    
    return {
        col: (df[col].isna().sum() / total_rows) * 100
        for col in df.columns
    }


def add_ingestion_metadata(
    df: pd.DataFrame,
    source: str,
    ingested_at: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Add source and ingestion timestamp columns.
    
    Args:
        df: DataFrame to annotate
        source: Source identifier ("history" or "current")
        ingested_at: Ingestion timestamp (default: now UTC)
        
    Returns:
        DataFrame with source and ingested_at_utc columns
    """
    df = df.copy()
    df["source"] = source
    df["ingested_at_utc"] = ingested_at or datetime.now(timezone.utc)
    return df


# =============================================================================
# WEATHER SCHEMA HELPERS
# =============================================================================

CANONICAL_POWER_COLUMNS: List[str] = [
    "date",
    "district",
    "latitude",
    "longitude",
    "precipitation",  # PRECTOTCORR
    "temperature_avg",  # T2M
    "temperature_max",  # T2M_MAX
    "temperature_min",  # T2M_MIN
    "relative_humidity",  # RH2M
]

POWER_COLUMN_MAP: Dict[str, str] = {
    "prectotcorr": "precipitation",
    "t2m": "temperature_avg",
    "t2m_max": "temperature_max",
    "t2m_min": "temperature_min",
    "rh2m": "relative_humidity",
}

CANONICAL_FORECAST_COLUMNS: List[str] = [
    "date",
    "district",
    "latitude",
    "longitude",
    "precipitation_sum",
    "precipitation_probability_max",
    "temperature_max",  # temperature_2m_max
    "temperature_min",  # temperature_2m_min
    "humidity_max",  # relative_humidity_2m_max
    "humidity_min",  # relative_humidity_2m_min
]

FORECAST_COLUMN_MAP: Dict[str, str] = {
    "temperature_2m_max": "temperature_max",
    "temperature_2m_min": "temperature_min",
    "relative_humidity_2m_max": "humidity_max",
    "relative_humidity_2m_min": "humidity_min",
}


def standardize_power_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize NASA POWER DataFrame columns."""
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in POWER_COLUMN_MAP:
            rename_map[col] = POWER_COLUMN_MAP[col_lower]
        else:
            rename_map[col] = col_lower
    
    return df.rename(columns=rename_map)


def standardize_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Open-Meteo forecast DataFrame columns."""
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in FORECAST_COLUMN_MAP:
            rename_map[col] = FORECAST_COLUMN_MAP[col_lower]
        else:
            rename_map[col] = col_lower
    
    return df.rename(columns=rename_map)
