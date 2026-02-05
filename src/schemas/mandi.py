"""
MANDIMITRA - Pandera Schemas for Mandi Data

Defines strict validation schemas for mandi price data with:
- Maharashtra-only enforcement
- Robust date parsing
- Column normalization
- Type coercion

Usage:
    from src.schemas.mandi import MandiSchema, validate_mandi_dataframe
    
    validated_df = validate_mandi_dataframe(raw_df, strict=True)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from pandera.typing import Series

from src.utils.maharashtra import MAHARASHTRA_STATE_NAME, is_maharashtra_state


# =============================================================================
# COLUMN MAPPING (source variations -> standard names)
# =============================================================================

COLUMN_VARIATIONS: Dict[str, List[str]] = {
    "state": ["state", "State", "STATE", "state_name", "State_Name"],
    "district": ["district", "District", "DISTRICT", "district_name", "District_Name"],
    "market": ["market", "Market", "MARKET", "market_name", "Market_Name", "mandi", "Mandi"],
    "commodity": ["commodity", "Commodity", "COMMODITY", "commodity_name", "Commodity_Name"],
    "variety": ["variety", "Variety", "VARIETY", "variety_name", "Variety_Name"],
    "grade": ["grade", "Grade", "GRADE"],
    "arrival_date": ["arrival_date", "Arrival_Date", "date", "Date", "arrival_dt", "reported_date", "Reported_Date", "price_date"],
    "min_price": ["min_price", "Min_Price", "minimum_price", "Minimum_Price", "price_min", "Min Price"],
    "max_price": ["max_price", "Max_Price", "maximum_price", "Maximum_Price", "price_max", "Max Price"],
    "modal_price": ["modal_price", "Modal_Price", "mode_price", "Mode_Price", "avg_price", "price", "Price", "Modal Price"],
}

# Date formats to try when parsing
DATE_FORMATS: List[str] = [
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%d-%m-%Y", 
    "%Y/%m/%d",
    "%d %b %Y",
    "%d-%b-%Y",
    "%d %B %Y",
    "%Y%m%d",
]


# =============================================================================
# SCHEMA DEFINITION
# =============================================================================

def create_mandi_schema(strict_maharashtra: bool = True) -> DataFrameSchema:
    """
    Create Pandera schema for mandi data validation.
    
    Args:
        strict_maharashtra: If True, enforce Maharashtra-only constraint
        
    Returns:
        Pandera DataFrameSchema
    """
    # Maharashtra check
    def check_maharashtra(state: str) -> bool:
        return is_maharashtra_state(state)
    
    checks_state = [Check.str_length(min_value=1)]
    if strict_maharashtra:
        checks_state.append(Check(check_maharashtra, error="State must be Maharashtra"))
    
    schema = DataFrameSchema(
        columns={
            "state": Column(
                str,
                checks=checks_state,
                nullable=False,
                coerce=True,
            ),
            "district": Column(
                str,
                checks=[Check.str_length(min_value=1)],
                nullable=False,
                coerce=True,
            ),
            "market": Column(
                str,
                checks=[Check.str_length(min_value=1)],
                nullable=False,
                coerce=True,
            ),
            "commodity": Column(
                str,
                checks=[Check.str_length(min_value=1)],
                nullable=False,
                coerce=True,
            ),
            "variety": Column(
                str,
                nullable=True,
                coerce=True,
            ),
            "grade": Column(
                str,
                nullable=True,
                coerce=True,
            ),
            "arrival_date": Column(
                "datetime64[ns]",
                nullable=False,
                coerce=True,
            ),
            "min_price": Column(
                float,
                checks=[Check.ge(0)],
                nullable=True,
                coerce=True,
            ),
            "max_price": Column(
                float,
                checks=[Check.ge(0)],
                nullable=True,
                coerce=True,
            ),
            "modal_price": Column(
                float,
                checks=[Check.ge(0)],
                nullable=True,
                coerce=True,
            ),
        },
        strict=False,  # Allow extra columns
        coerce=True,
    )
    
    return schema


# Global schema instance
MandiSchema = create_mandi_schema(strict_maharashtra=True)
MandiSchemaLenient = create_mandi_schema(strict_maharashtra=False)


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standard schema.
    
    Handles various column name variations from different data sources.
    
    Args:
        df: DataFrame with potentially non-standard column names
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    
    # Build reverse mapping: source_name -> standard_name
    reverse_map = {}
    for standard, variations in COLUMN_VARIATIONS.items():
        for var in variations:
            reverse_map[var.lower()] = standard
    
    # Rename columns
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in reverse_map:
            rename_map[col] = reverse_map[col_lower]
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


def parse_dates(df: pd.DataFrame, date_column: str = "arrival_date") -> pd.DataFrame:
    """
    Parse dates robustly, trying multiple formats.
    
    Args:
        df: DataFrame with date column
        date_column: Name of date column
        
    Returns:
        DataFrame with parsed datetime column
    """
    if date_column not in df.columns:
        return df
    
    df = df.copy()
    
    # If already datetime, return
    if pd.api.types.is_datetime64_any_dtype(df[date_column]):
        return df
    
    # Try each format
    parsed = None
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(df[date_column], format=fmt, errors="coerce")
            # If most values parsed successfully, use this format
            if parsed.notna().sum() > len(df) * 0.5:
                break
        except Exception:
            continue
    
    # Fallback to pandas auto-detection
    if parsed is None or parsed.isna().all():
        parsed = pd.to_datetime(df[date_column], infer_datetime_format=True, errors="coerce")
    
    df[date_column] = parsed
    return df


def normalize_strings(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize string columns: strip whitespace, consistent casing.
    
    Args:
        df: DataFrame to normalize
        columns: Specific columns to normalize (default: auto-detect string columns)
        
    Returns:
        Normalized DataFrame
    """
    df = df.copy()
    
    if columns is None:
        columns = ["state", "district", "market", "commodity", "variety", "grade"]
    
    for col in columns:
        if col in df.columns:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            # Title case for names (except state which should be exact)
            if col != "state":
                df[col] = df[col].str.title()
            # Handle "nan" strings
            df[col] = df[col].replace(["nan", "NaN", "None", ""], pd.NA)
    
    return df


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price columns to float, handling various formats.
    
    Args:
        df: DataFrame with price columns
        
    Returns:
        DataFrame with normalized prices
    """
    df = df.copy()
    
    price_cols = ["min_price", "max_price", "modal_price"]
    
    for col in price_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Ensure non-negative
            df.loc[df[col] < 0, col] = pd.NA
    
    return df


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_mandi_dataframe(
    df: pd.DataFrame,
    strict: bool = True,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and normalize a mandi DataFrame.
    
    Args:
        df: Raw DataFrame to validate
        strict: If True, enforce Maharashtra-only and raise on violations
        normalize: If True, normalize columns, dates, strings, prices
        
    Returns:
        Tuple of (validated_df, validation_report)
        
    Raises:
        pa.errors.SchemaError: If strict=True and validation fails
    """
    report = {
        "input_rows": len(df),
        "input_columns": list(df.columns),
        "normalized": normalize,
        "strict_maharashtra": strict,
        "errors": [],
        "warnings": [],
    }
    
    # Normalize if requested
    if normalize:
        df = normalize_columns(df)
        df = parse_dates(df)
        df = normalize_strings(df)
        df = normalize_prices(df)
    
    report["columns_after_normalization"] = list(df.columns)
    
    # Check for required columns
    required = ["state", "district", "market", "commodity", "arrival_date"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        report["errors"].append(f"Missing required columns: {missing}")
        if strict:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Filter to Maharashtra only
    if "state" in df.columns:
        non_mh_mask = ~df["state"].apply(is_maharashtra_state)
        non_mh_count = non_mh_mask.sum()
        
        if non_mh_count > 0:
            non_mh_states = df.loc[non_mh_mask, "state"].value_counts().head(5).to_dict()
            report["non_maharashtra_count"] = int(non_mh_count)
            report["non_maharashtra_samples"] = non_mh_states
            
            if strict:
                report["errors"].append(
                    f"HARD CONSTRAINT VIOLATION: {non_mh_count} non-Maharashtra records. "
                    f"Samples: {non_mh_states}"
                )
                raise ValueError(
                    f"Maharashtra-only constraint violated: {non_mh_count} non-MH records. "
                    f"Samples: {non_mh_states}"
                )
            else:
                report["warnings"].append(
                    f"Dropped {non_mh_count} non-Maharashtra records"
                )
                df = df[~non_mh_mask].copy()
    
    # Validate with Pandera schema
    schema = MandiSchema if strict else MandiSchemaLenient
    try:
        df = schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        report["schema_errors"] = str(e)
        if strict:
            raise
        else:
            report["warnings"].append(f"Schema validation issues: {len(e.failure_cases)} failures")
    
    # Drop duplicates
    dedup_cols = ["state", "district", "market", "commodity", "variety", "grade", "arrival_date"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    
    rows_before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="last")
    duplicates_removed = rows_before - len(df)
    
    if duplicates_removed > 0:
        report["duplicates_removed"] = duplicates_removed
    
    report["output_rows"] = len(df)
    report["success"] = len(report["errors"]) == 0
    
    return df, report


def summarize_mandi_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for mandi data.
    
    Args:
        df: Validated mandi DataFrame
        
    Returns:
        Summary dict
    """
    summary = {
        "total_rows": len(df),
    }
    
    if "state" in df.columns:
        summary["unique_states"] = df["state"].nunique()
        summary["states"] = df["state"].unique().tolist()
    
    if "district" in df.columns:
        summary["unique_districts"] = df["district"].nunique()
        summary["districts"] = sorted(df["district"].unique().tolist())
    
    if "market" in df.columns:
        summary["unique_markets"] = df["market"].nunique()
    
    if "commodity" in df.columns:
        summary["unique_commodities"] = df["commodity"].nunique()
        summary["top_commodities"] = df["commodity"].value_counts().head(10).to_dict()
    
    if "arrival_date" in df.columns:
        summary["date_range"] = {
            "min": str(df["arrival_date"].min()),
            "max": str(df["arrival_date"].max()),
        }
        summary["unique_dates"] = df["arrival_date"].nunique()
    
    if "modal_price" in df.columns:
        summary["price_stats"] = {
            "min": float(df["modal_price"].min()) if df["modal_price"].notna().any() else None,
            "max": float(df["modal_price"].max()) if df["modal_price"].notna().any() else None,
            "mean": float(df["modal_price"].mean()) if df["modal_price"].notna().any() else None,
        }
    
    return summary


# =============================================================================
# SCHEMA INFO EXPORT
# =============================================================================

MANDI_SCHEMA_COLUMNS = [
    "state",
    "district", 
    "market",
    "commodity",
    "variety",
    "grade",
    "arrival_date",
    "min_price",
    "max_price",
    "modal_price",
]

MANDI_DEDUP_KEYS = [
    "state",
    "district",
    "market",
    "commodity",
    "variety",
    "grade",
    "arrival_date",
]

__all__ = [
    "MandiSchema",
    "MandiSchemaLenient",
    "COLUMN_VARIATIONS",
    "DATE_FORMATS",
    "MANDI_SCHEMA_COLUMNS",
    "MANDI_DEDUP_KEYS",
    "normalize_columns",
    "parse_dates",
    "normalize_strings",
    "normalize_prices",
    "validate_mandi_dataframe",
    "summarize_mandi_data",
]
