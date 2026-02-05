#!/usr/bin/env python3
"""
MANDIMITRA - Pandera Schemas for Canonical Mandi Data

Defines strict validation schemas for the canonical mandi dataset with:
- Maharashtra-only enforcement
- Key uniqueness validation
- Price sanity checks
- District normalization validation

Canonical Schema:
    state, district, district_raw, market, commodity, variety, grade,
    arrival_date, min_price, max_price, modal_price, commodity_code,
    source, ingested_at_utc

Natural Key (must be unique):
    (state, district, market, commodity, variety, grade, arrival_date)

Usage:
    from src.schemas.mandi_canonical import (
        CanonicalMandiSchema,
        validate_canonical_mandi,
        check_key_uniqueness,
    )
    
    # Validate DataFrame
    validated_df = validate_canonical_mandi(df, strict=True)
    
    # Check for duplicates
    duplicates = check_key_uniqueness(df)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from pandera.typing import Series

# Import district normalizer
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.district_normalize import CANONICAL_DISTRICTS, get_normalizer


# =============================================================================
# CONSTANTS
# =============================================================================

MAHARASHTRA_STATE_NAME = "Maharashtra"

# Natural key columns for deduplication
NATURAL_KEY_COLUMNS: Tuple[str, ...] = (
    "state",
    "district",
    "market",
    "commodity",
    "variety",
    "grade",
    "arrival_date",
)

# All canonical columns
CANONICAL_COLUMNS: List[str] = [
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

# Required columns (non-nullable)
REQUIRED_COLUMNS: List[str] = [
    "state",
    "district",
    "market",
    "commodity",
    "arrival_date",
]

# Valid source values
VALID_SOURCES: Set[str] = {"history", "current"}


# =============================================================================
# CUSTOM CHECKS
# =============================================================================

def check_is_maharashtra(state: str) -> bool:
    """Check if state is Maharashtra (case-insensitive)."""
    if not state or not isinstance(state, str):
        return False
    return state.strip().lower() == "maharashtra"


def check_canonical_district(district: str) -> bool:
    """Check if district is in the canonical list of 36."""
    if not district or not isinstance(district, str):
        return False
    return district in CANONICAL_DISTRICTS


def check_valid_source(source: str) -> bool:
    """Check if source is valid."""
    if not source or not isinstance(source, str):
        return True  # Allow None
    return source in VALID_SOURCES


def check_price_non_negative(price: float) -> bool:
    """Check if price is non-negative."""
    if pd.isna(price):
        return True  # Allow nulls
    return price >= 0


def check_min_max_relationship(df: pd.DataFrame) -> bool:
    """Check that min_price <= max_price when both present."""
    mask = df["min_price"].notna() & df["max_price"].notna()
    if mask.sum() == 0:
        return True
    return (df.loc[mask, "min_price"] <= df.loc[mask, "max_price"]).all()


def check_modal_in_range(df: pd.DataFrame) -> bool:
    """Check that modal_price is between min and max when all present."""
    mask = (
        df["min_price"].notna() & 
        df["max_price"].notna() & 
        df["modal_price"].notna()
    )
    if mask.sum() == 0:
        return True
    subset = df.loc[mask]
    return (
        (subset["modal_price"] >= subset["min_price"]) &
        (subset["modal_price"] <= subset["max_price"])
    ).all()


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

def create_canonical_schema(
    strict_maharashtra: bool = True,
    strict_district: bool = True,
    strict_prices: bool = False,
) -> DataFrameSchema:
    """
    Create Pandera schema for canonical mandi data.
    
    Args:
        strict_maharashtra: Enforce Maharashtra-only
        strict_district: Enforce canonical district names
        strict_prices: Enforce price relationship rules
        
    Returns:
        DataFrameSchema for validation
    """
    # State checks
    state_checks = [Check.str_length(min_value=1)]
    if strict_maharashtra:
        state_checks.append(
            Check(check_is_maharashtra, element_wise=True, error="State must be Maharashtra")
        )
    
    # District checks
    district_checks = [Check.str_length(min_value=1)]
    if strict_district:
        district_checks.append(
            Check(check_canonical_district, element_wise=True, error="District must be canonical (one of 36)")
        )
    
    schema = DataFrameSchema(
        columns={
            "state": Column(
                str,
                checks=state_checks,
                nullable=False,
                coerce=True,
            ),
            "district": Column(
                str,
                checks=district_checks,
                nullable=False,
                coerce=True,
            ),
            "district_raw": Column(
                str,
                nullable=True,
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
                checks=[Check(check_price_non_negative, element_wise=True)] if strict_prices else [],
                nullable=True,
                coerce=True,
            ),
            "max_price": Column(
                float,
                checks=[Check(check_price_non_negative, element_wise=True)] if strict_prices else [],
                nullable=True,
                coerce=True,
            ),
            "modal_price": Column(
                float,
                checks=[Check(check_price_non_negative, element_wise=True)] if strict_prices else [],
                nullable=True,
                coerce=True,
            ),
            "commodity_code": Column(
                float,  # Nullable int as float
                nullable=True,
                coerce=True,
            ),
            "source": Column(
                str,
                checks=[Check(check_valid_source, element_wise=True)],
                nullable=True,
                coerce=True,
            ),
            "ingested_at_utc": Column(
                "datetime64[ns]",
                nullable=True,
                coerce=True,
            ),
        },
        strict=False,  # Allow extra columns
        coerce=True,
    )
    
    # Add dataframe-level checks
    if strict_prices:
        schema = schema.add_check(
            Check(check_min_max_relationship, error="min_price must be <= max_price")
        )
        schema = schema.add_check(
            Check(check_modal_in_range, error="modal_price must be between min and max")
        )
    
    return schema


# Pre-built schema instances
CanonicalMandiSchema = create_canonical_schema(strict_maharashtra=True, strict_district=True)
CanonicalMandiSchemaLenient = create_canonical_schema(strict_maharashtra=True, strict_district=False)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_canonical_mandi(
    df: pd.DataFrame,
    strict: bool = True,
    raise_on_error: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate a DataFrame against the canonical mandi schema.
    
    Args:
        df: DataFrame to validate
        strict: Use strict validation (enforces canonical districts)
        raise_on_error: Raise exception on validation failure
        
    Returns:
        Tuple of (validated DataFrame, list of error messages)
    """
    errors = []
    
    schema = CanonicalMandiSchema if strict else CanonicalMandiSchemaLenient
    
    try:
        validated = schema.validate(df, lazy=True)
        return validated, errors
    except pa.errors.SchemaErrors as e:
        errors = [str(err) for err in e.failure_cases["failure_case"].tolist()]
        if raise_on_error:
            raise
        return df, errors


def check_key_uniqueness(
    df: pd.DataFrame,
    key_columns: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """
    Check for duplicate keys in the DataFrame.
    
    Args:
        df: DataFrame to check
        key_columns: Columns that form the natural key
        
    Returns:
        DataFrame containing duplicate rows (empty if no duplicates)
    """
    key_columns = key_columns or NATURAL_KEY_COLUMNS
    
    # Ensure all key columns exist
    missing = set(key_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing key columns: {missing}")
    
    # Find duplicates
    duplicated_mask = df.duplicated(subset=list(key_columns), keep=False)
    duplicates = df[duplicated_mask].copy()
    
    return duplicates


def get_duplicate_stats(
    df: pd.DataFrame,
    key_columns: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """
    Get statistics about duplicates in the DataFrame.
    
    Args:
        df: DataFrame to analyze
        key_columns: Columns that form the natural key
        
    Returns:
        Dict with duplicate statistics
    """
    key_columns = key_columns or NATURAL_KEY_COLUMNS
    key_list = list(key_columns)
    
    total_rows = len(df)
    
    # Count duplicates
    duplicated_mask = df.duplicated(subset=key_list, keep=False)
    total_duplicated_rows = duplicated_mask.sum()
    
    # Count unique keys that have duplicates
    duplicate_keys = df[duplicated_mask].groupby(key_list).size()
    unique_duplicate_keys = len(duplicate_keys)
    
    # Rows that would remain after dedup
    unique_rows = df.drop_duplicates(subset=key_list).shape[0]
    
    return {
        "total_rows": total_rows,
        "unique_rows": unique_rows,
        "duplicate_rows": total_duplicated_rows,
        "duplicate_keys": unique_duplicate_keys,
        "rows_to_remove": total_rows - unique_rows,
        "dedup_pct": ((total_rows - unique_rows) / total_rows * 100) if total_rows > 0 else 0,
    }


def validate_maharashtra_only(df: pd.DataFrame) -> Tuple[bool, int]:
    """
    Validate that all rows are Maharashtra-only.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Tuple of (is_valid, count of non-MH rows)
    """
    if "state" not in df.columns:
        return False, len(df)
    
    non_mh_mask = ~df["state"].str.lower().str.strip().eq("maharashtra")
    non_mh_count = non_mh_mask.sum()
    
    return non_mh_count == 0, non_mh_count


def validate_canonical_districts(df: pd.DataFrame) -> Tuple[bool, Set[str]]:
    """
    Validate that all districts are in the canonical list.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Tuple of (is_valid, set of non-canonical districts)
    """
    if "district" not in df.columns:
        return False, set()
    
    unique_districts = set(df["district"].dropna().unique())
    canonical_set = set(CANONICAL_DISTRICTS)
    
    non_canonical = unique_districts - canonical_set
    
    return len(non_canonical) == 0, non_canonical


# =============================================================================
# SUMMARY FUNCTIONS
# =============================================================================

def summarize_canonical_mandi(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for canonical mandi dataset.
    
    Args:
        df: Canonical mandi DataFrame
        
    Returns:
        Dict with summary statistics
    """
    summary = {
        "total_rows": len(df),
        "columns": list(df.columns),
    }
    
    # Date range
    if "arrival_date" in df.columns:
        summary["date_min"] = str(df["arrival_date"].min())
        summary["date_max"] = str(df["arrival_date"].max())
        summary["unique_dates"] = df["arrival_date"].nunique()
    
    # Geographic coverage
    if "state" in df.columns:
        summary["unique_states"] = df["state"].nunique()
        summary["states"] = df["state"].unique().tolist()
    
    if "district" in df.columns:
        summary["unique_districts"] = df["district"].nunique()
    
    if "market" in df.columns:
        summary["unique_markets"] = df["market"].nunique()
    
    # Commodity coverage
    if "commodity" in df.columns:
        summary["unique_commodities"] = df["commodity"].nunique()
    
    # Source distribution
    if "source" in df.columns:
        summary["source_distribution"] = df["source"].value_counts().to_dict()
    
    # Price statistics
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            summary[f"{col}_mean"] = df[col].mean()
            summary[f"{col}_median"] = df[col].median()
            summary[f"{col}_null_pct"] = df[col].isna().sum() / len(df) * 100
    
    # Key uniqueness
    dup_stats = get_duplicate_stats(df)
    summary["duplicate_stats"] = dup_stats
    
    return summary


def generate_qc_report(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Mandi Data Quality Report",
) -> str:
    """
    Generate a Markdown QC report for the mandi dataset.
    
    Args:
        df: DataFrame to analyze
        output_path: Optional path to save report
        title: Report title
        
    Returns:
        Markdown report string
    """
    summary = summarize_canonical_mandi(df)
    dup_stats = summary.get("duplicate_stats", {})
    
    # Maharashtra validation
    mh_valid, non_mh_count = validate_maharashtra_only(df)
    
    # District validation
    dist_valid, non_canonical = validate_canonical_districts(df)
    
    lines = [
        f"# {title}",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Rows | {summary['total_rows']:,} |",
        f"| Unique Districts | {summary.get('unique_districts', 'N/A')} |",
        f"| Unique Markets | {summary.get('unique_markets', 'N/A')} |",
        f"| Unique Commodities | {summary.get('unique_commodities', 'N/A')} |",
        f"| Date Range | {summary.get('date_min', 'N/A')} to {summary.get('date_max', 'N/A')} |",
        "",
        "## Deduplication Stats",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Unique Rows (by key) | {dup_stats.get('unique_rows', 'N/A'):,} |",
        f"| Duplicate Rows | {dup_stats.get('duplicate_rows', 'N/A'):,} |",
        f"| Rows to Remove | {dup_stats.get('rows_to_remove', 'N/A'):,} |",
        f"| Dedup Percentage | {dup_stats.get('dedup_pct', 0):.2f}% |",
        "",
        "## Data Quality Checks",
        "",
        f"| Check | Status |",
        f"|-------|--------|",
        f"| Maharashtra Only | {'✅ PASS' if mh_valid else f'❌ FAIL ({non_mh_count} non-MH rows)'} |",
        f"| Canonical Districts | {'✅ PASS' if dist_valid else f'❌ FAIL ({len(non_canonical)} non-canonical)'} |",
        f"| Key Uniqueness | {'✅ PASS' if dup_stats.get('duplicate_rows', 0) == 0 else '⚠️ DUPLICATES EXIST'} |",
        "",
    ]
    
    if not dist_valid:
        lines.extend([
            "### Non-Canonical Districts Found",
            "",
            ", ".join(sorted(non_canonical)),
            "",
        ])
    
    if "source_distribution" in summary:
        lines.extend([
            "## Source Distribution",
            "",
            "| Source | Rows |",
            "|--------|------|",
        ])
        for source, count in summary["source_distribution"].items():
            lines.append(f"| {source} | {count:,} |")
        lines.append("")
    
    lines.extend([
        "## Price Statistics",
        "",
        "| Column | Mean | Median | Null % |",
        "|--------|------|--------|--------|",
    ])
    
    for col in ["min_price", "max_price", "modal_price"]:
        mean_val = summary.get(f"{col}_mean", 0)
        median_val = summary.get(f"{col}_median", 0)
        null_pct = summary.get(f"{col}_null_pct", 0)
        lines.append(f"| {col} | {mean_val:,.2f} | {median_val:,.2f} | {null_pct:.2f}% |")
    
    report = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    
    return report
