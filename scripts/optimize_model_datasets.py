#!/usr/bin/env python3
"""
MANDIMITRA - Optimize Model Datasets

This script addresses data quality issues for ML training:

1. Districts with no mandi data (Mumbai Suburban, Palghar, Sindhudurg)
   - Creates coverage metadata for UI handling
   - Documents which districts lack mandi data

2. Weather coverage gaps (1.1% missing)
   - Implements forward-fill imputation by district
   - Maintains data integrity with imputation flags

3. Creates optimized ML-ready dataset:
   - mandi_weather_optimized.parquet with 100% weather coverage
   - Clear imputation documentation

Usage:
    python scripts/optimize_model_datasets.py
    python scripts/optimize_model_datasets.py --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MODEL = DATA_PROCESSED / "model"
LOGS_DIR = PROJECT_ROOT / "logs"

# Input files
MANDI_CANONICAL = DATA_PROCESSED / "mandi" / "mandi_canonical.parquet"
POWER_WEATHER = DATA_PROCESSED / "weather" / "power_daily_maharashtra.parquet"
DIM_DISTRICTS = DATA_PROCESSED / "dim_districts.csv"
MANDI_WEATHER_INPUT = DATA_MODEL / "mandi_weather_2016plus.parquet"

# Output files
MANDI_WEATHER_OPTIMIZED = DATA_MODEL / "mandi_weather_optimized.parquet"
DISTRICT_COVERAGE = DATA_PROCESSED / "district_coverage.csv"
DATA_QUALITY_REPORT = LOGS_DIR / "data_quality_optimization_report.md"

# Weather columns to impute
WEATHER_COLUMNS = [
    "t2m_max", "t2m_min", "t2m_mean", "humidity",
    "precipitation", "wind_speed", "solar_radiation"
]


# =============================================================================
# DISTRICT COVERAGE ANALYSIS
# =============================================================================

def analyze_district_coverage() -> pd.DataFrame:
    """
    Analyze which districts have mandi data and weather data.
    
    Returns:
        DataFrame with coverage information per district
    """
    logger.info("Analyzing district coverage...")
    
    # Load canonical districts
    dim_districts = pd.read_csv(DIM_DISTRICTS)
    all_districts = set(dim_districts["canonical_district"].tolist())
    
    # Load mandi data
    mandi = pd.read_parquet(MANDI_CANONICAL)
    mandi_districts = set(mandi["district"].unique())
    
    # Load weather data
    weather = pd.read_parquet(POWER_WEATHER)
    weather_districts = set(weather["district"].unique())
    
    # Build coverage DataFrame
    coverage_data = []
    for district in sorted(all_districts):
        has_mandi = district in mandi_districts
        has_weather = district in weather_districts
        
        # Get mandi stats if available
        mandi_rows = 0
        mandi_markets = 0
        mandi_commodities = 0
        mandi_date_min = None
        mandi_date_max = None
        
        if has_mandi:
            district_mandi = mandi[mandi["district"] == district]
            mandi_rows = len(district_mandi)
            mandi_markets = district_mandi["market"].nunique()
            mandi_commodities = district_mandi["commodity"].nunique()
            mandi_date_min = str(district_mandi["arrival_date"].min().date())
            mandi_date_max = str(district_mandi["arrival_date"].max().date())
        
        coverage_data.append({
            "district": district,
            "has_mandi_data": has_mandi,
            "has_weather_data": has_weather,
            "mandi_rows": mandi_rows,
            "mandi_markets": mandi_markets,
            "mandi_commodities": mandi_commodities,
            "mandi_date_min": mandi_date_min,
            "mandi_date_max": mandi_date_max,
            "data_status": "full" if (has_mandi and has_weather) else 
                          "weather_only" if has_weather else
                          "mandi_only" if has_mandi else "none"
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    
    # Summary
    full_coverage = (coverage_df["data_status"] == "full").sum()
    weather_only = (coverage_df["data_status"] == "weather_only").sum()
    
    logger.info(f"  Districts with full coverage: {full_coverage}/36")
    logger.info(f"  Districts with weather only (no mandi): {weather_only}")
    logger.info(f"  Districts without mandi data: {sorted(coverage_df[~coverage_df['has_mandi_data']]['district'].tolist())}")
    
    return coverage_df


# =============================================================================
# WEATHER IMPUTATION
# =============================================================================

def impute_missing_weather(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing weather data using forward-fill by district.
    
    Strategy:
    1. Sort by district, date
    2. Forward-fill within each district (use previous day's weather)
    3. If still missing, backward-fill (for start of series)
    4. If still missing (entire district), use state-wide median
    
    Args:
        df: DataFrame with mandi+weather data
        
    Returns:
        Tuple of (imputed DataFrame, imputation statistics)
    """
    logger.info("Imputing missing weather data...")
    
    # Identify which weather columns exist in dataframe
    weather_cols_present = [c for c in WEATHER_COLUMNS if c in df.columns]
    logger.info(f"  Weather columns found: {weather_cols_present}")
    
    # Check initial weather coverage using has_weather column if present
    if "has_weather" in df.columns:
        has_weather_initial = df["has_weather"].astype(bool)
    else:
        has_weather_initial = df[weather_cols_present].notna().all(axis=1) if weather_cols_present else pd.Series([False] * len(df))
    
    # Track imputation stats
    stats = {
        "total_rows": len(df),
        "rows_with_weather_before": has_weather_initial.sum(),
        "rows_missing_before": len(df) - has_weather_initial.sum(),
        "imputation_method": {},
    }
    
    logger.info(f"  Initial weather coverage: {stats['rows_with_weather_before']:,} / {stats['total_rows']:,} ({stats['rows_with_weather_before']/stats['total_rows']*100:.1f}%)")
    
    # Add imputation flag column (True where weather was originally missing)
    df = df.copy()
    df["weather_imputed"] = (~has_weather_initial).astype(int)
    
    # Sort for proper forward-fill
    df = df.sort_values(["district", "arrival_date"]).reset_index(drop=True)
    
    # Calculate state-wide medians as fallback (before filling, using valid data)
    state_medians = {}
    for col in weather_cols_present:
        valid_vals = df[col].dropna()
        if len(valid_vals) > 0:
            state_medians[col] = valid_vals.median()
        else:
            state_medians[col] = 25.0  # Default fallback (reasonable temp)
    logger.info(f"  State medians calculated: { {k: round(v, 1) for k, v in state_medians.items()} }")
    
    # Impute by district using grouped operations for efficiency
    total_missing_before = sum(df[col].isna().sum() for col in weather_cols_present)
    
    # Forward fill and backward fill within each district
    for col in weather_cols_present:
        # Group by district and forward-fill, then backward-fill
        df[col] = df.groupby("district")[col].ffill()
        df[col] = df.groupby("district")[col].bfill()
        
        # Fill remaining NaNs with state median
        still_missing = df[col].isna().sum()
        if still_missing > 0:
            df[col] = df[col].fillna(state_medians[col])
    
    total_missing_after = sum(df[col].isna().sum() for col in weather_cols_present)
    
    # Update has_weather flag (now should be all True after imputation)
    has_weather_final = df[weather_cols_present].notna().all(axis=1) if weather_cols_present else pd.Series([False] * len(df))
    df["has_weather"] = has_weather_final.astype(int)
    
    stats["rows_with_weather_after"] = has_weather_final.sum()
    stats["rows_imputed_total"] = stats["rows_missing_before"]
    stats["total_missing_values_before"] = total_missing_before
    stats["total_missing_values_after"] = total_missing_after
    stats["imputation_coverage"] = has_weather_final.sum() / len(df) * 100 if len(df) > 0 else 0
    
    logger.info(f"  Before: {stats['rows_with_weather_before']:,} rows with weather ({stats['rows_with_weather_before']/stats['total_rows']*100:.1f}%)")
    logger.info(f"  After:  {stats['rows_with_weather_after']:,} rows with weather ({stats['imputation_coverage']:.1f}%)")
    logger.info(f"  Rows with imputed weather: {stats['rows_imputed_total']:,}")
    logger.info(f"  Missing values: {total_missing_before:,} → {total_missing_after:,}")
    
    return df, stats


# =============================================================================
# OPTIMIZED DATASET BUILDER
# =============================================================================

def build_optimized_dataset(dry_run: bool = False) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Build the optimized ML-ready dataset with imputed weather.
    
    Returns:
        Tuple of (optimized DataFrame, statistics dict)
    """
    logger.info("Building optimized ML dataset...")
    
    if not MANDI_WEATHER_INPUT.exists():
        logger.error(f"Input file not found: {MANDI_WEATHER_INPUT}")
        return None, {}
    
    # Load the joined dataset
    df = pd.read_parquet(MANDI_WEATHER_INPUT)
    logger.info(f"  Loaded {len(df):,} rows")
    
    # Impute missing weather
    df_imputed, impute_stats = impute_missing_weather(df)
    
    # Add derived features useful for ML
    logger.info("Adding derived features...")
    
    # Price features
    df_imputed["price_range"] = df_imputed["max_price"] - df_imputed["min_price"]
    df_imputed["price_volatility"] = (df_imputed["price_range"] / df_imputed["modal_price"] * 100).round(2)
    
    # Lag features (for time series)
    # These would normally be computed per commodity-market group, but we'll add placeholders
    df_imputed["is_weekend"] = df_imputed["arrival_day_of_week"].isin([5, 6]).astype(int)
    df_imputed["is_month_start"] = (df_imputed["arrival_date"].dt.day <= 5).astype(int)
    df_imputed["is_month_end"] = (df_imputed["arrival_date"].dt.day >= 25).astype(int)
    
    # Season features (Maharashtra agricultural seasons)
    def get_season(month):
        if month in [6, 7, 8, 9]:  # Kharif
            return "kharif"
        elif month in [10, 11, 12, 1, 2]:  # Rabi
            return "rabi"
        else:  # Summer/Zaid
            return "summer"
    
    df_imputed["season"] = df_imputed["arrival_month"].apply(get_season)
    
    # Final statistics
    final_stats = {
        **impute_stats,
        "final_rows": len(df_imputed),
        "final_columns": len(df_imputed.columns),
        "unique_districts": df_imputed["district"].nunique(),
        "unique_markets": df_imputed["market"].nunique(),
        "unique_commodities": df_imputed["commodity"].nunique(),
        "date_min": str(df_imputed["arrival_date"].min()),
        "date_max": str(df_imputed["arrival_date"].max()),
    }
    
    # Save
    if not dry_run:
        DATA_MODEL.mkdir(parents=True, exist_ok=True)
        df_imputed.to_parquet(MANDI_WEATHER_OPTIMIZED, index=False)
        logger.info(f"  Saved to: {MANDI_WEATHER_OPTIMIZED}")
    
    return df_imputed, final_stats


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_quality_report(
    coverage_df: pd.DataFrame,
    optimize_stats: Dict,
    output_path: Path,
) -> str:
    """
    Generate comprehensive data quality report.
    """
    timestamp = datetime.now().isoformat()
    
    # Districts without mandi data
    no_mandi = coverage_df[~coverage_df["has_mandi_data"]]["district"].tolist()
    
    lines = [
        "# MANDIMITRA Data Quality Optimization Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report documents data quality optimizations performed to create",
        "ML-ready datasets for price forecasting.",
        "",
        "### Key Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Mandi Rows | {optimize_stats.get('final_rows', 'N/A'):,} |",
        f"| Districts with Mandi Data | {optimize_stats.get('unique_districts', 'N/A')}/36 |",
        f"| Weather Coverage (Before) | {optimize_stats.get('rows_with_weather_before', 0)/optimize_stats.get('total_rows', 1)*100:.1f}% |",
        f"| Weather Coverage (After) | {optimize_stats.get('imputation_coverage', 0):.1f}% |",
        f"| Date Range | {optimize_stats.get('date_min', 'N/A')} to {optimize_stats.get('date_max', 'N/A')} |",
        "",
        "---",
        "",
        "## 1. District Coverage Analysis",
        "",
        "### Districts WITHOUT Mandi Data (3 districts)",
        "",
        "These districts have no reported mandi prices in the dataset:",
        "",
    ]
    
    for district in no_mandi:
        lines.append(f"- **{district}** - No agricultural markets reporting to AGMARKNET")
    
    lines.extend([
        "",
        "**Explanation:**",
        "- **Mumbai Suburban**: Urban area with no agricultural mandis",
        "- **Palghar**: Recently formed district (2014), markets may report under Thane",
        "- **Sindhudurg**: Coastal district with limited agricultural marketing infrastructure",
        "",
        "**UI Recommendation:**",
        "```",
        "Show message: 'Price data not available for this district.'",
        "Optionally show nearby district prices or regional averages.",
        "```",
        "",
        "### Full District Coverage Table",
        "",
        "| District | Mandi Data | Weather | Markets | Commodities | Status |",
        "|----------|------------|---------|---------|-------------|--------|",
    ])
    
    for _, row in coverage_df.iterrows():
        mandi_check = "✅" if row["has_mandi_data"] else "❌"
        weather_check = "✅" if row["has_weather_data"] else "❌"
        status_emoji = "🟢" if row["data_status"] == "full" else "🟡" if row["has_weather_data"] else "🔴"
        lines.append(
            f"| {row['district']} | {mandi_check} | {weather_check} | "
            f"{row['mandi_markets'] or '-'} | {row['mandi_commodities'] or '-'} | {status_emoji} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## 2. Weather Data Imputation",
        "",
        "### Strategy",
        "",
        "Missing weather data was imputed using the following strategy:",
        "",
        "1. **Forward-Fill (Primary)**: Use previous day's weather for same district",
        "2. **Backward-Fill (Secondary)**: For start-of-series gaps",
        "3. **State Median (Fallback)**: For districts with no historical weather",
        "",
        "### Imputation Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Rows | {optimize_stats.get('total_rows', 0):,} |",
        f"| Rows with Weather (Before) | {optimize_stats.get('rows_with_weather_before', 0):,} |",
        f"| Rows Missing Weather (Before) | {optimize_stats.get('rows_missing_before', 0):,} |",
        f"| Total Rows Imputed | {optimize_stats.get('rows_imputed_total', 0):,} |",
        f"| Final Weather Coverage | {optimize_stats.get('imputation_coverage', 0):.2f}% |",
        "",
        "### Imputation Flag",
        "",
        "The `weather_imputed` column indicates which rows have imputed weather:",
        "- `0` = Original weather data",
        "- `1` = Imputed weather data",
        "",
        "For strict ML validation, you can filter: `df[df['weather_imputed'] == 0]`",
        "",
        "---",
        "",
        "## 3. Weather Resolution Note",
        "",
        "Weather data is at **District HQ level** (single point per district).",
        "",
        "**Implications:**",
        "- Acceptable for competition/MVP level forecasting",
        "- Markets far from HQ may have different microclimates",
        "- Future improvement: Use gridded weather data (e.g., ERA5) for better resolution",
        "",
        "**Current Coordinates Source:** `configs/maharashtra_locations.csv`",
        "",
        "---",
        "",
        "## 4. Arrivals/Volume Data",
        "",
        "**Status:** ❌ Not Available",
        "",
        "The AGMARKNET dataset does not include arrival quantities (tonnes).",
        "This is a known limitation of the Kaggle dataset.",
        "",
        "**Impact on Forecasting:**",
        "- Price-only models are still viable",
        "- Volume data would improve supply-demand modeling",
        "- Consider scraping AGMARKNET directly for volume data in future",
        "",
        "**Alternative Approach:**",
        "Use price volatility as a proxy for supply conditions:",
        "- High volatility → supply uncertainty",
        "- Low volatility → stable supply",
        "",
        "---",
        "",
        "## 5. Optimized Dataset Schema",
        "",
        "### mandi_weather_optimized.parquet",
        "",
        "| Column | Type | Description |",
        "|--------|------|-------------|",
        "| state | string | Always 'Maharashtra' |",
        "| district | string | Canonical district name |",
        "| market | string | Mandi name |",
        "| commodity | string | Crop/commodity name |",
        "| variety | string | Variety (nullable) |",
        "| grade | string | Grade (nullable) |",
        "| arrival_date | datetime | Price observation date |",
        "| min_price | float | Minimum price (Rs/quintal) |",
        "| max_price | float | Maximum price (Rs/quintal) |",
        "| modal_price | float | Most common price (Rs/quintal) |",
        "| t2m_max | float | Max temperature (°C) |",
        "| t2m_min | float | Min temperature (°C) |",
        "| t2m_mean | float | Mean temperature (°C) |",
        "| humidity | float | Relative humidity (%) |",
        "| precipitation | float | Rainfall (mm) |",
        "| wind_speed | float | Wind speed (m/s) |",
        "| solar_radiation | float | Solar radiation (MJ/m²/day) |",
        "| has_weather | int | 1=has weather, 0=missing |",
        "| weather_imputed | int | 1=imputed, 0=original |",
        "| price_range | float | max_price - min_price |",
        "| price_volatility | float | (range/modal)*100 |",
        "| is_weekend | int | 1=Sat/Sun |",
        "| is_month_start | int | 1=days 1-5 |",
        "| is_month_end | int | 1=days 25-31 |",
        "| season | string | kharif/rabi/summer |",
        "",
        "---",
        "",
        "## 6. Recommendations for ML Training",
        "",
        "### Data Filtering",
        "",
        "```python",
        "import pandas as pd",
        "",
        "df = pd.read_parquet('data/processed/model/mandi_weather_optimized.parquet')",
        "",
        "# Option 1: Use all data (imputation included)",
        "train_df = df",
        "",
        "# Option 2: Use only rows with original weather (stricter)",
        "train_df = df[df['weather_imputed'] == 0]",
        "",
        "# Option 3: Filter by commodity",
        "onion_df = df[df['commodity'] == 'Onion']",
        "```",
        "",
        "### Recommended Train/Test Split",
        "",
        "```python",
        "# Time-based split (proper for time series)",
        "train = df[df['arrival_date'] < '2025-01-01']",
        "test = df[df['arrival_date'] >= '2025-01-01']",
        "```",
        "",
        "### Handling Missing Districts in Production",
        "",
        "```python",
        "DISTRICTS_NO_DATA = ['Mumbai Suburban', 'Palghar', 'Sindhudurg']",
        "",
        "def get_price_forecast(district, commodity):",
        "    if district in DISTRICTS_NO_DATA:",
        "        return {'status': 'no_data', 'message': 'Price data not available'}",
        "    # ... normal forecast logic",
        "```",
        "",
    ])
    
    report = "\n".join(lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report


# =============================================================================
# MAIN
# =============================================================================

def main(dry_run: bool = False):
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("MANDIMITRA - Optimize Model Datasets")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("DRY RUN MODE - no files will be saved")
    
    # 1. Analyze district coverage
    coverage_df = analyze_district_coverage()
    
    # Save coverage file
    if not dry_run:
        coverage_df.to_csv(DISTRICT_COVERAGE, index=False)
        logger.info(f"Saved district coverage to: {DISTRICT_COVERAGE}")
    
    # 2. Build optimized dataset with weather imputation
    optimized_df, stats = build_optimized_dataset(dry_run=dry_run)
    
    if optimized_df is None:
        logger.error("Failed to build optimized dataset")
        return False
    
    # 3. Generate quality report
    if not dry_run:
        generate_quality_report(coverage_df, stats, DATA_QUALITY_REPORT)
        logger.info(f"Saved quality report to: {DATA_QUALITY_REPORT}")
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Final dataset rows: {stats.get('final_rows', 0):,}")
    logger.info(f"  Weather coverage: {stats.get('imputation_coverage', 0):.2f}%")
    logger.info(f"  Districts with mandi data: {stats.get('unique_districts', 0)}/36")
    
    if not dry_run:
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  {MANDI_WEATHER_OPTIMIZED}")
        logger.info(f"  {DISTRICT_COVERAGE}")
        logger.info(f"  {DATA_QUALITY_REPORT}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize model datasets for ML training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save outputs",
    )
    
    args = parser.parse_args()
    success = main(dry_run=args.dry_run)
    sys.exit(0 if success else 1)
