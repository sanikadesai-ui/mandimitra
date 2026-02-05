#!/usr/bin/env python3
"""
MANDIMITRA - Build Model-Ready Datasets

Creates final join-ready datasets for ML training:

1. mandi_only_2001_2026.parquet
   - Full mandi history, all years
   - No weather join (for long-horizon analysis)

2. mandi_weather_2016plus.parquet
   - Mandi data joined with NASA POWER weather
   - Only 2016+ (where weather data overlaps)
   - Ready for weather-aware ML models

Prerequisites:
    - Run build_canonical_mandi.py first
    - Run process_weather.py first

Usage:
    python scripts/build_model_datasets.py
    python scripts/build_model_datasets.py --dry-run
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import duckdb
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED_MANDI = DATA_PROCESSED / "mandi"
DATA_PROCESSED_WEATHER = DATA_PROCESSED / "weather"
DATA_MODEL = DATA_PROCESSED / "model"
LOGS_DIR = PROJECT_ROOT / "logs"

# Input files
CANONICAL_MANDI = DATA_PROCESSED_MANDI / "mandi_canonical.parquet"
POWER_WEATHER = DATA_PROCESSED_WEATHER / "power_daily_maharashtra.parquet"
FORECAST_WEATHER = DATA_PROCESSED_WEATHER / "forecast_maharashtra.parquet"

# Output files
MANDI_ONLY_OUTPUT = DATA_MODEL / "mandi_only_2001_2026.parquet"
MANDI_WEATHER_OUTPUT = DATA_MODEL / "mandi_weather_2016plus.parquet"

# Weather join starts from this year (when NASA POWER data begins)
WEATHER_START_YEAR = 2016

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET BUILDERS
# =============================================================================

def build_mandi_only(dry_run: bool = False) -> Optional[pd.DataFrame]:
    """
    Build mandi-only dataset (no weather join).
    
    This is the full historical mandi dataset, useful for:
    - Long-term price trend analysis
    - Commodity-only models
    - Seasonal pattern analysis
    
    Args:
        dry_run: If True, don't save
        
    Returns:
        DataFrame or None
    """
    if not CANONICAL_MANDI.exists():
        logger.error(f"Canonical mandi not found: {CANONICAL_MANDI}")
        logger.error("Run build_canonical_mandi.py first!")
        return None
    
    logger.info(f"Building mandi-only dataset from: {CANONICAL_MANDI}")
    
    # Load canonical mandi
    df = pd.read_parquet(CANONICAL_MANDI)
    logger.info(f"  Loaded {len(df):,} rows")
    
    # Add useful derived columns
    df["arrival_year"] = df["arrival_date"].dt.year
    df["arrival_month"] = df["arrival_date"].dt.month
    df["arrival_day_of_week"] = df["arrival_date"].dt.dayofweek
    df["arrival_week_of_year"] = df["arrival_date"].dt.isocalendar().week.astype(int)
    
    # Add price spread
    df["price_spread"] = df["max_price"] - df["min_price"]
    df["price_spread_pct"] = (df["price_spread"] / df["modal_price"] * 100).round(2)
    
    # Sort by date and market
    df = df.sort_values(["arrival_date", "district", "market", "commodity"]).reset_index(drop=True)
    
    # Summary
    logger.info(f"  Final rows: {len(df):,}")
    logger.info(f"  Date range: {df['arrival_date'].min()} to {df['arrival_date'].max()}")
    logger.info(f"  Districts: {df['district'].nunique()}")
    logger.info(f"  Markets: {df['market'].nunique()}")
    logger.info(f"  Commodities: {df['commodity'].nunique()}")
    
    # Save
    if not dry_run:
        DATA_MODEL.mkdir(parents=True, exist_ok=True)
        df.to_parquet(MANDI_ONLY_OUTPUT, index=False)
        logger.info(f"  Saved to: {MANDI_ONLY_OUTPUT}")
    
    return df


def build_mandi_weather_joined(dry_run: bool = False) -> Optional[pd.DataFrame]:
    """
    Build mandi + weather joined dataset for ML training.
    
    Join strategy:
    - Join on (date, district)
    - Weather represents conditions on the day of price observation
    - Only includes data from 2016+ (weather data availability)
    
    Args:
        dry_run: If True, don't save
        
    Returns:
        DataFrame or None
    """
    if not CANONICAL_MANDI.exists():
        logger.error(f"Canonical mandi not found: {CANONICAL_MANDI}")
        return None
    
    if not POWER_WEATHER.exists():
        logger.error(f"POWER weather not found: {POWER_WEATHER}")
        logger.error("Run process_weather.py first!")
        return None
    
    logger.info("Building mandi + weather joined dataset")
    
    # Use DuckDB for efficient join
    con = duckdb.connect()
    
    # Register files directly
    con.execute(f"CREATE TABLE mandi AS SELECT * FROM read_parquet('{CANONICAL_MANDI}')")
    con.execute(f"CREATE TABLE weather AS SELECT * FROM read_parquet('{POWER_WEATHER}')")
    
    # Check data
    mandi_count = con.execute("SELECT COUNT(*) FROM mandi").fetchone()[0]
    weather_count = con.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
    logger.info(f"  Mandi rows: {mandi_count:,}")
    logger.info(f"  Weather rows: {weather_count:,}")
    
    # Filter mandi to weather years and join
    join_query = f"""
    SELECT 
        m.state,
        m.district,
        m.market,
        m.commodity,
        m.variety,
        m.grade,
        m.arrival_date,
        m.min_price,
        m.max_price,
        m.modal_price,
        m.commodity_code,
        m.source AS mandi_source,
        
        -- Derived date columns
        EXTRACT(YEAR FROM m.arrival_date) AS arrival_year,
        EXTRACT(MONTH FROM m.arrival_date) AS arrival_month,
        EXTRACT(DOW FROM m.arrival_date) AS arrival_day_of_week,
        EXTRACT(WEEK FROM m.arrival_date) AS arrival_week_of_year,
        
        -- Price derived columns
        m.max_price - m.min_price AS price_spread,
        
        -- Weather columns
        w.t2m_max,
        w.t2m_min,
        w.t2m_mean,
        w.rh2m AS humidity,
        w.precipitation,
        w.ws2m AS wind_speed,
        w.srad AS solar_radiation,
        
        -- Weather quality flag
        CASE WHEN w.date IS NOT NULL THEN 1 ELSE 0 END AS has_weather
        
    FROM mandi m
    LEFT JOIN weather w 
        ON DATE_TRUNC('day', m.arrival_date) = w.date 
        AND m.district = w.district
    WHERE EXTRACT(YEAR FROM m.arrival_date) >= {WEATHER_START_YEAR}
    ORDER BY m.arrival_date, m.district, m.market, m.commodity
    """
    
    logger.info(f"  Joining mandi with weather (year >= {WEATHER_START_YEAR})...")
    result_df = con.execute(join_query).fetchdf()
    
    con.close()
    
    # Convert date columns
    result_df["arrival_date"] = pd.to_datetime(result_df["arrival_date"])
    
    # Stats
    total_rows = len(result_df)
    with_weather = result_df["has_weather"].sum()
    weather_coverage = with_weather / total_rows * 100 if total_rows > 0 else 0
    
    logger.info(f"  Joined rows: {total_rows:,}")
    logger.info(f"  With weather: {with_weather:,} ({weather_coverage:.1f}%)")
    logger.info(f"  Date range: {result_df['arrival_date'].min()} to {result_df['arrival_date'].max()}")
    
    # Save
    if not dry_run:
        DATA_MODEL.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(MANDI_WEATHER_OUTPUT, index=False)
        logger.info(f"  Saved to: {MANDI_WEATHER_OUTPUT}")
    
    return result_df


def generate_model_report(
    mandi_only_df: Optional[pd.DataFrame],
    mandi_weather_df: Optional[pd.DataFrame],
    output_path: Path,
) -> str:
    """
    Generate summary report for model datasets.
    
    Args:
        mandi_only_df: Mandi-only DataFrame
        mandi_weather_df: Mandi+weather DataFrame
        output_path: Path to save report
        
    Returns:
        Report as string
    """
    lines = [
        "# MANDIMITRA Model Dataset Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
    ]
    
    if mandi_only_df is not None:
        lines.extend([
            "## mandi_only_2001_2026.parquet",
            "",
            "Full mandi history without weather join. Use for:",
            "- Long-term price trend analysis",
            "- Commodity-only forecasting models",
            "- Seasonal pattern discovery",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Rows | {len(mandi_only_df):,} |",
            f"| Date Range | {mandi_only_df['arrival_date'].min()} to {mandi_only_df['arrival_date'].max()} |",
            f"| Unique Districts | {mandi_only_df['district'].nunique()} |",
            f"| Unique Markets | {mandi_only_df['market'].nunique()} |",
            f"| Unique Commodities | {mandi_only_df['commodity'].nunique()} |",
            "",
            "### Columns",
            "",
            "```",
            ", ".join(mandi_only_df.columns.tolist()),
            "```",
            "",
        ])
    
    if mandi_weather_df is not None:
        with_weather = mandi_weather_df["has_weather"].sum() if "has_weather" in mandi_weather_df.columns else 0
        coverage = with_weather / len(mandi_weather_df) * 100 if len(mandi_weather_df) > 0 else 0
        
        lines.extend([
            "## mandi_weather_2016plus.parquet",
            "",
            "Mandi + weather joined dataset. Use for:",
            "- Weather-aware price prediction",
            "- Climate impact analysis",
            "- Feature engineering with weather variables",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Rows | {len(mandi_weather_df):,} |",
            f"| Date Range | {mandi_weather_df['arrival_date'].min()} to {mandi_weather_df['arrival_date'].max()} |",
            f"| Rows with Weather | {with_weather:,} ({coverage:.1f}%) |",
            f"| Unique Districts | {mandi_weather_df['district'].nunique()} |",
            f"| Unique Commodities | {mandi_weather_df['commodity'].nunique()} |",
            "",
            "### Weather Columns",
            "",
            "| Column | Description |",
            "|--------|-------------|",
            "| t2m_max | Temperature at 2m max (°C) |",
            "| t2m_min | Temperature at 2m min (°C) |",
            "| t2m_mean | Temperature at 2m mean (°C) |",
            "| humidity | Relative humidity at 2m (%) |",
            "| precipitation | Daily precipitation (mm) |",
            "| wind_speed | Wind speed at 2m (m/s) |",
            "| solar_radiation | Solar radiation (MJ/m²/day) |",
            "",
            "### Columns",
            "",
            "```",
            ", ".join(mandi_weather_df.columns.tolist()),
            "```",
            "",
        ])
    
    # Join key documentation
    lines.extend([
        "## Join Keys",
        "",
        "### Mandi Natural Key",
        "```",
        "(state, district, market, commodity, variety, grade, arrival_date)",
        "```",
        "",
        "### Weather Join Key",
        "```",
        "(date, district)",
        "```",
        "",
        "## Usage Examples",
        "",
        "### Load datasets",
        "```python",
        "import pandas as pd",
        "",
        "# Mandi only (full history)",
        "mandi_df = pd.read_parquet('data/processed/model/mandi_only_2001_2026.parquet')",
        "",
        "# Mandi + weather (2016+)",
        "weather_df = pd.read_parquet('data/processed/model/mandi_weather_2016plus.parquet')",
        "```",
        "",
        "### Filter by commodity",
        "```python",
        "onion_df = mandi_df[mandi_df['commodity'] == 'Onion']",
        "```",
        "",
        "### Aggregate by district-month",
        "```python",
        "monthly = mandi_df.groupby(['district', 'arrival_year', 'arrival_month']).agg({",
        "    'modal_price': ['mean', 'std', 'count']",
        "}).reset_index()",
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
    """
    Main entry point for building model datasets.
    
    Args:
        dry_run: If True, don't save outputs
    """
    logger.info("=" * 60)
    logger.info("MANDIMITRA - Build Model Datasets")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("DRY RUN MODE - no files will be saved")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build mandi-only dataset
    mandi_only_df = build_mandi_only(dry_run=dry_run)
    
    # Build mandi+weather joined dataset
    mandi_weather_df = build_mandi_weather_joined(dry_run=dry_run)
    
    # Generate report
    if not dry_run and (mandi_only_df is not None or mandi_weather_df is not None):
        report_path = LOGS_DIR / f"model_datasets_report_{timestamp}.md"
        generate_model_report(mandi_only_df, mandi_weather_df, report_path)
        logger.info(f"Report saved to: {report_path}")
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    
    outputs = []
    if mandi_only_df is not None:
        outputs.append(f"  mandi_only_2001_2026: {len(mandi_only_df):,} rows")
    if mandi_weather_df is not None:
        outputs.append(f"  mandi_weather_2016plus: {len(mandi_weather_df):,} rows")
    
    for out in outputs:
        logger.info(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build model-ready datasets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save outputs",
    )
    
    args = parser.parse_args()
    main(dry_run=args.dry_run)
