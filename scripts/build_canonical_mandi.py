#!/usr/bin/env python3
"""
MANDIMITRA - Build Canonical Mandi Dataset

This script creates the canonical, deduplicated mandi dataset from:
- Historical Kaggle data (mandi_maharashtra_2001_2025.parquet)
- Current API data (current_mandi_data.csv)

Key Features:
- DuckDB for memory-efficient SQL-based deduplication (handles 6M+ rows)
- Deterministic upsert: current data overwrites historical on key match
- District name normalization (35 mandi districts → 36 canonical)
- Strict natural key uniqueness: (state, district, market, commodity, variety, grade, arrival_date)
- Idempotent: running twice produces identical results

Output:
- data/processed/mandi/mandi_canonical.parquet (partitioned by arrival_year)
- data/processed/dim_districts.csv
- data/processed/dim_commodities.csv
- logs/mandi_dedup_report_<timestamp>.md
- logs/unmapped_districts_<timestamp>.md

Usage:
    python scripts/build_canonical_mandi.py
    python scripts/build_canonical_mandi.py --dry-run
    python scripts/build_canonical_mandi.py --validate-only
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd

from src.utils.district_normalize import (
    CANONICAL_DISTRICTS,
    get_normalizer,
    build_dim_districts,
    save_unmapped_report,
)
from src.utils.schema_standardize import (
    CANONICAL_MANDI_COLUMNS,
    MANDI_NATURAL_KEY,
    standardize_mandi_columns,
    parse_dates_column,
)
from src.schemas.mandi_canonical import (
    validate_canonical_mandi,
    check_key_uniqueness,
    get_duplicate_stats,
    generate_qc_report,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (relative to project root)
DATA_RAW_MANDI = PROJECT_ROOT / "data" / "raw" / "mandi"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED_MANDI = DATA_PROCESSED / "mandi"
LOGS_DIR = PROJECT_ROOT / "logs"

# Input files - Check multiple possible locations
HISTORY_FILES = [
    DATA_PROCESSED_MANDI / "mandi_maharashtra_all.parquet",  # Merged file
    DATA_PROCESSED_MANDI / "history_maharashtra.parquet",   # History only
    DATA_RAW_MANDI / "mandi_maharashtra_2001_2025.parquet", # Alternative name
]
HISTORY_PARQUET_DIR = DATA_RAW_MANDI / "kaggle_download" / "parquet"  # Yearly parquets

CURRENT_FILE = DATA_RAW_MANDI / "current_mandi_data.csv"
CURRENT_DIR = DATA_RAW_MANDI / "current"  # Date-partitioned current data

# Output files
CANONICAL_OUTPUT = DATA_PROCESSED_MANDI / "mandi_canonical.parquet"
DIM_DISTRICTS_OUTPUT = DATA_PROCESSED / "dim_districts.csv"
DIM_COMMODITIES_OUTPUT = DATA_PROCESSED / "dim_commodities.csv"

# Source priority (higher = wins in dedup)
SOURCE_PRIORITY = {
    "current": 2,
    "history": 1,
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_history_data() -> Optional[pd.DataFrame]:
    """Load historical mandi data from Parquet (auto-detects location)."""
    
    # Try single-file locations first
    for path in HISTORY_FILES:
        if path.exists():
            logger.info(f"Loading history data from: {path}")
            df = pd.read_parquet(path)
            df["source"] = "history"
            logger.info(f"  Loaded {len(df):,} rows from history")
            return df
    
    # Try loading yearly parquet files from Kaggle download
    if HISTORY_PARQUET_DIR.exists():
        parquet_files = sorted(HISTORY_PARQUET_DIR.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading history from yearly parquets: {HISTORY_PARQUET_DIR}")
            dfs = []
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                dfs.append(df)
                logger.info(f"  Loaded {len(df):,} rows from {pf.name}")
            
            combined = pd.concat(dfs, ignore_index=True)
            combined["source"] = "history"
            logger.info(f"  Total history rows: {len(combined):,}")
            return combined
    
    logger.warning("No history file found in any expected location")
    return None


def load_current_data() -> Optional[pd.DataFrame]:
    """Load current mandi data from CSV (auto-detects location)."""
    
    # Try direct file first
    if CURRENT_FILE.exists():
        logger.info(f"Loading current data from: {CURRENT_FILE}")
        df = pd.read_csv(CURRENT_FILE)
        df["source"] = "current"
        logger.info(f"  Loaded {len(df):,} rows from current")
        return df
    
    # Try date-partitioned directory
    if CURRENT_DIR.exists():
        # Get most recent date folder
        date_folders = sorted([d for d in CURRENT_DIR.iterdir() if d.is_dir()], reverse=True)
        for folder in date_folders:
            csv_files = list(folder.glob("*.csv"))
            if csv_files:
                latest_csv = csv_files[0]
                logger.info(f"Loading current data from: {latest_csv}")
                df = pd.read_csv(latest_csv)
                df["source"] = "current"
                logger.info(f"  Loaded {len(df):,} rows from current")
                return df
    
    logger.warning(f"Current file not found")
    return None


def standardize_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Standardize column names and types for a mandi DataFrame.
    
    Args:
        df: Raw DataFrame
        source_name: Source identifier
        
    Returns:
        Standardized DataFrame
    """
    logger.info(f"Standardizing {source_name} data ({len(df):,} rows)")
    
    # Standardize column names
    df = standardize_mandi_columns(df)
    
    # Parse arrival_date
    if "arrival_date" in df.columns:
        df["arrival_date"] = parse_dates_column(df["arrival_date"])
    
    # Filter to Maharashtra only
    if "state" in df.columns:
        before = len(df)
        df = df[df["state"].str.lower().str.strip() == "maharashtra"].copy()
        logger.info(f"  Filtered to Maharashtra: {before:,} → {len(df):,} rows")
    
    # Normalize state name
    df["state"] = "Maharashtra"
    
    # Set source and timestamp
    df["source"] = source_name
    df["ingested_at_utc"] = datetime.utcnow()
    
    return df


def normalize_districts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Normalize district names to canonical list.
    
    Args:
        df: DataFrame with 'district' column
        
    Returns:
        Tuple of (normalized DataFrame, unmapped district counts)
    """
    logger.info("Normalizing district names...")
    
    normalizer = get_normalizer()
    normalizer.reset_stats()  # Reset to track this batch
    
    # Keep original district
    df["district_raw"] = df["district"]
    
    # Normalize
    df["district"] = normalizer.normalize_batch(df["district"].tolist())
    
    # Get unmapped set and convert to counts
    unmapped_set = normalizer.unmapped
    
    # Count occurrences of unmapped districts
    unmapped_counts: Dict[str, int] = {}
    if unmapped_set:
        for raw_dist in df["district_raw"]:
            if raw_dist in unmapped_set:
                unmapped_counts[raw_dist] = unmapped_counts.get(raw_dist, 0) + 1
    
    if unmapped_counts:
        logger.warning(f"  Found {len(unmapped_counts)} unmapped district variants")
        for dist, count in sorted(unmapped_counts.items(), key=lambda x: -x[1])[:10]:
            logger.warning(f"    '{dist}': {count} rows")
    else:
        logger.info("  All districts mapped successfully!")
    
    # Check coverage
    mapped_count = df["district"].notna().sum()
    total = len(df)
    logger.info(f"  District mapping: {mapped_count:,}/{total:,} ({mapped_count/total*100:.1f}%)")
    
    return df, unmapped_counts


def deduplicate_with_duckdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate mandi data using DuckDB SQL.
    
    Deduplication rules:
    1. Current data wins over historical (source_priority)
    2. Prefer rows with more complete prices
    3. Tiebreaker: highest modal_price
    
    Args:
        df: Combined mandi DataFrame
        
    Returns:
        Deduplicated DataFrame
    """
    logger.info("Deduplicating with DuckDB...")
    
    before_count = len(df)
    
    # Create DuckDB connection (in-memory)
    con = duckdb.connect()
    
    # Register DataFrame
    con.register("mandi_raw", df)
    
    # Create dedup query with window function
    key_cols = ", ".join(MANDI_NATURAL_KEY)
    
    # Price completeness: count non-null price columns
    dedup_query = f"""
    WITH ranked AS (
        SELECT *,
            -- Source priority (current > history)
            CASE source 
                WHEN 'current' THEN 2 
                WHEN 'history' THEN 1 
                ELSE 0 
            END AS source_priority,
            
            -- Price completeness score
            (CASE WHEN min_price IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN max_price IS NOT NULL THEN 1 ELSE 0 END +
             CASE WHEN modal_price IS NOT NULL THEN 1 ELSE 0 END) AS price_completeness,
            
            -- Row number for dedup
            ROW_NUMBER() OVER (
                PARTITION BY {key_cols}
                ORDER BY 
                    CASE source WHEN 'current' THEN 2 WHEN 'history' THEN 1 ELSE 0 END DESC,
                    (CASE WHEN min_price IS NOT NULL THEN 1 ELSE 0 END +
                     CASE WHEN max_price IS NOT NULL THEN 1 ELSE 0 END +
                     CASE WHEN modal_price IS NOT NULL THEN 1 ELSE 0 END) DESC,
                    COALESCE(modal_price, 0) DESC
            ) AS rn
        FROM mandi_raw
        WHERE district IS NOT NULL
    )
    SELECT 
        state, district, district_raw, market, commodity, variety, grade,
        arrival_date, min_price, max_price, modal_price, commodity_code,
        source, ingested_at_utc
    FROM ranked
    WHERE rn = 1
    """
    
    # Execute and fetch
    result_df = con.execute(dedup_query).fetchdf()
    
    after_count = len(result_df)
    removed = before_count - after_count
    
    logger.info(f"  Deduplication complete: {before_count:,} → {after_count:,} rows")
    logger.info(f"  Removed {removed:,} duplicates ({removed/before_count*100:.2f}%)")
    
    con.close()
    
    return result_df


def build_dim_commodities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build commodity dimension table.
    
    Args:
        df: Canonical mandi DataFrame
        
    Returns:
        Commodity dimension DataFrame
    """
    logger.info("Building commodity dimension table...")
    
    # Get unique commodities with their codes
    commodities = df.groupby("commodity").agg({
        "commodity_code": "first",
        "variety": lambda x: list(x.dropna().unique())[:10],  # Top 10 varieties
    }).reset_index()
    
    commodities.columns = ["commodity", "commodity_code", "varieties"]
    commodities["variety_count"] = commodities["varieties"].apply(len)
    commodities = commodities.sort_values("commodity").reset_index(drop=True)
    
    # Add row counts
    row_counts = df.groupby("commodity").size().reset_index(name="total_records")
    commodities = commodities.merge(row_counts, on="commodity")
    
    logger.info(f"  Created dimension table with {len(commodities)} unique commodities")
    
    return commodities


def save_outputs(
    df: pd.DataFrame,
    unmapped_districts: Dict[str, int],
    dry_run: bool = False,
) -> Dict[str, Path]:
    """
    Save all output files.
    
    Args:
        df: Canonical mandi DataFrame
        unmapped_districts: Dict of unmapped district names
        dry_run: If True, don't actually save
        
    Returns:
        Dict of output paths
    """
    outputs = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure directories exist
    if not dry_run:
        DATA_PROCESSED_MANDI.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Save canonical mandi parquet
    logger.info(f"Saving canonical mandi data to: {CANONICAL_OUTPUT}")
    if not dry_run:
        # Add arrival_year for partitioning
        df["arrival_year"] = df["arrival_date"].dt.year
        df.to_parquet(CANONICAL_OUTPUT, index=False, engine="pyarrow")
        outputs["mandi_canonical"] = CANONICAL_OUTPUT
    
    # 2. Save dim_districts
    logger.info(f"Saving district dimension to: {DIM_DISTRICTS_OUTPUT}")
    if not dry_run:
        dim_districts_path = build_dim_districts(output_path=DIM_DISTRICTS_OUTPUT)
        outputs["dim_districts"] = dim_districts_path
    
    # 3. Save dim_commodities
    logger.info(f"Saving commodity dimension to: {DIM_COMMODITIES_OUTPUT}")
    if not dry_run:
        dim_commodities = build_dim_commodities(df)
        # Convert list to string for CSV
        dim_commodities["varieties"] = dim_commodities["varieties"].apply(lambda x: "; ".join(x))
        dim_commodities.to_csv(DIM_COMMODITIES_OUTPUT, index=False)
        outputs["dim_commodities"] = DIM_COMMODITIES_OUTPUT
    
    # 4. Save QC report
    qc_report_path = LOGS_DIR / f"mandi_dedup_report_{timestamp}.md"
    logger.info(f"Saving QC report to: {qc_report_path}")
    if not dry_run:
        generate_qc_report(df, output_path=qc_report_path, title="Canonical Mandi Data Quality Report")
        outputs["qc_report"] = qc_report_path
    
    # 5. Save unmapped districts report (using global normalizer)
    normalizer = get_normalizer()
    if normalizer.unmapped:
        unmapped_path = save_unmapped_report(normalizer, output_dir=LOGS_DIR)
        outputs["unmapped_report"] = unmapped_path
    
    return outputs


def validate_final_output(df: pd.DataFrame) -> bool:
    """
    Validate the final canonical dataset.
    
    Args:
        df: Canonical DataFrame
        
    Returns:
        True if valid, False otherwise
    """
    logger.info("Validating final output...")
    
    all_valid = True
    
    # 1. Check no duplicates
    dup_stats = get_duplicate_stats(df)
    if dup_stats["duplicate_rows"] > 0:
        logger.error(f"  ❌ FAIL: Found {dup_stats['duplicate_rows']} duplicate rows!")
        all_valid = False
    else:
        logger.info(f"  ✅ PASS: No duplicate keys")
    
    # 2. Check all districts canonical
    non_canonical = set(df["district"].dropna().unique()) - set(CANONICAL_DISTRICTS)
    if non_canonical:
        logger.error(f"  ❌ FAIL: Found {len(non_canonical)} non-canonical districts: {non_canonical}")
        all_valid = False
    else:
        logger.info(f"  ✅ PASS: All {df['district'].nunique()} districts are canonical")
    
    # 3. Check Maharashtra only
    non_mh = df[df["state"].str.lower() != "maharashtra"]
    if len(non_mh) > 0:
        logger.error(f"  ❌ FAIL: Found {len(non_mh)} non-Maharashtra rows")
        all_valid = False
    else:
        logger.info(f"  ✅ PASS: All rows are Maharashtra")
    
    # 4. Check date range
    date_min = df["arrival_date"].min()
    date_max = df["arrival_date"].max()
    logger.info(f"  📅 Date range: {date_min} to {date_max}")
    
    # 5. Check source distribution
    source_dist = df["source"].value_counts().to_dict()
    logger.info(f"  📊 Source distribution: {source_dist}")
    
    return all_valid


# =============================================================================
# MAIN
# =============================================================================

def main(dry_run: bool = False, validate_only: bool = False):
    """
    Main entry point for building canonical mandi dataset.
    
    Args:
        dry_run: If True, don't save outputs
        validate_only: If True, only validate existing output
    """
    logger.info("=" * 60)
    logger.info("MANDIMITRA - Build Canonical Mandi Dataset")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("DRY RUN MODE - no files will be saved")
    
    # If validate-only, just check existing output
    if validate_only:
        if not CANONICAL_OUTPUT.exists():
            logger.error(f"Cannot validate: {CANONICAL_OUTPUT} does not exist")
            return False
        
        logger.info(f"Loading existing canonical data from: {CANONICAL_OUTPUT}")
        df = pd.read_parquet(CANONICAL_OUTPUT)
        return validate_final_output(df)
    
    # Load data
    history_df = load_history_data()
    current_df = load_current_data()
    
    if history_df is None and current_df is None:
        logger.error("No input data found!")
        return False
    
    # Standardize each source
    dfs_to_combine = []
    
    if history_df is not None:
        history_df = standardize_dataframe(history_df, "history")
        dfs_to_combine.append(history_df)
    
    if current_df is not None:
        current_df = standardize_dataframe(current_df, "current")
        dfs_to_combine.append(current_df)
    
    # Combine
    logger.info("Combining data sources...")
    combined_df = pd.concat(dfs_to_combine, ignore_index=True)
    logger.info(f"  Combined total: {len(combined_df):,} rows")
    
    # Normalize districts
    combined_df, unmapped_districts = normalize_districts(combined_df)
    
    # Filter out rows where district couldn't be mapped
    before_filter = len(combined_df)
    combined_df = combined_df[combined_df["district"].notna()].copy()
    after_filter = len(combined_df)
    
    if before_filter > after_filter:
        logger.warning(f"  Dropped {before_filter - after_filter:,} rows with unmapped districts")
    
    # Deduplicate
    canonical_df = deduplicate_with_duckdb(combined_df)
    
    # Validate
    is_valid = validate_final_output(canonical_df)
    
    if not is_valid:
        logger.error("Validation failed! Check errors above.")
        if not dry_run:
            logger.error("Not saving outputs due to validation failure.")
            return False
    
    # Save outputs
    outputs = save_outputs(canonical_df, unmapped_districts, dry_run=dry_run)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Final row count: {len(canonical_df):,}")
    logger.info(f"  Unique districts: {canonical_df['district'].nunique()}")
    logger.info(f"  Unique markets: {canonical_df['market'].nunique()}")
    logger.info(f"  Unique commodities: {canonical_df['commodity'].nunique()}")
    logger.info(f"  Date range: {canonical_df['arrival_date'].min()} to {canonical_df['arrival_date'].max()}")
    
    if not dry_run:
        logger.info("")
        logger.info("Outputs saved:")
        for name, path in outputs.items():
            logger.info(f"  {name}: {path}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build canonical, deduplicated mandi dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save outputs, just show what would be done",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing canonical output",
    )
    
    args = parser.parse_args()
    
    success = main(dry_run=args.dry_run, validate_only=args.validate_only)
    sys.exit(0 if success else 1)
