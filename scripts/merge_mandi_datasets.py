#!/usr/bin/env python3
"""
MANDIMITRA - Merge Mandi Datasets (Historical + Current)

Merges historical and current mandi data into a unified training dataset.
Uses deterministic upsert: current data overwrites historical for same keys.

Deduplication key: [state, district, market, commodity, variety, grade, arrival_date]

Usage:
    python scripts/merge_mandi_datasets.py --merge
    python scripts/merge_mandi_datasets.py --merge --historical-path data/processed/mandi/history.parquet
    python scripts/merge_mandi_datasets.py --merge --current-dir data/raw/mandi/current
    python scripts/merge_mandi_datasets.py --help

Output:
    data/processed/mandi/mandi_maharashtra_all.parquet  # Unified dataset

⚠️  HARD CONSTRAINT: Maharashtra-only data.
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import ensure_directory, load_config, save_receipt
from src.utils.logging_utils import setup_logger, get_utc_timestamp_safe
from src.utils.maharashtra import MAHARASHTRA_STATE_NAME, is_maharashtra_state
from src.utils.audit import AuditLogger


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge historical and current mandi data (Maharashtra-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MERGE STRATEGY:
  - Historical data provides the base training dataset
  - Current data is "upserted" (insert or update) into historical
  - Dedup key: [state, district, market, commodity, variety, grade, arrival_date]
  - For same key, current data wins (overwrites historical)

BACKUP:
  - Before merging, creates backup of existing merged file
  - Keeps up to --keep-backups previous versions

Examples:
    # Merge with default paths
    python scripts/merge_mandi_datasets.py --merge
    
    # Specify historical file
    python scripts/merge_mandi_datasets.py --merge --historical-path data/custom/history.parquet
    
    # Merge specific current date only
    python scripts/merge_mandi_datasets.py --merge --current-date 2026-02-05
    
    # Dry run (show what would happen)
    python scripts/merge_mandi_datasets.py --merge --dry-run
        """,
    )
    
    # Actions
    parser.add_argument(
        "--merge",
        action="store_true",
        required=True,
        help="Merge historical and current datasets",
    )
    
    # Input paths
    parser.add_argument(
        "--historical-path",
        type=str,
        default=None,
        help="Path to historical dataset (parquet/csv)",
    )
    parser.add_argument(
        "--current-dir",
        type=str,
        default=None,
        help="Directory containing current date-partitioned data",
    )
    parser.add_argument(
        "--current-date",
        type=str,
        default=None,
        help="Specific current date to merge (YYYY-MM-DD). Default: all available",
    )
    
    # Output
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output merged file path",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    
    # Backup
    parser.add_argument(
        "--keep-backups",
        type=int,
        default=3,
        help="Number of backup versions to keep (default: 3)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation",
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources configuration",
    )
    
    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Strict mode: fail if non-Maharashtra data found",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_historical(path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Load historical dataset.
    
    Args:
        path: Path to historical file
        logger: Logger instance
        
    Returns:
        DataFrame or None if not found
    """
    if not path.exists():
        logger.warning(f"Historical file not found: {path}")
        return None
    
    logger.info(f"Loading historical: {path}")
    
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    logger.info(f"  Loaded {len(df):,} historical rows")
    return df


def load_current(
    current_dir: Path,
    specific_date: Optional[str],
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """
    Load current data from date-partitioned directory.
    
    Args:
        current_dir: Directory containing date folders
        specific_date: If provided, load only this date
        logger: Logger instance
        
    Returns:
        DataFrame or None if no data found
    """
    if not current_dir.exists():
        logger.warning(f"Current directory not found: {current_dir}")
        return None
    
    # Find date directories
    if specific_date:
        date_dirs = [current_dir / specific_date]
        date_dirs = [d for d in date_dirs if d.exists()]
    else:
        date_dirs = sorted(current_dir.glob("????-??-??"))
    
    if not date_dirs:
        logger.warning(f"No current data found in {current_dir}")
        return None
    
    logger.info(f"Found {len(date_dirs)} current date partitions")
    
    # Load all current CSVs
    dfs = []
    for date_dir in date_dirs:
        csv_file = date_dir / "mandi_current.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df["_source_date"] = date_dir.name  # Track source
            dfs.append(df)
            logger.info(f"  {date_dir.name}: {len(df):,} rows")
    
    if not dfs:
        logger.warning("No CSV files found in current directories")
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Total current: {len(combined):,} rows")
    
    return combined


# =============================================================================
# MERGE LOGIC
# =============================================================================

DEDUP_KEYS = ["state", "district", "market", "commodity", "variety", "grade", "arrival_date"]


def merge_datasets(
    historical: Optional[pd.DataFrame],
    current: Optional[pd.DataFrame],
    strict: bool,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Merge historical and current datasets with upsert logic.
    
    Current data wins for duplicate keys.
    
    Args:
        historical: Historical DataFrame (may be None)
        current: Current DataFrame (may be None)
        strict: Strict Maharashtra validation
        logger: Logger instance
        
    Returns:
        (merged_df, stats_dict)
    """
    stats = {
        "historical_rows": len(historical) if historical is not None else 0,
        "current_rows": len(current) if current is not None else 0,
        "merged_rows": 0,
        "new_from_current": 0,
        "updated_from_current": 0,
        "duplicates_removed": 0,
        "non_mh_dropped": 0,
    }
    
    # Handle empty cases
    if historical is None and current is None:
        logger.error("Both historical and current are empty!")
        return pd.DataFrame(), stats
    
    if historical is None:
        logger.info("No historical data, using current only")
        merged = current.copy()
    elif current is None:
        logger.info("No current data, using historical only")
        merged = historical.copy()
    else:
        # Actual merge
        logger.info("Merging historical + current (current wins for duplicates)")
        
        # Ensure date columns are compatible
        if "arrival_date" in historical.columns:
            historical["arrival_date"] = pd.to_datetime(historical["arrival_date"], errors="coerce")
        if "arrival_date" in current.columns:
            current["arrival_date"] = pd.to_datetime(current["arrival_date"], errors="coerce")
        
        # Add source markers
        historical = historical.copy()
        current = current.copy()
        historical["_source"] = "historical"
        current["_source"] = "current"
        
        # Concatenate
        merged = pd.concat([historical, current], ignore_index=True)
        
        # Find valid dedup columns
        dedup_cols = [c for c in DEDUP_KEYS if c in merged.columns]
        
        # Sort so current comes last (will be kept in drop_duplicates)
        merged = merged.sort_values("_source", ascending=True)  # historical first
        
        rows_before = len(merged)
        merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
        
        stats["duplicates_removed"] = rows_before - len(merged)
        
        # Count updates vs new
        if "_source" in merged.columns:
            current_count = (merged["_source"] == "current").sum()
            # New = current rows that weren't in historical
            # This is approximate - we'd need set operations for exact count
            stats["new_from_current"] = current_count
            
            # Remove source column
            merged = merged.drop(columns=["_source"], errors="ignore")
    
    # Remove tracking columns
    merged = merged.drop(columns=["_source_date"], errors="ignore")
    
    # Validate Maharashtra-only
    if "state" in merged.columns:
        non_mh_mask = ~merged["state"].apply(is_maharashtra_state)
        non_mh_count = non_mh_mask.sum()
        
        if non_mh_count > 0:
            non_mh_samples = merged.loc[non_mh_mask, "state"].value_counts().head(5).to_dict()
            stats["non_mh_dropped"] = non_mh_count
            stats["non_mh_samples"] = non_mh_samples
            
            if strict:
                logger.warning(
                    f"Dropping {non_mh_count} non-Maharashtra rows. "
                    f"Samples: {non_mh_samples}"
                )
                merged = merged[~non_mh_mask].copy()
    
    stats["merged_rows"] = len(merged)
    
    return merged, stats


# =============================================================================
# BACKUP
# =============================================================================

def create_backup(
    output_path: Path,
    backup_dir: Path,
    keep_backups: int,
    logger: logging.Logger,
) -> Optional[Path]:
    """
    Create backup of existing output file.
    
    Args:
        output_path: Path to file to backup
        backup_dir: Directory for backups
        keep_backups: Number of backups to keep
        logger: Logger instance
        
    Returns:
        Path to backup file, or None
    """
    if not output_path.exists():
        return None
    
    ensure_directory(backup_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
    backup_path = backup_dir / backup_name
    
    logger.info(f"Creating backup: {backup_path}")
    shutil.copy2(output_path, backup_path)
    
    # Clean old backups
    existing_backups = sorted(backup_dir.glob(f"{output_path.stem}_*{output_path.suffix}"))
    if len(existing_backups) > keep_backups:
        for old_backup in existing_backups[:-keep_backups]:
            logger.info(f"Removing old backup: {old_backup}")
            old_backup.unlink()
    
    return backup_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "merge_mandi.log"
    logger = setup_logger("merge_mandi", log_file, level=log_level)
    
    timestamp = get_utc_timestamp_safe()
    audit = AuditLogger("mandi_merge", PROJECT_ROOT / "logs", timestamp)
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Mandi Dataset Merger")
    logger.info("=" * 70)
    
    try:
        # Load config
        config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            config = load_config(config_path)
            merged_config = config.get("mandi", {}).get("merged", {})
        else:
            merged_config = {}
            logger.warning(f"Config not found: {config_path}")
        
        # Determine paths
        historical_path = args.historical_path
        if not historical_path:
            historical_path = config.get("mandi", {}).get("historical", {}).get("processing", {}).get("output_path")
        if not historical_path:
            historical_path = "data/processed/mandi/history_maharashtra.parquet"
        historical_path = PROJECT_ROOT / historical_path
        
        current_dir = args.current_dir
        if not current_dir:
            current_dir = config.get("mandi", {}).get("current", {}).get("output_dir")
        if not current_dir:
            current_dir = "data/raw/mandi/current"
        current_dir = PROJECT_ROOT / current_dir
        
        output_path = args.output_path
        if not output_path:
            output_path = merged_config.get("output_path")
        if not output_path:
            ext = ".parquet" if args.output_format == "parquet" else ".csv"
            output_path = f"data/processed/mandi/mandi_maharashtra_all{ext}"
        output_path = PROJECT_ROOT / output_path
        
        backup_dir = PROJECT_ROOT / merged_config.get("backup_path", "data/processed/mandi/backups")
        
        audit.add_section("Configuration", {
            "historical_path": str(historical_path),
            "current_dir": str(current_dir),
            "output_path": str(output_path),
            "strict": args.strict,
        })
        
        # Load datasets
        logger.info("Loading datasets...")
        historical = load_historical(historical_path, logger)
        current = load_current(current_dir, args.current_date, logger)
        
        # Dry run
        if args.dry_run:
            print("\n📋 DRY RUN - Would merge:")
            print(f"   Historical: {len(historical) if historical is not None else 0:,} rows")
            print(f"   Current: {len(current) if current is not None else 0:,} rows")
            print(f"   Output: {output_path}")
            return 0
        
        # Merge
        logger.info("Merging datasets...")
        merged, stats = merge_datasets(historical, current, args.strict, logger)
        
        audit.add_section("Merge Results", stats)
        
        if merged.empty:
            logger.error("Merged dataset is empty!")
            print("\n❌ Merge resulted in empty dataset")
            return 1
        
        # Backup existing
        if not args.no_backup and output_path.exists():
            create_backup(output_path, backup_dir, args.keep_backups, logger)
        
        # Save
        ensure_directory(output_path.parent)
        
        logger.info(f"Saving {len(merged):,} rows to {output_path}")
        
        if args.output_format == "parquet":
            merged.to_parquet(output_path, index=False, compression="snappy")
        else:
            merged.to_csv(output_path, index=False)
        
        # Generate summary
        summary = {
            "total_rows": len(merged),
        }
        if "state" in merged.columns:
            summary["states"] = merged["state"].unique().tolist()
        if "district" in merged.columns:
            summary["unique_districts"] = merged["district"].nunique()
        if "market" in merged.columns:
            summary["unique_markets"] = merged["market"].nunique()
        if "commodity" in merged.columns:
            summary["unique_commodities"] = merged["commodity"].nunique()
        if "arrival_date" in merged.columns:
            summary["date_range"] = {
                "min": str(merged["arrival_date"].min()),
                "max": str(merged["arrival_date"].max()),
            }
        
        # Receipt
        receipt = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "historical_rows": stats["historical_rows"],
            "current_rows": stats["current_rows"],
            "merged_rows": stats["merged_rows"],
            "duplicates_removed": stats["duplicates_removed"],
            "non_mh_dropped": stats["non_mh_dropped"],
            "output_path": str(output_path),
            "summary": summary,
        }
        
        receipt_path = output_path.parent / f"merge_receipt_{timestamp}.json"
        save_receipt(receipt_path, receipt)
        
        # Summary
        print(f"\n✅ Mandi Dataset Merge Complete!")
        print(f"   📊 Historical: {stats['historical_rows']:,} rows")
        print(f"   📊 Current: {stats['current_rows']:,} rows")
        print(f"   📊 Merged: {stats['merged_rows']:,} rows")
        print(f"   🔄 Duplicates removed: {stats['duplicates_removed']:,}")
        if stats["non_mh_dropped"]:
            print(f"   ⚠️  Non-MH dropped: {stats['non_mh_dropped']}")
        if summary.get("date_range"):
            dr = summary["date_range"]
            print(f"   📅 Date range: {dr['min']} to {dr['max']}")
        print(f"\n   📁 Output: {output_path}")
        
        audit_path = audit.save()
        print(f"   📋 Audit: {audit_path}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        audit.add_error(str(e))
        audit.save()
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
