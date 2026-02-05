#!/usr/bin/env python3
"""
MANDIMITRA - Data Completeness Report Generator

Generates a comprehensive Markdown report of data completeness for:
- Mandi data (historical + current + merged)
- Weather data (NASA POWER + Open-Meteo)

Checks:
- Date ranges and coverage gaps
- District coverage vs expected 36
- Record counts and statistics
- Missing data identification

Usage:
    python scripts/generate_completeness_report.py
    python scripts/generate_completeness_report.py --output logs/completeness_report.md
    python scripts/generate_completeness_report.py --help

Output:
    logs/data_completeness_<timestamp>.md

⚠️  HARD CONSTRAINT: Maharashtra-only verification.
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import load_config
from src.utils.logging_utils import setup_logger, get_utc_timestamp_safe
from src.utils.maharashtra import MAHARASHTRA_DISTRICTS, MAHARASHTRA_STATE_NAME


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate data completeness report for MANDIMITRA",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: logs/data_completeness_<timestamp>.md)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources configuration",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


# =============================================================================
# DATA INSPECTION
# =============================================================================

def inspect_mandi_historical(path: Path) -> Dict[str, Any]:
    """Inspect historical mandi dataset."""
    result = {
        "exists": False,
        "rows": 0,
        "date_range": None,
        "unique_districts": 0,
        "unique_markets": 0,
        "unique_commodities": 0,
        "states": [],
        "non_mh": 0,
        "error": None,
    }
    
    if not path.exists():
        result["error"] = f"File not found: {path}"
        return result
    
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        
        result["exists"] = True
        result["rows"] = len(df)
        
        if "state" in df.columns:
            result["states"] = df["state"].unique().tolist()
            non_mh = df[df["state"].str.lower() != "maharashtra"]
            result["non_mh"] = len(non_mh)
        
        if "district" in df.columns:
            result["unique_districts"] = df["district"].nunique()
            result["districts"] = sorted(df["district"].unique().tolist())
        
        if "market" in df.columns:
            result["unique_markets"] = df["market"].nunique()
        
        if "commodity" in df.columns:
            result["unique_commodities"] = df["commodity"].nunique()
        
        if "arrival_date" in df.columns:
            df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
            result["date_range"] = {
                "min": str(df["arrival_date"].min()),
                "max": str(df["arrival_date"].max()),
            }
            result["unique_dates"] = df["arrival_date"].nunique()
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def inspect_mandi_current(directory: Path) -> Dict[str, Any]:
    """Inspect current mandi data directory."""
    result = {
        "exists": False,
        "partitions": 0,
        "total_rows": 0,
        "date_partitions": [],
        "rows_per_date": {},
        "error": None,
    }
    
    if not directory.exists():
        result["error"] = f"Directory not found: {directory}"
        return result
    
    try:
        date_dirs = sorted(directory.glob("????-??-??"))
        result["exists"] = len(date_dirs) > 0
        result["partitions"] = len(date_dirs)
        
        for date_dir in date_dirs:
            csv_file = date_dir / "mandi_current.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                rows = len(df)
                result["date_partitions"].append(date_dir.name)
                result["rows_per_date"][date_dir.name] = rows
                result["total_rows"] += rows
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def inspect_mandi_merged(path: Path) -> Dict[str, Any]:
    """Inspect merged mandi dataset."""
    return inspect_mandi_historical(path)  # Same structure


def inspect_weather_power(directory: Path) -> Dict[str, Any]:
    """Inspect NASA POWER weather data directory."""
    result = {
        "exists": False,
        "districts_with_data": 0,
        "total_rows": 0,
        "districts": [],
        "date_ranges": {},
        "missing_districts": [],
        "error": None,
    }
    
    if not directory.exists():
        result["error"] = f"Directory not found: {directory}"
        return result
    
    try:
        expected = set(d.lower() for d in MAHARASHTRA_DISTRICTS)
        found = set()
        
        district_dirs = list(directory.glob("*"))
        
        for district_dir in district_dirs:
            if not district_dir.is_dir():
                continue
            
            csv_files = list(district_dir.glob("power_daily_*.csv"))
            if csv_files:
                district_name = district_dir.name
                found.add(district_name.lower())
                result["districts"].append(district_name)
                
                # Read most recent file
                latest = max(csv_files, key=lambda p: p.stat().st_mtime)
                df = pd.read_csv(latest)
                result["total_rows"] += len(df)
                
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    result["date_ranges"][district_name] = {
                        "min": str(df["date"].min()),
                        "max": str(df["date"].max()),
                        "days": df["date"].nunique(),
                    }
        
        result["exists"] = len(found) > 0
        result["districts_with_data"] = len(found)
        
        # Find missing (case-insensitive comparison)
        for expected_d in MAHARASHTRA_DISTRICTS:
            if expected_d.lower() not in found:
                result["missing_districts"].append(expected_d)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def inspect_weather_openmeteo(directory: Path) -> Dict[str, Any]:
    """Inspect Open-Meteo forecast data directory."""
    result = {
        "exists": False,
        "districts_with_data": 0,
        "total_rows": 0,
        "districts": [],
        "latest_forecasts": {},
        "missing_districts": [],
        "error": None,
    }
    
    if not directory.exists():
        result["error"] = f"Directory not found: {directory}"
        return result
    
    try:
        expected = set(d.lower() for d in MAHARASHTRA_DISTRICTS)
        found = set()
        
        district_dirs = list(directory.glob("*"))
        
        for district_dir in district_dirs:
            if not district_dir.is_dir():
                continue
            
            csv_files = list(district_dir.glob("forecast_*.csv"))
            if csv_files:
                district_name = district_dir.name
                found.add(district_name.lower())
                result["districts"].append(district_name)
                
                # Read most recent file
                latest = max(csv_files, key=lambda p: p.stat().st_mtime)
                df = pd.read_csv(latest)
                result["total_rows"] += len(df)
                
                result["latest_forecasts"][district_name] = {
                    "file": latest.name,
                    "days": len(df),
                }
        
        result["exists"] = len(found) > 0
        result["districts_with_data"] = len(found)
        
        # Find missing
        for expected_d in MAHARASHTRA_DISTRICTS:
            if expected_d.lower() not in found:
                result["missing_districts"].append(expected_d)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    historical: Dict[str, Any],
    current: Dict[str, Any],
    merged: Dict[str, Any],
    power: Dict[str, Any],
    openmeteo: Dict[str, Any],
    timestamp: str,
) -> str:
    """Generate Markdown completeness report."""
    
    lines = [
        "# MANDIMITRA - Data Completeness Report",
        f"",
        f"**Generated:** {timestamp}",
        f"**Scope:** Maharashtra Only (36 Districts)",
        f"",
        "---",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Summary table
    summary_items = []
    
    # Historical
    if historical["exists"]:
        summary_items.append(f"- ✅ **Historical Mandi:** {historical['rows']:,} rows")
        if historical["date_range"]:
            summary_items.append(f"  - Date range: {historical['date_range']['min']} to {historical['date_range']['max']}")
    else:
        summary_items.append(f"- ❌ **Historical Mandi:** Not found")
    
    # Current
    if current["exists"]:
        summary_items.append(f"- ✅ **Current Mandi:** {current['total_rows']:,} rows across {current['partitions']} days")
    else:
        summary_items.append(f"- ⚠️ **Current Mandi:** No data")
    
    # Merged
    if merged["exists"]:
        summary_items.append(f"- ✅ **Merged Dataset:** {merged['rows']:,} rows")
    else:
        summary_items.append(f"- ⚠️ **Merged Dataset:** Not created")
    
    # Weather
    summary_items.append(f"- {'✅' if power['exists'] else '❌'} **NASA POWER:** {power['districts_with_data']}/36 districts")
    summary_items.append(f"- {'✅' if openmeteo['exists'] else '❌'} **Open-Meteo:** {openmeteo['districts_with_data']}/36 districts")
    
    lines.extend(summary_items)
    
    # Detailed sections
    lines.extend([
        "",
        "---",
        "",
        "## 1. Mandi Data - Historical",
        "",
    ])
    
    if historical["exists"]:
        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Rows | {historical['rows']:,} |",
            f"| Unique Districts | {historical['unique_districts']} |",
            f"| Unique Markets | {historical['unique_markets']} |",
            f"| Unique Commodities | {historical['unique_commodities']} |",
            f"| Date Range | {historical['date_range']['min']} to {historical['date_range']['max']} |",
            f"| Unique Dates | {historical.get('unique_dates', 'N/A')} |",
            f"| States | {', '.join(historical['states'])} |",
            f"| Non-MH Rows | {historical['non_mh']} |",
            "",
        ])
        
        if historical['non_mh'] > 0:
            lines.append(f"⚠️ **WARNING:** {historical['non_mh']} non-Maharashtra rows detected!")
            lines.append("")
    else:
        lines.extend([
            f"❌ **Not Found:** {historical.get('error', 'Unknown error')}",
            "",
            "To create historical dataset:",
            "```bash",
            "python scripts/download_mandi_history_kaggle.py --download",
            "# OR",
            "python scripts/import_mandi_history.py --input-file /path/to/data.csv",
            "```",
            "",
        ])
    
    # Current mandi
    lines.extend([
        "## 2. Mandi Data - Current",
        "",
    ])
    
    if current["exists"]:
        lines.extend([
            f"| Date | Rows |",
            f"|------|------|",
        ])
        for date, rows in sorted(current["rows_per_date"].items(), reverse=True)[:10]:
            lines.append(f"| {date} | {rows:,} |")
        if len(current["rows_per_date"]) > 10:
            lines.append(f"| ... | ({len(current['rows_per_date'])-10} more days) |")
        lines.extend([
            "",
            f"**Total:** {current['total_rows']:,} rows across {current['partitions']} days",
            "",
        ])
    else:
        lines.extend([
            f"⚠️ **No current data found.**",
            "",
            "To download current data:",
            "```bash",
            "python scripts/download_mandi_current_datagov.py --download",
            "```",
            "",
        ])
    
    # Merged
    lines.extend([
        "## 3. Mandi Data - Merged (Training Dataset)",
        "",
    ])
    
    if merged["exists"]:
        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Rows | {merged['rows']:,} |",
            f"| Unique Districts | {merged['unique_districts']} |",
            f"| Unique Markets | {merged['unique_markets']} |",
            f"| Unique Commodities | {merged['unique_commodities']} |",
        ])
        if merged["date_range"]:
            lines.append(f"| Date Range | {merged['date_range']['min']} to {merged['date_range']['max']} |")
        lines.append("")
    else:
        lines.extend([
            f"⚠️ **Merged dataset not created.**",
            "",
            "To merge datasets:",
            "```bash",
            "python scripts/merge_mandi_datasets.py --merge",
            "```",
            "",
        ])
    
    # NASA POWER
    lines.extend([
        "## 4. Weather Data - NASA POWER (Historical)",
        "",
    ])
    
    if power["exists"]:
        lines.extend([
            f"**Coverage:** {power['districts_with_data']}/36 districts",
            f"**Total Rows:** {power['total_rows']:,}",
            "",
        ])
        
        if power["missing_districts"]:
            lines.append(f"⚠️ **Missing Districts:** {', '.join(power['missing_districts'][:10])}")
            if len(power["missing_districts"]) > 10:
                lines.append(f"   ... and {len(power['missing_districts'])-10} more")
            lines.append("")
        
        # Sample date ranges
        if power["date_ranges"]:
            lines.append("**Sample Date Ranges:**")
            lines.append("")
            lines.append("| District | Start | End | Days |")
            lines.append("|----------|-------|-----|------|")
            for district, dr in list(power["date_ranges"].items())[:5]:
                lines.append(f"| {district} | {dr['min'][:10]} | {dr['max'][:10]} | {dr['days']} |")
            lines.append("")
    else:
        lines.extend([
            f"❌ **No NASA POWER data found.**",
            "",
            "To download:",
            "```bash",
            "python scripts/download_weather_power_maharashtra.py --download",
            "```",
            "",
        ])
    
    # Open-Meteo
    lines.extend([
        "## 5. Weather Data - Open-Meteo (Forecast)",
        "",
    ])
    
    if openmeteo["exists"]:
        lines.extend([
            f"**Coverage:** {openmeteo['districts_with_data']}/36 districts",
            f"**Total Forecast Days:** {openmeteo['total_rows']:,}",
            "",
        ])
        
        if openmeteo["missing_districts"]:
            lines.append(f"⚠️ **Missing Districts:** {', '.join(openmeteo['missing_districts'][:10])}")
            lines.append("")
    else:
        lines.extend([
            f"❌ **No Open-Meteo forecast data found.**",
            "",
            "To download:",
            "```bash",
            "python scripts/download_weather_openmeteo_maharashtra.py --download",
            "```",
            "",
        ])
    
    # District coverage analysis
    lines.extend([
        "---",
        "",
        "## 6. Maharashtra District Coverage Analysis",
        "",
        f"**Expected Districts:** {len(MAHARASHTRA_DISTRICTS)}",
        "",
    ])
    
    # Compare mandi vs weather coverage
    mandi_districts = set(merged.get("districts", historical.get("districts", [])))
    power_districts = set(power.get("districts", []))
    openmeteo_districts = set(openmeteo.get("districts", []))
    
    lines.extend([
        "| District | Mandi | POWER | Forecast |",
        "|----------|-------|-------|----------|",
    ])
    
    for district in sorted(MAHARASHTRA_DISTRICTS):
        mandi_ok = "✅" if district.lower() in [d.lower() for d in mandi_districts] else "❌"
        power_ok = "✅" if district.lower() in [d.lower() for d in power_districts] else "❌"
        openmeteo_ok = "✅" if district.lower() in [d.lower() for d in openmeteo_districts] else "❌"
        lines.append(f"| {district} | {mandi_ok} | {power_ok} | {openmeteo_ok} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 7. Recommendations",
        "",
    ])
    
    recommendations = []
    
    if not historical["exists"]:
        recommendations.append("1. **Download historical mandi data** - Required for ML training")
    
    if not current["exists"]:
        recommendations.append("2. **Download current mandi data** - For latest price updates")
    
    if not merged["exists"] and historical["exists"]:
        recommendations.append("3. **Create merged dataset** - Combine historical + current")
    
    if power["missing_districts"]:
        recommendations.append(f"4. **Complete NASA POWER coverage** - Missing {len(power['missing_districts'])} districts")
    
    if openmeteo["missing_districts"]:
        recommendations.append(f"5. **Complete Open-Meteo coverage** - Missing {len(openmeteo['missing_districts'])} districts")
    
    if historical.get("non_mh", 0) > 0:
        recommendations.append(f"⚠️ **Remove non-Maharashtra data** - {historical['non_mh']} rows detected")
    
    if recommendations:
        lines.extend(recommendations)
    else:
        lines.append("✅ **All data sources are complete!**")
    
    lines.extend([
        "",
        "---",
        "",
        f"*Report generated by MANDIMITRA Data Completeness Check*",
    ])
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("completeness", None, level=log_level)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    timestamp_safe = get_utc_timestamp_safe()
    
    logger.info("Generating data completeness report...")
    
    try:
        # Load config
        config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            config = load_config(config_path)
        else:
            config = {}
        
        # Determine paths
        historical_path = PROJECT_ROOT / config.get("mandi", {}).get("historical", {}).get("processing", {}).get("output_path", "data/processed/mandi/history_maharashtra.parquet")
        current_dir = PROJECT_ROOT / config.get("mandi", {}).get("current", {}).get("output_dir", "data/raw/mandi/current")
        merged_path = PROJECT_ROOT / config.get("mandi", {}).get("merged", {}).get("output_path", "data/processed/mandi/mandi_maharashtra_all.parquet")
        power_dir = PROJECT_ROOT / config.get("weather", {}).get("nasa_power", {}).get("output_dir", "data/raw/weather/power_daily/maharashtra")
        openmeteo_dir = PROJECT_ROOT / config.get("weather", {}).get("openmeteo", {}).get("output_dir", "data/raw/weather/openmeteo_forecast/maharashtra")
        
        # Inspect all data sources
        logger.info("Inspecting historical mandi data...")
        historical = inspect_mandi_historical(historical_path)
        
        logger.info("Inspecting current mandi data...")
        current = inspect_mandi_current(current_dir)
        
        logger.info("Inspecting merged mandi data...")
        merged = inspect_mandi_merged(merged_path)
        
        logger.info("Inspecting NASA POWER data...")
        power = inspect_weather_power(power_dir)
        
        logger.info("Inspecting Open-Meteo data...")
        openmeteo = inspect_weather_openmeteo(openmeteo_dir)
        
        # Generate report
        report = generate_report(
            historical=historical,
            current=current,
            merged=merged,
            power=power,
            openmeteo=openmeteo,
            timestamp=timestamp,
        )
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = PROJECT_ROOT / "logs" / f"data_completeness_{timestamp_safe}.md"
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Report saved: {output_path}")
        
        # Print summary to console
        print(f"\n📋 Data Completeness Report Generated")
        print(f"   📁 Output: {output_path}")
        print(f"\n   Quick Summary:")
        print(f"   - Historical Mandi: {'✅' if historical['exists'] else '❌'} ({historical['rows']:,} rows)")
        print(f"   - Current Mandi: {'✅' if current['exists'] else '⚠️'} ({current['total_rows']:,} rows)")
        print(f"   - Merged Dataset: {'✅' if merged['exists'] else '⚠️'} ({merged['rows']:,} rows)")
        print(f"   - NASA POWER: {power['districts_with_data']}/36 districts")
        print(f"   - Open-Meteo: {openmeteo['districts_with_data']}/36 districts")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
