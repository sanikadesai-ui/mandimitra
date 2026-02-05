#!/usr/bin/env python3
"""
MANDIMITRA - Full Data Pipeline Orchestrator

Downloads, processes, and validates all data for the MANDIMITRA crop-risk model:
1. Historical mandi data (Kaggle OR local import)
2. Current daily mandi data (Data.gov.in)
3. Merged/upserted mandi dataset
4. Historical weather (NASA POWER)
5. Forecast weather (Open-Meteo)
6. Data completeness report

Usage:
    # Full pipeline with Kaggle download
    python scripts/download_all_data.py --historical-source kaggle
    
    # Full pipeline with local file import
    python scripts/download_all_data.py --historical-source local --historical-file /path/to/data.csv
    
    # Skip historical (already downloaded)
    python scripts/download_all_data.py --skip-historical
    
    # Skip weather data
    python scripts/download_all_data.py --skip-weather
    
    # Dry run (show what would be done)
    python scripts/download_all_data.py --dry-run
    
    # Full options
    python scripts/download_all_data.py --help

Exit Codes:
    0: All steps completed successfully
    1: One or more steps failed
    2: Critical failure (early abort)

⚠️  HARD CONSTRAINT: Maharashtra-only data throughout entire pipeline.

Environment Variables:
    KAGGLE_USERNAME: Required for Kaggle downloads
    KAGGLE_KEY: Required for Kaggle downloads
    DATA_GOV_API_KEY: Required for Data.gov.in API (via .env)
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import load_config
from src.utils.logging_utils import setup_logger, get_utc_timestamp_safe


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MANDIMITRA full data pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download everything (Kaggle historical)
    python scripts/download_all_data.py --historical-source kaggle
    
    # Use local historical file
    python scripts/download_all_data.py --historical-source local --historical-file data.csv
    
    # Skip historical, only update current + weather
    python scripts/download_all_data.py --skip-historical
    
    # Weather only
    python scripts/download_all_data.py --skip-mandi
        """,
    )
    
    # Historical data source
    parser.add_argument(
        "--historical-source",
        choices=["kaggle", "local", "skip"],
        default="skip",
        help="Source for historical mandi data (kaggle, local file, or skip)",
    )
    parser.add_argument(
        "--historical-file",
        type=str,
        default=None,
        help="Path to local historical data file (required if --historical-source=local)",
    )
    
    # Skip flags
    parser.add_argument(
        "--skip-historical",
        action="store_true",
        help="Skip historical mandi data download/import",
    )
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Skip current mandi data download",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge/upsert step",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip all weather data downloads",
    )
    parser.add_argument(
        "--skip-power",
        action="store_true",
        help="Skip NASA POWER historical weather",
    )
    parser.add_argument(
        "--skip-openmeteo",
        action="store_true",
        help="Skip Open-Meteo forecast weather",
    )
    parser.add_argument(
        "--skip-mandi",
        action="store_true",
        help="Skip all mandi data (historical + current + merge)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip completeness report generation",
    )
    
    # Control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure instead of continuing",
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
# STEP EXECUTION
# =============================================================================

class StepResult:
    """Result of a pipeline step."""
    
    def __init__(self, name: str, success: bool, message: str, duration: float = 0, skipped: bool = False):
        self.name = name
        self.success = success
        self.message = message
        self.duration = duration
        self.skipped = skipped
    
    def __str__(self):
        if self.skipped:
            return f"⏭️  {self.name}: SKIPPED"
        elif self.success:
            return f"✅ {self.name}: OK ({self.duration:.1f}s)"
        else:
            return f"❌ {self.name}: FAILED - {self.message}"


def run_script(
    script_path: Path,
    args: List[str],
    logger: logging.Logger,
    dry_run: bool = False,
    timeout: int = 3600,
) -> Tuple[bool, str, float]:
    """
    Run a Python script with arguments.
    
    Returns:
        Tuple of (success, message, duration)
    """
    cmd = [sys.executable, str(script_path)] + args
    cmd_str = " ".join(cmd)
    
    if dry_run:
        logger.info(f"[DRY RUN] Would run: {cmd_str}")
        return True, "Dry run", 0.0
    
    logger.info(f"Running: {cmd_str}")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        duration = time.time() - start
        
        if result.returncode == 0:
            return True, "Success", duration
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            return False, error_msg, duration
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return False, f"Timeout after {timeout}s", duration
    except Exception as e:
        duration = time.time() - start
        return False, str(e), duration


def step_historical_kaggle(logger: logging.Logger, dry_run: bool) -> StepResult:
    """Download historical mandi data from Kaggle."""
    script = PROJECT_ROOT / "scripts" / "download_mandi_history_kaggle.py"
    
    if not script.exists():
        return StepResult("Historical (Kaggle)", False, f"Script not found: {script}")
    
    # Check Kaggle credentials
    if not dry_run:
        if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
            return StepResult(
                "Historical (Kaggle)", 
                False, 
                "KAGGLE_USERNAME and KAGGLE_KEY environment variables required"
            )
    
    success, msg, duration = run_script(
        script, 
        ["--download"], 
        logger, 
        dry_run=dry_run,
        timeout=7200,  # 2 hours for large download
    )
    
    return StepResult("Historical (Kaggle)", success, msg, duration)


def step_historical_local(logger: logging.Logger, dry_run: bool, file_path: str) -> StepResult:
    """Import historical mandi data from local file."""
    script = PROJECT_ROOT / "scripts" / "import_mandi_history.py"
    
    if not script.exists():
        return StepResult("Historical (Local)", False, f"Script not found: {script}")
    
    if not file_path:
        return StepResult("Historical (Local)", False, "No file path provided")
    
    if not dry_run and not Path(file_path).exists():
        return StepResult("Historical (Local)", False, f"File not found: {file_path}")
    
    success, msg, duration = run_script(
        script,
        ["--input-file", file_path, "--import"],
        logger,
        dry_run=dry_run,
        timeout=3600,
    )
    
    return StepResult("Historical (Local)", success, msg, duration)


def step_current_datagov(logger: logging.Logger, dry_run: bool) -> StepResult:
    """Download current mandi data from Data.gov.in."""
    script = PROJECT_ROOT / "scripts" / "download_mandi_current_datagov.py"
    
    if not script.exists():
        return StepResult("Current (Data.gov)", False, f"Script not found: {script}")
    
    success, msg, duration = run_script(
        script,
        ["--download"],
        logger,
        dry_run=dry_run,
        timeout=1800,
    )
    
    return StepResult("Current (Data.gov)", success, msg, duration)


def step_merge(logger: logging.Logger, dry_run: bool) -> StepResult:
    """Merge historical and current mandi data."""
    script = PROJECT_ROOT / "scripts" / "merge_mandi_datasets.py"
    
    if not script.exists():
        return StepResult("Merge/Upsert", False, f"Script not found: {script}")
    
    success, msg, duration = run_script(
        script,
        ["--merge"],
        logger,
        dry_run=dry_run,
        timeout=1800,
    )
    
    return StepResult("Merge/Upsert", success, msg, duration)


def step_weather_power(logger: logging.Logger, dry_run: bool) -> StepResult:
    """Download NASA POWER historical weather."""
    script = PROJECT_ROOT / "scripts" / "download_weather_power_maharashtra.py"
    
    if not script.exists():
        return StepResult("Weather (NASA POWER)", False, f"Script not found: {script}")
    
    success, msg, duration = run_script(
        script,
        ["--download"],
        logger,
        dry_run=dry_run,
        timeout=7200,  # 2 hours for all districts
    )
    
    return StepResult("Weather (NASA POWER)", success, msg, duration)


def step_weather_openmeteo(logger: logging.Logger, dry_run: bool) -> StepResult:
    """Download Open-Meteo forecast weather."""
    script = PROJECT_ROOT / "scripts" / "download_weather_openmeteo_maharashtra.py"
    
    if not script.exists():
        return StepResult("Weather (Open-Meteo)", False, f"Script not found: {script}")
    
    success, msg, duration = run_script(
        script,
        ["--download"],
        logger,
        dry_run=dry_run,
        timeout=1800,
    )
    
    return StepResult("Weather (Open-Meteo)", success, msg, duration)


def step_completeness_report(logger: logging.Logger, dry_run: bool) -> StepResult:
    """Generate data completeness report."""
    script = PROJECT_ROOT / "scripts" / "generate_completeness_report.py"
    
    if not script.exists():
        return StepResult("Completeness Report", False, f"Script not found: {script}")
    
    success, msg, duration = run_script(
        script,
        [],
        logger,
        dry_run=dry_run,
        timeout=300,
    )
    
    return StepResult("Completeness Report", success, msg, duration)


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load environment
    load_dotenv(PROJECT_ROOT / ".env")
    
    # Setup logging
    timestamp_safe = get_utc_timestamp_safe()
    log_file = PROJECT_ROOT / "logs" / f"download_all_{timestamp_safe}.log"
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("orchestrator", log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("MANDIMITRA - Full Data Pipeline Orchestrator")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Print configuration
    print("\n" + "=" * 60)
    print("🚀 MANDIMITRA - Full Data Pipeline")
    print("=" * 60)
    
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made")
    
    # Determine what to run
    run_historical = not args.skip_historical and not args.skip_mandi and args.historical_source != "skip"
    run_current = not args.skip_current and not args.skip_mandi
    run_merge = not args.skip_merge and not args.skip_mandi
    run_power = not args.skip_power and not args.skip_weather
    run_openmeteo = not args.skip_openmeteo and not args.skip_weather
    run_report = not args.skip_report
    
    # Validate historical source
    if run_historical and args.historical_source == "local":
        if not args.historical_file:
            print("\n❌ Error: --historical-file required when --historical-source=local")
            return 2
    
    # Show plan
    print("\n📋 Execution Plan:")
    steps_planned = []
    
    if run_historical:
        if args.historical_source == "kaggle":
            steps_planned.append(("Historical Mandi (Kaggle)", "Download from Kaggle"))
        else:
            steps_planned.append(("Historical Mandi (Local)", f"Import from {args.historical_file}"))
    
    if run_current:
        steps_planned.append(("Current Mandi", "Download from Data.gov.in"))
    
    if run_merge:
        steps_planned.append(("Merge/Upsert", "Combine historical + current"))
    
    if run_power:
        steps_planned.append(("NASA POWER Weather", "Download 10-year historical"))
    
    if run_openmeteo:
        steps_planned.append(("Open-Meteo Weather", "Download 16-day forecast"))
    
    if run_report:
        steps_planned.append(("Completeness Report", "Generate data summary"))
    
    for i, (name, desc) in enumerate(steps_planned, 1):
        print(f"   {i}. {name}: {desc}")
    
    if not steps_planned:
        print("   (No steps to run)")
        return 0
    
    print()
    
    # Execute steps
    results: List[StepResult] = []
    
    # Step 1: Historical mandi data
    if run_historical:
        print("📥 Step 1: Historical Mandi Data...")
        if args.historical_source == "kaggle":
            result = step_historical_kaggle(logger, args.dry_run)
        else:
            result = step_historical_local(logger, args.dry_run, args.historical_file)
        results.append(result)
        print(f"   {result}")
        
        if not result.success and args.fail_fast:
            print("\n❌ Aborting due to failure (--fail-fast)")
            return 1
    else:
        results.append(StepResult("Historical", True, "Skipped", skipped=True))
    
    # Step 2: Current mandi data
    if run_current:
        print("📥 Step 2: Current Mandi Data...")
        result = step_current_datagov(logger, args.dry_run)
        results.append(result)
        print(f"   {result}")
        
        if not result.success and args.fail_fast:
            print("\n❌ Aborting due to failure (--fail-fast)")
            return 1
    else:
        results.append(StepResult("Current", True, "Skipped", skipped=True))
    
    # Step 3: Merge
    if run_merge:
        print("🔀 Step 3: Merge/Upsert...")
        result = step_merge(logger, args.dry_run)
        results.append(result)
        print(f"   {result}")
        
        if not result.success and args.fail_fast:
            print("\n❌ Aborting due to failure (--fail-fast)")
            return 1
    else:
        results.append(StepResult("Merge", True, "Skipped", skipped=True))
    
    # Step 4: NASA POWER weather
    if run_power:
        print("🌤️  Step 4: NASA POWER Weather...")
        result = step_weather_power(logger, args.dry_run)
        results.append(result)
        print(f"   {result}")
        
        if not result.success and args.fail_fast:
            print("\n❌ Aborting due to failure (--fail-fast)")
            return 1
    else:
        results.append(StepResult("NASA POWER", True, "Skipped", skipped=True))
    
    # Step 5: Open-Meteo weather
    if run_openmeteo:
        print("🌦️  Step 5: Open-Meteo Forecast...")
        result = step_weather_openmeteo(logger, args.dry_run)
        results.append(result)
        print(f"   {result}")
        
        if not result.success and args.fail_fast:
            print("\n❌ Aborting due to failure (--fail-fast)")
            return 1
    else:
        results.append(StepResult("Open-Meteo", True, "Skipped", skipped=True))
    
    # Step 6: Completeness report
    if run_report:
        print("📊 Step 6: Completeness Report...")
        result = step_completeness_report(logger, args.dry_run)
        results.append(result)
        print(f"   {result}")
    else:
        results.append(StepResult("Report", True, "Skipped", skipped=True))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Pipeline Summary")
    print("=" * 60)
    
    total_duration = sum(r.duration for r in results)
    success_count = sum(1 for r in results if r.success and not r.skipped)
    failed_count = sum(1 for r in results if not r.success and not r.skipped)
    skipped_count = sum(1 for r in results if r.skipped)
    
    for result in results:
        print(f"   {result}")
    
    print()
    print(f"   Total time: {total_duration:.1f}s")
    print(f"   ✅ Succeeded: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    
    logger.info(f"Pipeline completed: {success_count} succeeded, {failed_count} failed, {skipped_count} skipped")
    logger.info(f"Total duration: {total_duration:.1f}s")
    
    if failed_count > 0:
        print("\n⚠️  Some steps failed. Check logs for details.")
        print(f"   Log file: {log_file}")
        return 1
    
    print("\n✅ Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
