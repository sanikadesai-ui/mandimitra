#!/usr/bin/env python3
"""
MANDIMITRA - Build All Processed Datasets (Orchestrator)

Master script that runs the complete data processing pipeline:

1. build_canonical_mandi.py → Deduplicated mandi data
2. process_weather.py → Standardized weather data
3. build_model_datasets.py → ML-ready joined datasets

This ensures all datasets are built in the correct order with
proper dependencies. Running this script is IDEMPOTENT - you can
run it multiple times safely.

Output Structure:
    data/processed/
    ├── mandi/
    │   └── mandi_canonical.parquet       # Deduplicated mandi (6M+ rows)
    ├── weather/
    │   ├── power_daily_maharashtra.parquet  # NASA POWER historical
    │   └── forecast_maharashtra.parquet     # Open-Meteo forecast
    ├── model/
    │   ├── mandi_only_2001_2026.parquet     # Full mandi history
    │   └── mandi_weather_2016plus.parquet   # Mandi + weather joined
    ├── dim_districts.csv                    # District dimension table
    └── dim_commodities.csv                  # Commodity dimension table

    logs/
    ├── mandi_dedup_report_<timestamp>.md
    ├── weather_qc_report_<timestamp>.md
    ├── model_datasets_report_<timestamp>.md
    └── unmapped_districts_<timestamp>.md   # If any unmapped

Usage:
    python scripts/build_all_processed.py
    python scripts/build_all_processed.py --dry-run
    python scripts/build_all_processed.py --step mandi  # Run only mandi step
    python scripts/build_all_processed.py --skip-weather  # Skip weather processing
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"

# Pipeline steps in order
PIPELINE_STEPS = [
    {
        "name": "mandi",
        "description": "Build canonical mandi dataset",
        "script": "build_canonical_mandi.py",
        "required": True,
    },
    {
        "name": "weather",
        "description": "Process weather data",
        "script": "process_weather.py",
        "required": False,
    },
    {
        "name": "model",
        "description": "Build model datasets",
        "script": "build_model_datasets.py",
        "required": True,
    },
]

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

def run_script(
    script_name: str,
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """
    Run a Python script and return success status.
    
    Args:
        script_name: Name of script in scripts directory
        dry_run: Pass --dry-run flag to script
        extra_args: Additional arguments to pass
        
    Returns:
        True if successful, False otherwise
    """
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    
    if dry_run:
        cmd.append("--dry-run")
    
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return False


def check_prerequisites() -> bool:
    """
    Check that all required files and dependencies exist.
    
    Returns:
        True if all prerequisites met
    """
    logger.info("Checking prerequisites...")
    
    all_ok = True
    
    # Check for required Python packages
    required_packages = ["duckdb", "pandas", "pandera", "pyarrow"]
    for pkg in required_packages:
        try:
            __import__(pkg)
            logger.info(f"  ✓ {pkg}")
        except ImportError:
            logger.error(f"  ✗ {pkg} (not installed)")
            all_ok = False
    
    # Check for input data
    raw_mandi = PROJECT_ROOT / "data" / "raw" / "mandi"
    if not raw_mandi.exists():
        logger.warning(f"  ⚠ Raw mandi directory not found: {raw_mandi}")
    else:
        parquet_files = list(raw_mandi.glob("*.parquet"))
        csv_files = list(raw_mandi.glob("*.csv"))
        logger.info(f"  ✓ Raw mandi: {len(parquet_files)} parquet, {len(csv_files)} csv files")
    
    raw_weather = PROJECT_ROOT / "data" / "raw" / "weather"
    if not raw_weather.exists():
        logger.warning(f"  ⚠ Raw weather directory not found: {raw_weather}")
    else:
        weather_files = list(raw_weather.glob("*.csv"))
        logger.info(f"  ✓ Raw weather: {len(weather_files)} csv files")
    
    return all_ok


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# =============================================================================
# MAIN
# =============================================================================

def main(
    dry_run: bool = False,
    step: Optional[str] = None,
    skip_weather: bool = False,
):
    """
    Run the complete data processing pipeline.
    
    Args:
        dry_run: Don't save outputs
        step: Run only specific step
        skip_weather: Skip weather processing
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 70)
    logger.info("MANDIMITRA - Build All Processed Datasets")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now().isoformat()}")
    
    if dry_run:
        logger.info("MODE: DRY RUN (no files will be saved)")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Install missing packages:")
        logger.error("  pip install duckdb pandas pandera pyarrow")
        return False
    
    # Determine which steps to run
    steps_to_run = []
    
    if step:
        # Run only specific step
        matching = [s for s in PIPELINE_STEPS if s["name"] == step]
        if not matching:
            logger.error(f"Unknown step: {step}")
            logger.error(f"Available steps: {[s['name'] for s in PIPELINE_STEPS]}")
            return False
        steps_to_run = matching
    else:
        # Run all steps (optionally skipping weather)
        for s in PIPELINE_STEPS:
            if skip_weather and s["name"] == "weather":
                logger.info(f"Skipping step: {s['name']} (--skip-weather)")
                continue
            steps_to_run.append(s)
    
    # Run pipeline
    logger.info("")
    logger.info(f"Running {len(steps_to_run)} pipeline steps...")
    logger.info("-" * 70)
    
    results = {}
    
    for i, step_info in enumerate(steps_to_run, 1):
        step_name = step_info["name"]
        script_name = step_info["script"]
        description = step_info["description"]
        
        logger.info("")
        logger.info(f"[{i}/{len(steps_to_run)}] {description}")
        logger.info("-" * 50)
        
        step_start = time.time()
        success = run_script(script_name, dry_run=dry_run)
        step_duration = time.time() - step_start
        
        results[step_name] = {
            "success": success,
            "duration": step_duration,
        }
        
        if success:
            logger.info(f"✓ Step '{step_name}' completed in {format_duration(step_duration)}")
        else:
            logger.error(f"✗ Step '{step_name}' FAILED after {format_duration(step_duration)}")
            if step_info.get("required", True):
                logger.error("This is a required step. Aborting pipeline.")
                break
    
    # Summary
    total_duration = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    
    all_success = True
    for step_name, result in results.items():
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        duration = format_duration(result["duration"])
        logger.info(f"  {step_name}: {status} ({duration})")
        if not result["success"]:
            all_success = False
    
    logger.info("")
    logger.info(f"Total time: {format_duration(total_duration)}")
    logger.info(f"Completed: {datetime.now().isoformat()}")
    
    if all_success:
        logger.info("")
        logger.info("🎉 All steps completed successfully!")
        logger.info("")
        logger.info("Output files:")
        logger.info("  data/processed/mandi/mandi_canonical.parquet")
        logger.info("  data/processed/weather/power_daily_maharashtra.parquet")
        logger.info("  data/processed/weather/forecast_maharashtra.parquet")
        logger.info("  data/processed/model/mandi_only_2001_2026.parquet")
        logger.info("  data/processed/model/mandi_weather_2016plus.parquet")
        logger.info("  data/processed/dim_districts.csv")
        logger.info("  data/processed/dim_commodities.csv")
    else:
        logger.error("")
        logger.error("❌ Pipeline failed. Check errors above.")
    
    return all_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete MANDIMITRA data processing pipeline"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save outputs, just show what would be done",
    )
    parser.add_argument(
        "--step",
        choices=["mandi", "weather", "model"],
        help="Run only a specific step",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather processing step",
    )
    
    args = parser.parse_args()
    
    success = main(
        dry_run=args.dry_run,
        step=args.step,
        skip_weather=args.skip_weather,
    )
    
    sys.exit(0 if success else 1)
