#!/usr/bin/env python3
"""
MANDIMITRA - Maharashtra Data Validation Script

Validates downloaded mandi and weather data using strict Pandera schemas.
Enforces HARD CONSTRAINT: Maharashtra-only data.

⚠️  This script will FAIL validation if any non-Maharashtra records are found.

Usage:
    python scripts/validate_data.py --mandi data/raw/mandi/maharashtra/merged/*/mandi_*.csv
    python scripts/validate_data.py --all-recent
    python scripts/validate_data.py --strict
    python scripts/validate_data.py --help

Author: MANDIMITRA Team
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import setup_logger, ProgressLogger
from src.utils.io_utils import load_config
from src.utils.maharashtra import MAHARASHTRA_STATE_NAME, is_maharashtra_state
from src.utils.audit import AuditLogger
from src.utils.logging_utils import get_utc_timestamp_safe


# =============================================================================
# PANDERA SCHEMAS - MAHARASHTRA ONLY
# =============================================================================

def get_mandi_schema() -> DataFrameSchema:
    """
    Create Pandera schema for Maharashtra mandi price data.
    
    ⚠️  HARD CONSTRAINT: state MUST be "Maharashtra"
    
    Required columns:
    - state (must be Maharashtra)
    - district, market, commodity
    - arrival_date (parseable date)
    - modal_price (numeric, non-negative)
    
    Returns:
        DataFrameSchema for validation
    """
    return DataFrameSchema(
        columns={
            "state": Column(
                pa.String,
                nullable=False,
                checks=[
                    Check(
                        lambda s: s.apply(is_maharashtra_state).all(),
                        error=f"HARD CONSTRAINT VIOLATION: All records must be state='{MAHARASHTRA_STATE_NAME}'"
                    ),
                ],
                description="State name (MUST be Maharashtra)",
            ),
            "district": Column(
                pa.String,
                nullable=False,
                description="District name (required)",
            ),
            "market": Column(
                pa.String,
                nullable=False,
                description="Market/Mandi name (required)",
            ),
            "commodity": Column(
                pa.String,
                nullable=False,
                description="Commodity name (required)",
            ),
            "variety": Column(
                pa.String,
                nullable=True,
                description="Commodity variety (optional)",
            ),
            "arrival_date": Column(
                pa.String,
                nullable=False,
                description="Date of price recording (must be parseable)",
            ),
            "min_price": Column(
                pa.Float,
                nullable=True,
                checks=[
                    Check.ge(0, error="min_price must be non-negative"),
                ],
                coerce=True,
                description="Minimum price (Rs/Quintal)",
            ),
            "max_price": Column(
                pa.Float,
                nullable=True,
                checks=[
                    Check.ge(0, error="max_price must be non-negative"),
                ],
                coerce=True,
                description="Maximum price (Rs/Quintal)",
            ),
            "modal_price": Column(
                pa.Float,
                nullable=False,
                checks=[
                    Check.ge(0, error="modal_price must be non-negative"),
                    Check.le(1000000, error="modal_price suspiciously high (>10 lakh/quintal)"),
                ],
                coerce=True,
                description="Modal/Most common price (Rs/Quintal) - REQUIRED",
            ),
        },
        strict=False,  # Allow extra columns
        coerce=True,
        checks=[
            # Price consistency check
            Check(
                lambda df: (
                    df["modal_price"].isna() | 
                    df["min_price"].isna() | 
                    df["max_price"].isna() |
                    ((df["modal_price"] >= df["min_price"] * 0.9) &  # Allow 10% tolerance
                     (df["modal_price"] <= df["max_price"] * 1.1))
                ).all(),
                error="modal_price should be approximately between min_price and max_price",
            ),
        ],
        name="MaharashtraMandiPriceSchema",
        description="Schema for Maharashtra AGMARKNET mandi price data",
    )


def get_power_daily_schema() -> DataFrameSchema:
    """
    Create Pandera schema for NASA POWER daily data.
    
    Expected columns:
    - date (datetime)
    - PRECTOTCORR (precipitation in mm)
    - Optional: T2M, T2M_MAX, T2M_MIN, RH2M
    
    Returns:
        DataFrameSchema for validation
    """
    return DataFrameSchema(
        columns={
            "date": Column(
                pa.DateTime,
                nullable=False,
                coerce=True,
                description="Date of observation",
            ),
            "PRECTOTCORR": Column(
                pa.Float,
                nullable=True,
                checks=[
                    Check.ge(0, error="Precipitation cannot be negative"),
                    Check.le(500, error="Daily precipitation suspiciously high (>500mm)"),
                ],
                coerce=True,
                description="Precipitation corrected (mm/day)",
            ),
        },
        strict=False,  # Allow T2M, RH2M, etc.
        coerce=True,
        name="MaharashtraNASAPowerDailySchema",
        description="Schema for NASA POWER daily rainfall data (Maharashtra districts)",
    )


def get_openmeteo_forecast_schema() -> DataFrameSchema:
    """
    Create Pandera schema for Open-Meteo forecast data.
    
    Expected columns:
    - date (datetime)
    - precipitation_sum (mm)
    - Optional: precipitation_probability_max, temperature, etc.
    
    Returns:
        DataFrameSchema for validation
    """
    return DataFrameSchema(
        columns={
            "date": Column(
                pa.DateTime,
                nullable=False,
                coerce=True,
                description="Forecast date",
            ),
            "precipitation_sum": Column(
                pa.Float,
                nullable=True,
                checks=[
                    Check.ge(0, error="Precipitation cannot be negative"),
                    Check.le(500, error="Daily precipitation suspiciously high (>500mm)"),
                ],
                coerce=True,
                description="Total daily precipitation (mm)",
            ),
        },
        strict=False,
        coerce=True,
        name="MaharashtraOpenMeteoForecastSchema",
        description="Schema for Open-Meteo rainfall forecast data (Maharashtra districts)",
    )


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

class ValidationResult:
    """Container for validation results."""
    
    def __init__(
        self,
        file_path: Path,
        data_type: str,
        is_valid: bool,
        row_count: int,
        column_count: int,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        stats: Optional[Dict[str, Any]] = None,
        non_mh_count: int = 0,
    ):
        self.file_path = file_path
        self.data_type = data_type
        self.is_valid = is_valid
        self.row_count = row_count
        self.column_count = column_count
        self.errors = errors or []
        self.warnings = warnings or []
        self.stats = stats or {}
        self.non_mh_count = non_mh_count
    
    def __str__(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        lines = [
            f"\n{'='*60}",
            f"Validation Result: {status}",
            f"{'='*60}",
            f"File: {self.file_path}",
            f"Type: {self.data_type}",
            f"Rows: {self.row_count:,}",
            f"Columns: {self.column_count}",
        ]
        
        if self.non_mh_count > 0:
            lines.append(f"\n🚨 NON-MAHARASHTRA RECORDS: {self.non_mh_count}")
        
        if self.stats:
            lines.append("\nStatistics:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
        
        if self.warnings:
            lines.append("\n⚠️  Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        if self.errors:
            lines.append("\n❌ Errors:")
            for error in self.errors[:10]:  # Limit display
                lines.append(f"  - {error}")
            if len(self.errors) > 10:
                lines.append(f"  ... and {len(self.errors) - 10} more errors")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def validate_file(
    file_path: Path,
    schema: DataFrameSchema,
    data_type: str,
    logger,
    check_maharashtra: bool = False,
) -> ValidationResult:
    """
    Validate a CSV file against a Pandera schema.
    
    Args:
        file_path: Path to CSV file
        schema: Pandera DataFrameSchema
        data_type: Type description for reporting
        logger: Logger instance
        check_maharashtra: If True, enforce Maharashtra-only constraint
        
    Returns:
        ValidationResult object
    """
    errors = []
    warnings = []
    stats = {}
    non_mh_count = 0
    
    # Check file exists
    if not file_path.exists():
        return ValidationResult(
            file_path=file_path,
            data_type=data_type,
            is_valid=False,
            row_count=0,
            column_count=0,
            errors=[f"File not found: {file_path}"],
        )
    
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return ValidationResult(
            file_path=file_path,
            data_type=data_type,
            is_valid=False,
            row_count=0,
            column_count=0,
            errors=[f"Failed to read CSV: {e}"],
        )
    
    row_count = len(df)
    column_count = len(df.columns)
    
    # Check for empty file
    if row_count == 0:
        return ValidationResult(
            file_path=file_path,
            data_type=data_type,
            is_valid=False,
            row_count=0,
            column_count=column_count,
            errors=["File is empty (0 rows)"],
        )
    
    # =========================================================================
    # HARD CONSTRAINT: Maharashtra-only check for mandi data
    # =========================================================================
    if check_maharashtra and "state" in df.columns:
        non_mh_mask = ~df["state"].apply(is_maharashtra_state)
        non_mh_count = non_mh_mask.sum()
        
        if non_mh_count > 0:
            errors.append(
                f"HARD CONSTRAINT VIOLATION: {non_mh_count} non-Maharashtra records found! "
                f"States: {df.loc[non_mh_mask, 'state'].unique().tolist()}"
            )
            logger.error(f"CRITICAL: {non_mh_count} non-Maharashtra records in {file_path}")
    
    # Basic stats
    stats["columns"] = list(df.columns)
    
    # Required columns check
    required_cols = {"state", "district", "market", "commodity", "arrival_date", "modal_price"}
    if check_maharashtra:
        missing_required = required_cols - set(df.columns)
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")
    
    # Null checks
    stats["null_counts"] = df.isnull().sum().to_dict()
    stats["null_percentage"] = {
        col: f"{(count / row_count * 100):.1f}%"
        for col, count in stats["null_counts"].items()
        if count > 0
    }
    
    # Check for high null percentages in critical columns
    critical_cols = ["district", "market", "commodity", "modal_price"] if check_maharashtra else []
    for col in critical_cols:
        if col in df.columns:
            null_pct = df[col].isnull().sum() / row_count * 100
            if null_pct > 10:
                warnings.append(f"Critical column '{col}' has {null_pct:.1f}% null values")
    
    # Date parsing check for mandi data
    if check_maharashtra and "arrival_date" in df.columns:
        try:
            # Try common date formats
            parsed = pd.to_datetime(df["arrival_date"], errors="coerce", dayfirst=True)
            unparsed = parsed.isna().sum()
            if unparsed > 0:
                warnings.append(f"{unparsed} arrival_date values could not be parsed")
            else:
                stats["date_range"] = f"{parsed.min()} to {parsed.max()}"
        except Exception as e:
            warnings.append(f"Date parsing check failed: {e}")
    
    # Validate against schema
    try:
        schema.validate(df, lazy=True)
        schema_valid = True
    except pa.errors.SchemaErrors as e:
        schema_valid = False
        for _, row in e.failure_cases.iterrows():
            errors.append(
                f"Column '{row.get('column', 'N/A')}': "
                f"{row.get('check', 'N/A')} - "
                f"Index {row.get('index', 'N/A')}"
            )
    except Exception as e:
        schema_valid = False
        errors.append(f"Validation error: {e}")
    
    # Final validity determination
    # For Maharashtra data: non_mh_count > 0 is ALWAYS invalid
    is_valid = schema_valid and len(errors) == 0 and non_mh_count == 0
    
    return ValidationResult(
        file_path=file_path,
        data_type=data_type,
        is_valid=is_valid,
        row_count=row_count,
        column_count=column_count,
        errors=errors[:20],  # Limit to first 20 errors
        warnings=warnings,
        stats=stats,
        non_mh_count=non_mh_count,
    )


def validate_mandi(file_path: Path, logger) -> ValidationResult:
    """Validate Maharashtra mandi price data file."""
    return validate_file(
        file_path, 
        get_mandi_schema(), 
        "Maharashtra Mandi Price Data", 
        logger,
        check_maharashtra=True,  # HARD CONSTRAINT
    )


def validate_power(file_path: Path, logger) -> ValidationResult:
    """Validate NASA POWER daily data file."""
    return validate_file(file_path, get_power_daily_schema(), "NASA POWER Daily (Maharashtra)", logger)


def validate_openmeteo(file_path: Path, logger) -> ValidationResult:
    """Validate Open-Meteo forecast data file."""
    return validate_file(file_path, get_openmeteo_forecast_schema(), "Open-Meteo Forecast (Maharashtra)", logger)


def find_recent_files(
    data_dir: Path,
    pattern: str,
    limit: int = 10,
) -> List[Path]:
    """
    Find most recent files matching pattern.
    
    Args:
        data_dir: Base data directory
        pattern: Glob pattern (e.g., "mandi/maharashtra/**/*.csv")
        limit: Maximum files to return
        
    Returns:
        List of file paths, sorted by modification time (newest first)
    """
    files = list(data_dir.glob(pattern))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def find_maharashtra_mandi_files(data_dir: Path, limit: int = 10) -> List[Path]:
    """
    Find recent Maharashtra mandi data files.
    
    Searches in:
    - data/raw/mandi/maharashtra/merged/*.csv (merged files)
    - data/raw/mandi/maharashtra/**/*.csv (all files)
    
    Returns:
        List of CSV paths, prioritizing merged files
    """
    merged_pattern = "mandi/maharashtra/merged/**/*.csv"
    merged_files = find_recent_files(data_dir, merged_pattern, limit)
    
    if merged_files:
        return merged_files
    
    # Fallback to any Maharashtra mandi files
    all_pattern = "mandi/maharashtra/**/*.csv"
    return find_recent_files(data_dir, all_pattern, limit)


def find_maharashtra_weather_files(data_dir: Path, data_type: str, limit: int = 10) -> List[Path]:
    """
    Find recent Maharashtra weather data files.
    
    Args:
        data_dir: Base data directory
        data_type: "power" or "openmeteo"
        limit: Maximum files to return
        
    Returns:
        List of CSV paths
    """
    if data_type == "power":
        pattern = "weather/power_daily/maharashtra/**/*.csv"
    elif data_type == "openmeteo":
        pattern = "weather/openmeteo_forecast/maharashtra/**/*.csv"
    else:
        return []
    
    return find_recent_files(data_dir, pattern, limit)


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate downloaded mandi and weather data for MANDIMITRA.\n"
            "This script validates MAHARASHTRA-ONLY data with strict checks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAHARASHTRA-ONLY VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This validator enforces HARD CONSTRAINTS:
  - Mandi data MUST have state="Maharashtra" for all rows
  - Non-Maharashtra records will cause validation to FAIL
  - Weather data is expected from Maharashtra district HQs

Examples:
    # Validate a specific Maharashtra mandi file
    python scripts/validate_data.py --mandi data/raw/mandi/maharashtra/merged/merged_2025.csv
    
    # Validate NASA POWER data for Maharashtra districts
    python scripts/validate_data.py --power data/raw/weather/power_daily/maharashtra/Pune/*.csv
    
    # Validate all recent Maharashtra downloads
    python scripts/validate_data.py --all-recent
    
    # Strict mode (exit with error code 1 if any validation fails)
    python scripts/validate_data.py --all-recent --strict
    
    # Generate audit report
    python scripts/validate_data.py --all-recent --audit
        """,
    )
    
    # File type arguments
    parser.add_argument(
        "--mandi",
        type=str,
        nargs="*",
        help="Path(s) to Maharashtra mandi CSV file(s) to validate",
    )
    parser.add_argument(
        "--power",
        type=str,
        nargs="*",
        help="Path(s) to NASA POWER CSV file(s) to validate (Maharashtra districts)",
    )
    parser.add_argument(
        "--openmeteo",
        type=str,
        nargs="*",
        help="Path(s) to Open-Meteo CSV file(s) to validate (Maharashtra districts)",
    )
    
    # Auto-discovery
    parser.add_argument(
        "--all-recent",
        action="store_true",
        help="Find and validate most recent Maharashtra files of each type",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Base data directory for auto-discovery (default: data/raw)",
    )
    
    # Options
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code 1 if any validation fails",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary, not detailed results",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Generate Markdown audit report in logs/",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for data validation."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = PROJECT_ROOT / "logs" / "validation.log"
    logger = setup_logger("validate_data", log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("MANDIMITRA - Maharashtra Data Validation")
    logger.info("=" * 60)
    
    results: List[ValidationResult] = []
    
    # Initialize audit logger if requested
    audit = None
    if args.audit:
        audit = AuditLogger("Maharashtra Data Validation")
        audit.add_section("Configuration", {
            "Target State": MAHARASHTRA_STATE_NAME,
            "Strict Mode": str(args.strict),
            "Data Directory": str(PROJECT_ROOT / args.data_dir),
        })
    
    try:
        # Collect files to validate
        files_to_validate = {
            "mandi": [],
            "power": [],
            "openmeteo": [],
        }
        
        if args.all_recent:
            data_dir = PROJECT_ROOT / args.data_dir
            logger.info(f"Auto-discovering recent Maharashtra files in {data_dir}")
            
            # Find recent Maharashtra mandi files
            mandi_files = find_maharashtra_mandi_files(data_dir, limit=5)
            files_to_validate["mandi"].extend(mandi_files)
            logger.info(f"Found {len(mandi_files)} Maharashtra mandi file(s)")
            
            # Find recent NASA POWER files (Maharashtra districts)
            power_files = find_maharashtra_weather_files(data_dir, "power", limit=10)
            files_to_validate["power"].extend(power_files)
            logger.info(f"Found {len(power_files)} NASA POWER file(s) (Maharashtra)")
            
            # Find recent Open-Meteo files (Maharashtra districts)
            openmeteo_files = find_maharashtra_weather_files(data_dir, "openmeteo", limit=10)
            files_to_validate["openmeteo"].extend(openmeteo_files)
            logger.info(f"Found {len(openmeteo_files)} Open-Meteo file(s) (Maharashtra)")
        
        # Add explicitly specified files
        if args.mandi:
            files_to_validate["mandi"].extend([Path(p) for p in args.mandi])
        if args.power:
            files_to_validate["power"].extend([Path(p) for p in args.power])
        if args.openmeteo:
            files_to_validate["openmeteo"].extend([Path(p) for p in args.openmeteo])
        
        # Check if we have any files
        total_files = sum(len(v) for v in files_to_validate.values())
        if total_files == 0:
            logger.warning("No files to validate. Specify files or use --all-recent")
            print("\n⚠️  No files to validate.")
            print("Usage: python scripts/validate_data.py --help")
            return 0
        
        logger.info(f"Validating {total_files} file(s)")
        
        # Track Maharashtra-specific metrics
        total_non_mh_records = 0
        
        # Validate mandi files (with strict Maharashtra checks)
        for file_path in files_to_validate["mandi"]:
            logger.info(f"Validating Maharashtra mandi: {file_path}")
            result = validate_mandi(file_path, logger)
            results.append(result)
            total_non_mh_records += result.non_mh_count
            
            if result.non_mh_count > 0:
                logger.error(
                    f"HARD CONSTRAINT VIOLATION: {result.non_mh_count} "
                    f"non-Maharashtra records in {file_path.name}"
                )
                if audit:
                    audit.add_error(
                        f"Non-MH records in {file_path.name}: {result.non_mh_count}"
                    )
            
            if not args.summary_only:
                print(result)
        
        # Validate NASA POWER files
        for file_path in files_to_validate["power"]:
            logger.info(f"Validating NASA POWER: {file_path}")
            result = validate_power(file_path, logger)
            results.append(result)
            if not args.summary_only:
                print(result)
        
        # Validate Open-Meteo files
        for file_path in files_to_validate["openmeteo"]:
            logger.info(f"Validating Open-Meteo: {file_path}")
            result = validate_openmeteo(file_path, logger)
            results.append(result)
            if not args.summary_only:
                print(result)
        
        # Summary calculations
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count
        total_rows = sum(r.row_count for r in results)
        mandi_rows = sum(r.row_count for r in results if "Mandi" in r.data_type)
        
        # Print summary
        print("\n" + "=" * 60)
        print("MAHARASHTRA VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Target State: {MAHARASHTRA_STATE_NAME} (HARD CONSTRAINT)")
        print(f"Total files validated: {len(results)}")
        print(f"  ✅ Valid: {valid_count}")
        print(f"  ❌ Invalid: {invalid_count}")
        print(f"Total rows: {total_rows:,}")
        print(f"  Mandi rows: {mandi_rows:,}")
        
        if total_non_mh_records > 0:
            print(f"\n🚨 CRITICAL: {total_non_mh_records} non-Maharashtra records found!")
            print("   This is a HARD CONSTRAINT VIOLATION!")
        else:
            print(f"\n✅ Maharashtra-only constraint: PASSED")
        
        print("=" * 60)
        
        logger.info(f"Validation complete: {valid_count}/{len(results)} valid")
        
        # Generate audit report if requested
        if audit:
            audit.add_section("Summary", {
                "Total Files": str(len(results)),
                "Valid Files": str(valid_count),
                "Invalid Files": str(invalid_count),
                "Total Rows": f"{total_rows:,}",
                "Mandi Rows": f"{mandi_rows:,}",
                "Non-MH Records": str(total_non_mh_records),
            })
            
            if total_non_mh_records > 0:
                audit.add_error(
                    f"HARD CONSTRAINT VIOLATION: {total_non_mh_records} "
                    "non-Maharashtra records found"
                )
            
            # Save audit report
            audit_path = audit.save(PROJECT_ROOT / "logs")
            logger.info(f"Audit report saved: {audit_path}")
            print(f"\n📝 Audit report: {audit_path}")
        
        # Exit code - fail if strict mode and any invalid OR any non-MH records
        if total_non_mh_records > 0:
            logger.error("HARD CONSTRAINT VIOLATION: Non-Maharashtra data found!")
            return 2  # Special exit code for constraint violation
        
        if args.strict and invalid_count > 0:
            logger.error(f"{invalid_count} file(s) failed validation")
            return 1
        
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        if audit:
            audit.add_error(f"Unexpected error: {e}")
            audit.save(PROJECT_ROOT / "logs")
        return 99


if __name__ == "__main__":
    sys.exit(main())
