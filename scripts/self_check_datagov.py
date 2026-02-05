#!/usr/bin/env python3
"""
MANDIMITRA - Data.gov.in API Self-Check Script

Comprehensive diagnostic tool to verify API connectivity and filter behavior.
Run this before any downloads to catch issues early.

Usage:
    python scripts/self_check_datagov.py
    python scripts/self_check_datagov.py --verbose
    python scripts/self_check_datagov.py --test-filter-only

Checks performed:
    1. API Key: Valid and present in .env
    2. Connectivity: Can reach data.gov.in API
    3. Rate Limiting: Not currently blocked (429)
    4. Filter Test: `filters[state.keyword]` returns ONLY Maharashtra
    5. Data Availability: Maharashtra records exist (total > 0)

Exit codes:
    0 = All checks passed
    1 = Some checks failed (non-critical)
    2 = Critical failure (cannot proceed with downloads)

Author: MANDIMITRA Team
Version: 1.0.0
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import load_config
from src.utils.http import (
    create_session,
    redact_url,
    redact_params,
)
from src.utils.maharashtra import (
    MAHARASHTRA_STATE_NAME,
    build_maharashtra_api_filters,
)
from src.utils.logging_utils import setup_logger


# =============================================================================
# CHECK RESULT STRUCTURE
# =============================================================================

@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, critical
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfCheckReport:
    """Complete self-check report."""
    timestamp_utc: str
    checks: List[CheckResult] = field(default_factory=list)
    all_passed: bool = False
    critical_failure: bool = False
    
    def add_check(self, check: CheckResult) -> None:
        self.checks.append(check)
        if not check.passed and check.severity == "critical":
            self.critical_failure = True
    
    def finalize(self) -> None:
        self.all_passed = all(c.passed for c in self.checks)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "all_passed": self.all_passed,
            "critical_failure": self.critical_failure,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data.gov.in API self-check and diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-filter-only",
        action="store_true",
        help="Only run the filter behavior test",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project configuration",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Save report to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()


# =============================================================================
# DIAGNOSTIC CHECKS
# =============================================================================

def check_api_key() -> CheckResult:
    """Check 1: API key is present and not placeholder."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DATAGOV_API_KEY")
    
    if not api_key:
        return CheckResult(
            name="API Key Present",
            passed=False,
            message="DATAGOV_API_KEY not found in .env",
            severity="critical",
        )
    
    if api_key == "your_api_key_here":
        return CheckResult(
            name="API Key Present",
            passed=False,
            message="DATAGOV_API_KEY is still placeholder value",
            severity="critical",
        )
    
    # Redact for display
    redacted = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    
    return CheckResult(
        name="API Key Present",
        passed=True,
        message=f"API key found: {redacted}",
        details={"key_length": len(api_key)},
    )


def check_connectivity(api_url: str, api_key: str, timeout: int = 15) -> CheckResult:
    """Check 2: Can connect to API (simple health check)."""
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 1,
    }
    
    try:
        start = time.time()
        response = requests.get(api_url, params=params, timeout=timeout)
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            return CheckResult(
                name="API Connectivity",
                passed=True,
                message=f"Connected to API ({latency:.0f}ms)",
                details={
                    "status_code": 200,
                    "latency_ms": round(latency, 1),
                },
            )
        elif response.status_code == 429:
            return CheckResult(
                name="API Connectivity",
                passed=False,
                message="Rate limited (429). Wait and retry.",
                severity="critical",
                details={"status_code": 429},
            )
        else:
            return CheckResult(
                name="API Connectivity",
                passed=False,
                message=f"HTTP {response.status_code}: {response.text[:100]}",
                severity="critical",
                details={"status_code": response.status_code},
            )
            
    except requests.exceptions.Timeout:
        return CheckResult(
            name="API Connectivity",
            passed=False,
            message=f"Connection timeout after {timeout}s",
            severity="critical",
        )
    except requests.exceptions.ConnectionError as e:
        return CheckResult(
            name="API Connectivity",
            passed=False,
            message=f"Connection error: {str(e)[:100]}",
            severity="critical",
        )
    except Exception as e:
        return CheckResult(
            name="API Connectivity",
            passed=False,
            message=f"Error: {str(e)[:100]}",
            severity="critical",
        )


def check_filter_behavior(api_url: str, api_key: str, verbose: bool = False) -> CheckResult:
    """
    Check 3: Verify filter[state.keyword] returns ONLY Maharashtra.
    
    This is THE critical test. It:
    1. Requests 10 records with filters[state.keyword]=Maharashtra
    2. Verifies ALL returned records have state="Maharashtra"
    3. Reports any filter leakage
    """
    # Use correct state.keyword filter
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 10,
    }
    params.update(build_maharashtra_api_filters())
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code != 200:
            return CheckResult(
                name="Filter Behavior (state.keyword)",
                passed=False,
                message=f"HTTP {response.status_code}",
                severity="critical",
                details={
                    "params_used": redact_params(params),
                    "status_code": response.status_code,
                },
            )
        
        data = response.json()
        total = data.get("total", 0)
        records = data.get("records", [])
        
        if total == 0:
            return CheckResult(
                name="Filter Behavior (state.keyword)",
                passed=False,
                message="API returned 0 Maharashtra records",
                severity="warning",
                details={
                    "total": 0,
                    "params_used": redact_params(params),
                },
            )
        
        # Check each record's state
        states_found = {}
        non_mh_records = []
        
        for record in records:
            state = record.get("state", "UNKNOWN")
            states_found[state] = states_found.get(state, 0) + 1
            
            if state.lower() != "maharashtra":
                non_mh_records.append({
                    "state": state,
                    "district": record.get("district", "?"),
                    "market": record.get("market", "?"),
                })
        
        if non_mh_records:
            return CheckResult(
                name="Filter Behavior (state.keyword)",
                passed=False,
                message=f"FILTER LEAKAGE: Got {len(non_mh_records)} non-MH records",
                severity="critical",
                details={
                    "total_claimed": total,
                    "records_checked": len(records),
                    "states_found": states_found,
                    "non_mh_samples": non_mh_records[:3],
                    "params_used": redact_params(params),
                },
            )
        
        # Success - all records are Maharashtra
        sample_districts = list(set(r.get("district", "") for r in records if r.get("district")))[:5]
        
        return CheckResult(
            name="Filter Behavior (state.keyword)",
            passed=True,
            message=f"All {len(records)} records are Maharashtra (total={total:,})",
            details={
                "total": total,
                "records_verified": len(records),
                "sample_districts": sample_districts,
                "params_used": redact_params(params),
            },
        )
        
    except Exception as e:
        return CheckResult(
            name="Filter Behavior (state.keyword)",
            passed=False,
            message=f"Error: {str(e)[:100]}",
            severity="critical",
        )


def check_wrong_filter_behavior(api_url: str, api_key: str) -> CheckResult:
    """
    Check 4: Demonstrate that filters[state] (without .keyword) is WRONG.
    
    This test uses the OLD incorrect filter to show the difference.
    Expected: May return non-Maharashtra records (proving why we need .keyword)
    """
    # Use WRONG filter (no .keyword) - DO NOT USE THIS IN PRODUCTION
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 10,
        "filters[state]": "Maharashtra",  # WRONG - fuzzy matching
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code != 200:
            return CheckResult(
                name="Wrong Filter Comparison (filters[state])",
                passed=True,
                message="Skipped (API error)",
                severity="info",
            )
        
        data = response.json()
        records = data.get("records", [])
        
        states_found = {}
        for record in records:
            state = record.get("state", "UNKNOWN")
            states_found[state] = states_found.get(state, 0) + 1
        
        non_mh_count = sum(c for s, c in states_found.items() if s.lower() != "maharashtra")
        
        if non_mh_count > 0:
            return CheckResult(
                name="Wrong Filter Comparison (filters[state])",
                passed=True,
                message=f"Confirmed: filters[state] leaks {non_mh_count} non-MH records",
                severity="info",
                details={
                    "states_found": states_found,
                    "note": "This proves why filters[state.keyword] is required",
                },
            )
        else:
            return CheckResult(
                name="Wrong Filter Comparison (filters[state])",
                passed=True,
                message="filters[state] happened to return only MH (may vary by API state)",
                severity="info",
                details={"states_found": states_found},
            )
            
    except Exception as e:
        return CheckResult(
            name="Wrong Filter Comparison",
            passed=True,
            message=f"Skipped: {str(e)[:50]}",
            severity="info",
        )


def check_data_availability(api_url: str, api_key: str) -> CheckResult:
    """Check 5: Maharashtra data is actually available (total > threshold)."""
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": 1,
    }
    params.update(build_maharashtra_api_filters())
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code != 200:
            return CheckResult(
                name="Data Availability",
                passed=False,
                message=f"HTTP {response.status_code}",
                severity="warning",
            )
        
        data = response.json()
        total = data.get("total", 0)
        
        if total == 0:
            return CheckResult(
                name="Data Availability",
                passed=False,
                message="0 Maharashtra records available",
                severity="warning",
                details={"total": 0},
            )
        
        if total < 1000:
            return CheckResult(
                name="Data Availability",
                passed=True,
                message=f"Low record count: {total:,} (may be stale data)",
                severity="warning",
                details={"total": total},
            )
        
        return CheckResult(
            name="Data Availability",
            passed=True,
            message=f"{total:,} Maharashtra records available",
            details={"total": total},
        )
        
    except Exception as e:
        return CheckResult(
            name="Data Availability",
            passed=False,
            message=f"Error: {str(e)[:100]}",
            severity="warning",
        )


# =============================================================================
# MAIN
# =============================================================================

def print_report(report: SelfCheckReport) -> None:
    """Print formatted report to console."""
    print("\n" + "=" * 60)
    print("MANDIMITRA - Data.gov.in Self-Check Report")
    print(f"Timestamp: {report.timestamp_utc}")
    print("=" * 60 + "\n")
    
    for check in report.checks:
        icon = "✅" if check.passed else "❌"
        severity_icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🔴"}.get(check.severity, "")
        
        print(f"{icon} {check.name}")
        print(f"   {check.message}")
        
        if check.details and not check.passed:
            for key, value in check.details.items():
                if key != "params_used":
                    print(f"   - {key}: {value}")
        
        print()
    
    print("-" * 60)
    
    if report.all_passed:
        print("🎉 ALL CHECKS PASSED - Ready to download Maharashtra data")
    elif report.critical_failure:
        print("🔴 CRITICAL FAILURE - Cannot proceed with downloads")
        print("   Fix the issues above before running download scripts")
    else:
        print("⚠️  SOME CHECKS FAILED - Review warnings above")
    
    print("-" * 60 + "\n")


def main() -> int:
    args = parse_arguments()
    
    # Load config
    try:
        config = load_config(PROJECT_ROOT / args.config)
        mandi_config = config["mandi"]
        api_url = f"{mandi_config['api_base']}/{mandi_config['resource_id']}"
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 2
    
    # Initialize report
    report = SelfCheckReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    
    # Run checks
    print("\n🔍 Running Data.gov.in API self-check...\n")
    
    # Check 1: API Key
    if not args.test_filter_only:
        check = check_api_key()
        report.add_check(check)
        print(f"  {'✓' if check.passed else '✗'} {check.name}")
        
        if not check.passed:
            print(f"    → {check.message}")
            report.finalize()
            print_report(report)
            return 2
        
        api_key = os.getenv("DATAGOV_API_KEY")
    else:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("DATAGOV_API_KEY")
    
    # Check 2: Connectivity
    if not args.test_filter_only:
        check = check_connectivity(api_url, api_key)
        report.add_check(check)
        print(f"  {'✓' if check.passed else '✗'} {check.name}")
        
        if not check.passed:
            print(f"    → {check.message}")
            report.finalize()
            print_report(report)
            return 2
    
    # Check 3: Filter behavior (CRITICAL)
    check = check_filter_behavior(api_url, api_key, verbose=args.verbose)
    report.add_check(check)
    print(f"  {'✓' if check.passed else '✗'} {check.name}")
    
    if args.verbose and check.details:
        for key, value in check.details.items():
            if key not in ("params_used",):
                print(f"    - {key}: {value}")
    
    # Check 4: Compare with wrong filter (informational)
    if not args.test_filter_only:
        check = check_wrong_filter_behavior(api_url, api_key)
        report.add_check(check)
        if args.verbose:
            print(f"  {'✓' if check.passed else '✗'} {check.name}")
    
    # Check 5: Data availability
    if not args.test_filter_only:
        check = check_data_availability(api_url, api_key)
        report.add_check(check)
        print(f"  {'✓' if check.passed else '✗'} {check.name}")
    
    # Finalize
    report.finalize()
    
    # Save JSON if requested
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n  📄 Report saved: {output_path}")
    
    # Print full report
    print_report(report)
    
    # Exit code
    if report.critical_failure:
        return 2
    elif not report.all_passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
