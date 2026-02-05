#!/usr/bin/env python3
"""
MANDIMITRA - Self-Check Validation Script

Validates that the codebase meets production-grade standards:
1. Security: No exposed API keys
2. Memory safety: No unbounded data structures
3. Code quality: Proper error handling
4. Maharashtra constraint: Hard-coded filtering

Usage:
    python scripts/self_check.py
    python scripts/self_check.py --verbose
    python scripts/self_check.py --fix  # Auto-fix some issues

Author: MANDIMITRA Team
Version: 1.0.0
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def check_mark(passed: bool) -> str:
    """Return colored check/cross mark."""
    if passed:
        return f"{Colors.GREEN}✓{Colors.RESET}"
    return f"{Colors.RED}✗{Colors.RESET}"


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# CHECK: Security
# =============================================================================

def check_no_exposed_api_keys() -> Tuple[bool, List[str]]:
    """Check that no API keys are exposed in code or config files."""
    issues = []
    
    # Patterns that indicate exposed secrets
    secret_patterns = [
        (r'api[-_]?key\s*[=:]\s*["\'][a-zA-Z0-9]{20,}["\']', "Hardcoded API key"),
        (r'DATAGOV_API_KEY\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "Exposed DATAGOV key"),
        (r'password\s*[=:]\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'secret\s*[=:]\s*["\'][a-zA-Z0-9]{16,}["\']', "Exposed secret"),
    ]
    
    # Files to scan (exclude .env which should not exist)
    scan_extensions = ['.py', '.yaml', '.yml', '.json', '.md', '.txt']
    exclude_dirs = ['.git', 'venv', '.venv', '__pycache__', 'node_modules', '.env']
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if not any(file.endswith(ext) for ext in scan_extensions):
                continue
            
            filepath = Path(root) / file
            
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, issue_type in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        rel_path = filepath.relative_to(PROJECT_ROOT)
                        issues.append(f"{rel_path}: {issue_type}")
                        
            except Exception:
                pass
    
    # Check .env file doesn't exist (should be in .gitignore)
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        content = env_file.read_text(encoding='utf-8', errors='ignore')
        if re.search(r'DATAGOV_API_KEY\s*=\s*[a-zA-Z0-9]{10,}', content):
            issues.append(".env contains actual API key (should use .env.example)")
    
    return len(issues) == 0, issues


def check_gitignore_secrets() -> Tuple[bool, List[str]]:
    """Check that .gitignore properly excludes secrets."""
    issues = []
    
    gitignore = PROJECT_ROOT / ".gitignore"
    if not gitignore.exists():
        issues.append(".gitignore file missing")
        return False, issues
    
    content = gitignore.read_text(encoding='utf-8')
    
    required_patterns = [
        '.env',
        '*.env',
        'secrets/',
        '**/secrets/**',
    ]
    
    for pattern in required_patterns:
        # Check if pattern or equivalent exists
        if pattern not in content and pattern.replace('/', '') not in content:
            issues.append(f".gitignore missing pattern: {pattern}")
    
    return len(issues) == 0, issues


# =============================================================================
# CHECK: Memory Safety
# =============================================================================

def check_no_unbounded_lists() -> Tuple[bool, List[str]]:
    """Check for patterns that could cause memory issues."""
    warnings = []
    
    # Patterns that suggest unbounded memory usage
    risky_patterns = [
        (r'all_records\s*=\s*\[\].*while.*all_records\.extend', "Unbounded list growth in pagination"),
        (r'\.append\(.*\).*while\s+True', "Unbounded append in infinite loop"),
        (r'records\s*\+=.*while', "Growing list in pagination loop"),
    ]
    
    # Files that are exempt (they use controlled memory patterns)
    exempt_files = {
        'download_mandi_maharashtra.py',  # Uses chunked by-district, bounded
        'download_weather_maharashtra.py',  # Per-location downloads, bounded
        'discover_maharashtra_mandi_metadata.py',  # Uses streaming/generators
        'http.py',  # Has streaming generator
        'http_utils.py',  # Legacy module (deprecated)
    }
    
    python_files = list(PROJECT_ROOT.glob("**/*.py"))
    
    for filepath in python_files:
        if '__pycache__' in str(filepath) or 'venv' in str(filepath):
            continue
        
        if filepath.name in exempt_files:
            continue
            
        try:
            content = filepath.read_text(encoding='utf-8')
            
            # Skip if using generators (stream_paginated_records)
            if 'yield' in content or 'stream_paginated' in content:
                continue
            
            for pattern, issue_type in risky_patterns:
                if re.search(pattern, content, re.DOTALL):
                    rel_path = filepath.relative_to(PROJECT_ROOT)
                    warnings.append(f"{rel_path}: {issue_type}")
                    
        except Exception:
            pass
    
    return len(warnings) == 0, warnings


def check_csv_comment_handling() -> Tuple[bool, List[str]]:
    """Check that CSV readers handle comment lines."""
    issues = []
    
    python_files = list(PROJECT_ROOT.glob("**/*.py"))
    
    for filepath in python_files:
        if '__pycache__' in str(filepath) or 'venv' in str(filepath):
            continue
            
        try:
            content = filepath.read_text(encoding='utf-8')
            
            # Find pd.read_csv calls
            csv_reads = re.findall(r'pd\.read_csv\([^)]+\)', content, re.DOTALL)
            
            for csv_call in csv_reads:
                # Skip if comment parameter is present
                if 'comment=' in csv_call or 'comment =' in csv_call:
                    continue
                
                # Only flag if reading config files
                if 'locations' in csv_call or 'config' in csv_call:
                    rel_path = filepath.relative_to(PROJECT_ROOT)
                    issues.append(f"{rel_path}: pd.read_csv without comment='#'")
                    
        except Exception:
            pass
    
    return len(issues) == 0, issues


# =============================================================================
# CHECK: Maharashtra Constraint
# =============================================================================

def check_maharashtra_hardcoded() -> Tuple[bool, List[str]]:
    """Check that Maharashtra is hardcoded, not parameterized."""
    warnings = []
    
    # Patterns that suggest state is parameterized (BAD)
    # Note: Match argument definitions, not documentation
    bad_patterns = [
        (r'add_argument\s*\([^)]*["\']--state["\']', "CLI has --state argument"),
        (r'args\.state\b(?!\s+argument)', "Code reads args.state"),
    ]
    
    # Files that should have hardcoded Maharashtra
    target_files = [
        PROJECT_ROOT / "scripts" / "download_mandi_maharashtra.py",
        PROJECT_ROOT / "scripts" / "download_weather_maharashtra.py",
        PROJECT_ROOT / "scripts" / "discover_maharashtra_mandi_metadata.py",
    ]
    
    for filepath in target_files:
        if not filepath.exists():
            warnings.append(f"{filepath.name}: File not found")
            continue
            
        try:
            content = filepath.read_text(encoding='utf-8')
            
            for pattern, issue_type in bad_patterns:
                if re.search(pattern, content):
                    warnings.append(f"{filepath.name}: {issue_type}")
            
            # Check for hardcoded constant usage (GOOD)
            if 'MAHARASHTRA_STATE_NAME' not in content:
                warnings.append(f"{filepath.name}: Not using MAHARASHTRA_STATE_NAME constant")
                
        except Exception as e:
            warnings.append(f"{filepath.name}: Read error - {e}")
    
    return len(warnings) == 0, warnings


# =============================================================================
# CHECK: Code Quality
# =============================================================================

def check_imports_organized() -> Tuple[bool, List[str]]:
    """Check that http imports use the new consolidated module."""
    warnings = []
    
    # Old imports that should be replaced
    old_import_pattern = r'from src\.utils\.http_utils import'
    
    python_files = list(PROJECT_ROOT.glob("**/*.py"))
    
    for filepath in python_files:
        if '__pycache__' in str(filepath) or 'venv' in str(filepath):
            continue
        
        # Skip this file (self_check.py) - it contains the pattern as a string
        if filepath.name == 'self_check.py':
            continue
            
        try:
            content = filepath.read_text(encoding='utf-8')
            
            if re.search(old_import_pattern, content):
                rel_path = filepath.relative_to(PROJECT_ROOT)
                warnings.append(f"{rel_path}: Uses old http_utils import (should use src.utils.http)")
                    
        except Exception:
            pass
    
    return len(warnings) == 0, warnings


def check_error_handling() -> Tuple[bool, List[str]]:
    """Check for bare except clauses."""
    warnings = []
    
    python_files = list(PROJECT_ROOT.glob("**/*.py"))
    
    for filepath in python_files:
        if '__pycache__' in str(filepath) or 'venv' in str(filepath):
            continue
        
        # Skip this file - it has intentional broad exception handling
        if filepath.name == 'self_check.py':
            continue
            
        try:
            content = filepath.read_text(encoding='utf-8')
            
            # Find bare except: clauses (not except Exception:)
            bare_excepts = re.findall(r'\bexcept\s*:', content)
            if bare_excepts:
                rel_path = filepath.relative_to(PROJECT_ROOT)
                warnings.append(f"{rel_path}: Has {len(bare_excepts)} bare except: clauses")
                    
        except Exception:
            pass
    
    return len(warnings) == 0, warnings


# =============================================================================
# CHECK: Rate Limiting
# =============================================================================

def check_rate_limiting() -> Tuple[bool, List[str]]:
    """Check that download scripts use rate limiting."""
    warnings = []
    
    download_scripts = [
        PROJECT_ROOT / "scripts" / "download_mandi_maharashtra.py",
        PROJECT_ROOT / "scripts" / "download_weather_maharashtra.py",
    ]
    
    for filepath in download_scripts:
        if not filepath.exists():
            continue
            
        try:
            content = filepath.read_text(encoding='utf-8')
            
            if 'AdaptiveRateLimiter' not in content and 'rate_limiter' not in content:
                warnings.append(f"{filepath.name}: No rate limiting implemented")
            
            if 'ThreadPoolExecutor' in content and 'rate_limiter' not in content:
                warnings.append(f"{filepath.name}: Parallel downloads without shared rate limiter")
                    
        except Exception:
            pass
    
    return len(warnings) == 0, warnings


# =============================================================================
# CHECK: Progress Tracking
# =============================================================================

def check_progress_tracking() -> Tuple[bool, List[str]]:
    """Check that download scripts use progress tracking with batched saves."""
    warnings = []
    
    download_scripts = [
        PROJECT_ROOT / "scripts" / "download_mandi_maharashtra.py",
        PROJECT_ROOT / "scripts" / "download_weather_maharashtra.py",
    ]
    
    for filepath in download_scripts:
        if not filepath.exists():
            continue
            
        try:
            content = filepath.read_text(encoding='utf-8')
            
            if 'ProgressTracker' not in content:
                warnings.append(f"{filepath.name}: No ProgressTracker")
            
            if 'tracker.flush()' not in content and '.flush()' not in content:
                warnings.append(f"{filepath.name}: No explicit flush (relies on atexit)")
                    
        except Exception:
            pass
    
    return len(warnings) == 0, warnings


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MANDIMITRA Self-Check Validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--fix", action="store_true", help="Auto-fix some issues (not implemented)")
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{'=' * 60}")
    print("MANDIMITRA - Self-Check Validation")
    print(f"{'=' * 60}{Colors.RESET}\n")
    
    all_passed = True
    total_issues = 0
    
    checks = [
        ("Security: No exposed API keys", check_no_exposed_api_keys),
        ("Security: .gitignore protects secrets", check_gitignore_secrets),
        ("Memory: No unbounded list growth", check_no_unbounded_lists),
        ("Memory: CSV comment handling", check_csv_comment_handling),
        ("Constraint: Maharashtra hardcoded", check_maharashtra_hardcoded),
        ("Quality: Using new http module", check_imports_organized),
        ("Quality: No bare except clauses", check_error_handling),
        ("Perf: Rate limiting in downloads", check_rate_limiting),
        ("Perf: Progress tracking", check_progress_tracking),
    ]
    
    for name, check_fn in checks:
        try:
            passed, issues = check_fn()
        except Exception as e:
            passed = False
            issues = [f"Check failed: {e}"]
        
        status = check_mark(passed)
        print(f"  {status} {name}")
        
        if not passed:
            all_passed = False
            total_issues += len(issues)
            
            if args.verbose and issues:
                for issue in issues:
                    print(f"      {Colors.YELLOW}→ {issue}{Colors.RESET}")
    
    print()
    print(f"{Colors.BOLD}{'=' * 60}")
    
    if all_passed:
        print(f"{Colors.GREEN}✓ All checks passed!{Colors.RESET}")
        print(f"{'=' * 60}{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}✗ {total_issues} issues found{Colors.RESET}")
        print(f"{Colors.YELLOW}Run with --verbose for details{Colors.RESET}")
        print(f"{'=' * 60}{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
