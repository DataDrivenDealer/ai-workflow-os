"""
File System Convention Checker
==============================
Validates file naming, location, and extension rules against
configs/file_system_governance.yaml.

Used by:
  - hooks/pre-commit (on staged files)
  - .github/workflows/ci.yaml (on all tracked files)
  - Manual: python scripts/check_file_conventions.py [--staged | --all]

Exit codes:
  0 = all checks passed
  1 = violations found (BLOCK)
  2 = warnings found but no blocking violations
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

# ============================================================================
# Rules (hardcoded from configs/file_system_governance.yaml for zero-dep use)
# ============================================================================

# Files that must NOT contain spaces
SPACES_CHECK_ENABLED = True

# .yml is prohibited (except GitHub issue templates)
YML_EXCEPTIONS = {
    ".github/ISSUE_TEMPLATE",
}

# test_*.py files must only be in these directory patterns
VALID_TEST_DIRS = re.compile(
    r"(^|.*/)tests/.*$"
    r"|^kernel/tests/.*$"
)

# Directories that must be snake_case
DIR_CHECK_EXCLUDES = {
    ".github",
    ".git",
    ".venv",
    "htmlcov",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "archive",     # legacy content preserved as-is
    ".worktrees",
}

# Known report directories
REPORT_DIRS = re.compile(r"(^|.*/)reports/")
DATE_PATTERN = re.compile(r"_\d{4}[-_]?\d{2}[-_]?\d{2}[._]")


def get_staged_files() -> list[str]:
    """Get list of staged files (relative paths, forward-slash)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_all_tracked_files() -> list[str]:
    """Get list of all tracked files that exist on disk (relative paths, forward-slash)."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []
    # Filter out files deleted from disk but still in git index
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and Path(line.strip()).exists()
    ]


class Violation:
    def __init__(self, path: str, rule: str, message: str, severity: str = "BLOCK"):
        self.path = path
        self.rule = rule
        self.message = message
        self.severity = severity

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}\n  Rule: {self.rule}\n  {self.message}"


def check_spaces(path: str) -> Violation | None:
    """Check for spaces in file names."""
    name = PurePosixPath(path).name
    if " " in name:
        return Violation(
            path, "no_spaces_in_filenames",
            f"File name contains spaces. Rename to: {name.replace(' ', '_')}",
        )
    return None


def check_yml_extension(path: str) -> Violation | None:
    """Check for .yml extension (should be .yaml)."""
    if not path.endswith(".yml"):
        return None
    # Check exceptions
    for exc in YML_EXCEPTIONS:
        if path.startswith(exc):
            return None
    return Violation(
        path, "no_yml_extension",
        "Use .yaml extension instead of .yml.",
    )


def check_test_location(path: str) -> Violation | None:
    """Check that test_*.py files are in tests/ directories."""
    name = PurePosixPath(path).name
    if not (name.startswith("test_") and name.endswith(".py")):
        return None
    if VALID_TEST_DIRS.match(path):
        return None
    # It's a test file NOT in a tests/ directory
    return Violation(
        path, "no_test_files_in_scripts",
        "Test files must be in a tests/ directory, not here.",
    )


def check_directory_naming(path: str) -> Violation | None:
    """Check that directory components are snake_case."""
    parts = PurePosixPath(path).parts[:-1]  # exclude filename
    for part in parts:
        if part in DIR_CHECK_EXCLUDES:
            continue
        if part.startswith("."):
            continue
        # Check for uppercase letters (excluding ISSUE_TEMPLATE which GitHub requires)
        if part == "ISSUE_TEMPLATE":
            continue
        if any(c.isupper() for c in part):
            return Violation(
                path, "no_mixed_case_directories",
                f"Directory '{part}' should be snake_case (all lowercase).",
                severity="WARN",
            )
    return None


def check_report_date(path: str) -> Violation | None:
    """Check that report files have date suffix."""
    if not REPORT_DIRS.match(path):
        return None
    name = PurePosixPath(path).name
    # Skip .gitkeep, README, __init__.py
    if name in {".gitkeep", "README.md", "__init__.py"}:
        return None
    if not DATE_PATTERN.search(name):
        return Violation(
            path, "no_undated_reports",
            "Report files should include a date suffix (_YYYYMMDD).",
            severity="WARN",
        )
    return None


def run_checks(files: list[str]) -> tuple[list[Violation], list[Violation]]:
    """Run all checks on given files. Returns (blockers, warnings)."""
    blockers: list[Violation] = []
    warnings: list[Violation] = []

    checks = [
        check_spaces,
        check_yml_extension,
        check_test_location,
        check_directory_naming,
        check_report_date,
    ]

    for filepath in files:
        for check_fn in checks:
            violation = check_fn(filepath)
            if violation is not None:
                if violation.severity == "BLOCK":
                    blockers.append(violation)
                else:
                    warnings.append(violation)

    return blockers, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Check file naming conventions")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="Check only staged files")
    group.add_argument("--all", action="store_true", help="Check all tracked files")
    group.add_argument("files", nargs="*", default=[], help="Specific files to check")
    args = parser.parse_args()

    if args.staged:
        files = get_staged_files()
        print(f"Checking {len(files)} staged files...")
    elif args.all:
        files = get_all_tracked_files()
        print(f"Checking {len(files)} tracked files...")
    elif args.files:
        files = args.files
        print(f"Checking {len(files)} specified files...")
    else:
        files = get_staged_files()
        print(f"Checking {len(files)} staged files...")

    if not files:
        print("[OK] No files to check.")
        return 0

    blockers, warnings = run_checks(files)

    if warnings:
        print(f"\n{'='*60}")
        print(f"WARNINGS ({len(warnings)}):")
        print(f"{'='*60}")
        for w in warnings:
            print(f"\n{w}")

    if blockers:
        print(f"\n{'='*60}")
        print(f"BLOCKED ({len(blockers)}):")
        print(f"{'='*60}")
        for b in blockers:
            print(f"\n{b}")
        print(f"\n[FAIL] {len(blockers)} blocking violation(s) found.")
        print("  See: configs/file_system_governance.yaml for the full rules.")
        return 1

    if warnings:
        print(f"\n[WARN] {len(warnings)} warning(s), 0 blockers.")
        return 2

    print("[OK] All file convention checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
