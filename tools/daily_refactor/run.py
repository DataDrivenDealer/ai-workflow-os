"""
Daily Refactor Runner
=====================

Automated code cleanup and refactoring for files changed since last marker.

Usage:
    python tools/daily_refactor/run.py [OPTIONS]

Options:
    --since MARKER      Base for comparison (default: HEAD~1)
    --output-dir DIR    Output directory for reports
    --apply             Actually apply changes (default: dry-run)
    --safe-only         Only apply safe transformations
    --include-risky     Include risky transformations (requires confirmation)
    --commit            Auto-commit after successful refactor
    --verbose           Verbose output

Examples:
    # Dry-run, show what would change
    python tools/daily_refactor/run.py

    # Apply safe changes only
    python tools/daily_refactor/run.py --apply --safe-only

    # Full refactor with commit
    python tools/daily_refactor/run.py --apply --commit
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import re


# =============================================================================
# Configuration
# =============================================================================

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


CONFIG = load_config()


# =============================================================================
# Git Operations
# =============================================================================

def run_command(cmd: List[str], check: bool = True, capture: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            encoding="utf-8",
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"


def get_changed_files(base_ref: str = "HEAD~1") -> List[Path]:
    """Get list of changed Python files since base_ref."""
    code, stdout, _ = run_command(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", base_ref, "--", "*.py"],
        check=False
    )
    if code != 0:
        return []
    
    files = [Path(f.strip()) for f in stdout.strip().split("\n") if f.strip()]
    
    # Apply include/exclude patterns
    scope = CONFIG.get("scope", {})
    include_patterns = scope.get("include_patterns", ["**/*.py"])
    exclude_patterns = scope.get("exclude_patterns", [])
    
    filtered = []
    for f in files:
        # Check if matches any include pattern
        included = any(f.match(p) for p in include_patterns)
        # Check if matches any exclude pattern
        excluded = any(f.match(p) for p in exclude_patterns)
        
        if included and not excluded and f.exists():
            filtered.append(f)
    
    return filtered


def get_diff_stat(files: List[Path]) -> str:
    """Get diffstat for changed files."""
    if not files:
        return "No changes"
    
    code, stdout, _ = run_command(
        ["git", "diff", "--stat"] + [str(f) for f in files],
        check=False
    )
    return stdout if code == 0 else "Unable to get diffstat"


# =============================================================================
# Transformations
# =============================================================================

class RefactorResult:
    """Result of a refactor operation."""
    
    def __init__(self):
        self.files_processed: List[Path] = []
        self.files_changed: List[Path] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.tool_outputs: Dict[str, str] = {}
    
    def add_changed(self, path: Path):
        if path not in self.files_changed:
            self.files_changed.append(path)
    
    def add_error(self, msg: str):
        self.errors.append(msg)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)


def run_black(files: List[Path], dry_run: bool = True) -> Tuple[bool, str]:
    """Run black formatter."""
    if not files:
        return True, "No files to format"
    
    cmd = ["python", "-m", "black"]
    if dry_run:
        cmd.append("--check")
        cmd.append("--diff")
    cmd.extend(str(f) for f in files)
    
    code, stdout, stderr = run_command(cmd, check=False)
    output = stdout + stderr
    
    return code == 0, output


def run_isort(files: List[Path], dry_run: bool = True) -> Tuple[bool, str]:
    """Run isort import sorter."""
    if not files:
        return True, "No files to sort"
    
    cmd = ["python", "-m", "isort"]
    if dry_run:
        cmd.append("--check-only")
        cmd.append("--diff")
    cmd.append("--profile=black")
    cmd.extend(str(f) for f in files)
    
    code, stdout, stderr = run_command(cmd, check=False)
    output = stdout + stderr
    
    return code == 0, output


def run_ruff_fix(files: List[Path], dry_run: bool = True, safe_only: bool = True) -> Tuple[bool, str]:
    """Run ruff with auto-fix."""
    if not files:
        return True, "No files to lint"
    
    cmd = ["python", "-m", "ruff", "check"]
    if not dry_run:
        cmd.append("--fix")
    
    # Select rules based on safety level
    if safe_only:
        cmd.extend(["--select", "F401,W291,W292,W293,I"])
    else:
        cmd.extend(["--select", "F401,F841,W291,W292,W293,I"])
    
    cmd.extend(str(f) for f in files)
    
    code, stdout, stderr = run_command(cmd, check=False)
    output = stdout + stderr
    
    return code == 0, output


def run_pyright(files: List[Path]) -> Tuple[bool, str]:
    """Run pyright type checker (report only)."""
    if not files:
        return True, "No files to check"
    
    cmd = ["pyright", "--outputjson"]
    cmd.extend(str(f) for f in files)
    
    code, stdout, stderr = run_command(cmd, check=False)
    
    # Parse JSON output if available
    try:
        import json
        data = json.loads(stdout)
        error_count = data.get("summary", {}).get("errorCount", 0)
        warning_count = data.get("summary", {}).get("warningCount", 0)
        output = f"Pyright: {error_count} errors, {warning_count} warnings"
    except (json.JSONDecodeError, KeyError):
        output = stderr or stdout
    
    # Pyright is report-only, always "success" for our purposes
    return True, output


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    result: RefactorResult,
    files: List[Path],
    dry_run: bool,
    output_dir: Path,
) -> None:
    """Generate refactor report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # REPORT.md
    report_content = f"""# Daily Refactor Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Mode**: {'DRY-RUN' if dry_run else 'APPLIED'}
**Files Scanned**: {len(files)}
**Files Changed**: {len(result.files_changed)}

## Summary

| Metric | Value |
|--------|-------|
| Total files processed | {len(result.files_processed)} |
| Files with changes | {len(result.files_changed)} |
| Errors | {len(result.errors)} |
| Warnings | {len(result.warnings)} |

## Tool Outputs

"""
    
    for tool, output in result.tool_outputs.items():
        report_content += f"### {tool}\n\n```\n{output[:2000]}{'...' if len(output) > 2000 else ''}\n```\n\n"
    
    if result.errors:
        report_content += "## Errors\n\n"
        for err in result.errors:
            report_content += f"- {err}\n"
        report_content += "\n"
    
    if result.warnings:
        report_content += "## Warnings\n\n"
        for warn in result.warnings:
            report_content += f"- {warn}\n"
        report_content += "\n"
    
    report_content += f"""## Files Changed

"""
    for f in result.files_changed:
        report_content += f"- [{f}]({f})\n"
    
    (output_dir / "REPORT.md").write_text(report_content, encoding="utf-8")
    
    # DIFFSTAT.txt
    diffstat = get_diff_stat(result.files_changed) if not dry_run else "Dry-run mode - no actual changes"
    (output_dir / "DIFFSTAT.txt").write_text(diffstat, encoding="utf-8")
    
    # RISKS.md
    risks_content = f"""# Refactor Risks Assessment

**Date**: {timestamp}

## Potential Risks

| Risk | Level | Description |
|------|-------|-------------|
"""
    
    if len(result.files_changed) > 10:
        risks_content += "| Large changeset | MEDIUM | Over 10 files changed, review carefully |\n"
    
    if any("test" in str(f).lower() for f in result.files_changed):
        risks_content += "| Test files modified | LOW | Test files were included in refactor |\n"
    
    if result.errors:
        risks_content += f"| Tool errors | HIGH | {len(result.errors)} errors during refactor |\n"
    
    if not dry_run:
        risks_content += "| Applied changes | MEDIUM | Changes were applied, verify tests pass |\n"
    
    risks_content += """
## Recommended Actions

1. Review REPORT.md for detailed changes
2. Run tests to verify no regressions: `pytest kernel/tests -x`
3. If issues found, revert with: `git checkout -- .`
"""
    
    (output_dir / "RISKS.md").write_text(risks_content, encoding="utf-8")
    
    print(f"\n📄 Reports generated in: {output_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Daily Refactor Runner - Automated code cleanup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--since",
        default=CONFIG.get("scope", {}).get("base_ref", "HEAD~1"),
        help="Base reference for detecting changes (default: HEAD~1)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default: dry-run)"
    )
    parser.add_argument(
        "--safe-only",
        action="store_true",
        help="Only apply safe transformations"
    )
    parser.add_argument(
        "--include-risky",
        action="store_true",
        help="Include risky transformations"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Auto-commit after successful refactor"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    dry_run = not args.apply
    safe_only = args.safe_only or (not args.include_risky)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_dir = Path(CONFIG.get("output", {}).get("base_dir", "docs/refactor"))
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_dir = base_dir / date_str
    
    print("=" * 60)
    print("Daily Refactor Runner")
    print("=" * 60)
    print(f"Mode: {'DRY-RUN' if dry_run else 'APPLY'}")
    print(f"Safety: {'SAFE-ONLY' if safe_only else 'MODERATE'}")
    print(f"Base: {args.since}")
    print()
    
    # Get changed files
    print("📂 Detecting changed files...")
    files = get_changed_files(args.since)
    
    if not files:
        print("✅ No files changed since", args.since)
        return 0
    
    print(f"Found {len(files)} file(s) to process:")
    for f in files[:10]:
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
    print()
    
    # Run transformations
    result = RefactorResult()
    result.files_processed = files
    
    print("🔧 Running transformations...")
    
    # Black
    print("  [1/4] Black (formatter)...")
    success, output = run_black(files, dry_run)
    result.tool_outputs["black"] = output
    if not success and not dry_run:
        result.add_warning("Black formatting had issues")
    if args.verbose:
        print(output[:500])
    
    # isort
    print("  [2/4] isort (import sorter)...")
    success, output = run_isort(files, dry_run)
    result.tool_outputs["isort"] = output
    if args.verbose:
        print(output[:500])
    
    # ruff
    print("  [3/4] Ruff (linter + fix)...")
    success, output = run_ruff_fix(files, dry_run, safe_only)
    result.tool_outputs["ruff"] = output
    if args.verbose:
        print(output[:500])
    
    # pyright (report only)
    print("  [4/4] Pyright (type check, report only)...")
    success, output = run_pyright(files)
    result.tool_outputs["pyright"] = output
    if args.verbose:
        print(output[:500])
    
    print()
    
    # Detect which files actually changed (if applied)
    if not dry_run:
        code, stdout, _ = run_command(["git", "diff", "--name-only"], check=False)
        if code == 0:
            for line in stdout.strip().split("\n"):
                if line.strip():
                    result.add_changed(Path(line.strip()))
    else:
        # In dry-run, mark all files as potentially changed
        result.files_changed = files.copy()
    
    # Generate reports
    generate_report(result, files, dry_run, output_dir)
    
    # Create marker file
    if not dry_run and CONFIG.get("git", {}).get("create_marker", True):
        marker_file = Path(CONFIG.get("git", {}).get("marker_file", ".refactor_marker"))
        marker_file.write_text(datetime.now().isoformat(), encoding="utf-8")
        print(f"📌 Created marker: {marker_file}")
    
    # Auto-commit if requested
    if not dry_run and args.commit and result.files_changed:
        print("\n📦 Committing changes...")
        file_count = len(result.files_changed)
        date_str = datetime.now().strftime("%Y-%m-%d")
        msg = f"refactor(daily): {date_str} auto-cleanup ({file_count} files)"
        
        run_command(["git", "add"] + [str(f) for f in result.files_changed], check=False)
        run_command(["git", "commit", "-m", msg], check=False)
        print(f"✅ Committed: {msg}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files processed: {len(result.files_processed)}")
    print(f"Files changed: {len(result.files_changed)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Reports: {output_dir}")
    
    if dry_run:
        print("\n💡 This was a DRY-RUN. To apply changes, run with --apply")
    
    return 0 if not result.errors else 1


if __name__ == "__main__":
    sys.exit(main())
