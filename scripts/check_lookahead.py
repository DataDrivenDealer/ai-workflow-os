"""
Lookahead Bias Detection Script

Detects potential lookahead (future data leakage) in:
- Feature/factor construction
- Signal generation
- Data pipelines

This script is referenced by configs/gates.yaml for Gate G1 (Data Quality)
and Gate G2 (Sanity Checks).

Usage:
    python scripts/check_lookahead.py --data-dir data/project/
    python scripts/check_lookahead.py --code-dir strategies/project/
    python scripts/check_lookahead.py --task-id DATA_2_XXX
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LookaheadViolation:
    """Represents a detected lookahead violation."""
    file_path: str
    line_number: int
    violation_type: str
    description: str
    severity: str = "error"  # error, warning, info
    suggestion: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line_number,
            "type": self.violation_type,
            "description": self.description,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class LookaheadCheckResult:
    """Result of lookahead check."""
    passed: bool
    violations: List[LookaheadViolation] = field(default_factory=list)
    files_checked: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "files_checked": self.files_checked,
            "violations": [v.to_dict() for v in self.violations],
            "timestamp": self.timestamp.isoformat(),
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            "# Lookahead Bias Check Report",
            "",
            f"**Status**: {status}",
            f"**Timestamp**: {self.timestamp.isoformat()}",
            f"**Files Checked**: {self.files_checked}",
            f"**Errors**: {self.error_count}",
            f"**Warnings**: {self.warning_count}",
            "",
        ]
        
        if self.violations:
            lines.append("## Violations")
            lines.append("")
            lines.append("| File | Line | Type | Severity | Description |")
            lines.append("|------|------|------|----------|-------------|")
            
            for v in self.violations:
                lines.append(
                    f"| `{v.file_path}` | {v.line_number} | {v.violation_type} | {v.severity} | {v.description} |"
                )
            lines.append("")
            
            # Detailed violations with suggestions
            lines.append("## Violation Details")
            for i, v in enumerate(self.violations, 1):
                lines.append(f"### {i}. {v.violation_type}")
                lines.append(f"- **File**: `{v.file_path}:{v.line_number}`")
                lines.append(f"- **Description**: {v.description}")
                if v.suggestion:
                    lines.append(f"- **Suggestion**: {v.suggestion}")
                lines.append("")
        else:
            lines.append("## Result")
            lines.append("")
            lines.append("No lookahead violations detected.")
        
        return "\n".join(lines)


# ===========================================================================
# Code-based Lookahead Detection
# ===========================================================================

# Patterns that commonly indicate lookahead bias
LOOKAHEAD_PATTERNS = {
    # Pandas operations that can cause lookahead
    "shift_negative": {
        "pattern": r"\.shift\s*\(\s*-\d+",
        "description": "Negative shift can access future data",
        "severity": "error",
        "suggestion": "Use positive shift values only: df.shift(1) for lag",
    },
    "rolling_center": {
        "pattern": r"\.rolling\s*\([^)]*center\s*=\s*True",
        "description": "Centered rolling window uses future data",
        "severity": "error",
        "suggestion": "Remove center=True or use center=False",
    },
    "ewm_adjust_false": {
        "pattern": r"\.ewm\s*\([^)]*adjust\s*=\s*False",
        "description": "EWM with adjust=False may cause issues at series start",
        "severity": "warning",
        "suggestion": "Consider using adjust=True for cleaner behavior",
    },
    "fillna_method_bfill": {
        "pattern": r"\.fillna\s*\([^)]*method\s*=\s*['\"]bfill['\"]",
        "description": "Backward fill uses future data",
        "severity": "error",
        "suggestion": "Use method='ffill' or specify a constant value",
    },
    "bfill_direct": {
        "pattern": r"\.bfill\s*\(",
        "description": "Backward fill uses future data",
        "severity": "error",
        "suggestion": "Use .ffill() instead",
    },
    "interpolate_backward": {
        "pattern": r"\.interpolate\s*\([^)]*method\s*=\s*['\"]backward",
        "description": "Backward interpolation uses future data",
        "severity": "error",
        "suggestion": "Use method='linear' with limit_direction='forward'",
    },
    # Scikit-learn fit on full data
    "fit_transform_full": {
        "pattern": r"fit_transform\s*\(\s*X\s*\)",
        "description": "fit_transform on full dataset may leak test info",
        "severity": "warning",
        "suggestion": "Use fit on train, transform on test separately",
    },
    # Future date access
    "future_date_access": {
        "pattern": r"df\[df\[['\"]date['\"]\]\s*>\s*current_date\]",
        "description": "Accessing data with future dates",
        "severity": "error",
        "suggestion": "Ensure date filtering only accesses past/current data",
    },
    # Index-based forward looking
    "iloc_future": {
        "pattern": r"\.iloc\s*\[\s*i\s*\+\s*\d+",
        "description": "iloc with positive offset may access future rows",
        "severity": "warning",
        "suggestion": "Verify this is intentional and not a data leak",
    },
}

# AST-based checks for function calls
DANGEROUS_FUNCTION_CALLS = {
    "resample": {
        "check_args": ["label", "closed"],
        "dangerous_values": {"label": "right", "closed": "right"},
        "description": "Resampling with right label/closed uses future data",
        "severity": "error",
    },
}


class LookaheadASTVisitor(ast.NodeVisitor):
    """AST visitor to detect lookahead patterns in Python code."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.violations: List[LookaheadViolation] = []
    
    def visit_Call(self, node: ast.Call) -> Any:
        """Check function calls for dangerous patterns."""
        # Check for method calls like df.shift(-1)
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            # Check shift with negative argument
            if method_name == "shift" and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.UnaryOp) and isinstance(first_arg.op, ast.USub):
                    self.violations.append(LookaheadViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        violation_type="negative_shift",
                        description="Negative shift accesses future data",
                        severity="error",
                        suggestion="Use positive shift: df.shift(n) where n > 0",
                    ))
                elif isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, (int, float)):
                    if first_arg.value < 0:
                        self.violations.append(LookaheadViolation(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            violation_type="negative_shift",
                            description=f"shift({first_arg.value}) accesses future data",
                            severity="error",
                            suggestion="Use positive shift: df.shift(n) where n > 0",
                        ))
            
            # Check for bfill
            if method_name in ("bfill", "backfill"):
                self.violations.append(LookaheadViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    violation_type="backward_fill",
                    description="Backward fill uses future data",
                    severity="error",
                    suggestion="Use .ffill() for forward fill",
                ))
            
            # Check rolling with center=True
            if method_name == "rolling":
                for keyword in node.keywords:
                    if keyword.arg == "center":
                        if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            self.violations.append(LookaheadViolation(
                                file_path=self.file_path,
                                line_number=node.lineno,
                                violation_type="centered_rolling",
                                description="Centered rolling window uses future data",
                                severity="error",
                                suggestion="Remove center=True or use center=False",
                            ))
        
        self.generic_visit(node)
        return None


def check_code_file(file_path: Path) -> List[LookaheadViolation]:
    """Check a single Python file for lookahead violations."""
    violations = []
    
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        # Regex-based pattern matching
        for pattern_name, pattern_info in LOOKAHEAD_PATTERNS.items():
            regex = re.compile(pattern_info["pattern"], re.IGNORECASE)
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    violations.append(LookaheadViolation(
                        file_path=str(file_path),
                        line_number=i,
                        violation_type=pattern_name,
                        description=pattern_info["description"],
                        severity=pattern_info["severity"],
                        suggestion=pattern_info.get("suggestion", ""),
                    ))
        
        # AST-based analysis (for .py files)
        if file_path.suffix == ".py":
            try:
                tree = ast.parse(content)
                visitor = LookaheadASTVisitor(str(file_path))
                visitor.visit(tree)
                violations.extend(visitor.violations)
            except SyntaxError:
                pass  # Skip files with syntax errors
    
    except Exception as e:
        print(f"Warning: Could not check {file_path}: {e}", file=sys.stderr)
    
    return violations


def check_code_directory(code_dir: Path) -> LookaheadCheckResult:
    """Check all Python files in a directory for lookahead violations."""
    violations = []
    files_checked = 0
    
    # Find all Python files
    py_files = list(code_dir.rglob("*.py"))
    
    for py_file in py_files:
        file_violations = check_code_file(py_file)
        violations.extend(file_violations)
        files_checked += 1
    
    # Check notebooks too
    ipynb_files = list(code_dir.rglob("*.ipynb"))
    for nb_file in ipynb_files:
        try:
            import json
            nb_content = json.loads(nb_file.read_text(encoding="utf-8"))
            for cell_idx, cell in enumerate(nb_content.get("cells", [])):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))
                    # Create temp path for reporting
                    temp_path = f"{nb_file}#cell{cell_idx}"
                    for pattern_name, pattern_info in LOOKAHEAD_PATTERNS.items():
                        regex = re.compile(pattern_info["pattern"], re.IGNORECASE)
                        for i, line in enumerate(source.split("\n"), 1):
                            if regex.search(line):
                                violations.append(LookaheadViolation(
                                    file_path=temp_path,
                                    line_number=i,
                                    violation_type=pattern_name,
                                    description=pattern_info["description"],
                                    severity=pattern_info["severity"],
                                    suggestion=pattern_info.get("suggestion", ""),
                                ))
            files_checked += 1
        except Exception:
            pass
    
    # Determine pass/fail based on errors
    error_count = sum(1 for v in violations if v.severity == "error")
    passed = error_count == 0
    
    return LookaheadCheckResult(
        passed=passed,
        violations=violations,
        files_checked=files_checked,
    )


# ===========================================================================
# Data-based Lookahead Detection
# ===========================================================================

def check_data_timestamps(data_dir: Path) -> List[LookaheadViolation]:
    """Check data files for timestamp issues that could indicate lookahead."""
    violations = []
    
    # Check for metadata files that might indicate data versioning
    metadata_files = list(data_dir.rglob("*.yaml")) + list(data_dir.rglob("*.yml"))
    
    for meta_file in metadata_files:
        try:
            content = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
            if content and isinstance(content, dict):
                # Check for snapshot dates
                snapshot_date = content.get("snapshot_date") or content.get("as_of_date")
                data_end_date = content.get("data_end_date") or content.get("end_date")
                
                if snapshot_date and data_end_date:
                    # If data_end_date is after snapshot_date, might be lookahead
                    if str(data_end_date) > str(snapshot_date):
                        violations.append(LookaheadViolation(
                            file_path=str(meta_file),
                            line_number=0,
                            violation_type="data_timestamp_mismatch",
                            description=f"Data end date ({data_end_date}) is after snapshot date ({snapshot_date})",
                            severity="error",
                            suggestion="Ensure data_end_date <= snapshot_date",
                        ))
        except Exception:
            pass
    
    return violations


def check_data_directory(data_dir: Path) -> LookaheadCheckResult:
    """Check data directory for lookahead issues."""
    violations = []
    files_checked = 0
    
    # Check timestamps
    violations.extend(check_data_timestamps(data_dir))
    
    # Count files checked
    files_checked = len(list(data_dir.rglob("*")))
    
    # Check for any Python files in data dir (processing scripts)
    py_files = list(data_dir.rglob("*.py"))
    for py_file in py_files:
        file_violations = check_code_file(py_file)
        violations.extend(file_violations)
    
    error_count = sum(1 for v in violations if v.severity == "error")
    passed = error_count == 0
    
    return LookaheadCheckResult(
        passed=passed,
        violations=violations,
        files_checked=files_checked,
    )


# ===========================================================================
# Task-based Check
# ===========================================================================

def check_task(task_id: str) -> LookaheadCheckResult:
    """Check a task for lookahead based on its type and associated directories."""
    all_violations = []
    total_files = 0
    
    # Determine what to check based on task prefix
    if task_id.startswith("DATA_"):
        # Check data directories
        data_dirs = [
            ROOT / "data",
            ROOT / "projects" / "dgsf" / "data",
        ]
        for data_dir in data_dirs:
            if data_dir.exists():
                result = check_data_directory(data_dir)
                all_violations.extend(result.violations)
                total_files += result.files_checked
    
    elif task_id.startswith("DEV_") or task_id.startswith("EVAL_"):
        # Check code directories
        code_dirs = [
            ROOT / "strategies",
            ROOT / "kernel",
            ROOT / "scripts",
        ]
        for code_dir in code_dirs:
            if code_dir.exists():
                result = check_code_directory(code_dir)
                all_violations.extend(result.violations)
                total_files += result.files_checked
    
    else:
        # Check everything
        for check_dir in [ROOT / "kernel", ROOT / "scripts"]:
            if check_dir.exists():
                result = check_code_directory(check_dir)
                all_violations.extend(result.violations)
                total_files += result.files_checked
    
    error_count = sum(1 for v in all_violations if v.severity == "error")
    passed = error_count == 0
    
    return LookaheadCheckResult(
        passed=passed,
        violations=all_violations,
        files_checked=total_files,
    )


# ===========================================================================
# CLI
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect lookahead bias in code and data"
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        help="Directory containing Python code to check",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing data files to check",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Task ID to check (auto-detects relevant directories)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for report (markdown)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json", "yaml"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    
    args = parser.parse_args()
    
    # Determine what to check
    result: Optional[LookaheadCheckResult] = None
    
    if args.task_id:
        result = check_task(args.task_id)
    elif args.code_dir:
        if not args.code_dir.exists():
            print(f"Error: Code directory not found: {args.code_dir}", file=sys.stderr)
            return 1
        result = check_code_directory(args.code_dir)
    elif args.data_dir:
        if not args.data_dir.exists():
            print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
            return 1
        result = check_data_directory(args.data_dir)
    else:
        # Default: check kernel and scripts
        result = check_code_directory(ROOT / "kernel")
        scripts_result = check_code_directory(ROOT / "scripts")
        result.violations.extend(scripts_result.violations)
        result.files_checked += scripts_result.files_checked
        result.passed = result.error_count == 0
    
    # Apply strict mode
    if args.strict and result.warning_count > 0:
        result.passed = False
    
    # Output results
    if args.format == "markdown":
        output = result.to_markdown()
    elif args.format == "json":
        import json
        output = json.dumps(result.to_dict(), indent=2)
    elif args.format == "yaml":
        output = yaml.safe_dump(result.to_dict(), sort_keys=False)
    else:
        # Text format
        status = "PASSED" if result.passed else "FAILED"
        lines = [
            f"Lookahead Check: {status}",
            f"Files checked: {result.files_checked}",
            f"Errors: {result.error_count}",
            f"Warnings: {result.warning_count}",
        ]
        if result.violations:
            lines.append("")
            lines.append("Violations:")
            for v in result.violations:
                lines.append(f"  [{v.severity.upper()}] {v.file_path}:{v.line_number}")
                lines.append(f"    {v.violation_type}: {v.description}")
                if v.suggestion:
                    lines.append(f"    Suggestion: {v.suggestion}")
        output = "\n".join(lines)
    
    # Write output
    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(output)
    
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
