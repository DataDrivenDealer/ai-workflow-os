#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate G2 (Sanity Checks) Executable Validator

Checks:
- Unit tests pass (kernel/tests)
- Type checking (pyright)
- Lookahead detection (optional)

Exit codes:
  0 - All checks passed
  1 - Warnings only
  2 - Errors found
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass
class G2CheckResult:
    check_id: str
    status: str  # passed, warning, failed
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class G2GateReport:
    timestamp: str
    checks: List[G2CheckResult] = field(default_factory=list)
    passed: int = 0
    warnings: int = 0
    errors: int = 0

    @property
    def gate_passed(self) -> bool:
        return self.errors == 0

    def add(self, result: G2CheckResult) -> None:
        self.checks.append(result)
        if result.status == "passed":
            self.passed += 1
        elif result.status == "warning":
            self.warnings += 1
        else:
            self.errors += 1


def run_command(cmd: List[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        ok = result.returncode == 0
        output = (result.stdout or "") + (result.stderr or "")
        return ok, output.strip()
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gate G2 sanity checks")
    parser.add_argument("--output", "--format", dest="output", choices=["text", "json"], default="text")
    args = parser.parse_args()

    report = G2GateReport(timestamp=datetime.now(timezone.utc).isoformat())

    # 1) Unit tests
    ok, output = run_command([sys.executable, "-m", "pytest", "kernel/tests/", "-q"])
    report.add(G2CheckResult(
        check_id="unit_tests_pass",
        status="passed" if ok else "failed",
        message="Kernel unit tests" if ok else "Kernel unit tests failed",
        details={"output": output} if output else None,
    ))

    # 2) Type checking (pyright)
    ok, output = run_command([sys.executable, "-m", "pyright", "kernel", "scripts"])
    report.add(G2CheckResult(
        check_id="type_hints",
        status="passed" if ok else "warning",
        message="Pyright type check" if ok else "Pyright reported issues",
        details={"output": output} if output else None,
    ))

    # 3) Lookahead check (optional)
    lookahead_script = ROOT_DIR / "scripts" / "check_lookahead.py"
    code_dir = ROOT_DIR / "kernel"
    if lookahead_script.exists() and code_dir.exists():
        ok, output = run_command([sys.executable, str(lookahead_script), "--code-dir", str(code_dir)])
        report.add(G2CheckResult(
            check_id="no_lookahead",
            status="passed" if ok else "failed",
            message="Lookahead check" if ok else "Lookahead check failed",
            details={"output": output} if output else None,
        ))
    else:
        report.add(G2CheckResult(
            check_id="no_lookahead",
            status="warning",
            message="Lookahead check script not found",
        ))

    # Emit report
    if args.output == "json":
        print(json.dumps({
            "timestamp": report.timestamp,
            "passed": report.passed,
            "warnings": report.warnings,
            "errors": report.errors,
            "checks": [c.__dict__ for c in report.checks],
            "gate_passed": report.gate_passed,
        }, ensure_ascii=False, indent=2))
    else:
        print("Gate G2 - Sanity Checks")
        for c in report.checks:
            status = "✅" if c.status == "passed" else "⚠️" if c.status == "warning" else "❌"
            print(f"{status} {c.check_id}: {c.message}")
        print(f"Passed: {report.passed} | Warnings: {report.warnings} | Errors: {report.errors}")

    if report.errors > 0:
        return 2
    if report.warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
