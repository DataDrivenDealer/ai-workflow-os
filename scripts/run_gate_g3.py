#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate G3 (Performance & Robustness) Executable Validator

This gate requires domain-specific evaluation artifacts. When artifacts
are missing, checks are marked as warnings and require manual review.

Exit codes:
  0 - All checks passed
  1 - Warnings only
  2 - Errors found
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class G3CheckResult:
    check_id: str
    status: str  # passed, warning, failed
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class G3GateReport:
    timestamp: str
    checks: List[G3CheckResult] = field(default_factory=list)
    passed: int = 0
    warnings: int = 0
    errors: int = 0

    @property
    def gate_passed(self) -> bool:
        return self.errors == 0

    def add(self, result: G3CheckResult) -> None:
        self.checks.append(result)
        if result.status == "passed":
            self.passed += 1
        elif result.status == "warning":
            self.warnings += 1
        else:
            self.errors += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gate G3 performance checks")
    parser.add_argument("--output", "--format", dest="output", choices=["text", "json"], default="text")
    parser.add_argument("--report", dest="report_path", default="reports/performance_report.json")
    args = parser.parse_args()

    report = G3GateReport(timestamp=datetime.now(timezone.utc).isoformat())

    report_path = Path(args.report_path)
    if report_path.exists():
        report.add(G3CheckResult(
            check_id="performance_report",
            status="passed",
            message=f"Found performance report: {report_path}",
        ))
    else:
        report.add(G3CheckResult(
            check_id="performance_report",
            status="warning",
            message="Performance report missing (manual review required)",
        ))

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
        print("Gate G3 - Performance & Robustness")
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
