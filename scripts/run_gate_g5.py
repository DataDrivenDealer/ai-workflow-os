#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate G5 (Code Review) Executable Validator

Checks task review status from state/tasks.yaml.

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

import yaml


@dataclass
class G5CheckResult:
    check_id: str
    status: str  # passed, warning, failed
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class G5GateReport:
    timestamp: str
    checks: List[G5CheckResult] = field(default_factory=list)
    passed: int = 0
    warnings: int = 0
    errors: int = 0

    @property
    def gate_passed(self) -> bool:
        return self.errors == 0

    def add(self, result: G5CheckResult) -> None:
        self.checks.append(result)
        if result.status == "passed":
            self.passed += 1
        elif result.status == "warning":
            self.warnings += 1
        else:
            self.errors += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gate G5 code review checks")
    parser.add_argument("--output", "--format", dest="output", choices=["text", "json"], default="text")
    parser.add_argument("--tasks", dest="tasks_path", default="state/tasks.yaml")
    args = parser.parse_args()

    report = G5GateReport(timestamp=datetime.now(timezone.utc).isoformat())

    tasks_path = Path(args.tasks_path)
    if not tasks_path.exists():
        report.add(G5CheckResult(
            check_id="code_review_state",
            status="warning",
            message="tasks.yaml not found (manual review required)",
        ))
    else:
        data = yaml.safe_load(tasks_path.read_text(encoding="utf-8")) or {}
        reviews = data.get("reviews", {})
        pending = []
        for task_id, review_data in reviews.items():
            if review_data.get("status") == "pending_review":
                pending.append(task_id)

        if pending:
            report.add(G5CheckResult(
                check_id="code_review_state",
                status="failed",
                message=f"Pending reviews detected: {', '.join(pending)}",
                details={"pending": pending},
            ))
        else:
            report.add(G5CheckResult(
                check_id="code_review_state",
                status="passed",
                message="No pending code reviews",
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
        print("Gate G5 - Code Review")
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
