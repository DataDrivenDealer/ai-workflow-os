#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate G6 (Release Readiness) Executable Validator

Validates presence of release readiness artifacts.

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
class G6CheckResult:
    check_id: str
    status: str  # passed, warning, failed
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class G6GateReport:
    timestamp: str
    checks: List[G6CheckResult] = field(default_factory=list)
    passed: int = 0
    warnings: int = 0
    errors: int = 0

    @property
    def gate_passed(self) -> bool:
        return self.errors == 0

    def add(self, result: G6CheckResult) -> None:
        self.checks.append(result)
        if result.status == "passed":
            self.passed += 1
        elif result.status == "warning":
            self.warnings += 1
        else:
            self.errors += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gate G6 release readiness checks")
    parser.add_argument("--output", "--format", dest="output", choices=["text", "json"], default="text")
    parser.add_argument("--release-notes", dest="release_notes", default="release_notes.txt")
    args = parser.parse_args()

    report = G6GateReport(timestamp=datetime.now(timezone.utc).isoformat())

    notes_path = Path(args.release_notes)
    if notes_path.exists():
        report.add(G6CheckResult(
            check_id="release_notes",
            status="passed",
            message=f"Found release notes: {notes_path}",
        ))
    else:
        report.add(G6CheckResult(
            check_id="release_notes",
            status="warning",
            message="Release notes missing (manual review required)",
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
        print("Gate G6 - Release Readiness")
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
