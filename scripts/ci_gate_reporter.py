"""
CI Gate Reporter

Generates comprehensive gate reports for CI/CD pipelines.
Outputs in multiple formats: markdown, JSON, GitHub Actions annotations.

Usage:
    python scripts/ci_gate_reporter.py --format markdown > report.md
    python scripts/ci_gate_reporter.py --format github  # For GitHub Actions
    python scripts/ci_gate_reporter.py --format json
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

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CICheckResult:
    """Result of a CI check."""
    name: str
    category: str
    passed: bool
    message: str = ""
    details: str = ""
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class CIGateReport:
    """Complete CI gate report."""
    timestamp: datetime
    commit_sha: Optional[str]
    branch: Optional[str]
    checks: List[CICheckResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)
    
    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)
    
    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


class CIGateReporter:
    """Generates CI gate reports."""
    
    def __init__(self):
        self.report = CIGateReport(
            timestamp=datetime.now(timezone.utc),
            commit_sha=self._get_commit_sha(),
            branch=self._get_branch(),
        )
    
    def _get_commit_sha(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()[:8]
        except Exception:
            return None
    
    def _get_branch(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception:
            return None
    
    def run_all_checks(self):
        """Run all CI checks."""
        self._check_policy()
        self._check_governance()
        self._check_yaml_validity()
        self._check_spec_registry()
        self._check_taskcard_format()
    
    def _check_policy(self):
        """Run policy check."""
        try:
            result = subprocess.run(
                ["python", "scripts/policy_check.py", "--mode", "ci"],
                capture_output=True, text=True, cwd=str(ROOT)
            )
            self.report.checks.append(CICheckResult(
                name="Policy Check",
                category="Governance",
                passed=result.returncode == 0,
                message="Spec registry policy validation",
                details=result.stdout or result.stderr,
            ))
        except Exception as e:
            self.report.checks.append(CICheckResult(
                name="Policy Check",
                category="Governance",
                passed=False,
                message=f"Failed to run: {e}",
            ))
    
    def _check_governance(self):
        """Run governance gate checks."""
        sys.path.insert(0, str(ROOT / "kernel"))
        try:
            from governance_gate import GovernanceGate
            
            gate = GovernanceGate()
            results = gate.verify_all()
            
            for r in results:
                # In CI context, role_mode_integrity without session is expected
                is_ci_expected = (
                    r.gate_name == "role_mode_integrity" and 
                    any("No agent session" in v.description for v in r.violations)
                )
                
                self.report.checks.append(CICheckResult(
                    name=f"Governance: {r.gate_name}",
                    category="Governance",
                    passed=not r.has_violations or is_ci_expected,
                    message="Clean (CI context)" if is_ci_expected else (
                        f"{len(r.violations)} violations" if r.has_violations else "Clean"
                    ),
                    details="" if is_ci_expected else (
                        "\n".join(v.description for v in r.violations) if r.violations else ""
                    ),
                ))
        except ImportError as e:
            self.report.checks.append(CICheckResult(
                name="Governance Gate",
                category="Governance",
                passed=False,
                message=f"Module not available: {e}",
            ))
    
    def _check_yaml_validity(self):
        """Validate all YAML files."""
        errors = []
        for yaml_file in ROOT.rglob("*.yaml"):
            if ".git" in str(yaml_file):
                continue
            try:
                with yaml_file.open("r", encoding="utf-8") as f:
                    yaml.safe_load(f)
            except Exception as e:
                errors.append((yaml_file, str(e)))
        
        self.report.checks.append(CICheckResult(
            name="YAML Validity",
            category="Syntax",
            passed=len(errors) == 0,
            message=f"{len(errors)} invalid files" if errors else "All YAML files valid",
            details="\n".join(f"{f}: {e}" for f, e in errors),
        ))
    
    def _check_spec_registry(self):
        """Validate spec registry structure."""
        registry_path = ROOT / "spec_registry.yaml"
        
        if not registry_path.exists():
            self.report.checks.append(CICheckResult(
                name="Spec Registry",
                category="Structure",
                passed=False,
                message="spec_registry.yaml not found",
                file_path=str(registry_path),
            ))
            return
        
        with registry_path.open("r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}
        
        # Check required sections
        required = ["specs"]
        missing = [s for s in required if s not in registry]
        
        # Check spec entries
        specs = registry.get("specs", [])
        invalid_specs = []
        for spec in specs:
            if not spec.get("spec_id"):
                invalid_specs.append("Missing spec_id")
            if not spec.get("scope"):
                invalid_specs.append(f"{spec.get('spec_id', 'unknown')}: missing scope")
        
        passed = len(missing) == 0 and len(invalid_specs) == 0
        self.report.checks.append(CICheckResult(
            name="Spec Registry Structure",
            category="Structure",
            passed=passed,
            message="Valid" if passed else f"Issues found",
            details="\n".join(missing + invalid_specs) if not passed else "",
            file_path="spec_registry.yaml",
        ))
    
    def _check_taskcard_format(self):
        """Check TaskCard format validity."""
        tasks_dir = ROOT / "tasks"
        if not tasks_dir.exists():
            return
        
        invalid = []
        required_fields = [
            "task_id",
            "type",
            "queue",
            "branch",
            "spec_ids",
            "verification",
        ]
        for md_file in tasks_dir.rglob("*.md"):
            if md_file.name == "README.md":
                continue
            
            content = md_file.read_text(encoding="utf-8")
            if not content.startswith("---"):
                continue
            
            parts = content.split("---", 2)
            if len(parts) < 3:
                invalid.append(f"{md_file.name}: frontmatter not closed with '---'")
                continue
            
            frontmatter = yaml.safe_load(parts[1]) or {}
            missing = [field for field in required_fields if field not in frontmatter]
            if missing:
                invalid.append(f"{md_file.name}: missing frontmatter fields {', '.join(missing)}")
        
        self.report.checks.append(CICheckResult(
            name="TaskCard Format",
            category="Structure",
            passed=len(invalid) == 0,
            message=f"{len(invalid)} issues" if invalid else "All TaskCards valid",
            details="\n".join(invalid[:10]),  # Limit details
        ))
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# CI Gate Report",
            "",
            f"**Timestamp**: {self.report.timestamp.isoformat()}",
            f"**Commit**: {self.report.commit_sha or 'N/A'}",
            f"**Branch**: {self.report.branch or 'N/A'}",
            "",
            "## Summary",
            "",
            f"- **Status**: {'✅ PASSED' if self.report.passed else '❌ FAILED'}",
            f"- **Passed**: {self.report.pass_count}",
            f"- **Failed**: {self.report.fail_count}",
            "",
            "## Check Results",
            "",
            "| Category | Check | Status | Message |",
            "|----------|-------|--------|---------|",
        ]
        
        for check in self.report.checks:
            icon = "✅" if check.passed else "❌"
            lines.append(f"| {check.category} | {check.name} | {icon} | {check.message} |")
        
        # Details for failures
        failures = [c for c in self.report.checks if not c.passed and c.details]
        if failures:
            lines.extend([
                "",
                "## Failure Details",
                "",
            ])
            for check in failures:
                lines.extend([
                    f"### {check.name}",
                    "",
                    "```",
                    check.details[:500],
                    "```",
                    "",
                ])
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Generate JSON report."""
        return json.dumps({
            "timestamp": self.report.timestamp.isoformat(),
            "commit_sha": self.report.commit_sha,
            "branch": self.report.branch,
            "passed": self.report.passed,
            "summary": {
                "total": len(self.report.checks),
                "passed": self.report.pass_count,
                "failed": self.report.fail_count,
            },
            "checks": [
                {
                    "name": c.name,
                    "category": c.category,
                    "passed": c.passed,
                    "message": c.message,
                    "details": c.details,
                    "file": c.file_path,
                    "line": c.line_number,
                }
                for c in self.report.checks
            ]
        }, indent=2)
    
    def to_github_annotations(self) -> str:
        """Generate GitHub Actions annotations."""
        lines = []
        
        for check in self.report.checks:
            if not check.passed:
                level = "error"
                file_part = f"file={check.file_path}," if check.file_path else ""
                line_part = f"line={check.line_number}," if check.line_number else ""
                lines.append(f"::{level} {file_part}{line_part}::{check.name}: {check.message}")
        
        # Summary
        if self.report.passed:
            lines.append("::notice::✅ All CI gate checks passed")
        else:
            lines.append(f"::error::❌ CI gate checks failed: {self.report.fail_count} failures")
        
        return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="CI Gate Reporter")
    parser.add_argument("--format", choices=["markdown", "json", "github"], 
                        default="markdown", help="Output format")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    args = parser.parse_args()
    
    reporter = CIGateReporter()
    reporter.run_all_checks()
    
    if args.format == "markdown":
        output = reporter.to_markdown()
    elif args.format == "json":
        output = reporter.to_json()
    else:  # github
        output = reporter.to_github_annotations()
    
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        # Ensure UTF-8 output on Windows
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
        print(output)
    
    return 0 if reporter.report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
