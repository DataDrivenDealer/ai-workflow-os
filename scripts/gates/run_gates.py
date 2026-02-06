"""
Gate Runner - Execute gate checks for pipeline stages.

This module provides deterministic enforcement of gates defined in configs/gates.yaml.
Each gate check must have a corresponding script or be explicitly marked as manual_review.

Version: 1.0.0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add kernel to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from kernel.paths import ROOT
except ImportError:
    ROOT = Path(__file__).parent.parent.parent


@dataclass
class CheckResult:
    """Result of a single gate check."""
    check_id: str
    status: str  # pass, fail, skip, warn, error
    actual_value: Any = None
    expected_value: Any = None
    message: str = ""
    evidence_path: Optional[str] = None


@dataclass
class GateResult:
    """Result of a complete gate evaluation."""
    gate_id: str
    stage: int
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gate_id": self.gate_id,
            "stage": self.stage,
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {
                    "check_id": c.check_id,
                    "status": c.status,
                    "actual_value": c.actual_value,
                    "expected_value": c.expected_value,
                    "message": c.message,
                    "evidence_path": c.evidence_path,
                }
                for c in self.checks
            ],
        }


def load_gates_config() -> Dict[str, Any]:
    """Load gate configuration from configs/gates.yaml."""
    config_path = ROOT / "configs" / "gates.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_check_script_path(gate_id: str, check_id: str) -> Optional[Path]:
    """Get the path to a check script if it exists.
    
    Convention: scripts/gates/check_{gate_id}_{check_id}.py
    """
    script_name = f"check_{gate_id.lower()}_{check_id}.py"
    script_path = ROOT / "scripts" / "gates" / script_name
    if script_path.exists():
        return script_path
    
    # Also check for generic check
    generic_path = ROOT / "scripts" / "gates" / f"check_{check_id}.py"
    if generic_path.exists():
        return generic_path
    
    return None


def run_check_script(script_path: Path, check_config: Dict[str, Any]) -> CheckResult:
    """Run a check script and return the result.
    
    The script should define a function `run_check(config: dict) -> dict`
    that returns {"status": "pass|fail|warn", "actual": ..., "message": ...}
    """
    check_id = script_path.stem.replace("check_", "")
    
    try:
        spec = importlib.util.spec_from_file_location(check_id, script_path)
        if spec is None or spec.loader is None:
            return CheckResult(
                check_id=check_id,
                status="error",
                message=f"Could not load script: {script_path}",
            )
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if not hasattr(module, "run_check"):
            return CheckResult(
                check_id=check_id,
                status="error",
                message=f"Script missing run_check function: {script_path}",
            )
        
        result = module.run_check(check_config)
        
        return CheckResult(
            check_id=check_id,
            status=result.get("status", "error"),
            actual_value=result.get("actual"),
            expected_value=check_config.get("threshold"),
            message=result.get("message", ""),
            evidence_path=result.get("evidence_path"),
        )
    
    except Exception as e:
        return CheckResult(
            check_id=check_id,
            status="error",
            message=f"Script error: {str(e)}",
        )


def run_gate(gate_id: str, config: Optional[Dict[str, Any]] = None) -> GateResult:
    """Run all checks for a gate.
    
    Args:
        gate_id: Gate identifier (e.g., "G1", "G2").
        config: Optional override config.
    
    Returns:
        GateResult with all check results.
    """
    if config is None:
        config = load_gates_config()
    
    gate_config = config.get("gates", {}).get(gate_id)
    if not gate_config:
        return GateResult(
            gate_id=gate_id,
            stage=-1,
            passed=False,
            checks=[CheckResult(
                check_id="gate_lookup",
                status="error",
                message=f"Gate {gate_id} not found in configuration",
            )],
        )
    
    stage = gate_config.get("stage", -1)
    checks = gate_config.get("checks", {})
    
    results = []
    all_passed = True
    
    for check_id, check_config in checks.items():
        # Check if manual review
        if check_config.get("manual_review", False):
            results.append(CheckResult(
                check_id=check_id,
                status="skip",
                message="Manual review required",
            ))
            continue
        
        # Check if auto_check is disabled
        if not check_config.get("auto_check", True):
            results.append(CheckResult(
                check_id=check_id,
                status="skip",
                message="Auto-check disabled",
            ))
            continue
        
        # Find and run check script
        script_path = get_check_script_path(gate_id, check_id)
        
        if script_path:
            result = run_check_script(script_path, check_config)
        else:
            # No script found
            severity = check_config.get("severity", "error")
            if severity == "warning":
                result = CheckResult(
                    check_id=check_id,
                    status="warn",
                    message=f"No check script found (scripts/gates/check_{gate_id.lower()}_{check_id}.py), but severity is warning",
                )
            else:
                result = CheckResult(
                    check_id=check_id,
                    status="fail",
                    message=f"No check script found: scripts/gates/check_{gate_id.lower()}_{check_id}.py",
                )
        
        results.append(result)
        
        # Determine if gate fails
        if result.status == "fail":
            if check_config.get("severity", "error") == "error":
                all_passed = False
        elif result.status == "error":
            all_passed = False
    
    return GateResult(
        gate_id=gate_id,
        stage=stage,
        passed=all_passed,
        checks=results,
    )


def save_gate_report(result: GateResult) -> Path:
    """Save gate result to artifacts directory.
    
    Returns:
        Path to the saved report.
    """
    artifacts_dir = ROOT / "artifacts" / "gate_reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
    report_path = artifacts_dir / f"{result.gate_id}_{timestamp}.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    return report_path


def list_missing_scripts() -> Dict[str, List[str]]:
    """List all gates that have auto_check=true but no corresponding script.
    
    Returns:
        Dict mapping gate_id to list of missing check_ids.
    """
    config = load_gates_config()
    missing = {}
    
    for gate_id, gate_config in config.get("gates", {}).items():
        gate_missing = []
        for check_id, check_config in gate_config.get("checks", {}).items():
            if check_config.get("auto_check", True) and not check_config.get("manual_review", False):
                script_path = get_check_script_path(gate_id, check_id)
                if not script_path:
                    gate_missing.append(check_id)
        
        if gate_missing:
            missing[gate_id] = gate_missing
    
    return missing


def main():
    """CLI interface for gate runner."""
    parser = argparse.ArgumentParser(description="Gate Runner - Execute pipeline gate checks")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run a gate")
    run_parser.add_argument("gate_id", help="Gate ID (e.g., G1, G2)")
    run_parser.add_argument("--save", action="store_true", help="Save report to artifacts")
    run_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # audit command
    subparsers.add_parser("audit", help="Audit for missing check scripts")
    
    # list command
    subparsers.add_parser("list", help="List all gates")
    
    args = parser.parse_args()
    
    if args.command == "run":
        result = run_gate(args.gate_id)
        
        if args.save:
            report_path = save_gate_report(result)
            print(f"📄 Report saved: {report_path}")
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status_icon = "✅" if result.passed else "❌"
            print(f"\n{status_icon} Gate {result.gate_id} (Stage {result.stage}): {'PASSED' if result.passed else 'FAILED'}")
            print(f"   Timestamp: {result.timestamp.isoformat()}\n")
            
            for check in result.checks:
                if check.status == "pass":
                    icon = "✅"
                elif check.status == "fail":
                    icon = "❌"
                elif check.status == "warn":
                    icon = "⚠️"
                elif check.status == "skip":
                    icon = "⏭️"
                else:
                    icon = "💥"
                
                print(f"   {icon} {check.check_id}: {check.status}")
                if check.message:
                    print(f"      {check.message}")
        
        sys.exit(0 if result.passed else 1)
    
    elif args.command == "audit":
        missing = list_missing_scripts()
        
        if not missing:
            print("✅ All auto_check gates have corresponding scripts")
        else:
            print("⚠️ Missing gate check scripts:\n")
            for gate_id, checks in missing.items():
                print(f"  {gate_id}:")
                for check_id in checks:
                    script_name = f"check_{gate_id.lower()}_{check_id}.py"
                    print(f"    - scripts/gates/{script_name}")
            
            print("\n📝 Create these scripts with a `run_check(config: dict) -> dict` function")
    
    elif args.command == "list":
        config = load_gates_config()
        print("📋 Configured Gates:\n")
        for gate_id, gate_config in config.get("gates", {}).items():
            check_count = len(gate_config.get("checks", {}))
            print(f"  {gate_id}: {gate_config.get('name', 'Unnamed')} (Stage {gate_config.get('stage', '?')})")
            print(f"      Checks: {check_count}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
