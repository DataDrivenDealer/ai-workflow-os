"""
Pipeline Gate Checker

Automated verification for PROJECT_DELIVERY_PIPELINE gates:
- G1: Data Quality (Stage 2 exit)
- G2: Sanity Checks (Stage 3 exit)
- G3: Performance & Robustness (Stage 4 exit)
- G4: Approval (Stage 5 exit)
- G5: Live Safety (Stage 6 continuous)

Usage:
    python scripts/gate_check.py --gate G1 --task-id DATA_2_XXX
    python scripts/gate_check.py --gate all --task-id EVAL_4_XXX
    python scripts/gate_check.py --ci  # Run in CI mode
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernel"
CONFIGS_DIR = ROOT / "configs"
GATE_CONFIG_PATH = ROOT / "configs" / "gates.yaml"


@dataclass
class CheckResult:
    """Result of a single check."""
    check_id: str
    name: str
    passed: bool
    actual_value: Any = None
    threshold: Any = None
    message: str = ""
    severity: str = "error"  # error, warning, info


@dataclass
class GateResult:
    """Result of a gate evaluation."""
    gate_id: str
    gate_name: str
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_id: Optional[str] = None
    
    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "checks": [
                {
                    "check_id": c.check_id,
                    "name": c.name,
                    "passed": c.passed,
                    "actual": c.actual_value,
                    "threshold": c.threshold,
                    "message": c.message,
                    "severity": c.severity,
                }
                for c in self.checks
            ],
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"# Gate {self.gate_id}: {self.gate_name}",
            f"",
            f"**Status**: {status}",
            f"**Timestamp**: {self.timestamp.isoformat()}",
            f"**Task**: {self.task_id or 'N/A'}",
            f"",
            f"## Check Results",
            f"",
            f"| Check | Status | Actual | Threshold | Message |",
            f"|-------|--------|--------|-----------|---------|",
        ]
        
        for c in self.checks:
            icon = "✅" if c.passed else ("⚠️" if c.severity == "warning" else "❌")
            actual = str(c.actual_value) if c.actual_value is not None else "-"
            threshold = str(c.threshold) if c.threshold is not None else "-"
            lines.append(f"| {c.name} | {icon} | {actual} | {threshold} | {c.message} |")
        
        return "\n".join(lines)


class GateChecker:
    """Pipeline Gate verification system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or GATE_CONFIG_PATH
        self._config: Optional[Dict[str, Any]] = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Load gate configuration."""
        if self._config is None:
            if self.config_path.exists():
                with self.config_path.open("r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                self._config = self._default_config()
        assert self._config is not None
        return self._config
    
    def _default_config(self) -> Dict[str, Any]:
        """Default gate configuration."""
        return {
            "gates": {
                "G1": {
                    "name": "Data Quality",
                    "stage": 2,
                    "checks": {
                        "schema_valid": {"threshold": True, "severity": "error"},
                        "missing_rate": {"threshold": 0.05, "severity": "error"},
                        "no_lookahead": {"threshold": True, "severity": "error"},
                        "snapshot_immutable": {"threshold": True, "severity": "error"},
                        "checksum_present": {"threshold": True, "severity": "error"},
                    }
                },
                "G2": {
                    "name": "Sanity Checks",
                    "stage": 3,
                    "checks": {
                        "unit_tests_pass": {"threshold": True, "severity": "error"},
                        "no_lookahead": {"threshold": True, "severity": "error"},
                        "cost_assumptions_valid": {"threshold": True, "severity": "error"},
                        "signal_range_valid": {"threshold": True, "severity": "error"},
                        "reproducible": {"threshold": True, "severity": "error"},
                    }
                },
                "G3": {
                    "name": "Performance & Robustness",
                    "stage": 4,
                    "checks": {
                        "oos_sharpe": {"threshold": 0.5, "severity": "error"},
                        "sharpe_decay": {"threshold": 0.5, "severity": "warning"},
                        "max_drawdown": {"threshold": 0.25, "severity": "error"},
                        "subperiod_stability": {"threshold": True, "severity": "warning"},
                        "param_sensitivity": {"threshold": "low", "severity": "warning"},
                        "stress_test_survival": {"threshold": True, "severity": "error"},
                    }
                },
                "G4": {
                    "name": "Approval",
                    "stage": 5,
                    "checks": {
                        "g3_passed": {"threshold": True, "severity": "error"},
                        "decision_memo_complete": {"threshold": True, "severity": "error"},
                        "kill_switch_defined": {"threshold": True, "severity": "error"},
                        "risk_spec_compliant": {"threshold": True, "severity": "error"},
                        "runbook_ready": {"threshold": True, "severity": "warning"},
                    }
                },
                "G5": {
                    "name": "Live Safety",
                    "stage": 6,
                    "checks": {
                        "kill_switch_available": {"threshold": True, "severity": "error"},
                        "monitoring_active": {"threshold": True, "severity": "error"},
                        "data_feed_healthy": {"threshold": True, "severity": "error"},
                        "execution_channel_healthy": {"threshold": True, "severity": "error"},
                        "risk_limits_enforced": {"threshold": True, "severity": "error"},
                    }
                },
            }
        }
    
    def check_gate(
        self,
        gate_id: str,
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        """
        Run gate verification.
        
        Args:
            gate_id: Gate identifier (G1-G5)
            task_id: Associated TaskCard ID
            context: Additional context for checks
        
        Returns:
            GateResult with check outcomes
        """
        gate_config = self.config.get("gates", {}).get(gate_id)
        if not gate_config:
            return GateResult(
                gate_id=gate_id,
                gate_name="Unknown",
                passed=False,
                checks=[CheckResult(
                    check_id="GATE_CONFIG",
                    name="Gate Configuration",
                    passed=False,
                    message=f"Gate {gate_id} not configured",
                )],
                task_id=task_id,
            )
        
        checker_map = {
            "G1": self._check_g1_data_quality,
            "G2": self._check_g2_sanity,
            "G3": self._check_g3_robustness,
            "G4": self._check_g4_approval,
            "G5": self._check_g5_live_safety,
        }
        
        checker = checker_map.get(gate_id)
        if not checker:
            return GateResult(
                gate_id=gate_id,
                gate_name=gate_config.get("name", "Unknown"),
                passed=False,
                checks=[CheckResult(
                    check_id="CHECKER_MISSING",
                    name="Checker Implementation",
                    passed=False,
                    message=f"No checker implemented for {gate_id}",
                )],
                task_id=task_id,
            )
        
        checks = checker(gate_config, context or {})
        
        # Gate passes if no error-severity checks failed
        passed = all(
            c.passed or c.severity != "error"
            for c in checks
        )
        
        return GateResult(
            gate_id=gate_id,
            gate_name=gate_config.get("name", "Unknown"),
            passed=passed,
            checks=checks,
            task_id=task_id,
        )
    
    def _check_g1_data_quality(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[CheckResult]:
        """G1: Data Quality checks."""
        checks = []
        check_config = config.get("checks", {})
        
        # Schema validation
        schema_valid = context.get("schema_valid", None)
        if schema_valid is not None:
            checks.append(CheckResult(
                check_id="G1_SCHEMA",
                name="Schema Validation",
                passed=bool(schema_valid),
                actual_value=schema_valid,
                threshold=True,
                message="" if schema_valid else "Schema validation failed",
                severity=check_config.get("schema_valid", {}).get("severity", "error"),
            ))
        else:
            checks.append(self._check_schema_files(context))
        
        # Missing rate
        missing_rate = context.get("missing_rate")
        threshold = check_config.get("missing_rate", {}).get("threshold", 0.05)
        if missing_rate is not None:
            checks.append(CheckResult(
                check_id="G1_MISSING",
                name="Missing Rate",
                passed=missing_rate <= threshold,
                actual_value=f"{missing_rate:.2%}",
                threshold=f"≤{threshold:.0%}",
                message="" if missing_rate <= threshold else f"Missing rate {missing_rate:.2%} exceeds threshold",
                severity=check_config.get("missing_rate", {}).get("severity", "error"),
            ))
        
        # No look-ahead check
        no_lookahead = context.get("no_lookahead")
        if no_lookahead is not None:
            checks.append(CheckResult(
                check_id="G1_LOOKAHEAD",
                name="No Look-ahead",
                passed=bool(no_lookahead),
                actual_value=no_lookahead,
                threshold=True,
                message="" if no_lookahead else "Look-ahead bias detected",
                severity=check_config.get("no_lookahead", {}).get("severity", "error"),
            ))
        
        # Snapshot immutability
        snapshot_immutable = context.get("snapshot_immutable")
        if snapshot_immutable is not None:
            checks.append(CheckResult(
                check_id="G1_IMMUTABLE",
                name="Snapshot Immutable",
                passed=bool(snapshot_immutable),
                actual_value=snapshot_immutable,
                threshold=True,
                message="" if snapshot_immutable else "Snapshot has been modified",
                severity=check_config.get("snapshot_immutable", {}).get("severity", "error"),
            ))
        
        # Checksum presence
        checksums = context.get("checksums")
        has_checksums = checksums is not None and len(checksums) > 0
        checks.append(CheckResult(
            check_id="G1_CHECKSUM",
            name="Checksums Present",
            passed=has_checksums,
            actual_value=len(checksums) if checksums else 0,
            threshold="> 0",
            message="" if has_checksums else "No checksums found",
            severity=check_config.get("checksum_present", {}).get("severity", "error"),
        ))
        
        return checks
    
    def _check_schema_files(self, context: Dict[str, Any]) -> CheckResult:
        """Check for schema definition files."""
        project_path = context.get("project_path", ROOT / "projects" / "dgsf")
        schema_patterns = ["**/schema*.json", "**/schema*.yaml", "**/*_schema*.json"]
        
        found_schemas = []
        for pattern in schema_patterns:
            found_schemas.extend(Path(project_path).glob(pattern))
        
        return CheckResult(
            check_id="G1_SCHEMA",
            name="Schema Files",
            passed=len(found_schemas) > 0,
            actual_value=len(found_schemas),
            threshold="> 0",
            message="" if found_schemas else "No schema files found",
            severity="error",
        )
    
    def _check_g2_sanity(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[CheckResult]:
        """G2: Sanity checks."""
        checks = []
        check_config = config.get("checks", {})
        
        # Unit tests
        if context.get("run_tests", True):
            test_result = self._run_unit_tests(context)
            checks.append(test_result)
        else:
            tests_passed = context.get("unit_tests_pass")
            if tests_passed is not None:
                checks.append(CheckResult(
                    check_id="G2_TESTS",
                    name="Unit Tests",
                    passed=bool(tests_passed),
                    actual_value=tests_passed,
                    threshold=True,
                    message="" if tests_passed else "Unit tests failed",
                    severity=check_config.get("unit_tests_pass", {}).get("severity", "error"),
                ))
        
        # No look-ahead
        no_lookahead = context.get("no_lookahead")
        if no_lookahead is not None:
            checks.append(CheckResult(
                check_id="G2_LOOKAHEAD",
                name="No Look-ahead",
                passed=bool(no_lookahead),
                actual_value=no_lookahead,
                threshold=True,
                message="" if no_lookahead else "Look-ahead detected in signal generation",
                severity=check_config.get("no_lookahead", {}).get("severity", "error"),
            ))
        
        # Cost assumptions
        cost_valid = context.get("cost_assumptions_valid")
        if cost_valid is not None:
            checks.append(CheckResult(
                check_id="G2_COST",
                name="Cost Assumptions",
                passed=bool(cost_valid),
                actual_value=cost_valid,
                threshold=True,
                message="" if cost_valid else "Cost assumptions unrealistic",
                severity=check_config.get("cost_assumptions_valid", {}).get("severity", "error"),
            ))
        
        # Signal range
        signal_range = context.get("signal_range")
        if signal_range is not None:
            valid = signal_range.get("min", -999) >= -1 and signal_range.get("max", 999) <= 1
            checks.append(CheckResult(
                check_id="G2_SIGNAL",
                name="Signal Range",
                passed=valid,
                actual_value=f"[{signal_range.get('min')}, {signal_range.get('max')}]",
                threshold="[-1, 1]",
                message="" if valid else "Signal outside expected range",
                severity=check_config.get("signal_range_valid", {}).get("severity", "error"),
            ))
        
        # Reproducibility
        reproducible = context.get("reproducible")
        if reproducible is not None:
            checks.append(CheckResult(
                check_id="G2_REPRO",
                name="Reproducibility",
                passed=bool(reproducible),
                actual_value=reproducible,
                threshold=True,
                message="" if reproducible else "Results not reproducible with same seed",
                severity=check_config.get("reproducible", {}).get("severity", "error"),
            ))
        
        return checks
    
    def _run_unit_tests(self, context: Dict[str, Any]) -> CheckResult:
        """Run pytest and return result."""
        import sys
        project_path = context.get("project_path", ROOT / "projects" / "dgsf" / "repo")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(project_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(ROOT),
            )
            passed = result.returncode == 0
            return CheckResult(
                check_id="G2_TESTS",
                name="Unit Tests",
                passed=passed,
                actual_value=f"exit code {result.returncode}",
                threshold="exit code 0",
                message="" if passed else result.stdout[-500:] if result.stdout else result.stderr[-500:],
                severity="error",
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                check_id="G2_TESTS",
                name="Unit Tests",
                passed=False,
                message="Tests timed out after 300s",
                severity="error",
            )
        except Exception as e:
            return CheckResult(
                check_id="G2_TESTS",
                name="Unit Tests",
                passed=False,
                message=f"Failed to run tests: {e}",
                severity="error",
            )
    
    def _check_g3_robustness(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[CheckResult]:
        """G3: Performance & Robustness checks."""
        checks = []
        check_config = config.get("checks", {})
        
        # OOS Sharpe
        oos_sharpe = context.get("oos_sharpe")
        threshold = check_config.get("oos_sharpe", {}).get("threshold", 0.5)
        if oos_sharpe is not None:
            checks.append(CheckResult(
                check_id="G3_SHARPE",
                name="OOS Sharpe Ratio",
                passed=oos_sharpe >= threshold,
                actual_value=f"{oos_sharpe:.2f}",
                threshold=f"≥{threshold}",
                message="" if oos_sharpe >= threshold else "OOS Sharpe below threshold",
                severity=check_config.get("oos_sharpe", {}).get("severity", "error"),
            ))
        
        # Sharpe decay (IS to OOS)
        is_sharpe = context.get("is_sharpe")
        decay_threshold = check_config.get("sharpe_decay", {}).get("threshold", 0.5)
        if oos_sharpe is not None and is_sharpe is not None and is_sharpe > 0:
            decay = 1 - (oos_sharpe / is_sharpe)
            checks.append(CheckResult(
                check_id="G3_DECAY",
                name="Sharpe Decay",
                passed=decay <= decay_threshold,
                actual_value=f"{decay:.0%}",
                threshold=f"≤{decay_threshold:.0%}",
                message="" if decay <= decay_threshold else "Excessive Sharpe decay from IS to OOS",
                severity=check_config.get("sharpe_decay", {}).get("severity", "warning"),
            ))
        
        # Max drawdown
        max_dd = context.get("max_drawdown")
        dd_threshold = check_config.get("max_drawdown", {}).get("threshold", 0.25)
        if max_dd is not None:
            checks.append(CheckResult(
                check_id="G3_MAXDD",
                name="Max Drawdown",
                passed=max_dd <= dd_threshold,
                actual_value=f"{max_dd:.1%}",
                threshold=f"≤{dd_threshold:.0%}",
                message="" if max_dd <= dd_threshold else "Max drawdown exceeds threshold",
                severity=check_config.get("max_drawdown", {}).get("severity", "error"),
            ))
        
        # Subperiod stability
        subperiod_results = context.get("subperiod_sharpes", [])
        if subperiod_results:
            all_positive = all(s > 0 for s in subperiod_results)
            checks.append(CheckResult(
                check_id="G3_SUBPERIOD",
                name="Subperiod Stability",
                passed=all_positive,
                actual_value=f"{sum(1 for s in subperiod_results if s > 0)}/{len(subperiod_results)} positive",
                threshold="All positive",
                message="" if all_positive else "Some subperiods have negative Sharpe",
                severity=check_config.get("subperiod_stability", {}).get("severity", "warning"),
            ))
        
        # Stress test survival
        stress_survived = context.get("stress_test_survival")
        if stress_survived is not None:
            checks.append(CheckResult(
                check_id="G3_STRESS",
                name="Stress Test Survival",
                passed=bool(stress_survived),
                actual_value=stress_survived,
                threshold=True,
                message="" if stress_survived else "Strategy failed stress tests",
                severity=check_config.get("stress_test_survival", {}).get("severity", "error"),
            ))
        
        return checks
    
    def _check_g4_approval(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[CheckResult]:
        """G4: Approval checks."""
        checks = []
        check_config = config.get("checks", {})
        
        # G3 passed
        g3_passed = context.get("g3_passed")
        if g3_passed is not None:
            checks.append(CheckResult(
                check_id="G4_G3",
                name="G3 Gate Passed",
                passed=bool(g3_passed),
                actual_value=g3_passed,
                threshold=True,
                message="" if g3_passed else "G3 gate not passed",
                severity=check_config.get("g3_passed", {}).get("severity", "error"),
            ))
        
        # Decision memo
        decision_memo = context.get("decision_memo_path")
        memo_exists = decision_memo and Path(decision_memo).exists()
        checks.append(CheckResult(
            check_id="G4_MEMO",
            name="Decision Memo",
            passed=bool(memo_exists),
            actual_value=decision_memo if memo_exists else "Not found",
            threshold="Exists",
            message="" if memo_exists else "Decision memo not found",
            severity=check_config.get("decision_memo_complete", {}).get("severity", "error"),
        ))
        
        # Kill-switch defined
        kill_switch = context.get("kill_switch_defined")
        if kill_switch is not None:
            checks.append(CheckResult(
                check_id="G4_KILLSWITCH",
                name="Kill-switch Defined",
                passed=bool(kill_switch),
                actual_value=kill_switch,
                threshold=True,
                message="" if kill_switch else "Kill-switch criteria not defined",
                severity=check_config.get("kill_switch_defined", {}).get("severity", "error"),
            ))
        
        # Risk spec compliance
        risk_compliant = context.get("risk_spec_compliant")
        if risk_compliant is not None:
            checks.append(CheckResult(
                check_id="G4_RISK",
                name="Risk Spec Compliant",
                passed=bool(risk_compliant),
                actual_value=risk_compliant,
                threshold=True,
                message="" if risk_compliant else "Risk spec violations found",
                severity=check_config.get("risk_spec_compliant", {}).get("severity", "error"),
            ))
        
        # Runbook ready
        runbook_path = context.get("runbook_path")
        runbook_exists = runbook_path and Path(runbook_path).exists()
        checks.append(CheckResult(
            check_id="G4_RUNBOOK",
            name="Runbook Ready",
            passed=bool(runbook_exists),
            actual_value=runbook_path if runbook_exists else "Not found",
            threshold="Exists",
            message="" if runbook_exists else "Runbook not found",
            severity=check_config.get("runbook_ready", {}).get("severity", "warning"),
        ))
        
        return checks
    
    def _check_g5_live_safety(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[CheckResult]:
        """G5: Live Safety checks."""
        checks = []
        check_config = config.get("checks", {})
        
        # Kill-switch available
        kill_switch_available = context.get("kill_switch_available")
        if kill_switch_available is not None:
            checks.append(CheckResult(
                check_id="G5_KILLSWITCH",
                name="Kill-switch Available",
                passed=bool(kill_switch_available),
                actual_value=kill_switch_available,
                threshold=True,
                message="" if kill_switch_available else "Kill-switch not operational",
                severity=check_config.get("kill_switch_available", {}).get("severity", "error"),
            ))
        
        # Monitoring active
        monitoring = context.get("monitoring_active")
        if monitoring is not None:
            checks.append(CheckResult(
                check_id="G5_MONITOR",
                name="Monitoring Active",
                passed=bool(monitoring),
                actual_value=monitoring,
                threshold=True,
                message="" if monitoring else "Monitoring not active",
                severity=check_config.get("monitoring_active", {}).get("severity", "error"),
            ))
        
        # Data feed healthy
        data_feed = context.get("data_feed_healthy")
        if data_feed is not None:
            checks.append(CheckResult(
                check_id="G5_DATA",
                name="Data Feed Healthy",
                passed=bool(data_feed),
                actual_value=data_feed,
                threshold=True,
                message="" if data_feed else "Data feed unhealthy",
                severity=check_config.get("data_feed_healthy", {}).get("severity", "error"),
            ))
        
        # Execution channel healthy
        exec_channel = context.get("execution_channel_healthy")
        if exec_channel is not None:
            checks.append(CheckResult(
                check_id="G5_EXEC",
                name="Execution Channel",
                passed=bool(exec_channel),
                actual_value=exec_channel,
                threshold=True,
                message="" if exec_channel else "Execution channel unhealthy",
                severity=check_config.get("execution_channel_healthy", {}).get("severity", "error"),
            ))
        
        # Risk limits enforced
        risk_limits = context.get("risk_limits_enforced")
        if risk_limits is not None:
            checks.append(CheckResult(
                check_id="G5_RISK",
                name="Risk Limits Enforced",
                passed=bool(risk_limits),
                actual_value=risk_limits,
                threshold=True,
                message="" if risk_limits else "Risk limits not enforced",
                severity=check_config.get("risk_limits_enforced", {}).get("severity", "error"),
            ))
        
        return checks


def run_ci_gates(changed_files: Optional[Set[Path]] = None) -> List[GateResult]:
    """
    Run gates appropriate for CI context.
    
    In CI, we run:
    - G2 (unit tests) always
    - G1 if data files changed
    """
    checker = GateChecker()
    results = []
    
    # Always run G2 sanity checks (unit tests)
    g2_result = checker.check_gate("G2", context={"run_tests": True})
    results.append(g2_result)
    
    # Check for governance violations
    sys.path.insert(0, str(KERNEL_DIR))
    try:
        from governance_gate import GovernanceGate, has_violations
        gov_gate = GovernanceGate()
        gov_results = gov_gate.verify_all()
        
        gov_checks = []
        for gr in gov_results:
            gov_checks.append(CheckResult(
                check_id=f"GOV_{gr.gate_name.upper()}",
                name=f"Governance: {gr.gate_name}",
                passed=not gr.has_violations,
                actual_value=f"{len(gr.violations)} violations" if gr.has_violations else "Clean",
                threshold="0 violations",
                message=gr.violations[0].description if gr.violations else "",
                severity="error" if gr.has_violations else "info",
            ))
        
        results.append(GateResult(
            gate_id="GOV",
            gate_name="Governance Verification",
            passed=not any(c.severity == "error" and not c.passed for c in gov_checks),
            checks=gov_checks,
        ))
    except ImportError:
        pass  # Governance module not available
    
    return results


def main() -> int:
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
    
    parser = argparse.ArgumentParser(description="Pipeline Gate Checker")
    parser.add_argument("--gate", choices=["G1", "G2", "G3", "G4", "G5", "all"], 
                        help="Gate to check")
    parser.add_argument("--task-id", help="TaskCard ID")
    parser.add_argument("--ci", action="store_true", help="Run in CI mode")
    parser.add_argument("--output", choices=["text", "json", "markdown"], default="text",
                        help="Output format")
    parser.add_argument("--context", type=str, help="JSON context for checks")
    args = parser.parse_args()
    
    context = {}
    if args.context:
        context = json.loads(args.context)
    
    checker = GateChecker()
    results: List[GateResult] = []
    
    if args.ci:
        results = run_ci_gates()
    elif args.gate == "all":
        for gate_id in ["G1", "G2", "G3", "G4", "G5"]:
            results.append(checker.check_gate(gate_id, args.task_id, context))
    elif args.gate:
        results.append(checker.check_gate(args.gate, args.task_id, context))
    else:
        parser.print_help()
        return 1
    
    # Output results
    all_passed = all(r.passed for r in results)
    
    if args.output == "json":
        print(json.dumps([r.to_dict() for r in results], indent=2))
    elif args.output == "markdown":
        for r in results:
            print(r.to_markdown())
            print("\n---\n")
    else:
        for r in results:
            status = "✅ PASSED" if r.passed else "❌ FAILED"
            print(f"\n{r.gate_id}: {r.gate_name} - {status}")
            print(f"  Errors: {r.error_count}, Warnings: {r.warning_count}")
            for c in r.checks:
                icon = "✅" if c.passed else ("⚠️" if c.severity == "warning" else "❌")
                print(f"  {icon} {c.name}: {c.message or 'OK'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
