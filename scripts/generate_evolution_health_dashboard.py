#!/usr/bin/env python3
"""
Meta-Evolution Health Dashboard Generator (AEP-6)

Generates a comprehensive health dashboard for the evolution system itself.
This is the central monitoring point for meta-evolution metrics.

Usage:
    python scripts/generate_evolution_health_dashboard.py [--project dgsf]

Output:
    reports/meta_evolution_health_{date}.md
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json


# =============================================================================
# METRIC COLLECTORS
# =============================================================================

def collect_signal_coverage_metrics(project_root: Path, project_id: str) -> Dict[str, Any]:
    """Run signal coverage measurement and collect results."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "measure_signal_coverage.py"),
                "--project", project_id,
                "--format", "yaml",
                "--output", str(project_root / "reports" / "temp_coverage.yaml")
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        temp_file = project_root / "reports" / "temp_coverage.yaml"
        if temp_file.exists():
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            temp_file.unlink()
            return data
    except Exception as e:
        print(f"⚠️ Error collecting signal coverage: {e}", file=sys.stderr)
    
    return {"error": "Failed to collect signal coverage metrics"}


def collect_evolution_velocity(signals_file: Path) -> Dict[str, Any]:
    """
    Calculate evolution velocity: time from signal to action.
    """
    if not signals_file.exists():
        return {"avg_days": None, "signals_actioned": 0, "signals_pending": 0}
    
    with open(signals_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    signals = data.get('signals', [])
    
    actioned = []
    pending = []
    velocities = []
    
    for sig in signals:
        status = sig.get('status', 'new')
        if status == 'actioned':
            actioned.append(sig)
            # Try to calculate velocity (would need actioned_at field)
            actioned_at = sig.get('actioned_at')
            created_at = sig.get('timestamp')
            if actioned_at and created_at:
                try:
                    created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    done = datetime.fromisoformat(actioned_at.replace('Z', '+00:00'))
                    velocities.append((done - created).days)
                except:
                    pass
        elif status in ['new', 'aggregated', 'reviewed']:
            pending.append(sig)
    
    avg_velocity = sum(velocities) / len(velocities) if velocities else None
    
    return {
        "avg_days": avg_velocity,
        "signals_actioned": len(actioned),
        "signals_pending": len(pending),
        "total_signals": len(signals),
        "meets_target": avg_velocity is None or avg_velocity <= 14  # Target: ≤14 days
    }


def collect_regression_rate(project_root: Path) -> Dict[str, Any]:
    """
    Calculate regression rate: proportion of evolutions rolled back.
    """
    # Check git log for revert commits related to evolution
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--since=30 days ago", "--grep=rollback\\|revert"],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        revert_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        
        # Count total evolution-related commits
        result2 = subprocess.run(
            ["git", "log", "--oneline", "--since=30 days ago", "--grep=evolution\\|AEP"],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        total_evolution = len(result2.stdout.strip().split('\n')) if result2.stdout.strip() else 0
        
        rate = revert_count / total_evolution if total_evolution > 0 else 0.0
        
        return {
            "rate": rate,
            "reverts": revert_count,
            "total_evolutions": total_evolution,
            "meets_target": rate <= 0.1  # Target: ≤10%
        }
    except Exception as e:
        return {"rate": None, "error": str(e)}


def collect_signal_noise_ratio(signals_file: Path) -> Dict[str, Any]:
    """
    Calculate signal-to-noise ratio: actionable signals / total signals.
    """
    if not signals_file.exists():
        return {"ratio": None, "actionable": 0, "total": 0}
    
    with open(signals_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    signals = data.get('signals', [])
    
    actionable = sum(1 for s in signals if s.get('status') in ['actioned', 'reviewed'])
    dismissed = sum(1 for s in signals if s.get('status') == 'dismissed')
    total = len(signals)
    
    ratio = actionable / total if total > 0 else 0.0
    
    return {
        "ratio": ratio,
        "actionable": actionable,
        "dismissed": dismissed,
        "total": total,
        "meets_target": ratio >= 0.3  # Target: ≥30%
    }


def collect_test_health(project_root: Path) -> Dict[str, Any]:
    """
    Run kernel tests and collect health status.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "kernel/tests/", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env={**dict(__import__('os').environ), "PYTHONPATH": str(project_root)}
        )
        
        # Parse pytest output
        output = result.stdout
        if "passed" in output:
            import re
            match = re.search(r'(\d+) passed', output)
            passed = int(match.group(1)) if match else 0
            match = re.search(r'(\d+) failed', output)
            failed = int(match.group(1)) if match else 0
            
            return {
                "passed": passed,
                "failed": failed,
                "total": passed + failed,
                "success_rate": passed / (passed + failed) if (passed + failed) > 0 else 1.0,
                "meets_target": failed == 0
            }
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "Could not parse test results"}


# =============================================================================
# HEALTH ASSESSMENT
# =============================================================================

def assess_overall_health(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess overall evolution system health based on all metrics.
    """
    issues = []
    warnings = []
    status = "healthy"
    
    # Check signal coverage
    coverage = metrics.get("signal_coverage", {}).get("summary", {})
    blind_spot = coverage.get("blind_spot_proxy", 0)
    if blind_spot > 0.3:
        issues.append(f"Critical: Blind spot proxy at {blind_spot:.1%} exceeds 30% threshold")
        status = "critical"
    elif blind_spot > 0.2:
        warnings.append(f"Blind spot proxy at {blind_spot:.1%} exceeds 20% target")
        if status == "healthy":
            status = "warning"
    
    # Check evolution velocity
    velocity = metrics.get("evolution_velocity", {})
    if velocity.get("avg_days") and velocity["avg_days"] > 21:
        issues.append(f"Evolution velocity slow: {velocity['avg_days']:.1f} days > 21 day threshold")
        if status != "critical":
            status = "warning"
    
    # Check regression rate
    regression = metrics.get("regression_rate", {})
    if regression.get("rate") and regression["rate"] > 0.15:
        issues.append(f"High regression rate: {regression['rate']:.1%} > 15% threshold")
        status = "critical"
    elif regression.get("rate") and regression["rate"] > 0.1:
        warnings.append(f"Regression rate at {regression['rate']:.1%} exceeds 10% target")
        if status == "healthy":
            status = "warning"
    
    # Check signal noise ratio
    noise = metrics.get("signal_noise_ratio", {})
    if noise.get("ratio") is not None and noise["ratio"] < 0.2:
        warnings.append(f"Low signal quality: only {noise['ratio']:.1%} actionable")
        if status == "healthy":
            status = "warning"
    
    # Check tests
    tests = metrics.get("test_health", {})
    if tests.get("failed", 0) > 0:
        issues.append(f"Test failures: {tests['failed']} tests failing")
        status = "critical"
    
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "score": _calculate_health_score(metrics),
        "trend": "stable"  # Would need historical data to calculate
    }


def _calculate_health_score(metrics: Dict[str, Any]) -> float:
    """Calculate overall health score (0-100)."""
    score = 100.0
    
    # Signal coverage contribution (30 points)
    coverage = metrics.get("signal_coverage", {}).get("summary", {})
    module_cov = coverage.get("module_coverage", 0)
    score -= (1 - module_cov) * 15
    blind_spot = coverage.get("blind_spot_proxy", 0)
    score -= blind_spot * 15
    
    # Evolution velocity contribution (20 points)
    velocity = metrics.get("evolution_velocity", {})
    if velocity.get("avg_days"):
        if velocity["avg_days"] > 14:
            score -= min(20, (velocity["avg_days"] - 14) * 2)
    
    # Regression rate contribution (25 points)
    regression = metrics.get("regression_rate", {})
    if regression.get("rate"):
        score -= regression["rate"] * 250  # 10% regression = -25 points
    
    # Signal quality contribution (15 points)
    noise = metrics.get("signal_noise_ratio", {})
    if noise.get("ratio") is not None:
        score -= (1 - noise["ratio"]) * 15 * 0.5  # Half weight
    
    # Test health contribution (10 points)
    tests = metrics.get("test_health", {})
    if tests.get("failed", 0) > 0:
        score -= 10
    
    return max(0, min(100, score))


# =============================================================================
# DASHBOARD GENERATION
# =============================================================================

def generate_dashboard(metrics: Dict[str, Any], project_id: str) -> str:
    """Generate markdown dashboard."""
    health = metrics["overall_health"]
    status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🔴"}.get(health["status"], "❓")
    
    dashboard = f"""# Meta-Evolution Health Dashboard

**Project**: {project_id}  
**Generated**: {datetime.now().isoformat()}  
**Overall Status**: {status_emoji} **{health['status'].upper()}** (Score: {health['score']:.0f}/100)

---

## 📊 Health Overview

```
╔══════════════════════════════════════════════════════════════╗
║  EVOLUTION SYSTEM HEALTH                                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Overall Score: {'█' * int(health['score'] / 5)}{'░' * (20 - int(health['score'] / 5))} {health['score']:.0f}%       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 🎯 Key Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
"""
    
    # Signal coverage
    coverage = metrics.get("signal_coverage", {}).get("summary", {})
    module_cov = coverage.get("module_coverage", 0)
    dashboard += f"| Module Coverage | {module_cov:.1%} | ≥80% | {'✅' if module_cov >= 0.8 else '⚠️'} |\n"
    
    blind_spot = coverage.get("blind_spot_proxy", 0)
    dashboard += f"| Blind Spot Proxy | {blind_spot:.1%} | ≤20% | {'✅' if blind_spot <= 0.2 else '⚠️'} |\n"
    
    # Evolution velocity
    velocity = metrics.get("evolution_velocity", {})
    vel_days = velocity.get("avg_days")
    vel_str = f"{vel_days:.1f}d" if vel_days else "N/A"
    dashboard += f"| Evolution Velocity | {vel_str} | ≤14d | {'✅' if velocity.get('meets_target', True) else '⚠️'} |\n"
    
    # Regression rate
    regression = metrics.get("regression_rate", {})
    reg_rate = regression.get("rate")
    reg_str = f"{reg_rate:.1%}" if reg_rate is not None else "N/A"
    dashboard += f"| Regression Rate | {reg_str} | ≤10% | {'✅' if regression.get('meets_target', True) else '⚠️'} |\n"
    
    # Signal noise ratio
    noise = metrics.get("signal_noise_ratio", {})
    noise_ratio = noise.get("ratio")
    noise_str = f"{noise_ratio:.1%}" if noise_ratio is not None else "N/A"
    dashboard += f"| Signal Quality | {noise_str} | ≥30% | {'✅' if noise.get('meets_target', True) else '⚠️'} |\n"
    
    # Test health
    tests = metrics.get("test_health", {})
    test_str = f"{tests.get('passed', 0)}/{tests.get('total', 0)}"
    dashboard += f"| Test Health | {test_str} | 100% | {'✅' if tests.get('meets_target', False) else '🔴'} |\n"
    
    dashboard += """
---

## 🚨 Issues & Warnings

"""
    
    if health["issues"]:
        dashboard += "### Critical Issues\n\n"
        for issue in health["issues"]:
            dashboard += f"- 🔴 {issue}\n"
        dashboard += "\n"
    
    if health["warnings"]:
        dashboard += "### Warnings\n\n"
        for warning in health["warnings"]:
            dashboard += f"- ⚠️ {warning}\n"
        dashboard += "\n"
    
    if not health["issues"] and not health["warnings"]:
        dashboard += "✅ No issues or warnings detected.\n\n"
    
    # Cold zones section
    cold_zones = metrics.get("signal_coverage", {}).get("cold_zones", [])
    dashboard += """---

## 🧊 Cold Zones (Top 10)

Modules without friction signal coverage:

| Module | Lines | Risk |
|--------|-------|------|
"""
    
    for zone in cold_zones[:10]:
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(zone.get("risk_level", "low"), "⚪")
        dashboard += f"| `{zone.get('module', 'unknown')}` | {zone.get('line_count', 0)} | {risk_emoji} |\n"
    
    if not cold_zones:
        dashboard += "| (none) | - | - |\n"
    
    # Signal statistics
    dashboard += f"""
---

## 📈 Signal Statistics

| Category | Count |
|----------|-------|
| Total Signals | {noise.get('total', 0)} |
| Actionable | {noise.get('actionable', 0)} |
| Dismissed | {noise.get('dismissed', 0)} |
| Pending | {velocity.get('signals_pending', 0)} |

---

## 🔧 Recommended Actions

"""
    
    if health["status"] == "critical":
        dashboard += """1. **IMMEDIATE**: Address critical issues above
2. Review and triage pending high-severity signals
3. Investigate test failures
"""
    elif health["status"] == "warning":
        dashboard += """1. Review cold zones for potential blind spots
2. Improve signal-to-action velocity
3. Consider adding proactive signal collection to uncovered modules
"""
    else:
        dashboard += """1. Continue monitoring metrics
2. Periodic review of cold zones
3. Consider expanding coverage to new modules
"""
    
    dashboard += """
---

## 📅 Next Review

Based on current health status:
"""
    
    if health["status"] == "critical":
        dashboard += "- **Next Review**: Within 24 hours\n"
    elif health["status"] == "warning":
        dashboard += "- **Next Review**: Within 3 days\n"
    else:
        dashboard += "- **Next Review**: Weekly\n"
    
    dashboard += """
---

*Generated by AEP-6 Meta-Evolution Monitoring System*
"""
    
    return dashboard


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Meta-Evolution Health Dashboard (AEP-6)"
    )
    parser.add_argument(
        "--project", "-p",
        default="dgsf",
        help="Project ID (default: dgsf)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path (default: reports/meta_evolution_health_{date}.md)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "yaml", "json"],
        default="markdown",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    project_dir = project_root / "projects" / args.project
    signals_file = project_dir / "evolution_signals.yaml"
    
    print(f"📊 Collecting metrics for project: {args.project}")
    
    # Collect all metrics
    metrics = {
        "project_id": args.project,
        "generated_at": datetime.now().isoformat(),
        "signal_coverage": collect_signal_coverage_metrics(project_root, args.project),
        "evolution_velocity": collect_evolution_velocity(signals_file),
        "regression_rate": collect_regression_rate(project_root),
        "signal_noise_ratio": collect_signal_noise_ratio(signals_file),
        "test_health": collect_test_health(project_root),
    }
    
    # Assess overall health
    metrics["overall_health"] = assess_overall_health(metrics)
    
    # Generate output
    if args.format == "yaml":
        output = yaml.dump(metrics, allow_unicode=True, sort_keys=False, default_flow_style=False)
    elif args.format == "json":
        output = json.dumps(metrics, indent=2, default=str)
    else:
        output = generate_dashboard(metrics, args.project)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "reports" / f"meta_evolution_health_{datetime.now().strftime('%Y-%m-%d')}.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"✅ Dashboard generated: {output_path}")
    
    # Print summary
    health = metrics["overall_health"]
    status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🔴"}.get(health["status"], "❓")
    print(f"\n{status_emoji} Overall Health: {health['status'].upper()} (Score: {health['score']:.0f}/100)")
    
    if health["issues"]:
        print("\n🚨 Critical Issues:")
        for issue in health["issues"]:
            print(f"   - {issue}")
    
    # Exit code based on health
    if health["status"] == "critical":
        sys.exit(2)
    elif health["status"] == "warning":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
