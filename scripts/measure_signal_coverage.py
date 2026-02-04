#!/usr/bin/env python3
"""
Signal Coverage Measurement Script (AEP-6)

Measures the proportion of code paths covered by evolution signal detection.
This is a key meta-monitoring metric to detect blind spots in the evolution system.

Usage:
    python scripts/measure_signal_coverage.py [--project dgsf] [--output reports/signal_coverage.md]

Metrics produced:
    - signal_coverage: % of modules with at least one signal in past 30d
    - cold_zones: List of modules with no signals for >30d
    - blind_spot_proxy: % of new code paths without signal coverage
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import yaml
import json


# =============================================================================
# CODE PATH DISCOVERY
# =============================================================================

def discover_code_modules(project_root: Path, source_path: str) -> List[Dict[str, Any]]:
    """
    Discover all Python modules in the project source.
    
    Returns list of module info: {path, name, last_modified, line_count}
    """
    source_dir = project_root / source_path
    modules = []
    
    if not source_dir.exists():
        print(f"⚠️ Source directory not found: {source_dir}", file=sys.stderr)
        return modules
    
    for py_file in source_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        try:
            stat = py_file.stat()
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for _ in f)
            
            # Calculate relative path from source
            rel_path = py_file.relative_to(source_dir)
            module_name = str(rel_path).replace("/", ".").replace("\\", ".").replace(".py", "")
            
            modules.append({
                "path": str(py_file),
                "relative_path": str(rel_path),
                "module_name": module_name,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "line_count": line_count,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            })
        except Exception as e:
            print(f"⚠️ Error processing {py_file}: {e}", file=sys.stderr)
    
    return modules


def discover_skills(prompts_dir: Path) -> List[str]:
    """Discover all skill prompt files."""
    skills = []
    if prompts_dir.exists():
        for prompt_file in prompts_dir.glob("*.prompt.md"):
            skill_name = prompt_file.stem.replace(".prompt", "")
            skills.append(skill_name)
    return skills


def discover_rules(copilot_instructions: Path) -> List[str]:
    """Extract rule IDs from copilot-instructions.md."""
    rules = []
    if copilot_instructions.exists():
        with open(copilot_instructions, 'r', encoding='utf-8') as f:
            content = f.read()
            import re
            # Match rule patterns like | R1 | or R1:
            rule_matches = re.findall(r'\bR(\d+)\b', content)
            rules = list(set(f"R{r}" for r in rule_matches))
    return sorted(rules)


# =============================================================================
# SIGNAL LOADING
# =============================================================================

def load_evolution_signals(signals_file: Path, window_days: int = 30) -> List[Dict[str, Any]]:
    """Load evolution signals from YAML file, filtered by time window."""
    if not signals_file.exists():
        return []
    
    with open(signals_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    signals = data.get('signals', [])
    cutoff = datetime.now() - timedelta(days=window_days)
    
    filtered = []
    for sig in signals:
        ts_str = sig.get('timestamp', '')
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                if ts.replace(tzinfo=None) >= cutoff:
                    filtered.append(sig)
            except (ValueError, TypeError):
                filtered.append(sig)  # Include if can't parse
    
    return filtered


def extract_signal_coverage(signals: List[Dict]) -> Dict[str, Set[str]]:
    """
    Extract which code areas are covered by signals.
    
    Returns:
        Dict with coverage sets for modules, skills, rules
    """
    coverage = {
        "modules": set(),
        "skills": set(),
        "rules": set(),
        "signal_types": set(),
    }
    
    for sig in signals:
        # Extract module from context (heuristic)
        context = sig.get('context', '')
        if context:
            # Look for path patterns like "src/dgsf/module/file.py"
            import re
            path_matches = re.findall(r'[\w/\\]+\.py', context)
            for path in path_matches:
                module = path.replace("/", ".").replace("\\", ".").replace(".py", "")
                coverage["modules"].add(module)
        
        # Extract rule coverage
        rule_id = sig.get('rule_id')
        if rule_id:
            coverage["rules"].add(rule_id)
        
        # Extract signal type
        sig_type = sig.get('signal_type')
        if sig_type:
            coverage["signal_types"].add(sig_type)
    
    return coverage


# =============================================================================
# COLD ZONE DETECTION
# =============================================================================

def detect_cold_zones(
    modules: List[Dict],
    covered_modules: Set[str],
    cold_threshold_days: int = 30
) -> List[Dict[str, Any]]:
    """
    Identify modules that have had no friction signals.
    
    Returns list of cold zone info with risk assessment.
    """
    cold_zones = []
    now = datetime.now()
    
    for module in modules:
        module_name = module.get("module_name", "")
        
        # Check if module or any parent/child is covered
        is_covered = any(
            module_name.startswith(covered) or covered.startswith(module_name)
            for covered in covered_modules
        )
        
        if not is_covered:
            # Calculate age
            try:
                last_mod = datetime.fromisoformat(module.get("last_modified", ""))
                age_days = (now - last_mod).days
            except:
                age_days = 0
            
            # Risk assessment based on line count and age
            line_count = module.get("line_count", 0)
            if line_count > 200:
                risk = "high"
            elif line_count > 50:
                risk = "medium"
            else:
                risk = "low"
            
            cold_zones.append({
                "module": module_name,
                "path": module.get("relative_path", ""),
                "line_count": line_count,
                "days_since_modified": age_days,
                "risk_level": risk,
                "recommendation": "manual_review" if risk in ["high", "medium"] else "monitor"
            })
    
    # Sort by risk level and line count
    risk_order = {"high": 0, "medium": 1, "low": 2}
    cold_zones.sort(key=lambda x: (risk_order.get(x["risk_level"], 3), -x["line_count"]))
    
    return cold_zones


def detect_new_code_without_signals(
    modules: List[Dict],
    covered_modules: Set[str],
    new_code_window_days: int = 7
) -> Tuple[int, int, List[Dict]]:
    """
    Detect new code paths that don't have signal coverage.
    
    Returns: (total_new, uncovered_new, list of uncovered new modules)
    """
    now = datetime.now()
    cutoff = now - timedelta(days=new_code_window_days)
    
    new_modules = []
    uncovered_new = []
    
    for module in modules:
        try:
            created = datetime.fromisoformat(module.get("created_at", ""))
            if created >= cutoff:
                new_modules.append(module)
                
                module_name = module.get("module_name", "")
                is_covered = any(
                    module_name.startswith(covered) or covered.startswith(module_name)
                    for covered in covered_modules
                )
                
                if not is_covered:
                    uncovered_new.append(module)
        except:
            pass
    
    return len(new_modules), len(uncovered_new), uncovered_new


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_coverage_metrics(
    modules: List[Dict],
    signals: List[Dict],
    rules: List[str],
    skills: List[str]
) -> Dict[str, Any]:
    """
    Calculate all signal coverage metrics.
    """
    coverage = extract_signal_coverage(signals)
    cold_zones = detect_cold_zones(modules, coverage["modules"])
    
    total_new, uncovered_new, uncovered_new_list = detect_new_code_without_signals(
        modules, coverage["modules"]
    )
    
    # Calculate coverage ratios
    total_modules = len(modules)
    covered_modules = len(coverage["modules"])
    
    total_rules = len(rules)
    covered_rules = len(coverage["rules"] & set(rules))
    
    signal_types_all = {
        "rule_friction", "missing_skill", "ambiguous_guidance",
        "threshold_tension", "scope_escape", "tooling_gap"
    }
    covered_types = len(coverage["signal_types"] & signal_types_all)
    
    # Blind spot proxy
    blind_spot_proxy = uncovered_new / total_new if total_new > 0 else 0.0
    
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_modules": total_modules,
            "modules_with_signals": covered_modules,
            "module_coverage": covered_modules / total_modules if total_modules > 0 else 0.0,
            "total_rules": total_rules,
            "rules_with_signals": covered_rules,
            "rule_coverage": covered_rules / total_rules if total_rules > 0 else 0.0,
            "signal_type_coverage": covered_types / len(signal_types_all),
            "blind_spot_proxy": blind_spot_proxy,
            "cold_zone_count": len(cold_zones),
            "high_risk_cold_zones": len([z for z in cold_zones if z["risk_level"] == "high"]),
        },
        "cold_zones": cold_zones[:20],  # Top 20 by risk
        "uncovered_new_modules": [
            {"module": m["module_name"], "path": m["relative_path"], "lines": m["line_count"]}
            for m in uncovered_new_list
        ],
        "coverage_by_type": {
            "modules": list(coverage["modules"])[:10],
            "rules": list(coverage["rules"]),
            "signal_types": list(coverage["signal_types"]),
        },
        "health_status": _assess_health(blind_spot_proxy, len(cold_zones), covered_modules / total_modules if total_modules > 0 else 0)
    }


def _assess_health(blind_spot_proxy: float, cold_zone_count: int, module_coverage: float) -> Dict[str, Any]:
    """Assess overall health based on metrics."""
    issues = []
    status = "healthy"
    
    if blind_spot_proxy > 0.3:
        issues.append(f"High blind spot proxy: {blind_spot_proxy:.1%} > 30% threshold")
        status = "warning"
    if blind_spot_proxy > 0.5:
        status = "critical"
    
    if cold_zone_count > 20:
        issues.append(f"Many cold zones: {cold_zone_count} modules without signal coverage")
        if status != "critical":
            status = "warning"
    
    if module_coverage < 0.5:
        issues.append(f"Low module coverage: {module_coverage:.1%}")
        if status != "critical":
            status = "warning"
    
    return {
        "status": status,
        "issues": issues,
        "recommendation": "Review cold zones and consider adding proactive signal collection" if issues else "Coverage is adequate"
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(metrics: Dict[str, Any], project_id: str) -> str:
    """Generate markdown report from metrics."""
    summary = metrics["summary"]
    health = metrics["health_status"]
    
    status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🔴"}.get(health["status"], "❓")
    
    report = f"""# Signal Coverage Report

**Project**: {project_id}  
**Generated**: {metrics['timestamp']}  
**Status**: {status_emoji} {health['status'].upper()}

---

## Executive Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Module Coverage | {summary['module_coverage']:.1%} | ≥80% | {'✅' if summary['module_coverage'] >= 0.8 else '⚠️'} |
| Rule Coverage | {summary['rule_coverage']:.1%} | ≥80% | {'✅' if summary['rule_coverage'] >= 0.8 else '⚠️'} |
| Blind Spot Proxy | {summary['blind_spot_proxy']:.1%} | ≤20% | {'✅' if summary['blind_spot_proxy'] <= 0.2 else '⚠️'} |
| Cold Zones (High Risk) | {summary['high_risk_cold_zones']} | 0 | {'✅' if summary['high_risk_cold_zones'] == 0 else '⚠️'} |

---

## Health Issues

"""
    if health["issues"]:
        for issue in health["issues"]:
            report += f"- ⚠️ {issue}\n"
    else:
        report += "No issues detected.\n"
    
    report += f"""
**Recommendation**: {health['recommendation']}

---

## Cold Zones (Top 20 by Risk)

Modules without friction signal coverage in the past 30 days:

| Module | Lines | Risk | Recommendation |
|--------|-------|------|----------------|
"""
    
    for zone in metrics["cold_zones"][:20]:
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(zone["risk_level"], "⚪")
        report += f"| `{zone['module']}` | {zone['line_count']} | {risk_emoji} {zone['risk_level']} | {zone['recommendation']} |\n"
    
    if not metrics["cold_zones"]:
        report += "| (none) | - | - | - |\n"
    
    report += f"""
---

## New Code Without Coverage

Modules created in the last 7 days without signal coverage:

"""
    if metrics["uncovered_new_modules"]:
        for mod in metrics["uncovered_new_modules"]:
            report += f"- `{mod['module']}` ({mod['lines']} lines)\n"
    else:
        report += "All new modules have signal coverage. ✅\n"
    
    report += f"""
---

## Coverage Distribution

### By Signal Type

"""
    for sig_type in metrics["coverage_by_type"]["signal_types"]:
        report += f"- {sig_type}\n"
    
    if not metrics["coverage_by_type"]["signal_types"]:
        report += "No signals recorded.\n"
    
    report += f"""
### Rules with Signals

"""
    for rule in metrics["coverage_by_type"]["rules"]:
        report += f"- {rule}\n"
    
    if not metrics["coverage_by_type"]["rules"]:
        report += "No rule-specific signals recorded.\n"
    
    report += """
---

*Generated by AEP-6 Meta-Evolution Monitoring*
"""
    
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure evolution signal coverage (AEP-6)"
    )
    parser.add_argument(
        "--project", "-p",
        default="dgsf",
        help="Project ID (default: dgsf)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output report path (default: reports/signal_coverage_{date}.md)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "yaml", "json"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--cold-threshold",
        type=int,
        default=30,
        help="Days without signal to be considered cold (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load project adapter
    adapter_path = project_root / "projects" / args.project / "adapter.yaml"
    if not adapter_path.exists():
        print(f"❌ Project adapter not found: {adapter_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(adapter_path, 'r', encoding='utf-8') as f:
        adapter = yaml.safe_load(f)
    
    source_path = adapter.get("paths", {}).get("source", "repo/src")
    
    # Discover code modules
    project_dir = project_root / "projects" / args.project
    modules = discover_code_modules(project_dir, source_path)
    
    # Discover skills and rules
    prompts_dir = project_root / ".github" / "prompts"
    skills = discover_skills(prompts_dir)
    
    copilot_instructions = project_root / ".github" / "copilot-instructions.md"
    rules = discover_rules(copilot_instructions)
    
    # Load signals
    signals_file = project_dir / "evolution_signals.yaml"
    signals = load_evolution_signals(signals_file, window_days=args.cold_threshold)
    
    # Also check reports directory for manually generated signals
    reports_signals = project_root / "reports" / f"evolution_signals_{datetime.now().strftime('%Y-%m-%d')}.yaml"
    if reports_signals.exists():
        signals.extend(load_evolution_signals(reports_signals, window_days=args.cold_threshold))
    
    # Calculate metrics
    metrics = calculate_coverage_metrics(modules, signals, rules, skills)
    
    # Output
    if args.format == "yaml":
        output = yaml.dump(metrics, allow_unicode=True, sort_keys=False)
    elif args.format == "json":
        output = json.dumps(metrics, indent=2, default=str)
    else:
        output = generate_markdown_report(metrics, args.project)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "reports" / f"signal_coverage_{datetime.now().strftime('%Y-%m-%d')}.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"✅ Report generated: {output_path}")
    
    # Print summary
    summary = metrics["summary"]
    health = metrics["health_status"]
    status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🔴"}.get(health["status"], "❓")
    
    print(f"\n{status_emoji} Health Status: {health['status'].upper()}")
    print(f"   Module Coverage: {summary['module_coverage']:.1%}")
    print(f"   Blind Spot Proxy: {summary['blind_spot_proxy']:.1%}")
    print(f"   Cold Zones: {summary['cold_zone_count']} ({summary['high_risk_cold_zones']} high risk)")
    
    # Exit code based on health
    if health["status"] == "critical":
        sys.exit(2)
    elif health["status"] == "warning":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
