#!/usr/bin/env python3
"""
Evolution Signal Aggregator (Enhanced)

Aggregates evolution signals from project-level files and generates
review reports based on evolution_policy.yaml thresholds.

Part of AEP-2: Evolution Closed-loop Automation

Usage:
    python scripts/aggregate_evolution_signals.py [--project PROJECT_ID] [--generate-report]
    python scripts/aggregate_evolution_signals.py --check-thresholds
    python scripts/aggregate_evolution_signals.py --dashboard
    python scripts/aggregate_evolution_signals.py --path PATH  # Legacy mode
    
Output:
    - Aggregated signal statistics
    - Review reports (markdown)
    - Threshold alerts
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from kernel.yaml_utils import load_yaml, safe_load_yaml, safe_dump_yaml
except ImportError:
    import yaml
    def load_yaml(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    safe_load_yaml = load_yaml
    def safe_dump_yaml(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_evolution_policy() -> Optional[Dict[str, Any]]:
    """Load evolution policy configuration if it exists."""
    policy_path = PROJECT_ROOT / "configs" / "evolution_policy.yaml"
    if policy_path.exists():
        return safe_load_yaml(policy_path)
    return None


def load_project_signals(project_id: str) -> List[Dict[str, Any]]:
    """Load evolution signals for a specific project."""
    signals_path = PROJECT_ROOT / "projects" / project_id / "evolution_signals.yaml"
    if not signals_path.exists():
        return []
    
    data = safe_load_yaml(signals_path)
    if not data:
        return []
    
    # Support both 'signals' key and direct list
    return data.get("signals", data) if isinstance(data, dict) else data


def get_all_project_ids() -> List[str]:
    """Discover all project IDs from projects/ directory."""
    projects_dir = PROJECT_ROOT / "projects"
    if not projects_dir.exists():
        return []
    
    return [
        d.name for d in projects_dir.iterdir() 
        if d.is_dir() and (d / "adapter.yaml").exists()
    ]


# =============================================================================
# SIGNAL PROCESSING (Enhanced)
# =============================================================================

def calculate_severity_score(
    signals: List[Dict[str, Any]], 
    policy: Optional[Dict[str, Any]] = None
) -> float:
    """Calculate weighted severity score for signals."""
    if policy:
        severity_levels = policy.get("collection", {}).get("severity_levels", {})
        signal_types = {
            st["id"]: st.get("severity_weight", 1.0)
            for st in policy.get("collection", {}).get("signal_types", [])
        }
    else:
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 5}
        signal_types = {}
    
    total_score = 0.0
    for signal in signals:
        severity = signal.get("severity", "medium")
        signal_type = signal.get("signal_type", signal.get("type", "rule_friction"))
        
        severity_value = severity_levels.get(severity, 2)
        type_weight = signal_types.get(signal_type, 1.0)
        
        total_score += severity_value * type_weight
    
    return total_score


def filter_signals_by_window(
    signals: List[Dict[str, Any]], 
    window: str
) -> List[Dict[str, Any]]:
    """Filter signals within a time window."""
    if window.endswith("d"):
        days = int(window[:-1])
    elif window.endswith("w"):
        days = int(window[:-1]) * 7
    else:
        days = 7
    
    cutoff = datetime.now() - timedelta(days=days)
    
    filtered = []
    for signal in signals:
        timestamp_str = signal.get("timestamp", signal.get("date", ""))
        if timestamp_str:
            try:
                if "T" in str(timestamp_str):
                    timestamp = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
                else:
                    timestamp = datetime.strptime(str(timestamp_str), "%Y-%m-%d")
                
                if timestamp.replace(tzinfo=None) >= cutoff:
                    filtered.append(signal)
            except (ValueError, TypeError):
                filtered.append(signal)
        else:
            filtered.append(signal)
    
    return filtered


def check_review_thresholds(
    signals: List[Dict[str, Any]], 
    policy: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Check if any review thresholds are exceeded."""
    alerts = []
    
    # Group by rule
    by_rule = Counter(s.get('rule_id', s.get('rule', 'UNKNOWN')) for s in signals)
    
    # Check for high frequency rules (default: >= 3)
    threshold = 3
    if policy:
        threshold = (policy.get("aggregation", {})
                    .get("triggers", {})
                    .get("signal_count", {})
                    .get("threshold", 3))
    
    for rule, count in by_rule.items():
        if count >= threshold:
            alerts.append({
                "type": "aggregated",
                "reason": f"Rule {rule} friction: {count} signals (threshold: {threshold})",
                "action": "flag_for_rule_review",
                "rule_id": rule,
                "count": count
            })
    
    # Check for critical signals
    critical_signals = [s for s in signals if s.get("severity") == "critical"]
    for signal in critical_signals:
        alerts.append({
            "type": "immediate",
            "reason": f"Critical severity signal: {signal.get('rule_id', signal.get('rule', 'unknown'))}",
            "signal": signal
        })
    
    return alerts


# =============================================================================
# LEGACY AGGREGATION (backward compatible)
# =============================================================================


def aggregate_signals(yaml_path: Path) -> dict:
    """
    Legacy: Parse evolution_signals.yaml and aggregate statistics.
    
    Returns:
        {
            'by_rule': Counter({'R2': 3, 'R1': 1, ...}),
            'by_type': Counter({'false-positive': 2, ...}),
            'pending': [list of pending signals],
            'high_frequency': [rules with count >= 3]
        }
    """
    data = load_yaml(yaml_path)
    signals = data.get('signals', []) if isinstance(data, dict) else (data or [])
    
    if not signals:
        return {
            'by_rule': Counter(),
            'by_type': Counter(),
            'pending': [],
            'high_frequency': []
        }
    
    by_rule = Counter(s.get('rule', s.get('rule_id', 'UNKNOWN')) for s in signals)
    by_type = Counter(s.get('type', s.get('signal_type', 'unknown')) for s in signals)
    pending = [s for s in signals if s.get('resolution') == 'pending']
    high_frequency = [rule for rule, count in by_rule.items() if count >= 3]
    
    return {
        'by_rule': by_rule,
        'by_type': by_type,
        'pending': pending,
        'high_frequency': high_frequency
    }


def format_report(stats: dict) -> str:
    """格式化输出报告。"""
    lines = [
        "=" * 60,
        "EVOLUTION SIGNALS AGGREGATION REPORT",
        "=" * 60,
        "",
        "## Signal Count by Rule",
        ""
    ]
    
    if stats['by_rule']:
        for rule, count in stats['by_rule'].most_common():
            marker = " ⚠️ HIGH FREQUENCY" if rule in stats['high_frequency'] else ""
            lines.append(f"  {rule}: {count}{marker}")
    else:
        lines.append("  (no signals recorded)")
    
    lines.extend([
        "",
        "## Signal Count by Type",
        ""
    ])
    
    if stats['by_type']:
        for signal_type, count in stats['by_type'].most_common():
            lines.append(f"  {signal_type}: {count}")
    else:
        lines.append("  (no signals recorded)")
    
    lines.extend([
        "",
        "## Pending Signals (need attention)",
        ""
    ])
    
    if stats['pending']:
        for s in stats['pending']:
            rule = s.get('rule', s.get('rule_id', 'unknown'))
            date = s.get('date', s.get('timestamp', 'unknown'))
            context = s.get('context', 'no context')
            lines.append(f"  - [{rule}] {date}: {context}")
    else:
        lines.append("  (none pending)")
    
    lines.extend([
        "",
        "## Recommendations",
        ""
    ])
    
    if stats['high_frequency']:
        lines.append(f"  ⚠️ Rules with >= 3 occurrences: {', '.join(stats['high_frequency'])}")
        lines.append("     Consider triggering evolution review for these rules.")
    else:
        lines.append("  ✅ No rules have reached the review threshold (3 occurrences).")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# ENHANCED REPORT GENERATION
# =============================================================================

def generate_markdown_report(
    signals: List[Dict[str, Any]], 
    policy: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None
) -> str:
    """Generate a comprehensive markdown review report."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    
    severity_score = calculate_severity_score(signals, policy)
    alerts = check_review_thresholds(signals, policy)
    recent_signals = filter_signals_by_window(signals, "7d")
    
    # Group by type and rule
    by_type = defaultdict(list)
    by_rule = defaultdict(list)
    for s in signals:
        by_type[s.get("signal_type", s.get("type", "unknown"))].append(s)
        by_rule[s.get("rule_id", s.get("rule", "unknown"))].append(s)
    
    report = []
    report.append(f"# Evolution Signal Review Report")
    report.append(f"\n**Generated**: {now.isoformat()}")
    report.append(f"**Project**: {project_id or 'All Projects'}")
    report.append(f"**Period**: Last 7 days\n")
    
    # Summary
    report.append("## Summary Statistics\n")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Total Signals | {len(signals)} |")
    report.append(f"| Signals (Last 7d) | {len(recent_signals)} |")
    report.append(f"| Severity Score | {severity_score:.1f} |")
    report.append(f"| Active Alerts | {len(alerts)} |")
    report.append("")
    
    # Alerts
    if alerts:
        report.append("## ⚠️ Active Alerts\n")
        for alert in alerts:
            report.append(f"- **{alert['type'].title()}**: {alert['reason']}")
            if "action" in alert:
                report.append(f"  - Recommended: `{alert['action']}`")
        report.append("")
    
    # By Type
    report.append("## Signals by Type\n")
    report.append("| Type | Count |")
    report.append("|------|-------|")
    for signal_type, type_signals in sorted(by_type.items()):
        report.append(f"| {signal_type} | {len(type_signals)} |")
    report.append("")
    
    # By Rule
    report.append("## Signals by Rule\n")
    report.append("| Rule | Count | High Frequency |")
    report.append("|------|-------|----------------|")
    for rule, rule_signals in sorted(by_rule.items(), key=lambda x: -len(x[1])):
        hf = "⚠️ YES" if len(rule_signals) >= 3 else ""
        report.append(f"| {rule} | {len(rule_signals)} | {hf} |")
    report.append("")
    
    # Recent signals detail
    report.append("## Recent Signals (Last 7 Days)\n")
    if recent_signals:
        for s in recent_signals[:10]:
            report.append(f"### {s.get('signal_id', 'signal')}")
            report.append(f"- **Type**: {s.get('signal_type', s.get('type', 'unknown'))}")
            report.append(f"- **Rule**: {s.get('rule_id', s.get('rule', 'N/A'))}")
            report.append(f"- **Severity**: {s.get('severity', 'unknown')}")
            report.append(f"- **Context**: {s.get('context', 'No context')}")
            if s.get("suggested_evolution"):
                report.append(f"- **Suggested**: {s['suggested_evolution']}")
            report.append("")
    else:
        report.append("*No signals in the last 7 days.*\n")
    
    report.append("---")
    report.append(f"*Generated by aggregate_evolution_signals.py on {date_str}*")
    
    return "\n".join(report)


def save_report(report: str, output_path: Optional[Path] = None) -> Path:
    """Save report to file."""
    if output_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = PROJECT_ROOT / "reports" / f"evolution_review_{date_str}.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evolution signals and generate reports"
    )
    parser.add_argument(
        '--path',
        type=Path,
        default=None,
        help='Legacy: Path to evolution_signals.yaml'
    )
    parser.add_argument(
        '--project', '-p',
        help='Project ID to aggregate (default: all projects)'
    )
    parser.add_argument(
        '--generate-report', '-r',
        action='store_true',
        help='Generate markdown review report'
    )
    parser.add_argument(
        '--check-thresholds', '-t',
        action='store_true',
        help='Check if review thresholds are exceeded'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output path for report'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Generate dashboard summary'
    )
    
    args = parser.parse_args()
    
    # Legacy mode: single file path
    if args.path:
        if not args.path.exists():
            print(f"ERROR: File not found: {args.path}", file=sys.stderr)
            sys.exit(1)
        
        stats = aggregate_signals(args.path)
        print(format_report(stats))
        sys.exit(1 if stats['high_frequency'] else 0)
    
    # Enhanced mode: multi-project aggregation
    policy = load_evolution_policy()
    
    all_signals = []
    if args.project:
        project_ids = [args.project]
    else:
        project_ids = get_all_project_ids()
        if not project_ids:
            # Fallback to default dgsf
            project_ids = ["dgsf"]
    
    for project_id in project_ids:
        signals = load_project_signals(project_id)
        for signal in signals:
            signal["project_id"] = project_id
        all_signals.extend(signals)
    
    print(f"Loaded {len(all_signals)} signals from {len(project_ids)} project(s)")
    
    # Check thresholds
    if args.check_thresholds:
        alerts = check_review_thresholds(all_signals, policy)
        if alerts:
            print(f"\n⚠️  {len(alerts)} alert(s) triggered:")
            for alert in alerts:
                print(f"  - [{alert['type']}] {alert['reason']}")
            sys.exit(1)
        else:
            print("✓ No review thresholds exceeded")
            sys.exit(0)
    
    # Generate report
    if args.generate_report:
        report = generate_markdown_report(all_signals, policy, args.project)
        output_path = Path(args.output) if args.output else None
        saved_path = save_report(report, output_path)
        print(f"✓ Report saved to: {saved_path}")
    
    # Dashboard
    if args.dashboard:
        print("\n=== Evolution Dashboard ===")
        print(f"Total Signals: {len(all_signals)}")
        print(f"Severity Score: {calculate_severity_score(all_signals, policy):.1f}")
        
        recent = filter_signals_by_window(all_signals, "7d")
        print(f"Signals (Last 7d): {len(recent)}")
        
        by_type = defaultdict(int)
        for s in all_signals:
            by_type[s.get("signal_type", s.get("type", "unknown"))] += 1
        
        print("\nBy Type:")
        for signal_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {signal_type}: {count}")
    
    # If no specific action, run legacy report
    if not (args.generate_report or args.check_thresholds or args.dashboard):
        # Default: show legacy text report
        default_path = PROJECT_ROOT / 'projects' / 'dgsf' / 'evolution_signals.yaml'
        if default_path.exists():
            stats = aggregate_signals(default_path)
            print(format_report(stats))
            sys.exit(1 if stats['high_frequency'] else 0)
        else:
            print("No evolution_signals.yaml found. Use --generate-report to create a report.")
            sys.exit(0)


if __name__ == '__main__':
    main()
