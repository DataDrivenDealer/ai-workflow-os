#!/usr/bin/env python3
"""
Evolution Signal Collector (Enhanced)

Provides automatic capture of rule friction during agent execution.
Signals are categorized by type and aggregated for human review.

Part of AEP-2: Now integrates with evolution_policy.yaml for threshold-based
alerting and automatic review triggering.

Usage:
    from evolution_signal import EvolutionSignalCollector
    
    collector = EvolutionSignalCollector()
    collector.log_friction(
        rule_id="R2",
        context="Needed to run 2 tasks in parallel for A/B comparison",
        severity="medium",
        suggested_evolution="Parameterize R2 for controlled parallelism"
    )
"""

import yaml
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
from dataclasses import dataclass, asdict
import hashlib

# =============================================================================
# SIGNAL TYPES
# =============================================================================

SignalType = Literal[
    "rule_friction",      # Rule blocked legitimate action
    "missing_skill",      # Needed capability not available
    "ambiguous_guidance", # Instruction unclear
    "threshold_tension",  # Success criteria too strict/loose
    "scope_escape",       # Task needed to touch out-of-scope area
    "tooling_gap",        # Missing automation/script
    # AEP-6: Meta-monitoring signal types
    "meta_blind_spot",    # Evolution system not monitoring an area
    "meta_slow_velocity", # Signal-to-action time exceeds threshold
    "meta_high_noise",    # Too many non-actionable signals
    "meta_coverage_gap",  # Code path without signal coverage
    "meta_regression",    # Evolution caused regression
]

Severity = Literal["low", "medium", "high", "critical"]

# AEP-6: Confidence scoring for evolution proposals
ConfidenceLevel = Literal["low", "medium", "high", "very_high"]

CONFIDENCE_THRESHOLDS = {
    "very_high": {"min_signals": 5, "min_agreement": 0.9, "recurrence_days": 30},
    "high": {"min_signals": 3, "min_agreement": 0.7, "recurrence_days": 14},
    "medium": {"min_signals": 2, "min_agreement": 0.5, "recurrence_days": 7},
    "low": {"min_signals": 1, "min_agreement": 0.0, "recurrence_days": 0},
}

@dataclass
class EvolutionSignal:
    """Structured evolution signal for audit and aggregation."""
    signal_id: str
    signal_type: SignalType
    timestamp: str
    rule_id: Optional[str]
    context: str
    severity: Severity
    suggested_evolution: Optional[str]
    session_id: Optional[str]
    experiment_id: Optional[str]
    project_id: Optional[str] = None
    status: str = "new"  # new, aggregated, reviewed, actioned, dismissed
    # AEP-6: Meta-monitoring fields
    confidence: Optional[str] = None  # low, medium, high, very_high
    related_signals: Optional[List[str]] = None  # IDs of related signals
    evidence_links: Optional[List[str]] = None  # File paths / URLs as evidence
    
    def to_dict(self):
        return asdict(self)


@dataclass
class EvolutionProposal:
    """AEP-6: Structured evolution proposal with confidence scoring."""
    proposal_id: str
    title: str
    description: str
    confidence: ConfidenceLevel
    supporting_signals: List[str]  # Signal IDs
    affected_files: List[str]
    estimated_impact: Literal["low", "medium", "high"]
    created_at: str
    status: str = "draft"  # draft, proposed, approved, rejected, implemented
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ThresholdAlert:
    """Alert triggered when evolution thresholds are exceeded."""
    alert_type: str  # "immediate" | "aggregated"
    reason: str
    rule_id: Optional[str]
    signal_count: int
    action: str
    triggered_at: str

# =============================================================================
# POLICY LOADER
# =============================================================================

def load_evolution_policy(project_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load evolution policy configuration if it exists."""
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    policy_path = project_root / "configs" / "evolution_policy.yaml"
    if policy_path.exists():
        with open(policy_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None


# =============================================================================
# COLLECTOR
# =============================================================================

class EvolutionSignalCollector:
    """Collects and persists evolution signals for later human review.
    
    Enhanced with AEP-2 features:
    - Policy-based threshold checking
    - Automatic alert generation
    - Multi-project support
    """
    
    def __init__(
        self, 
        signals_file: str = "projects/dgsf/evolution_signals.yaml",
        project_id: str = "dgsf"
    ):
        self.signals_file = Path(signals_file)
        self.signals_file.parent.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id
        self._policy = None
        
    @property
    def policy(self) -> Optional[Dict[str, Any]]:
        """Lazy-load evolution policy."""
        if self._policy is None:
            project_root = self.signals_file.parent.parent.parent
            self._policy = load_evolution_policy(project_root)
        return self._policy
        
    def _generate_signal_id(self, context: str) -> str:
        """Generate deterministic signal ID for deduplication."""
        hash_input = f"{context}".encode()
        return f"SIG-{hashlib.sha256(hash_input).hexdigest()[:8].upper()}"
    
    def _load_existing_signals(self) -> list:
        """Load existing signals from file."""
        if self.signals_file.exists():
            with open(self.signals_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                return data.get('signals', [])
        return []
    
    def _save_signals(self, signals: list):
        """Atomically save signals to file."""
        data = {
            'version': '1.0.0',
            'last_updated': datetime.now().isoformat(),
            'signals': signals
        }
        temp_file = self.signals_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        temp_file.replace(self.signals_file)
    
    def log_friction(
        self,
        signal_type: SignalType,
        context: str,
        severity: Severity,
        rule_id: Optional[str] = None,
        suggested_evolution: Optional[str] = None,
        session_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> str:
        """
        Log an evolution signal.
        
        Returns:
            Signal ID for reference
        """
        signal = EvolutionSignal(
            signal_id=self._generate_signal_id(context),
            signal_type=signal_type,
            timestamp=datetime.now().isoformat(),
            rule_id=rule_id,
            context=context,
            severity=severity,
            suggested_evolution=suggested_evolution,
            session_id=session_id,
            experiment_id=experiment_id,
            project_id=self.project_id,
            status="new",
        )
        
        signals = self._load_existing_signals()
        
        # Deduplicate by signal_id
        existing_ids = {s.get('signal_id') for s in signals}
        if signal.signal_id not in existing_ids:
            signals.append(signal.to_dict())
            self._save_signals(signals)
            
            # Check thresholds after logging
            self._check_thresholds_and_alert(signals, signal)
            
        return signal.signal_id
    
    def _check_thresholds_and_alert(
        self, 
        signals: List[Dict], 
        new_signal: EvolutionSignal
    ) -> Optional[ThresholdAlert]:
        """Check if thresholds are exceeded and generate alert if needed."""
        policy = self.policy
        if not policy:
            return None
        
        # Check immediate alert conditions
        immediate_conditions = (
            policy.get("review", {})
            .get("alert_conditions", {})
            .get("immediate", [])
        )
        
        for condition in immediate_conditions:
            if "severity" in condition and new_signal.severity == condition["severity"]:
                alert = ThresholdAlert(
                    alert_type="immediate",
                    reason=f"Critical severity signal logged",
                    rule_id=new_signal.rule_id,
                    signal_count=1,
                    action="immediate_review",
                    triggered_at=datetime.now().isoformat()
                )
                self._log_alert(alert)
                return alert
        
        # Check aggregated thresholds
        threshold_config = (
            policy.get("aggregation", {})
            .get("triggers", {})
            .get("signal_count", {})
        )
        
        if threshold_config.get("enabled", False):
            threshold = threshold_config.get("threshold", 5)
            window = threshold_config.get("window", "7d")
            
            recent_signals = self._filter_by_window(signals, window)
            
            if len(recent_signals) >= threshold:
                alert = ThresholdAlert(
                    alert_type="aggregated",
                    reason=f"Signal count threshold reached: {len(recent_signals)} >= {threshold}",
                    rule_id=None,
                    signal_count=len(recent_signals),
                    action="trigger_review",
                    triggered_at=datetime.now().isoformat()
                )
                self._log_alert(alert)
                return alert
        
        return None
    
    def _filter_by_window(self, signals: List[Dict], window: str) -> List[Dict]:
        """Filter signals within a time window."""
        if window.endswith("d"):
            days = int(window[:-1])
        elif window.endswith("w"):
            days = int(window[:-1]) * 7
        else:
            days = 7
        
        cutoff = datetime.now() - timedelta(days=days)
        
        filtered = []
        for sig in signals:
            ts_str = sig.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts.replace(tzinfo=None) >= cutoff:
                        filtered.append(sig)
                except (ValueError, TypeError):
                    filtered.append(sig)
            else:
                filtered.append(sig)
        
        return filtered
    
    def _log_alert(self, alert: ThresholdAlert):
        """Log alert to stderr for visibility."""
        import sys
        print(
            f"⚠️  EVOLUTION ALERT [{alert.alert_type}]: {alert.reason}",
            file=sys.stderr
        )
        if alert.action:
            print(f"   Recommended action: {alert.action}", file=sys.stderr)
    
    def get_aggregated_report(self) -> dict:
        """
        Aggregate signals for human review.
        
        Returns:
            Dict with signals grouped by type and rule, with frequency counts
        """
        signals = self._load_existing_signals()
        
        by_type = {}
        by_rule = {}
        by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for sig in signals:
            # By type
            sig_type = sig.get('signal_type', 'unknown')
            if sig_type not in by_type:
                by_type[sig_type] = []
            by_type[sig_type].append(sig)
            
            # By rule
            rule_id = sig.get('rule_id')
            if rule_id:
                if rule_id not in by_rule:
                    by_rule[rule_id] = []
                by_rule[rule_id].append(sig)
            
            # By severity
            sev = sig.get('severity', 'low')
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {
            'total_signals': len(signals),
            'by_severity': by_severity,
            'by_type': {k: len(v) for k, v in by_type.items()},
            'by_rule': {k: len(v) for k, v in by_rule.items()},
            'top_friction_rules': sorted(
                by_rule.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:5],
            'high_priority_signals': [
                s for s in signals 
                if s.get('severity') in ['high', 'critical']
            ],
        }
    
    # =========================================================================
    # AEP-6: CONFIDENCE SCORING
    # =========================================================================
    
    def calculate_confidence(
        self, 
        signal_ids: List[str],
        recurrence_check: bool = True
    ) -> ConfidenceLevel:
        """
        Calculate confidence level for an evolution proposal based on
        supporting signals.
        
        Args:
            signal_ids: List of signal IDs supporting the proposal
            recurrence_check: Whether to check for recurring patterns
            
        Returns:
            Confidence level: low, medium, high, or very_high
        """
        signals = self._load_existing_signals()
        supporting = [s for s in signals if s.get('signal_id') in signal_ids]
        
        if not supporting:
            return "low"
        
        # Count signals
        signal_count = len(supporting)
        
        # Check agreement (signals pointing to same evolution suggestion)
        suggestions = [s.get('suggested_evolution') for s in supporting if s.get('suggested_evolution')]
        if suggestions:
            # Simple similarity: count most common suggestion
            from collections import Counter
            counts = Counter(suggestions)
            most_common_count = counts.most_common(1)[0][1]
            agreement_ratio = most_common_count / len(suggestions)
        else:
            agreement_ratio = 0.0
        
        # Check recurrence over time
        recurrence_days = 0
        if recurrence_check and len(supporting) > 1:
            timestamps = []
            for s in supporting:
                ts_str = s.get('timestamp', '')
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                        timestamps.append(ts.replace(tzinfo=None))
                    except (ValueError, TypeError):
                        pass
            if len(timestamps) >= 2:
                timestamps.sort()
                recurrence_days = (timestamps[-1] - timestamps[0]).days
        
        # Determine confidence level
        for level in ["very_high", "high", "medium", "low"]:
            thresholds = CONFIDENCE_THRESHOLDS[level]
            if (signal_count >= thresholds["min_signals"] and
                agreement_ratio >= thresholds["min_agreement"] and
                recurrence_days >= thresholds["recurrence_days"]):
                return level
        
        return "low"
    
    def create_proposal(
        self,
        title: str,
        description: str,
        signal_ids: List[str],
        affected_files: List[str],
        estimated_impact: Literal["low", "medium", "high"] = "medium"
    ) -> EvolutionProposal:
        """
        Create an evolution proposal with automatic confidence scoring.
        
        Args:
            title: Proposal title
            description: Detailed description
            signal_ids: Supporting signal IDs
            affected_files: Files that would be modified
            estimated_impact: Expected impact level
            
        Returns:
            EvolutionProposal with calculated confidence
        """
        proposal_id = f"PROP-{hashlib.sha256(title.encode()).hexdigest()[:8].upper()}"
        confidence = self.calculate_confidence(signal_ids)
        
        proposal = EvolutionProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            confidence=confidence,
            supporting_signals=signal_ids,
            affected_files=affected_files,
            estimated_impact=estimated_impact,
            created_at=datetime.now().isoformat(),
            status="draft"
        )
        
        # Save proposal
        self._save_proposal(proposal)
        
        return proposal
    
    def _save_proposal(self, proposal: EvolutionProposal):
        """Save proposal to proposals file."""
        proposals_file = self.signals_file.parent / "evolution_proposals.yaml"
        
        if proposals_file.exists():
            with open(proposals_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {'proposals': []}
        
        # Add or update proposal
        existing_ids = [p.get('proposal_id') for p in data.get('proposals', [])]
        if proposal.proposal_id in existing_ids:
            # Update existing
            for i, p in enumerate(data['proposals']):
                if p.get('proposal_id') == proposal.proposal_id:
                    data['proposals'][i] = proposal.to_dict()
                    break
        else:
            data['proposals'].append(proposal.to_dict())
        
        data['last_updated'] = datetime.now().isoformat()
        
        with open(proposals_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    
    def log_meta_signal(
        self,
        signal_type: SignalType,
        context: str,
        severity: Severity,
        evidence_links: Optional[List[str]] = None,
        related_signals: Optional[List[str]] = None
    ) -> str:
        """
        Log a meta-monitoring signal (AEP-6).
        
        These signals are about the evolution system itself.
        
        Args:
            signal_type: Must be a meta_* type
            context: Description of the issue
            severity: low, medium, high, critical
            evidence_links: File paths or URLs as evidence
            related_signals: IDs of related signals
            
        Returns:
            Signal ID
        """
        valid_meta_types = [
            "meta_blind_spot", "meta_slow_velocity", 
            "meta_high_noise", "meta_coverage_gap", "meta_regression"
        ]
        
        if signal_type not in valid_meta_types:
            raise ValueError(f"signal_type must be one of {valid_meta_types}")
        
        signal = EvolutionSignal(
            signal_id=self._generate_signal_id(f"meta:{context}"),
            signal_type=signal_type,
            timestamp=datetime.now().isoformat(),
            rule_id=None,
            context=context,
            severity=severity,
            suggested_evolution=None,
            session_id=None,
            experiment_id=None,
            project_id=self.project_id,
            status="new",
            confidence=None,
            related_signals=related_signals,
            evidence_links=evidence_links
        )
        
        signals = self._load_existing_signals()
        
        existing_ids = {s.get('signal_id') for s in signals}
        if signal.signal_id not in existing_ids:
            signals.append(signal.to_dict())
            self._save_signals(signals)
            
        return signal.signal_id
    
    def get_meta_health_summary(self) -> Dict[str, Any]:
        """
        Get meta-monitoring health summary for the evolution system.
        
        Returns:
            Dict with meta-health metrics
        """
        signals = self._load_existing_signals()
        
        # Filter meta signals
        meta_signals = [s for s in signals if s.get('signal_type', '').startswith('meta_')]
        
        # Count by type
        meta_by_type = {}
        for sig in meta_signals:
            sig_type = sig.get('signal_type')
            meta_by_type[sig_type] = meta_by_type.get(sig_type, 0) + 1
        
        # Count by severity
        meta_by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for sig in meta_signals:
            sev = sig.get('severity', 'low')
            meta_by_severity[sev] = meta_by_severity.get(sev, 0) + 1
        
        # Calculate health score
        health_score = 100
        health_score -= meta_by_severity["critical"] * 20
        health_score -= meta_by_severity["high"] * 10
        health_score -= meta_by_severity["medium"] * 5
        health_score -= meta_by_severity["low"] * 2
        health_score = max(0, min(100, health_score))
        
        return {
            "total_meta_signals": len(meta_signals),
            "by_type": meta_by_type,
            "by_severity": meta_by_severity,
            "health_score": health_score,
            "needs_attention": health_score < 70 or meta_by_severity["critical"] > 0
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evolution Signal Collector")
    subparsers = parser.add_subparsers(dest='command')
    
    # Log command
    log_parser = subparsers.add_parser('log', help='Log a new signal')
    log_parser.add_argument('--type', required=True, choices=[
        'rule_friction', 'missing_skill', 'ambiguous_guidance',
        'threshold_tension', 'scope_escape', 'tooling_gap',
        # AEP-6: Meta signal types
        'meta_blind_spot', 'meta_slow_velocity', 'meta_high_noise',
        'meta_coverage_gap', 'meta_regression'
    ])
    log_parser.add_argument('--context', required=True)
    log_parser.add_argument('--severity', required=True, choices=['low', 'medium', 'high', 'critical'])
    log_parser.add_argument('--rule', help='Related rule ID (e.g., R2)')
    log_parser.add_argument('--suggestion', help='Suggested evolution')
    log_parser.add_argument('--evidence', nargs='*', help='Evidence links (file paths or URLs)')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate aggregated report')
    report_parser.add_argument('--format', choices=['yaml', 'summary'], default='summary')
    
    # AEP-6: Meta health command
    meta_parser = subparsers.add_parser('meta-health', help='Show meta-monitoring health')
    meta_parser.add_argument('--format', choices=['yaml', 'summary'], default='summary')
    
    # AEP-6: Propose command
    propose_parser = subparsers.add_parser('propose', help='Create evolution proposal')
    propose_parser.add_argument('--title', required=True, help='Proposal title')
    propose_parser.add_argument('--description', required=True, help='Proposal description')
    propose_parser.add_argument('--signals', nargs='+', required=True, help='Supporting signal IDs')
    propose_parser.add_argument('--files', nargs='+', required=True, help='Affected files')
    propose_parser.add_argument('--impact', choices=['low', 'medium', 'high'], default='medium')
    
    args = parser.parse_args()
    
    collector = EvolutionSignalCollector()
    
    if args.command == 'log':
        # Check if it's a meta signal
        if args.type.startswith('meta_'):
            signal_id = collector.log_meta_signal(
                signal_type=args.type,
                context=args.context,
                severity=args.severity,
                evidence_links=args.evidence,
            )
        else:
            signal_id = collector.log_friction(
                signal_type=args.type,
                context=args.context,
                severity=args.severity,
                rule_id=args.rule,
                suggested_evolution=args.suggestion,
            )
        print(f"Logged signal: {signal_id}")
        
    elif args.command == 'report':
        report = collector.get_aggregated_report()
        if args.format == 'yaml':
            print(yaml.dump(report, allow_unicode=True, sort_keys=False))
        else:
            print(f"=== Evolution Signal Report ===")
            print(f"Total signals: {report['total_signals']}")
            print(f"\nBy severity:")
            for sev, count in report['by_severity'].items():
                print(f"  {sev}: {count}")
            print(f"\nTop friction rules:")
            for rule, signals in report['top_friction_rules']:
                print(f"  {rule}: {len(signals)} signals")
            print(f"\nHigh priority signals: {len(report['high_priority_signals'])}")
    
    elif args.command == 'meta-health':
        health = collector.get_meta_health_summary()
        if args.format == 'yaml':
            print(yaml.dump(health, allow_unicode=True, sort_keys=False))
        else:
            print(f"=== Meta-Evolution Health ===")
            print(f"Total meta signals: {health['total_meta_signals']}")
            print(f"Health score: {health['health_score']}/100")
            print(f"\nBy type:")
            for sig_type, count in health['by_type'].items():
                print(f"  {sig_type}: {count}")
            print(f"\nBy severity:")
            for sev, count in health['by_severity'].items():
                print(f"  {sev}: {count}")
            if health['needs_attention']:
                print(f"\n⚠️  Meta-evolution system needs attention!")
    
    elif args.command == 'propose':
        proposal = collector.create_proposal(
            title=args.title,
            description=args.description,
            signal_ids=args.signals,
            affected_files=args.files,
            estimated_impact=args.impact
        )
        print(f"Created proposal: {proposal.proposal_id}")
        print(f"  Title: {proposal.title}")
        print(f"  Confidence: {proposal.confidence}")
        print(f"  Status: {proposal.status}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
