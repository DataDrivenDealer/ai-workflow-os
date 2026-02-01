"""
Governance Gate Module

Implements the 5-dimensional governance verification system as defined in:
- GOVERNANCE_INVARIANTS
- ROLE_MODE_CANON
- AUTHORITY_CANON
- MULTI_AGENT_CANON

This module provides stateless verification that execution does not violate
governance constraints. It does NOT:
- Produce pass/fail status
- Correct violations
- Suggest remediation
- Trigger governance actions

Its sole function is to DETECT governance violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class ViolationType(Enum):
    """Categories of governance violations."""
    AUTHORITY_INTEGRITY = "authority_integrity"
    ROLE_MODE_INTEGRITY = "role_mode_integrity"
    STATE_INTEGRITY = "state_integrity"
    WORKFLOW_SPINE_INTEGRITY = "workflow_spine_integrity"
    CONCURRENCY_ISOLATION = "concurrency_isolation"


@dataclass
class Violation:
    """Represents a detected governance violation."""
    violation_type: ViolationType
    rule_id: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.value,
            "rule_id": self.rule_id,
            "description": self.description,
            "context": self.context,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class GateResult:
    """Result of governance gate verification."""
    gate_name: str
    violations: List[Violation] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate_name,
            "has_violations": self.has_violations,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "checked_at": self.checked_at.isoformat(),
        }


class GovernanceGate:
    """
    Governance verification gate implementing 5-dimensional checks.
    
    Verification Dimensions:
    1. Authority Integrity - No execution output claims authority
    2. Role Mode Integrity - Actions attributable to single Role Mode
    3. State Integrity - State reconstructable from artifacts only
    4. Workflow Spine Integrity - No implicit stage progression
    5. Concurrency Isolation - Parallel executions don't combine authority
    """
    
    # Keywords that indicate authority claims in outputs
    AUTHORITY_CLAIM_KEYWORDS = {
        "i accept", "accepted", "i approve", "approve this", "approved",
        "i freeze", "frozen", "this is authoritative",
        "binding decision", "i authorize", "authorized",
        "i grant", "granted authority", "hereby approve", "hereby accept",
    }
    
    # Keywords indicating governance actions
    GOVERNANCE_ACTION_KEYWORDS = {
        "supersede", "superseded", "revoke", "revoked",
        "override", "overridden", "commit state",
    }
    
    def __init__(self, state_machine_path: Optional[Path] = None):
        """Initialize the governance gate with state machine configuration."""
        self.state_machine_path = state_machine_path or Path(__file__).parent / "state_machine.yaml"
        self._config: Optional[Dict[str, Any]] = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Load and cache state machine configuration."""
        if self._config is None:
            if self.state_machine_path.exists():
                with self.state_machine_path.open("r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            else:
                self._config = {}
        return self._config
    
    def verify_all(
        self,
        output_text: Optional[str] = None,
        agent_session: Optional[Dict[str, Any]] = None,
        artifact_changes: Optional[List[Dict[str, Any]]] = None,
        state_mutations: Optional[List[Dict[str, Any]]] = None,
        concurrent_sessions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[GateResult]:
        """
        Run all 5 governance gate checks.
        
        Args:
            output_text: Text output from execution to check for authority claims
            agent_session: Current agent session info (agent_id, role_mode, etc.)
            artifact_changes: List of artifact modifications attempted
            state_mutations: List of state changes attempted
            concurrent_sessions: Other active sessions (for concurrency check)
        
        Returns:
            List of GateResult for each dimension
        """
        results = []
        
        results.append(self.check_authority_integrity(
            output_text=output_text,
            artifact_changes=artifact_changes,
        ))
        
        results.append(self.check_role_mode_integrity(
            agent_session=agent_session,
            artifact_changes=artifact_changes,
        ))
        
        results.append(self.check_state_integrity(
            state_mutations=state_mutations,
            agent_session=agent_session,
        ))
        
        results.append(self.check_workflow_spine_integrity(
            artifact_changes=artifact_changes,
            state_mutations=state_mutations,
        ))
        
        results.append(self.check_concurrency_isolation(
            agent_session=agent_session,
            concurrent_sessions=concurrent_sessions,
            artifact_changes=artifact_changes,
        ))
        
        return results
    
    def check_authority_integrity(
        self,
        output_text: Optional[str] = None,
        artifact_changes: Optional[List[Dict[str, Any]]] = None,
    ) -> GateResult:
        """
        Dimension 1: Authority Integrity
        
        Checks:
        - No execution output claims authority
        - No governance action is implied
        - No artifact is treated as binding without explicit record
        """
        result = GateResult(gate_name="authority_integrity")
        
        # Check output text for authority claims
        if output_text:
            text_lower = output_text.lower()
            
            for keyword in self.AUTHORITY_CLAIM_KEYWORDS:
                if keyword in text_lower:
                    result.violations.append(Violation(
                        violation_type=ViolationType.AUTHORITY_INTEGRITY,
                        rule_id="AUTH-CLAIM-001",
                        description=f"Output contains authority claim keyword: '{keyword}'",
                        context={"keyword": keyword, "output_snippet": output_text[:200]},
                    ))
            
            for keyword in self.GOVERNANCE_ACTION_KEYWORDS:
                if keyword in text_lower:
                    result.violations.append(Violation(
                        violation_type=ViolationType.AUTHORITY_INTEGRITY,
                        rule_id="AUTH-ACTION-001",
                        description=f"Output implies governance action: '{keyword}'",
                        context={"keyword": keyword},
                    ))
        
        # Check artifact changes for unauthorized authority changes
        if artifact_changes:
            for change in artifact_changes:
                if change.get("authority_state") in ("accepted", "frozen"):
                    if not change.get("authorized_by"):
                        result.violations.append(Violation(
                            violation_type=ViolationType.AUTHORITY_INTEGRITY,
                            rule_id="AUTH-UNAUTH-001",
                            description="Artifact authority change without authorization record",
                            context={"artifact": change.get("path")},
                        ))
        
        return result
    
    def check_role_mode_integrity(
        self,
        agent_session: Optional[Dict[str, Any]] = None,
        artifact_changes: Optional[List[Dict[str, Any]]] = None,
    ) -> GateResult:
        """
        Dimension 2: Role Mode Integrity
        
        Checks:
        - Each action attributable to exactly one Role Mode
        - No implicit Role Mode switching
        - No escalation inferred from execution
        """
        result = GateResult(gate_name="role_mode_integrity")
        
        if not agent_session:
            result.violations.append(Violation(
                violation_type=ViolationType.ROLE_MODE_INTEGRITY,
                rule_id="ROLE-SESSION-001",
                description="No agent session provided - action not attributable",
                context={},
            ))
            return result
        
        role_mode = agent_session.get("role_mode")
        if not role_mode:
            result.violations.append(Violation(
                violation_type=ViolationType.ROLE_MODE_INTEGRITY,
                rule_id="ROLE-MODE-001",
                description="No Role Mode specified in session",
                context={"session": agent_session},
            ))
            return result
        
        # Check if role mode is valid
        valid_modes = set(self.config.get("role_modes", {}).keys())
        if role_mode not in valid_modes:
            result.violations.append(Violation(
                violation_type=ViolationType.ROLE_MODE_INTEGRITY,
                rule_id="ROLE-INVALID-001",
                description=f"Invalid Role Mode: {role_mode}",
                context={"role_mode": role_mode, "valid_modes": list(valid_modes)},
            ))
            return result
        
        # Check artifact changes against role mode permissions
        if artifact_changes:
            role_config = self.config.get("role_modes", {}).get(role_mode, {})
            prohibitions = set(role_config.get("prohibitions", []))
            
            for change in artifact_changes:
                artifact_path = change.get("path", "")
                
                # Check L0/L1 modification restrictions
                if "specs/canon" in artifact_path or "l0" in artifact_path.lower():
                    if "modify_l0_artifacts" in prohibitions:
                        result.violations.append(Violation(
                            violation_type=ViolationType.ROLE_MODE_INTEGRITY,
                            rule_id="ROLE-PROHIB-001",
                            description=f"Role Mode '{role_mode}' cannot modify L0 artifacts",
                            context={"artifact": artifact_path, "role_mode": role_mode},
                        ))
                
                if "specs/framework" in artifact_path or "l1" in artifact_path.lower():
                    if "modify_l1_artifacts" in prohibitions:
                        result.violations.append(Violation(
                            violation_type=ViolationType.ROLE_MODE_INTEGRITY,
                            rule_id="ROLE-PROHIB-002",
                            description=f"Role Mode '{role_mode}' cannot modify L1 artifacts",
                            context={"artifact": artifact_path, "role_mode": role_mode},
                        ))
                
                # Check accept/freeze restrictions
                if change.get("action") in ("accept", "freeze"):
                    if "accept_artifacts" in prohibitions or "freeze_artifacts" in prohibitions:
                        result.violations.append(Violation(
                            violation_type=ViolationType.ROLE_MODE_INTEGRITY,
                            rule_id="ROLE-PROHIB-003",
                            description=f"Role Mode '{role_mode}' cannot accept/freeze artifacts",
                            context={"artifact": artifact_path, "action": change.get("action")},
                        ))
        
        return result
    
    def check_state_integrity(
        self,
        state_mutations: Optional[List[Dict[str, Any]]] = None,
        agent_session: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        """
        Dimension 3: State Integrity
        
        Checks:
        - System state reconstructable from artifacts only
        - No execution-side memory treated as state
        - No implicit state transition
        """
        result = GateResult(gate_name="state_integrity")
        
        if state_mutations:
            for mutation in state_mutations:
                # Check if mutation is backed by artifact
                if not mutation.get("artifact_backed"):
                    result.violations.append(Violation(
                        violation_type=ViolationType.STATE_INTEGRITY,
                        rule_id="STATE-ARTIFACT-001",
                        description="State mutation without artifact backing",
                        context={"mutation": mutation},
                    ))
                
                # Check if mutation is explicit
                if mutation.get("implicit"):
                    result.violations.append(Violation(
                        violation_type=ViolationType.STATE_INTEGRITY,
                        rule_id="STATE-IMPLICIT-001",
                        description="Implicit state mutation detected",
                        context={"mutation": mutation},
                    ))
                
                # Check if transition is valid per state machine
                if mutation.get("type") == "task_state":
                    from_state = mutation.get("from")
                    to_state = mutation.get("to")
                    valid_transitions = self.config.get("transitions", [])
                    
                    is_valid = any(
                        t.get("from") == from_state and t.get("to") == to_state
                        for t in valid_transitions
                    )
                    
                    if not is_valid:
                        result.violations.append(Violation(
                            violation_type=ViolationType.STATE_INTEGRITY,
                            rule_id="STATE-TRANS-001",
                            description=f"Invalid state transition: {from_state} → {to_state}",
                            context={"from": from_state, "to": to_state},
                        ))
        
        return result
    
    def check_workflow_spine_integrity(
        self,
        artifact_changes: Optional[List[Dict[str, Any]]] = None,
        state_mutations: Optional[List[Dict[str, Any]]] = None,
    ) -> GateResult:
        """
        Dimension 4: Workflow Spine Integrity
        
        Checks:
        - No implicit stage progression
        - Execution does not produce validation effect
        - No inferred completion
        """
        result = GateResult(gate_name="workflow_spine_integrity")
        
        if state_mutations:
            for mutation in state_mutations:
                # Check for execution-validation collapse
                if mutation.get("type") == "task_state":
                    if mutation.get("to") == "reviewing" and mutation.get("self_validated"):
                        result.violations.append(Violation(
                            violation_type=ViolationType.WORKFLOW_SPINE_INTEGRITY,
                            rule_id="SPINE-COLLAPSE-001",
                            description="Execution-validation collapse: self-validation detected",
                            context={"mutation": mutation},
                        ))
                
                # Check for implicit completion
                if mutation.get("inferred_completion"):
                    result.violations.append(Violation(
                        violation_type=ViolationType.WORKFLOW_SPINE_INTEGRITY,
                        rule_id="SPINE-IMPLICIT-001",
                        description="Implicit completion inferred without explicit governance action",
                        context={"mutation": mutation},
                    ))
        
        return result
    
    def check_concurrency_isolation(
        self,
        agent_session: Optional[Dict[str, Any]] = None,
        concurrent_sessions: Optional[List[Dict[str, Any]]] = None,
        artifact_changes: Optional[List[Dict[str, Any]]] = None,
    ) -> GateResult:
        """
        Dimension 5: Concurrency Isolation
        
        Checks:
        - Parallel executions do not combine or amplify authority
        - Conflicting drafts do not override authoritative artifacts
        - No collective legitimacy emerges from concurrency
        """
        result = GateResult(gate_name="concurrency_isolation")
        
        if not concurrent_sessions or not artifact_changes:
            return result
        
        # Get artifacts being modified by current session
        current_artifacts = {
            change.get("path") for change in artifact_changes if change.get("path")
        }
        
        # Check for conflicts with other sessions
        for other_session in concurrent_sessions:
            if other_session.get("session_token") == agent_session.get("session_token"):
                continue
            
            other_artifacts = set(other_session.get("pending_artifacts", []))
            conflicts = current_artifacts & other_artifacts
            
            if conflicts:
                result.violations.append(Violation(
                    violation_type=ViolationType.CONCURRENCY_ISOLATION,
                    rule_id="CONCUR-CONFLICT-001",
                    description="Concurrent modification of same artifacts detected",
                    context={
                        "conflicting_artifacts": list(conflicts),
                        "current_session": agent_session.get("session_token"),
                        "other_session": other_session.get("session_token"),
                    },
                ))
        
        # Check for authority amplification attempts
        if artifact_changes:
            for change in artifact_changes:
                if change.get("claims_collective_authority"):
                    result.violations.append(Violation(
                        violation_type=ViolationType.CONCURRENCY_ISOLATION,
                        rule_id="CONCUR-COLLECTIVE-001",
                        description="Attempt to claim collective authority from concurrent execution",
                        context={"artifact": change.get("path")},
                    ))
        
        return result


def verify_governance(
    output_text: Optional[str] = None,
    agent_session: Optional[Dict[str, Any]] = None,
    artifact_changes: Optional[List[Dict[str, Any]]] = None,
    state_mutations: Optional[List[Dict[str, Any]]] = None,
    concurrent_sessions: Optional[List[Dict[str, Any]]] = None,
) -> List[GateResult]:
    """
    Convenience function to run all governance checks.
    
    Returns list of GateResult for each dimension.
    """
    gate = GovernanceGate()
    return gate.verify_all(
        output_text=output_text,
        agent_session=agent_session,
        artifact_changes=artifact_changes,
        state_mutations=state_mutations,
        concurrent_sessions=concurrent_sessions,
    )


def has_violations(results: List[GateResult]) -> bool:
    """Check if any gate result has violations."""
    return any(r.has_violations for r in results)


def get_all_violations(results: List[GateResult]) -> List[Violation]:
    """Get all violations from all gate results."""
    violations = []
    for result in results:
        violations.extend(result.violations)
    return violations


if __name__ == "__main__":
    # Example usage
    import json
    
    # Simulate an execution check
    results = verify_governance(
        output_text="I have completed the task. The changes are ready for review.",
        agent_session={
            "agent_id": "claude-001",
            "role_mode": "executor",
            "session_token": "sess-12345",
        },
        artifact_changes=[
            {"path": "tasks/TASK_001.md", "action": "modify"},
        ],
    )
    
    print("Governance Gate Results:")
    print("=" * 50)
    for result in results:
        status = "❌ VIOLATIONS" if result.has_violations else "✅ PASS"
        print(f"{result.gate_name}: {status}")
        if result.has_violations:
            for v in result.violations:
                print(f"  - [{v.rule_id}] {v.description}")
