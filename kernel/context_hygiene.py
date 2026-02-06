#!/usr/bin/env python3
"""
Context Hygiene Manager — Token Threshold & Auto-Delegation

Purpose: Detect context overload conditions and trigger automatic delegation
         to subagents to maintain context hygiene.

Enforcement for: PP-020 (Context Hygiene 无强制委托)

Usage:
    python kernel/context_hygiene.py assess --files 15 --tokens 80000
    python kernel/context_hygiene.py checkpoint --session-id S001
    python kernel/context_hygiene.py restore --session-id S001
    python kernel/context_hygiene.py recommend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
_KERNEL_DIR = Path(__file__).parent
_ROOT = _KERNEL_DIR.parent
sys.path.insert(0, str(_ROOT))

try:
    from kernel.paths import ROOT
except ImportError:
    ROOT = _ROOT

# -----------------------------------------------------------------------------
# Constants & Thresholds
# -----------------------------------------------------------------------------
CONTEXT_STATE_PATH = ROOT / "state" / "context_hygiene.yaml"
CHECKPOINT_DIR = ROOT / "state" / "context_checkpoints"

# Default thresholds (can be overridden via config)
DEFAULT_THRESHOLDS = {
    "token_warning": 50000,       # Warn when approaching limit
    "token_critical": 80000,      # Force delegation when exceeded
    "file_count_warning": 10,     # Warn with many files in context
    "file_count_critical": 20,    # Force delegation with too many files
    "depth_warning": 5,           # Call stack depth warning
    "depth_critical": 10,         # Force checkpoint at this depth
}

# Subagent recommendations based on context type
DELEGATION_RULES = {
    "spec_heavy": ["repo_specs_retrieval"],
    "research_heavy": ["external_research", "repo_specs_retrieval"],
    "code_review": ["quant_risk_review"],
    "cross_layer": ["repo_specs_retrieval", "spec_drift"],
    "general_overload": ["repo_specs_retrieval"],
}


class ContextStatus(Enum):
    """Context health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERLOADED = "overloaded"


class DelegationAction(Enum):
    """Recommended delegation actions."""
    NONE = "none"
    SUGGEST = "suggest"         # Suggest delegation
    RECOMMEND = "recommend"     # Strongly recommend
    FORCE = "force"             # Force delegation or checkpoint


@dataclass
class ContextMetrics:
    """Current context metrics."""
    estimated_tokens: int = 0
    file_count: int = 0
    unique_directories: int = 0
    call_depth: int = 0
    session_duration_minutes: int = 0
    files_accessed: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "estimated_tokens": self.estimated_tokens,
            "file_count": self.file_count,
            "unique_directories": self.unique_directories,
            "call_depth": self.call_depth,
            "session_duration_minutes": self.session_duration_minutes,
            "files_accessed": self.files_accessed[:20],  # Limit stored files
            "topics": self.topics,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ContextMetrics":
        return cls(
            estimated_tokens=data.get("estimated_tokens", 0),
            file_count=data.get("file_count", 0),
            unique_directories=data.get("unique_directories", 0),
            call_depth=data.get("call_depth", 0),
            session_duration_minutes=data.get("session_duration_minutes", 0),
            files_accessed=data.get("files_accessed", []),
            topics=data.get("topics", []),
        )


@dataclass
class ContextAssessment:
    """Assessment of context hygiene."""
    status: ContextStatus
    action: DelegationAction
    metrics: ContextMetrics
    reasons: list[str] = field(default_factory=list)
    recommended_subagents: list[str] = field(default_factory=list)
    checkpoint_suggested: bool = False
    assessed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "action": self.action.value,
            "metrics": self.metrics.to_dict(),
            "reasons": self.reasons,
            "recommended_subagents": self.recommended_subagents,
            "checkpoint_suggested": self.checkpoint_suggested,
            "assessed_at": self.assessed_at,
        }


@dataclass
class ContextCheckpoint:
    """Saved context state for restoration."""
    session_id: str
    checkpoint_id: str
    created_at: str
    metrics: ContextMetrics
    summary: str
    files_snapshot: list[str]
    key_findings: list[str]
    pending_tasks: list[str]
    restoration_notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "metrics": self.metrics.to_dict(),
            "summary": self.summary,
            "files_snapshot": self.files_snapshot,
            "key_findings": self.key_findings,
            "pending_tasks": self.pending_tasks,
            "restoration_notes": self.restoration_notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ContextCheckpoint":
        return cls(
            session_id=data["session_id"],
            checkpoint_id=data["checkpoint_id"],
            created_at=data["created_at"],
            metrics=ContextMetrics.from_dict(data.get("metrics", {})),
            summary=data.get("summary", ""),
            files_snapshot=data.get("files_snapshot", []),
            key_findings=data.get("key_findings", []),
            pending_tasks=data.get("pending_tasks", []),
            restoration_notes=data.get("restoration_notes", ""),
        )


# -----------------------------------------------------------------------------
# Threshold Loading
# -----------------------------------------------------------------------------

def load_thresholds() -> dict:
    """Load thresholds from config or use defaults."""
    config_path = ROOT / "configs" / "operating_modes.yaml"
    thresholds = DEFAULT_THRESHOLDS.copy()
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            
            if "context_hygiene" in config:
                thresholds.update(config["context_hygiene"].get("thresholds", {}))
        except Exception:
            pass
    
    return thresholds


# -----------------------------------------------------------------------------
# Core Assessment Logic
# -----------------------------------------------------------------------------

def assess_context(metrics: ContextMetrics) -> ContextAssessment:
    """
    Assess context health and recommend actions.
    
    This is the core enforcement mechanism for PP-020.
    """
    thresholds = load_thresholds()
    
    status = ContextStatus.HEALTHY
    action = DelegationAction.NONE
    reasons = []
    recommended_subagents = []
    checkpoint_suggested = False
    
    # Token assessment
    if metrics.estimated_tokens >= thresholds["token_critical"]:
        status = ContextStatus.CRITICAL
        action = DelegationAction.FORCE
        reasons.append(f"Token count ({metrics.estimated_tokens}) exceeds critical threshold ({thresholds['token_critical']})")
        checkpoint_suggested = True
    elif metrics.estimated_tokens >= thresholds["token_warning"]:
        if status.value == "healthy":
            status = ContextStatus.WARNING
        if action.value == "none":
            action = DelegationAction.SUGGEST
        reasons.append(f"Token count ({metrics.estimated_tokens}) approaching limit ({thresholds['token_warning']})")
    
    # File count assessment
    if metrics.file_count >= thresholds["file_count_critical"]:
        status = ContextStatus.CRITICAL
        action = DelegationAction.FORCE
        reasons.append(f"File count ({metrics.file_count}) exceeds critical threshold ({thresholds['file_count_critical']})")
        recommended_subagents.extend(["repo_specs_retrieval"])
        checkpoint_suggested = True
    elif metrics.file_count >= thresholds["file_count_warning"]:
        if status.value == "healthy":
            status = ContextStatus.WARNING
        if action.value == "none":
            action = DelegationAction.SUGGEST
        reasons.append(f"File count ({metrics.file_count}) is high ({thresholds['file_count_warning']})")
        recommended_subagents.extend(["repo_specs_retrieval"])
    
    # Call depth assessment
    if metrics.call_depth >= thresholds["depth_critical"]:
        status = ContextStatus.OVERLOADED
        action = DelegationAction.FORCE
        reasons.append(f"Call depth ({metrics.call_depth}) exceeds critical threshold ({thresholds['depth_critical']})")
        checkpoint_suggested = True
    elif metrics.call_depth >= thresholds["depth_warning"]:
        if status.value == "healthy":
            status = ContextStatus.WARNING
        reasons.append(f"Call depth ({metrics.call_depth}) is deep ({thresholds['depth_warning']})")
    
    # Topic-based recommendations
    topics = set(t.lower() for t in metrics.topics)
    if "spec" in topics or "specification" in topics:
        recommended_subagents.extend(DELEGATION_RULES["spec_heavy"])
    if "research" in topics or "paper" in topics or "literature" in topics:
        recommended_subagents.extend(DELEGATION_RULES["research_heavy"])
    if "review" in topics or "audit" in topics:
        recommended_subagents.extend(DELEGATION_RULES["code_review"])
    if len(topics) > 3:
        recommended_subagents.extend(DELEGATION_RULES["general_overload"])
    
    # Deduplicate subagents
    recommended_subagents = list(dict.fromkeys(recommended_subagents))
    
    return ContextAssessment(
        status=status,
        action=action,
        metrics=metrics,
        reasons=reasons,
        recommended_subagents=recommended_subagents,
        checkpoint_suggested=checkpoint_suggested,
    )


def should_delegate(metrics: ContextMetrics) -> tuple[bool, list[str], str]:
    """
    Quick check: should we delegate to subagent?
    
    Returns:
        (should_delegate, subagent_ids, reason)
    """
    assessment = assess_context(metrics)
    
    if assessment.action == DelegationAction.FORCE:
        return True, assessment.recommended_subagents, "; ".join(assessment.reasons)
    elif assessment.action == DelegationAction.RECOMMEND:
        return True, assessment.recommended_subagents, "; ".join(assessment.reasons)
    
    return False, [], ""


# -----------------------------------------------------------------------------
# Checkpoint Management
# -----------------------------------------------------------------------------

def create_checkpoint(
    session_id: str,
    metrics: ContextMetrics,
    summary: str,
    files_snapshot: list[str],
    key_findings: list[str],
    pending_tasks: list[str],
) -> ContextCheckpoint:
    """Create and save a context checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate checkpoint ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content_hash = hashlib.md5(summary.encode()).hexdigest()[:8]
    checkpoint_id = f"CP-{timestamp}-{content_hash}"
    
    checkpoint = ContextCheckpoint(
        session_id=session_id,
        checkpoint_id=checkpoint_id,
        created_at=datetime.now().isoformat(),
        metrics=metrics,
        summary=summary,
        files_snapshot=files_snapshot,
        key_findings=key_findings,
        pending_tasks=pending_tasks,
    )
    
    # Save to file
    checkpoint_path = CHECKPOINT_DIR / f"{session_id}_{checkpoint_id}.yaml"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        yaml.dump(checkpoint.to_dict(), f, allow_unicode=True, default_flow_style=False)
    
    # Update index
    update_checkpoint_index(checkpoint)
    
    return checkpoint


def restore_checkpoint(session_id: str, checkpoint_id: Optional[str] = None) -> Optional[ContextCheckpoint]:
    """Restore a context checkpoint."""
    if not CHECKPOINT_DIR.exists():
        return None
    
    # Find checkpoint file
    if checkpoint_id:
        pattern = f"{session_id}_{checkpoint_id}.yaml"
    else:
        # Get latest checkpoint for session
        pattern = f"{session_id}_*.yaml"
    
    matching_files = list(CHECKPOINT_DIR.glob(pattern))
    if not matching_files:
        return None
    
    # Use latest if multiple matches
    latest = max(matching_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return ContextCheckpoint.from_dict(data)


def list_checkpoints(session_id: Optional[str] = None) -> list[dict]:
    """List available checkpoints."""
    if not CHECKPOINT_DIR.exists():
        return []
    
    pattern = f"{session_id}_*.yaml" if session_id else "*.yaml"
    checkpoints = []
    
    for path in CHECKPOINT_DIR.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            checkpoints.append({
                "session_id": data.get("session_id"),
                "checkpoint_id": data.get("checkpoint_id"),
                "created_at": data.get("created_at"),
                "summary": data.get("summary", "")[:100],
            })
        except Exception:
            continue
    
    return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)


def update_checkpoint_index(checkpoint: ContextCheckpoint) -> None:
    """Update the checkpoint index file."""
    index_path = CHECKPOINT_DIR / "INDEX.md"
    
    # Load existing or create new
    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        lines = content.split("\n")
    else:
        lines = [
            "# Context Checkpoint Index",
            "",
            "> Auto-generated by `kernel/context_hygiene.py`",
            "",
            "---",
            "",
            "| Session | Checkpoint | Created | Summary |",
            "|---------|------------|---------|---------|",
        ]
    
    # Add new entry after header
    entry_line = f"| `{checkpoint.session_id}` | `{checkpoint.checkpoint_id}` | {checkpoint.created_at[:19]} | {checkpoint.summary[:50]}... |"
    
    # Find table and insert after header
    for i, line in enumerate(lines):
        if line.startswith("|------"):
            lines.insert(i + 1, entry_line)
            break
    
    index_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# State Persistence
# -----------------------------------------------------------------------------

def save_context_state(assessment: ContextAssessment) -> None:
    """Save current context assessment to state file."""
    CONTEXT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTEXT_STATE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(assessment.to_dict(), f, allow_unicode=True, default_flow_style=False)


def load_context_state() -> Optional[ContextAssessment]:
    """Load previous context assessment."""
    if not CONTEXT_STATE_PATH.exists():
        return None
    
    with open(CONTEXT_STATE_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not data:
        return None
    
    return ContextAssessment(
        status=ContextStatus(data["status"]),
        action=DelegationAction(data["action"]),
        metrics=ContextMetrics.from_dict(data.get("metrics", {})),
        reasons=data.get("reasons", []),
        recommended_subagents=data.get("recommended_subagents", []),
        checkpoint_suggested=data.get("checkpoint_suggested", False),
        assessed_at=data.get("assessed_at", ""),
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Context Hygiene Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # assess
    assess_parser = subparsers.add_parser("assess", help="Assess current context health")
    assess_parser.add_argument("--tokens", type=int, default=0, help="Estimated token count")
    assess_parser.add_argument("--files", type=int, default=0, help="Number of files in context")
    assess_parser.add_argument("--depth", type=int, default=0, help="Call depth")
    assess_parser.add_argument("--topics", nargs="*", default=[], help="Current topics")
    assess_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # checkpoint
    cp_parser = subparsers.add_parser("checkpoint", help="Create a context checkpoint")
    cp_parser.add_argument("--session-id", required=True, help="Session ID")
    cp_parser.add_argument("--summary", default="", help="Session summary")
    cp_parser.add_argument("--findings", nargs="*", default=[], help="Key findings")
    cp_parser.add_argument("--pending", nargs="*", default=[], help="Pending tasks")
    
    # restore
    restore_parser = subparsers.add_parser("restore", help="Restore a checkpoint")
    restore_parser.add_argument("--session-id", required=True, help="Session ID")
    restore_parser.add_argument("--checkpoint-id", help="Specific checkpoint ID")
    
    # list
    list_parser = subparsers.add_parser("list", help="List checkpoints")
    list_parser.add_argument("--session-id", help="Filter by session ID")
    
    # recommend
    subparsers.add_parser("recommend", help="Get delegation recommendations")
    
    # status
    subparsers.add_parser("status", help="Show current context status")
    
    args = parser.parse_args()
    
    if args.command == "assess":
        metrics = ContextMetrics(
            estimated_tokens=args.tokens,
            file_count=args.files,
            call_depth=args.depth,
            topics=args.topics,
        )
        
        assessment = assess_context(metrics)
        save_context_state(assessment)
        
        if args.json:
            print(json.dumps(assessment.to_dict(), indent=2))
        else:
            print(f"📊 Context Assessment")
            print(f"   Status: {assessment.status.value.upper()}")
            print(f"   Action: {assessment.action.value}")
            
            if assessment.reasons:
                print(f"\n⚠️  Reasons:")
                for r in assessment.reasons:
                    print(f"   • {r}")
            
            if assessment.recommended_subagents:
                print(f"\n🤖 Recommended Subagents:")
                for s in assessment.recommended_subagents:
                    print(f"   • {s}")
            
            if assessment.checkpoint_suggested:
                print(f"\n💾 Checkpoint recommended!")
        
        # Exit with non-zero if critical
        if assessment.status == ContextStatus.CRITICAL:
            sys.exit(1)
        sys.exit(0)
    
    elif args.command == "checkpoint":
        # Get current assessment for metrics
        prev_state = load_context_state()
        metrics = prev_state.metrics if prev_state else ContextMetrics()
        
        checkpoint = create_checkpoint(
            session_id=args.session_id,
            metrics=metrics,
            summary=args.summary or "Manual checkpoint",
            files_snapshot=[],
            key_findings=args.findings,
            pending_tasks=args.pending,
        )
        
        print(f"✅ Created checkpoint: {checkpoint.checkpoint_id}")
        print(f"   Session: {checkpoint.session_id}")
        print(f"   Path: {CHECKPOINT_DIR / f'{checkpoint.session_id}_{checkpoint.checkpoint_id}.yaml'}")
        sys.exit(0)
    
    elif args.command == "restore":
        checkpoint = restore_checkpoint(args.session_id, args.checkpoint_id)
        
        if not checkpoint:
            print(f"❌ No checkpoint found for session '{args.session_id}'")
            sys.exit(1)
        
        print(f"📂 Restored Checkpoint: {checkpoint.checkpoint_id}")
        print(f"   Created: {checkpoint.created_at}")
        print(f"   Summary: {checkpoint.summary}")
        
        if checkpoint.key_findings:
            print(f"\n🔍 Key Findings:")
            for f in checkpoint.key_findings:
                print(f"   • {f}")
        
        if checkpoint.pending_tasks:
            print(f"\n📋 Pending Tasks:")
            for t in checkpoint.pending_tasks:
                print(f"   • {t}")
        
        print(f"\n📊 Context at Checkpoint:")
        print(f"   Tokens: {checkpoint.metrics.estimated_tokens}")
        print(f"   Files: {checkpoint.metrics.file_count}")
        
        sys.exit(0)
    
    elif args.command == "list":
        checkpoints = list_checkpoints(args.session_id)
        
        if not checkpoints:
            print("No checkpoints found.")
            sys.exit(0)
        
        print(f"{'Session':<15} {'Checkpoint':<25} {'Created':<20} {'Summary':<40}")
        print("-" * 100)
        for cp in checkpoints:
            print(f"{cp['session_id']:<15} {cp['checkpoint_id']:<25} {cp['created_at'][:19]:<20} {cp['summary']:<40}")
        sys.exit(0)
    
    elif args.command == "recommend":
        prev_state = load_context_state()
        
        if not prev_state:
            print("No context assessment available. Run 'assess' first.")
            sys.exit(1)
        
        should, subagents, reason = should_delegate(prev_state.metrics)
        
        if should:
            print(f"🚨 DELEGATION RECOMMENDED")
            print(f"   Reason: {reason}")
            print(f"\n🤖 Delegate to:")
            for s in subagents:
                print(f"   • {s}")
        else:
            print(f"✅ Context is healthy, no delegation needed.")
        
        sys.exit(0)
    
    elif args.command == "status":
        prev_state = load_context_state()
        
        if not prev_state:
            print("📊 No context assessment recorded yet.")
            print("   Run: python kernel/context_hygiene.py assess --tokens N --files M")
            sys.exit(0)
        
        status_icons = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "🔴",
            "overloaded": "💥",
        }
        
        icon = status_icons.get(prev_state.status.value, "❓")
        print(f"{icon} Context Status: {prev_state.status.value.upper()}")
        print(f"   Tokens: ~{prev_state.metrics.estimated_tokens}")
        print(f"   Files: {prev_state.metrics.file_count}")
        print(f"   Assessed: {prev_state.assessed_at[:19]}")
        
        if prev_state.action != DelegationAction.NONE:
            print(f"\n   Recommended Action: {prev_state.action.value}")
        
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
