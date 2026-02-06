"""
Destructive Operations Guard Module for Copilot Runtime OS.

Enforces safety guardrails for bulk delete, large refactor, and other
potentially dangerous operations.

Version: 1.0.0
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add kernel to path for imports
_KERNEL_DIR = Path(__file__).parent
_ROOT = _KERNEL_DIR.parent
sys.path.insert(0, str(_ROOT))

try:
    from kernel.paths import ROOT
except ImportError:
    ROOT = _ROOT


class OperationType(Enum):
    """Types of destructive operations."""
    DELETE = "delete"
    BULK_DELETE = "bulk_delete"
    REFACTOR = "refactor"
    LARGE_REFACTOR = "large_refactor"
    MOVE = "move"
    RENAME = "rename"
    CLEANUP = "cleanup"


class RiskLevel(Enum):
    """Risk levels for operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(Enum):
    """Approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class DestructiveOpRequest:
    """A request to perform a destructive operation."""
    request_id: str
    operation_type: OperationType
    risk_level: RiskLevel
    affected_files: List[str]
    reason: str
    status: ApprovalStatus
    created_at: datetime
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    backup_branch: Optional[str] = None
    rollback_plan: Optional[str] = None


def get_state_dir() -> Path:
    """Get the directory for destructive op state."""
    return ROOT / "state" / "destructive_ops"


def assess_risk(operation_type: OperationType, file_count: int) -> RiskLevel:
    """Assess the risk level of an operation.
    
    Args:
        operation_type: Type of operation.
        file_count: Number of affected files.
    
    Returns:
        Assessed RiskLevel.
    """
    # High-risk operations
    if operation_type in (OperationType.DELETE, OperationType.BULK_DELETE):
        if file_count >= 10:
            return RiskLevel.CRITICAL
        elif file_count >= 5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.MEDIUM
    
    # Medium-risk operations
    if operation_type in (OperationType.REFACTOR, OperationType.LARGE_REFACTOR):
        if file_count >= 10:
            return RiskLevel.HIGH
        elif file_count >= 5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    # Low-risk operations
    return RiskLevel.LOW


def create_request(
    operation_type: str,
    affected_files: List[str],
    reason: str,
) -> DestructiveOpRequest:
    """Create a destructive operation request.
    
    Args:
        operation_type: Type of operation (delete, refactor, etc.).
        affected_files: List of affected file paths.
        reason: Explanation for why this operation is needed.
    
    Returns:
        Created request.
    """
    now = datetime.now(timezone.utc)
    op_type = OperationType(operation_type)
    risk_level = assess_risk(op_type, len(affected_files))
    
    request_id = f"DO-{now.strftime('%Y%m%d-%H%M%S')}"
    
    request = DestructiveOpRequest(
        request_id=request_id,
        operation_type=op_type,
        risk_level=risk_level,
        affected_files=affected_files,
        reason=reason,
        status=ApprovalStatus.PENDING,
        created_at=now,
    )
    
    # Generate rollback plan
    request.rollback_plan = _generate_rollback_plan(request)
    
    # Save to file
    state_dir = get_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    
    request_path = state_dir / f"{request_id}.yaml"
    _save_request(request, request_path)
    
    return request


def _generate_rollback_plan(request: DestructiveOpRequest) -> str:
    """Generate a rollback plan for an operation."""
    plans = []
    
    if request.operation_type in (OperationType.DELETE, OperationType.BULK_DELETE):
        plans.append(f"1. Restore from backup branch: {request.backup_branch or 'TBD'}")
        plans.append("2. Or restore from git reflog:")
        plans.append("   git reflog")
        plans.append("   git checkout <commit-before-delete> -- <files>")
    
    elif request.operation_type in (OperationType.REFACTOR, OperationType.LARGE_REFACTOR):
        plans.append(f"1. Reset to backup branch: git reset --hard {request.backup_branch or 'TBD'}")
        plans.append("2. Or revert specific commit: git revert <commit>")
    
    else:
        plans.append("1. Use git revert or checkout to restore previous state")
    
    return "\n".join(plans)


def _save_request(request: DestructiveOpRequest, path: Path) -> None:
    """Save a request to YAML file."""
    data = {
        "request_id": request.request_id,
        "operation_type": request.operation_type.value,
        "risk_level": request.risk_level.value,
        "affected_files": request.affected_files,
        "affected_count": len(request.affected_files),
        "reason": request.reason,
        "status": request.status.value,
        "created_at": request.created_at.isoformat(),
        "approved_at": request.approved_at.isoformat() if request.approved_at else None,
        "approved_by": request.approved_by,
        "backup_branch": request.backup_branch,
        "rollback_plan": request.rollback_plan,
    }
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def _load_request(path: Path) -> DestructiveOpRequest:
    """Load a request from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return DestructiveOpRequest(
        request_id=data["request_id"],
        operation_type=OperationType(data["operation_type"]),
        risk_level=RiskLevel(data["risk_level"]),
        affected_files=data.get("affected_files", []),
        reason=data.get("reason", ""),
        status=ApprovalStatus(data["status"]),
        created_at=datetime.fromisoformat(data["created_at"]),
        approved_at=datetime.fromisoformat(data["approved_at"]) if data.get("approved_at") else None,
        approved_by=data.get("approved_by"),
        backup_branch=data.get("backup_branch"),
        rollback_plan=data.get("rollback_plan"),
    )


def approve_request(request_id: str, approved_by: str = "user") -> Optional[DestructiveOpRequest]:
    """Approve a pending request.
    
    Args:
        request_id: The request ID to approve.
        approved_by: Who is approving.
    
    Returns:
        Updated request, or None if not found.
    """
    state_dir = get_state_dir()
    request_path = state_dir / f"{request_id}.yaml"
    
    if not request_path.exists():
        return None
    
    request = _load_request(request_path)
    
    if request.status != ApprovalStatus.PENDING:
        return request  # Already processed
    
    now = datetime.now(timezone.utc)
    request.status = ApprovalStatus.APPROVED
    request.approved_at = now
    request.approved_by = approved_by
    
    # Create backup branch
    backup_branch = f"backup/{request.operation_type.value}/{now.strftime('%Y%m%d_%H%M%S')}"
    try:
        subprocess.run(
            ["git", "branch", backup_branch],
            capture_output=True,
            cwd=str(ROOT),
        )
        request.backup_branch = backup_branch
    except Exception:
        pass
    
    _save_request(request, request_path)
    return request


def list_pending() -> List[DestructiveOpRequest]:
    """List all pending requests."""
    state_dir = get_state_dir()
    pending = []
    
    if not state_dir.exists():
        return pending
    
    for path in state_dir.glob("DO-*.yaml"):
        try:
            request = _load_request(path)
            if request.status == ApprovalStatus.PENDING:
                pending.append(request)
        except Exception:
            continue
    
    return sorted(pending, key=lambda x: x.created_at, reverse=True)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for destructive ops management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Destructive Operations Guard")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # request command
    req_parser = subparsers.add_parser("request", help="Create a request")
    req_parser.add_argument("--type", required=True, 
                           choices=["delete", "bulk_delete", "refactor", "large_refactor", "move", "rename"])
    req_parser.add_argument("--files", required=True, help="Comma-separated file list or count")
    req_parser.add_argument("--reason", required=True, help="Reason for operation")
    
    # approve command
    approve_parser = subparsers.add_parser("approve", help="Approve a request")
    approve_parser.add_argument("request_id", help="Request ID to approve")
    approve_parser.add_argument("--by", default="user", help="Approver name")
    
    # list command
    subparsers.add_parser("list", help="List pending requests")
    
    args = parser.parse_args()
    
    if args.command == "request":
        # Parse files
        if args.files.isdigit():
            files = [f"file_{i}" for i in range(int(args.files))]
        else:
            files = [f.strip() for f in args.files.split(",")]
        
        request = create_request(args.type, files, args.reason)
        
        print(f"✅ Created request: {request.request_id}")
        print(f"   Type: {request.operation_type.value}")
        print(f"   Risk: {request.risk_level.value}")
        print(f"   Files: {len(request.affected_files)}")
        print(f"   Status: {request.status.value}")
        print(f"\n   To approve: python kernel/destructive_ops.py approve {request.request_id}")
    
    elif args.command == "approve":
        request = approve_request(args.request_id, args.by)
        if request:
            print(f"✅ Approved: {request.request_id}")
            print(f"   By: {request.approved_by}")
            print(f"   Backup: {request.backup_branch}")
        else:
            print(f"❌ Request not found: {args.request_id}")
    
    elif args.command == "list":
        pending = list_pending()
        if not pending:
            print("📭 No pending requests")
        else:
            print(f"📋 Pending Requests ({len(pending)}):\n")
            for r in pending:
                risk_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}[r.risk_level.value]
                print(f"  {risk_icon} [{r.request_id}]")
                print(f"     Type: {r.operation_type.value} | Files: {len(r.affected_files)}")
                print(f"     Reason: {r.reason[:50]}...")
                print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
