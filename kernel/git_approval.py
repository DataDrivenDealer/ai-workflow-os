"""
Git Approval Artifact Module for Copilot Runtime OS.

Enforces approval artifacts for CONFIRM/BLOCK level Git operations.
No artifact = No commit for sensitive changes.

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
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


class ApprovalStatus(Enum):
    """Approval status for Git operations."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class GitApproval:
    """Represents a Git operation approval artifact."""
    approval_id: str
    commit_plan_hash: str
    files: List[str]
    confirm_level: str  # CONFIRM, BLOCK
    status: ApprovalStatus
    created_at: datetime
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    commit_message: Optional[str] = None
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None


def get_approval_dir() -> Path:
    """Get the directory for approval artifacts."""
    return ROOT / "state" / "git_approvals"


def compute_plan_hash(files: List[str], commit_message: str) -> str:
    """Compute a hash for a commit plan.
    
    Args:
        files: List of files to be committed.
        commit_message: The proposed commit message.
    
    Returns:
        SHA256 hash of the plan.
    """
    content = json.dumps({
        "files": sorted(files),
        "message": commit_message,
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_approval_request(
    files: List[str],
    commit_message: str,
    confirm_level: str,
    reason: Optional[str] = None,
) -> GitApproval:
    """Create an approval request artifact.
    
    Args:
        files: Files to be committed.
        commit_message: Proposed commit message.
        confirm_level: CONFIRM or BLOCK.
        reason: Why this commit requires approval.
    
    Returns:
        Created GitApproval.
    """
    now = datetime.now(timezone.utc)
    plan_hash = compute_plan_hash(files, commit_message)
    approval_id = f"GA-{now.strftime('%Y%m%d-%H%M%S')}-{plan_hash[:8]}"
    
    approval = GitApproval(
        approval_id=approval_id,
        commit_plan_hash=plan_hash,
        files=files,
        confirm_level=confirm_level,
        status=ApprovalStatus.PENDING,
        created_at=now,
        commit_message=commit_message,
        reason=reason,
    )
    
    # Save to file
    approval_dir = get_approval_dir()
    approval_dir.mkdir(parents=True, exist_ok=True)
    
    approval_path = approval_dir / f"{approval_id}.yaml"
    with open(approval_path, "w", encoding="utf-8") as f:
        data = {
            "approval_id": approval.approval_id,
            "commit_plan_hash": approval.commit_plan_hash,
            "files": approval.files,
            "confirm_level": approval.confirm_level,
            "status": approval.status.value,
            "created_at": approval.created_at.isoformat(),
            "commit_message": approval.commit_message,
            "reason": approval.reason,
            "approved_at": None,
            "approved_by": None,
        }
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    return approval


def approve_request(
    approval_id: str,
    approved_by: str = "user",
) -> Optional[GitApproval]:
    """Approve a pending request.
    
    Args:
        approval_id: The approval ID to approve.
        approved_by: Who is approving.
    
    Returns:
        Updated GitApproval, or None if not found.
    """
    approval_dir = get_approval_dir()
    approval_path = approval_dir / f"{approval_id}.yaml"
    
    if not approval_path.exists():
        return None
    
    with open(approval_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    now = datetime.now(timezone.utc)
    data["status"] = ApprovalStatus.APPROVED.value
    data["approved_at"] = now.isoformat()
    data["approved_by"] = approved_by
    
    with open(approval_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    return _load_approval(data)


def check_approval_exists(files: List[str], commit_message: str) -> Dict[str, Any]:
    """Check if a valid approval exists for a commit plan.
    
    Args:
        files: Files to be committed.
        commit_message: Commit message.
    
    Returns:
        Dict with 'approved' (bool), 'approval_id' (str), 'reason' (str).
    """
    plan_hash = compute_plan_hash(files, commit_message)
    approval_dir = get_approval_dir()
    
    if not approval_dir.exists():
        return {
            "approved": False,
            "approval_id": None,
            "reason": "No approval directory exists",
        }
    
    # Look for matching approval
    for approval_file in approval_dir.glob("GA-*.yaml"):
        try:
            with open(approval_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if data.get("commit_plan_hash") == plan_hash:
                if data.get("status") == ApprovalStatus.APPROVED.value:
                    return {
                        "approved": True,
                        "approval_id": data.get("approval_id"),
                        "approved_by": data.get("approved_by"),
                        "approved_at": data.get("approved_at"),
                    }
                elif data.get("status") == ApprovalStatus.PENDING.value:
                    return {
                        "approved": False,
                        "approval_id": data.get("approval_id"),
                        "reason": "Approval is pending - awaiting human confirmation",
                    }
                elif data.get("status") == ApprovalStatus.REJECTED.value:
                    return {
                        "approved": False,
                        "approval_id": data.get("approval_id"),
                        "reason": "Approval was rejected",
                    }
        except Exception:
            continue
    
    return {
        "approved": False,
        "approval_id": None,
        "reason": "No matching approval found for this commit plan",
    }


def list_pending_approvals() -> List[GitApproval]:
    """List all pending approval requests.
    
    Returns:
        List of pending GitApproval objects.
    """
    approval_dir = get_approval_dir()
    pending = []
    
    if not approval_dir.exists():
        return pending
    
    for approval_file in approval_dir.glob("GA-*.yaml"):
        try:
            with open(approval_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if data.get("status") == ApprovalStatus.PENDING.value:
                pending.append(_load_approval(data))
        except Exception:
            continue
    
    return sorted(pending, key=lambda x: x.created_at, reverse=True)


def _load_approval(data: Dict[str, Any]) -> GitApproval:
    """Load a GitApproval from dict data."""
    return GitApproval(
        approval_id=data["approval_id"],
        commit_plan_hash=data["commit_plan_hash"],
        files=data.get("files", []),
        confirm_level=data.get("confirm_level", "CONFIRM"),
        status=ApprovalStatus(data.get("status", "pending")),
        created_at=datetime.fromisoformat(data["created_at"]),
        approved_at=datetime.fromisoformat(data["approved_at"]) if data.get("approved_at") else None,
        approved_by=data.get("approved_by"),
        commit_message=data.get("commit_message"),
        reason=data.get("reason"),
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for git approval management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Git Approval Artifact Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # request command
    req_parser = subparsers.add_parser("request", help="Create approval request")
    req_parser.add_argument("--files", nargs="+", required=True, help="Files to commit")
    req_parser.add_argument("--message", required=True, help="Commit message")
    req_parser.add_argument("--level", default="CONFIRM", choices=["CONFIRM", "BLOCK"])
    req_parser.add_argument("--reason", help="Why approval is needed")
    
    # approve command
    approve_parser = subparsers.add_parser("approve", help="Approve a request")
    approve_parser.add_argument("approval_id", help="Approval ID to approve")
    approve_parser.add_argument("--by", default="user", help="Approver name")
    
    # check command
    check_parser = subparsers.add_parser("check", help="Check if approval exists")
    check_parser.add_argument("--files", nargs="+", required=True, help="Files to check")
    check_parser.add_argument("--message", required=True, help="Commit message")
    
    # list command
    subparsers.add_parser("list", help="List pending approvals")
    
    args = parser.parse_args()
    
    if args.command == "request":
        approval = create_approval_request(
            files=args.files,
            commit_message=args.message,
            confirm_level=args.level,
            reason=args.reason,
        )
        print(f"✅ Created approval request: {approval.approval_id}")
        print(f"   Files: {len(approval.files)}")
        print(f"   Level: {approval.confirm_level}")
        print(f"   Status: {approval.status.value}")
        print(f"\n   To approve: python kernel/git_approval.py approve {approval.approval_id}")
    
    elif args.command == "approve":
        approval = approve_request(args.approval_id, args.by)
        if approval:
            print(f"✅ Approved: {approval.approval_id}")
            print(f"   By: {approval.approved_by}")
            print(f"   At: {approval.approved_at}")
        else:
            print(f"❌ Approval not found: {args.approval_id}")
    
    elif args.command == "check":
        result = check_approval_exists(args.files, args.message)
        if result["approved"]:
            print(f"✅ Approved: {result['approval_id']}")
            print(f"   By: {result.get('approved_by', 'unknown')}")
        else:
            print(f"❌ Not approved")
            print(f"   Reason: {result['reason']}")
            if result.get("approval_id"):
                print(f"   Pending ID: {result['approval_id']}")
    
    elif args.command == "list":
        pending = list_pending_approvals()
        if not pending:
            print("📭 No pending approvals")
        else:
            print(f"📋 Pending Approvals ({len(pending)}):\n")
            for a in pending:
                print(f"  [{a.approval_id}]")
                print(f"    Level: {a.confirm_level}")
                print(f"    Files: {len(a.files)}")
                print(f"    Message: {a.commit_message[:50]}...")
                print(f"    Created: {a.created_at.isoformat()}")
                print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
