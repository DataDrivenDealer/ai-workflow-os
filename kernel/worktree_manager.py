#!/usr/bin/env python3
"""
Worktree Manager — Git Worktree-based Parallel Execution

Purpose: Enable parallel task/subagent execution via git worktrees,
         providing isolation without affecting the main working directory.

Enforcement for: PP-018 (Worktree 隔离无实现)

Usage:
    python kernel/worktree_manager.py create --name subagent-research --branch feat/research
    python kernel/worktree_manager.py list
    python kernel/worktree_manager.py remove --name subagent-research
    python kernel/worktree_manager.py cleanup
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

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
# Constants
# -----------------------------------------------------------------------------
WORKTREE_MAP_PATH = ROOT / "docs" / "state" / "WORKTREE_MAP.md"
WORKTREE_STATE_PATH = ROOT / "state" / "worktrees.yaml"
DEFAULT_WORKTREE_BASE = ROOT / ".worktrees"
MAX_WORKTREES = 5  # Safety limit


class WorktreeStatus(Enum):
    """Worktree status enumeration."""
    ACTIVE = "active"
    IDLE = "idle"
    STALE = "stale"
    REMOVED = "removed"


class WorktreePurpose(Enum):
    """Worktree purpose categories."""
    SUBAGENT = "subagent"       # For subagent parallel execution
    EXPERIMENT = "experiment"    # For isolated experiment runs
    HOTFIX = "hotfix"           # For urgent fixes
    REVIEW = "review"           # For code review isolation
    BACKUP = "backup"           # For backup before destructive ops


@dataclass
class Worktree:
    """Represents a git worktree instance."""
    name: str
    path: str
    branch: str
    purpose: WorktreePurpose
    status: WorktreeStatus = WorktreeStatus.IDLE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    task_id: Optional[str] = None
    owner: str = "agent"  # agent | subagent:{id} | human
    
    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "branch": self.branch,
            "purpose": self.purpose.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "task_id": self.task_id,
            "owner": self.owner,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Worktree":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            branch=data["branch"],
            purpose=WorktreePurpose(data["purpose"]),
            status=WorktreeStatus(data.get("status", "idle")),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_used=data.get("last_used"),
            task_id=data.get("task_id"),
            owner=data.get("owner", "agent"),
        )


@dataclass
class WorktreeState:
    """Global worktree state."""
    worktrees: dict[str, Worktree] = field(default_factory=dict)
    base_path: str = str(DEFAULT_WORKTREE_BASE)
    max_worktrees: int = MAX_WORKTREES
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "worktrees": {name: wt.to_dict() for name, wt in self.worktrees.items()},
            "base_path": self.base_path,
            "max_worktrees": self.max_worktrees,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WorktreeState":
        """Create from dictionary."""
        worktrees = {}
        for name, wt_data in data.get("worktrees", {}).items():
            worktrees[name] = Worktree.from_dict(wt_data)
        return cls(
            worktrees=worktrees,
            base_path=data.get("base_path", str(DEFAULT_WORKTREE_BASE)),
            max_worktrees=data.get("max_worktrees", MAX_WORKTREES),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
        )


# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------

def load_worktree_state() -> WorktreeState:
    """Load worktree state from YAML file."""
    if WORKTREE_STATE_PATH.exists():
        with open(WORKTREE_STATE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return WorktreeState.from_dict(data)
    return WorktreeState()


def save_worktree_state(state: WorktreeState) -> None:
    """Save worktree state to YAML file."""
    state.last_updated = datetime.now().isoformat()
    WORKTREE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WORKTREE_STATE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(state.to_dict(), f, allow_unicode=True, default_flow_style=False)


def update_worktree_map(state: WorktreeState) -> None:
    """Update the human-readable WORKTREE_MAP.md file."""
    WORKTREE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# WORKTREE MAP",
        "",
        f"> **Last Updated**: {state.last_updated}",
        f"> **Base Path**: `{state.base_path}`",
        f"> **Max Worktrees**: {state.max_worktrees}",
        "",
        "---",
        "",
        "## Active Worktrees",
        "",
    ]
    
    active = [wt for wt in state.worktrees.values() if wt.status != WorktreeStatus.REMOVED]
    
    if not active:
        lines.append("*No active worktrees.*")
    else:
        lines.append("| Name | Branch | Purpose | Status | Owner | Task ID |")
        lines.append("|------|--------|---------|--------|-------|---------|")
        for wt in active:
            task = wt.task_id or "—"
            lines.append(f"| `{wt.name}` | `{wt.branch}` | {wt.purpose.value} | {wt.status.value} | {wt.owner} | {task} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Usage Commands",
        "",
        "```bash",
        "# Create a new worktree for subagent",
        "python kernel/worktree_manager.py create --name subagent-001 --branch feat/task-001 --purpose subagent",
        "",
        "# List all worktrees",
        "python kernel/worktree_manager.py list",
        "",
        "# Mark worktree as active with task",
        "python kernel/worktree_manager.py activate --name subagent-001 --task-id T3.1",
        "",
        "# Mark worktree as idle",
        "python kernel/worktree_manager.py idle --name subagent-001",
        "",
        "# Remove a worktree",
        "python kernel/worktree_manager.py remove --name subagent-001",
        "",
        "# Cleanup stale worktrees",
        "python kernel/worktree_manager.py cleanup",
        "```",
        "",
        "---",
        "",
        "*Auto-generated by `kernel/worktree_manager.py`*",
    ])
    
    with open(WORKTREE_MAP_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# Git Worktree Operations
# -----------------------------------------------------------------------------

def run_git_command(args: list[str], cwd: Optional[Path] = None) -> tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd or ROOT,
            check=False,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


def create_worktree(
    name: str,
    branch: str,
    purpose: WorktreePurpose,
    task_id: Optional[str] = None,
    owner: str = "agent",
    create_branch: bool = True,
) -> tuple[bool, str, Optional[Worktree]]:
    """
    Create a new git worktree.
    
    Returns:
        (success, message, worktree_or_none)
    """
    state = load_worktree_state()
    
    # Check limit
    active_count = sum(1 for wt in state.worktrees.values() if wt.status != WorktreeStatus.REMOVED)
    if active_count >= state.max_worktrees:
        return False, f"❌ Max worktrees limit reached ({state.max_worktrees}). Remove existing worktrees first.", None
    
    # Check duplicate
    if name in state.worktrees and state.worktrees[name].status != WorktreeStatus.REMOVED:
        return False, f"❌ Worktree '{name}' already exists.", None
    
    # Prepare path
    base_path = Path(state.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    worktree_path = base_path / name
    
    # Create worktree
    if create_branch:
        # Create new branch from current HEAD
        success, output = run_git_command(["worktree", "add", "-b", branch, str(worktree_path)])
    else:
        # Use existing branch
        success, output = run_git_command(["worktree", "add", str(worktree_path), branch])
    
    if not success:
        return False, f"❌ Failed to create worktree: {output}", None
    
    # Create worktree record
    worktree = Worktree(
        name=name,
        path=str(worktree_path),
        branch=branch,
        purpose=purpose,
        status=WorktreeStatus.IDLE,
        task_id=task_id,
        owner=owner,
    )
    
    state.worktrees[name] = worktree
    save_worktree_state(state)
    update_worktree_map(state)
    
    return True, f"✅ Created worktree '{name}' at {worktree_path}", worktree


def remove_worktree(name: str, force: bool = False) -> tuple[bool, str]:
    """Remove a git worktree."""
    state = load_worktree_state()
    
    if name not in state.worktrees:
        return False, f"❌ Worktree '{name}' not found."
    
    worktree = state.worktrees[name]
    
    if worktree.status == WorktreeStatus.ACTIVE and not force:
        return False, f"❌ Worktree '{name}' is active. Use --force to remove."
    
    # Remove git worktree
    args = ["worktree", "remove", worktree.path]
    if force:
        args.insert(2, "--force")
    
    success, output = run_git_command(args)
    
    if not success:
        # Try force remove if path doesn't exist
        if "is not a working tree" in output or "does not exist" in output.lower():
            pass  # Already removed, just update state
        else:
            return False, f"❌ Failed to remove worktree: {output}"
    
    # Update state
    worktree.status = WorktreeStatus.REMOVED
    save_worktree_state(state)
    update_worktree_map(state)
    
    return True, f"✅ Removed worktree '{name}'."


def activate_worktree(name: str, task_id: Optional[str] = None) -> tuple[bool, str]:
    """Mark a worktree as active."""
    state = load_worktree_state()
    
    if name not in state.worktrees:
        return False, f"❌ Worktree '{name}' not found."
    
    worktree = state.worktrees[name]
    if worktree.status == WorktreeStatus.REMOVED:
        return False, f"❌ Worktree '{name}' has been removed."
    
    worktree.status = WorktreeStatus.ACTIVE
    worktree.last_used = datetime.now().isoformat()
    if task_id:
        worktree.task_id = task_id
    
    save_worktree_state(state)
    update_worktree_map(state)
    
    return True, f"✅ Worktree '{name}' marked as active."


def idle_worktree(name: str) -> tuple[bool, str]:
    """Mark a worktree as idle."""
    state = load_worktree_state()
    
    if name not in state.worktrees:
        return False, f"❌ Worktree '{name}' not found."
    
    worktree = state.worktrees[name]
    if worktree.status == WorktreeStatus.REMOVED:
        return False, f"❌ Worktree '{name}' has been removed."
    
    worktree.status = WorktreeStatus.IDLE
    worktree.last_used = datetime.now().isoformat()
    
    save_worktree_state(state)
    update_worktree_map(state)
    
    return True, f"✅ Worktree '{name}' marked as idle."


def list_worktrees() -> list[Worktree]:
    """List all non-removed worktrees."""
    state = load_worktree_state()
    return [wt for wt in state.worktrees.values() if wt.status != WorktreeStatus.REMOVED]


def cleanup_stale_worktrees(max_age_hours: int = 24) -> tuple[int, list[str]]:
    """Remove worktrees that have been idle for too long."""
    from datetime import timedelta
    
    state = load_worktree_state()
    removed = []
    now = datetime.now()
    
    for name, wt in list(state.worktrees.items()):
        if wt.status in (WorktreeStatus.IDLE, WorktreeStatus.STALE):
            if wt.last_used:
                last_used = datetime.fromisoformat(wt.last_used)
            else:
                last_used = datetime.fromisoformat(wt.created_at)
            
            age = now - last_used
            if age > timedelta(hours=max_age_hours):
                success, msg = remove_worktree(name, force=True)
                if success:
                    removed.append(name)
    
    return len(removed), removed


def get_worktree_for_subagent(subagent_id: str) -> Optional[Worktree]:
    """Get or create a worktree for a specific subagent."""
    state = load_worktree_state()
    
    # Check for existing idle worktree for this subagent
    for wt in state.worktrees.values():
        if wt.owner == f"subagent:{subagent_id}" and wt.status == WorktreeStatus.IDLE:
            return wt
    
    # Check for any idle subagent worktree
    for wt in state.worktrees.values():
        if wt.purpose == WorktreePurpose.SUBAGENT and wt.status == WorktreeStatus.IDLE:
            wt.owner = f"subagent:{subagent_id}"
            save_worktree_state(state)
            return wt
    
    # Create new worktree if under limit
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"subagent-{subagent_id}-{timestamp}"
    branch = f"worktree/{name}"
    
    success, msg, wt = create_worktree(
        name=name,
        branch=branch,
        purpose=WorktreePurpose.SUBAGENT,
        owner=f"subagent:{subagent_id}",
    )
    
    return wt if success else None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Git Worktree Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # create
    create_parser = subparsers.add_parser("create", help="Create a new worktree")
    create_parser.add_argument("--name", required=True, help="Worktree name")
    create_parser.add_argument("--branch", required=True, help="Branch name")
    create_parser.add_argument("--purpose", default="subagent", 
                               choices=["subagent", "experiment", "hotfix", "review", "backup"],
                               help="Worktree purpose")
    create_parser.add_argument("--task-id", help="Associated task ID")
    create_parser.add_argument("--owner", default="agent", help="Owner (agent, subagent:id, human)")
    create_parser.add_argument("--existing-branch", action="store_true", 
                               help="Use existing branch instead of creating new")
    
    # list
    subparsers.add_parser("list", help="List all worktrees")
    
    # activate
    activate_parser = subparsers.add_parser("activate", help="Mark worktree as active")
    activate_parser.add_argument("--name", required=True, help="Worktree name")
    activate_parser.add_argument("--task-id", help="Task ID being worked on")
    
    # idle
    idle_parser = subparsers.add_parser("idle", help="Mark worktree as idle")
    idle_parser.add_argument("--name", required=True, help="Worktree name")
    
    # remove
    remove_parser = subparsers.add_parser("remove", help="Remove a worktree")
    remove_parser.add_argument("--name", required=True, help="Worktree name")
    remove_parser.add_argument("--force", action="store_true", help="Force removal")
    
    # cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove stale worktrees")
    cleanup_parser.add_argument("--max-age", type=int, default=24, 
                                help="Max idle age in hours (default: 24)")
    
    # get-subagent
    get_parser = subparsers.add_parser("get-subagent", help="Get worktree for subagent")
    get_parser.add_argument("--id", required=True, help="Subagent ID")
    
    args = parser.parse_args()
    
    if args.command == "create":
        success, msg, wt = create_worktree(
            name=args.name,
            branch=args.branch,
            purpose=WorktreePurpose(args.purpose),
            task_id=args.task_id,
            owner=args.owner,
            create_branch=not args.existing_branch,
        )
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "list":
        worktrees = list_worktrees()
        if not worktrees:
            print("No active worktrees.")
        else:
            print(f"{'Name':<25} {'Branch':<30} {'Purpose':<12} {'Status':<10} {'Owner':<20}")
            print("-" * 100)
            for wt in worktrees:
                print(f"{wt.name:<25} {wt.branch:<30} {wt.purpose.value:<12} {wt.status.value:<10} {wt.owner:<20}")
        sys.exit(0)
    
    elif args.command == "activate":
        success, msg = activate_worktree(args.name, args.task_id)
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "idle":
        success, msg = idle_worktree(args.name)
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "remove":
        success, msg = remove_worktree(args.name, args.force)
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "cleanup":
        count, removed = cleanup_stale_worktrees(args.max_age)
        if count == 0:
            print("No stale worktrees to clean up.")
        else:
            print(f"✅ Removed {count} stale worktree(s): {', '.join(removed)}")
        sys.exit(0)
    
    elif args.command == "get-subagent":
        wt = get_worktree_for_subagent(args.id)
        if wt:
            print(f"✅ Worktree for subagent '{args.id}': {wt.path}")
            print(f"   Branch: {wt.branch}")
            print(f"   Status: {wt.status.value}")
        else:
            print(f"❌ Could not get or create worktree for subagent '{args.id}'")
            sys.exit(1)
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
