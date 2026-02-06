#!/usr/bin/env python3
"""
GitHub Integration — Issue/PR Binding for Task Workflow

Purpose: Bind execution queue tasks to GitHub Issues and PRs,
         enabling native GitHub-based workflow tracking.

Enforcement for: PP-019 (Issue/PR 绑定无实现)

Usage:
    python kernel/github_integration.py create-issue --task-id T3.1 --title "Implement feature X"
    python kernel/github_integration.py bind-issue --task-id T3.1 --issue 42
    python kernel/github_integration.py create-pr --task-id T3.1 --branch feat/T3.1
    python kernel/github_integration.py check-pr --pr 15
    python kernel/github_integration.py sync --task-id T3.1
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
# Constants
# -----------------------------------------------------------------------------
GITHUB_STATE_PATH = ROOT / "state" / "github_bindings.yaml"
EXECUTION_QUEUE_PATH = ROOT / "state" / "execution_queue.yaml"


class IssueState(Enum):
    """GitHub Issue states."""
    OPEN = "open"
    CLOSED = "closed"


class PRState(Enum):
    """GitHub PR states."""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    DRAFT = "draft"


class TaskGitHubStatus(Enum):
    """Task GitHub integration status."""
    UNBOUND = "unbound"           # No Issue/PR created
    ISSUE_CREATED = "issue_created"
    PR_CREATED = "pr_created"
    PR_READY = "pr_ready"         # PR ready for review
    PR_APPROVED = "pr_approved"   # PR approved
    MERGED = "merged"             # PR merged
    CLOSED = "closed"             # Issue/PR closed without merge


@dataclass
class GitHubBinding:
    """Binding between a task and GitHub Issue/PR."""
    task_id: str
    issue_number: Optional[int] = None
    issue_url: Optional[str] = None
    issue_state: Optional[str] = None
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None
    pr_state: Optional[str] = None
    pr_branch: Optional[str] = None
    status: TaskGitHubStatus = TaskGitHubStatus.UNBOUND
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    labels: list[str] = field(default_factory=list)
    review_artifacts: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "issue_number": self.issue_number,
            "issue_url": self.issue_url,
            "issue_state": self.issue_state,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "pr_state": self.pr_state,
            "pr_branch": self.pr_branch,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "labels": self.labels,
            "review_artifacts": self.review_artifacts,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GitHubBinding":
        return cls(
            task_id=data["task_id"],
            issue_number=data.get("issue_number"),
            issue_url=data.get("issue_url"),
            issue_state=data.get("issue_state"),
            pr_number=data.get("pr_number"),
            pr_url=data.get("pr_url"),
            pr_state=data.get("pr_state"),
            pr_branch=data.get("pr_branch"),
            status=TaskGitHubStatus(data.get("status", "unbound")),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            labels=data.get("labels", []),
            review_artifacts=data.get("review_artifacts", []),
        )


@dataclass
class GitHubState:
    """Global GitHub bindings state."""
    bindings: dict[str, GitHubBinding] = field(default_factory=dict)
    repo_owner: str = ""
    repo_name: str = ""
    last_synced: str = ""
    
    def to_dict(self) -> dict:
        return {
            "bindings": {k: v.to_dict() for k, v in self.bindings.items()},
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "last_synced": self.last_synced,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GitHubState":
        bindings = {}
        for k, v in data.get("bindings", {}).items():
            bindings[k] = GitHubBinding.from_dict(v)
        return cls(
            bindings=bindings,
            repo_owner=data.get("repo_owner", ""),
            repo_name=data.get("repo_name", ""),
            last_synced=data.get("last_synced", ""),
        )


# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------

def load_github_state() -> GitHubState:
    """Load GitHub state from YAML file."""
    if GITHUB_STATE_PATH.exists():
        with open(GITHUB_STATE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return GitHubState.from_dict(data)
    return GitHubState()


def save_github_state(state: GitHubState) -> None:
    """Save GitHub state to YAML file."""
    state.last_synced = datetime.now().isoformat()
    GITHUB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GITHUB_STATE_PATH, "w", encoding="utf-8") as f:
        yaml.dump(state.to_dict(), f, allow_unicode=True, default_flow_style=False)


def get_binding(task_id: str) -> Optional[GitHubBinding]:
    """Get binding for a task."""
    state = load_github_state()
    return state.bindings.get(task_id)


def update_binding(binding: GitHubBinding) -> None:
    """Update or create a binding."""
    state = load_github_state()
    binding.updated_at = datetime.now().isoformat()
    state.bindings[binding.task_id] = binding
    save_github_state(state)


# -----------------------------------------------------------------------------
# GitHub CLI Helpers
# -----------------------------------------------------------------------------

def run_gh_command(args: list[str], check: bool = True) -> tuple[bool, str]:
    """Run a GitHub CLI command."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr.strip()
    except FileNotFoundError:
        return False, "GitHub CLI (gh) not installed. Install from https://cli.github.com/"
    except Exception as e:
        return False, str(e)


def get_repo_info() -> tuple[str, str]:
    """Get current repo owner and name."""
    success, output = run_gh_command(["repo", "view", "--json", "owner,name"])
    if success:
        try:
            data = json.loads(output)
            return data.get("owner", {}).get("login", ""), data.get("name", "")
        except json.JSONDecodeError:
            pass
    
    # Fallback: parse from git remote
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Parse github.com/owner/repo or git@github.com:owner/repo
            match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
            if match:
                return match.group(1), match.group(2)
    except Exception:
        pass
    
    return "", ""


def check_gh_auth() -> tuple[bool, str]:
    """Check if gh CLI is authenticated."""
    success, output = run_gh_command(["auth", "status"])
    if success or "Logged in" in output:
        return True, "GitHub CLI authenticated"
    return False, f"GitHub CLI not authenticated: {output}"


# -----------------------------------------------------------------------------
# Issue Operations
# -----------------------------------------------------------------------------

def create_issue(
    task_id: str,
    title: str,
    body: str = "",
    labels: Optional[list[str]] = None,
) -> tuple[bool, str, Optional[int]]:
    """Create a GitHub Issue for a task."""
    # Check existing binding
    binding = get_binding(task_id)
    if binding and binding.issue_number:
        return False, f"Task {task_id} already has Issue #{binding.issue_number}", binding.issue_number
    
    # Prepare issue body
    issue_body = f"""## Task ID: `{task_id}`

{body}

---
*Created by AI Workflow OS*
"""
    
    # Build command
    cmd = ["issue", "create", "--title", f"[{task_id}] {title}", "--body", issue_body]
    
    if labels:
        for label in labels:
            cmd.extend(["--label", label])
    
    success, output = run_gh_command(cmd)
    
    if success:
        # Parse issue URL to get number
        issue_url = output.strip()
        match = re.search(r"/issues/(\d+)", issue_url)
        issue_number = int(match.group(1)) if match else None
        
        # Update binding
        if not binding:
            binding = GitHubBinding(task_id=task_id)
        
        binding.issue_number = issue_number
        binding.issue_url = issue_url
        binding.issue_state = "open"
        binding.status = TaskGitHubStatus.ISSUE_CREATED
        binding.labels = labels or []
        update_binding(binding)
        
        return True, f"Created Issue #{issue_number}: {issue_url}", issue_number
    
    return False, f"Failed to create issue: {output}", None


def bind_issue(task_id: str, issue_number: int) -> tuple[bool, str]:
    """Bind an existing Issue to a task."""
    # Verify issue exists
    success, output = run_gh_command(["issue", "view", str(issue_number), "--json", "url,state,title"])
    
    if not success:
        return False, f"Issue #{issue_number} not found: {output}"
    
    try:
        data = json.loads(output)
        issue_url = data.get("url", "")
        issue_state = data.get("state", "").lower()
    except json.JSONDecodeError:
        return False, "Failed to parse issue data"
    
    # Update binding
    binding = get_binding(task_id) or GitHubBinding(task_id=task_id)
    binding.issue_number = issue_number
    binding.issue_url = issue_url
    binding.issue_state = issue_state
    binding.status = TaskGitHubStatus.ISSUE_CREATED
    update_binding(binding)
    
    return True, f"Bound Issue #{issue_number} to task {task_id}"


def close_issue(task_id: str, reason: str = "completed") -> tuple[bool, str]:
    """Close the Issue for a task."""
    binding = get_binding(task_id)
    if not binding or not binding.issue_number:
        return False, f"Task {task_id} has no bound Issue"
    
    success, output = run_gh_command([
        "issue", "close", str(binding.issue_number),
        "--reason", reason,
    ])
    
    if success:
        binding.issue_state = "closed"
        update_binding(binding)
        return True, f"Closed Issue #{binding.issue_number}"
    
    return False, f"Failed to close issue: {output}"


# -----------------------------------------------------------------------------
# PR Operations
# -----------------------------------------------------------------------------

def create_pr(
    task_id: str,
    branch: str,
    title: Optional[str] = None,
    body: str = "",
    draft: bool = False,
    base: str = "main",
) -> tuple[bool, str, Optional[int]]:
    """Create a PR for a task."""
    binding = get_binding(task_id)
    if binding and binding.pr_number:
        return False, f"Task {task_id} already has PR #{binding.pr_number}", binding.pr_number
    
    # Default title from task ID
    pr_title = title or f"[{task_id}] Implementation"
    
    # Prepare PR body with checklist
    pr_body = f"""## Task ID: `{task_id}`

{body}

---

## Gate Checklist

- [ ] **G1**: Data quality checks pass
- [ ] **G2**: Unit tests pass
- [ ] **G3**: Performance benchmarks pass (if applicable)
- [ ] **Review**: Pair review completed (REVIEW artifact exists)

### Linked Issue
{f"Closes #{binding.issue_number}" if binding and binding.issue_number else "N/A"}

---
*Created by AI Workflow OS*
"""
    
    # Build command
    cmd = [
        "pr", "create",
        "--title", pr_title,
        "--body", pr_body,
        "--head", branch,
        "--base", base,
    ]
    
    if draft:
        cmd.append("--draft")
    
    success, output = run_gh_command(cmd)
    
    if success:
        # Parse PR URL to get number
        pr_url = output.strip()
        match = re.search(r"/pull/(\d+)", pr_url)
        pr_number = int(match.group(1)) if match else None
        
        # Update binding
        if not binding:
            binding = GitHubBinding(task_id=task_id)
        
        binding.pr_number = pr_number
        binding.pr_url = pr_url
        binding.pr_branch = branch
        binding.pr_state = "draft" if draft else "open"
        binding.status = TaskGitHubStatus.PR_CREATED
        update_binding(binding)
        
        return True, f"Created PR #{pr_number}: {pr_url}", pr_number
    
    return False, f"Failed to create PR: {output}", None


def bind_pr(task_id: str, pr_number: int) -> tuple[bool, str]:
    """Bind an existing PR to a task."""
    success, output = run_gh_command([
        "pr", "view", str(pr_number),
        "--json", "url,state,headRefName,isDraft",
    ])
    
    if not success:
        return False, f"PR #{pr_number} not found: {output}"
    
    try:
        data = json.loads(output)
        pr_url = data.get("url", "")
        pr_state = "draft" if data.get("isDraft") else data.get("state", "").lower()
        pr_branch = data.get("headRefName", "")
    except json.JSONDecodeError:
        return False, "Failed to parse PR data"
    
    # Update binding
    binding = get_binding(task_id) or GitHubBinding(task_id=task_id)
    binding.pr_number = pr_number
    binding.pr_url = pr_url
    binding.pr_branch = pr_branch
    binding.pr_state = pr_state
    binding.status = TaskGitHubStatus.PR_CREATED
    update_binding(binding)
    
    return True, f"Bound PR #{pr_number} to task {task_id}"


def check_pr_status(pr_number: int) -> tuple[bool, dict]:
    """Check PR status including reviews and checks."""
    success, output = run_gh_command([
        "pr", "view", str(pr_number),
        "--json", "state,isDraft,reviewDecision,statusCheckRollup,mergeable",
    ])
    
    if not success:
        return False, {"error": output}
    
    try:
        data = json.loads(output)
        
        # Parse status checks
        checks = data.get("statusCheckRollup", []) or []
        checks_pass = all(
            c.get("conclusion") == "SUCCESS" or c.get("state") == "SUCCESS"
            for c in checks
        )
        
        return True, {
            "state": data.get("state", "").lower(),
            "is_draft": data.get("isDraft", False),
            "review_decision": data.get("reviewDecision", ""),
            "mergeable": data.get("mergeable", ""),
            "checks_pass": checks_pass,
            "checks_count": len(checks),
        }
    except json.JSONDecodeError:
        return False, {"error": "Failed to parse PR data"}


def sync_task_status(task_id: str) -> tuple[bool, str]:
    """Sync task GitHub status from remote."""
    binding = get_binding(task_id)
    if not binding:
        return False, f"No binding found for task {task_id}"
    
    updates = []
    
    # Sync Issue
    if binding.issue_number:
        success, output = run_gh_command([
            "issue", "view", str(binding.issue_number),
            "--json", "state",
        ])
        if success:
            try:
                data = json.loads(output)
                binding.issue_state = data.get("state", "").lower()
                updates.append(f"Issue #{binding.issue_number}: {binding.issue_state}")
            except json.JSONDecodeError:
                pass
    
    # Sync PR
    if binding.pr_number:
        success, pr_status = check_pr_status(binding.pr_number)
        if success:
            binding.pr_state = pr_status.get("state", "")
            if pr_status.get("is_draft"):
                binding.pr_state = "draft"
            
            # Update status based on PR state
            if binding.pr_state == "merged":
                binding.status = TaskGitHubStatus.MERGED
            elif pr_status.get("review_decision") == "APPROVED":
                binding.status = TaskGitHubStatus.PR_APPROVED
            elif not pr_status.get("is_draft"):
                binding.status = TaskGitHubStatus.PR_READY
            
            updates.append(f"PR #{binding.pr_number}: {binding.pr_state}")
    
    update_binding(binding)
    
    return True, f"Synced: {', '.join(updates)}" if updates else "No updates"


# -----------------------------------------------------------------------------
# Batch Operations
# -----------------------------------------------------------------------------

def list_bindings(status_filter: Optional[str] = None) -> list[GitHubBinding]:
    """List all bindings, optionally filtered by status."""
    state = load_github_state()
    bindings = list(state.bindings.values())
    
    if status_filter:
        bindings = [b for b in bindings if b.status.value == status_filter]
    
    return bindings


def get_unbound_tasks() -> list[str]:
    """Get task IDs from execution queue that have no GitHub binding."""
    if not EXECUTION_QUEUE_PATH.exists():
        return []
    
    with open(EXECUTION_QUEUE_PATH, "r", encoding="utf-8") as f:
        queue = yaml.safe_load(f) or {}
    
    state = load_github_state()
    
    unbound = []
    for task in queue.get("queue", []):
        task_id = task.get("id", "")
        if task_id and task_id not in state.bindings:
            unbound.append(task_id)
    
    return unbound


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="GitHub Integration for Task Workflow")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # status
    subparsers.add_parser("status", help="Show GitHub integration status")
    
    # create-issue
    ci_parser = subparsers.add_parser("create-issue", help="Create Issue for task")
    ci_parser.add_argument("--task-id", required=True, help="Task ID")
    ci_parser.add_argument("--title", required=True, help="Issue title")
    ci_parser.add_argument("--body", default="", help="Issue body")
    ci_parser.add_argument("--labels", nargs="*", help="Labels to add")
    
    # bind-issue
    bi_parser = subparsers.add_parser("bind-issue", help="Bind existing Issue to task")
    bi_parser.add_argument("--task-id", required=True, help="Task ID")
    bi_parser.add_argument("--issue", type=int, required=True, help="Issue number")
    
    # create-pr
    cp_parser = subparsers.add_parser("create-pr", help="Create PR for task")
    cp_parser.add_argument("--task-id", required=True, help="Task ID")
    cp_parser.add_argument("--branch", required=True, help="Source branch")
    cp_parser.add_argument("--title", help="PR title")
    cp_parser.add_argument("--body", default="", help="PR body")
    cp_parser.add_argument("--draft", action="store_true", help="Create as draft")
    cp_parser.add_argument("--base", default="main", help="Target branch")
    
    # bind-pr
    bp_parser = subparsers.add_parser("bind-pr", help="Bind existing PR to task")
    bp_parser.add_argument("--task-id", required=True, help="Task ID")
    bp_parser.add_argument("--pr", type=int, required=True, help="PR number")
    
    # check-pr
    chk_parser = subparsers.add_parser("check-pr", help="Check PR status")
    chk_parser.add_argument("--pr", type=int, required=True, help="PR number")
    
    # sync
    sync_parser = subparsers.add_parser("sync", help="Sync task status from GitHub")
    sync_parser.add_argument("--task-id", required=True, help="Task ID")
    
    # list
    list_parser = subparsers.add_parser("list", help="List all bindings")
    list_parser.add_argument("--status", help="Filter by status")
    
    # unbound
    subparsers.add_parser("unbound", help="Show tasks without GitHub binding")
    
    args = parser.parse_args()
    
    if args.command == "status":
        # Check auth
        auth_ok, auth_msg = check_gh_auth()
        print(f"🔐 Auth: {'✅' if auth_ok else '❌'} {auth_msg}")
        
        # Get repo info
        owner, name = get_repo_info()
        if owner and name:
            print(f"📦 Repo: {owner}/{name}")
        else:
            print("📦 Repo: Not detected")
        
        # Count bindings
        state = load_github_state()
        print(f"\n📊 Bindings: {len(state.bindings)}")
        
        by_status = {}
        for b in state.bindings.values():
            by_status[b.status.value] = by_status.get(b.status.value, 0) + 1
        
        for status, count in sorted(by_status.items()):
            print(f"   • {status}: {count}")
        
        sys.exit(0)
    
    elif args.command == "create-issue":
        success, msg, issue_num = create_issue(
            args.task_id, args.title, args.body, args.labels
        )
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "bind-issue":
        success, msg = bind_issue(args.task_id, args.issue)
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "create-pr":
        success, msg, pr_num = create_pr(
            args.task_id, args.branch, args.title, args.body, args.draft, args.base
        )
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "bind-pr":
        success, msg = bind_pr(args.task_id, args.pr)
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "check-pr":
        success, status = check_pr_status(args.pr)
        if success:
            print(f"📋 PR #{args.pr} Status:")
            print(f"   State: {status['state']}")
            print(f"   Draft: {'Yes' if status['is_draft'] else 'No'}")
            print(f"   Review: {status['review_decision'] or 'Pending'}")
            print(f"   Checks: {'✅ Pass' if status['checks_pass'] else '⏳ Pending'} ({status['checks_count']} checks)")
            print(f"   Mergeable: {status['mergeable']}")
        else:
            print(f"❌ {status.get('error', 'Unknown error')}")
        sys.exit(0 if success else 1)
    
    elif args.command == "sync":
        success, msg = sync_task_status(args.task_id)
        print(msg)
        sys.exit(0 if success else 1)
    
    elif args.command == "list":
        bindings = list_bindings(args.status)
        if not bindings:
            print("No bindings found.")
        else:
            print(f"{'Task ID':<15} {'Issue':<8} {'PR':<8} {'Status':<20}")
            print("-" * 55)
            for b in bindings:
                issue = f"#{b.issue_number}" if b.issue_number else "—"
                pr = f"#{b.pr_number}" if b.pr_number else "—"
                print(f"{b.task_id:<15} {issue:<8} {pr:<8} {b.status.value:<20}")
        sys.exit(0)
    
    elif args.command == "unbound":
        unbound = get_unbound_tasks()
        if not unbound:
            print("✅ All tasks have GitHub bindings.")
        else:
            print(f"📋 {len(unbound)} task(s) without GitHub binding:")
            for task_id in unbound:
                print(f"   • {task_id}")
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
