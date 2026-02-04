"""
Git Operations Module for Copilot Runtime OS.

Provides structured Git workflow integration as an internal ops subprocess.
Supports status checking, commit planning, and execution with confirmation levels.

Version: 1.0.0
"""

from __future__ import annotations

import subprocess
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kernel.paths import ROOT


class ConfirmLevel(Enum):
    """Confirmation level for Git operations.
    
    AUTO: Execute automatically, log only
    NOTIFY: Execute and notify user
    CONFIRM: Output plan, wait for [Y/n]
    BLOCK: Output commands, human executes
    """
    AUTO = 0
    NOTIFY = 1
    CONFIRM = 2
    BLOCK = 3


@dataclass
class GitStatus:
    """Represents the current state of a Git repository."""
    is_clean: bool
    branch: str
    staged: List[str] = field(default_factory=list)
    unstaged: List[str] = field(default_factory=list)
    untracked: List[str] = field(default_factory=list)
    latest_tag: Optional[str] = None
    ahead: int = 0
    behind: int = 0
    repo_path: Path = field(default_factory=lambda: ROOT)


@dataclass
class CommitPlan:
    """A plan for Git commit, tag, and push operations."""
    message: str
    files_to_stage: List[str] = field(default_factory=list)
    tag: Optional[str] = None
    tag_message: Optional[str] = None
    push_remote: bool = False
    push_tags: bool = False
    confirm_level: ConfirmLevel = ConfirmLevel.CONFIRM
    context: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CHANGE CLASSIFICATION
# =============================================================================

CHANGE_CATEGORIES = {
    "kernel": ["kernel/"],
    "prompts": [".github/prompts/"],
    "experiments": ["experiments/", "projects/dgsf/experiments/"],
    "docs": ["docs/", "README"],
    "config": ["configs/", ".github/copilot-instructions.md", "spec_registry.yaml"],
    "tests": ["tests/", "test_"],
    "data": ["data/"],
}

CATEGORY_CONFIRM_LEVELS = {
    "kernel": ConfirmLevel.CONFIRM,
    "prompts": ConfirmLevel.CONFIRM,
    "experiments": ConfirmLevel.NOTIFY,
    "docs": ConfirmLevel.AUTO,
    "config": ConfirmLevel.CONFIRM,
    "tests": ConfirmLevel.NOTIFY,
    "data": ConfirmLevel.BLOCK,
}


def classify_path(path: str) -> str:
    """Classify a file path into a category."""
    path_lower = path.lower().replace("\\", "/")
    for category, patterns in CHANGE_CATEGORIES.items():
        for pattern in patterns:
            if pattern.lower() in path_lower:
                return category
    return "other"


def classify_changes(files: List[str]) -> Dict[str, List[str]]:
    """Classify a list of changed files by category.
    
    Args:
        files: List of file paths
        
    Returns:
        Dict mapping category names to lists of files
    """
    classified: Dict[str, List[str]] = {}
    for f in files:
        category = classify_path(f)
        classified.setdefault(category, []).append(f)
    return classified


def determine_confirm_level(classified: Dict[str, List[str]]) -> ConfirmLevel:
    """Determine the highest required confirmation level from classified changes.
    
    Higher enum values = more restrictive confirmation.
    """
    max_level = ConfirmLevel.AUTO
    for category in classified.keys():
        level = CATEGORY_CONFIRM_LEVELS.get(category, ConfirmLevel.CONFIRM)
        if level.value > max_level.value:
            max_level = level
    return max_level


# =============================================================================
# GIT STATUS OPERATIONS
# =============================================================================

def _run_git(args: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def is_git_repo(repo_path: Optional[Path] = None) -> bool:
    """Check if the path is inside a Git repository."""
    code, out, _ = _run_git(["rev-parse", "--is-inside-work-tree"], repo_path)
    return code == 0 and out.strip() == "true"


def get_git_status(repo_path: Optional[Path] = None) -> GitStatus:
    """Get comprehensive Git repository status.
    
    Args:
        repo_path: Path to repository (defaults to ROOT)
        
    Returns:
        GitStatus object with current repository state
        
    Raises:
        RuntimeError: If not inside a git repository
    """
    path = repo_path or ROOT
    
    if not is_git_repo(path):
        raise RuntimeError(f"Not inside a git repository: {path}")
    
    # Get current branch
    code, branch_out, _ = _run_git(["branch", "--show-current"], path)
    branch = branch_out.strip() if code == 0 else "HEAD"
    
    # Get porcelain status
    code, status_out, _ = _run_git(["status", "--porcelain"], path)
    
    staged: List[str] = []
    unstaged: List[str] = []
    untracked: List[str] = []
    
    if code == 0 and status_out.strip():
        # Split by newlines but preserve leading spaces in each line
        # (important for git porcelain format where ' M' means unstaged modified)
        for line in status_out.rstrip("\n").split("\n"):
            if len(line) < 3:
                continue
            index_status = line[0]
            worktree_status = line[1]
            # Git porcelain format: "XY filename" where XY are status chars
            # Position 2 is always a space, filename starts at position 3
            # Don't strip - filename might start with spaces (rare but valid)
            filename = line[3:]
            
            # Handle renames (format: "R  old -> new")
            if " -> " in filename:
                filename = filename.split(" -> ")[1]
            
            if index_status == "?":
                untracked.append(filename)
            elif index_status != " ":
                staged.append(filename)
            if worktree_status != " " and worktree_status != "?":
                unstaged.append(filename)
    
    is_clean = not (staged or unstaged or untracked)
    
    # Get latest tag
    code, tag_out, _ = _run_git(["describe", "--tags", "--always", "--abbrev=0"], path)
    latest_tag = tag_out.strip() if code == 0 else None
    
    # Get ahead/behind counts
    ahead, behind = 0, 0
    code, remote_out, _ = _run_git(["rev-parse", "--abbrev-ref", "@{upstream}"], path)
    if code == 0:
        upstream = remote_out.strip()
        code, ab_out, _ = _run_git(
            ["rev-list", "--left-right", "--count", f"HEAD...{upstream}"],
            path
        )
        if code == 0:
            parts = ab_out.strip().split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])
    
    return GitStatus(
        is_clean=is_clean,
        branch=branch,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        latest_tag=latest_tag,
        ahead=ahead,
        behind=behind,
        repo_path=path,
    )


def get_diff_stat(repo_path: Optional[Path] = None) -> str:
    """Get git diff --stat output."""
    code, out, _ = _run_git(["diff", "--stat"], repo_path)
    return out.strip() if code == 0 else ""


# =============================================================================
# COMMIT MESSAGE GENERATION
# =============================================================================

CONVENTIONAL_TYPES = {
    "kernel": "feat",
    "prompts": "feat",
    "experiments": "experiment",
    "docs": "docs",
    "config": "chore",
    "tests": "test",
    "data": "data",
    "other": "chore",
}


def generate_commit_message(
    classified: Dict[str, List[str]],
    context: str = "",
    task_id: Optional[str] = None,
) -> str:
    """Generate a Conventional Commits format message.
    
    Args:
        classified: Dict of category -> files from classify_changes()
        context: Additional context to include
        task_id: Optional TaskCard ID to reference
        
    Returns:
        Formatted commit message
    """
    # Determine primary category (most files or highest priority)
    priority_order = ["kernel", "prompts", "config", "experiments", "tests", "docs", "other"]
    primary = "other"
    for cat in priority_order:
        if cat in classified and classified[cat]:
            primary = cat
            break
    
    commit_type = CONVENTIONAL_TYPES.get(primary, "chore")
    scope = primary if primary != "other" else None
    
    # Build summary line
    file_count = sum(len(files) for files in classified.values())
    if file_count == 1:
        # Single file change - use filename
        single_file = list(classified.values())[0][0]
        summary = Path(single_file).name
    else:
        # Multiple files - summarize
        summary = f"update {file_count} files across {len(classified)} categories"
    
    # Format type(scope): summary
    if scope:
        first_line = f"{commit_type}({scope}): {summary}"
    else:
        first_line = f"{commit_type}: {summary}"
    
    # Build body
    body_parts = []
    
    if context:
        body_parts.append(context)
    
    if task_id:
        body_parts.append(f"Task: {task_id}")
    
    # List changed files by category
    body_parts.append("\nChanges:")
    for cat, files in classified.items():
        for f in files[:5]:  # Limit to 5 per category
            body_parts.append(f"  - [{cat}] {f}")
        if len(files) > 5:
            body_parts.append(f"  - ... and {len(files) - 5} more")
    
    return first_line + "\n\n" + "\n".join(body_parts)


# =============================================================================
# COMMIT PLAN GENERATION
# =============================================================================

def generate_commit_plan(
    status: GitStatus,
    trigger_context: str = "",
    task_id: Optional[str] = None,
    auto_tag: bool = False,
    tag_prefix: str = "exp",
    experiment_metrics: Optional[Dict[str, Any]] = None,
) -> CommitPlan:
    """Generate a commit plan based on current Git status.
    
    Args:
        status: GitStatus from get_git_status()
        trigger_context: Context about what triggered this (e.g., "dgsf_execute complete")
        task_id: Optional TaskCard ID
        auto_tag: Whether to auto-generate a tag
        tag_prefix: Prefix for auto-generated tags (exp, milestone, release)
        experiment_metrics: Dict with OOS Sharpe, OOS/IS ratio etc. for tag message
        
    Returns:
        CommitPlan ready for execution or review
    """
    if status.is_clean:
        return CommitPlan(
            message="",
            confirm_level=ConfirmLevel.AUTO,
            context={"status": "clean", "trigger": trigger_context},
        )
    
    # Collect all changed files
    all_files = status.staged + status.unstaged + status.untracked
    classified = classify_changes(all_files)
    
    # Determine confirmation level
    confirm_level = determine_confirm_level(classified)
    
    # Generate commit message
    message = generate_commit_message(classified, trigger_context, task_id)
    
    # Determine files to stage (unstaged + untracked)
    files_to_stage = status.unstaged + status.untracked
    
    # Auto-tag logic
    tag = None
    tag_message = None
    if auto_tag and experiment_metrics:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tag = f"{tag_prefix}/{task_id or 'snapshot'}/{timestamp}"
        
        # Build tag message with metrics
        tag_lines = [f"Tag: {tag}"]
        if "oos_sharpe" in experiment_metrics:
            tag_lines.append(f"OOS Sharpe: {experiment_metrics['oos_sharpe']}")
        if "oos_is_ratio" in experiment_metrics:
            tag_lines.append(f"OOS/IS Ratio: {experiment_metrics['oos_is_ratio']}")
        tag_message = "\n".join(tag_lines)
    
    return CommitPlan(
        message=message,
        files_to_stage=files_to_stage,
        tag=tag,
        tag_message=tag_message,
        push_remote=False,  # Default to not pushing
        push_tags=auto_tag,
        confirm_level=confirm_level,
        context={
            "trigger": trigger_context,
            "task_id": task_id,
            "classified": classified,
            "metrics": experiment_metrics,
        },
    )


# =============================================================================
# PLAN EXECUTION
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of executing a CommitPlan."""
    success: bool
    actions_taken: List[str] = field(default_factory=list)
    commit_sha: Optional[str] = None
    tag_created: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    dry_run: bool = False


def format_plan_for_review(plan: CommitPlan, status: GitStatus) -> str:
    """Format a CommitPlan for human review.
    
    Returns a Markdown-formatted string suitable for display.
    """
    lines = [
        "## 📦 Git Commit Plan",
        "",
        f"**Branch**: `{status.branch}`",
        f"**Confirm Level**: {plan.confirm_level.name}",
        "",
    ]
    
    if status.ahead > 0 or status.behind > 0:
        lines.append(f"**Remote Status**: {status.ahead} ahead, {status.behind} behind")
        lines.append("")
    
    # Changes section
    lines.append("### Changes")
    classified = plan.context.get("classified", {})
    for category, files in classified.items():
        for f in files:
            lines.append(f"  - [{category[0].upper()}] `{f}`")
    
    lines.append("")
    lines.append("### Commit Message")
    lines.append("```")
    lines.append(plan.message)
    lines.append("```")
    
    if plan.tag:
        lines.append("")
        lines.append(f"### Tag: `{plan.tag}`")
        if plan.tag_message:
            lines.append("```")
            lines.append(plan.tag_message)
            lines.append("```")
    
    if plan.push_remote:
        lines.append("")
        lines.append("### Push")
        lines.append(f"  - Push commits to `origin/{status.branch}`")
        if plan.push_tags:
            lines.append(f"  - Push tag `{plan.tag}`")
    
    # Action prompt based on confirm level
    lines.append("")
    if plan.confirm_level == ConfirmLevel.AUTO:
        lines.append("*Auto-executing...*")
    elif plan.confirm_level == ConfirmLevel.NOTIFY:
        lines.append("*Executing with notification...*")
    elif plan.confirm_level == ConfirmLevel.CONFIRM:
        lines.append("**Proceed? [Y/n]**")
    elif plan.confirm_level == ConfirmLevel.BLOCK:
        lines.append("**⚠️ Manual execution required. Commands:**")
        lines.append("```bash")
        if plan.files_to_stage:
            lines.append(f"git add {' '.join(plan.files_to_stage)}")
        lines.append(f'git commit -m "{plan.message.split(chr(10))[0]}"')
        if plan.tag:
            lines.append(f'git tag -a {plan.tag} -m "{plan.tag_message or plan.tag}"')
        if plan.push_remote:
            lines.append(f"git push origin {status.branch}")
        if plan.push_tags and plan.tag:
            lines.append(f"git push origin {plan.tag}")
        lines.append("```")
    
    return "\n".join(lines)


def execute_plan(
    plan: CommitPlan,
    dry_run: bool = True,
    repo_path: Optional[Path] = None,
) -> ExecutionResult:
    """Execute a CommitPlan.
    
    Args:
        plan: The CommitPlan to execute
        dry_run: If True, only simulate actions (print commands)
        repo_path: Path to repository
        
    Returns:
        ExecutionResult with status and details
    """
    result = ExecutionResult(success=True, dry_run=dry_run)
    path = repo_path or ROOT
    
    if not plan.message:
        result.actions_taken.append("No changes to commit (clean tree)")
        return result
    
    try:
        # Stage files
        if plan.files_to_stage:
            if dry_run:
                result.actions_taken.append(f"[DRY-RUN] git add {' '.join(plan.files_to_stage)}")
            else:
                code, _, err = _run_git(["add"] + plan.files_to_stage, path)
                if code != 0:
                    result.errors.append(f"Failed to stage files: {err}")
                    result.success = False
                    return result
                result.actions_taken.append(f"Staged {len(plan.files_to_stage)} files")
        
        # Commit
        if dry_run:
            result.actions_taken.append(f"[DRY-RUN] git commit -m '{plan.message.split(chr(10))[0]}...'")
        else:
            code, out, err = _run_git(["commit", "-m", plan.message], path)
            if code != 0:
                result.errors.append(f"Failed to commit: {err}")
                result.success = False
                return result
            # Extract commit SHA
            sha_match = re.search(r"\[[\w/-]+ ([a-f0-9]+)\]", out)
            if sha_match:
                result.commit_sha = sha_match.group(1)
            result.actions_taken.append(f"Committed: {result.commit_sha or 'success'}")
        
        # Tag
        if plan.tag:
            if dry_run:
                result.actions_taken.append(f"[DRY-RUN] git tag -a {plan.tag}")
            else:
                tag_args = ["tag", "-a", plan.tag]
                if plan.tag_message:
                    tag_args.extend(["-m", plan.tag_message])
                else:
                    tag_args.extend(["-m", plan.tag])
                code, _, err = _run_git(tag_args, path)
                if code != 0:
                    result.errors.append(f"Failed to create tag: {err}")
                    result.success = False
                    return result
                result.tag_created = plan.tag
                result.actions_taken.append(f"Created tag: {plan.tag}")
        
        # Push
        if plan.push_remote and not dry_run:
            code, _, err = _run_git(["push", "origin", "HEAD"], path)
            if code != 0:
                result.errors.append(f"Failed to push: {err}")
                result.success = False
                return result
            result.actions_taken.append("Pushed to origin")
            
            if plan.push_tags and plan.tag:
                code, _, err = _run_git(["push", "origin", plan.tag], path)
                if code != 0:
                    result.errors.append(f"Failed to push tag: {err}")
                    result.success = False
                    return result
                result.actions_taken.append(f"Pushed tag: {plan.tag}")
        elif plan.push_remote and dry_run:
            result.actions_taken.append("[DRY-RUN] git push origin HEAD")
            if plan.push_tags and plan.tag:
                result.actions_taken.append(f"[DRY-RUN] git push origin {plan.tag}")
    
    except Exception as e:
        result.success = False
        result.errors.append(str(e))
    
    return result


# =============================================================================
# HIGH-LEVEL WORKFLOW FUNCTIONS
# =============================================================================

def run_git_ops_workflow(
    trigger_context: str = "",
    task_id: Optional[str] = None,
    auto_tag: bool = False,
    tag_prefix: str = "exp",
    experiment_metrics: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
    repo_path: Optional[Path] = None,
) -> Tuple[CommitPlan, ExecutionResult, str]:
    """Run the complete Git ops workflow.
    
    This is the main entry point for the Git ops subprocess.
    
    Args:
        trigger_context: What triggered this workflow
        task_id: Associated TaskCard ID
        auto_tag: Whether to create a tag
        tag_prefix: Tag prefix (exp/milestone/release)
        experiment_metrics: Metrics for tag message
        dry_run: If True, simulate only
        repo_path: Repository path
        
    Returns:
        Tuple of (plan, result, formatted_output)
    """
    path = repo_path or ROOT
    
    # Phase 1: Status check
    status = get_git_status(path)
    
    # Phase 2: Generate plan
    plan = generate_commit_plan(
        status=status,
        trigger_context=trigger_context,
        task_id=task_id,
        auto_tag=auto_tag,
        tag_prefix=tag_prefix,
        experiment_metrics=experiment_metrics,
    )
    
    # Format for review
    formatted = format_plan_for_review(plan, status)
    
    # Phase 3: Execute (or simulate)
    if plan.confirm_level == ConfirmLevel.BLOCK:
        # BLOCK level = human must execute
        result = ExecutionResult(
            success=True,
            dry_run=True,
            actions_taken=["Blocked: Manual execution required"],
        )
    elif dry_run or plan.confirm_level == ConfirmLevel.CONFIRM:
        # Dry run or needs confirmation
        result = execute_plan(plan, dry_run=True, repo_path=path)
    else:
        # AUTO or NOTIFY - execute
        result = execute_plan(plan, dry_run=False, repo_path=path)
    
    return plan, result, formatted
