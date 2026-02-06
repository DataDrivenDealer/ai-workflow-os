"""
Git Branch Validator — Branch naming policy enforcement.

Loads rules from configs/git_branch_policy.yaml and validates branch names.
Supports BLOCK / WARN / NOTIFY enforcement levels.

Version: 1.0.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from kernel.paths import ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLICY_PATH = ROOT / "configs" / "git_branch_policy.yaml"


class EnforcementLevel(Enum):
    """How strictly to enforce branch naming rules."""
    NOTIFY = "NOTIFY"   # Informational only
    WARN = "WARN"       # Warn but allow
    BLOCK = "BLOCK"     # Reject non-conforming names


@dataclass
class BranchType:
    """Describes one allowed branch type from policy."""
    name: str
    pattern: str
    description: str
    example: str
    merge_target: Optional[str]
    require_task_id: bool = False
    max_lifetime_days: int = 30

    # Compiled regex (computed once)
    _compiled: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    def regex(self) -> re.Pattern:
        if self._compiled is None:
            self._compiled = re.compile(self.pattern)
        return self._compiled


@dataclass
class BranchPolicy:
    """Full branch policy loaded from YAML."""
    strategy: str
    main_branch: str
    branch_types: Dict[str, BranchType]
    enforcement_level: EnforcementLevel
    protected_branches: List[str]
    allow_override: bool
    exempt_patterns: List[re.Pattern]

    @staticmethod
    def load(path: Optional[Path] = None) -> "BranchPolicy":
        """Load policy from YAML file.

        Args:
            path: Override config path (mostly for testing).

        Returns:
            Populated BranchPolicy.

        Raises:
            FileNotFoundError: Config file missing.
            ValueError: Config file malformed.
        """
        cfg_path = path or POLICY_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(f"Branch policy not found: {cfg_path}")

        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ValueError(f"Invalid branch policy: expected dict, got {type(raw)}")

        # Parse branch types
        branch_types: Dict[str, BranchType] = {}
        for name, cfg in raw.get("branches", {}).get("types", {}).items():
            branch_types[name] = BranchType(
                name=name,
                pattern=cfg["pattern"],
                description=cfg.get("description", ""),
                example=cfg.get("example", ""),
                merge_target=cfg.get("merge_target"),
                require_task_id=cfg.get("require_task_id", False),
                max_lifetime_days=cfg.get("max_lifetime_days", 30),
            )

        enforcement_raw = raw.get("enforcement", {})
        level_str = enforcement_raw.get("level", "WARN").upper()
        try:
            level = EnforcementLevel(level_str)
        except ValueError:
            level = EnforcementLevel.WARN

        exempt_compiled = [
            re.compile(p)
            for p in enforcement_raw.get("exempt_patterns", [])
        ]

        return BranchPolicy(
            strategy=raw.get("strategy", "github_flow"),
            main_branch=raw.get("branches", {}).get("main_branch", "main"),
            branch_types=branch_types,
            enforcement_level=level,
            protected_branches=enforcement_raw.get("protected_branches", ["main"]),
            allow_override=enforcement_raw.get("allow_override", True),
            exempt_patterns=exempt_compiled,
        )


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of branch name validation."""
    valid: bool
    branch: str
    matched_type: Optional[str] = None
    message: str = ""
    enforcement: EnforcementLevel = EnforcementLevel.BLOCK
    examples: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core validation functions
# ---------------------------------------------------------------------------

_cached_policy: Optional[BranchPolicy] = None


def _get_policy(path: Optional[Path] = None) -> BranchPolicy:
    """Return cached policy (loads once)."""
    global _cached_policy
    if _cached_policy is None or path is not None:
        _cached_policy = BranchPolicy.load(path)
    return _cached_policy


def reload_policy(path: Optional[Path] = None) -> BranchPolicy:
    """Force-reload the policy (useful after config changes)."""
    global _cached_policy
    _cached_policy = None
    return _get_policy(path)


def validate_branch_name(
    branch: str,
    policy_path: Optional[Path] = None,
) -> ValidationResult:
    """Validate a branch name against the configured policy.

    Args:
        branch: The branch name to validate.
        policy_path: Override path to policy YAML (for testing).

    Returns:
        ValidationResult with valid=True/False and details.
    """
    policy = _get_policy(policy_path)

    # Exempt patterns (HEAD, detached, etc.)
    for pat in policy.exempt_patterns:
        if pat.match(branch):
            return ValidationResult(
                valid=True,
                branch=branch,
                matched_type="exempt",
                message="Branch name is exempt from policy.",
                enforcement=policy.enforcement_level,
            )

    # Main branch is always valid
    if branch == policy.main_branch:
        return ValidationResult(
            valid=True,
            branch=branch,
            matched_type="main",
            message=f"Main branch '{policy.main_branch}'.",
            enforcement=policy.enforcement_level,
        )

    # Check against each branch type
    for type_name, bt in policy.branch_types.items():
        if bt.regex().match(branch):
            return ValidationResult(
                valid=True,
                branch=branch,
                matched_type=type_name,
                message=f"Matches '{type_name}' pattern.",
                enforcement=policy.enforcement_level,
            )

    # No match — gather examples for error message
    examples = [
        f"  {bt.name}: {bt.example}"
        for bt in policy.branch_types.values()
    ]
    hint = "\n".join(examples)

    return ValidationResult(
        valid=False,
        branch=branch,
        message=(
            f"Branch name '{branch}' does not match any allowed pattern.\n"
            f"Enforcement level: {policy.enforcement_level.value}\n"
            f"\nAllowed formats:\n{hint}"
        ),
        enforcement=policy.enforcement_level,
        examples=[bt.example for bt in policy.branch_types.values()],
    )


def is_protected_branch(
    branch: str,
    policy_path: Optional[Path] = None,
) -> bool:
    """Check whether *branch* is listed as protected."""
    policy = _get_policy(policy_path)
    return branch in policy.protected_branches


def get_branch_type(
    branch: str,
    policy_path: Optional[Path] = None,
) -> Optional[str]:
    """Return the branch type name or None if unrecognized."""
    result = validate_branch_name(branch, policy_path)
    return result.matched_type


def suggest_branch_name(
    branch_type: str,
    task_id: str = "",
    description: str = "",
    experiment_number: int = 0,
    policy_path: Optional[Path] = None,
) -> Optional[str]:
    """Suggest a conforming branch name.

    Args:
        branch_type: One of feature, experiment, hotfix, release, worktree.
        task_id: Task ID (required for feature, hotfix).
        description: Short slug description.
        experiment_number: Experiment number (for experiment branches).
        policy_path: Override policy path.

    Returns:
        Suggested branch name string, or None if type unknown.
    """
    _get_policy(policy_path)  # ensure loaded

    desc_slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-") or "unnamed"

    if branch_type == "feature":
        tid = task_id or "TASK_000"
        return f"feature/{tid}-{desc_slug}"
    elif branch_type == "experiment":
        nn = f"{experiment_number:02d}" if experiment_number < 100 else str(experiment_number)
        return f"experiment/t{nn}_{desc_slug.replace('-', '_')}"
    elif branch_type == "hotfix":
        tid = task_id or "URGENT_000"
        return f"hotfix/{tid}-{desc_slug}"
    elif branch_type == "release":
        return f"release/v{description or '0.0.0'}"
    elif branch_type == "worktree":
        return f"worktree/{description or 'backup-unnamed'}"
    return None


# ---------------------------------------------------------------------------
# Format helpers (for hooks / CLI output)
# ---------------------------------------------------------------------------

def format_validation_error(result: ValidationResult) -> str:
    """Format a validation failure into a human-friendly message for hooks."""
    lines = [
        "",
        "═" * 60,
        "  GIT BRANCH POLICY VIOLATION",
        "═" * 60,
        "",
        f"  Branch:      {result.branch}",
        f"  Enforcement: {result.enforcement.value}",
        "",
        "  This branch name does not conform to the project",
        "  naming policy defined in configs/git_branch_policy.yaml.",
        "",
        "  Allowed formats:",
    ]
    for ex in result.examples:
        lines.append(f"    ✓ {ex}")
    lines.extend([
        "",
        "  To create a conforming branch:",
        "    git checkout -b feature/TASK_ID-description",
        "    git checkout -b experiment/t05_my_experiment",
        "",
        "═" * 60,
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point for branch validation."""
    import sys
    import subprocess

    # Get current branch
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("[ERROR] Not inside a git repository.", file=sys.stderr)
        return 1

    branch = result.stdout.strip()
    if not branch:
        print("[WARN] Detached HEAD — skipping branch validation.")
        return 0

    vr = validate_branch_name(branch)
    if vr.valid:
        print(f"[OK] Branch '{branch}' → type='{vr.matched_type}'")
        return 0

    # Print error
    print(format_validation_error(vr), file=sys.stderr)

    if vr.enforcement == EnforcementLevel.BLOCK:
        return 1
    elif vr.enforcement == EnforcementLevel.WARN:
        print("[WARN] Proceeding despite non-conforming branch name.")
        return 0
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
