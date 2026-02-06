"""
Tests for kernel/git_branch_validator.py and kernel/git_setup_check.py.

Covers:
  - Branch name validation against policy
  - Enforcement levels (BLOCK / WARN / NOTIFY)
  - Protected branch detection
  - Branch name suggestion
  - Hooks setup checking
  - Policy loading
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_policy(tmp_path: Path) -> Path:
    """Write a minimal branch policy YAML and return its path."""
    policy = {
        "version": "1.0.0",
        "strategy": "github_flow",
        "branches": {
            "main_branch": "main",
            "types": {
                "feature": {
                    "pattern": r"^feature/[A-Z][A-Z0-9_]+-[a-z0-9][a-z0-9_-]+$",
                    "description": "Feature branches",
                    "example": "feature/GIT_001-branch-policy",
                    "merge_target": "main",
                    "require_task_id": True,
                    "max_lifetime_days": 30,
                },
                "experiment": {
                    "pattern": r"^experiment/t\d{2,3}_[a-z0-9][a-z0-9_-]+$",
                    "description": "Experiment branches",
                    "example": "experiment/t05_sharpe_validation",
                    "merge_target": "main",
                    "require_task_id": False,
                    "max_lifetime_days": 60,
                },
                "hotfix": {
                    "pattern": r"^hotfix/[A-Z][A-Z0-9_]+-[a-z0-9][a-z0-9_-]+$",
                    "description": "Hotfix branches",
                    "example": "hotfix/URGENT_001-fix-crash",
                    "merge_target": "main",
                    "require_task_id": True,
                    "max_lifetime_days": 7,
                },
                "release": {
                    "pattern": r"^release/v\d+\.\d+\.\d+$",
                    "description": "Release branches",
                    "example": "release/v1.0.0",
                    "merge_target": "main",
                    "require_task_id": False,
                    "max_lifetime_days": 14,
                },
            },
        },
        "enforcement": {
            "level": "BLOCK",
            "protected_branches": ["main"],
            "allow_override": True,
            "exempt_patterns": [r"^HEAD$", r"^(no branch)$"],
        },
    }
    p = tmp_path / "git_branch_policy.yaml"
    p.write_text(yaml.dump(policy, default_flow_style=False), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _reset_cache() -> Generator:
    """Reset the cached policy between tests."""
    from kernel import git_branch_validator
    git_branch_validator._cached_policy = None
    yield
    git_branch_validator._cached_policy = None


# =========================================================================
# Branch Validator Tests
# =========================================================================

class TestValidateBranchName:
    """Tests for validate_branch_name()."""

    # ── Valid names ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("branch", [
        "feature/GIT_001-branch-policy",
        "feature/TASK_123-add-feature",
        "feature/DE7_V2-factor-panel",
    ])
    def test_valid_feature_branches(self, tmp_policy: Path, branch: str) -> None:
        from kernel.git_branch_validator import validate_branch_name
        result = validate_branch_name(branch, policy_path=tmp_policy)
        assert result.valid, f"Expected valid: {branch} — {result.message}"
        assert result.matched_type == "feature"

    @pytest.mark.parametrize("branch", [
        "experiment/t05_sharpe_validation",
        "experiment/t12_oos_test",
        "experiment/t100_large_scale",
    ])
    def test_valid_experiment_branches(self, tmp_policy: Path, branch: str) -> None:
        from kernel.git_branch_validator import validate_branch_name
        result = validate_branch_name(branch, policy_path=tmp_policy)
        assert result.valid, f"Expected valid: {branch} — {result.message}"
        assert result.matched_type == "experiment"

    @pytest.mark.parametrize("branch", [
        "hotfix/URGENT_001-fix-crash",
        "hotfix/BUG_042-null-pointer",
    ])
    def test_valid_hotfix_branches(self, tmp_policy: Path, branch: str) -> None:
        from kernel.git_branch_validator import validate_branch_name
        result = validate_branch_name(branch, policy_path=tmp_policy)
        assert result.valid
        assert result.matched_type == "hotfix"

    @pytest.mark.parametrize("branch", [
        "release/v1.0.0",
        "release/v2.3.1",
    ])
    def test_valid_release_branches(self, tmp_policy: Path, branch: str) -> None:
        from kernel.git_branch_validator import validate_branch_name
        result = validate_branch_name(branch, policy_path=tmp_policy)
        assert result.valid
        assert result.matched_type == "release"

    def test_main_branch_valid(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import validate_branch_name
        result = validate_branch_name("main", policy_path=tmp_policy)
        assert result.valid
        assert result.matched_type == "main"

    def test_exempt_head(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import validate_branch_name
        result = validate_branch_name("HEAD", policy_path=tmp_policy)
        assert result.valid
        assert result.matched_type == "exempt"

    # ── Invalid names ────────────────────────────────────────────────────

    @pytest.mark.parametrize("branch", [
        "my-random-branch",
        "fix-something",
        "feature_bad_name",
        "feature/lowercase_task-desc",
        "Feature/GIT_001-branch",       # Capital F
        "experiment/5_too_short",        # t missing
        "experiment/abc_not_number",     # no t prefix
        "release/1.0.0",                 # no v prefix
        "hotfix/lowertask-fix",          # no uppercase task
    ])
    def test_invalid_branches_blocked(self, tmp_policy: Path, branch: str) -> None:
        from kernel.git_branch_validator import validate_branch_name, EnforcementLevel
        result = validate_branch_name(branch, policy_path=tmp_policy)
        assert not result.valid, f"Expected invalid: {branch}"
        assert result.enforcement == EnforcementLevel.BLOCK
        assert len(result.examples) > 0

    # ── Format error output ──────────────────────────────────────────────

    def test_format_validation_error(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import validate_branch_name, format_validation_error
        result = validate_branch_name("bad-branch", policy_path=tmp_policy)
        assert not result.valid
        err = format_validation_error(result)
        assert "BRANCH POLICY VIOLATION" in err
        assert "bad-branch" in err
        assert "feature/" in err


class TestProtectedBranch:

    def test_main_is_protected(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import is_protected_branch
        assert is_protected_branch("main", policy_path=tmp_policy)

    def test_feature_not_protected(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import is_protected_branch
        assert not is_protected_branch("feature/X_001-stuff", policy_path=tmp_policy)


class TestSuggestBranchName:

    def test_suggest_feature(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import suggest_branch_name, validate_branch_name
        name = suggest_branch_name("feature", task_id="GIT_001",
                                   description="branch policy", policy_path=tmp_policy)
        assert name is not None
        assert name.startswith("feature/")
        result = validate_branch_name(name, policy_path=tmp_policy)
        assert result.valid

    def test_suggest_experiment(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import suggest_branch_name, validate_branch_name
        name = suggest_branch_name("experiment", experiment_number=5,
                                   description="sharpe validation", policy_path=tmp_policy)
        assert name is not None
        assert name.startswith("experiment/t05_")
        result = validate_branch_name(name, policy_path=tmp_policy)
        assert result.valid

    def test_suggest_hotfix(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import suggest_branch_name, validate_branch_name
        name = suggest_branch_name("hotfix", task_id="URGENT_001",
                                   description="fix crash", policy_path=tmp_policy)
        assert name is not None
        result = validate_branch_name(name, policy_path=tmp_policy)
        assert result.valid

    def test_suggest_release(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import suggest_branch_name, validate_branch_name
        name = suggest_branch_name("release", description="1.0.0", policy_path=tmp_policy)
        assert name == "release/v1.0.0"
        result = validate_branch_name(name, policy_path=tmp_policy)
        assert result.valid

    def test_suggest_unknown_type(self, tmp_policy: Path) -> None:
        from kernel.git_branch_validator import suggest_branch_name
        assert suggest_branch_name("foobar", policy_path=tmp_policy) is None


class TestPolicyLoading:

    def test_missing_file_raises(self) -> None:
        from kernel.git_branch_validator import BranchPolicy
        with pytest.raises(FileNotFoundError):
            BranchPolicy.load(Path("/nonexistent/policy.yaml"))

    def test_load_real_policy(self) -> None:
        """Ensure the actual configs/git_branch_policy.yaml loads without error."""
        from kernel.git_branch_validator import BranchPolicy
        from kernel.paths import ROOT
        real_path = ROOT / "configs" / "git_branch_policy.yaml"
        if not real_path.exists():
            pytest.skip("Real policy file not present")
        policy = BranchPolicy.load(real_path)
        assert policy.strategy == "github_flow"
        assert len(policy.branch_types) >= 4


# =========================================================================
# Setup Check Tests
# =========================================================================

class TestGitSetupCheck:

    def test_check_no_git_dir(self, tmp_path: Path) -> None:
        from kernel.git_setup_check import SetupStatus
        # Patch GIT_DIR to a non-existent path
        with patch("kernel.git_setup_check.GIT_DIR", tmp_path / ".git"):
            from kernel.git_setup_check import check_git_hooks
            status = check_git_hooks()
            assert not status.git_available

    def test_install_hooks(self, tmp_path: Path) -> None:
        """Simulate installing hooks to a temp .git/hooks/ directory."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir()

        # Create fake source hooks
        src_dir = tmp_path / "hooks_src"
        src_dir.mkdir()
        for h in ["pre-commit", "pre-push"]:
            (src_dir / h).write_text(f"#!/bin/sh\necho {h}\n")

        with patch("kernel.git_setup_check.GIT_DIR", git_dir), \
             patch("kernel.git_setup_check.GIT_HOOKS_DIR", hooks_dir), \
             patch("kernel.git_setup_check.HOOKS_SOURCE_DIR", src_dir), \
             patch("kernel.git_setup_check.REQUIRED_HOOKS", ["pre-commit", "pre-push"]):
            from kernel.git_setup_check import install_hooks
            status = install_hooks(force=True)
            assert status.hooks_installed
            assert (hooks_dir / "pre-commit").exists()
            assert (hooks_dir / "pre-push").exists()

    def test_format_setup_status(self) -> None:
        from kernel.git_setup_check import SetupStatus, format_setup_status
        status = SetupStatus(
            hooks_installed=False,
            missing_hooks=["pre-commit", "pre-push"],
            installed_hooks=["post-tag"],
        )
        output = format_setup_status(status)
        assert "Missing" in output
        assert "pre-commit" in output
