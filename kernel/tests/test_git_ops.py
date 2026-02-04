"""
Unit tests for kernel/git_ops.py

Tests Git operations module: status checking, commit plan generation, dry-run execution.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kernel.git_ops import (
    ConfirmLevel,
    GitStatus,
    CommitPlan,
    ExecutionResult,
    classify_path,
    classify_changes,
    determine_confirm_level,
    generate_commit_message,
    generate_commit_plan,
    format_plan_for_review,
    execute_plan,
    is_git_repo,
    get_git_status,
    run_git_ops_workflow,
    CHANGE_CATEGORIES,
    CATEGORY_CONFIRM_LEVELS,
)


# =============================================================================
# CLASSIFICATION TESTS
# =============================================================================

class TestClassifyPath:
    """Tests for classify_path function."""

    def test_kernel_files(self):
        assert classify_path("kernel/git_ops.py") == "kernel"
        assert classify_path("kernel/tests/test_git_ops.py") == "kernel"

    def test_prompt_files(self):
        assert classify_path(".github/prompts/dgsf_execute.prompt.md") == "prompts"

    def test_experiment_files(self):
        assert classify_path("experiments/t05_oos/config.yaml") == "experiments"
        assert classify_path("projects/dgsf/experiments/t01/run.log") == "experiments"

    def test_docs_files(self):
        assert classify_path("docs/ARCHITECTURE.md") == "docs"
        assert classify_path("README.md") == "docs"

    def test_config_files(self):
        assert classify_path("configs/gates.yaml") == "config"
        assert classify_path(".github/copilot-instructions.md") == "config"

    def test_test_files(self):
        assert classify_path("tests/test_something.py") == "tests"
        assert classify_path("test_foo.py") == "tests"

    def test_data_files(self):
        assert classify_path("data/processed/features.parquet") == "data"

    def test_other_files(self):
        assert classify_path("random_file.txt") == "other"
        assert classify_path("scripts/run.sh") == "other"

    def test_case_insensitive(self):
        assert classify_path("KERNEL/OS.PY") == "kernel"
        assert classify_path("Docs/README.md") == "docs"

    def test_backslash_paths(self):
        assert classify_path("kernel\\git_ops.py") == "kernel"
        assert classify_path(".github\\prompts\\test.md") == "prompts"


class TestClassifyChanges:
    """Tests for classify_changes function."""

    def test_empty_list(self):
        result = classify_changes([])
        assert result == {}

    def test_single_category(self):
        files = ["kernel/a.py", "kernel/b.py"]
        result = classify_changes(files)
        assert "kernel" in result
        assert len(result["kernel"]) == 2

    def test_multiple_categories(self):
        files = [
            "kernel/git_ops.py",
            "docs/README.md",
            "experiments/t01/config.yaml",
        ]
        result = classify_changes(files)
        assert "kernel" in result
        assert "docs" in result
        assert "experiments" in result


class TestDetermineConfirmLevel:
    """Tests for determine_confirm_level function."""

    def test_empty_returns_auto(self):
        assert determine_confirm_level({}) == ConfirmLevel.AUTO

    def test_docs_only_returns_auto(self):
        result = determine_confirm_level({"docs": ["README.md"]})
        assert result == ConfirmLevel.AUTO

    def test_kernel_returns_confirm(self):
        result = determine_confirm_level({"kernel": ["os.py"]})
        assert result == ConfirmLevel.CONFIRM

    def test_data_returns_block(self):
        result = determine_confirm_level({"data": ["raw/file.csv"]})
        assert result == ConfirmLevel.BLOCK

    def test_mixed_returns_highest(self):
        result = determine_confirm_level({
            "docs": ["README.md"],
            "kernel": ["os.py"],
            "data": ["file.csv"],
        })
        assert result == ConfirmLevel.BLOCK

    def test_experiments_returns_notify(self):
        result = determine_confirm_level({"experiments": ["t01/config.yaml"]})
        assert result == ConfirmLevel.NOTIFY


# =============================================================================
# COMMIT MESSAGE GENERATION TESTS
# =============================================================================

class TestGenerateCommitMessage:
    """Tests for generate_commit_message function."""

    def test_single_kernel_file(self):
        classified = {"kernel": ["kernel/git_ops.py"]}
        msg = generate_commit_message(classified)
        assert msg.startswith("feat(kernel):")
        assert "git_ops.py" in msg

    def test_docs_file(self):
        classified = {"docs": ["docs/README.md"]}
        msg = generate_commit_message(classified)
        assert msg.startswith("docs(docs):")

    def test_multiple_categories(self):
        classified = {
            "kernel": ["kernel/a.py", "kernel/b.py"],
            "docs": ["README.md"],
        }
        msg = generate_commit_message(classified)
        # Should use kernel as primary (higher priority)
        assert "feat(kernel):" in msg
        assert "3 files" in msg

    def test_with_context(self):
        classified = {"kernel": ["kernel/git_ops.py"]}
        msg = generate_commit_message(classified, context="After dgsf_execute")
        assert "After dgsf_execute" in msg

    def test_with_task_id(self):
        classified = {"kernel": ["kernel/git_ops.py"]}
        msg = generate_commit_message(classified, task_id="TASK-001")
        assert "Task: TASK-001" in msg

    def test_experiments_type(self):
        classified = {"experiments": ["experiments/t05/config.yaml"]}
        msg = generate_commit_message(classified)
        assert msg.startswith("experiment(experiments):")


# =============================================================================
# GIT STATUS TESTS
# =============================================================================

class TestGitStatus:
    """Tests for GitStatus dataclass."""

    def test_clean_status(self):
        status = GitStatus(is_clean=True, branch="main")
        assert status.is_clean
        assert status.branch == "main"
        assert status.staged == []
        assert status.unstaged == []
        assert status.untracked == []

    def test_dirty_status(self):
        status = GitStatus(
            is_clean=False,
            branch="feature/test",
            staged=["a.py"],
            unstaged=["b.py"],
            untracked=["c.py"],
        )
        assert not status.is_clean
        assert len(status.staged) == 1
        assert len(status.unstaged) == 1
        assert len(status.untracked) == 1


class TestIsGitRepo:
    """Tests for is_git_repo function."""

    @patch("kernel.git_ops._run_git")
    def test_inside_git_repo(self, mock_run):
        mock_run.return_value = (0, "true\n", "")
        assert is_git_repo() is True

    @patch("kernel.git_ops._run_git")
    def test_outside_git_repo(self, mock_run):
        mock_run.return_value = (128, "", "fatal: not a git repository")
        assert is_git_repo() is False


class TestGetGitStatus:
    """Tests for get_git_status function."""

    @patch("kernel.git_ops._run_git")
    def test_clean_repo(self, mock_run):
        def side_effect(args, cwd=None):
            if args[0] == "rev-parse":
                return (0, "true\n", "")
            if args[0] == "branch":
                return (0, "main\n", "")
            if args[0] == "status":
                return (0, "", "")
            if args[0] == "describe":
                return (0, "v1.0.0\n", "")
            return (0, "", "")

        mock_run.side_effect = side_effect
        status = get_git_status()
        assert status.is_clean
        assert status.branch == "main"
        assert status.latest_tag == "v1.0.0"

    @patch("kernel.git_ops._run_git")
    def test_dirty_repo(self, mock_run):
        def side_effect(args, cwd=None):
            if args[0] == "rev-parse":
                if args[1] == "--is-inside-work-tree":
                    return (0, "true\n", "")
                return (1, "", "")
            if args[0] == "branch":
                return (0, "feature/test\n", "")
            if args[0] == "status":
                # Git porcelain format: XY filename
                # ' M' = not staged, modified in worktree
                # '??' = untracked
                # 'A ' = added to index
                return (0, " M kernel/os.py\n?? new_file.txt\nA  staged.py\n", "")
            if args[0] == "describe":
                return (0, "v1.0.0\n", "")
            return (0, "", "")

        mock_run.side_effect = side_effect
        status = get_git_status()
        assert not status.is_clean
        assert "kernel/os.py" in status.unstaged
        assert "new_file.txt" in status.untracked
        assert "staged.py" in status.staged

    @patch("kernel.git_ops._run_git")
    def test_not_git_repo_raises(self, mock_run):
        mock_run.return_value = (128, "", "fatal: not a git repository")
        with pytest.raises(RuntimeError, match="Not inside a git repository"):
            get_git_status()


# =============================================================================
# COMMIT PLAN GENERATION TESTS
# =============================================================================

class TestGenerateCommitPlan:
    """Tests for generate_commit_plan function."""

    def test_clean_status_returns_empty_plan(self):
        status = GitStatus(is_clean=True, branch="main")
        plan = generate_commit_plan(status)
        assert plan.message == ""
        assert plan.confirm_level == ConfirmLevel.AUTO

    def test_dirty_status_generates_plan(self):
        status = GitStatus(
            is_clean=False,
            branch="main",
            unstaged=["kernel/git_ops.py"],
        )
        plan = generate_commit_plan(status, trigger_context="test trigger")
        assert plan.message != ""
        assert "kernel/git_ops.py" in plan.files_to_stage
        assert plan.confirm_level == ConfirmLevel.CONFIRM

    def test_auto_tag_with_metrics(self):
        status = GitStatus(
            is_clean=False,
            branch="main",
            unstaged=["experiments/t05/results.json"],
        )
        metrics = {"oos_sharpe": 1.67, "oos_is_ratio": 0.94}
        plan = generate_commit_plan(
            status,
            auto_tag=True,
            tag_prefix="exp",
            task_id="t05",
            experiment_metrics=metrics,
        )
        assert plan.tag is not None
        assert plan.tag.startswith("exp/t05/")
        assert "OOS Sharpe: 1.67" in plan.tag_message

    def test_docs_only_returns_auto_level(self):
        status = GitStatus(
            is_clean=False,
            branch="main",
            unstaged=["docs/README.md"],
        )
        plan = generate_commit_plan(status)
        assert plan.confirm_level == ConfirmLevel.AUTO


# =============================================================================
# PLAN FORMATTING TESTS
# =============================================================================

class TestFormatPlanForReview:
    """Tests for format_plan_for_review function."""

    def test_basic_format(self):
        status = GitStatus(is_clean=False, branch="main")
        plan = CommitPlan(
            message="feat(kernel): add git_ops",
            files_to_stage=["kernel/git_ops.py"],
            confirm_level=ConfirmLevel.CONFIRM,
            context={"classified": {"kernel": ["kernel/git_ops.py"]}},
        )
        output = format_plan_for_review(plan, status)
        assert "## 📦 Git Commit Plan" in output
        assert "**Branch**: `main`" in output
        assert "feat(kernel): add git_ops" in output
        assert "**Proceed? [Y/n]**" in output

    def test_block_level_shows_commands(self):
        status = GitStatus(is_clean=False, branch="main")
        plan = CommitPlan(
            message="data: update raw files",
            files_to_stage=["data/file.csv"],
            confirm_level=ConfirmLevel.BLOCK,
            context={"classified": {"data": ["data/file.csv"]}},
        )
        output = format_plan_for_review(plan, status)
        assert "**⚠️ Manual execution required" in output
        assert "git add" in output
        assert "git commit" in output

    def test_with_tag(self):
        status = GitStatus(is_clean=False, branch="main")
        plan = CommitPlan(
            message="experiment: t05 complete",
            tag="exp/t05/v1",
            tag_message="OOS Sharpe: 1.67",
            confirm_level=ConfirmLevel.NOTIFY,
            context={"classified": {"experiments": ["t05/results.json"]}},
        )
        output = format_plan_for_review(plan, status)
        assert "### Tag: `exp/t05/v1`" in output
        assert "OOS Sharpe: 1.67" in output


# =============================================================================
# EXECUTION TESTS
# =============================================================================

class TestExecutePlan:
    """Tests for execute_plan function."""

    def test_empty_plan_succeeds(self):
        plan = CommitPlan(message="", confirm_level=ConfirmLevel.AUTO)
        result = execute_plan(plan, dry_run=True)
        assert result.success
        assert "No changes to commit" in result.actions_taken[0]

    @patch("kernel.git_ops._run_git")
    def test_dry_run_stages_files(self, mock_run):
        mock_run.return_value = (0, "", "")
        plan = CommitPlan(
            message="test commit",
            files_to_stage=["file.py"],
            confirm_level=ConfirmLevel.AUTO,
        )
        result = execute_plan(plan, dry_run=True)
        assert result.success
        assert result.dry_run
        assert any("[DRY-RUN] git add" in a for a in result.actions_taken)
        assert any("[DRY-RUN] git commit" in a for a in result.actions_taken)

    @patch("kernel.git_ops._run_git")
    def test_actual_commit(self, mock_run):
        def side_effect(args, cwd=None):
            if args[0] == "add":
                return (0, "", "")
            if args[0] == "commit":
                return (0, "[main abc1234] test commit\n", "")
            return (0, "", "")

        mock_run.side_effect = side_effect
        plan = CommitPlan(
            message="test commit",
            files_to_stage=["file.py"],
            confirm_level=ConfirmLevel.AUTO,
        )
        result = execute_plan(plan, dry_run=False)
        assert result.success
        assert not result.dry_run
        assert result.commit_sha == "abc1234"

    @patch("kernel.git_ops._run_git")
    def test_tag_creation(self, mock_run):
        mock_run.return_value = (0, "[main abc1234] test\n", "")
        plan = CommitPlan(
            message="test commit",
            tag="v1.0.0",
            tag_message="Release v1.0.0",
            confirm_level=ConfirmLevel.AUTO,
        )
        result = execute_plan(plan, dry_run=True)
        assert result.success
        assert any("[DRY-RUN] git tag" in a for a in result.actions_taken)

    @patch("kernel.git_ops._run_git")
    def test_commit_failure(self, mock_run):
        def side_effect(args, cwd=None):
            if args[0] == "add":
                return (0, "", "")
            if args[0] == "commit":
                return (1, "", "error: nothing to commit")
            return (0, "", "")

        mock_run.side_effect = side_effect
        plan = CommitPlan(
            message="test commit",
            files_to_stage=["file.py"],
            confirm_level=ConfirmLevel.AUTO,
        )
        result = execute_plan(plan, dry_run=False)
        assert not result.success
        assert len(result.errors) > 0


# =============================================================================
# WORKFLOW INTEGRATION TESTS
# =============================================================================

class TestRunGitOpsWorkflow:
    """Tests for run_git_ops_workflow function."""

    @patch("kernel.git_ops.get_git_status")
    def test_clean_repo_workflow(self, mock_status):
        mock_status.return_value = GitStatus(is_clean=True, branch="main")
        plan, result, output = run_git_ops_workflow(
            trigger_context="test",
            dry_run=True,
        )
        assert plan.message == ""
        assert result.success

    @patch("kernel.git_ops.get_git_status")
    @patch("kernel.git_ops.execute_plan")
    def test_dirty_repo_workflow(self, mock_execute, mock_status):
        mock_status.return_value = GitStatus(
            is_clean=False,
            branch="feature/test",
            unstaged=["kernel/git_ops.py"],
        )
        mock_execute.return_value = ExecutionResult(
            success=True,
            dry_run=True,
            actions_taken=["[DRY-RUN] git add kernel/git_ops.py"],
        )
        plan, result, output = run_git_ops_workflow(
            trigger_context="dgsf_execute complete",
            task_id="TASK-001",
            dry_run=True,
        )
        assert plan.message != ""
        assert "📦 Git Commit Plan" in output
        assert result.success

    @patch("kernel.git_ops.get_git_status")
    def test_block_level_skips_execution(self, mock_status):
        mock_status.return_value = GitStatus(
            is_clean=False,
            branch="main",
            unstaged=["data/raw/file.csv"],
        )
        plan, result, output = run_git_ops_workflow(dry_run=False)
        # BLOCK level should not execute even with dry_run=False
        assert plan.confirm_level == ConfirmLevel.BLOCK
        assert "Manual execution required" in result.actions_taken[0]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_rename_file_parsing(self):
        """Test that renamed files are handled correctly."""
        # Git status shows renames as "R  old -> new"
        status = GitStatus(
            is_clean=False,
            branch="main",
            staged=["new_name.py"],  # After parsing " -> "
        )
        plan = generate_commit_plan(status)
        assert "new_name.py" in plan.files_to_stage or plan.message != ""

    def test_very_long_file_list(self):
        """Test commit message truncation for many files."""
        files = [f"kernel/file_{i}.py" for i in range(20)]
        classified = classify_changes(files)
        msg = generate_commit_message(classified)
        # Should truncate to 5 files per category
        assert "... and" in msg

    def test_empty_branch_name(self):
        """Test handling of detached HEAD state."""
        status = GitStatus(is_clean=True, branch="HEAD")
        plan = generate_commit_plan(status)
        # Should not crash
        assert plan is not None

    def test_unicode_in_files(self):
        """Test handling of unicode in file paths."""
        classified = {"docs": ["docs/中文文档.md"]}
        msg = generate_commit_message(classified)
        assert "中文文档.md" in msg
