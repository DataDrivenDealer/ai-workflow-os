"""
Tests for kernel.paths module.

Verifies that path constants are correctly defined and utility functions work.
"""

import pytest
from pathlib import Path
import sys

# Import paths module
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kernel.paths import (
    ROOT, KERNEL_DIR, STATE_DIR, TASKS_DIR, SPECS_DIR,
    CONFIGS_DIR, TEMPLATES_DIR, SCRIPTS_DIR, OPS_DIR, DOCS_DIR,
    STATE_MACHINE_PATH, REGISTRY_PATH, GATES_CONFIG_PATH,
    TASKS_STATE_PATH, AGENTS_STATE_PATH, SESSIONS_STATE_PATH,
    TASKCARD_TEMPLATE_PATH,
    ensure_dirs, get_task_path, get_ops_audit_path,
    TASKS_DONE_DIR, TASKS_INBOX_DIR, TASKS_RUNNING_DIR,
    OPS_AUDIT_DIR,
)


class TestPathConstants:
    """Test that all path constants are correctly defined."""
    
    def test_root_is_absolute(self):
        """ROOT should be an absolute path."""
        assert ROOT.is_absolute()
    
    def test_root_exists(self):
        """ROOT directory should exist."""
        assert ROOT.exists()
        assert ROOT.is_dir()
    
    def test_primary_directories_relative_to_root(self):
        """Primary directories should be children of ROOT."""
        primary_dirs = [
            KERNEL_DIR, STATE_DIR, TASKS_DIR, SPECS_DIR,
            CONFIGS_DIR, TEMPLATES_DIR, SCRIPTS_DIR, OPS_DIR, DOCS_DIR
        ]
        
        for dir_path in primary_dirs:
            # Should be child of ROOT
            assert ROOT in dir_path.parents
            # Should be absolute
            assert dir_path.is_absolute()
    
    def test_kernel_dir_contains_paths_module(self):
        """KERNEL_DIR should contain the paths.py module."""
        paths_file = KERNEL_DIR / "paths.py"
        assert paths_file.exists()
    
    def test_config_files_exist(self):
        """Critical config files should exist."""
        assert STATE_MACHINE_PATH.exists(), f"Missing: {STATE_MACHINE_PATH}"
        assert REGISTRY_PATH.exists(), f"Missing: {REGISTRY_PATH}"
        assert GATES_CONFIG_PATH.exists(), f"Missing: {GATES_CONFIG_PATH}"
    
    def test_state_dir_structure(self):
        """STATE_DIR should exist and contain expected files."""
        assert STATE_DIR.exists()
        assert STATE_DIR.is_dir()
        
        # State files should be defined
        assert TASKS_STATE_PATH.parent == STATE_DIR
        assert AGENTS_STATE_PATH.parent == STATE_DIR
        assert SESSIONS_STATE_PATH.parent == STATE_DIR
    
    def test_template_paths(self):
        """Template paths should be under TEMPLATES_DIR."""
        assert TASKCARD_TEMPLATE_PATH.parent == TEMPLATES_DIR


class TestEnsureDirs:
    """Test the ensure_dirs() utility function."""
    
    def test_ensure_dirs_creates_missing_directories(self, tmp_path):
        """ensure_dirs() should create missing directories."""
        # This test just verifies the function runs without error
        # Actual directory creation is tested in integration
        ensure_dirs()  # Should not raise
    
    def test_ensure_dirs_is_idempotent(self):
        """Calling ensure_dirs() multiple times should be safe."""
        ensure_dirs()
        ensure_dirs()  # Second call should not raise


class TestGetTaskPath:
    """Test the get_task_path() utility function."""
    
    def test_get_task_path_no_status(self):
        """get_task_path() without status returns tasks/ path."""
        path = get_task_path("TEST_TASK_001")
        assert path == TASKS_DIR / "TEST_TASK_001.md"
    
    def test_get_task_path_done(self):
        """get_task_path() with 'done' status returns tasks/done/ path."""
        path = get_task_path("TEST_TASK_001", "done")
        assert path == TASKS_DONE_DIR / "TEST_TASK_001.md"
    
    def test_get_task_path_inbox(self):
        """get_task_path() with 'inbox' status returns tasks/inbox/ path."""
        path = get_task_path("TEST_TASK_001", "inbox")
        assert path == TASKS_INBOX_DIR / "TEST_TASK_001.md"
    
    def test_get_task_path_running(self):
        """get_task_path() with 'running' status returns tasks/running/ path."""
        path = get_task_path("TEST_TASK_001", "running")
        assert path == TASKS_RUNNING_DIR / "TEST_TASK_001.md"


class TestGetOpsAuditPath:
    """Test the get_ops_audit_path() utility function."""
    
    def test_get_ops_audit_path(self):
        """get_ops_audit_path() returns correct path."""
        path = get_ops_audit_path("EXEC_20260201_TEST")
        assert path == OPS_AUDIT_DIR / "EXEC_20260201_TEST.md"
    
    def test_get_ops_audit_path_with_special_chars(self):
        """get_ops_audit_path() handles special characters."""
        path = get_ops_audit_path("EXEC_20260201_DGSF_INIT")
        assert path.name == "EXEC_20260201_DGSF_INIT.md"


class TestPathIntegration:
    """Integration tests for paths module."""
    
    def test_can_navigate_from_root_to_kernel(self):
        """Should be able to navigate from ROOT to KERNEL_DIR."""
        kernel_from_root = ROOT / "kernel"
        assert kernel_from_root == KERNEL_DIR
    
    def test_can_navigate_to_state_files(self):
        """Should be able to construct paths to state files."""
        tasks_yaml = STATE_DIR / "tasks.yaml"
        assert tasks_yaml == TASKS_STATE_PATH
    
    def test_paths_are_reusable(self):
        """Paths can be combined with other Path operations."""
        # Should be able to combine paths
        test_task = TASKS_DIR / "TEST_001.md"
        assert test_task.parent == TASKS_DIR
        
        # Should be able to use path methods
        assert KERNEL_DIR.name == "kernel"
        assert STATE_DIR.name == "state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
