"""
Unit tests for kernel/os.py (renamed to avoid conflict with stdlib os module)

Author: 李质量 (QA Test Engineer)
Date: 2026-02-01
"""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

import sys
KERNEL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(KERNEL_DIR))

from state_store import read_yaml, write_yaml


def load_kernel_os():
    """Load kernel/os.py as a module without naming conflict."""
    spec = importlib.util.spec_from_file_location("kernel_os", KERNEL_DIR / "os.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStateMachine:
    """Tests for state machine loading and transitions."""

    def test_minimal_state_machine_structure(self):
        """MINIMAL_STATE_MACHINE should have required structure."""
        kernel_os = load_kernel_os()
        
        assert "states" in kernel_os.MINIMAL_STATE_MACHINE
        assert "transitions" in kernel_os.MINIMAL_STATE_MACHINE
        
        states = kernel_os.MINIMAL_STATE_MACHINE["states"]
        assert "draft" in states
        assert "ready" in states
        assert "running" in states
        assert "reviewing" in states
        assert "merged" in states
        assert "released" in states

    def test_can_transition_valid(self):
        """can_transition should return True for valid transitions."""
        kernel_os = load_kernel_os()
        sm = kernel_os.MINIMAL_STATE_MACHINE
        
        assert kernel_os.can_transition(sm, "draft", "ready") is True
        assert kernel_os.can_transition(sm, "draft", "running") is True
        assert kernel_os.can_transition(sm, "running", "reviewing") is True
        assert kernel_os.can_transition(sm, "reviewing", "merged") is True

    def test_can_transition_invalid(self):
        """can_transition should return False for invalid transitions."""
        kernel_os = load_kernel_os()
        sm = kernel_os.MINIMAL_STATE_MACHINE
        
        # Cannot skip states
        assert kernel_os.can_transition(sm, "draft", "merged") is False
        # Cannot go backwards
        assert kernel_os.can_transition(sm, "running", "draft") is False
        # Invalid states
        assert kernel_os.can_transition(sm, "nonexistent", "running") is False


class TestLoadStateMachine:
    """Tests for load_state_machine function."""

    def test_load_state_machine_creates_file(self, tmp_path: Path):
        """load_state_machine should create file if not exists."""
        kernel_os = load_kernel_os()
        
        # Create a temp state machine path
        temp_sm_path = tmp_path / "kernel" / "state_machine.yaml"
        original_path = kernel_os.STATE_MACHINE_PATH
        
        try:
            kernel_os.STATE_MACHINE_PATH = temp_sm_path
            result = kernel_os.load_state_machine()
            
            assert temp_sm_path.exists()
            assert "states" in result
            assert "transitions" in result
        finally:
            kernel_os.STATE_MACHINE_PATH = original_path


class TestLoadRegistrySpecIds:
    """Tests for load_registry_spec_ids function."""

    def test_load_registry_spec_ids(self, tmp_path: Path):
        """load_registry_spec_ids should extract spec_id list from registry."""
        kernel_os = load_kernel_os()
        
        registry_path = tmp_path / "spec_registry.yaml"
        registry_data = {
            "specs": [
                {"spec_id": "SPEC_001", "title": "Test Spec 1"},
                {"spec_id": "SPEC_002", "title": "Test Spec 2"},
            ]
        }
        write_yaml(registry_path, registry_data)
        
        original_path = kernel_os.REGISTRY_PATH
        try:
            kernel_os.REGISTRY_PATH = registry_path
            result = kernel_os.load_registry_spec_ids()
            
            assert "SPEC_001" in result
            assert "SPEC_002" in result
        finally:
            kernel_os.REGISTRY_PATH = original_path


class TestCmdInit:
    """Tests for cmd_init function."""

    def test_cmd_init_creates_directories(self, tmp_path: Path):
        """cmd_init should create required directories."""
        kernel_os = load_kernel_os()
        
        original_root = kernel_os.ROOT
        original_tasks = kernel_os.TASKS_DIR
        original_sm = kernel_os.STATE_MACHINE_PATH
        
        try:
            kernel_os.ROOT = tmp_path
            kernel_os.TASKS_DIR = tmp_path / "tasks"
            kernel_os.STATE_MACHINE_PATH = tmp_path / "kernel" / "state_machine.yaml"
            
            args = argparse.Namespace()
            kernel_os.cmd_init(args)
            
            assert (tmp_path / "tasks").exists()
            assert (tmp_path / "ops" / "audit").exists()
            assert (tmp_path / "ops" / "decision-log").exists()
            assert (tmp_path / "state").exists()
        finally:
            kernel_os.ROOT = original_root
            kernel_os.TASKS_DIR = original_tasks
            kernel_os.STATE_MACHINE_PATH = original_sm


class TestCmdTaskNew:
    """Tests for cmd_task_new function."""

    def test_cmd_task_new_creates_taskcard(self, tmp_path: Path):
        """cmd_task_new should create a new TaskCard file."""
        kernel_os = load_kernel_os()
        
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(parents=True)
        
        template_path = tmp_path / "templates" / "TASKCARD_TEMPLATE.md"
        template_path.parent.mkdir(parents=True)
        template_content = """---
task_id: {{TASK_ID}}
type: dev
queue: dev
branch: feature/{{TASK_ID}}
spec_ids: []
verification: []
---

# {{TASK_ID}}
"""
        template_path.write_text(template_content, encoding="utf-8")
        
        original_root = kernel_os.ROOT
        original_tasks = kernel_os.TASKS_DIR
        original_template = kernel_os.TEMPLATE_PATH
        
        try:
            kernel_os.ROOT = tmp_path
            kernel_os.TASKS_DIR = tasks_dir
            kernel_os.TEMPLATE_PATH = template_path
            
            args = argparse.Namespace(task_id="TASK_TEST_001")
            kernel_os.cmd_task_new(args)
            
            created_file = tasks_dir / "TASK_TEST_001.md"
            assert created_file.exists()
            
            content = created_file.read_text(encoding="utf-8")
            assert "TASK_TEST_001" in content
        finally:
            kernel_os.ROOT = original_root
            kernel_os.TASKS_DIR = original_tasks
            kernel_os.TEMPLATE_PATH = original_template

    def test_cmd_task_new_raises_if_exists(self, tmp_path: Path):
        """cmd_task_new should raise error if TaskCard already exists."""
        kernel_os = load_kernel_os()
        
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(parents=True)
        
        existing_task = tasks_dir / "TASK_EXISTING.md"
        existing_task.write_text("existing content", encoding="utf-8")
        
        original_tasks = kernel_os.TASKS_DIR
        
        try:
            kernel_os.TASKS_DIR = tasks_dir
            
            args = argparse.Namespace(task_id="TASK_EXISTING")
            
            with pytest.raises(RuntimeError, match="already exists"):
                kernel_os.cmd_task_new(args)
        finally:
            kernel_os.TASKS_DIR = original_tasks
