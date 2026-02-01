"""
Unit tests for kernel/state_store.py

Author: 李质量 (QA Test Engineer)
Date: 2026-02-01
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from state_store import (
    read_yaml,
    write_yaml,
    init_state,
    get_task,
    upsert_task,
    append_event,
)


class TestReadWriteYaml:
    """Tests for read_yaml and write_yaml functions."""

    def test_read_yaml_nonexistent_file(self, tmp_path: Path):
        """Reading a non-existent file should return empty dict."""
        result = read_yaml(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_write_and_read_yaml(self, tmp_path: Path):
        """Write and read should roundtrip correctly."""
        test_file = tmp_path / "test.yaml"
        data = {"key": "value", "nested": {"a": 1, "b": 2}}
        
        write_yaml(test_file, data)
        result = read_yaml(test_file)
        
        assert result == data

    def test_write_yaml_creates_parent_dirs(self, tmp_path: Path):
        """write_yaml should create parent directories."""
        nested_path = tmp_path / "a" / "b" / "c" / "test.yaml"
        data = {"test": True}
        
        write_yaml(nested_path, data)
        
        assert nested_path.exists()
        assert read_yaml(nested_path) == data

    def test_read_yaml_empty_file(self, tmp_path: Path):
        """Reading an empty YAML file should return empty dict."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        
        result = read_yaml(empty_file)
        assert result == {}

    def test_write_yaml_unicode(self, tmp_path: Path):
        """write_yaml should handle unicode correctly."""
        test_file = tmp_path / "unicode.yaml"
        data = {"中文": "测试", "日本語": "テスト"}
        
        write_yaml(test_file, data)
        result = read_yaml(test_file)
        
        assert result == data


class TestInitState:
    """Tests for init_state function."""

    def test_init_state_creates_directories(self, tmp_path: Path):
        """init_state should create state directory structure."""
        init_state(tmp_path)
        
        state_dir = tmp_path / "state"
        assert state_dir.exists()
        assert (state_dir / "project.yaml").exists()
        assert (state_dir / "tasks.yaml").exists()

    def test_init_state_project_yaml_content(self, tmp_path: Path):
        """init_state should create valid project.yaml."""
        init_state(tmp_path)
        
        project_data = read_yaml(tmp_path / "state" / "project.yaml")
        assert project_data["version"] == "0.1"
        assert "initialized_at" in project_data

    def test_init_state_tasks_yaml_content(self, tmp_path: Path):
        """init_state should create valid tasks.yaml."""
        init_state(tmp_path)
        
        tasks_data = read_yaml(tmp_path / "state" / "tasks.yaml")
        assert tasks_data["version"] == "0.1"
        assert tasks_data["queues"] == {}
        assert tasks_data["tasks"] == {}

    def test_init_state_idempotent(self, tmp_path: Path):
        """init_state should be idempotent - not overwrite existing data."""
        init_state(tmp_path)
        
        # Modify tasks.yaml
        tasks_path = tmp_path / "state" / "tasks.yaml"
        tasks_data = read_yaml(tasks_path)
        tasks_data["tasks"]["TASK_001"] = {"status": "running"}
        write_yaml(tasks_path, tasks_data)
        
        # Run init_state again
        init_state(tmp_path)
        
        # Should preserve existing task
        tasks_after = read_yaml(tasks_path)
        assert "TASK_001" in tasks_after["tasks"]


class TestTaskOperations:
    """Tests for get_task, upsert_task, append_event functions."""

    def test_get_task_nonexistent(self):
        """get_task should return empty dict for nonexistent task."""
        tasks_state = {"tasks": {}}
        result = get_task(tasks_state, "TASK_MISSING")
        assert result == {}

    def test_get_task_existing(self):
        """get_task should return task data for existing task."""
        tasks_state = {
            "tasks": {
                "TASK_001": {"status": "running", "queue": "dev"}
            }
        }
        result = get_task(tasks_state, "TASK_001")
        assert result == {"status": "running", "queue": "dev"}

    def test_upsert_task_new(self):
        """upsert_task should create new task."""
        tasks_state = {"tasks": {}}
        upsert_task(tasks_state, "TASK_NEW", {"status": "draft", "queue": "dev"})
        
        assert "TASK_NEW" in tasks_state["tasks"]
        assert tasks_state["tasks"]["TASK_NEW"]["status"] == "draft"
        assert "last_updated" in tasks_state["tasks"]["TASK_NEW"]

    def test_upsert_task_update_existing(self):
        """upsert_task should update existing task."""
        tasks_state = {
            "tasks": {
                "TASK_001": {"status": "draft", "queue": "dev"}
            }
        }
        upsert_task(tasks_state, "TASK_001", {"status": "running"})
        
        # Status updated, queue preserved
        assert tasks_state["tasks"]["TASK_001"]["status"] == "running"
        assert tasks_state["tasks"]["TASK_001"]["queue"] == "dev"

    def test_append_event(self):
        """append_event should add event to task events list."""
        tasks_state = {
            "tasks": {
                "TASK_001": {"status": "running", "events": []}
            }
        }
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "task_start",
            "from": "draft",
            "to": "running",
        }
        append_event(tasks_state, "TASK_001", event)
        
        assert len(tasks_state["tasks"]["TASK_001"]["events"]) == 1
        assert tasks_state["tasks"]["TASK_001"]["events"][0]["action"] == "task_start"

    def test_append_event_creates_events_list(self):
        """append_event should create events list if not exists."""
        tasks_state = {
            "tasks": {
                "TASK_001": {"status": "running"}
            }
        }
        event = {"action": "test"}
        append_event(tasks_state, "TASK_001", event)
        
        assert "events" in tasks_state["tasks"]["TASK_001"]
        assert len(tasks_state["tasks"]["TASK_001"]["events"]) == 1
