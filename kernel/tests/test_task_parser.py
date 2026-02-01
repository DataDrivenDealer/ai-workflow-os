"""
Unit tests for kernel/task_parser.py

Author: 李质量 (QA Test Engineer)
Date: 2026-02-01
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_parser import (
    parse_taskcard,
    validate_taskcard,
    REQUIRED_FIELDS,
    PRIORITY_LEVELS,
    DEFAULT_PRIORITY,
    get_priority,
    get_priority_order,
)


class TestParseTaskcard:
    """Tests for parse_taskcard function."""

    def test_parse_valid_taskcard(self, tmp_path: Path):
        """parse_taskcard should parse valid TaskCard correctly."""
        taskcard = tmp_path / "TASK_001.md"
        content = """---
task_id: TASK_001
type: dev
queue: dev
branch: feature/TASK_001
spec_ids:
  - ARCH_BLUEPRINT_MASTER
verification:
  - test passes
---

# Task Body

This is the task description.
"""
        taskcard.write_text(content, encoding="utf-8")
        
        result = parse_taskcard(taskcard)
        
        assert result["frontmatter"]["task_id"] == "TASK_001"
        assert result["frontmatter"]["type"] == "dev"
        assert result["frontmatter"]["queue"] == "dev"
        assert "ARCH_BLUEPRINT_MASTER" in result["frontmatter"]["spec_ids"]
        assert "# Task Body" in result["body"]

    def test_parse_taskcard_missing_frontmatter(self, tmp_path: Path):
        """parse_taskcard should raise error for missing frontmatter."""
        taskcard = tmp_path / "TASK_BAD.md"
        content = """# No Frontmatter

This TaskCard has no YAML frontmatter.
"""
        taskcard.write_text(content, encoding="utf-8")
        
        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            parse_taskcard(taskcard)

    def test_parse_taskcard_unclosed_frontmatter(self, tmp_path: Path):
        """parse_taskcard should raise error for unclosed frontmatter."""
        taskcard = tmp_path / "TASK_UNCLOSED.md"
        content = """---
task_id: TASK_001

Body without closing frontmatter.
"""
        taskcard.write_text(content, encoding="utf-8")
        
        with pytest.raises(ValueError, match="not closed"):
            parse_taskcard(taskcard)

    def test_parse_taskcard_empty_frontmatter(self, tmp_path: Path):
        """parse_taskcard should handle empty frontmatter."""
        taskcard = tmp_path / "TASK_EMPTY.md"
        content = """---
---

# Body only
"""
        taskcard.write_text(content, encoding="utf-8")
        
        result = parse_taskcard(taskcard)
        assert result["frontmatter"] == {}
        assert "# Body only" in result["body"]


class TestValidateTaskcard:
    """Tests for validate_taskcard function."""

    def test_validate_valid_taskcard(self):
        """validate_taskcard should pass for valid fields."""
        fields = {
            "task_id": "TASK_001",
            "type": "dev",
            "queue": "dev",
            "branch": "feature/TASK_001",
            "spec_ids": ["ARCH_BLUEPRINT_MASTER"],
            "verification": ["test passes"],
        }
        
        # Should not raise
        validate_taskcard(fields)

    def test_validate_missing_required_field(self):
        """validate_taskcard should raise for missing required fields."""
        fields = {
            "task_id": "TASK_001",
            "type": "dev",
            # Missing: queue, branch, spec_ids, verification
        }
        
        with pytest.raises(ValueError, match="missing required fields"):
            validate_taskcard(fields)

    def test_validate_spec_ids_not_list(self):
        """validate_taskcard should raise if spec_ids is not a list."""
        fields = {
            "task_id": "TASK_001",
            "type": "dev",
            "queue": "dev",
            "branch": "feature/TASK_001",
            "spec_ids": "ARCH_BLUEPRINT_MASTER",  # String instead of list
            "verification": ["test passes"],
        }
        
        with pytest.raises(ValueError, match="spec_ids must be a list"):
            validate_taskcard(fields)

    def test_validate_verification_not_list(self):
        """validate_taskcard should raise if verification is not a list."""
        fields = {
            "task_id": "TASK_001",
            "type": "dev",
            "queue": "dev",
            "branch": "feature/TASK_001",
            "spec_ids": ["ARCH_BLUEPRINT_MASTER"],
            "verification": "test passes",  # String instead of list
        }
        
        with pytest.raises(ValueError, match="verification must be a list"):
            validate_taskcard(fields)

    def test_required_fields_constant(self):
        """REQUIRED_FIELDS should contain expected fields."""
        assert "task_id" in REQUIRED_FIELDS
        assert "type" in REQUIRED_FIELDS
        assert "queue" in REQUIRED_FIELDS
        assert "branch" in REQUIRED_FIELDS
        assert "spec_ids" in REQUIRED_FIELDS
        assert "verification" in REQUIRED_FIELDS


class TestPriorityValidation:
    """Tests for priority validation and utility functions."""

    def test_priority_levels_constant(self):
        """PRIORITY_LEVELS should be P0-P3 in order."""
        assert PRIORITY_LEVELS == ["P0", "P1", "P2", "P3"]
        assert DEFAULT_PRIORITY == "P3"

    def test_validate_valid_priority(self):
        """validate_taskcard should accept valid priority values."""
        for priority in PRIORITY_LEVELS:
            fields = {
                "task_id": "TASK_001",
                "type": "dev",
                "queue": "dev",
                "branch": "feature/TASK_001",
                "spec_ids": ["ARCH_BLUEPRINT_MASTER"],
                "verification": ["test passes"],
                "priority": priority,
            }
            # Should not raise
            validate_taskcard(fields)

    def test_validate_invalid_priority(self):
        """validate_taskcard should reject invalid priority values."""
        fields = {
            "task_id": "TASK_001",
            "type": "dev",
            "queue": "dev",
            "branch": "feature/TASK_001",
            "spec_ids": ["ARCH_BLUEPRINT_MASTER"],
            "verification": ["test passes"],
            "priority": "P9",  # Invalid
        }
        
        with pytest.raises(ValueError, match="priority must be one of"):
            validate_taskcard(fields)

    def test_validate_taskcard_without_priority(self):
        """validate_taskcard should pass when priority is not specified."""
        fields = {
            "task_id": "TASK_001",
            "type": "dev",
            "queue": "dev",
            "branch": "feature/TASK_001",
            "spec_ids": ["ARCH_BLUEPRINT_MASTER"],
            "verification": ["test passes"],
            # No priority field
        }
        # Should not raise
        validate_taskcard(fields)

    def test_get_priority_with_value(self):
        """get_priority should return the specified priority."""
        fields = {"priority": "P1"}
        assert get_priority(fields) == "P1"

    def test_get_priority_default(self):
        """get_priority should return P3 when not specified."""
        fields = {}
        assert get_priority(fields) == "P3"

    def test_get_priority_order_valid(self):
        """get_priority_order should return correct order values."""
        assert get_priority_order("P0") == 0
        assert get_priority_order("P1") == 1
        assert get_priority_order("P2") == 2
        assert get_priority_order("P3") == 3

    def test_get_priority_order_invalid(self):
        """get_priority_order should return high value for invalid priority."""
        assert get_priority_order("invalid") == 4
        assert get_priority_order("") == 4

    def test_priority_sorting_order(self):
        """Priorities should sort P0 < P1 < P2 < P3."""
        priorities = ["P3", "P0", "P2", "P1"]
        sorted_priorities = sorted(priorities, key=get_priority_order)
        assert sorted_priorities == ["P0", "P1", "P2", "P3"]
