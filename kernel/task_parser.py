from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

REQUIRED_FIELDS = [
    "task_id",
    "type",
    "queue",
    "branch",
    "spec_ids",
    "verification",
]

# Priority levels from highest (P0) to lowest (P3)
PRIORITY_LEVELS = ["P0", "P1", "P2", "P3"]
DEFAULT_PRIORITY = "P3"


def parse_taskcard(path: Path) -> Dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        raise ValueError("TaskCard missing YAML frontmatter")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("TaskCard frontmatter is not closed with '---'")

    frontmatter_raw = parts[1]
    body = parts[2].lstrip("\n")
    frontmatter = yaml.safe_load(frontmatter_raw) or {}
    return {"frontmatter": frontmatter, "body": body}


def validate_taskcard(fields: Dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"TaskCard missing required fields: {', '.join(missing)}")

    if not isinstance(fields.get("spec_ids"), list):
        raise ValueError("TaskCard spec_ids must be a list")

    if not isinstance(fields.get("verification"), list):
        raise ValueError("TaskCard verification must be a list")

    # Validate priority if provided
    priority = fields.get("priority")
    if priority is not None and priority not in PRIORITY_LEVELS:
        raise ValueError(
            f"TaskCard priority must be one of {PRIORITY_LEVELS}, got: {priority}"
        )


def get_priority(fields: Dict[str, Any]) -> str:
    """Get task priority, defaulting to P3 if not specified."""
    return fields.get("priority", DEFAULT_PRIORITY)


def get_priority_order(priority: str) -> int:
    """Convert priority to sortable integer (lower = higher priority)."""
    try:
        return PRIORITY_LEVELS.index(priority)
    except ValueError:
        return len(PRIORITY_LEVELS)  # Unknown priorities sort last
