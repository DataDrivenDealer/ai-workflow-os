from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def init_state(root: Path) -> None:
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    project_path = state_dir / "project.yaml"
    tasks_path = state_dir / "tasks.yaml"

    if not project_path.exists():
        project_data = {
            "version": "0.1",
            "initialized_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        write_yaml(project_path, project_data)

    if not tasks_path.exists():
        tasks_data = {
            "version": "0.1",
            "queues": {},
            "tasks": {},
        }
        write_yaml(tasks_path, tasks_data)


def get_task(tasks_state: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    return tasks_state.setdefault("tasks", {}).get(task_id, {})


def upsert_task(tasks_state: Dict[str, Any], task_id: str, fields: Dict[str, Any]) -> None:
    tasks = tasks_state.setdefault("tasks", {})
    task = tasks.get(task_id, {})
    task.update(fields)
    task["last_updated"] = datetime.now(timezone.utc).isoformat()
    tasks[task_id] = task


def append_event(tasks_state: Dict[str, Any], task_id: str, event: Dict[str, Any]) -> None:
    tasks = tasks_state.setdefault("tasks", {})
    task = tasks.get(task_id, {})
    events = task.get("events", [])
    events.append(event)
    task["events"] = events
    tasks[task_id] = task
