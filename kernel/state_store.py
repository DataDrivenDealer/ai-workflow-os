from __future__ import annotations

from datetime import datetime, timezone
import contextlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict

import yaml


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _acquire_lock(path: Path, timeout_seconds: float = 2.0, interval_seconds: float = 0.05) -> Path:
    lock_path = path.with_suffix(path.suffix + ".lock")
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            return lock_path
        except FileExistsError:
            if time.time() - start >= timeout_seconds:
                raise TimeoutError(f"Timeout acquiring lock for {path}")
            time.sleep(interval_seconds)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


@contextlib.contextmanager
def atomic_update(path: Path):
    """
    Context manager for atomic read-modify-write operations.
    
    Usage:
        with atomic_update(path) as data:
            data['key'] = 'value'
    
    The data dictionary is read with lock held, and written back when exiting.
    """
    lock_path = _acquire_lock(path)
    try:
        # Read current data with lock held
        data = read_yaml(path)
        yield data
        # Write back modified data
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
    finally:
        _release_lock(lock_path)


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_lock(path)
    tmp_path = None
    try:
        serialized = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        os.replace(tmp_path, path)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        _release_lock(lock_path)


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
