from __future__ import annotations

from datetime import datetime, timezone
import contextlib
import os
import tempfile
import time
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import yaml


# =============================================================================
# VERSION CONTROL FOR OPTIMISTIC CONCURRENCY
# =============================================================================

@dataclass
class VersionedData:
    """Data with version information for optimistic concurrency control."""
    data: Dict[str, Any]
    version: int
    checksum: str
    last_modified_by: Optional[str] = None
    last_modified_at: Optional[str] = None


def compute_checksum(data: Dict[str, Any]) -> str:
    """Compute SHA256 checksum of data for conflict detection."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def get_version(data: Dict[str, Any]) -> int:
    """Extract version from data, defaulting to 0."""
    return data.get("_version", 0)


def set_version(data: Dict[str, Any], version: int, agent_id: Optional[str] = None) -> None:
    """Set version metadata in data."""
    data["_version"] = version
    data["_checksum"] = compute_checksum({k: v for k, v in data.items() if not k.startswith("_")})
    data["_last_modified_at"] = datetime.now(timezone.utc).isoformat()
    if agent_id:
        data["_last_modified_by"] = agent_id


# =============================================================================
# CONFLICT DETECTION AND RESOLUTION
# =============================================================================

@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    path: Path
    expected_version: int
    actual_version: int
    expected_checksum: str
    actual_checksum: str
    conflicting_agent: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConflictError(Exception):
    """Raised when optimistic concurrency conflict is detected."""
    def __init__(self, conflict: ConflictInfo):
        self.conflict = conflict
        super().__init__(
            f"Conflict detected on {conflict.path}: "
            f"expected version {conflict.expected_version}, "
            f"found version {conflict.actual_version}"
        )


def detect_conflict(path: Path, expected_version: int, expected_checksum: str) -> Optional[ConflictInfo]:
    """Check if file has been modified since we read it."""
    if not path.exists():
        return None
    
    current_data = read_yaml(path)
    current_version = get_version(current_data)
    current_checksum = current_data.get("_checksum", "")
    
    if current_version != expected_version or current_checksum != expected_checksum:
        return ConflictInfo(
            path=path,
            expected_version=expected_version,
            actual_version=current_version,
            expected_checksum=expected_checksum,
            actual_checksum=current_checksum,
            conflicting_agent=current_data.get("_last_modified_by")
        )
    return None


def log_conflict(conflict: ConflictInfo, resolution: str) -> None:
    """Log conflict to audit trail."""
    # Import here to avoid circular dependency
    try:
        from kernel.audit import log_event
        log_event(
            event_type="concurrency_conflict",
            details={
                "path": str(conflict.path),
                "expected_version": conflict.expected_version,
                "actual_version": conflict.actual_version,
                "resolution": resolution,
                "timestamp": conflict.timestamp
            }
        )
    except ImportError:
        pass  # Audit not available


# =============================================================================
# CORE READ/WRITE FUNCTIONS
# =============================================================================

def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_yaml_versioned(path: Path) -> VersionedData:
    """Read YAML with version information for optimistic concurrency."""
    data = read_yaml(path)
    return VersionedData(
        data=data,
        version=get_version(data),
        checksum=data.get("_checksum", compute_checksum(data)),
        last_modified_by=data.get("_last_modified_by"),
        last_modified_at=data.get("_last_modified_at")
    )


def _acquire_lock(path: Path, timeout_seconds: float = 30.0, interval_seconds: float = 0.05) -> Path:
    """Acquire file lock with configurable timeout (default increased to 30s for multi-agent)."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            # Write lock metadata for debugging
            lock_info = {
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
            }
            os.write(fd, json.dumps(lock_info).encode())
            os.close(fd)
            return lock_path
        except FileExistsError:
            # Check for stale lock (older than 5 minutes)
            if _is_stale_lock(lock_path, stale_threshold_seconds=300):
                try:
                    lock_path.unlink(missing_ok=True)
                    continue
                except Exception:
                    pass
            
            if time.time() - start >= timeout_seconds:
                raise TimeoutError(f"Timeout acquiring lock for {path} after {timeout_seconds}s")
            time.sleep(interval_seconds)


def _is_stale_lock(lock_path: Path, stale_threshold_seconds: float = 300) -> bool:
    """Check if a lock file is stale (too old)."""
    try:
        stat = lock_path.stat()
        age = time.time() - stat.st_mtime
        return age > stale_threshold_seconds
    except Exception:
        return False


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


@contextlib.contextmanager
def atomic_update(path: Path, agent_id: Optional[str] = None, use_optimistic: bool = True):
    """
    Context manager for atomic read-modify-write operations with concurrency control.
    
    Supports both optimistic (version-based) and pessimistic (lock-based) concurrency.
    
    Args:
        path: Path to the YAML file
        agent_id: Optional agent identifier for audit trail
        use_optimistic: If True, use optimistic concurrency with conflict detection
    
    Usage:
        with atomic_update(path, agent_id="agent_001") as data:
            data['key'] = 'value'
    
    The data dictionary is read with lock held, and written back when exiting.
    Version is automatically incremented for conflict detection.
    """
    lock_path = _acquire_lock(path)
    try:
        # Read current data with lock held
        versioned = read_yaml_versioned(path)
        data = versioned.data
        initial_version = versioned.version
        initial_checksum = versioned.checksum
        
        yield data
        
        # Check for conflicts before writing (optimistic check even with lock)
        if use_optimistic:
            conflict = detect_conflict(path, initial_version, initial_checksum)
            if conflict:
                log_conflict(conflict, "detected_during_atomic_update")
                # With lock held, this shouldn't happen, but log it
        
        # Increment version and set metadata
        set_version(data, initial_version + 1, agent_id)
        
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


@contextlib.contextmanager
def optimistic_update(path: Path, agent_id: Optional[str] = None, max_retries: int = 3):
    """
    Context manager for optimistic concurrency control without holding lock during mutation.
    
    This allows higher concurrency but requires conflict detection and retry logic.
    
    Args:
        path: Path to the YAML file
        agent_id: Optional agent identifier
        max_retries: Maximum retry attempts on conflict
        
    Raises:
        ConflictError: If conflict persists after max_retries
    """
    for attempt in range(max_retries + 1):
        # Read without lock
        versioned = read_yaml_versioned(path)
        data = versioned.data.copy()  # Work on a copy
        initial_version = versioned.version
        initial_checksum = versioned.checksum
        
        yield data
        
        # Acquire lock only for write
        lock_path = _acquire_lock(path)
        try:
            # Check for conflict
            conflict = detect_conflict(path, initial_version, initial_checksum)
            if conflict:
                _release_lock(lock_path)
                if attempt < max_retries:
                    log_conflict(conflict, f"retry_attempt_{attempt + 1}")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    log_conflict(conflict, "max_retries_exceeded")
                    raise ConflictError(conflict)
            
            # No conflict, write data
            set_version(data, initial_version + 1, agent_id)
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
            return  # Success
        finally:
            _release_lock(lock_path)


def write_yaml(path: Path, data: Dict[str, Any], agent_id: Optional[str] = None) -> None:
    """Write YAML with version tracking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_lock(path)
    tmp_path = None
    try:
        # Read current version
        current_version = get_version(read_yaml(path)) if path.exists() else 0
        set_version(data, current_version + 1, agent_id)
        
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


def get_running_tasks_count(tasks_state: Dict[str, Any]) -> int:
    """
    Get the count of tasks currently in 'running' status.
    
    Args:
        tasks_state: The tasks state dictionary
        
    Returns:
        int: Number of tasks with status='running'
    """
    tasks = tasks_state.get("tasks", {})
    return sum(1 for task in tasks.values() if task.get("status") == "running")


def check_wip_limit(tasks_state: Dict[str, Any], limit: Optional[int] = None) -> None:
    """
    Check WIP (Work-In-Progress) limit and raise error if exceeded.
    
    This implements Kanban-style WIP limits to prevent multitasking overhead
    and improve flow efficiency (Gene Kim's Theory of Constraints).
    
    Args:
        tasks_state: The tasks state dictionary
        limit: Maximum number of running tasks (default: read from config)
        
    Raises:
        RuntimeError: If WIP limit would be exceeded
    """
    if limit is None:
        # Import here to avoid circular dependency
        from kernel.config import config
        limit = config.get_wip_limit()
    
    count = get_running_tasks_count(tasks_state)
    
    if count >= limit:
        # Get running task IDs for helpful error message
        running_tasks = [
            task_id for task_id, task in tasks_state.get("tasks", {}).items()
            if task.get("status") == "running"
        ]
        
        raise RuntimeError(
            f"WIP limit exceeded: {count}/{limit} tasks already running.\n"
            f"Currently running: {', '.join(running_tasks)}\n\n"
            f"To start a new task, first complete or pause one of the running tasks.\n"
            f"This limit prevents multitasking overhead and improves flow efficiency.\n"
            f"(Based on Gene Kim's Theory of Constraints and The Phoenix Project)"
        )
