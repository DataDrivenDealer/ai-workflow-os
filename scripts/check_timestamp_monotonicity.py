#!/usr/bin/env python3
"""
Verify event timestamp monotonicity in task state files.

Checks that event timestamps within each task are non-decreasing
(later events have timestamps >= earlier events).

Exit codes:
  0 - all timestamps monotonic
  1 - violations found
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kernel.paths import STATE_DIR

TASKS_STATE_PATH = STATE_DIR / "tasks.yaml"


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO 8601 timestamp with timezone support."""
    # Handle both 'Z' suffix and explicit timezone offsets
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    
    # Parse timestamp
    dt = datetime.fromisoformat(ts_str)
    
    # If naive (no timezone), assume UTC for comparison purposes
    if dt.tzinfo is None:
        from datetime import timezone as tz
        dt = dt.replace(tzinfo=tz.utc)
    
    return dt


def _load_tasks() -> Dict[str, Dict]:
    """Load all tasks from state/tasks.yaml."""
    if not TASKS_STATE_PATH.exists():
        return {}
    data = yaml.safe_load(TASKS_STATE_PATH.read_text(encoding="utf-8")) or {}
    return data.get("tasks", {})


def main() -> int:
    """
    Verify timestamp monotonicity for all tasks.
    
    Returns:
        0 if all timestamps are monotonic
        1 if violations are found
    """
    tasks = _load_tasks()
    violations: List[Dict] = []
    
    for task_id, task_data in tasks.items():
        events = task_data.get("events", []) or []
        
        # Check consecutive event pairs
        for i in range(len(events) - 1):
            ts1_str = events[i].get("timestamp", "")
            ts2_str = events[i+1].get("timestamp", "")
            
            if not ts1_str or not ts2_str:
                continue
            
            try:
                ts1 = _parse_timestamp(ts1_str)
                ts2 = _parse_timestamp(ts2_str)
                
                # Check monotonicity (ts2 should be >= ts1)
                if ts2 < ts1:
                    delta_seconds = (ts1 - ts2).total_seconds()
                    violations.append({
                        "task_id": task_id,
                        "event1_index": i,
                        "event2_index": i + 1,
                        "ts1": ts1_str,
                        "ts2": ts2_str,
                        "delta_seconds": delta_seconds,
                        "status1": events[i].get("status", "unknown"),
                        "status2": events[i+1].get("status", "unknown"),
                    })
            except (ValueError, TypeError) as e:
                # Invalid timestamp format - report as violation
                violations.append({
                    "task_id": task_id,
                    "event1_index": i,
                    "event2_index": i + 1,
                    "ts1": ts1_str,
                    "ts2": ts2_str,
                    "error": f"Timestamp parse error: {e}",
                })
    
    # Report results
    if violations:
        print(f"❌ Found {len(violations)} timestamp monotonicity violations:")
        for v in violations:
            task_id = v["task_id"]
            idx1 = v["event1_index"]
            idx2 = v["event2_index"]
            
            if "error" in v:
                print(f"  [{task_id}] Event {idx1} → {idx2}: {v['error']}")
            else:
                status1 = v.get("status1", "?")
                status2 = v.get("status2", "?")
                delta = v["delta_seconds"]
                print(f"  [{task_id}] Event {idx1} ({status1}) @ {v['ts1']}")
                print(f"              > Event {idx2} ({status2}) @ {v['ts2']}")
                print(f"              Δ = -{delta:.2f} seconds (backward!)")
        return 1
    
    print("✅ All event timestamps are monotonic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
