#!/usr/bin/env python3
"""
Check WIP Limit (SYSTEM_INVARIANTS INV-2)

Verifies that the number of running tasks does not exceed the configured limit.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.config import config
from kernel.state_store import read_yaml
from kernel.paths import TASKS_STATE_PATH


def get_running_tasks_count(tasks_state: dict) -> int:
    """Count tasks with status 'running'."""
    count = 0
    for task in tasks_state.get("tasks", {}).values():
        if task.get("status") == "running":
            count += 1
    return count


def check_wip_limit() -> bool:
    """Check if WIP limit is violated."""
    # Access config attributes directly (not as dict)
    wip_limits = getattr(config, 'wip_limits', {})
    if isinstance(wip_limits, dict):
        max_running = wip_limits.get("max_running_tasks", 3)
    else:
        max_running = getattr(wip_limits, 'max_running_tasks', 3) if wip_limits else 3
    
    tasks_state = read_yaml(TASKS_STATE_PATH)
    running_count = get_running_tasks_count(tasks_state)
    
    print(f"WIP Limit Check")
    print(f"===============")
    print(f"Max allowed: {max_running}")
    print(f"Currently running: {running_count}")
    
    if running_count > max_running:
        print(f"❌ VIOLATION: {running_count} > {max_running}")
        return False
    else:
        print(f"✅ PASS: {running_count} <= {max_running}")
        return True


if __name__ == "__main__":
    passed = check_wip_limit()
    sys.exit(0 if passed else 1)
