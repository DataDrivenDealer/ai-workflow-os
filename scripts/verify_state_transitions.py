#!/usr/bin/env python3
"""
Verify task state transitions against kernel/state_machine.yaml.

Exit codes:
  0 - all transitions valid
  1 - violations found
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kernel.paths import STATE_DIR

STATE_MACHINE_PATH = Path(__file__).resolve().parents[1] / "kernel" / "state_machine.yaml"
TASKS_STATE_PATH = STATE_DIR / "tasks.yaml"


def _load_state_machine() -> Set[Tuple[str, str]]:
    data = yaml.safe_load(STATE_MACHINE_PATH.read_text(encoding="utf-8")) or {}
    transitions = data.get("transitions", [])
    allowed: Set[Tuple[str, str]] = set()
    for item in transitions:
        src = item.get("from")
        dst = item.get("to")
        if src and dst:
            allowed.add((src, dst))
    return allowed


def _load_tasks() -> Dict[str, Dict]:
    if not TASKS_STATE_PATH.exists():
        return {}
    data = yaml.safe_load(TASKS_STATE_PATH.read_text(encoding="utf-8")) or {}
    return data.get("tasks", {})


def main() -> int:
    allowed = _load_state_machine()
    tasks = _load_tasks()

    violations: List[Dict[str, str]] = []
    for task_id, task_data in tasks.items():
        events = task_data.get("events", []) or []
        for event in events:
            src = event.get("from")
            dst = event.get("to")
            if not src or not dst:
                continue
            if (src, dst) not in allowed:
                violations.append({
                    "task_id": task_id,
                    "from": src,
                    "to": dst,
                    "timestamp": event.get("timestamp", ""),
                })

    if violations:
        print(f"❌ Found {len(violations)} invalid state transitions")
        for v in violations:
            print(f"  {v['task_id']}: {v['from']} → {v['to']} @ {v['timestamp']}")
        return 1

    print("✅ All task state transitions are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
