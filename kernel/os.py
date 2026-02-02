from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from audit import write_audit
from paths import (
    ROOT, STATE_MACHINE_PATH, REGISTRY_PATH,
    TASKCARD_TEMPLATE_PATH as TEMPLATE_PATH, TASKS_DIR
)
from state_store import append_event, get_task, init_state, read_yaml, upsert_task, write_yaml
from task_parser import parse_taskcard, validate_taskcard, get_priority, get_priority_order, PRIORITY_LEVELS


MINIMAL_STATE_MACHINE = {
    "states": [
        "draft",
        "ready",
        "running",
        "reviewing",
        "merged",
        "released",
        "blocked",
        "abandoned",
    ],
    "transitions": [
        {"from": "draft", "to": "ready"},
        {"from": "draft", "to": "running"},
        {"from": "ready", "to": "running"},
        {"from": "running", "to": "reviewing"},
        {"from": "reviewing", "to": "merged"},
        {"from": "merged", "to": "released"},
    ],
}


def load_state_machine() -> Dict[str, Any]:
    if not STATE_MACHINE_PATH.exists():
        STATE_MACHINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATE_MACHINE_PATH.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(MINIMAL_STATE_MACHINE, handle, sort_keys=False)
    return read_yaml(STATE_MACHINE_PATH)


def can_transition(state_machine: Dict[str, Any], current: str, target: str) -> bool:
    transitions = state_machine.get("transitions", [])
    return any(t.get("from") == current and t.get("to") == target for t in transitions)


def load_registry_spec_ids() -> List[str]:
    registry = read_yaml(REGISTRY_PATH)
    specs = registry.get("specs", [])
    return [spec.get("spec_id") for spec in specs if "spec_id" in spec]


def ensure_git_repo() -> None:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise RuntimeError("Not inside a git work tree")


def ensure_branch(branch_name: str) -> None:
    exists = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        check=False,
    )
    if exists.returncode == 0:
        print(f"Branch already exists: {branch_name}")
        return
    subprocess.run(["git", "branch", branch_name], check=True)
    print(f"Created branch: {branch_name}")


def cmd_init(_: argparse.Namespace) -> None:
    init_state(ROOT)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "ops" / "audit").mkdir(parents=True, exist_ok=True)
    (ROOT / "ops" / "decision-log").mkdir(parents=True, exist_ok=True)
    load_state_machine()
    print("Initialized state and ops directories.")


def cmd_task_new(args: argparse.Namespace) -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    target = TASKS_DIR / f"{args.task_id}.md"
    if target.exists():
        raise RuntimeError(f"TaskCard already exists: {target}")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    content = template.replace("{{TASK_ID}}", args.task_id)
    target.write_text(content, encoding="utf-8")
    print(f"Created TaskCard: {target}")


def cmd_task_start(args: argparse.Namespace) -> None:
    ensure_git_repo()

    task_path = TASKS_DIR / f"{args.task_id}.md"
    if not task_path.exists():
        raise RuntimeError(f"TaskCard not found: {task_path}")

    parsed = parse_taskcard(task_path)
    fields = parsed["frontmatter"]
    validate_taskcard(fields)

    registry_spec_ids = set(load_registry_spec_ids())
    missing_specs = [spec for spec in fields["spec_ids"] if spec not in registry_spec_ids]
    if missing_specs:
        raise RuntimeError(f"Spec IDs not found in registry: {', '.join(missing_specs)}")

    state_machine = load_state_machine()
    tasks_state_path = ROOT / "state" / "tasks.yaml"
    tasks_state = read_yaml(tasks_state_path)

    current_task = get_task(tasks_state, args.task_id)
    current_status = current_task.get("status", "draft")

    if not can_transition(state_machine, current_status, "running"):
        raise RuntimeError(f"Invalid transition {current_status} -> running")

    queue = fields["queue"]
    queues = tasks_state.setdefault("queues", {})
    locked_by = queues.get(queue)
    if locked_by and locked_by != args.task_id:
        raise RuntimeError(f"Queue '{queue}' is locked by task {locked_by}")

    branch_name = fields["branch"]
    ensure_branch(branch_name)

    # Get priority from TaskCard (defaults to P3)
    priority = get_priority(fields)

    queues[queue] = args.task_id
    upsert_task(
        tasks_state,
        args.task_id,
        {
            "status": "running",
            "queue": queue,
            "branch": branch_name,
            "priority": priority,
        },
    )
    append_event(
        tasks_state,
        args.task_id,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "task_start",
            "from": current_status,
            "to": "running",
        },
    )
    write_yaml(tasks_state_path, tasks_state)
    print(f"Task {args.task_id} started.")


def cmd_task_finish(args: argparse.Namespace) -> None:
    task_path = TASKS_DIR / f"{args.task_id}.md"
    if not task_path.exists():
        raise RuntimeError(f"TaskCard not found: {task_path}")

    parsed = parse_taskcard(task_path)
    fields = parsed["frontmatter"]
    validate_taskcard(fields)

    state_machine = load_state_machine()
    tasks_state_path = ROOT / "state" / "tasks.yaml"
    tasks_state = read_yaml(tasks_state_path)

    current_task = get_task(tasks_state, args.task_id)
    current_status = current_task.get("status", "draft")

    if current_status != "running":
        raise RuntimeError(f"Task {args.task_id} is not running (current: {current_status})")

    if not can_transition(state_machine, current_status, "reviewing"):
        raise RuntimeError(f"Invalid transition {current_status} -> reviewing")

    queue = current_task.get("queue")
    queues = tasks_state.setdefault("queues", {})
    if queue and queues.get(queue) == args.task_id:
        queues.pop(queue, None)

    upsert_task(
        tasks_state,
        args.task_id,
        {
            "status": "reviewing",
        },
    )
    append_event(
        tasks_state,
        args.task_id,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "task_finish",
            "from": current_status,
            "to": "reviewing",
        },
    )
    write_yaml(tasks_state_path, tasks_state)

    audit_path = write_audit(
        ROOT,
        args.task_id,
        "reviewing",
        {
            "queue": fields["queue"],
            "branch": fields["branch"],
            "verification": ", ".join(fields.get("verification", [])),
        },
    )
    print(f"Task {args.task_id} moved to reviewing. Audit: {audit_path}")


def cmd_task_merge(args: argparse.Namespace) -> None:
    """Merge a task from reviewing state."""
    task_path = TASKS_DIR / f"{args.task_id}.md"
    if not task_path.exists():
        raise RuntimeError(f"TaskCard not found: {task_path}")

    parsed = parse_taskcard(task_path)
    fields = parsed["frontmatter"]
    validate_taskcard(fields)

    state_machine = load_state_machine()
    tasks_state_path = ROOT / "state" / "tasks.yaml"
    tasks_state = read_yaml(tasks_state_path)

    current_task = get_task(tasks_state, args.task_id)
    current_status = current_task.get("status", "draft")

    if current_status != "reviewing":
        raise RuntimeError(f"Task {args.task_id} is not in reviewing state (current: {current_status})")

    if not can_transition(state_machine, current_status, "merged"):
        raise RuntimeError(f"Invalid transition {current_status} -> merged")

    upsert_task(
        tasks_state,
        args.task_id,
        {
            "status": "merged",
        },
    )
    append_event(
        tasks_state,
        args.task_id,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "task_merge",
            "from": current_status,
            "to": "merged",
        },
    )
    write_yaml(tasks_state_path, tasks_state)

    audit_path = write_audit(
        ROOT,
        args.task_id,
        "merged",
        {
            "queue": fields["queue"],
            "branch": fields["branch"],
            "merged_by": args.merged_by if hasattr(args, 'merged_by') and args.merged_by else "system",
        },
    )
    print(f"Task {args.task_id} merged. Audit: {audit_path}")


def cmd_task_release(args: argparse.Namespace) -> None:
    """Release a merged task."""
    task_path = TASKS_DIR / f"{args.task_id}.md"
    if not task_path.exists():
        raise RuntimeError(f"TaskCard not found: {task_path}")

    state_machine = load_state_machine()
    tasks_state_path = ROOT / "state" / "tasks.yaml"
    tasks_state = read_yaml(tasks_state_path)

    current_task = get_task(tasks_state, args.task_id)
    current_status = current_task.get("status", "draft")

    if current_status != "merged":
        raise RuntimeError(f"Task {args.task_id} is not in merged state (current: {current_status})")

    if not can_transition(state_machine, current_status, "released"):
        raise RuntimeError(f"Invalid transition {current_status} -> released")

    upsert_task(
        tasks_state,
        args.task_id,
        {
            "status": "released",
        },
    )
    append_event(
        tasks_state,
        args.task_id,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "task_release",
            "from": current_status,
            "to": "released",
        },
    )
    write_yaml(tasks_state_path, tasks_state)

    # Move TaskCard to done folder
    done_dir = TASKS_DIR / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    done_path = done_dir / f"{args.task_id}.md"
    task_path.rename(done_path)

    print(f"Task {args.task_id} released and moved to {done_path}")


def cmd_task_status(args: argparse.Namespace) -> None:
    tasks_state_path = ROOT / "state" / "tasks.yaml"
    tasks_state = read_yaml(tasks_state_path)
    current_task = get_task(tasks_state, args.task_id)
    status = current_task.get("status", "draft")
    print(f"Task {args.task_id} status: {status}")


def cmd_task_list(args: argparse.Namespace) -> None:
    """List tasks, optionally sorted by priority."""
    tasks_state_path = ROOT / "state" / "tasks.yaml"
    tasks_state = read_yaml(tasks_state_path)
    tasks = tasks_state.get("tasks", {})

    if not tasks:
        print("No tasks found.")
        return

    # Collect task info with priorities
    task_list = []
    for task_id, task_data in tasks.items():
        status = task_data.get("status", "draft")
        priority = task_data.get("priority", "P3")
        queue = task_data.get("queue", "-")
        task_list.append({
            "task_id": task_id,
            "status": status,
            "priority": priority,
            "queue": queue,
        })

    # Filter by status if specified
    if args.status:
        task_list = [t for t in task_list if t["status"] == args.status]

    # Filter by queue if specified
    if args.queue:
        task_list = [t for t in task_list if t["queue"] == args.queue]

    # Sort by priority (P0 first) then by task_id
    task_list.sort(key=lambda t: (get_priority_order(t["priority"]), t["task_id"]))

    if not task_list:
        print("No tasks match the filter criteria.")
        return

    # Print header
    print(f"{'Task ID':<25} {'Status':<12} {'Priority':<10} {'Queue':<10}")
    print("-" * 60)

    for t in task_list:
        priority_marker = "🔴" if t["priority"] == "P0" else "🟠" if t["priority"] == "P1" else "🟡" if t["priority"] == "P2" else "🟢"
        print(f"{t['task_id']:<25} {t['status']:<12} {priority_marker} {t['priority']:<7} {t['queue']:<10}")

    print(f"\nTotal: {len(task_list)} task(s)")


def get_sorted_task_ids(tasks_state: Dict[str, Any], status_filter: str = None, queue_filter: str = None) -> List[str]:
    """Return task IDs sorted by priority (P0 first, then P1, etc.)."""
    tasks = tasks_state.get("tasks", {})
    
    filtered_tasks = []
    for task_id, task_data in tasks.items():
        if status_filter and task_data.get("status") != status_filter:
            continue
        if queue_filter and task_data.get("queue") != queue_filter:
            continue
        priority = task_data.get("priority", "P3")
        filtered_tasks.append((task_id, get_priority_order(priority)))
    
    # Sort by priority order (lower = higher priority), then by task_id
    filtered_tasks.sort(key=lambda x: (x[1], x[0]))
    
    return [task_id for task_id, _ in filtered_tasks]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Workflow OS Kernel v0 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize OS state")
    init_parser.set_defaults(func=cmd_init)

    task_parser = subparsers.add_parser("task", help="Task operations")
    task_sub = task_parser.add_subparsers(dest="task_command", required=True)

    task_new = task_sub.add_parser("new", help="Create a new TaskCard")
    task_new.add_argument("task_id")
    task_new.set_defaults(func=cmd_task_new)

    task_start = task_sub.add_parser("start", help="Start a task")
    task_start.add_argument("task_id")
    task_start.set_defaults(func=cmd_task_start)

    task_finish = task_sub.add_parser("finish", help="Finish a task")
    task_finish.add_argument("task_id")
    task_finish.set_defaults(func=cmd_task_finish)

    task_merge = task_sub.add_parser("merge", help="Merge a reviewed task")
    task_merge.add_argument("task_id")
    task_merge.add_argument("--merged-by", default="system", help="Who merged the task")
    task_merge.set_defaults(func=cmd_task_merge)

    task_release = task_sub.add_parser("release", help="Release a merged task")
    task_release.add_argument("task_id")
    task_release.set_defaults(func=cmd_task_release)

    task_status = task_sub.add_parser("status", help="Show task status")
    task_status.add_argument("task_id")
    task_status.set_defaults(func=cmd_task_status)

    task_list = task_sub.add_parser("list", help="List tasks sorted by priority")
    task_list.add_argument("--status", help="Filter by status (draft, running, reviewing, etc.)")
    task_list.add_argument("--queue", help="Filter by queue (dev, research, data, gov)")
    task_list.set_defaults(func=cmd_task_list)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
