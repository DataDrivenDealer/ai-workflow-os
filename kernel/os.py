from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from audit import write_audit
from state_store import append_event, get_task, init_state, read_yaml, upsert_task, write_yaml
from task_parser import parse_taskcard, validate_taskcard

ROOT = Path(__file__).resolve().parents[1]
STATE_MACHINE_PATH = ROOT / "kernel" / "state_machine.yaml"
REGISTRY_PATH = ROOT / "spec_registry.yaml"
TEMPLATE_PATH = ROOT / "templates" / "TASKCARD_TEMPLATE.md"
TASKS_DIR = ROOT / "tasks"


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

    queues[queue] = args.task_id
    upsert_task(
        tasks_state,
        args.task_id,
        {
            "status": "running",
            "queue": queue,
            "branch": branch_name,
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


def cmd_task_status(args: argparse.Namespace) -> None:
    tasks_state_path = ROOT / "state" / "tasks.yaml"
    tasks_state = read_yaml(tasks_state_path)
    current_task = get_task(tasks_state, args.task_id)
    status = current_task.get("status", "draft")
    print(f"Task {args.task_id} status: {status}")


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

    task_status = task_sub.add_parser("status", help="Show task status")
    task_status.add_argument("task_id")
    task_status.set_defaults(func=cmd_task_status)

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
