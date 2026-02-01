from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def ensure_ops_dirs(root: Path) -> None:
    (root / "ops" / "audit").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "decision-log").mkdir(parents=True, exist_ok=True)


def write_audit(root: Path, task_id: str, status: str, details: Dict[str, Any]) -> Path:
    ensure_ops_dirs(root)
    audit_path = root / "ops" / "audit" / f"{task_id}.md"

    timestamp = datetime.now(timezone.utc).isoformat()
    entry = [
        f"## {timestamp}",
        f"- status: {status}",
        "- details:",
    ]
    for key, value in details.items():
        entry.append(f"  - {key}: {value}")
    entry.append("")

    if audit_path.exists():
        existing = audit_path.read_text(encoding="utf-8")
    else:
        existing = f"# Audit Log — {task_id}\n\n"

    audit_path.write_text(existing + "\n".join(entry), encoding="utf-8")
    return audit_path


def write_decision_draft(root: Path, task_id: str, summary: str) -> Path:
    ensure_ops_dirs(root)
    decision_path = root / "ops" / "decision-log" / f"{task_id}.md"
    if not decision_path.exists():
        decision_path.write_text(
            f"# Decision Draft — {task_id}\n\n{summary}\n",
            encoding="utf-8",
        )
    return decision_path
