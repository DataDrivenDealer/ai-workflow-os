import json
import os
import re
from pathlib import Path


def _normalize_heading(heading: str) -> str:
    heading = heading.strip().lower()
    heading = re.sub(r"\s*\(.*\)\s*", "", heading)
    heading = heading.replace("_", " ")
    heading = re.sub(r"\s+", " ", heading)
    return heading


def _parse_sections(body: str) -> dict:
    sections = {}
    current_key = None

    for line in body.splitlines():
        match = re.match(r"^###\s+(.*)\s*$", line.strip())
        if match:
            heading = _normalize_heading(match.group(1))
            current_key = heading
            sections.setdefault(current_key, [])
            continue
        if current_key is not None:
            sections[current_key].append(line)

    def get_section(name: str, *aliases: str) -> str:
        keys = (name,) + aliases
        for key in keys:
            if key in sections:
                text = "\n".join(sections[key]).strip()
                return text
        return ""

    return {
        "objective": get_section("objective"),
        "constraints": get_section("constraints"),
        "scope": get_section("scope"),
        "acceptance_criteria": get_section("acceptance criteria", "acceptance"),
    }


def _read_event_payload(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_text(value: str, fallback: str = "TBD") -> str:
    value = (value or "").strip()
    return value if value else fallback


def main() -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise SystemExit("GITHUB_EVENT_PATH is not set")

    payload = _read_event_payload(event_path)
    issue = payload.get("issue") or {}

    number = issue.get("number")
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    author = (issue.get("user") or {}).get("login") or "unknown"
    issue_url = issue.get("html_url") or ""

    if not number:
        raise SystemExit("Issue number not found in event payload")

    sections = _parse_sections(body)

    task_id = f"TASK_ISSUE_{number}"
    task_path = Path("tasks") / "inbox" / f"{task_id}.md"
    task_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""---
task_id: {task_id}
type: dev
queue: dev
branch: feature/issue-{number}
requires_review: false
spec_ids:
  - ARCH_BLUEPRINT_MASTER
  - TASK_STATE_MACHINE
artifacts_expected: []
verification:
  - "policy_check (ci)"
  - "CI"
---

# Summary
{_safe_text(title)}

# Objective
{_safe_text(sections.get('objective'))}

# Constraints
{_safe_text(sections.get('constraints'))}

# Scope
{_safe_text(sections.get('scope'))}

# Acceptance Criteria
{_safe_text(sections.get('acceptance_criteria'))}

# Metadata
- Source Issue: {issue_url}
- Issue Author: {author}
"""

    task_path.write_text(content, encoding="utf-8")
    print(f"Wrote TaskCard: {task_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
