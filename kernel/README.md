# Kernel v0 CLI

Minimal runnable kernel CLI for AI Workflow OS.

## Commands
- `python kernel/os.py init`
- `python kernel/os.py task new <TASK_ID>`
- `python kernel/os.py task start <TASK_ID>`
- `python kernel/os.py task finish <TASK_ID>`
- `python kernel/os.py task status <TASK_ID>`

## Notes
- State is stored in `state/project.yaml` and `state/tasks.yaml`.
- Audits are written to `ops/audit/<task_id>.md`.
- TaskCards are created from `templates/TASKCARD_TEMPLATE.md` into `tasks/`.
