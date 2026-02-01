# AI Workflow OS

Canonical architecture pack and governance system for AI Workflow OS.

## Structure
- docs/: Canonical architecture pack artifacts
- specs/: Canonical and framework specs
- projects/dgsf/: Linked DGSF project reference

## Status
Bootstrap repository skeleton initialized. Legacy documents preserved in `legacy/ai_workflow_os_upgrade_dgsf_project`.

## Quickstart
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe kernel/os.py init
.venv\Scripts\python.exe kernel/os.py task new TASK_DEMO_0002
.venv\Scripts\python.exe kernel/os.py task start TASK_DEMO_0002
.venv\Scripts\python.exe kernel/os.py task finish TASK_DEMO_0002
```

## Hooks (local enforcement)
Copy the hooks into your local git hooks folder:

```powershell
Copy-Item -Force hooks\pre-commit .git\hooks\pre-commit
Copy-Item -Force hooks\pre-push .git\hooks\pre-push
```

Then ensure your `python` resolves to the intended environment.
