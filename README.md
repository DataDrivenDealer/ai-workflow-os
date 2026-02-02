# AI Workflow OS

Canonical architecture pack and governance system for AI Workflow OS.

## Structure
- docs/: Canonical architecture pack artifacts
- specs/: Canonical and framework specs
- projects/dgsf/: Linked DGSF project reference

## Architecture

This project follows a canonical architecture pack model:
- 📘 [Architecture Pack Index](docs/ARCHITECTURE_PACK_INDEX.md) - Complete architecture overview
- 📐 [Architecture Blueprint](docs/ARCH_BLUEPRINT_MASTER.mmd) - System structure
- 🔒 [Governance Invariants](specs/canon/GOVERNANCE_INVARIANTS.md) - Constitutional rules
- 🎭 [Role Mode Canon](specs/canon/ROLE_MODE_CANON.md) - Role-based authorization

## Documentation

- [MCP Usage Guide](docs/MCP_USAGE_GUIDE.md) - How to use the MCP Server
- [Pair Programming Guide](docs/PAIR_PROGRAMMING_GUIDE.md) - Code review process
- [System Invariants](docs/SYSTEM_INVARIANTS.md) - Verifiable system guarantees
- [Project Playbook](docs/PROJECT_PLAYBOOK.md) - Development workflows
- [Spec Registry Schema](docs/SPEC_REGISTRY_SCHEMA.md) - Specification governance

## Status
Bootstrap repository skeleton initialized. Legacy documents preserved in `legacy/ai_workflow_os_upgrade_dgsf_project`.

## Quickstart
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
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

## Install local hooks
```powershell
.\scripts\install_hooks.ps1
```
