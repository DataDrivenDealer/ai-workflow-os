# 🚀 AI Workflow OS - Start Here

> **One-command demo path to get started with AI Workflow OS**

---

## What is AI Workflow OS?

AI Workflow OS is a **governance-first operating system** for AI-assisted software development, specifically designed for:

- 🏗️ **Structured AI Agent Workflows** - Define how AI agents operate within your organization
- 📋 **Task Lifecycle Management** - Track tasks from inception to release
- 🔒 **Governance & Compliance** - Enforce company policies and audit trails
- 🤖 **Multi-Agent Coordination** - MCP-based protocol for AI agent integration

---

## ⚡ Quick Start (5 minutes)

### Prerequisites

- Python 3.10+
- Git

### Step 1: Setup Environment

```powershell
# Clone and enter the repository
cd "e:\AI Tools\AI Workflow OS"

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Initialize the OS

```powershell
# Initialize state directories and configuration
python kernel/os.py init
```

Expected output:
```
Initialized state and ops directories.
```

### Step 3: Run Your First Task

```powershell
# Create a new task
python kernel/os.py task new TASK_MY_FIRST

# Start working on the task
python kernel/os.py task start TASK_MY_FIRST

# Complete the task
python kernel/os.py task finish TASK_MY_FIRST
```

### Step 4: Check Task Status

```powershell
python kernel/os.py task status TASK_MY_FIRST
# Output: Task TASK_MY_FIRST status: reviewing
```

🎉 **Congratulations!** You've completed your first task lifecycle.

---

## 📁 Project Structure

```
AI Workflow OS/
├── kernel/           # Core runtime (CLI, MCP Server, State Machine)
│   ├── os.py         # Main CLI entry point
│   ├── mcp_server.py # MCP protocol server for AI agents
│   └── state_machine.yaml
├── docs/             # Architecture blueprints (Mermaid diagrams)
├── specs/            # Canonical specifications
│   ├── canon/        # L0 - Company Constitution
│   └── framework/    # L1 - Platform Standards
├── state/            # Runtime state (YAML files)
├── tasks/            # TaskCards (task definitions)
├── templates/        # TaskCard templates
├── configs/          # Gate configurations
├── scripts/          # Utility scripts
├── hooks/            # Git hooks for enforcement
└── ops/              # Operational artifacts
    ├── audit/        # Execution audit logs
    ├── decision-log/ # Governance decisions
    └── freeze/       # Freeze records
```

---

## 🔧 Common Operations

### Task Management

| Command | Description |
|---------|-------------|
| `python kernel/os.py task new <ID>` | Create new TaskCard |
| `python kernel/os.py task start <ID>` | Start working on task |
| `python kernel/os.py task finish <ID>` | Mark task complete |
| `python kernel/os.py task status <ID>` | Check task status |

### Task Lifecycle

```
draft → ready → running → reviewing → merged → released
```

---

## 🤖 AI Agent Integration (MCP)

AI Workflow OS exposes its capabilities via the **Model Context Protocol (MCP)**.

### Start MCP Server

```powershell
python kernel/mcp_stdio.py
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `agent_register` | Register new AI agent |
| `session_create` | Create authorized session |
| `task_list` | List all tasks |
| `task_start` | Start a task |
| `task_finish` | Complete a task |
| `governance_check` | Verify compliance |

See [MCP_USAGE_GUIDE.md](docs/MCP_USAGE_GUIDE.md) for detailed integration instructions.

---

## 📚 Documentation Map

| Document | Purpose | Audience |
|----------|---------|----------|
| [ARCHITECTURE_PACK_INDEX](docs/ARCHITECTURE_PACK_INDEX.md) | Master blueprint index | Architects |
| [ARCH_BLUEPRINT_MASTER](docs/ARCH_BLUEPRINT_MASTER.mmd) | System architecture | All |
| [CO_OS_CAPABILITY_MAP](docs/CO_OS_CAPABILITY_MAP.md) | OS capabilities | Engineers |
| [SPEC_GOVERNANCE_MODEL](docs/SPEC_GOVERNANCE_MODEL.mmd) | Spec governance | Governance |
| [MCP_SERVER_TEST_REPORT](docs/MCP_SERVER_TEST_REPORT.md) | MCP test results | DevOps |

---

## 🔐 Governance Model

AI Workflow OS enforces a three-layer specification system:

| Layer | Name | Mutability | Example |
|-------|------|------------|---------|
| **L0** | Canon | Frozen | `GOVERNANCE_INVARIANTS.md` |
| **L1** | Framework | Controlled | `AGENT_SESSION.md` |
| **L2** | Project | Flexible | Project-specific rules |

Key principles:
- **Artifact over Conversation** - All truth is externalized
- **Explicit Authority** - No implicit permissions
- **Full Auditability** - Every action is logged

---

## 🛠️ Git Hooks Setup

Install local enforcement hooks:

```powershell
# Run setup script
.\scripts\install_hooks.ps1

# Or manual copy
Copy-Item -Force hooks\pre-commit .git\hooks\pre-commit
Copy-Item -Force hooks\pre-push .git\hooks\pre-push
```

---

## 🧪 Run Tests

```powershell
# MCP Server unit tests
python scripts/test_mcp_server.py

# End-to-end tests
python scripts/test_mcp_e2e.py

# Gate check validation
python scripts/gate_check.py --gate G1 --task-id TASK_DEMO_0001
```

---

## 🎯 Next Steps

1. **Explore the Architecture**: Read [ARCHITECTURE_PACK_INDEX.md](docs/ARCHITECTURE_PACK_INDEX.md)
2. **Understand Governance**: Review [GOVERNANCE_INVARIANTS.md](specs/canon/GOVERNANCE_INVARIANTS.md)
3. **Create Your First Project Task**: Use pipeline templates in `templates/pipeline/`
4. **Integrate AI Agent**: Follow [MCP_USAGE_GUIDE.md](docs/MCP_USAGE_GUIDE.md)

---

## 🆘 Troubleshooting

### "Not inside a git work tree"

```powershell
git init
```

### "TaskCard not found"

Ensure the task exists in `tasks/` directory:
```powershell
ls tasks/
```

### "Queue is locked"

Another task is using the queue. Finish or abandon it first:
```powershell
python kernel/os.py task status <BLOCKING_TASK_ID>
```

---

## 📞 Support

- **Architecture Questions**: Review `docs/ARCHITECTURE_PACK_INDEX.md`
- **Governance Questions**: Check `specs/canon/` directory
- **Bug Reports**: Create TaskCard with type `bug`

---

> **Remember**: In AI Workflow OS, **if it's not an artifact, it doesn't exist**.
