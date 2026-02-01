# MCP Server Usage Guide

This guide explains how to connect AI agents to the AI Workflow OS via MCP (Model Context Protocol).

## Quick Start

### Running the Server

```bash
# Run in stdio mode (for AI client integration)
python -m kernel.mcp_stdio

# Run with debug logging
python -m kernel.mcp_stdio --debug

# Run self-test
python -m kernel.mcp_stdio --test
```

### Connecting from Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-workflow-os": {
      "command": "python",
      "args": ["-m", "kernel.mcp_stdio"],
      "cwd": "/path/to/ai-workflow-os"
    }
  }
}
```

## Protocol Overview

The server implements **MCP 2024-11-05** over **stdio transport** using **JSON-RPC 2.0**.

### Handshake Flow

```
Client                                    Server
  |                                          |
  |-------- initialize ------------------>   |
  |<------- capabilities, serverInfo -----   |
  |-------- initialized (notification) -->   |
  |                                          |
  |  (Server is ready for tool calls)        |
```

### Available Methods

| Method | Type | Description |
|--------|------|-------------|
| `initialize` | Request | Start handshake, exchange capabilities |
| `initialized` | Notification | Confirm initialization complete |
| `tools/list` | Request | List available tools |
| `tools/call` | Request | Execute a tool |
| `resources/list` | Request | List available resources |
| `resources/read` | Request | Read a resource by URI |
| `prompts/list` | Request | List prompt templates |
| `prompts/get` | Request | Get a prompt with arguments |
| `shutdown` | Request | Prepare for termination |
| `exit` | Notification | Terminate server |

## Available Tools

### Agent Management

#### `agent_register`
Register a new AI agent before creating sessions.

```json
{
  "name": "agent_register",
  "arguments": {
    "agent_type": "ai_claude",
    "display_name": "Claude Assistant",
    "allowed_role_modes": ["executor", "builder"]
  }
}
```

Returns: `agent_id` for use in session creation.

### Session Management

#### `session_create`
Create an authorized session for operations.

```json
{
  "name": "session_create",
  "arguments": {
    "agent_id": "ai_claude-abc123",
    "role_mode": "executor",
    "authorized_by": "user_request"
  }
}
```

Returns: `session_token` (required for all other operations).

#### `session_validate`
Check if a session is still active.

```json
{
  "name": "session_validate",
  "arguments": {
    "session_token": "sess-xxx..."
  }
}
```

#### `session_terminate`
End a session when work is complete.

```json
{
  "name": "session_terminate",
  "arguments": {
    "session_token": "sess-xxx...",
    "reason": "work_complete"
  }
}
```

### Task Operations

#### `task_list`
List all tasks with optional filtering.

```json
{
  "name": "task_list",
  "arguments": {
    "session_token": "sess-xxx...",
    "status_filter": "running"
  }
}
```

#### `task_get`
Get details of a specific task.

```json
{
  "name": "task_get",
  "arguments": {
    "session_token": "sess-xxx...",
    "task_id": "TASK_DEMO_0001"
  }
}
```

#### `task_start`
Start working on a task (changes status to 'running').

```json
{
  "name": "task_start",
  "arguments": {
    "session_token": "sess-xxx...",
    "task_id": "TASK_DEMO_0001"
  }
}
```

#### `task_finish`
Mark a task as complete (changes status to 'reviewing').

```json
{
  "name": "task_finish",
  "arguments": {
    "session_token": "sess-xxx...",
    "task_id": "TASK_DEMO_0001",
    "artifacts": ["output.py"],
    "result": "success"
  }
}
```

### Governance

#### `governance_check`
Verify output against governance rules before submission.

```json
{
  "name": "governance_check",
  "arguments": {
    "session_token": "sess-xxx...",
    "output_text": "I will implement the requested feature."
  }
}
```

**Important**: Use this before any output to ensure compliance. The following will be rejected:
- Authority claims ("I approve", "I authorize", etc.)
- Governance actions ("supersede", "revoke", etc.)

### Artifacts

#### `artifact_list`
List files in a directory.

```json
{
  "name": "artifact_list",
  "arguments": {
    "session_token": "sess-xxx...",
    "path": "specs/canon"
  }
}
```

#### `artifact_read`
Read a file's content.

```json
{
  "name": "artifact_read",
  "arguments": {
    "session_token": "sess-xxx...",
    "path": "README.md"
  }
}
```

### Specifications

#### `spec_list`
List registered specifications.

```json
{
  "name": "spec_list",
  "arguments": {
    "session_token": "sess-xxx..."
  }
}
```

## Resources

Resources provide read access to workspace files via URI:

| URI Pattern | Description |
|-------------|-------------|
| `file://spec_registry.yaml` | Spec registry |
| `file://specs/canon/*.md` | Canon specifications |
| `file://tasks/TASK_*.md` | TaskCards |

### Reading a Resource

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/read",
  "params": {
    "uri": "file://specs/canon/GOVERNANCE_INVARIANTS.md"
  }
}
```

## Role Modes

Agents operate under specific Role Modes that determine permissions:

| Role Mode | Permissions |
|-----------|-------------|
| `architect` | Design specs, cannot write code |
| `planner` | Create tasks, cannot implement |
| `executor` | Run tasks, restricted scope |
| `builder` | Write code, must follow specs |

Your allowed role modes are set during `agent_register`.

## Complete Workflow Example

```python
# 1. Register agent
agent = tools.call("agent_register", {
    "agent_type": "ai_claude",
    "display_name": "Claude",
    "allowed_role_modes": ["executor", "builder"]
})
agent_id = agent["agent"]["agent_id"]

# 2. Create session
session = tools.call("session_create", {
    "agent_id": agent_id,
    "role_mode": "executor",
    "authorized_by": "user"
})
token = session["session"]["session_token"]

# 3. List tasks
tasks = tools.call("task_list", {"session_token": token})

# 4. Start a task
tools.call("task_start", {
    "session_token": token,
    "task_id": "TASK_001"
})

# 5. Check governance before output
check = tools.call("governance_check", {
    "session_token": token,
    "output_text": "Implementation complete."
})
if check["passed"]:
    # 6. Finish task
    tools.call("task_finish", {
        "session_token": token,
        "task_id": "TASK_001",
        "artifacts": ["output.py"],
        "result": "success"
    })

# 7. Terminate session
tools.call("session_terminate", {
    "session_token": token,
    "reason": "work_complete"
})
```

## Error Handling

All tool calls return either success data or an error:

```json
{
  "success": false,
  "error": "SESSION_NOT_FOUND",
  "message": "Session not found or expired"
}
```

Common errors:
- `SESSION_NOT_FOUND`: Invalid or expired session token
- `INVALID_PARAMS`: Missing required parameters
- `GOVERNANCE_VIOLATION`: Output contains authority claims
- `ROLE_MODE_DENIED`: Action not allowed for current role mode

## Testing

```bash
# Run unit tests
python scripts/test_mcp_server.py --verbose

# Run end-to-end tests
python scripts/test_mcp_e2e.py

# Run workflow simulation
python scripts/simulate_agent_workflow.py
```

## Configuration Files

| File | Purpose |
|------|---------|
| `.vscode/mcp.json` | VS Code MCP settings |
| `mcp_server_manifest.json` | Server manifest for discovery |
| `kernel/mcp_stdio.py` | Stdio protocol handler |
| `kernel/mcp_server.py` | Tool implementations |
