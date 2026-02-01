# AGENT_SESSION Protocol

**Spec ID**: AGENT_SESSION
**Scope**: L1 (Framework)
**Status**: Active
**Version**: 0.1.0
**Derived From**: MULTI_AGENT_CANON §10
**Depends On**: ROLE_MODE_CANON, AUTHORITY_CANON, MULTI_AGENT_CANON

---

## 0. Purpose

This specification defines the **Agent Session Protocol** — the standard interface for AI agents to connect to and operate within AI Workflow OS.

An Agent Session provides:
- **Identity**: Who is acting
- **Authorization**: What they are allowed to do (Role Mode)
- **Isolation**: Separation from other sessions
- **Auditability**: Complete action trace

---

## 1. Session Lifecycle

### 1.1 State Diagram

```
┌─────────────────┐
│                 │
│   ANONYMOUS     │ ◄── Agent connects, no authorization yet
│                 │
└────────┬────────┘
         │
         │ authorize(agent_id, role_mode, authorized_by)
         ▼
┌─────────────────┐
│                 │
│     ACTIVE      │ ◄── Agent can perform actions
│                 │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    │ suspend │ terminate/expire
    ▼         ▼
┌─────────┐  ┌─────────────────┐
│SUSPENDED│  │   TERMINATED    │
└────┬────┘  └─────────────────┘
     │              ▲
     │ resume       │
     └──────────────┘ (if expired while suspended)
```

### 1.2 State Definitions

| State | Description | Actions Allowed |
|-------|-------------|-----------------|
| `ANONYMOUS` | Connected but not authorized | None (read-only metadata) |
| `ACTIVE` | Authorized and operating | Per Role Mode permissions |
| `SUSPENDED` | Temporarily paused | None (session preserved) |
| `TERMINATED` | Ended permanently | None (session archived) |

---

## 2. Session Structure

### 2.1 Required Fields

Every session MUST have:

```yaml
session:
  session_token: "sess-abc123..."      # Unique, cryptographically secure
  agent_id: "ai_claude-f8e2a1b3"       # Registered agent identity
  role_mode: "executor"                 # Active Role Mode
  state: "active"                       # Current lifecycle state
  started_at: "2026-02-01T10:00:00Z"   # ISO 8601 timestamp
  authorized_by: "project_owner"        # Who authorized this session
```

### 2.2 Optional Fields

```yaml
session:
  # ... required fields ...
  expires_at: "2026-02-01T18:00:00Z"   # Session expiration
  task_scope: ["TASK_001", "TASK_002"] # Bound task IDs
  parent_session: "sess-parent123"      # For delegation chains
  metadata:                             # Agent-specific data
    model_version: "claude-3.5-opus"
    context_window: 200000
```

### 2.3 Session Token Format

```
sess-{32_hex_chars}
Example: sess-a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4
```

- Generated using cryptographically secure random
- Unique across all sessions (past and present)
- Used for all session-scoped operations

---

## 3. Agent Registration

### 3.1 Registration Request

Before creating sessions, an agent must be registered:

```yaml
register_agent:
  agent_type: "ai_claude"              # "human", "ai_claude", "ai_gpt", etc.
  display_name: "Research Agent Alpha"
  allowed_role_modes:                   # Project Owner determines
    - executor
    - builder
  metadata:
    provider: "anthropic"
    model: "claude-3.5-opus"
```

### 3.2 Registration Response

```yaml
agent:
  agent_id: "ai_claude-f8e2a1b3"       # Generated unique ID
  registered_at: "2026-02-01T09:00:00Z"
  allowed_role_modes: ["executor", "builder"]
```

### 3.3 Agent ID Format

```
{agent_type}-{8_hex_hash}
Example: ai_claude-f8e2a1b3
```

---

## 4. Session Operations

### 4.1 Create Session

**Request:**
```yaml
create_session:
  agent_id: "ai_claude-f8e2a1b3"
  role_mode: "executor"
  authorized_by: "project_owner"
  timeout_minutes: 480                  # Optional, default 8 hours
  task_scope: ["TASK_001"]              # Optional
```

**Response:**
```yaml
session:
  session_token: "sess-a1b2c3d4..."
  state: "active"
  started_at: "2026-02-01T10:00:00Z"
  expires_at: "2026-02-01T18:00:00Z"
```

**Constraints:**
- Agent must be registered
- Role Mode must be in agent's `allowed_role_modes`
- Max 1 active session per agent

### 4.2 Validate Session

**Request:**
```yaml
validate_session:
  session_token: "sess-a1b2c3d4..."
```

**Response:**
```yaml
valid: true
session:
  agent_id: "ai_claude-f8e2a1b3"
  role_mode: "executor"
  state: "active"
  remaining_seconds: 28800
```

### 4.3 Terminate Session

**Request:**
```yaml
terminate_session:
  session_token: "sess-a1b2c3d4..."
  reason: "task_completed"              # Or "user_request", "violation", etc.
```

**Response:**
```yaml
terminated: true
final_state:
  session_token: "sess-a1b2c3d4..."
  state: "terminated"
  ended_at: "2026-02-01T12:30:00Z"
  reason: "task_completed"
```

### 4.4 Switch Role Mode

**Request:**
```yaml
switch_role_mode:
  session_token: "sess-a1b2c3d4..."
  new_role_mode: "builder"
  authorized_by: "project_owner"
```

**Response:**
```yaml
switched: true
session:
  role_mode: "builder"
  previous_role_mode: "executor"
```

**Constraints:**
- No self-escalation (see §5)
- New mode must be in agent's `allowed_role_modes`
- Requires explicit authorization

---

## 5. Escalation Rules

### 5.1 Authority Levels

Role Modes have implicit authority levels:

| Level | Role Mode | Authority |
|-------|-----------|-----------|
| 4 | Architect | Highest (can propose L0/L1) |
| 3 | Planner | Can define tasks, L2 specs |
| 2 | Builder | Can execute with limited planning |
| 1 | Executor | Lowest (execution only) |

### 5.2 Escalation Prohibition

**Strictly prohibited** (MA-20):
- Self-escalation to higher authority level
- Switching from Executor → Planner/Architect
- Switching from Builder → Architect

**Allowed transitions:**
- Executor ↔ Builder (same level group)
- Planner → Builder (downgrade)
- Any mode → same mode (no-op)

**Escalation path:**
```
To escalate: Terminate current session → Request new session at higher level
             (Requires Project Owner authorization)
```

---

## 6. Concurrency Control

### 6.1 Session Isolation

Each session is fully isolated:
- Own Role Mode
- Own action history
- Own artifact locks
- Cannot access other session's state

### 6.2 Artifact Locking

Before modifying an artifact, session must acquire lock:

**Request:**
```yaml
lock_artifact:
  session_token: "sess-a1b2c3d4..."
  artifact_path: "tasks/TASK_001.md"
```

**Response (success):**
```yaml
locked: true
lock_holder: "sess-a1b2c3d4..."
```

**Response (conflict):**
```yaml
locked: false
lock_holder: "sess-other123..."
conflict: true
```

### 6.3 Lock Release

Locks are released:
- Explicitly via `unlock_artifact`
- Automatically on session termination
- NOT automatically on session suspend (preserved)

---

## 7. Audit Trail

### 7.1 Event Recording

Every session maintains an event log:

```yaml
events:
  - timestamp: "2026-02-01T10:00:00Z"
    action: "session_created"
    details:
      role_mode: "executor"
      authorized_by: "project_owner"
  
  - timestamp: "2026-02-01T10:05:00Z"
    action: "artifact_locked"
    details:
      artifact: "tasks/TASK_001.md"
  
  - timestamp: "2026-02-01T10:30:00Z"
    action: "task_started"
    details:
      task_id: "TASK_001"
  
  - timestamp: "2026-02-01T12:00:00Z"
    action: "session_terminated"
    details:
      reason: "completed"
```

### 7.2 Audit Requirements

Per AUTHORITY_CANON §9:
- All events are timestamped
- All events are attributable to session
- Event log is immutable (append-only)
- Event log persists after session termination

---

## 8. Error Conditions

### 8.1 Session Errors

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `SESSION_NOT_FOUND` | Invalid session token | Create new session |
| `SESSION_EXPIRED` | Session past expiration | Create new session |
| `SESSION_TERMINATED` | Session already ended | Create new session |
| `SESSION_SUSPENDED` | Session temporarily paused | Resume or create new |

### 8.2 Authorization Errors

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `AGENT_NOT_FOUND` | Agent ID not registered | Register agent first |
| `ROLE_MODE_NOT_ALLOWED` | Role Mode not in allowed list | Request allowed mode |
| `ESCALATION_PROHIBITED` | Attempted self-escalation | Request new session |
| `CONCURRENT_SESSION` | Agent already has active session | Terminate existing first |

### 8.3 Concurrency Errors

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `ARTIFACT_LOCKED` | Artifact locked by another session | Wait or request release |
| `LOCK_NOT_HELD` | Trying to unlock artifact not locked | Check lock state |

---

## 9. Integration Points

### 9.1 CLI Integration

```bash
# Create session for CLI operations
./os session create --agent-id ai_claude-xxx --role-mode executor

# Validate current session
./os session validate --token sess-xxx

# Terminate session
./os session terminate --token sess-xxx --reason completed
```

### 9.2 MCP Integration

Sessions are passed in MCP tool calls:

```json
{
  "tool": "task_start",
  "arguments": {
    "session_token": "sess-a1b2c3d4...",
    "task_id": "TASK_001"
  }
}
```

### 9.3 HTTP API (Future)

```http
POST /api/v1/sessions
Authorization: Bearer {project_owner_token}
Content-Type: application/json

{
  "agent_id": "ai_claude-xxx",
  "role_mode": "executor"
}
```

---

## 10. Security Considerations

### 10.1 Token Security

- Session tokens must be treated as secrets
- Tokens should not be logged in full
- Tokens should not be shared between agents

### 10.2 Least Privilege

- Agents should request minimal Role Mode needed
- Sessions should be scoped to specific tasks when possible
- Sessions should have reasonable timeouts

### 10.3 Audit for Security Events

Log and alert on:
- Repeated failed session creation attempts
- Escalation attempts
- Unusual session patterns (very short/very long)

---

## 11. Change Control

- **Edit Policy**: Controlled review (Framework-level)
- **Required Artifacts**: ops/proposals/* for breaking changes
- **Approval**: Platform Engineering + Project Owner
- **Backward Compatibility**: Required for minor versions
