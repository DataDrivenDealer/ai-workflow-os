# OS_OPERATING_MODEL

**Document ID**: OS_OPERATING_MODEL  
**Type**: Operational  
**Scope**: Company OS  
**Status**: Active  
**Version**: 0.1.0  
**Owner**: Company Governance + Platform Engineering

---

## 0. Purpose

This document explains **how the AI Workflow OS is actually operated day-to-day**.

It answers:
- Who does what?
- When are different operations performed?
- How do humans and AI agents collaborate?
- What are the escalation paths?

---

## 1. Operating Roles

### 1.1 Role Definitions

| Role | Responsibility | Authority Level |
|------|----------------|-----------------|
| **Project Owner** | Ultimate governance authority | L0 - Can freeze/unfreeze |
| **Platform Engineer** | Maintain kernel, tools, infrastructure | L1 - Framework changes |
| **Project Lead** | Manage project-level specs and tasks | L2 - Project adaptive |
| **AI Agent** | Execute tasks within defined boundaries | Speculative only |
| **Human Operator** | Day-to-day task management | Per-role assignment |

### 1.2 Role-Mode Matrix for AI Agents

| Role Mode | Can Do | Cannot Do |
|-----------|--------|-----------|
| `architect` | Define specs, propose changes | Execute code, modify production |
| `planner` | Decompose tasks, create TaskCards | Approve governance changes |
| `executor` | Run defined tasks, produce artifacts | Create new tasks, change specs |
| `builder` | Implement features, write code | Approve releases, change canon |

---

## 2. Daily Operations

### 2.1 Morning Standup Pattern

```
08:00 - Review overnight agent activity
      - Check ops/audit/ for new entries
      - Review state/tasks.yaml for status changes

08:30 - Task triage
      - Review tasks/inbox/ for new requests
      - Assign priorities and queues

09:00 - Start execution cycle
      - Agents begin working on running tasks
      - Human oversight via task status checks
```

### 2.2 Task Workflow (Human + Agent)

```
┌─────────────────────────────────────────────────────────────┐
│                    Daily Task Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Human]                    [AI Agent]                     │
│     │                           │                          │
│     │ 1. Create TaskCard        │                          │
│     │ ──────────────────────>   │                          │
│     │                           │                          │
│     │ 2. Approve task start     │                          │
│     │ ──────────────────────>   │                          │
│     │                           │                          │
│     │                    3. Execute task                   │
│     │                    (speculative)                     │
│     │                           │                          │
│     │ <────────────────────     │                          │
│     │    4. Submit for review   │                          │
│     │                           │                          │
│     │ 5. Review & approve       │                          │
│     │ ──────────────────────>   │                          │
│     │                           │                          │
│     │         6. Merge (authority granted)                 │
│     │                           │                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Pair Programming Workflow (NEW)

The Pair Programming feature adds automated code review between code generation and final review.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Pair Programming Workflow                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Builder Agent]              [Reviewer Agent]           [Human]        │
│        │                            │                       │           │
│        │ 1. Generate Code           │                       │           │
│        │ ─────────────>             │                       │           │
│        │                            │                       │           │
│        │ 2. review_submit           │                       │           │
│        │ ─────────────────────────> │                       │           │
│        │    [code_review state]     │                       │           │
│        │                            │                       │           │
│        │                     3. review_create_session       │           │
│        │                     4. Quality Check (Q)           │           │
│        │                     5. Requirements Check (R)      │           │
│        │                     6. Completeness Check (C)      │           │
│        │                     7. Optimization Check (O)      │           │
│        │                            │                       │           │
│        │ <──────────────────────────│                       │           │
│        │    8. Review Report        │                       │           │
│        │                            │                       │           │
│   [If NEEDS_REVISION]               │                       │           │
│        │                            │                       │           │
│        │ 9. Revise Code             │                       │           │
│        │ ─────────────────────────> │                       │           │
│        │    (repeat 3-8)            │                       │           │
│        │                            │                       │           │
│   [If APPROVED]                     │                       │           │
│        │                     10. review_approve             │           │
│        │ <──────────────────────────│                       │           │
│        │    [reviewing state]       │                       │           │
│        │                            │                       │           │
│        │                                                    │           │
│        │ ──────────────────────────────────────────────────>│           │
│        │                            11. Human Final Review  │           │
│        │ <──────────────────────────────────────────────────│           │
│        │                            12. Merge (authority)   │           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Review Dimensions

The Pair Programming review covers four mandatory dimensions:

| Dimension | Code | Focus |
|-----------|------|-------|
| Quality Check | Q | Bugs, type safety, error handling, security |
| Requirements Check | R | Spec compliance, feature completeness |
| Completeness Check | C | All requirements addressed, tests included |
| Optimization Check | O | Code elegance, efficiency, maintainability |

### 2.5 Review Verdicts

| Verdict | Meaning | Next Action |
|---------|---------|-------------|
| APPROVED | All checks passed | Advance to `reviewing` |
| NEEDS_REVISION | Critical/Major issues found | Revise and resubmit |
| BLOCKED | Fundamental problem | Escalate to human |

### 2.6 Expert Personas

Reviewers can apply specialized perspectives:

| Persona | Focus |
|---------|-------|
| `security_expert` | OWASP, injection, auth, data protection |
| `performance_expert` | Algorithms, caching, I/O optimization |
| `architecture_expert` | SOLID, design patterns, coupling |
| `domain_expert` | Business logic, requirements alignment |
| `testing_expert` | Test coverage, edge cases |

### 2.7 Command Reference (Daily Operations)

| Time | Command | Purpose |
|------|---------|---------|
| Morning | `python kernel/os.py task status <ID>` | Check overnight progress |
| Triage | `ls tasks/inbox/` | Review incoming tasks |
| Start | `python kernel/os.py task start <ID>` | Begin task execution |
| Monitor | `cat state/tasks.yaml` | View all task states |
| Review | `ls ops/audit/` | Check audit trail |
| End-of-day | `python kernel/os.py task finish <ID>` | Complete tasks |

---

## 3. Weekly Operations

### 3.1 Weekly Governance Review

**When**: Every Friday  
**Duration**: 1 hour  
**Participants**: Project Owner, Platform Engineer, Project Leads

**Agenda**:
1. Review `ops/decision-log/` for pending decisions
2. Review `ops/proposals/` for spec change requests
3. Check `ops/deviations/` for L2 overrides
4. Update `docs/BLUEPRINT_FREEZE_RECORD.md` if needed

### 3.2 Weekly Checklist

```yaml
weekly_review:
  governance:
    - [ ] Review all new decision-log entries
    - [ ] Process pending proposals
    - [ ] Audit deviation declarations
  
  system_health:
    - [ ] Check for orphaned tasks (running > 7 days)
    - [ ] Review agent registration in state/agents.yaml
    - [ ] Validate spec_registry.yaml consistency
  
  documentation:
    - [ ] Update ARCHITECTURE_PACK_INDEX.md if needed
    - [ ] Refresh capability map if new features added
```

---

## 4. Task Queue Management

### 4.1 Queue Types

| Queue | Purpose | Concurrency |
|-------|---------|-------------|
| `research` | Exploratory work | 2 parallel |
| `data` | Data engineering | 1 sequential |
| `dev` | Development work | 3 parallel |
| `governance` | Spec changes | 1 sequential |

### 4.2 Queue Lock Rules

```yaml
queue_policy:
  dev:
    max_concurrent: 3
    lock_on_start: true
    auto_release_on_finish: true
    
  governance:
    max_concurrent: 1
    requires_human_approval: true
    audit_all_actions: true
```

### 4.3 Queue Commands

```powershell
# Check queue status
cat state/tasks.yaml | grep -A 5 "queues:"

# View tasks in specific queue
cat state/tasks.yaml | grep "queue: dev"

# Release stuck queue (emergency)
# Edit state/tasks.yaml to remove queue lock
```

---

## 5. Audit & Compliance

### 5.1 Audit Trail Structure

```
ops/audit/
├── TASK_DEMO_0001.md      # Per-task audit
├── TASK_DATA_001.md
└── ...

Each audit file contains:
- Task ID
- State transitions
- Timestamps
- Agent/human actions
- Artifacts produced
```

### 5.2 Compliance Checks

| Check | Frequency | Tool |
|-------|-----------|------|
| TaskCard schema | On commit | `hooks/pre-commit` |
| Spec consistency | On push | `hooks/pre-push` |
| Gate validation | On PR | `scripts/gate_check.py` |
| Governance verify | On merge | `kernel/governance_gate.py` |

### 5.3 Audit Commands

```powershell
# View task audit trail
cat ops/audit/TASK_DEMO_0001.md

# Check for governance violations
python scripts/policy_check.py

# Generate gate report
python scripts/gate_check.py --gate all --task-id TASK_DEMO_0001
```

---

## 6. Escalation Paths

### 6.1 Escalation Matrix

| Issue Type | First Response | Escalation | Final Authority |
|------------|----------------|------------|-----------------|
| Task blocked | Human Operator | Project Lead | Platform Engineer |
| Spec conflict | Project Lead | Platform Engineer | Project Owner |
| Security concern | Platform Engineer | Project Owner | Immediate freeze |
| Agent misbehavior | Human Operator | Platform Engineer | Session terminate |

### 6.2 Emergency Procedures

#### Agent Session Termination

```powershell
# Via MCP tool
python -c "
from kernel.mcp_server import MCPServer
server = MCPServer()
server.session_terminate({'session_token': 'TOKEN', 'reason': 'Emergency'})
"

# Via direct state edit
# Edit state/sessions.yaml to set state: terminated
```

#### Task Emergency Stop

```powershell
# Edit state/tasks.yaml
# Change task status to: blocked
# Add event: emergency_stop

# Or via CLI (if task is running)
python kernel/os.py task finish <ID>  # Forces to reviewing
```

#### System Freeze

```powershell
# Lock all queues
# Edit state/tasks.yaml:
# queues:
#   dev: SYSTEM_LOCK
#   research: SYSTEM_LOCK
#   data: SYSTEM_LOCK
#   governance: SYSTEM_LOCK

# Notify all operators
# Create freeze record in ops/freeze/
```

---

## 7. AI Agent Operations

### 7.1 Agent Lifecycle

```
1. Registration
   └─ agent_register → receives agent_id

2. Session Creation
   └─ session_create → receives session_token
   └─ Specifies role_mode and task_scope

3. Task Execution
   └─ task_start → marks task running
   └─ (Agent performs work)
   └─ task_finish → marks task reviewing

4. Session Termination
   └─ session_terminate → cleans up
```

### 7.2 Agent Monitoring

```powershell
# List registered agents
cat state/agents.yaml

# List active sessions
cat state/sessions.yaml

# View agent activity in audit
grep -r "agent_id" ops/audit/
```

### 7.3 Agent Boundaries

| Boundary | Enforcement |
|----------|-------------|
| Task scope | Session `task_scope` field |
| Role permissions | `RoleMode` enum checks |
| Time limits | Session expiry (configurable) |
| Output validation | `governance_check` tool |

---

## 8. Maintenance Operations

### 8.1 Monthly Maintenance

| Task | Command/Action |
|------|----------------|
| Archive old audits | `mv ops/audit/*.md ops/archive/` |
| Clean test agents | Edit `state/agents.yaml` |
| Update dependencies | `pip install -r requirements.txt --upgrade` |
| Validate all specs | `python scripts/policy_check.py --all` |

### 8.2 State File Backup

```powershell
# Backup state directory
$date = Get-Date -Format "yyyy-MM-dd"
Copy-Item -Recurse state/ "backups/state-$date/"

# Verify backup
ls "backups/state-$date/"
```

### 8.3 Recovery Procedures

```powershell
# Restore from backup
Copy-Item -Recurse "backups/state-$date/*" state/

# Re-initialize if corrupted
python kernel/os.py init

# Validate state
cat state/project.yaml
cat state/tasks.yaml
```

---

## 9. Metrics & Observability

### 9.1 Key Metrics

| Metric | Source | Target |
|--------|--------|--------|
| Tasks completed/week | `state/tasks.yaml` | ≥10 |
| Average task cycle time | Audit timestamps | <3 days |
| Governance violations | `ops/deviations/` | 0 |
| Agent session success rate | `state/sessions.yaml` | >95% |

### 9.2 Health Indicators

```yaml
health_check:
  green:
    - All queues operational
    - No tasks blocked >24h
    - No unresolved deviations
    
  yellow:
    - Queue at capacity
    - Task blocked 24-72h
    - Pending proposals >3
    
  red:
    - Queue locked
    - Task blocked >72h
    - Unaudited actions detected
```

---

## 10. References

- [README_START_HERE.md](../README_START_HERE.md) - Quick start guide
- [CO_OS_CAPABILITY_MAP.md](CO_OS_CAPABILITY_MAP.md) - System capabilities
- [SECURITY_TRUST_BOUNDARY.mmd](SECURITY_TRUST_BOUNDARY.mmd) - Security model
- [GOVERNANCE_INVARIANTS.md](../specs/canon/GOVERNANCE_INVARIANTS.md) - Core invariants

---

## 11. Change Log

| Date | Version | Change |
|------|---------|--------|
| 2026-02-01 | 0.1.0 | Initial operating model |
