# Playbook: Plan to Execute Transition

## Context
Use this playbook when transitioning from Plan Mode (P9 complete) to Execute Mode.
This ensures proper handoff and execution queue readiness.

## Prerequisites
- Plan Mode session completed (P9)
- Execution queue populated
- All specs consistent

## Steps

### 1. Verify Plan Mode Exit State
```bash
python kernel/plan_mode_phases.py status
```
- **Verification**: Current phase is P9 or COMPLETE, active=false

### 2. Verify Execution Queue
```bash
python kernel/state_store.py read execution_queue
```
- **Verification**: Queue has items with status="pending"

### 3. Check GitHub Bindings (if applicable)
```bash
python kernel/github_integration.py list
```
- **Verification**: Tasks bound to issues, or explicitly unbound

### 4. Verify First Task Readiness
First task should have:
- Clear acceptance criteria
- Verification method defined
- Required subagents listed
- Spec pointers

### 5. Announce Mode Switch
```
🎯 Switching to EXECUTE MODE

First task: {task_id} - {title}
Acceptance criteria: ...
Verification: ...
```

### 6. Start First Task
```bash
python kernel/github_integration.py start-task <task_id>
```

## Verification Checklist
- [ ] Plan mode ended properly (P9)
- [ ] Execution queue populated
- [ ] First task has clear criteria
- [ ] Subagent requirements identified
- [ ] GitHub bindings established (optional)

## Decision Points

| Situation | Action |
|-----------|--------|
| No tasks in queue | Return to Plan Mode |
| First task missing criteria | Return to Plan Mode (P8) |
| Escalations pending | Resolve before executing |
| Spec inconsistency detected | Return to Plan Mode |

## Common Issues

| Issue | Solution |
|-------|----------|
| Queue empty after Plan Mode | Re-run P8 to populate queue |
| Task criteria unclear | Use `/dgsf_spec_triage` to clarify |
| Subagent requirements missing | Add to task before starting |

## Related Playbooks
- [execute_to_plan.md](execute_to_plan.md)
- [experiment_lifecycle.md](experiment_lifecycle.md)

## Evolution Log
| Date | Change | Reason |
|------|--------|--------|
| 2025-02-01 | Initial creation | Codify mode transition pattern |
