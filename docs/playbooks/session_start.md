# Playbook: Session Start

## Context
Use this playbook when beginning a new work session on the AI Workflow OS.
This ensures proper context restoration and state verification.

## Prerequisites
- Terminal access to workspace
- Python environment configured

## Steps

### 1. Check Context Hygiene
```bash
python kernel/context_hygiene.py status
```
- **Verification**: No critical warnings displayed

### 2. Check Plan Mode State
```bash
python kernel/plan_mode_phases.py status
```
- **Verification**: Know if there's an active Plan Mode session to resume

### 3. Check Execution Queue
```bash
python kernel/state_store.py read execution_queue
```
- **Verification**: Understand pending tasks from previous session

### 4. Check Knowledge Sync Status
```bash
python kernel/knowledge_sync.py check
```
- **Verification**: No overdue syncs (or plan to address them)

### 5. Review Git Status
```bash
git status
git log --oneline -5
```
- **Verification**: Clean working directory or understand uncommitted changes

### 6. Check for Escalations
```bash
python kernel/state_store.py read escalation_queue
```
- **Verification**: No pending escalations or plan to address them

## Verification Checklist
- [ ] Context hygiene assessed
- [ ] Plan mode state understood
- [ ] Execution queue reviewed
- [ ] Knowledge sync status checked
- [ ] Git state clean or understood
- [ ] Escalations checked

## Decision Point

Based on the checks above:

| Situation | Action |
|-----------|--------|
| Active Plan Mode session exists | Resume Plan Mode |
| Pending tasks in execution queue | Resume Execute Mode |
| Pending escalations | Enter Plan Mode to resolve |
| Fresh start | Begin with Plan Mode if new initiative |

## Common Issues

| Issue | Solution |
|-------|----------|
| Context overload warning | Run `python kernel/context_hygiene.py checkpoint save` first |
| Plan Mode interrupted mid-phase | Use `python kernel/plan_mode_phases.py resume` to see state |
| Stale execution queue | Check if tasks are still relevant |

## Related Playbooks
- [plan_to_execute.md](plan_to_execute.md)
- [session_end.md](session_end.md)

## Evolution Log
| Date | Change | Reason |
|------|--------|--------|
| 2025-02-01 | Initial creation | Codify session start pattern |
