# Playbook: Session End

## Context
Use this playbook when ending a work session to ensure clean state persistence
and proper handoff for the next session.

## Prerequisites
- Current task at a stable checkpoint
- All critical work committed

## Steps

### 1. Save Context Checkpoint
```bash
python kernel/context_hygiene.py checkpoint save --notes "End of session $(date +%Y-%m-%d)"
```
- **Verification**: Checkpoint saved successfully

### 2. Update Task State
If you were working on a task:
```bash
python kernel/state_store.py update tasks <task_id> status paused
```
- **Verification**: Task marked as paused, not abandoned

### 3. Commit Any In-Progress Work
```bash
git status
git add -A
git commit -m "WIP: Session checkpoint [$(date +%Y-%m-%d)]"
```
- **Verification**: Clean working directory

### 4. Record Plan Mode Checkpoint (if active)
If in Plan Mode:
```bash
python kernel/plan_mode_phases.py checkpoint <current_phase> --notes "Session end"
```
- **Verification**: Phase progress persisted

### 5. Document Session Summary
Create a brief note of what was accomplished:
```bash
echo "## Session $(date +%Y-%m-%d)
- Completed: ...
- In progress: ...
- Next session: ...
" >> docs/state/SESSION_LOG.md
```

### 6. Verify State Files
```bash
python kernel/context_hygiene.py status
```
- **Verification**: No critical state warnings

## Verification Checklist
- [ ] Context checkpoint saved
- [ ] Task state updated
- [ ] Work committed to git
- [ ] Plan mode state persisted (if applicable)
- [ ] Session documented
- [ ] State files verified

## Common Issues

| Issue | Solution |
|-------|----------|
| Uncommitted changes in protected files | Review and commit or stash |
| Execution queue has in-progress tasks | Mark as paused with reason |
| Plan mode mid-phase | Use checkpoint command to persist |

## Related Playbooks
- [session_start.md](session_start.md)
- [plan_to_execute.md](plan_to_execute.md)

## Evolution Log
| Date | Change | Reason |
|------|--------|--------|
| 2025-02-01 | Initial creation | Codify session end pattern |
