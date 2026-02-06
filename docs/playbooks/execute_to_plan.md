# Playbook: Execute to Plan Escalation

## Context
Use this playbook when encountering an issue during Execute Mode that cannot
be resolved without returning to Plan Mode. This ensures proper escalation
and queue management.

## Prerequisites
- Currently in Execute Mode
- Encountered a blocking issue

## Problem Classification

First, classify the problem:

| Problem Type | Severity | Queue Impact | Mode Switch |
|--------------|----------|--------------|-------------|
| `spec_unclear` | low | None | Optional |
| `spec_error` | medium | Block current task | Recommended |
| `spec_missing` | high | Pause queue | Recommended |
| `research_needed` | medium | Block current task | Optional |
| `refactor_required` | high | Pause queue | Recommended |
| `blocker` | critical | Stop queue | Required |

## Steps

### 1. Document the Issue
```bash
# Capture what you know about the problem
echo "## Escalation: $(date +%Y-%m-%d)
Type: {problem_type}
Task: {task_id}
Description: {description}
Affected specs: {spec_paths}
" > /tmp/escalation_notes.txt
```

### 2. Create Escalation Record
```bash
python kernel/escalation.py create \
  --type {problem_type} \
  --task {task_id} \
  --description "{description}" \
  --severity {severity}
```
- **Verification**: Escalation ID returned

### 3. Update Execution Queue
Based on severity:
```bash
# For high/critical severity
python kernel/state_store.py update execution_queue metadata.paused true
python kernel/state_store.py update execution_queue metadata.paused_reason "ESC-{id}"

# For medium severity - just block current task
python kernel/state_store.py update execution_queue items[{idx}].status blocked
```

### 4. Announce Mode Switch
```
⚠️ Escalating to PLAN MODE

Escalation ID: ESC-{id}
Type: {problem_type}
Reason: {description}

Switching to Plan Mode for resolution.
```

### 5. Enter Plan Mode
Say: "开启PLAN MODE" or "PLAN MODE"

Plan Mode will automatically detect the escalation in Phase 0.5.

## Verification Checklist
- [ ] Problem classified correctly
- [ ] Escalation recorded in state
- [ ] Execution queue updated appropriately
- [ ] Mode switch announced
- [ ] Evidence preserved

## Do NOT Escalate For

These can usually be resolved in Execute Mode:
- Simple bugs (fix and retry)
- Missing imports (add and retry)
- Test failures (diagnose and fix)
- Git merge conflicts (resolve directly)

## Do Escalate For

- Spec contradictions
- Unclear acceptance criteria
- Missing design decisions
- Architectural questions
- Research requirements

## Common Issues

| Issue | Solution |
|-------|----------|
| Unsure if should escalate | Check if spec is involved; if yes, escalate |
| Multiple issues | Create separate escalations |
| Already in Plan Mode | Already correct; no action needed |

## Related Playbooks
- [plan_to_execute.md](plan_to_execute.md)
- [incident_response.md](incident_response.md)

## Evolution Log
| Date | Change | Reason |
|------|--------|--------|
| 2025-02-01 | Initial creation | Codify escalation pattern |
