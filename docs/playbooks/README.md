# Living Playbooks

This directory contains executable workflow playbooks that codify common patterns
observed during OS operations. Each playbook is a living document that evolves
based on actual usage.

## What is a Living Playbook?

A **Living Playbook** is:
- A documented workflow pattern extracted from successful operations
- Executable commands and verification steps
- Updated based on real-world experience
- Referenced by skills and prompts

## Playbook Index

| Playbook | Use Case | Last Updated |
|----------|----------|--------------|
| [session_start.md](session_start.md) | Beginning a new work session | 2025-02-01 |
| [session_end.md](session_end.md) | Ending a work session cleanly | 2025-02-01 |
| [plan_to_execute.md](plan_to_execute.md) | Transitioning from Plan to Execute Mode | 2025-02-01 |
| [execute_to_plan.md](execute_to_plan.md) | Escalating from Execute back to Plan | 2025-02-01 |
| [experiment_lifecycle.md](experiment_lifecycle.md) | Running a complete experiment | 2025-02-01 |
| [spec_change.md](spec_change.md) | Making a specification change | 2025-02-01 |
| [code_review.md](code_review.md) | Pair programming review cycle | 2025-02-01 |
| [incident_response.md](incident_response.md) | Handling unexpected failures | 2025-02-01 |

## Playbook Structure

Each playbook follows this structure:

```markdown
# Playbook: {Name}

## Context
When to use this playbook

## Prerequisites
What must be true before starting

## Steps
1. Step with command
   ```bash
   command to run
   ```
   - Verification: How to verify it worked

## Verification Checklist
- [ ] Check 1
- [ ] Check 2

## Common Issues
| Issue | Solution |
|-------|----------|

## Related Playbooks
Links to related playbooks

## Evolution Log
When and why this playbook was updated
```

## Creating New Playbooks

1. Identify a repeating pattern in your workflow
2. Document it in this directory
3. Add to the index above
4. Reference in relevant skills/prompts
5. Update based on usage feedback

## Governance

- Playbooks are owned by the team
- Changes should be PR'd like code
- Include an evolution log entry for each change
