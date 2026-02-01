---
task_id: "TASK_INFRA_0002"
type: dev
queue: dev
branch: "feature/TASK_INFRA_0002"
priority: P1
spec_ids:
  - ARCH_BLUEPRINT_MASTER
  - TASK_STATE_MACHINE
verification:
  - "Priority field in TaskCard frontmatter is parsed correctly"
  - "Tasks with higher priority are processed first in queue"
  - "Unit tests for priority scheduling pass"
  - "Documentation updated to reflect priority feature"
---

# Task TASK_INFRA_0002: Implement Priority Scheduling

## Summary

Implement a priority scheduling system for the AI Workflow OS task queue. Currently, tasks are processed in FIFO order without consideration for urgency. This task adds support for:

1. **Priority Field**: Add `priority` field to TaskCard frontmatter (P0/P1/P2/P3)
2. **Queue Sorting**: Sort task queue by priority when processing
3. **Priority Preemption**: Higher priority tasks can interrupt queue processing

## Context

According to the [CO_OS_CAPABILITY_MAP](../docs/CO_OS_CAPABILITY_MAP.md), Priority Scheduling is listed as "Planned (⏳)" under Orchestration Engine capabilities. This is identified as a P1 task in the project gap analysis.

## Implementation Notes

### 1. TaskCard Schema Update

Add `priority` field to `kernel/task_parser.py`:

```python
PRIORITY_LEVELS = ["P0", "P1", "P2", "P3"]  # P0 = highest

# Add to validation
if "priority" in fields and fields["priority"] not in PRIORITY_LEVELS:
    raise ValueError(f"Invalid priority: {fields['priority']}")
```

### 2. Queue Sorting Logic

Update `kernel/os.py` to sort tasks by priority:

```python
def get_sorted_tasks(tasks_state: Dict) -> List[str]:
    """Return task IDs sorted by priority (P0 first)."""
    tasks = tasks_state.get("tasks", {})
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    
    return sorted(
        tasks.keys(),
        key=lambda t: priority_order.get(tasks[t].get("priority", "P3"), 3)
    )
```

### 3. State Machine Update

Consider adding priority to state transitions (optional):

```yaml
# kernel/state_machine.yaml
priority_rules:
  - P0 tasks can preempt running P2/P3 tasks
  - P0 tasks require immediate human attention flag
```

## Files to Modify

| File | Change |
|------|--------|
| `kernel/task_parser.py` | Add priority validation |
| `kernel/os.py` | Add queue sorting |
| `kernel/state_machine.yaml` | Add priority rules (optional) |
| `templates/TASKCARD_TEMPLATE.md` | Add priority field |
| `kernel/tests/test_task_parser.py` | Add priority tests |
| `docs/CO_OS_CAPABILITY_MAP.md` | Update status to ✅ |

## Acceptance Criteria

- [ ] TaskCard with `priority: P0` is validated correctly
- [ ] `task list` command shows tasks sorted by priority
- [ ] Unit tests cover all priority levels
- [ ] Documentation updated

## Dependencies

- **Upstream**: None (standalone feature)
- **Downstream**: Will benefit all future task management

## Verification

1. Create test TaskCards with different priorities
2. Run `python kernel/os.py task list` and verify order
3. Run `pytest kernel/tests/ -v` - all tests pass
4. Review [CO_OS_CAPABILITY_MAP](../docs/CO_OS_CAPABILITY_MAP.md) shows Priority Scheduling as ✅

## Estimated Effort

- **Development**: 4 hours
- **Testing**: 2 hours
- **Documentation**: 1 hour
- **Total**: 7 hours

---

**Created by**: Dr. 林建筑 (System Architect)  
**Created at**: 2026-02-01  
**Decision Reference**: DEC_20260201_TEST_INFRASTRUCTURE
