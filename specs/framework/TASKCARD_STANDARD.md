# TASKCARD_STANDARD

**Spec ID**: TASKCARD_STANDARD  
**Scope**: L1 Framework  
**Version**: 1.0.0  
**Status**: Active  
**Owner**: System Architect

---

## 1. Purpose

This specification defines the canonical structure and requirements for TaskCards within AI Workflow OS. TaskCards are the fundamental unit of work tracking and governance.

---

## 2. TaskCard Structure

### 2.1 Frontmatter (Required)

```yaml
---
task_id: "TASK_TYPE_N_PROJECT_NNN"  # Required: Unique identifier
type: research | data | build | eval | deploy | ops  # Required
queue: dev | data | prod  # Required: Assignment queue
branch: "feature/TASK_ID"  # Required: Git branch
priority: P0 | P1 | P2 | P3  # Required
spec_ids:  # Required: Referenced specifications
  - ARCH_BLUEPRINT_MASTER
  - TASK_STATE_MACHINE
verification:  # Required: Exit criteria
  - "Criterion 1"
  - "Criterion 2"
---
```

### 2.2 Task ID Format

```
{TYPE}_{STAGE}_{PROJECT}_{SEQUENCE}

Examples:
- RESEARCH_0_DGSF_001  (Research Stage 0, DGSF project, #001)
- DATA_2_DGSF_001      (Data Stage 2, DGSF project, #001)
- BUILD_3_DGSF_001     (Build Stage 3, DGSF project, #001)
```

### 2.3 Priority Levels

| Priority | Label | SLA | Use Case |
|----------|-------|-----|----------|
| P0 | 🔴 Critical | < 4 hours | Production incident |
| P1 | 🟠 High | < 1 day | Blocking issues |
| P2 | 🟡 Medium | < 1 week | Normal development |
| P3 | 🟢 Low | Best effort | Nice-to-have |

---

## 3. TaskCard Lifecycle

```
draft → running → reviewing → merged → released
                ↓
              blocked (can return to running)
```

### 3.1 State Transitions

| From | To | Action | Gate |
|------|-----|--------|------|
| draft | running | `task start` | - |
| running | reviewing | `task finish` | Verification checks |
| reviewing | merged | `task merge` | Code review |
| merged | released | `task release` | G{N} gate pass |
| running | blocked | `task block` | - |
| blocked | running | `task unblock` | - |

---

## 4. Required Sections

Every TaskCard MUST include:

1. **元信息** - Metadata table
2. **目标/假设** - Objectives or hypotheses
3. **交付物** - Expected artifacts
4. **验证标准** - Verification criteria
5. **Authority声明** - Authorization scope
6. **Audit Trail** - Change history

---

## 5. Governance Integration

### 5.1 Spec References
TaskCards MUST reference applicable specs in `spec_ids`:
- Always include `ARCH_BLUEPRINT_MASTER`
- Include stage-specific specs (e.g., `PROJECT_DELIVERY_PIPELINE`)
- Include domain specs as needed

### 5.2 Verification
All items in `verification` array must be checkable:
- Boolean conditions preferred
- Quantitative thresholds when possible
- Link to evidence/artifacts

---

## 6. Examples

See templates at: `templates/TASKCARD_TEMPLATE.md`

---

*Spec maintained by: System Architect*  
*Last updated: 2026-02-01*
