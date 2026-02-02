---
task_id: "{{TASK_ID}}"
type: dev
queue: dev
branch: "feature/{{TASK_ID}}"
priority: P2
spec_ids:
  - ARCH_BLUEPRINT_MASTER
  - TASK_STATE_MACHINE
  - PAIR_PROGRAMMING_STANDARD

# Pair Programming Configuration
review_config:
  required: true
  required_personas:
    - security_expert
    - architecture_expert
  optional_personas:
    - performance_expert
    - domain_expert
  min_coverage_pct: 80
  max_revision_cycles: 3

verification:
  - "Code passes all four review dimensions (Q, R, C, O)"
  - "No CRITICAL or MAJOR issues remain"
  - "Requirements coverage >= 80%"
---

# Task {{TASK_ID}}

## Summary
Describe the task intent. Be specific about what needs to be implemented.

## Requirements

### Functional Requirements
- [ ] FR-01: Describe functional requirement 1
- [ ] FR-02: Describe functional requirement 2

### Non-Functional Requirements
- [ ] NFR-01: Performance requirement
- [ ] NFR-02: Security requirement

## Acceptance Criteria
1. [ ] AC-01: Specific, testable acceptance criterion
2. [ ] AC-02: Another acceptance criterion
3. [ ] AC-03: All unit tests pass
4. [ ] AC-04: Code review approved

## Implementation Notes
- Design considerations
- Dependencies
- Constraints

## Verification (Post-Review)
- [ ] Quality Check (Q-Check) passed
- [ ] Requirements Check (R-Check) passed
- [ ] Completeness Check (C-Check) passed
- [ ] Optimization Check (O-Check) reviewed

---

## Review History

| Revision | Date | Reviewer | Verdict | Notes |
|----------|------|----------|---------|-------|
| 1 | - | - | - | Initial submission |

