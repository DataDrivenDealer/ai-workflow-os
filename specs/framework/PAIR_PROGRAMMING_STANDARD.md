# PAIR_PROGRAMMING_STANDARD

**Spec ID**: PAIR_PROGRAMMING_STANDARD  
**Scope**: L1 (Framework)  
**Status**: Active  
**Version**: 0.1.0  
**Derived From**: ROLE_MODE_CANON, GOVERNANCE_INVARIANTS  
**Depends On**: ROLE_MODE_CANON, TASKCARD_STANDARD  

---

## 0. Purpose

This standard defines the **Pair Programming** workflow — an automated code review process that engages immediately after code generation, simulating the collaboration between two expert developers.

**Goals**:
1. **Quality Assurance** — Detect bugs, anti-patterns, and security issues
2. **Requirements Validation** — Verify code meets specified requirements
3. **Completeness Check** — Ensure all requirements are addressed
4. **Code Optimization** — Suggest improvements for elegance and efficiency

---

## 1. Reviewer Role Mode

### 1.1 Definition

The **Reviewer** Role Mode is a specialized mode for agents performing code review.

```yaml
reviewer:
  description: "Code review and quality assurance"
  permissions:
    - read_all_code_artifacts
    - read_task_requirements
    - read_spec_references
    - generate_review_report
    - propose_code_improvements
    - flag_quality_issues
    - request_revision
  prohibitions:
    - execute_code_changes_directly
    - accept_artifacts
    - freeze_artifacts
    - modify_task_scope
    - self_review  # Cannot review own code
  authority_level: none  # All review outputs are speculative
```

### 1.2 Reviewer Independence Principle

> **A reviewer MUST NOT be the same agent that produced the code.**

This ensures objective evaluation and prevents self-validation bias.

---

## 2. Review Workflow

### 2.1 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Pair Programming Workflow                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Builder Agent]              [Reviewer Agent]                      │
│        │                            │                               │
│        │ 1. Generate Code           │                               │
│        │ ──────────────────>        │                               │
│        │                            │                               │
│        │    [REVIEW_PENDING]        │                               │
│        │                            │                               │
│        │                     2. Quality Check                       │
│        │                     3. Requirements Check                  │
│        │                     4. Completeness Check                  │
│        │                     5. Optimization Check                  │
│        │                            │                               │
│        │ <──────────────────────────│                               │
│        │    6. Review Report        │                               │
│        │                            │                               │
│   [If NEEDS_REVISION]               │                               │
│        │                            │                               │
│        │ 7. Revise Code             │                               │
│        │ ──────────────────>        │                               │
│        │         (repeat 2-6)       │                               │
│        │                            │                               │
│   [If APPROVED]                     │                               │
│        │                            │                               │
│        │ ──> Task continues to 'reviewing' state                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 State Transitions

```yaml
review_states:
  - code_generated       # Code produced by builder
  - review_pending       # Awaiting reviewer assignment
  - under_review         # Reviewer actively reviewing
  - revision_requested   # Issues found, revision needed
  - review_approved      # Review passed

review_transitions:
  - from: code_generated
    to: review_pending
    trigger: builder_submits_for_review
  - from: review_pending
    to: under_review
    trigger: reviewer_assigned
  - from: under_review
    to: revision_requested
    trigger: issues_found
  - from: under_review
    to: review_approved
    trigger: all_checks_passed
  - from: revision_requested
    to: review_pending
    trigger: revision_submitted
```

---

## 3. Review Dimensions

The Pair Programming review covers **four mandatory dimensions**:

### 3.1 Quality Check (Q-Check)

**Purpose**: Detect code quality issues

**Checklist**:
| Check | Description |
|-------|-------------|
| Q-001 | Syntax correctness |
| Q-002 | Type safety / type hints |
| Q-003 | Error handling coverage |
| Q-004 | Null/undefined handling |
| Q-005 | Resource cleanup (files, connections) |
| Q-006 | Security vulnerabilities (injection, exposure) |
| Q-007 | Performance anti-patterns |
| Q-008 | Code duplication |

### 3.2 Requirements Check (R-Check)

**Purpose**: Verify code implements specified requirements

**Checklist**:
| Check | Description |
|-------|-------------|
| R-001 | Functional requirements met |
| R-002 | Input validation per spec |
| R-003 | Output format per spec |
| R-004 | Edge cases handled per spec |
| R-005 | Integration points correct |
| R-006 | API contracts honored |

### 3.3 Completeness Check (C-Check)

**Purpose**: Ensure all requirements are addressed

**Checklist**:
| Check | Description |
|-------|-------------|
| C-001 | All TaskCard requirements covered |
| C-002 | All acceptance criteria addressed |
| C-003 | All referenced specs implemented |
| C-004 | Necessary tests included |
| C-005 | Documentation updated |

### 3.4 Optimization Check (O-Check)

**Purpose**: Identify opportunities for improvement

**Checklist**:
| Check | Description |
|-------|-------------|
| O-001 | Can code be simplified? |
| O-002 | Are there more efficient algorithms? |
| O-003 | Can abstractions be improved? |
| O-004 | Is naming clear and consistent? |
| O-005 | Is code idiomatic for the language? |
| O-006 | Can complexity be reduced? |

---

## 4. Review Report Format

### 4.1 Structure

```yaml
review_report:
  report_id: "REV-{TASK_ID}-{TIMESTAMP}"
  reviewer_agent_id: string
  code_artifact_path: string
  reviewed_at: ISO8601 timestamp
  
  summary:
    verdict: APPROVED | NEEDS_REVISION | BLOCKED
    issue_count:
      critical: int
      major: int
      minor: int
      suggestion: int
  
  quality_check:
    passed: boolean
    issues: list[Issue]
  
  requirements_check:
    passed: boolean
    issues: list[Issue]
    coverage_percentage: float
  
  completeness_check:
    passed: boolean
    issues: list[Issue]
    missing_items: list[string]
  
  optimization_check:
    suggestions: list[Suggestion]
  
  revision_requests: list[RevisionRequest]
```

### 4.2 Issue Severity

| Severity | Definition | Action Required |
|----------|------------|-----------------|
| CRITICAL | Security vulnerability, data loss risk, crash | Must fix before merge |
| MAJOR | Functional bug, missing requirement | Must fix before merge |
| MINOR | Code smell, minor bug | Should fix |
| SUGGESTION | Optimization opportunity | Optional |

### 4.3 Verdict Logic

```python
def determine_verdict(issues):
    if has_critical(issues) or has_major(issues):
        return "NEEDS_REVISION"
    if has_blocker_missing(issues):
        return "BLOCKED"
    return "APPROVED"
```

---

## 5. Review Governance

### 5.1 Review Requirements

| Artifact Type | Review Required | Min Review Depth |
|---------------|-----------------|------------------|
| Kernel code | Yes | Full (Q,R,C,O) |
| Spec documents | Yes | Requirements only |
| Test code | Yes | Quality + Coverage |
| Config files | Yes | Security focus |
| Documentation | Optional | Clarity check |

### 5.2 Review Bypass (Emergency Only)

Review bypass requires:
1. Emergency declaration by Project Owner
2. Deviation record in `ops/deviations/`
3. Post-hoc review commitment

### 5.3 Review Metrics

Track and report:
- Review cycle count per task
- Issue density per code type
- Time to first review
- Revision frequency

---

## 6. Expert Reviewer Personas

The system can simulate specialized reviewer perspectives:

### 6.1 Available Personas

| Persona | Focus Area | Expertise |
|---------|------------|-----------|
| `security_expert` | Security vulnerabilities | OWASP, injection, auth |
| `performance_expert` | Performance optimization | Algorithms, caching, I/O |
| `architecture_expert` | Design patterns | SOLID, coupling, cohesion |
| `domain_expert` | Business logic | Requirements alignment |
| `testing_expert` | Test coverage | Edge cases, assertions |

### 6.2 Persona Selection

```yaml
# In TaskCard frontmatter
review_config:
  required_personas:
    - security_expert
    - architecture_expert
  optional_personas:
    - performance_expert
```

---

## 7. Integration with Task Lifecycle

### 7.1 Automatic Trigger

Pair Programming review is **automatically triggered** when:
- Builder agent completes code generation
- `task_submit_for_review` command is invoked

### 7.2 Task State Integration

```
draft -> ready -> running -> [code_generated -> review -> approved] -> reviewing -> merged
                              └─────── Pair Programming Zone ───────┘
```

### 7.3 MCP Tool Integration

New MCP tools for Pair Programming:
- `review_submit` — Submit code for review
- `review_get_status` — Check review status
- `review_get_report` — Get review report
- `review_respond` — Respond to review (accept/revise)

---

## 8. Implementation Checklist

```yaml
implementation:
  kernel:
    - [ ] Add reviewer role mode to state_machine.yaml
    - [ ] Create code_review.py module
    - [ ] Add review tools to mcp_server.py
    - [ ] Update task lifecycle for review states
  
  templates:
    - [ ] Create TASKCARD_REVIEW.md template
    - [ ] Create REVIEW_REPORT.md template
  
  config:
    - [ ] Add review config to gates.yaml
    - [ ] Define review thresholds
  
  docs:
    - [ ] Update OS_OPERATING_MODEL.md
    - [ ] Update PROJECT_PLAYBOOK.md
```

---

## 9. Change Control

- **Edit Policy**: Proposal required (Framework Standard Amendment)
- **Required Artifacts**: ops/proposals/*, ops/decision-log/*
- **Approval**: Project Owner or Platform Engineer
- **Impact Analysis Required**: Yes (affects code generation workflow)

---

## Appendix A: Review Prompt Templates

### A.1 Quality Review Prompt

```
You are a Senior Code Reviewer conducting a Quality Check.

CODE TO REVIEW:
{code_content}

TASK CONTEXT:
{taskcard_content}

REVIEW CHECKLIST:
- Q-001: Syntax correctness
- Q-002: Type safety
- Q-003: Error handling
- Q-004: Null/undefined handling
- Q-005: Resource cleanup
- Q-006: Security vulnerabilities
- Q-007: Performance anti-patterns
- Q-008: Code duplication

For each issue found, provide:
- Check ID
- Severity (CRITICAL/MAJOR/MINOR/SUGGESTION)
- Line number(s)
- Description
- Suggested fix
```

### A.2 Requirements Review Prompt

```
You are a Senior Code Reviewer conducting a Requirements Check.

CODE TO REVIEW:
{code_content}

REQUIREMENTS (from TaskCard):
{requirements}

REFERENCED SPECS:
{spec_content}

Verify each requirement is correctly implemented.
Flag any deviations or missing implementations.
```

### A.3 Optimization Review Prompt

```
You are a Senior Code Reviewer conducting an Optimization Review.

CODE TO REVIEW:
{code_content}

Consider:
- Can this code be simplified while maintaining functionality?
- Are there more efficient algorithms or data structures?
- Is the code idiomatic for {language}?
- Can complexity be reduced?

Provide specific, actionable suggestions with code examples.
```
