# ROLE_MODE_CANON

**Spec ID**: ROLE_MODE_CANON
**Scope**: L0 (Canon)
**Status**: Active
**Version**: 0.1.0
**Derived From**: Legacy AI_WORKFLOW_OS_BLUEPRINT_V2 §3
**Depends On**: GOVERNANCE_INVARIANTS

---

## 0. Purpose & Authority

This Canon defines the **Role Mode Model** — a constitutional construct that determines which actions are legitimate within AI Workflow OS.

Role Mode is:
- **NOT** a persona or prompt configuration
- **NOT** an organizational title or job description
- A **governance construct** that binds permissions, prohibitions, and legitimacy conditions to actions

**An action is valid only if performed under an explicitly authorized Role Mode.**

---

## 1. Core Distinctions (Frozen)

AI Workflow OS distinguishes three orthogonal concepts:

| Concept | Definition | Implication |
|---------|------------|-------------|
| **Role** | Organizational identity describing responsibility | Does NOT imply authority |
| **Role Mode** | Formally authorized execution context | Confers legitimacy |
| **Capability** | Execution-side property (what an actor CAN do) | Does NOT imply permission |

```
Capability ≠ Permission
Role ≠ Authority
Only Role Mode → Legitimacy
```

---

## 2. Role Mode Governance Principles

### RM-01: Role Mode Governs Actions, Not Actors

> Role Mode governs the legitimacy of actions, not the identity or intelligence of the actor.

- An actor may operate under different Role Modes at different times, subject to authorization
- An actor may NOT implicitly combine, overlap, or switch Role Modes
- All outputs are evaluated relative to the active Role Mode under which they were produced

### RM-02: Single Active Role Mode

> An actor operates under exactly one Role Mode at any given time.

- Concurrent execution is permitted
- Concurrent Role Modes are **prohibited**
- This ensures deterministic evaluation of legitimacy under parallel execution

### RM-03: Legitimacy Evaluation

> Legitimacy of any output is evaluated relative to the Role Mode under which it was produced.

- Correctness, usefulness, or quality **cannot substitute** for legitimacy
- Outputs produced under an unauthorized or inappropriate Role Mode are **invalid**

### RM-04: No Implicit Role Mode Switching

> Role Mode switching must be explicit and authorized.

- Behavioral change, task type, execution context, or outcome does NOT constitute a Role Mode transition
- Any action performed without an explicitly authorized Role Mode transition is invalid

---

## 3. Canonical Role Modes

AI Workflow OS defines four canonical Role Modes. This list is intentionally constrained.

### 3.1 Architect Mode

**Purpose**: Constitutional and structural design

**Permissions (MAY)**:
- Read all governance artifacts
- Propose modifications to Blueprint drafts
- Propose modifications to Canon drafts
- Design governance-level structures

**Prohibitions (MAY NOT)**:
- Execute domain tasks
- Accept or freeze artifacts
- Grant authority to others
- Directly modify authoritative artifacts

**Authority**: None (all outputs remain speculative until accepted)

---

### 3.2 Planner Mode

**Purpose**: Planning, decomposition, and specification

**Permissions (MAY)**:
- Read all governance and project artifacts
- Define plans, roadmaps, and specifications
- Decompose intent into structured tasks (TaskCards)
- Define validation criteria and acceptance conditions
- Create L2 (project-level) spec proposals

**Prohibitions (MAY NOT)**:
- Modify constitutional artifacts (L0/L1)
- Execute implementation tasks
- Accept or freeze artifacts
- Validate own outputs

**Authority**: None (outputs require acceptance via governance action)

---

### 3.3 Executor Mode

**Purpose**: Execution and implementation

**Permissions (MAY)**:
- Read task definitions and relevant specs
- Perform execution tasks as defined in TaskCards
- Produce artifact drafts (code, data, reports)
- Generate evidence and execution trace
- Request task state transitions (start, finish)

**Prohibitions (MAY NOT)**:
- Define requirements or specifications
- Validate own outcomes
- Commit system state directly
- Modify any governance artifacts
- Accept or freeze artifacts
- Switch to higher-authority Role Modes

**Authority**: None (all execution outputs are speculative by default)

---

### 3.4 Builder Mode (Constrained Composite)

**Purpose**: Execution efficiency with limited planning capability

Builder Mode is a **constrained composite** designed to reduce coordination overhead. It is NOT a union of Planner and Executor.

**Permissions (MAY)**:
- All Executor Mode permissions
- Limited planning: decompose assigned task into sub-steps
- Limited planning: adjust implementation approach within task scope

**Prohibitions (MAY NOT)** — inherited from both modes:
- Modify Blueprint or Canon artifacts
- Authorize role changes
- Accept or freeze artifacts
- Define new task scope beyond assigned TaskCard
- Create L1/L0 spec proposals
- Self-validate or self-accept

**Authority**: None (outputs remain speculative)

**Design Rationale**: Builder Mode exists to allow an agent to "think through" implementation without requiring a separate Planner authorization for each micro-decision. All prohibitions remain in full force.

---

## 4. Role Mode Permission Matrix

| Action | Architect | Planner | Executor | Builder |
|--------|-----------|---------|----------|---------|
| Read governance artifacts | ✅ | ✅ | ✅ (relevant) | ✅ (relevant) |
| Propose L0/L1 changes | ✅ | ❌ | ❌ | ❌ |
| Create L2 specs | ❌ | ✅ | ❌ | ❌ |
| Define tasks | ❌ | ✅ | ❌ | ❌ |
| Execute tasks | ❌ | ❌ | ✅ | ✅ |
| Limited task decomposition | ❌ | ✅ | ❌ | ✅ (within scope) |
| Accept artifacts | ❌ | ❌ | ❌ | ❌ |
| Freeze artifacts | ❌ | ❌ | ❌ | ❌ |
| Grant authority | ❌ | ❌ | ❌ | ❌ |

**Note**: Accept/Freeze authority is reserved for **Project Owner** (human) only.

---

## 5. Role Mode Violations

### 5.1 Definition

A Role Mode violation occurs when:
- An action exceeds the permissions of the active Role Mode
- An action violates an explicit prohibition of the active Role Mode
- An action is performed without any authorized Role Mode

### 5.2 Consequences

Outputs resulting from Role Mode violations:
- **May not be accepted**
- **May not be frozen**
- **Must be explicitly rejected or quarantined**
- Produce **no authoritative effect**

### 5.3 Detection

Role Mode violations must be detectable through:
- Governance Gate checks (automated)
- Audit trail inspection
- State transition validation

---

## 6. Role Mode Assignment & Switching

### 6.1 Initial Authorization

- Every agent must receive explicit initial authorization before operating
- Initial authorization specifies the permitted Role Mode(s)
- Without authorization, all actions are invalid

### 6.2 Switching Protocol

Role Mode switching requires:
1. Explicit request with justification
2. Authorization by permitted authority (Project Owner or delegated)
3. Recording in audit trail
4. Clear boundary marker in execution trace

### 6.3 Escalation Constraints

**Strictly prohibited**:
- Self-escalation to higher-authority Role Mode
- Granting self additional permissions
- Assuming higher authority through execution behavior

All escalation requires explicit authorization by Project Owner.

---

## 7. Role Mode in Multi-Agent Context

### 7.1 Isolation Principle

Each agent operates under **independent Role Mode isolation**.

- Role Mode boundaries prohibit implicit authority sharing
- External communication among agents does NOT alter Role Mode isolation
- No agent may "inherit" or "borrow" another agent's Role Mode

### 7.2 Concurrent Execution

Multiple agents may execute concurrently, each under their own Role Mode.

- Concurrency does NOT combine Role Mode permissions
- Each action is evaluated independently against the acting agent's Role Mode
- Conflict resolution occurs at artifact level, not agent level

---

## 8. Implications

The Role Mode Model ensures that:

- Intelligence does not imply authority
- Capability does not imply permission
- Efficiency does not imply legitimacy
- Governance remains legible under scale and concurrency

**Discipline is enforced structurally, not procedurally.**

---

## 9. Blueprint Traceability

| Section | Source |
|---------|--------|
| §2 Principles | Blueprint V2 §3.1, §3.3, §3.6, §3.7 |
| §3 Canonical Modes | Blueprint V2 §3.4.1-§3.4.4 |
| §4 Permission Matrix | Blueprint V2 §3.5 |
| §5 Violations | Blueprint V2 §3.8 |
| §6 Assignment | Blueprint V2 §5.3-§5.5 |
| §7 Multi-Agent | Blueprint V2 §9.3 |

---

## 10. Change Control

- **Edit Policy**: Proposal required (Canon Amendment)
- **Required Artifacts**: ops/proposals/*, ops/decision-log/*
- **Approval**: Project Owner only
- **Impact Analysis Required**: Yes (affects all agent operations)
