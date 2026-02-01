# AUTHORITY_CANON

**Spec ID**: AUTHORITY_CANON
**Scope**: L0 (Canon)
**Status**: Active
**Version**: 0.1.0
**Derived From**: Legacy AI_WORKFLOW_OS_BLUEPRINT_V2 §4, §5
**Depends On**: GOVERNANCE_INVARIANTS, ROLE_MODE_CANON

---

## 0. Purpose & Authority

This Canon defines the **Authority & Permission System** — the structural foundation for how truth is established, preserved, and changed within AI Workflow OS.

Authority is:
- A **structural property** of the system, not a behavioral attribute of actors
- Does NOT arise from intelligence, capability, seniority, productivity, or execution quality
- Arises solely from explicit governance structure

---

## 1. The Authority Triad (Frozen)

AI Workflow OS enforces strict separation of three concepts:

| Concept | Definition | How Established |
|---------|------------|-----------------|
| **Authority** | What may become system truth | Acceptance + Freeze |
| **Legitimacy** | Whether an action is eligible to receive authority | Role Mode compliance |
| **Validity** | Whether an action conforms to structure/procedure | State machine + Schema |

### Non-Exchangeability Principle

```
Authority ≠ Legitimacy ≠ Validity
Validity → does NOT imply → Legitimacy
Legitimacy → does NOT imply → Authority
```

These properties are **distinct and non-exchangeable**:
- No degree of validity or legitimacy shall produce authority
- Authority shall not be inferred from any other property

---

## 2. Authority Hierarchy

AI Workflow OS enforces a strict, non-circular authority hierarchy.

From highest to lowest:

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **1. Project Owner** | Human with ultimate governance authority | Can accept, freeze, supersede |
| **2. Frozen Artifacts** | Immutable historical truth | Cannot be modified, only superseded |
| **3. Accepted Artifacts** | Current authoritative state | Active, mutable via governance |
| **4. Speculative Outputs** | Execution results | No authority until accepted |

**Authority flows downward only. Lower levels may not override higher levels.**

---

## 3. Artifact-Centric Authority

### AUTH-01: Authority Bound to Artifact State

> Authority is bound to artifact state, not to authorship, role, or capability.

An artifact's authority is determined exclusively by:
- Its classification (L0/L1/L2)
- Its acceptance status
- Its freeze status

**The producer of an artifact has no bearing on its authority.**

### AUTH-02: Acceptance as Sole Grant

> Acceptance is the sole mechanism by which authority is conferred.

Acceptance must be:
- **Explicit** — no implicit or inferred acceptance
- **Attributable** — traceable to authorized actor
- **Recorded** — persisted as artifact

An unaccepted artifact has **no authority**, regardless of correctness or utility.

### AUTH-03: Freeze as Preservation

> Freeze preserves authority across time.

A frozen artifact:
- Defines **immutable historical truth**
- May NOT be altered
- May only be superseded through explicit governance action

Freeze separates historical truth from active intent.

---

## 4. Authority States

Every artifact exists in exactly one authority state:

```
┌─────────────┐     accept      ┌─────────────┐     freeze      ┌─────────────┐
│  SPECULATIVE │ ─────────────► │   ACCEPTED   │ ─────────────► │   FROZEN    │
│  (no auth)   │                │  (active)    │                │ (immutable) │
└─────────────┘                └─────────────┘                └─────────────┘
       ▲                              │                              │
       │         reject/discard       │         supersede            │
       └──────────────────────────────┴──────────────────────────────┘
```

| State | Authority Level | Mutability | Governance Actions |
|-------|-----------------|------------|-------------------|
| Speculative | None | Freely mutable | May be discarded |
| Accepted | Active | Controlled | May be superseded, frozen |
| Frozen | Historical | Immutable | May only be superseded |

---

## 5. Permission System

### 5.1 Permission vs Authority

| Permission | Authority |
|------------|-----------|
| Allows an action to be attempted | Determines if outcome becomes truth |
| Constrained by Role Mode | Conferred by Acceptance + Freeze |
| May be revoked | Once frozen, persists |

**Permission cannot be converted into authority by execution, repetition, or success.**

### 5.2 Permission Boundaries

Permissions specify:
- Which artifact types may be **read**
- Which artifact types may be **proposed/drafted**
- Which actions are **explicitly prohibited**

Exceeding a permission boundary renders the action **invalid**.

### 5.3 Explicit Prohibitions

Explicit prohibitions are **first-class governance constraints**.

Actions that violate constitutional or role-based prohibitions:
- Are invalid by definition
- Produce no authoritative effect
- Must be rejected or quarantined

---

## 6. Authorization Events

### 6.1 Authorization as Governance Event

> Authorization is an explicit governance event, not an implicit state.

Authorization defines the Role Mode under which an actor's actions may be considered legitimate.

Authorization exists only if it is:
- Explicitly granted
- Explicitly recorded
- Attributable to an authorized authority

### 6.2 Classes of Authorization Events

| Event Type | Description |
|------------|-------------|
| Initial Authorization | Permits actor to operate under specified Role Mode |
| Role Mode Assignment | Binds actor to specific Role Mode |
| Role Mode Switching | Changes active Role Mode (requires explicit action) |
| Authorization Downgrade | Reduces permissions |
| Authorization Revocation | Terminates all permissions |

Each constitutes an **independent governance event**. No class implies or subsumes another.

### 6.3 Temporal Scope

> Authorization is temporally scoped.

- Authorization applies only within its explicitly defined validity window
- Authorization cannot retroactively legitimize prior actions
- Actions performed outside the valid authorization window are **invalid**

### 6.4 Revocation

Authorization is revocable:
- Takes effect **immediately** upon record
- Does NOT require actor consent
- Does NOT invalidate previously accepted artifacts
- Subsequent actions under revoked authorization are **invalid**

---

## 7. Authority Constraints

### AUTH-04: No Authority Accumulation

> Authority does not accumulate through execution success, repetition, reliability, performance, or volume.

No execution system, agent, or actor acquires authority by producing correct, useful, repeated, or superior outcomes.

### AUTH-05: No Authority Amplification

> Concurrency does not amplify authority.

Parallel execution, multiple agents, or repeated submissions do NOT increase authority.
For any artifact lineage, authority exists **only** through explicit acceptance and freeze.

### AUTH-06: Authority Persistence

> Authority, once granted and frozen, persists independently of subsequent execution outcomes.

Conflicting proposals, alternative executions, or later results do NOT weaken, override, or erode existing authority.
Authority change occurs **only** through explicit supersession governed by higher authority.

### AUTH-07: No Implicit Authority Transitions

> Authority transitions must be explicit.

Authority shall NOT be inferred from:
- Workflow position
- Execution stage completion
- Validation outcome
- Elapsed time
- Operational convenience

---

## 8. Conflict Resolution

In the event of conflict:

| Rule | Priority |
|------|----------|
| Higher authority overrides lower authority | 1 |
| Frozen artifacts override active drafts | 2 |
| Explicit governance overrides implicit assumption | 3 |

**Conflicts may not be resolved implicitly.**

---

## 9. Auditability

### 9.1 Audit Requirement

All authority-conferring actions must be auditable:
- **Explicit** — clearly recorded
- **Attributable** — traceable to actor
- **Traceable** — across time

### 9.2 Audit Scope

Audit scope is limited to:
- Governance artifacts
- Governed state transitions
- Authorization events

Execution internals are NOT authoritative audit material unless externalized as artifacts.

---

## 10. Implications

This system ensures that:

- Authority cannot drift through automation or intelligence
- Permissions constrain behavior without granting power
- Historical truth remains stable under scale and concurrency
- Governance remains legible, enforceable, and reversible

**Authority is exercised deliberately, sparingly, and visibly.**

---

## 11. Blueprint Traceability

| Section | Source |
|---------|--------|
| §1 Authority Triad | Blueprint V2 §2.1 (terminology) |
| §2 Hierarchy | Blueprint V2 §4.2 |
| §3 Artifact-Centric | Blueprint V2 §4.3-§4.5 |
| §4 Authority States | Blueprint V2 §4.4-§4.5 |
| §5 Permission System | Blueprint V2 §4.6-§4.8 |
| §6 Authorization Events | Blueprint V2 §5.1-§5.7 |
| §7 Constraints | Blueprint V2 §4.9-§4.10 |
| §8 Conflict Resolution | Blueprint V2 §4.12 |
| §9 Auditability | Blueprint V2 §4.11 |

---

## 12. Change Control

- **Edit Policy**: Proposal required (Canon Amendment)
- **Required Artifacts**: ops/proposals/*, ops/decision-log/*
- **Approval**: Project Owner only
- **Impact Analysis Required**: Yes (affects all governance operations)
