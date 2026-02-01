# Architecture Pack Index (Frozen Draft)

> This document is the **single entry point** to the entire AI Workflow OS architecture.
> It defines **which top-level design blueprints exist**, **what each blueprint is responsible for**,
> **who should read it**, and **how often it is allowed to change**.
>
> The goal is to turn a large, evolving design discussion into a **finite, auditable, and enforceable
> architecture pack**, similar to engineering drawings in an aerospace or quantitative fund R&D organization.

---

## 0. How to Read This Pack

This architecture pack describes a **two-layer system**:

- **Layer A – AI Workflow OS (Company / Platform Layer)**
  - Governs how work is organized, executed, audited, and evolved
- **Layer B – Quant Trading System Projects (Product / Delivery Layer)**
  - Governs how individual trading systems are specified, built, validated, and released

Each blueprint below is classified as:
- **Constitutional**: slow-changing, high-authority
- **Structural**: defines system shape and boundaries
- **Operational**: defines execution and workflows
- **Project-level**: applied per quant trading system

---

## 0A. Blueprint Control Console

This section is the **control surface** for the entire architecture pack.
It answers, at any point in time:
- What exists now?
- Who owns it?
- How risky is it to change?
- What is the next gate to freeze it?

### 0A.1 Status Table

| Blueprint ID | File(s) | Type | Owner | Status | Change policy | Update cadence | Target freeze |
|---|---|---|---|---|---|---|---|
| ARCH_BLUEPRINT_MASTER | `docs/ARCH_BLUEPRINT_MASTER.mmd` | constitutional / structural | company governance | ready-to-freeze | proposal required | rare | v1.0.0 |
| SPEC_GOVERNANCE_MODEL | `docs/SPEC_GOVERNANCE_MODEL.mmd` | constitutional | company governance | ready-to-freeze | proposal required | rare | v0.1.0 |
| SPEC_REGISTRY_SCHEMA | `spec_registry.yaml`, `docs/SPEC_REGISTRY_SCHEMA.md` | structural / operational | platform engineering | active | controlled review | controlled | v0.1.0 |
| KERNEL_V0_RUNTIME_FLOW | `docs/KERNEL_V0_RUNTIME_FLOW.mmd` | operational | platform engineering | active | controlled review | iterative | v0.1.0 |
| TASK_STATE_MACHINE | `docs/TASK_STATE_MACHINE.mmd`, `kernel/state_machine.yaml` | constitutional / operational | platform engineering | active | controlled review | controlled | v0.1.0 |
| PROJECT_DELIVERY_PIPELINE | `docs/PROJECT_DELIVERY_PIPELINE.mmd` | project-level structural | project leads + governance | active | project direct + audit | iterative | v0.1.0 |
| INTERFACE_LAYER_MAP | `docs/INTERFACE_LAYER_MAP.mmd` | structural / operational | platform engineering | active | controlled review | iterative | v0.1.0 |
| SECURITY_TRUST_BOUNDARY | `docs/SECURITY_TRUST_BOUNDARY.mmd` | constitutional | security owner | ready-to-freeze | proposal required | rare | v0.1.0 |
| CO_OS_CAPABILITY_MAP | `docs/CO_OS_CAPABILITY_MAP.md` | structural | platform engineering | planned | controlled review | controlled | v0.1.0 |
| SPEC_DEPENDENCY_GRAPH | `docs/SPEC_DEPENDENCY_GRAPH.mmd` | structural | platform engineering | planned (auto-gen) | auto-generated | iterative | v0.1.0 |
| README_START_HERE | `README_START_HERE.md` | operational | platform engineering | planned | direct allowed | iterative | v0.1.0 |
| BLUEPRINT_FREEZE_RECORD | `docs/BLUEPRINT_FREEZE_RECORD.md` | constitutional | company governance | planned | proposal required | rare | v0.1.0 |
| OS_OPERATING_MODEL | `docs/OS_OPERATING_MODEL.md` | operational | company governance | planned | controlled review | controlled | v0.1.0 |
| PROJECT_PLAYBOOK | `docs/PROJECT_PLAYBOOK.md` | operational / project-level | company governance + project leads | planned | controlled review | controlled | v0.1.0 |

### 0A.2 Status Legend

- **planned**: placeholder exists in the pack, content not yet written
- **active**: content exists and is the current working reference
- **ready-to-freeze**: content is stable; next step is freeze record + consistency review
- **frozen**: change-controlled; only via proposal + migration notes

### 0A.3 Freeze Gate Checklist (for any blueprint)

To move a blueprint from **active → ready-to-freeze → frozen**:
1. Cross-consistency check against `ARCH_BLUEPRINT_MASTER`
2. Scope/permission alignment with `SPEC_GOVERNANCE_MODEL`
3. Tooling alignment (hooks/CI) for any enforceable rule
4. Create a Freeze Record entry in `docs/BLUEPRINT_FREEZE_RECORD.md`

---

## 1. Master Architecture Blueprints (Must Exist)

### 1.1 ARCH_BLUEPRINT_MASTER
- **File**: `docs/ARCH_BLUEPRINT_MASTER.mmd`
- **Type**: Structural / Constitutional
- **Scope**: Company OS + Project Systems
- **Purpose**:
  - Provide a single, high-level view of the entire system
  - Clearly separate Company-level OS responsibilities from Project-level responsibilities
- **Key Questions Answered**:
  - What is the AI Workflow OS?
  - What is a quant project in this company?
  - How do they interact?
- **Change Policy**: Rare, requires architecture freeze record

---

### 1.2 CO_OS_CAPABILITY_MAP
- **File**: `docs/CO_OS_CAPABILITY_MAP.md`
- **Type**: Structural
- **Scope**: Company OS
- **Purpose**:
  - Enumerate all capabilities the Company OS must provide
  - Prevent missing system responsibilities
- **Includes**:
  - Orchestration
  - Governance
  - Memory
  - Runtime
  - Interfaces
  - Security
- **Change Policy**: Controlled, reviewed

---

## 2. Specification Governance & Evolution Blueprints

### 2.1 SPEC_GOVERNANCE_MODEL
- **File**: `docs/SPEC_GOVERNANCE_MODEL.mmd`
- **Type**: Constitutional
- **Scope**: Company OS + All Projects
- **Purpose**:
  - Define how specifications are layered, modified, and promoted
- **Defines**:
  - Layer 0: Canonical (Company Constitution)
  - Layer 1: Framework (Company Platform Specs)
  - Layer 2: Project Adaptive Specs
  - Allowed modification paths and escalation rules
- **Change Policy**: Extremely strict

---

### 2.2 SPEC_REGISTRY_SCHEMA
- **Files**:
  - `spec_registry.yaml`
  - `docs/SPEC_REGISTRY_SCHEMA.md`
- **Type**: Structural / Operational
- **Scope**: Company OS + Projects
- **Purpose**:
  - Treat specifications as first-class system objects
- **Defines**:
  - spec_id, scope, version
  - dependencies and consumers
  - compatibility expectations
- **Change Policy**: Controlled, versioned

---

### 2.3 SPEC_DEPENDENCY_GRAPH
- **File**: `docs/SPEC_DEPENDENCY_GRAPH.mmd`
- **Type**: Structural
- **Scope**: Company OS
- **Purpose**:
  - Visualize how spec changes propagate through the system
- **Used For**:
  - Impact analysis
  - Upgrade proposals
- **Change Policy**: Auto-generated or semi-auto

---

## 3. Kernel & Execution Blueprints (From Constitution to Running System)

### 3.1 KERNEL_V0_RUNTIME_FLOW
- **File**: `docs/KERNEL_V0_RUNTIME_FLOW.mmd`
- **Type**: Operational
- **Scope**: Company OS Kernel
- **Purpose**:
  - Show how the OS actually runs
  - Bridge design and executable behavior
- **Includes**:
  - CLI entry points
  - State loading and validation
  - Task execution lifecycle
  - Audit generation
- **Change Policy**: Iterative but controlled

---

### 3.2 TASK_STATE_MACHINE
- **Files**:
  - `kernel/state_machine.yaml`
  - `docs/TASK_STATE_MACHINE.mmd`
- **Type**: Constitutional / Operational
- **Scope**: Company OS Kernel
- **Purpose**:
  - Define legal task states and transitions
- **Defines**:
  - Task lifecycle
  - Required artifacts per transition
  - Forbidden transitions
- **Change Policy**: Strict, requires migration notes

---

## 4. Project Delivery & Quant System Blueprints

### 4.1 PROJECT_DELIVERY_PIPELINE
- **File**: `docs/PROJECT_DELIVERY_PIPELINE.mmd`
- **Type**: Project-level Structural
- **Scope**: Quant Trading System Projects
- **Purpose**:
  - Standardize how trading systems are built and validated
- **Includes**:
  - Research
  - Data versioning
  - Model & factor construction
  - Backtesting
  - Risk gates
  - Release & monitoring
- **Change Policy**: Adaptable per project, logged

---

## 5. Interface & Automation Layer Blueprints

### 5.1 INTERFACE_LAYER_MAP
- **File**: `docs/INTERFACE_LAYER_MAP.mmd`
- **Type**: Structural / Operational
- **Scope**: Company OS
- **Purpose**:
  - Clarify where hooks, skills, and MCP belong
- **Defines**:
  - hooks (pre-commit, pre-push, CI)
  - skills (tool wrappers)
  - MCP exposure boundaries
- **Change Policy**: Iterative

---

## 6. Security & Trust Boundary Blueprints

### 6.1 SECURITY_TRUST_BOUNDARY
- **File**: `docs/SECURITY_TRUST_BOUNDARY.mmd`
- **Type**: Constitutional
- **Scope**: Entire System
- **Purpose**:
  - Prevent catastrophic failures from over-privileged agents
- **Defines**:
  - Token scopes
  - Execution environments
  - Audit requirements
  - Secret handling
- **Change Policy**: Extremely strict

---

## 7. Accompanying Operating Documents (Non-Graphical)

### 7.1 README_START_HERE
- **File**: `README_START_HERE.md`
- **Purpose**: Onboarding and orientation

### 7.2 BLUEPRINT_FREEZE_RECORD
- **File**: `docs/BLUEPRINT_FREEZE_RECORD.md`
- **Purpose**: Track architectural freezes and upgrades

### 7.3 OS_OPERATING_MODEL
- **File**: `docs/OS_OPERATING_MODEL.md`
- **Purpose**: Explain how the Company OS is actually operated day-to-day

### 7.4 PROJECT_PLAYBOOK
- **File**: `docs/PROJECT_PLAYBOOK.md`
- **Purpose**: Step-by-step guide to creating and running a new quant project

---

## 8. Update & Freeze Rules (Summary)

- Not all files are equal
- Constitutional blueprints require explicit freeze records
- Structural blueprints require review
- Operational blueprints evolve but must remain backward compatible
- Project-level blueprints are flexible but auditable

---

## 9. Status

- **Current State**: Draft – Architecture Pack Assembly
- **Completed (content exists)**:
  - `docs/ARCH_BLUEPRINT_MASTER.mmd`
  - `docs/SPEC_GOVERNANCE_MODEL.mmd`
  - `spec_registry.yaml` (starter) + `docs/SPEC_REGISTRY_SCHEMA.md`
  - `docs/KERNEL_V0_RUNTIME_FLOW.mmd`
  - `docs/TASK_STATE_MACHINE.mmd` + `kernel/state_machine.yaml`
  - `docs/PROJECT_DELIVERY_PIPELINE.mmd`
  - `docs/INTERFACE_LAYER_MAP.mmd`
  - `docs/SECURITY_TRUST_BOUNDARY.mmd`

### 9.1 Next Actions

1. Create `docs/BLUEPRINT_FREEZE_RECORD.md` (so “ready-to-freeze” can become “frozen”)
2. Write `docs/CO_OS_CAPABILITY_MAP.md` (ensures no missing company-OS responsibilities)
3. Create `README_START_HERE.md` (one-command demo path)
4. Decide **Pack Version** for the first freeze (recommended: `v0.1.0` for the pack; `v1.0.0` for the master architecture diagram)

### 9.2 Definition of Done for v0.1.0 Pack Freeze

- Index control console table is accurate (files exist, owners assigned)
- All “ready-to-freeze” items have freeze records
- Hooks/CI enforce at least:
  - scope-based spec edits (L0/L1 via proposals)
  - deviation requirement for L2 overrides
  - impact analysis requirement for spec changes
- One end-to-end demo task can run through:
  - `os init` → `os task new/start/finish` → PR → CI → merge → audit

---

> This index is the **only authoritative table of contents** for the AI Workflow OS architecture.
> Any document not referenced here is considered non-canonical.
