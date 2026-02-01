# SPEC Registry Schema (v0)

This document defines the **minimum viable, machine-readable spec registry** for the two-layer architecture:

- **Layer A (Company / Platform OS)** publishes **L0 Canon** and **L1 Framework** specs
- **Layer B (Project / Product)** owns **L2 Project** specs, and may override L1 behavior via **Deviation Declarations**

The registry turns specs into **first-class objects** so the OS can enforce:
- scope-based permissions (L0/L1/L2)
- change pathways (direct edit vs proposal)
- impact analysis (who consumes what)
- auditability (every spec change is traceable)

---

## 1) Files

- **Primary registry file (machine source of truth)**
  - `spec_registry.yaml`
- **This schema & rules doc (human reference)**
  - `docs/SPEC_REGISTRY_SCHEMA.md`

> Implementation note: You can keep `spec_registry.yaml` at repo root or under `/specs/`. The OS kernel should treat this file as authoritative.

---

## 2) Core Concepts

### 2.1 Spec Scopes

- **L0 / canon**: Company constitution-level, non-negotiable invariants
- **L1 / framework**: Company platform standards and shared schemas
- **L2 / project**: Project-owned adaptive specs

### 2.2 Change Pathways (enforced by kernel + hooks/CI)

- **L2**: direct PR allowed (must update registry + audit)
- **L1**: changes only through **Framework Upgrade Proposal**
- **L0**: changes only through **Canon Amendment Proposal** (rare, strict)

### 2.3 Overrides & Deviations

A project may **override L1 behavior** only if:
- it declares a **Deviation Declaration** referencing the base spec
- it is timeboxed or reviewable
- it includes rollback + impact assessment

---

## 3) Minimal Schema (v0)

### 3.1 Top-level layout

```yaml
registry_version: 0.1
updated_at: "2026-02-01"
owners:
  company_governance:
    name: "AI Workflow OS Governance"
    contact: "(optional)"
  registry_maintainer:
    name: "(optional)"

specs:
  - spec_id: "..."
    title: "..."
    scope: "canon|framework|project"
    status: "draft|active|deprecated"
    version:
      semver: "0.1.0"
      frozen: false
    location:
      path: "specs/.../SPEC_NAME.md"
      format: "markdown"
    applies_to:
      - "company_os"
      - "project:*"  # or explicit project IDs
    owners:
      - "role:company_governance"   # or team/person string
    depends_on: []
    consumers:
      - kind: "kernel|hook|skill|pipeline|project"
        ref: "..."
    compatibility:
      level: "strict|compatible|experimental"
      notes: "..."
    change_control:
      edit_policy: "proposal_required|direct_allowed"
      proposal_type: "none|framework_upgrade|canon_amendment"
      required_artifacts:
        - "ops/audit/*"
        - "ops/decision-log/*"
    links:
      issues: []
      prs: []
      freeze_records: []

proposals:
  - proposal_id: "PROP_..."
    type: "framework_upgrade|canon_amendment"
    target_spec_id: "..."
    originating_project: "project:..."
    status: "draft|in_review|pilot|adopted|rejected"
    rationale: "..."
    migration_required: true
    impacted_specs: []
    decision_ref: "ops/decision-log/....md"

deviations:
  - deviation_id: "DEV_..."
    project: "project:..."
    base_spec_id: "..."           # usually an L1 spec
    override_spec_id: "..."       # the L2 spec implementing the override
    status: "active|expired|reverted"
    risk_level: "low|medium|high"
    review_by: "2026-03-01"
    rollback_plan_ref: "ops/..."
    impact_summary: "..."
    decision_ref: "ops/decision-log/..."
```

---

## 4) Field Definitions (v0)

### 4.1 `specs[]`

- `spec_id` *(required, unique)*
  - Stable identifier. Recommended format: `L{0|1|2}_{DOMAIN}_{NAME}`
- `scope` *(required)*
  - `canon` (L0), `framework` (L1), `project` (L2)
- `status` *(required)*
  - `draft` / `active` / `deprecated`
- `version.semver` *(required)*
  - Semantic version string
- `version.frozen` *(required)*
  - `true` if this spec is currently frozen and change-controlled
- `location.path` *(required)*
  - Repository path to the canonical spec file
- `applies_to` *(required)*
  - Targets: `company_os`, `project:*`, or `project:<id>`
- `depends_on` *(optional)*
  - List of upstream spec IDs
- `consumers[]` *(recommended)*
  - Who uses/enforces this spec. The OS uses this for impact analysis.
- `compatibility.level` *(recommended)*
  - `strict`: breaking changes are not allowed without migration
  - `compatible`: backward-compatible evolution expected
  - `experimental`: may change frequently
- `change_control.edit_policy` *(required)*
  - `proposal_required` for L0/L1; `direct_allowed` for L2
- `change_control.proposal_type` *(required)*
  - `framework_upgrade` for L1, `canon_amendment` for L0, `none` for L2

### 4.2 `proposals[]`

Used for L1/L0 evolution via Channel B (Promotion / Feedback).

- `type`
  - `framework_upgrade` or `canon_amendment`
- `target_spec_id`
  - The spec being upgraded
- `originating_project`
  - Project requesting institutional adoption
- `status`
  - `draft` → `in_review` → (`pilot`) → `adopted` | `rejected`
- `migration_required`
  - If adoption requires updates elsewhere (hooks/kernel/pipelines)

### 4.3 `deviations[]`

Used for L2 overrides of L1 via Channel A (Local Evolution).

- `base_spec_id`
  - L1 spec being overridden
- `override_spec_id`
  - L2 spec implementing the override
- `review_by`
  - Timebox or review deadline. Prevents silent drift.

---

## 5) Required Enforcement Rules (Kernel + Hooks/CI)

These are **non-optional** if you want “spec-driven development” to be real.

1. **Every spec file must appear in `spec_registry.yaml`**
2. **`scope` implies edit policy**
   - L0/L1: direct edits blocked unless an approved proposal exists
   - L2: direct edits allowed, but audit + registry update required
3. **Spec changes require impact analysis**
   - At minimum: list affected consumers (kernel/hooks/skills/pipelines/projects)
4. **Overrides require deviation declarations**
   - Any L2 override of L1 behavior must reference `deviations[]`
5. **All merges write audit & decision refs**
   - PR merge must append/produce entries under `/ops/audit` and `/ops/decision-log`

---

## 6) Minimal Example Registry (copy/paste starter)

Below is a small starter `spec_registry.yaml` you can paste into your repo and iterate.

```yaml
registry_version: 0.1
updated_at: "2026-02-01"
owners:
  company_governance:
    name: "AI Workflow OS Governance"
  registry_maintainer:
    name: "Alan"

specs:
  - spec_id: "L0_GOV_CANON_RULES"
    title: "Company Canon Rules"
    scope: "canon"
    status: "active"
    version: { semver: "1.0.0", frozen: true }
    location: { path: "specs/company/L0_GOV_CANON_RULES.md", format: "markdown" }
    applies_to: ["company_os", "project:*"]
    owners: ["role:company_governance"]
    depends_on: []
    consumers:
      - { kind: "hook", ref: "hooks/pre-commit" }
      - { kind: "hook", ref: ".github/workflows/ci.yml" }
    compatibility: { level: "strict", notes: "Defines non-negotiable invariants" }
    change_control:
      edit_policy: "proposal_required"
      proposal_type: "canon_amendment"
      required_artifacts: ["ops/audit/*", "ops/decision-log/*", "ops/freeze/*"]
    links: { issues: [], prs: [], freeze_records: [] }

  - spec_id: "L1_KERNEL_TASK_STATE_MACHINE"
    title: "Task State Machine"
    scope: "framework"
    status: "active"
    version: { semver: "0.1.0", frozen: false }
    location: { path: "kernel/state_machine.yaml", format: "yaml" }
    applies_to: ["company_os", "project:*"]
    owners: ["role:company_governance"]
    depends_on: ["L0_GOV_CANON_RULES"]
    consumers:
      - { kind: "kernel", ref: "kernel/os.py" }
      - { kind: "hook", ref: "hooks/pre-push" }
    compatibility: { level: "compatible", notes: "Backwards-compatible transitions preferred" }
    change_control:
      edit_policy: "proposal_required"
      proposal_type: "framework_upgrade"
      required_artifacts: ["ops/audit/*", "ops/decision-log/*"]
    links: { issues: [], prs: [], freeze_records: [] }

  - spec_id: "L2_PROJECT_ALPHAPIPELINE"
    title: "Project Alpha Delivery Pipeline"
    scope: "project"
    status: "draft"
    version: { semver: "0.1.0", frozen: false }
    location: { path: "projects/alpha/specs/L2_PROJECT_ALPHAPIPELINE.md", format: "markdown" }
    applies_to: ["project:alpha"]
    owners: ["team:project_alpha"]
    depends_on: ["L1_KERNEL_TASK_STATE_MACHINE"]
    consumers:
      - { kind: "pipeline", ref: "projects/alpha/.github/workflows/pipeline.yml" }
    compatibility: { level: "experimental", notes: "May iterate rapidly" }
    change_control:
      edit_policy: "direct_allowed"
      proposal_type: "none"
      required_artifacts: ["ops/audit/*"]
    links: { issues: [], prs: [], freeze_records: [] }

proposals: []

deviations:
  - deviation_id: "DEV_ALPHA_0001"
    project: "project:alpha"
    base_spec_id: "L1_KERNEL_TASK_STATE_MACHINE"
    override_spec_id: "L2_PROJECT_ALPHAPIPELINE"
    status: "active"
    risk_level: "low"
    review_by: "2026-03-01"
    rollback_plan_ref: "projects/alpha/ops/rollback/DEV_ALPHA_0001.md"
    impact_summary: "Alpha project needs an extra pre-review state; implemented locally as a pipeline gate without changing L1."
    decision_ref: "projects/alpha/ops/decision-log/DEV_ALPHA_0001.md"
```

---

## 7) Future Extensions (v1+)

When you outgrow v0, add:
- `tags` for filtering (kernel, governance, data, research)
- `security_classification` (public/internal/restricted)
- `test_requirements` per spec
- automated graph generation (`SPEC_DEPENDENCY_GRAPH.mmd`) from the registry

---

## 8) Definition of Done (for adopting this schema)

You can consider the SPEC registry “active” when:
- `spec_registry.yaml` exists and is referenced in `ARCHITECTURE_PACK_INDEX.md`
- CI blocks changes to L0/L1 files unless a matching proposal exists
- L2 overrides cannot merge without a deviation declaration
- every merged PR that touches specs writes an audit entry

