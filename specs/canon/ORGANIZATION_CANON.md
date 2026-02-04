# ORGANIZATION_CANON

**Spec ID**: ORGANIZATION_CANON  
**Scope**: L0 (Canon)  
**Status**: Active  
**Version**: 1.0.0  
**Derived From**: AEP-9 Organizational Evolution Architecture  
**Depends On**: GOVERNANCE_INVARIANTS, MULTI_AGENT_CANON, AUTHORITY_CANON

---

## 0. Purpose & Authority

This Canon defines the **Organization Layer (L-1)** — the constitutional framework enabling multi-project coordination, trust transfer, and portfolio governance within AI Workflow OS.

This Canon addresses:
- How multiple projects coexist within an organization
- How trust and authority transfer across project boundaries
- How portfolio-level constraints are enforced
- How organization-level governance interacts with kernel and project layers

**The Organization Layer is optional. Systems operating without it fall back to single-project mode as defined in GOVERNANCE_INVARIANTS.**

---

## 1. Foundational Definitions

### ORG-01: Organization as Governance Scope

> An **Organization** is a bounded governance scope that may contain one or more Projects.

Organization defines:
- Identity and ownership
- Portfolio membership
- Cross-project policies
- Trust transfer rules
- Resource quotas

### ORG-02: Layer Hierarchy

```
Organization (L-1) ─┬─ Optional, enables multi-project mode
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 Project A       Project B       Project C
 (L2 via L1)     (L2 via L1)     (L2 via L1)
```

> The Organization Layer sits above the Kernel Layer (L0) in the conceptual hierarchy but does NOT override kernel invariants.

**Layer Precedence**: Kernel invariants > Organization policies > Project rules > Experiment configs

### ORG-03: Organization Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Organization Config | `configs/organization.yaml` | Identity, portfolio, policies |
| Trust Registry | `state/trust_registry.yaml` | Agent trust records across projects |
| Portfolio State | `state/portfolio.yaml` | Cross-project resource tracking |
| Organization Rules | `configs/rules/org_*.rel` | Organization-level REL rules |

---

## 2. Portfolio Governance

### ORG-04: Portfolio Membership

> Projects are members of an Organization's portfolio through explicit declaration.

```yaml
# configs/organization.yaml
portfolio:
  projects:
    - id: "dgsf"
      status: "active"      # active | paused | archived
      priority: 1           # Lower = higher priority
      quota_weight: 0.4     # Share of organization resources
```

A project may belong to at most ONE organization at a time.

### ORG-05: Portfolio Constraints

> Organization may define constraints that apply across all member projects.

Constraint types:
- **Resource quotas**: Compute, storage, API calls
- **Concurrency limits**: Maximum parallel tasks across portfolio
- **Dependency rules**: Cross-project reference policies
- **Compliance requirements**: Audit, retention, security

```yaml
constraints:
  max_concurrent_projects: 5
  shared_resource_quotas:
    compute_hours_daily: 100
    storage_tb: 10
  compliance:
    audit_retention_days: 365
    pii_handling: "prohibited"
```

### ORG-06: Inter-Project References

> Cross-project data access requires explicit lineage declaration.

When Project A references data from Project B:
1. Reference must be declared in experiment's `lineage.yaml`
2. Project B must allow external access in its `adapter.yaml`
3. Organization policy must permit the reference pattern

```yaml
# In experiment lineage.yaml
external_references:
  - source_project: "alpha_model"
    artifact: "data/processed/factors.parquet"
    access_granted: "2026-01-15"
    purpose: "Factor universe cross-validation"
```

---

## 3. Trust Transfer Protocol

### ORG-07: Trust Portability

> Agent trust earned in one project may be partially transferred to other projects within the same organization.

Trust is NOT:
- Automatically granted
- Transferred at full strength
- Applicable across organizations

### ORG-08: Trust Decay

> Transferred trust decays over time and must be reinforced through project-specific performance.

```yaml
trust:
  transfer_policy: "portable_with_decay"
  decay_rate: 0.9              # Per 30 days of inactivity
  minimum_level_for_transfer: 1 # Must be Level 1+ to transfer
  max_transfer_level: 2        # Cannot transfer Level 3+
```

Decay calculation:
```
transferred_trust = source_trust * decay_rate^(days_since_last_activity / 30)
```

### ORG-09: Trust Registry Schema

```yaml
# state/trust_registry.yaml
agents:
  - agent_id: "copilot_main_01"
    trust_records:
      - project_id: "dgsf"
        authority_level: 2
        achieved_at: "2026-01-15T10:00:00Z"
        evidence:
          task_success_rate: 0.96
          tasks_completed: 47
          violations: []
        
      - project_id: "alpha_model"
        authority_level: 1
        transferred_from: "dgsf"
        transferred_at: "2026-02-01T00:00:00Z"
        decay_applied: 0.9
```

### ORG-10: Trust Boundaries

> Trust transfer respects hard boundaries that cannot be crossed.

Hard boundaries:
- Canon (L0) modifications: Always require human approval regardless of trust
- Data protection (R4): No trust level exempts from data protection rules
- Authority claims: No agent may claim authority regardless of trust level

Soft boundaries (trust-dependent):
- WIP limits: Higher trust may allow more parallel tasks
- Review requirements: Higher trust may allow self-merge to feature branches
- Audit frequency: Lower trust requires more frequent audit checkpoints

---

## 4. Organization-Level Rules

### ORG-11: Rule Namespace

> Organization-level rules use the `O` prefix to distinguish from kernel rules (`R`) and project rules (`P`).

| Namespace | Scope | Examples |
|-----------|-------|----------|
| R1-R9 | Kernel (global) | R4: Data protection |
| O1-O9 | Organization | O1: Cross-project lineage |
| P1-P9 | Project | P1: Point-in-time correctness |
| E1-E9 | Experiment | E1: Experiment naming |

### ORG-12: Rule Inheritance

```
Kernel Rules (R)
      │
      ▼ (inherit all)
Organization Rules (O)
      │
      ▼ (inherit all, may add)
Project Rules (P)
      │
      ▼ (inherit all, may add)
Experiment Rules (E)
```

> Lower layers inherit all rules from upper layers. They may ADD rules but NOT remove or weaken inherited rules.

### ORG-13: Rule Override Protocol

> Rules may be parameterized but NOT overridden in violation of their intent.

Valid override:
```yaml
# In organization.yaml
rule_parameters:
  R2:
    max_parallel: 3  # Increase from default of 1
```

Invalid override (PROHIBITED):
```yaml
# ILLEGAL - Would disable data protection
rule_parameters:
  R4:
    enabled: false
```

---

## 5. Deployment Modes

### ORG-14: Deployment Mode Declaration

> Organization declares its deployment mode to enable appropriate governance scaling.

| Mode | Description | Typical Scale |
|------|-------------|---------------|
| `individual` | Single researcher, single project | 1 project, 1 agent |
| `team` | Small team, few projects | 2-5 projects, 1-5 agents |
| `institutional` | Large organization, many projects | 5+ projects, 5-20 agents |

```yaml
organization:
  deployment_mode: "institutional"
```

### ORG-15: Mode-Specific Defaults

Each mode has different default behaviors:

| Behavior | Individual | Team | Institutional |
|----------|------------|------|---------------|
| `R2.max_parallel` | 1 | 2 | 3 |
| Trust transfer | Disabled | Manual | Automatic |
| Audit frequency | On-demand | Daily | Continuous |
| Review requirements | Relaxed | Standard | Strict |

---

## 6. Meta-Evolution at Organization Level

### ORG-16: Organization-Level Evolution Signals

> Evolution signals are aggregated at both project and organization levels.

```yaml
meta_evolution:
  enabled: true
  aggregation_scope: "organization"
  dashboard_path: "reports/org_evolution_health.md"
  
  thresholds:
    signal_velocity_days: 14
    blind_spot_alert: true
    cross_project_pattern_detection: true
```

### ORG-17: Cross-Project Pattern Detection

> The organization layer may detect patterns that span multiple projects.

Pattern types:
- Repeated friction with same rule across projects → Rule evolution candidate
- Similar missing skills across projects → Shared skill development
- Correlated failures across projects → Systemic issue detection

---

## 7. Governance Interactions

### ORG-18: Human Authority Preservation

> Organization layer does NOT diminish human authority requirements for governance actions.

Actions always requiring human approval:
- Canon modifications (any layer)
- Trust level promotion beyond Level 2
- Organization configuration changes
- New project addition to portfolio

### ORG-19: Audit Trail Continuity

> All organization-level actions are audited with the same rigor as project actions.

Audit must include:
- Actor (human or agent with ID)
- Action type
- Affected scope (organization, project, experiment)
- Timestamp
- Outcome

---

## 8. Implementation Requirements

### ORG-20: Graceful Degradation

> Systems without organization configuration must operate correctly in single-project mode.

When `configs/organization.yaml` is absent or disabled:
- Trust transfer is disabled
- Portfolio constraints do not apply
- Organization rules (O*) are not loaded
- Kernel and project rules remain fully enforced

### ORG-21: Configuration Validation

> Organization configuration must be validated before system initialization.

Validation checks:
1. All referenced projects exist and have valid adapters
2. Resource quotas sum to <= 100%
3. Trust decay rate is in valid range (0.5 - 1.0)
4. Rule parameters do not violate kernel invariants

---

## 9. Change Control

| Change Type | Required Approval | Artifacts |
|-------------|-------------------|-----------|
| Organization creation | Platform Owner | organization.yaml, audit record |
| Project addition | Org Admin + Project Owner | organization.yaml update |
| Trust policy change | Org Admin | organization.yaml, decision record |
| Resource quota change | Org Admin | organization.yaml, impact analysis |

---

## 10. Compatibility

| Component | Compatibility |
|-----------|--------------|
| Kernel v7.0+ | Required |
| Adapter v1.0+ | Required |
| Project Interface v1.0+ | Required |
| MCP Server | Optional (enhances but not required) |

---

## Appendix A: Organization Config Reference

```yaml
# configs/organization.yaml - Full Schema Reference
version: "1.0.0"

organization:
  id: string                    # Unique identifier
  name: string                  # Human-readable name
  deployment_mode: enum         # individual | team | institutional
  
  portfolio:
    projects:
      - id: string
        status: enum            # active | paused | archived
        priority: integer       # Lower = higher priority
        quota_weight: float     # 0.0 - 1.0
    
    constraints:
      max_concurrent_projects: integer
      shared_resource_quotas:
        compute_hours_daily: integer
        storage_tb: integer
  
  trust:
    transfer_policy: enum       # disabled | manual | portable_with_decay
    decay_rate: float           # 0.5 - 1.0
    minimum_level_for_transfer: integer
    max_transfer_level: integer
    registry_path: string
  
  governance:
    threshold_policy: enum      # inherit_or_stricter | inherit_exact
    rules: map[string, rule]
  
  meta_evolution:
    enabled: boolean
    aggregation_scope: enum     # project | organization
    dashboard_path: string
    alert_channel: string
```

---

## Appendix B: Traceability

| Section | Source Requirement |
|---------|-------------------|
| §1 Definitions | AEP-9 §2.1 |
| §2 Portfolio | AEP-9 §2.3 |
| §3 Trust Transfer | AEP-9 §2.4 |
| §4 Organization Rules | AEP-9 §2.2 |
| §5 Deployment Modes | Evolution Signal sig_2026020403 |
| §6 Meta-Evolution | AEP-9 §2.6 |

---

*ORGANIZATION_CANON v1.0.0 — Constitutional Framework for Multi-Project AI Workflow OS*
