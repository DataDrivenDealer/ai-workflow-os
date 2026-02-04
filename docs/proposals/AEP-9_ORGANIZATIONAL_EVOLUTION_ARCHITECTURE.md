# AEP-9: Organizational Evolution Architecture

**Proposal ID**: AEP-9  
**Title**: Organizational Evolution Architecture — Toward Institutional-Scale AI Workflow OS  
**Status**: Draft  
**Created**: 2026-02-04  
**Author**: Copilot (Architecture Agent)  
**Type**: Major Architecture Evolution  
**Affects**: All Layers (Kernel, Adapter, Project, Experiment) + New Organization Layer

---

## 0. Executive Summary

This proposal represents a **paradigm shift** from project-centric to organization-centric architecture, addressing fundamental scalability, governance, and evolutionary tensions discovered through systematic analysis.

**Key Innovations:**
1. **Organization Layer (L-1)**: New layer above Kernel for multi-project coordination
2. **Compositional Rule Expression Language (REL)**: Move from natural language rules to structured, parameterized governance
3. **Fast-Track Evolution Circuit**: Parallel A/B testing for rapid iteration without compromising safety
4. **Trust Transfer Protocol**: Cross-project agent trust portability for institutional deployment
5. **Meta-Evolution Monitoring**: Self-referential system health metrics to detect blind spots
6. **Capability Maturity Model**: Formal progression framework for system evolution assessment

---

## 1. Problem Statement: Identified Tensions

### 1.1 Scalability Tensions (Category A)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| A1 | Single-project scope | Kernel binds to one project at a time | Cannot manage portfolio of research projects |
| A2 | State machine concurrency | Undefined locking semantics | Race conditions in multi-agent deployment |
| A3 | Cross-project references | No runtime validation | Dependency breaks silently |

### 1.2 Trust & Authority Tensions (Category B)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| B1 | Human approval bottleneck | Every project restarts from Level 0 | Blocks institutional scale-out |
| B2 | No trust transfer | Agent reputation isolated per-project | Knowledge loss across projects |
| B3 | Static authority levels | Promotion criteria fixed | Cannot adapt to different risk profiles |

### 1.3 Evolution Tensions (Category C)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| C1 | Self-referential measurement | Friction signals may create blind spots | Evolution success unmeasurable |
| C2 | Static rule priority | P4>P3>P2>P1 fixed | Cannot adapt to contextual needs |
| C3 | Slow feedback loop | 37+ day minimum cycle | Too slow for quantitative research |

### 1.4 Governance Tensions (Category D)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| D1 | Static gate thresholds | Fixed values in YAML | Cannot adapt to market regimes |
| D2 | Natural language rules | Cannot be formally verified | Interpretation drift over time |
| D3 | No conditional governance | Rules apply uniformly | Over-constrains low-risk scenarios |

### 1.5 Expressiveness Tensions (Category E)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| E1 | Non-compositional rules | R1-R6 are atomic | Cannot express conditional or parameterized rules |
| E2 | Implicit skill dependencies | Skills reference each other informally | No formal skill algebra |
| E3 | Flat namespace | All skills at same level | Cannot express skill hierarchy |

---

## 2. Proposed Architecture

### 2.1 5-Layer Architecture (Introducing Organization Layer)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORGANIZATION LAYER (L-1)                              │
│  - Multi-project coordination          - Cross-project trust transfer        │
│  - Portfolio governance rules           - Institutional authority config      │
│  Artifacts: org_config.yaml, trust_registry.yaml, portfolio_governance.yaml │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KERNEL LAYER (L0)                                   │
│  - Platform invariants (unchanged)      - Rule Expression Engine             │
│  - Compositional rule evaluation        - Meta-evolution monitoring          │
│  Artifacts: copilot-instructions.md, state_machine.yaml, rules/*.rel         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ADAPTER LAYER (L1)                                  │
│  - Project-to-Kernel binding (unchanged)                                     │
│  - Now also binds to Organization layer                                      │
│  Artifacts: adapter.yaml (extended with org_binding section)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROJECT LAYER (L2)                                  │
│  - Domain-specific configuration (unchanged)                                 │
│  - Can reference organization-level thresholds                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT LAYER (L3)                                │
│  - Individual experiment instances (unchanged)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Rule Expression Language (REL) Specification

**Purpose**: Replace natural language rules with structured, formally verifiable expressions.

```yaml
# Example: R2 in REL format
rule:
  id: "R2"
  version: "2.0.0"
  name: "task_serialization"
  expression: |
    WHEN agent.action.type == "start_task"
    AND context.concurrent_tasks_by(agent.id, agent.project_id).count >= threshold.max_parallel
    THEN BLOCK with "WIP limit exceeded"
  
  parameters:
    - name: threshold.max_parallel
      type: integer
      default: 1
      min: 1
      max: 10
      source: "adapter.behavior.max_parallel_tasks"
  
  conditions:
    - name: "multi_agent_exception"
      when: "org.deployment_mode == 'institutional'"
      override:
        threshold.max_parallel: 3
    
    - name: "hotfix_exception"
      when: "task.type == 'hotfix' AND task.priority == 'P0'"
      action: ALLOW
      requires_audit: true
  
  metadata:
    category: "flow_control"
    severity: "P2"
    rationale: "Prevent context switching overhead while allowing institutional scale"
    verifiable: true
    test_file: "kernel/tests/test_r2_rule.py"
```

### 2.3 Organization Layer Schema

```yaml
# configs/organization.yaml
version: "1.0.0"

organization:
  id: "quant_research_org"
  name: "Quantitative Research Organization"
  
  # Multi-project portfolio
  portfolio:
    projects:
      - id: "dgsf"
        status: "active"
        priority: 1
      - id: "alpha_model"
        status: "active"
        priority: 2
      - id: "risk_engine"
        status: "planned"
        priority: 3
    
    # Portfolio-level constraints
    constraints:
      max_concurrent_projects: 5
      shared_resource_quotas:
        compute_hours_daily: 100
        storage_tb: 10
  
  # Trust transfer registry
  trust:
    transfer_policy: "portable_with_decay"
    decay_rate: 0.9  # Per 30 days of inactivity
    minimum_level_for_transfer: 1
    
    registry_path: "state/trust_registry.yaml"
  
  # Organization-level governance
  governance:
    # Override project thresholds for portfolio consistency
    threshold_policy: "inherit_or_stricter"
    
    # Organization-level rules (O-prefixed)
    rules:
      O1:
        description: "Cross-project data access requires lineage declaration"
        applies_to: "projects/**/data/**"
      
      O2:
        description: "Shared resources require quota reservation"
        applies_to: "all_projects"
  
  # Meta-evolution monitoring
  meta_evolution:
    enabled: true
    dashboard_path: "reports/org_evolution_health.md"
    alert_channel: "governance-alerts"
```

### 2.4 Trust Transfer Protocol

```yaml
# state/trust_registry.yaml
version: "1.0.0"

agents:
  - agent_id: "copilot_main_01"
    trust_records:
      - project_id: "dgsf"
        authority_level: 2
        achieved_at: "2026-01-15T10:00:00Z"
        evidence:
          - task_success_rate: 0.96
          - tasks_completed: 47
          - violations: []
        
      - project_id: "alpha_model"
        authority_level: 1  # Transferred from dgsf with decay
        transferred_from: "dgsf"
        transferred_at: "2026-02-01T00:00:00Z"
        decay_applied: 0.9
        initial_trust_score: 0.864  # 0.96 * 0.9
    
    # Aggregate trust score for new projects
    portable_trust:
      level: 1
      confidence: 0.85
      valid_until: "2026-03-01T00:00:00Z"
```

### 2.5 Fast-Track Evolution Circuit

```yaml
# configs/fast_track_evolution.yaml
version: "1.0.0"

fast_track:
  enabled: true
  
  eligibility:
    # Evolutions that can use fast track
    allowed_types:
      - "threshold_adjustment"
      - "parameter_tuning"
      - "cosmetic_rule_change"
    
    # Must not affect these
    excluded_scope:
      - "L0_canon"
      - "authority_boundaries"
      - "data_protection"
  
  protocol:
    # A/B testing window (much shorter than standard 37 days)
    test_window: "7d"
    
    # Automatic rollback triggers
    rollback_triggers:
      - metric: "rule_violation_rate"
        threshold: 0.05  # 5% increase
      - metric: "task_completion_rate"
        threshold: -0.1  # 10% decrease
    
    # Success criteria for graduation
    graduation_criteria:
      confidence: 0.95
      min_observations: 50
    
    # On graduation, move to standard evolution track
    graduation_action: "merge_to_evolution_policy"
  
  audit:
    required: true
    output: "reports/fast_track_experiments.yaml"
```

### 2.6 Meta-Evolution Monitoring

```yaml
# configs/meta_evolution_monitoring.yaml
version: "1.0.0"

monitoring:
  # Self-referential health metrics
  metrics:
    signal_velocity:
      description: "Time from signal to action"
      target: "< 14 days"
      alert: "> 30 days"
      measurement: |
        avg(signal.actioned_at - signal.created_at) for signals where status == 'actioned'
    
    blind_spot_detection:
      description: "Code areas without evolution coverage"
      method: "correlation_analysis"
      input:
        - code_change_frequency: "git log --numstat"
        - evolution_signals: "evolution_signals.yaml"
      alert: "Areas with high change frequency but low signal rate"
    
    signal_quality:
      description: "Ratio of actionable to noise signals"
      target: "> 0.6"
      alert: "< 0.4"
      calculation: "count(status='actioned') / count(status='dismissed')"
    
    evolution_effectiveness:
      description: "Do evolutions reduce friction?"
      method: "before_after_comparison"
      window: "30d"
      target: "friction_rate decrease > 20%"
    
    regression_rate:
      description: "Evolutions that caused regressions"
      target: "< 5%"
      alert: "> 10%"
  
  # Confidence scoring for proposals
  proposal_confidence:
    very_high:
      min_signals: 5
      min_agreement: 0.9
      recurrence_days: 30
      action: "auto_propose_to_human"
    
    high:
      min_signals: 3
      min_agreement: 0.7
      recurrence_days: 14
      action: "queue_for_review"
    
    medium:
      min_signals: 2
      min_agreement: 0.5
      recurrence_days: 7
      action: "aggregate_only"
    
    low:
      min_signals: 1
      action: "log_only"
```

---

## 3. Capability Maturity Model (CMM)

**Purpose**: Provide a progression framework for assessing and planning system evolution.

### 3.1 Maturity Levels

| Level | Name | Characteristics | Indicators |
|-------|------|-----------------|------------|
| **CMM-1** | Initial | Ad-hoc governance, natural language rules | Rules in prose, manual verification |
| **CMM-2** | Managed | Structured rules, basic evolution tracking | YAML configs, evolution_signal.py |
| **CMM-3** | Defined | Compositional rules, organization layer | REL expressions, trust transfer |
| **CMM-4** | Quantified | Meta-evolution monitoring, confidence scoring | Blind spot detection, regression tracking |
| **CMM-5** | Optimizing | Self-improving evolution, predictive governance | Auto-proposal, A/B testing at scale |

### 3.2 Current Assessment

| Dimension | Current Level | Target Level (AEP-9) |
|-----------|---------------|---------------------|
| Rule Expressiveness | CMM-1 | CMM-3 |
| Evolution Tracking | CMM-2 | CMM-4 |
| Multi-Project Coordination | CMM-1 | CMM-3 |
| Trust Management | CMM-1 | CMM-3 |
| Self-Monitoring | CMM-1 | CMM-4 |

---

## 4. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create Organization Layer spec (`specs/canon/ORGANIZATION_CANON.md`)
- [ ] Extend meta_model.yaml with L-1 layer
- [ ] Define REL syntax and parser skeleton

### Phase 2: Rule Expression (Weeks 3-4)
- [ ] Convert R1-R6 to REL format
- [ ] Implement REL evaluation engine in kernel
- [ ] Add REL validation to pre-commit hooks

### Phase 3: Trust & Coordination (Weeks 5-6)
- [ ] Implement trust_registry.yaml schema
- [ ] Add trust transfer protocol to kernel
- [ ] Create organization.yaml configuration

### Phase 4: Meta-Evolution (Weeks 7-8)
- [ ] Implement blind spot detection
- [ ] Add confidence scoring to evolution_signal.py
- [ ] Create fast-track evolution circuit

### Phase 5: Integration (Weeks 9-10)
- [ ] Update copilot-instructions.md with new architecture
- [ ] Update all prompts for compositional rules
- [ ] Comprehensive testing and validation

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| REL complexity increases onboarding cost | Medium | Medium | Provide natural language aliases |
| Trust transfer creates false confidence | Low | High | Decay rate + project-specific recertification |
| Fast-track evolution introduces instability | Medium | Medium | Strict rollback triggers + scope limits |
| Meta-evolution creates overhead | Low | Low | Lazy evaluation + optional enablement |

---

## 6. Success Metrics

| Metric | Current | Target (Post-AEP-9) |
|--------|---------|---------------------|
| Evolution feedback cycle | 37+ days | 7-14 days (fast track) |
| Rule violation interpretation variance | Unmeasured | < 5% |
| Cross-project agent onboarding time | Full restart | 1 day (trust transfer) |
| Blind spot detection coverage | 0% | > 80% |
| Evolution proposal confidence accuracy | N/A | > 90% |

---

## 7. Appendices

### 7.1 REL Grammar Sketch (EBNF)

```ebnf
rule        ::= 'rule:' header expression parameters? conditions? metadata
header      ::= 'id:' STRING 'version:' SEMVER 'name:' STRING
expression  ::= 'expression:' when_clause then_clause
when_clause ::= 'WHEN' condition ('AND' condition)*
then_clause ::= 'THEN' action
condition   ::= path operator value
action      ::= 'BLOCK' message | 'ALLOW' | 'WARN' message | 'AUDIT'
parameters  ::= 'parameters:' parameter+
parameter   ::= 'name:' path 'type:' type 'default:' value constraints?
conditions  ::= 'conditions:' named_condition+
named_condition ::= 'name:' STRING 'when:' expr 'override:' override | 'action:' action
```

### 7.2 Related AEPs

- **AEP-1**: Kernel-Project Decoupling (foundation for adapter layer) ✅ Implemented
- **AEP-2**: Evolution Closed-loop Automation ✅ Implemented
- **AEP-4**: Tiered Authority Levels ✅ Implemented
- **AEP-6**: Meta-Evolution Monitoring (partial) 🔄 Extended in this proposal
- **AEP-7**: Organization Layer (proposed in signals) → Subsumed by AEP-9
- **AEP-8**: Rule Expression Language (proposed in signals) → Subsumed by AEP-9

---

**Approval Required From**: Platform Owner, Project Leads (all active projects)

**Estimated Effort**: 10 weeks (2 engineers)

**Breaking Changes**: 
- copilot-instructions.md structure change (major version bump)
- New required configs for organization-enabled deployments
- REL migration required for custom rules

---

*AEP-9 — Toward Institutional-Scale AI Workflow OS*
