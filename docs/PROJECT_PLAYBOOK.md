# PROJECT_PLAYBOOK

**Document ID**: PROJECT_PLAYBOOK  
**Type**: Operational / Project-level  
**Scope**: Quant Trading System Projects  
**Status**: Active  
**Version**: 0.1.0  
**Owner**: Company Governance + Project Leads

---

## 0. Purpose

This playbook provides a **step-by-step guide** to creating and running a new quant trading system project within AI Workflow OS.

It covers:
- Project initialization
- Stage-by-stage workflow
- Gate requirements
- Artifact management
- Handoff procedures

---

## 1. Project Lifecycle Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                    Project Delivery Pipeline                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Stage 0      Stage 1       Stage 2      Stage 3      Stage 4     │
│  ┌─────┐      ┌─────┐       ┌─────┐      ┌─────┐      ┌─────┐     │
│  │Idea │ ──>  │Rsrch│ ──>   │Data │ ──>  │Build│ ──>  │Eval │     │
│  │Triage│     │Design│      │Eng  │      │Model│      │Test │     │
│  └─────┘      └─────┘       └─────┘      └─────┘      └─────┘     │
│                               │G1│         │G2│        │G3│       │
│                                                                    │
│  Stage 5      Stage 6       Stage 7                               │
│  ┌─────┐      ┌─────┐       ┌─────┐                               │
│  │Release│──> │Ops  │ ──>   │Gov  │                               │
│  │Deploy │    │Monitor│     │Review│                              │
│  └─────┘      └─────┘       └─────┘                               │
│    │G4│        │G5│                                               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pre-Project Checklist

Before starting a new project, ensure:

```yaml
pre_project_checklist:
  governance:
    - [ ] Project sponsor identified
    - [ ] Budget/resource allocation approved
    - [ ] Risk tolerance defined
    
  technical:
    - [ ] Data sources identified
    - [ ] Infrastructure access confirmed
    - [ ] Development environment ready
    
  specs:
    - [ ] L2 project specs drafted
    - [ ] Deviation from L1/L0 declared (if any)
    - [ ] Risk limits defined
```

---

## 3. Stage-by-Stage Guide

### Stage 0: Idea Intake & Triage

**Goal**: Filter ideas and decide whether to proceed

**Inputs**:
- Research papers / practitioner notes
- Internal alpha ideas
- Post-mortem learnings

**Activities**:
1. Create `RESEARCH_0_<PROJECT>` TaskCard
2. Document hypothesis and expected edge
3. Assess data availability
4. Estimate implementation cost
5. Make GO/NO-GO decision

**TaskCard Template**: [templates/pipeline/TASKCARD_RESEARCH_0.md](../templates/pipeline/TASKCARD_RESEARCH_0.md)

**Output Artifacts**:
| Artifact | Location | Format |
|----------|----------|--------|
| Triage Report | `ops/decision-log/RESEARCH_0_<ID>_triage.md` | Markdown |
| Decision | GO / NO-GO / DEFER | Logged |

**Exit Criteria**:
- [ ] Triage report completed
- [ ] Decision logged in decision-log
- [ ] If GO: `RESEARCH_1_<ID>` TaskCard created

---

### Stage 1: Research Design (Reproducibility First)

**Goal**: Design experiment with reproducibility as first-class requirement

**Inputs**:
- Approved triage report
- Initial hypothesis

**Activities**:
1. Create `RESEARCH_1_<PROJECT>` TaskCard
2. Define hypothesis and success metrics
3. Plan experiments (ablations, baselines)
4. Draft reproducibility package (config, seeds, scripts)

**TaskCard Template**: [templates/pipeline/TASKCARD_RESEARCH_1.md](../templates/pipeline/TASKCARD_RESEARCH_1.md)

**Output Artifacts**:
| Artifact | Location | Format |
|----------|----------|--------|
| Research Design Doc | `docs/research/<ID>_design.md` | Markdown |
| Experiment Config | `configs/<ID>.yaml` | YAML |
| Repro Package Manifest | `docs/research/<ID>_repro.md` | Markdown |

**Exit Criteria**:
- [ ] Hypothesis clearly stated
- [ ] Success metrics defined
- [ ] Experiment plan documented
- [ ] Reproducibility package drafted

---

### Stage 2: Data Engineering (Versioned Data)

**Goal**: Build clean, versioned, leakage-free data pipeline

**Inputs**:
- Research design
- Data contract spec

**Activities**:
1. Create `DATA_2_<PROJECT>` TaskCard
2. Ingest raw data (vendors/APIs)
3. Clean and align (corporate actions, missing data)
4. Build feature/factor panel
5. Create immutable data snapshot
6. **Pass Gate G1: Data Quality**

**TaskCard Template**: [templates/pipeline/TASKCARD_DATA_2.md](../templates/pipeline/TASKCARD_DATA_2.md)

**Gate G1 Requirements**:
| Check | Threshold | Severity |
|-------|-----------|----------|
| Schema valid | 100% | error |
| Missing rate | <5% | error |
| No lookahead | true | error |
| Snapshot immutable | checksum verified | error |
| Survivorship handled | documented | warning |

**Output Artifacts**:
| Artifact | Location | Format |
|----------|----------|--------|
| Data Snapshot | `data/<ID>/snapshot_v1/` | Parquet |
| Checksums | `data/<ID>/checksums.yaml` | YAML |
| Data Quality Report | `reports/gates/G1_<ID>.md` | Markdown |

**Exit Criteria**:
- [ ] Gate G1 passed
- [ ] Data snapshot created with checksum
- [ ] No lookahead bias confirmed

---

### Stage 3: Model/Strategy Build

**Goal**: Construct modular, testable strategy

**Inputs**:
- Verified data snapshot
- Research design

**Activities**:
1. Create `DEV_3_<PROJECT>` TaskCard
2. Build signal module (factor transforms, inference)
3. Build portfolio construction module
4. Define execution assumptions (slippage, costs)
5. Create strategy package artifact
6. **Pass Gate G2: Sanity Checks**

**TaskCard Template**: [templates/pipeline/TASKCARD_DEV_3.md](../templates/pipeline/TASKCARD_DEV_3.md)

**Gate G2 Requirements**:
| Check | Threshold | Severity |
|-------|-----------|----------|
| Unit tests pass | 100% | error |
| No lookahead in signals | true | error |
| Cost assumptions valid | documented | error |
| Constraints satisfied | true | error |

**Output Artifacts**:
| Artifact | Location | Format |
|----------|----------|--------|
| Strategy Package | `strategies/<ID>/` | Python |
| Unit Tests | `tests/<ID>/` | pytest |
| G2 Report | `reports/gates/G2_<ID>.md` | Markdown |

**Exit Criteria**:
- [ ] Gate G2 passed
- [ ] Strategy package created
- [ ] All unit tests pass

---

### Stage 4: Evaluation & Backtesting

**Goal**: Rigorous historical validation

**Inputs**:
- Strategy package
- Data snapshot

**Activities**:
1. Create `EVAL_4_<PROJECT>` TaskCard
2. Run in-sample backtest
3. Run out-of-sample backtest
4. Stress testing (regime, drawdown)
5. Compare vs benchmarks
6. **Pass Gate G3: Performance & Robustness**

**TaskCard Template**: [templates/pipeline/TASKCARD_EVAL_4.md](../templates/pipeline/TASKCARD_EVAL_4.md)

**Gate G3 Requirements**:
| Check | Threshold | Severity |
|-------|-----------|----------|
| Sharpe ratio | >1.0 (or project spec) | error |
| Max drawdown | <20% (or project spec) | error |
| OOS degradation | <30% | warning |
| P-value (bootstrap) | <0.05 | warning |

**Output Artifacts**:
| Artifact | Location | Format |
|----------|----------|--------|
| Backtest Report | `reports/backtest/<ID>.md` | Markdown |
| Performance Metrics | `reports/metrics/<ID>.yaml` | YAML |
| G3 Report | `reports/gates/G3_<ID>.md` | Markdown |

**Exit Criteria**:
- [ ] Gate G3 passed
- [ ] Backtest report reviewed
- [ ] Risk committee approval (if required)

---

### Stage 5: Release & Deployment

**Goal**: Package and deploy to production

**Inputs**:
- Approved strategy
- All gate reports

**Activities**:
1. Create `RELEASE_5_<PROJECT>` TaskCard
2. Create release candidate bundle
3. Final compliance review
4. **Pass Gate G4: Approval**
5. Deploy to paper trading (first)
6. Deploy to live (after paper validation)

**TaskCard Template**: [templates/pipeline/TASKCARD_RELEASE_5.md](../templates/pipeline/TASKCARD_RELEASE_5.md)

**Gate G4 Requirements**:
| Check | Threshold | Severity |
|-------|-----------|----------|
| All prior gates passed | true | error |
| Risk committee sign-off | true | error |
| Compliance review | approved | error |
| Rollback plan documented | true | error |

**Output Artifacts**:
| Artifact | Location | Format |
|----------|----------|--------|
| Release Candidate | `releases/RC_<ID>/` | Bundle |
| Deployment Config | `configs/deploy/<ID>.yaml` | YAML |
| G4 Report | `reports/gates/G4_<ID>.md` | Markdown |

**Exit Criteria**:
- [ ] Gate G4 passed
- [ ] Release bundle created
- [ ] Deployed to paper trading

---

### Stage 6: Operations & Monitoring

**Goal**: Continuous monitoring and maintenance

**Inputs**:
- Live strategy
- Monitoring specs

**Activities**:
1. Create `OPS_6_<PROJECT>` TaskCard
2. Monitor P&L and risk metrics
3. Track vs backtest expectations
4. **Continuous Gate G5: Live Safety**
5. Handle incidents and anomalies

**TaskCard Template**: [templates/pipeline/TASKCARD_OPS_6.md](../templates/pipeline/TASKCARD_OPS_6.md)

**Gate G5 (Continuous)**:
| Check | Action |
|-------|--------|
| Drawdown > limit | Auto-reduce position |
| Anomaly detected | Alert + review |
| Kill-switch triggered | Halt trading |

**Output Artifacts**:
| Artifact | Location | Format |
|----------|----------|--------|
| Daily P&L Report | `reports/ops/<ID>/daily/` | CSV |
| Incident Reports | `ops/incidents/<ID>/` | Markdown |
| Monthly Review | `reports/ops/<ID>/monthly/` | Markdown |

---

### Stage 7: Governance Review

**Goal**: Periodic strategy review and lifecycle decisions

**Inputs**:
- Operations history
- Performance data

**Activities**:
1. Create `GOV_7_<PROJECT>` TaskCard
2. Quarterly performance review
3. Assess continued viability
4. Decide: CONTINUE / MODIFY / SUNSET

**TaskCard Template**: [templates/pipeline/TASKCARD_GOV_7.md](../templates/pipeline/TASKCARD_GOV_7.md)

**Output Artifacts**:
| Artifact | Location | Format |
|----------|----------|--------|
| Governance Report | `reports/gov/<ID>_Q<N>.md` | Markdown |
| Decision Record | `ops/decision-log/GOV_<ID>.md` | Markdown |

---

## 4. Gate Summary

| Gate | Stage Exit | Key Checks | Enforced By |
|------|------------|------------|-------------|
| **G1** | Stage 2 | Data quality, no leakage | `scripts/gate_check.py` |
| **G2** | Stage 3 | Tests pass, sanity checks | `scripts/gate_check.py` |
| **G3** | Stage 4 | Performance, robustness | `scripts/gate_check.py` |
| **G4** | Stage 5 | Approval, compliance | Human + script |
| **G5** | Stage 6 | Live safety (continuous) | Monitoring system |

---

## 5. Quick Reference Commands

```powershell
# Create new project task
python kernel/os.py task new RESEARCH_0_MYPROJECT

# Start task
python kernel/os.py task start RESEARCH_0_MYPROJECT

# Run gate check
python scripts/gate_check.py --gate G1 --task-id DATA_2_MYPROJECT

# Finish task
python kernel/os.py task finish DATA_2_MYPROJECT

# Check all gates
python scripts/gate_check.py --gate all --task-id EVAL_4_MYPROJECT
```

---

## 6. Project Spec Template

Create `specs/projects/<PROJECT_ID>.yaml`:

```yaml
project_id: MYPROJECT
name: "My Quant Strategy"
created_at: "2026-02-01"
owner: "project_lead_name"

# Inherits from L1
inherits:
  - specs/framework/PROJECT_PIPELINE_SPEC.yaml

# L2 Overrides (requires deviation declaration)
overrides:
  sharpe_threshold: 1.5  # stricter than L1 default
  max_drawdown: 0.15     # stricter than L1 default

# Risk Limits
risk:
  max_leverage: 2.0
  max_position_size: 0.05
  max_sector_exposure: 0.20

# Data Sources
data:
  universe: "US_EQUITY_LARGE"
  frequency: "daily"
  history_start: "2010-01-01"
```

---

## 7. Handoff Procedures

### Stage Transition Handoff

When transitioning between stages:

1. **Finish current TaskCard**
   ```powershell
   python kernel/os.py task finish <CURRENT_TASK>
   ```

2. **Pass required gate**
   ```powershell
   python scripts/gate_check.py --gate <GATE> --task-id <TASK>
   ```

3. **Create next stage TaskCard**
   ```powershell
   python kernel/os.py task new <NEXT_TASK>
   ```

4. **Link artifacts**
   - Reference previous task outputs in new TaskCard
   - Ensure data lineage is documented

### Team Handoff

When handing off to another team member:

1. Ensure all artifacts are committed to git
2. Update TaskCard with current status
3. Log handoff in audit trail
4. Brief new owner on context

---

## 8. Troubleshooting

### Gate Failed

```powershell
# View detailed gate report
cat reports/gates/G<N>_<TASK_ID>.md

# Re-run specific check
python scripts/gate_check.py --gate G1 --task-id <TASK> --verbose
```

### Task Stuck

```powershell
# Check task status
python kernel/os.py task status <TASK_ID>

# Check queue locks
cat state/tasks.yaml | grep -A 3 "queues:"
```

### Data Snapshot Mismatch

```powershell
# Verify checksum
python scripts/verify_checksum.py data/<ID>/snapshot_v1/

# Regenerate snapshot (if needed)
# Document reason in deviation log
```

---

## 9. References

- [PROJECT_DELIVERY_PIPELINE.mmd](PROJECT_DELIVERY_PIPELINE.mmd) - Visual pipeline
- [configs/gates.yaml](../configs/gates.yaml) - Gate configurations
- [templates/pipeline/](../templates/pipeline/) - TaskCard templates
- [GOVERNANCE_INVARIANTS.md](../specs/canon/GOVERNANCE_INVARIANTS.md) - Core rules

---

## 10. Change Log

| Date | Version | Change |
|------|---------|--------|
| 2026-02-01 | 0.1.0 | Initial playbook |
