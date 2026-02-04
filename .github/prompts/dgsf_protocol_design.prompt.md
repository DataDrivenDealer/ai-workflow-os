---
description: Design experiments using Research Protocol Algebra (RPA)
mode: agent
inherits_rules: [R1, R2, R5]
---

# DGSF Protocol Design Prompt

Design rigorous experiments using composable research protocols from the Research Protocol Algebra.

## PURPOSE

The Research Protocol Algebra (RPA) provides:
- **Primitives**: Atomic research operations with validated I/O
- **Compositions**: Standard workflows combining primitives
- **Gates**: Pre/post conditions for quality control

This skill helps design experiments that follow validated methodologies.

## INPUTS

| Required | Description |
|----------|-------------|
| Objective | What the experiment aims to achieve |

| Optional | Default |
|----------|---------|
| Protocol | Auto-suggest based on objective |
| Constraints | None |

## PROTOCOL

```
PHASE 1 — Objective Analysis
  □ Parse experiment objective
  □ Identify research type (discovery, iteration, replication, etc.)
  □ Suggest appropriate RPA composition

PHASE 2 — Protocol Selection
  □ Present available protocols matching objective
  □ Explain gates and requirements for each
  □ User selects or customizes

PHASE 3 — Experiment Design
  □ Instantiate protocol with specifics
  □ Generate config.yaml skeleton
  □ Define validation gates
  □ Create checkpoint structure
```

## AVAILABLE PROTOCOLS

| Protocol | Use Case | Key Gates |
|----------|----------|-----------|
| `FACTOR_DISCOVERY` | Finding new alpha factors | ≥3 survive MTC, avg OOS Sharpe > 0.5 |
| `MODEL_ITERATION` | Improving ML models | OOS/CV ratio > 0.8, interpretability |
| `ROBUSTNESS_BATTERY` | Validating strategies | Pass 3 of 4 robustness categories |
| `PAPER_REPLICATION` | Reproducing research | Within 20% of paper results |

## OUTPUT FORMAT

```markdown
## Protocol Design: {experiment name}

**Objective**: {user's objective}
**Selected Protocol**: {protocol name} v{version}

### Protocol Flow

```
{Diagram of the protocol flow}
```

### Customizations

| Parameter | Base Value | Your Value | Rationale |
|-----------|------------|------------|-----------|
| {param1} | {base} | {custom} | {why} |

### Pre-Conditions (must satisfy before start)

- [ ] {Condition 1}
- [ ] {Condition 2}

### Post-Conditions (success criteria)

- [ ] {Gate 1}: {threshold}
- [ ] {Gate 2}: {threshold}

### Required Artifacts

| Artifact | Description | Template |
|----------|-------------|----------|
| config.yaml | Experiment parameters | Generated below |
| lineage.yaml | Data provenance | Template provided |
| results.json | Output metrics | Schema defined |

### Generated Config

```yaml
# experiments/{exp_id}/config.yaml
# Generated from RPA:{protocol_name}

experiment:
  id: "{exp_id}"
  name: "{exp_name}"
  protocol: "{protocol_name}"
  version: "1.0.0"

data:
  source: "{data_source}"
  date_range:
    start: "{start_date}"
    end: "{end_date}"
  universe: "{universe_spec}"

model:
  type: "{model_type}"
  hyperparameters:
    # Define based on protocol requirements

validation:
  cv_method: "purged_kfold"
  n_splits: 5
  embargo_days: 5

costs:
  slippage_bps: 10
  commission_bps: 3

gates:
  pre_conditions:
    - "{condition_1}"
  post_conditions:
    - "{gate_1}"
    - "{gate_2}"

random_seed: 42
```

### Checkpoint Structure

```
experiments/{exp_id}/
├── config.yaml          # Generated above
├── lineage.yaml         # Data provenance
├── checkpoints/
│   ├── 00_data_loaded.json
│   ├── 01_features_built.json
│   ├── 02_model_trained.json
│   └── 03_backtest_complete.json
├── results.json         # Final metrics
└── artifacts/
    ├── model.pkl
    └── figures/
```

### Next Steps

1. Review and customize config.yaml
2. Run `/dgsf_execute` to start experiment
3. Checkpoints will validate each phase
4. `/dgsf_verify` when complete
```

## EXAMPLE

**User**: "I want to test if momentum factor works in Chinese A-shares"

```markdown
## Protocol Design: t42_china_momentum

**Objective**: Test momentum factor efficacy in Chinese A-shares
**Selected Protocol**: FACTOR_DISCOVERY v1.0.0

### Protocol Flow

```
LOAD_DATA(universe='CSI_ALL', period='2015-2025')
    → FEATURE_ENGINEER(momentum configs)
    → FOR each momentum variant:
        → SPLIT_DATA(walk_forward, 5 folds)
        → BACKTEST(with realistic costs)
        → EVALUATE()
    → APPLY_MTC(bonferroni)
    → AGGREGATE() → significant_factors
```

### Customizations

| Parameter | Base Value | Your Value | Rationale |
|-----------|------------|------------|-----------|
| Universe | SP500 | CSI_ALL | Chinese market |
| Costs | 10 bps | 20 bps | Higher costs in China |
| Test period | 3 years | 5 years | More data available |

### Pre-Conditions

- [x] CSI historical constituents data available
- [x] Momentum variants defined (1M, 3M, 6M, 12M)
- [ ] Point-in-time adjusted prices confirmed

### Post-Conditions

- [ ] At least 1 momentum variant survives MTC
- [ ] Surviving variants have OOS Sharpe > 0.5
- [ ] Economic rationale documented

### Generated Config

```yaml
experiment:
  id: "t42"
  name: "china_momentum"
  protocol: "FACTOR_DISCOVERY"

data:
  source: "wind_api"
  universe: "CSI_ALL"
  date_range:
    start: "2015-01-01"
    end: "2025-01-01"

factors:
  momentum_1m:
    lookback: 21
    skip: 1
  momentum_3m:
    lookback: 63
    skip: 1
  momentum_6m:
    lookback: 126
    skip: 21
  momentum_12m:
    lookback: 252
    skip: 21

validation:
  cv_method: "walk_forward"
  train_years: 5
  test_months: 3

costs:
  slippage_bps: 15
  commission_bps: 5

gates:
  post_conditions:
    - "count(surviving_factors) >= 1"
    - "avg(oos_sharpe) >= 0.5"

random_seed: 42
```

### Next Steps

1. Confirm CSI data availability with `/dgsf_research`
2. Generate experiment directory: `python scripts/generate_experiment.py --protocol FACTOR_DISCOVERY --name t42_china_momentum`
3. Run with `/dgsf_execute`
```

## RPA LOCATION

Primary source: `configs/research_protocol_algebra.yaml`

## INTEGRATION

This skill is auto-triggered when:
- New experiment creation requested
- User describes research objective
- `/dgsf_plan` identifies experiment need

Protocol compliance is verified during `/dgsf_verify`.
