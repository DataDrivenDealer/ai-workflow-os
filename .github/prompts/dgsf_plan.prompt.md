---
description: Create a research or development plan for DGSF work
mode: agent
---

# DGSF Plan Prompt

Create a single, actionable plan for DGSF research or development.

## CORE RULES (from Kernel)

- **R1**: Verify before asserting (check paths exist before planning to modify them)
- **R2**: One task at a time (plan ONE thing)

## PREREQUISITES

- If uncertain about approach, run `/dgsf_research` first
- If need to understand current state, run `/dgsf_repo_scan` first

## INPUTS

| Required | Description |
|----------|-------------|
| Objective | What to achieve (one sentence) |

| Optional | Default |
|----------|---------|
| Module | Auto-detected from objective |
| Prior Work | None |

## OUTPUT FORMAT

```markdown
## Plan: {3-5 word title}

**Objective**: {one sentence}
**Module**: src/dgsf/{module}/
**Prior**: {experiment ref or "None"}

### Hypothesis
{What we expect and why — 2 sentences max}

### Steps (max 5)
1. {Verify}: Check {path} exists
2. {Read}: Understand current {component}
3. {Modify}: Change {specific thing}
4. {Test}: Run pytest {specific test}
5. {Record}: Write results to {path}

### Success Criteria
- [ ] {Measurable criterion 1}
- [ ] {Measurable criterion 2}

### Artifacts
- `{path}` — {what it contains}
```

## EXAMPLE: Good Plan

```markdown
## Plan: Add Early Stopping to SDF

**Objective**: Reduce overfitting by stopping training when validation loss plateaus
**Module**: src/dgsf/sdf/
**Prior**: experiments/t4_baseline/results.json

### Hypothesis
Adding patience-based early stopping will improve OOS/IS ratio by preventing
late-stage overfitting observed in t4_baseline (OOS/IS = 0.82).

### Steps
1. Verify: ls projects/dgsf/repo/src/dgsf/sdf/trainer.py
2. Read: Understand current training loop (lines 50-120)
3. Modify: Add EarlyStopping callback with patience=10
4. Test: pytest tests/test_sdf.py -v -k "train"
5. Record: Save config + results to experiments/t4_early_stopping/

### Success Criteria
- [ ] Training stops before max_epochs when loss plateaus
- [ ] OOS/IS ratio ≥ 0.90

### Artifacts
- `experiments/t4_early_stopping/config.yaml` — hyperparameters
- `experiments/t4_early_stopping/results.json` — metrics
```

## ANTI-PATTERNS (Do Not Do)

❌ **Vague objective**: "Improve the model"  
✅ **Specific objective**: "Reduce validation loss by 10% via dropout"

❌ **Unverified paths**: "Modify src/dgsf/core/engine.py"  
✅ **Verified paths**: First run `ls src/dgsf/` then reference

❌ **Multiple objectives**: "Add dropout AND change optimizer AND..."  
✅ **Single objective**: "Add dropout to SDF forward pass"

## BOUNDARIES

- Plan ONE task (not a roadmap)
- ALL paths must be verified before including in plan
- NO financial parameter assumptions (fees, leverage, etc.)
