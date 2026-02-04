---
description: Update DGSF project state and track progress
mode: agent
---

# DGSF State Update Prompt

Record verified progress or blockers. Evidence required.

## CORE RULES (from Kernel)

- **R1**: Verify before asserting — only record what's proven
- **R3**: Stop on failure — blocker = stop and report

## INPUTS

| Required | Description |
|----------|-------------|
| Type | `complete` / `blocked` / `milestone` |
| Subject | What changed |

## NEXT STEPS (based on type)

| Type | Action |
|------|--------|
| complete | **→ `/dgsf_git_ops` with auto_tag** → Consider if milestone reached → `/dgsf_research_summary` |
| blocked | Invoke `/dgsf_diagnose` for root cause |
| milestone | **→ `/dgsf_git_ops` with milestone tag** → `/dgsf_research_summary` to synthesize findings |

## OUTPUT FORMATS

### Experiment Complete
```markdown
## ✅ Complete: {experiment_name}

**Evidence**: {path}/results.json exists, run.log shows "complete"
**Key Metric**: OOS Sharpe = {value}
**Next**: {what to do now}
```

### Blocker Found
```markdown
## 🚫 Blocked: {component}

**Error**: {exact message}
**Location**: {file}:{line}
**Impact**: Cannot proceed with {what}
**Resolution**: {suggested fix}
```

### Milestone Reached
```markdown
## 🎯 Milestone: {name}

**Criteria Met**:
- [x] {criterion 1}
- [x] {criterion 2}

**Unlocks**: {what can now proceed}
```

## EXAMPLE: Complete

```markdown
## ✅ Complete: t5_oos_validation

**Evidence**: experiments/t5_oos_validation/results.json exists
             run.log last line: "Validation complete at epoch 47"
**Key Metric**: OOS Sharpe = 1.67, OOS/IS = 0.94
**Next**: Summarize T4-T5 results, decide on t6 deployment
```

## EXAMPLE: Blocked

```markdown
## 🚫 Blocked: SDF training

**Error**: CUDA out of memory
**Location**: scripts/train_sdf_optimized.py:234
**Impact**: Cannot run t5_large_batch experiment
**Resolution**: Reduce batch_size from 64 to 32, or use gradient accumulation
```

## BOUNDARIES

- NO fabricating completion status
- MUST cite evidence file paths
- Blockers require error message verbatim
