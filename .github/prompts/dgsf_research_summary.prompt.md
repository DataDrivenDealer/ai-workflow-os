---
description: Summarize DGSF research findings or experiment results
mode: agent
---

# DGSF Research Summary Prompt

Synthesize experiment results into actionable insights. Evidence-based only.

## WHEN TO INVOKE

- After completing a milestone (multiple related experiments)
- Before starting new research direction
- When user requests synthesis of findings

## CORE RULES (from Kernel)

- **R1**: Verify before asserting — all metrics must come from actual files
- **R3**: Stop on failure — flag missing data, don't interpolate

## INPUTS

| Required | Description |
|----------|-------------|
| Topic | What to summarize |
| Experiments | Which t{N}_* directories |

## OUTPUT FORMAT

```markdown
# Summary: {topic}

**Experiments**: {list}
**Date**: {timestamp}

## Key Finding
{One sentence: what did we learn?}

## Comparison

| Experiment | OOS Sharpe | OOS/IS | Status |
|------------|------------|--------|--------|
| {name} | {value} | {value} | ✅/❌ |

## Insight
{Why this matters for DGSF}

## Next Step
{Single actionable recommendation}

## Sources
- {experiment}/results.json
```

## EXAMPLE

```markdown
# Summary: T4 Training Optimization

**Experiments**: t4_baseline, t4_regularization, t4_final
**Date**: 2026-02-04

## Key Finding
Dropout + early stopping improved OOS/IS ratio from 0.82 to 0.94.

## Comparison

| Experiment | OOS Sharpe | OOS/IS | Status |
|------------|------------|--------|--------|
| t4_baseline | 1.42 | 0.82 | ❌ below target |
| t4_regularization | 1.55 | 0.89 | ⚠️ close |
| t4_final | 1.67 | 0.94 | ✅ meets target |

## Insight
Regularization alone insufficient; combination with early stopping critical.

## Next Step
Proceed to t5 OOS validation with t4_final config.

## Sources
- experiments/t4_baseline/results.json
- experiments/t4_regularization/results.json  
- experiments/t4_final/results.json
```

## BOUNDARIES

- NO speculation or interpolation
- ALL values must be extracted from actual files
- MUST flag missing experiments
