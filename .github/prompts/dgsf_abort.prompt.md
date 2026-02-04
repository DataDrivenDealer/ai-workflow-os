---
description: Abort a research/plan direction when proven infeasible
mode: agent
inherits_rules: [R1, R5]
---

# DGSF Abort Prompt

Structured exit when a direction is proven infeasible. Not giving up — learning.

## WHEN TO INVOKE

- After `/dgsf_research` concludes approach is not viable
- After `/dgsf_diagnose` identifies unfixable root cause
- After multiple `/dgsf_execute` failures with same pattern

## CORE RULES (Inherit from Kernel)

- **R1**: Verify before asserting — abort requires evidence of infeasibility
- **R5**: No assumptions — document WHY it failed, not just THAT it failed

## INPUTS

| Required | Description |
|----------|-------------|
| Direction | What was being attempted |
| Evidence | Why it cannot succeed |

## OUTPUT FORMAT

```markdown
## ⛔ Abort: {direction in 3-5 words}

**Attempted**: {what we tried}
**Duration**: {time/effort spent}

### Evidence of Infeasibility
{Why this cannot work — cite specific failures}

### Root Cause Category
- [ ] Technical limitation (e.g., CUDA OOM, package incompatibility)
- [ ] Data limitation (e.g., insufficient samples, data quality)
- [ ] Theoretical limitation (e.g., assumption violated)
- [ ] Resource limitation (e.g., compute budget exceeded)

### Lessons Learned
{What we now know that we didn't before}

### Alternative Directions
| Option | Viability | Notes |
|--------|-----------|-------|
| {A} | High/Medium/Low | {why} |
| {B} | High/Medium/Low | {why} |

### Artifacts to Preserve
- `{path}` — {what it contains, why keep it}

### Next Step
{Specific action: /dgsf_research new direction, /dgsf_plan alternative, etc.}
```

## EXAMPLE

```markdown
## ⛔ Abort: Transformer-based SDF architecture

**Attempted**: Replace MLP with Transformer for factor learning
**Duration**: 2 experiments over 3 days

### Evidence of Infeasibility
- t4_transformer_v1: OOM at batch_size=32 (need 64+ for stability)
- t4_transformer_v2: OOS Sharpe = 0.87 (below 1.5 threshold)
- Attention patterns show no meaningful cross-asset learning

### Root Cause Category
- [x] Technical limitation (memory)
- [ ] Data limitation
- [x] Theoretical limitation (cross-asset attention not beneficial)
- [ ] Resource limitation

### Lessons Learned
Cross-asset attention in our panel data does not improve factor learning.
MLP with proper regularization remains more effective for this data regime.

### Alternative Directions
| Option | Viability | Notes |
|--------|-----------|-------|
| Larger MLP with dropout | High | Already proven in t4_final |
| CNN on sorted characteristics | Medium | Untested, may capture ordinal relations |
| Keep current architecture | High | Focus on data/features instead |

### Artifacts to Preserve
- `experiments/t4_transformer_v1/` — OOM investigation notes
- `experiments/t4_transformer_v2/results.json` — baseline comparison

### Next Step
/dgsf_plan "Improve feature engineering for current MLP architecture"
```

## ANTI-PATTERNS (Do NOT do)

❌ Abort without evidence: "This doesn't seem to work"
❌ Abort too early: First failure should trigger /diagnose, not /abort
❌ Abort without alternatives: Must propose at least one next direction
❌ Silent abort: All aborts must be logged for institutional learning
```
