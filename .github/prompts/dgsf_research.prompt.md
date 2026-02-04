---
description: Research approaches or methods before planning DGSF work
mode: agent
inherits_rules: [R1, R5]
---

# DGSF Research Prompt

Explore options before committing to a plan. Evidence-based exploration.

## CORE RULES (Inherit from Kernel)

- **R1**: Verify before asserting — check what already exists
- **R5**: No assumptions — cite sources for all claims

## INPUTS

| Required | Description |
|----------|-------------|
| Question | What to research (e.g., "Best dropout rate for SDF?") |

| Optional | Default |
|----------|---------|
| Scope | Internal (codebase) + External (literature) |
| Constraints | None |

## RESEARCH PROTOCOL

```
PHASE 1 — Internal Discovery (always first)
  □ What exists in our codebase related to this?
  □ What have we tried before? (check experiments/)
  □ What prior decisions were made? (check decisions/)
  □ What configs/parameters are currently used?

PHASE 2 — External Research (if needed)
  □ What does literature/best practice say?
  □ What are common approaches?
  □ What are the tradeoffs?

PHASE 3 — Synthesis
  □ Compare options on relevant criteria
  □ Identify most promising approach(es)
  □ Note uncertainties and risks
```

## OUTPUT FORMAT

```markdown
## Research: {question in 5-10 words}

**Scope**: Internal / External / Both
**Date**: {timestamp}

### Current State
{What exists now in our codebase — cite paths}

### Prior Work
{What we've tried before — cite experiments}

### Options Analysis

| Option | Pros | Cons | Evidence |
|--------|------|------|----------|
| {A} | {list} | {list} | {source} |
| {B} | {list} | {list} | {source} |

### Recommendation
{Which option to pursue and why}

### Uncertainties
- {What we don't know}
- {What could invalidate recommendation}

### Next Step
{Specific action: /plan X, run experiment Y, etc.}
```

## EXAMPLE: Hyperparameter Research

```markdown
## Research: Optimal dropout rate for SDF model

**Scope**: Both
**Date**: 2026-02-04

### Current State
- Current dropout: 0.3 (src/dgsf/sdf/model.py:45)
- No ablation study found in experiments/

### Prior Work
- t4_regularization used dropout=0.3, achieved OOS/IS=0.89
- No experiments varying dropout specifically

### Options Analysis

| Option | Pros | Cons | Evidence |
|--------|------|------|----------|
| 0.1 | Less underfitting risk | May overfit | Common in small models |
| 0.3 | Current default | Possibly too aggressive | t4_regularization |
| 0.5 | Strong regularization | High underfitting risk | Literature (Srivastava 2014) |
| Scheduled | Adaptive | More complex | Recent papers show benefit |

### Recommendation
Run ablation study with {0.1, 0.3, 0.5} to empirically determine best value.

### Uncertainties
- Optimal dropout may depend on dataset size (ours is ~50K samples)
- Interaction with other regularization (weight decay, early stopping)

### Next Step
/plan "Dropout ablation study"
```

## EXAMPLE: Architecture Research

```markdown
## Research: Transformer vs MLP for SDF pricing kernel

**Scope**: External
**Date**: 2026-02-04

### Current State
- Current architecture: 3-layer MLP (src/dgsf/sdf/model.py)
- No transformer implementation exists

### Prior Work
- All t4 experiments used MLP
- No transformer experiments

### Options Analysis

| Option | Pros | Cons | Evidence |
|--------|------|------|----------|
| MLP | Simple, fast, proven | Limited cross-asset attention | Our t4 results |
| Transformer | Cross-asset attention | Higher compute, data hungry | Gu et al. 2021 |
| Hybrid | Best of both | Complex implementation | No direct evidence |

### Recommendation
Keep MLP for now. Transformer exploration is t6+ scope.

### Uncertainties
- Transformer may need >100K samples to benefit
- Cross-asset attention relevance unclear for SDF

### Next Step
Continue with MLP optimization in t5.
```

## BOUNDARIES

- Internal research ALWAYS comes first (don't reinvent)
- External research must cite sources (papers, docs, repos)
- NO fabricating experimental results
- Research is EXPLORATION, not commitment — /plan decides action

## SEARCH COMMANDS

```powershell
# Find related code
Get-ChildItem -Recurse -Filter "*.py" | Select-String -Pattern "dropout"

# Check prior experiments
Get-ChildItem projects/dgsf/experiments -Directory | ForEach-Object {
    Write-Host $_.Name
    if (Test-Path "$($_.FullName)/config.yaml") {
        Get-Content "$($_.FullName)/config.yaml" | Select-String "dropout"
    }
}

# List all configs
Get-ChildItem projects/dgsf -Recurse -Filter "config.yaml"
```
```
