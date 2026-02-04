---
description: Log key decisions with rationale for institutional memory
mode: agent
---

# DGSF Decision Log Prompt

Record WHY we chose A over B. Future self will thank you.

## WHEN TO INVOKE

- After `/dgsf_research` when selecting from multiple options
- After `/dgsf_abort` when pivoting to new direction
- When user explicitly makes a non-obvious choice
- At architecture-level decision points

## CORE RULES (Inherit from Kernel)

- **R1**: Verify before asserting — cite evidence for decision rationale
- **R5**: No assumptions — document assumptions explicitly

## INPUTS

| Required | Description |
|----------|-------------|
| Decision | What was decided |
| Options | What alternatives were considered |

| Optional | Default |
|----------|---------|
| Urgency | Normal |
| Reversibility | Reversible |

## OUTPUT FORMAT

```markdown
## 📋 Decision: {decision in 5-10 words}

**Date**: {timestamp}
**Context**: {what triggered this decision}
**Urgency**: Low / Normal / High
**Reversibility**: Easily reversible / Reversible with effort / Irreversible

### Options Considered

| Option | Pros | Cons | Estimated Effort |
|--------|------|------|------------------|
| {A} ✅ | {list} | {list} | {hours/days} |
| {B} | {list} | {list} | {hours/days} |
| {C} | {list} | {list} | {hours/days} |

### Decision Rationale
{Why option A was chosen — 2-3 sentences max}

### Key Assumptions
- {Assumption 1} — if violated, reconsider
- {Assumption 2} — if violated, reconsider

### Success Criteria
- [ ] {How we'll know this was the right choice}

### Review Trigger
{When to revisit this decision}

### Related
- Experiment: {t{N}_{name} if applicable}
- Prior decision: {link if this reverses/extends a prior decision}
```

## EXAMPLE

```markdown
## 📋 Decision: Use EarlyStopping with patience=10

**Date**: 2026-02-04 14:00
**Context**: t4_baseline showed OOS/IS ratio of 0.82, indicating overfitting
**Urgency**: Normal
**Reversibility**: Easily reversible

### Options Considered

| Option | Pros | Cons | Estimated Effort |
|--------|------|------|------------------|
| EarlyStopping patience=10 ✅ | Standard, well-understood | May stop too early | 1 hour |
| EarlyStopping patience=5 | Aggressive, less overfit | May underfit | 1 hour |
| Reduce epochs to 50 | Simple | Arbitrary, not adaptive | 30 min |
| Add more dropout | Orthogonal to stopping | Slower convergence | 2 hours |

### Decision Rationale
patience=10 is the literature default for financial time series (López de Prado 2018).
Combined with existing dropout, should balance under/overfit without excessive tuning.

### Key Assumptions
- Validation loss is a good proxy for OOS performance — if violated, reconsider
- 10 epochs of patience is sufficient for convergence signals — if violated, try 15

### Success Criteria
- [ ] OOS/IS ratio improves to ≥ 0.90

### Review Trigger
After t4_early_stopping experiment completes

### Related
- Experiment: t4_early_stopping
- Prior decision: None (first regularization decision)
```

## STORAGE (Structured)

**Primary**: Write YAML to `projects/dgsf/decisions/YYYY-MM-DD_DEC-{seq}_{slug}.yaml`

```yaml
# Example: projects/dgsf/decisions/2026-02-04_DEC-001_early-stopping.yaml
id: DEC-001
date: 2026-02-04
title: Use EarlyStopping with patience=10
context: t4_baseline OOS/IS ratio 0.82 indicates overfitting
urgency: normal
reversibility: easily_reversible
options:
  - name: EarlyStopping patience=10
    chosen: true
    pros: [standard, well-understood]
    cons: [may stop early]
    effort: 1h
  - name: EarlyStopping patience=5
    chosen: false
    pros: [aggressive]
    cons: [may underfit]
    effort: 1h
rationale: Literature default for financial time series (López de Prado 2018)
assumptions:
  - Validation loss proxies OOS performance
  - 10 epochs sufficient for convergence signal
success_criteria:
  - OOS/IS ratio >= 0.90
review_trigger: After t4_early_stopping completes
related_experiments: [t4_early_stopping]
related_decisions: []
```

**Secondary**: Echo Markdown summary to conversation (see OUTPUT FORMAT above)

**Index**: Run `ls projects/dgsf/decisions/*.yaml | Sort-Object` to list all decisions
```
