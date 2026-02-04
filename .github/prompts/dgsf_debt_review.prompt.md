---
description: Review and prioritize technical and research debt
mode: agent
inherits_rules: [R1, R5]
---

# DGSF Debt Review Prompt

Review the Strategic Debt Ledger to prioritize and plan debt remediation.

## PURPOSE

The Strategic Debt Ledger (SDL) tracks:
- **Technical Debt**: Code quality, test coverage, infrastructure
- **Research Debt**: Outdated methods, unreplicated results, missing documentation
- **Evolutionary Debt**: Spec drift, knowledge decay

This skill helps maintain system health through systematic debt management.

## INPUTS

| Required | Description |
|----------|-------------|
| Mode | "review" (analyze) or "log" (add new debt) |

| Optional | Default |
|----------|---------|
| Category | "all" |
| Sprint | Current sprint |

## REVIEW PROTOCOL

```
PHASE 1 — Debt Inventory
  □ Load current debt items from SDL
  □ Filter by category if specified
  □ Calculate priority scores

PHASE 2 — Prioritization
  □ Rank by priority_score
  □ Identify high-interest items (getting worse)
  □ Flag items approaching critical threshold

PHASE 3 — Remediation Planning
  □ Select top items for sprint
  □ Estimate effort and impact
  □ Create remediation tasks
```

## OUTPUT FORMAT (Review Mode)

```markdown
## Debt Review: {date}

**Total Debt Items**: {count}
**High Priority**: {count}
**Items Remediated This Sprint**: {count}

### Priority Queue

| Rank | ID | Category | Description | Priority | Interest | Effort |
|------|----|---------:|-------------|----------|----------|--------|
| 1 | {id} | {cat} | {desc} | {score} | {rate} | {days} |
| 2 | {id} | {cat} | {desc} | {score} | {rate} | {days} |

### High-Interest Items ⚠️

These items are getting worse and should be addressed soon:

#### {Debt ID}: {Title}
**Category**: {category}
**Interest Rate**: High (actively degrading)
**Current State**: {description}
**If Unaddressed**: {consequence}
**Remediation**: {how to fix}
**Effort**: {X} days

---

### Recommended Sprint Allocation

Per SDL policy, 20% of capacity reserved for debt remediation.

**Available Capacity**: {X} days
**Recommended Items**:

1. **{Debt ID}** ({effort} days)
   - {Brief description}
   - Impact: {expected improvement}

2. **{Debt ID}** ({effort} days)
   - {Brief description}
   - Impact: {expected improvement}

### Debt Trends

| Category | Last Month | This Month | Trend |
|----------|------------|------------|-------|
| Technical | {n} | {n} | ↑/↓/→ |
| Research | {n} | {n} | ↑/↓/→ |
| Evolutionary | {n} | {n} | ↑/↓/→ |

### Next Actions

1. {Action 1}
2. {Action 2}
```

## OUTPUT FORMAT (Log Mode)

```markdown
## New Debt Logged

**Debt ID**: {generated_id}
**Category**: {category}
**Title**: {title}

**Details**:
- Location: {file/area}
- Description: {what is the debt}
- Interest Rate: {low/medium/high}
- Remediation: {how to fix}
- Effort Estimate: {days}

**Priority Score**: {calculated}

**Added to**: `configs/strategic_debt_ledger.yaml`
```

## EXAMPLE REVIEW

```markdown
## Debt Review: 2026-02-04

**Total Debt Items**: 12
**High Priority**: 4
**Items Remediated This Sprint**: 2

### Priority Queue

| Rank | ID | Category | Description | Priority | Interest | Effort |
|------|----|---------:|-------------|----------|----------|--------|
| 1 | RD-002 | Research | MTC not applied to early experiments | 0.90 | Medium | 1d |
| 2 | TD-001 | Technical | Backtest engine high complexity | 0.85 | High | 3d |
| 3 | RD-001 | Research | Purged CV missing in t01-t03 | 0.82 | Medium | 2d |
| 4 | TD-003 | Technical | Slow feature pipeline | 0.78 | Low | 2d |

### High-Interest Items ⚠️

#### TD-001: Backtest Engine Complexity
**Category**: Technical Debt / Code Quality
**Interest Rate**: High (each new strategy increases debt)
**Current State**: 
- Cyclomatic complexity > 20
- 3 related bugs in last month
- New developers struggle to modify

**If Unaddressed**: 
- Bug rate will increase
- Feature development slows
- Knowledge silos form

**Remediation**:
1. Extract strategy execution to StrategyRunner class
2. Create position sizing module
3. Add integration tests

**Effort**: 3 days

---

### Recommended Sprint Allocation

**Available Capacity**: 2 days (20% of 10 day sprint)

**Recommended Items**:

1. **RD-002** (1 day)
   - Apply Bonferroni correction to t01-t05 conclusions
   - Impact: Prevent false positive factor discoveries

2. **TD-001** (partial - 1 day)
   - Extract StrategyRunner class
   - Impact: Reduce complexity, enable parallel work

### Next Actions

1. Create task card for RD-002 remediation
2. Schedule TD-001 for next sprint continuation
3. Add monitoring for new debt in backtest module
```

## PRIORITY FORMULA

```
priority_score = (
    0.4 * impact_score +
    0.3 * interest_rate_score +
    0.3 * (1 - effort_normalized)
)
```

Where:
- `impact_score`: How much it affects research quality (0-1)
- `interest_rate_score`: How fast it's getting worse (0-1)
- `effort_normalized`: Remediation effort / max_effort (0-1)

## SDL LOCATION

Primary source: `configs/strategic_debt_ledger.yaml`

## INTEGRATION

This skill is:
- Run weekly as part of sprint planning
- Triggered when new debt is discovered
- Connected to evolution signals for debt tracking

When debt is remediated:
1. Log to evolution_signals with `category=debt_remediated`
2. Update SDL entry status to `remediated`
3. Record actual effort vs estimated
