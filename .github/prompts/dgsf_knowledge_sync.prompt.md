---
description: Synchronize and query the Quant Knowledge Base for current research insights
mode: agent
inherits_rules: [R1, R5]
---

# DGSF Knowledge Sync Prompt

Access and update the Quant Knowledge Base (QKB) to ensure research is aligned with current frontier knowledge.

## PURPOSE

The QKB tracks quantitative finance research frontiers including:
- Factor research and anomalies
- Statistical methods and inference
- Machine learning in finance
- Backtesting methodology

This skill ensures we use current best practices and don't miss important advances.

## INPUTS

| Required | Description |
|----------|-------------|
| Query | What knowledge to retrieve (e.g., "best practice for cross-validation") |

| Optional | Default |
|----------|---------|
| Domain | Auto-detect from query |
| Update | false (set true to trigger refresh) |

## PROTOCOL

```
PHASE 1 — Knowledge Lookup
  □ Parse query to identify relevant domain(s)
  □ Retrieve current_consensus for domain
  □ Check methodology_standards if applicable
  □ Note last_updated dates

PHASE 2 — Relevance Assessment
  □ Match query to specific consensus items
  □ Identify applicable methodology_standards
  □ Check for any caveats or recent changes

PHASE 3 — Response Formulation
  □ Cite specific QKB references
  □ Provide actionable guidance
  □ Note any uncertainties or emerging areas
```

## OUTPUT FORMAT

```markdown
## Knowledge Query: {query summary}

**Domain**: {QKB domain}
**Last Updated**: {date}

### Current Consensus

{Summary of relevant consensus item}

**Implications for your work**:
- {Specific implication 1}
- {Specific implication 2}

### Methodology Standard

{If applicable, the current best practice}

**Correct Pattern**:
```python
{code example from QKB}
```

**Avoid**:
```python
{anti-pattern from QKB}
```

### References

- {Key reference 1}
- {Key reference 2}

### Caveats

{Any limitations or emerging changes}
```

## EXAMPLE

```markdown
## Knowledge Query: best cross-validation for time series

**Domain**: statistical_methods / backtesting
**Last Updated**: 2026-01

### Current Consensus

For time series financial data, standard K-fold cross-validation is inappropriate
because it shuffles temporal order, causing data leakage.

**Implications for your work**:
- Never use `shuffle=True` with financial time series
- Use walk-forward or purged K-fold instead
- Apply embargo period between train and test

### Methodology Standard

**Correct Pattern**:
```python
from kernel.quant_utils import PurgedKFold

cv = PurgedKFold(
    n_splits=5,
    embargo_td=pd.Timedelta(days=5)
)
```

**Avoid**:
```python
# WRONG - shuffles time series
train_test_split(X, y, test_size=0.2, shuffle=True)
```

### References

- de Prado (2018) AFML, Chapter 7
- configs/code_practice_registry.yaml#BT-02

### Caveats

For very long time series (>20 years), consider using multiple walk-forward
windows to assess regime changes.
```

## REFRESH TRIGGER

If user asks to update knowledge:

```markdown
## Knowledge Refresh: {domain}

**Sources Checked**:
- arxiv.org/list/q-fin/new
- SSRN Finance top downloads
- Practitioner sources

**Updates Found**: {count}

**Summary**:
{Brief summary of any new relevant papers/insights}

**Action Required**: 
{If any consensus updates needed, suggest /dgsf_spec_propose}
```

## QKB LOCATION

Primary source: `configs/quant_knowledge_base.yaml`

## INTEGRATION

This skill is auto-triggered when:
- User asks about "best practice" for something
- User asks about methodology or statistics
- New experiment design begins

Always cross-reference with Code Practice Registry (CPR) for enforcement rules.
