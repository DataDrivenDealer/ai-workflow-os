---
description: Query the Institutional Memory Graph for decisions and learnings
mode: agent
inherits_rules: [R1, R5]
---

# DGSF Memory Query Prompt

Query the Institutional Memory Graph to recall past decisions, learnings, and failures.

## PURPOSE

The Institutional Memory Graph (IMG) captures:
- **Decisions**: Why things were done a certain way
- **Experiments**: What was tried and what happened
- **Learnings**: Insights gained from experience
- **Failures**: What went wrong and how to prevent it

This skill prevents repeated mistakes and enables organizational learning.

## INPUTS

| Required | Description |
|----------|-------------|
| Query | What to look up (e.g., "why did we choose XGBoost?") |

| Optional | Default |
|----------|---------|
| Query Type | auto-detect |
| Time Range | all |

## QUERY TYPES

| Type | Pattern | What it returns |
|------|---------|-----------------|
| `why_decided` | "Why did we..." | Decision rationale and alternatives |
| `what_learned` | "What did we learn about..." | Insights on topic |
| `what_failed` | "What failed when..." | Past failures and causes |
| `is_valid` | "Is X still valid?" | Check for invalidations |
| `related_to` | "What's related to..." | Connected nodes |

## PROTOCOL

```
PHASE 1 — Query Parsing
  □ Identify query type
  □ Extract key entities (experiments, decisions, concepts)
  □ Formulate graph query

PHASE 2 — Memory Retrieval
  □ Search nodes matching query
  □ Follow relevant edges
  □ Gather context from connected nodes

PHASE 3 — Response Synthesis
  □ Present findings with sources
  □ Note any superseded or invalidated information
  □ Suggest related queries if helpful
```

## OUTPUT FORMAT

```markdown
## Memory Query: {query summary}

**Query Type**: {type}
**Nodes Found**: {count}

### Primary Finding

{Main answer to the query}

### Source Nodes

#### {Node Type}: {Node ID}
**Created**: {date}
**Status**: Active / Superseded / Invalidated

{Node content}

**Connected To**:
- [{edge_type}] → {other_node}
- [{edge_type}] → {other_node}

---

### Related Insights

{Other relevant information from connected nodes}

### Validity Check

**Last Validated**: {date}
**Superseded By**: {if any}
**Still Applicable**: Yes/No/Partially

### Related Queries

- "What happened after {decision}?"
- "Were there alternatives to {choice}?"
```

## EXAMPLE QUERIES

### Query: "Why did we set OOS Sharpe threshold at 1.5?"

```markdown
## Memory Query: OOS Sharpe threshold rationale

**Query Type**: why_decided
**Nodes Found**: 3

### Primary Finding

The OOS Sharpe threshold of 1.5 was set in January 2026 based on:
1. Literature consensus (Harvey et al 2016 suggests t > 3.0)
2. Our capacity constraints requiring high conviction
3. Historical backtest showing 1.5+ strategies survive real trading

### Source Nodes

#### Decision: DEC-2026-001
**Created**: 2026-01-10
**Status**: Active

> Set OOS Sharpe threshold at 1.5 for DGSF project.
> 
> **Rationale**: 
> - Factor replication crisis suggests high bar needed
> - Our trading costs require strategies with strong edge
> - Prevents false positives from multiple testing
>
> **Alternatives Considered**:
> - 1.0: Too permissive, many would fail live
> - 2.0: Too strict, would reject viable strategies
>
> **Decision Maker**: Research Lead
> **Evidence**: Analysis of 50 backtest-to-live comparisons

**Connected To**:
- [LED_TO] → Learning: LEARN-2026-005 (threshold validated in Q1)
- [SUPPORTS] → Experiment: t04_threshold_calibration

---

### Related Insights

From LEARN-2026-005:
> Strategies with OOS Sharpe > 1.5 had 75% success rate in live trading,
> while 1.0-1.5 range had only 40% success rate.

### Validity Check

**Last Validated**: 2026-02-01
**Superseded By**: None
**Still Applicable**: Yes

### Related Queries

- "What was the live trading success rate by Sharpe bucket?"
- "Should we adjust threshold for different strategy types?"
```

### Query: "What failed when we tried LSTM for returns?"

```markdown
## Memory Query: LSTM for returns prediction

**Query Type**: what_failed
**Nodes Found**: 2

### Primary Finding

LSTM for returns prediction was attempted in t08 and t15, both failed:
- t08: Severe overfitting, OOS Sharpe negative
- t15: Marginal improvement over baseline, not worth complexity

### Source Nodes

#### Failure: FAIL-t08-001
**Created**: 2026-01-20
**Experiment**: t08_lstm_returns

> **What Failed**: LSTM model for daily return prediction
> 
> **Root Cause**: 
> - Overfitting on training data (train Sharpe 3.2, OOS -0.4)
> - Insufficient regularization
> - Sequence length too long (60 days)
>
> **Prevention Measure**:
> - Use shorter sequences (10-20 days)
> - Heavy dropout (0.5+)
> - Early stopping on validation

**Connected To**:
- [LED_TO] → Learning: "Deep learning needs 10x more regularization in finance"

---

#### Failure: FAIL-t15-001
**Created**: 2026-01-28
**Experiment**: t15_lstm_improved

> **What Failed**: LSTM with regularization
>
> **Root Cause**:
> - Only marginal improvement over XGBoost (+0.1 Sharpe)
> - 10x computational cost
> - Much harder to interpret
>
> **Prevention Measure**:
> - For tabular data, prefer tree ensembles
> - Use LSTM only for genuine sequential patterns

### Validity Check

**Still Applicable**: Yes - LSTM remains unsuitable for our use case

### Lessons Synthesized

For return prediction:
1. Tree ensembles (XGBoost, LightGBM) are preferred for tabular features
2. Deep learning requires much more data and regularization
3. Computational cost rarely justified by marginal gains
```

## AUTO-CAPTURE

The IMG is automatically updated when:
- Decision logged via `/dgsf_decision_log`
- Experiment completed (success or failure)
- Spec change committed
- `/dgsf_diagnose` identifies root cause

## IMG LOCATION

Primary storage: `state/memory_graph/`
- `nodes/decisions/*.yaml`
- `nodes/experiments/*.yaml`
- `nodes/learnings/*.yaml`
- `nodes/failures/*.yaml`
- `edges.yaml`

## INTEGRATION

This skill is:
- Triggered when user asks "why", "what happened", "did we try"
- Consulted during `/dgsf_research` for prior work
- Updated after significant decisions or failures
