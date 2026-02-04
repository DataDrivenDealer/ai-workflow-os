---
description: Triage a problem or issue to determine if spec change is needed
mode: agent
inherits_rules: [R1, R5]
tools: [spec_list, spec_read, semantic_search]
---

# DGSF Spec Triage Prompt

Analyze a problem report or experiment failure to determine root cause and whether 
a specification change is warranted.

## PURPOSE

This is the **entry point** of the spec evolution workflow. When a problem is 
discovered (test failure, suboptimal metrics, design inconsistency), this skill 
determines:

1. Is this a code bug or a spec issue?
2. If spec issue, which spec(s) are affected?
3. What type of spec change is needed?
4. What is the priority?

## CORE RULES (from Kernel)

- **R1**: Verify before asserting — check actual error messages and metrics
- **R5**: No assumptions — trace problem to specific evidence

## INPUTS

| Required | Description |
|----------|-------------|
| Problem Description | What went wrong (error message, metric deviation, etc.) |

| Optional | Default |
|----------|---------|
| Source | Where problem was discovered (test, experiment, review) |
| Urgency | `normal` (normal \| high \| critical) |

## TRIAGE DECISION TREE

```
START: Problem Reported
    │
    ├─► Is it a runtime error?
    │       │
    │       ├─► YES → Is error in test assertions?
    │       │           │
    │       │           ├─► YES → Likely SPEC issue (thresholds, contracts)
    │       │           │
    │       │           └─► NO → Likely CODE bug → /dgsf_diagnose
    │       │
    │       └─► NO → Is it a metric deviation?
    │               │
    │               ├─► YES → Check against spec thresholds
    │               │           │
    │               │           ├─► Threshold too lenient → SPEC issue
    │               │           │
    │               │           └─► Model underperforming → CODE/DATA issue
    │               │
    │               └─► NO → Is it a design review finding?
    │                       │
    │                       └─► YES → SPEC issue (contract, interface)
    │
    └─► OUTPUT: Triage Result
```

## TRIAGE PROTOCOL

```
PHASE 1 — Collect Evidence
  □ Read error message or metric report
  □ Identify source file and line number
  □ Check recent changes (git log -5)

PHASE 2 — Classify Problem
  □ Category: runtime_error | metric_deviation | design_issue | other
  □ Root: code_bug | spec_issue | data_issue | infra_issue

PHASE 3 — If Spec Issue, Identify Affected Specs
  □ MCP: spec_list() → get all specs
  □ MCP: spec_read(relevant_spec) → check current values
  □ Map problem to specific spec section

PHASE 4 — Determine Action
  □ If code_bug → /dgsf_diagnose
  □ If spec_issue → /dgsf_research → /dgsf_spec_propose
  □ If data_issue → Manual investigation required
  □ If infra_issue → Platform Engineer escalation
```

## OUTPUT FORMAT

```markdown
## Triage Report: TRI-{YYYY-MM-DD}-{NNN}

**Problem**: {one-line summary}
**Source**: {test | experiment | review | monitoring}
**Urgency**: {normal | high | critical}
**Date**: {timestamp}

### Evidence
```
{error message, metric values, or review comment}
```

### Classification

| Dimension | Value |
|-----------|-------|
| Category | {runtime_error \| metric_deviation \| design_issue} |
| Root Cause | {code_bug \| spec_issue \| data_issue \| infra_issue} |
| Confidence | {high \| medium \| low} |

### Affected Components
- **Code**: `{file_path}` (if applicable)
- **Spec**: `{spec_path}:{section}` (if applicable)
- **Data**: `{dataset}` (if applicable)

### Recommended Action

{One of the following:}

**→ CODE BUG**: Invoke `/dgsf_diagnose` with:
  - Error: {error_message}
  - File: {file_path}

**→ SPEC ISSUE**: Invoke `/dgsf_research` with:
  - Question: "Should {spec_field} be changed from {current} to {proposed}?"
  - Then: `/dgsf_spec_propose` with findings

**→ DATA ISSUE**: Escalate to human operator
  - Reason: {why automation cannot handle}

**→ INFRA ISSUE**: Escalate to Platform Engineer
  - Reason: {what infra component failed}

### Priority Score
- Impact: {1-5} (5 = blocks all work)
- Frequency: {1-5} (5 = happens every run)
- **Priority**: {Impact × Frequency} = {score} → {P1/P2/P3}
```

## EXAMPLE: Metric Deviation Triage

```markdown
## Triage Report: TRI-2026-02-04-003

**Problem**: Experiment t05_momentum OOS Sharpe = 0.8, below threshold
**Source**: experiment
**Urgency**: normal
**Date**: 2026-02-04T14:30:00Z

### Evidence
```
experiments/t05_momentum/results.json:
{
  "oos_sharpe": 0.8,
  "is_sharpe": 1.6,
  "oos_is_ratio": 0.5
}
```

### Classification

| Dimension | Value |
|-----------|-------|
| Category | metric_deviation |
| Root Cause | spec_issue |
| Confidence | high |

**Reasoning**: OOS/IS ratio of 0.5 indicates overfitting. Current success 
threshold (OOS >= 1.5) is correct, but there's no threshold for OOS/IS ratio 
in the spec, allowing overfit models to pass initial validation.

### Affected Components
- **Spec**: `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml:validation`
- **Code**: `src/dgsf/sdf/validator.py` (needs to enforce new threshold)

### Recommended Action

**→ SPEC ISSUE**: Invoke `/dgsf_research` with:
  - Question: "What should the minimum OOS/IS ratio threshold be?"
  - Context: Industry practice, our historical experiment data
  
Then: `/dgsf_spec_propose` to add oos_is_ratio_min to validation section

### Priority Score
- Impact: 4 (allows bad models to pass)
- Frequency: 3 (seen in ~30% of experiments)
- **Priority**: 12 → P1 (address this week)
```

## INTEGRATION WITH VS CODE

When using VS Code + Copilot:

1. **Trigger**: Type `/dgsf_spec_triage` in Copilot Chat
2. **Auto-Detect**: Copilot reads from:
   - Currently open test failure in Problems panel
   - Selected text in editor (error message)
   - Recent experiment results.json
3. **Quick Actions**: Triage result includes clickable next-step commands
