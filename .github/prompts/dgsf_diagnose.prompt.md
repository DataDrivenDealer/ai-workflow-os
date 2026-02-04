---
description: Diagnose failures in DGSF experiments or code execution
mode: agent
inherits_rules: [R1, R3]
---

# DGSF Diagnose Prompt

Systematically identify root cause of failures. No guessing.

## CORE RULES (Inherit from Kernel)

- **R1**: Verify before asserting — read actual error messages
- **R3**: Stop on failure — diagnose is triggered BY failure, don't compound it

## INPUTS

| Required | Description |
|----------|-------------|
| Failure | What failed (test, experiment, command) |
| Evidence | Error message or log path |

## DIAGNOSTIC PROTOCOL

```
STEP 1 — Collect Evidence
  □ Read full error message (not just last line)
  □ Identify error type: Syntax / Runtime / Logic / Environment
  □ Note file:line where error originated

STEP 2 — Reproduce (if possible)
  □ Run minimal command to trigger same error
  □ Confirm error is consistent (not flaky)

STEP 3 — Isolate
  □ Check: Is error in OUR code or dependency?
  □ Check: Did it work before? What changed?
  □ Check: Environment issue (CUDA, packages, paths)?

STEP 4 — Root Cause Statement
  □ One sentence: "X fails because Y"
  □ Must be specific, not "something is wrong"
```

## OUTPUT FORMAT

```markdown
## Diagnosis: {failure in 3-5 words}

**Error Type**: Syntax / Runtime / Logic / Environment / Data
**Location**: {file}:{line}

### Evidence
```
{verbatim error message, max 20 lines}
```

### Root Cause
{One sentence explaining WHY it fails}

### Fix Recommendation
{Specific action to resolve}

### Verification Command
{Command to confirm fix worked}
```

## EXAMPLE: Runtime Error

```markdown
## Diagnosis: CUDA OOM during training

**Error Type**: Environment
**Location**: scripts/train_sdf_optimized.py:234

### Evidence
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 8.00 GiB total capacity; 6.12 GiB already allocated)
```

### Root Cause
Batch size 64 with model size 120M params exceeds 8GB VRAM.

### Fix Recommendation
Reduce `batch_size` from 64 to 32 in config.yaml, or enable gradient accumulation.

### Verification Command
pytest tests/test_sdf.py -k "train" -v  # Should pass without OOM
```

## EXAMPLE: Logic Error

```markdown
## Diagnosis: OOS Sharpe calculation wrong

**Error Type**: Logic
**Location**: src/dgsf/eval/metrics.py:45

### Evidence
```
AssertionError: Expected Sharpe > 0 for positive returns
Actual: Sharpe = -0.23, returns mean = 0.02
```

### Root Cause
Sharpe formula uses wrong denominator: `std(returns)` instead of `std(excess_returns)`.

### Fix Recommendation
Change line 45: `return mean / std` → `return excess_mean / excess_std`

### Verification Command
pytest tests/eval/test_metrics.py -v -x
```

## BOUNDARIES

- MUST include verbatim error message
- NO speculation without evidence
- If root cause unclear after protocol, state "INCONCLUSIVE" and request more info
- DO NOT attempt fix without user approval if change is non-trivial

## ESCALATION

If diagnosis is INCONCLUSIVE after full protocol:

```markdown
## Diagnosis: INCONCLUSIVE

**Attempted**:
- [x] Read error message
- [x] Reproduced issue
- [x] Checked environment

**Blocked By**: {what information is missing}

**Next Step**: {what user should provide}
```
```
