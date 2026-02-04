---
description: Verify DGSF code, experiments, or research claims
mode: agent
inherits_rules: [R1, R3]
---

# DGSF Verify Prompt

Validate claims with primary evidence. No claim accepted without proof.

## CORE RULES (from Kernel)

- **R1**: Verify before asserting — this prompt IS the verification
- **R3**: Stop on failure — if evidence missing, verdict is FAIL

## INPUTS

| Required | Description |
|----------|-------------|
| Claim | What to verify (e.g., "OOS Sharpe > 1.5") |
| Evidence Path | Where to find proof |

## NEXT STEPS (based on verdict)

| Verdict | Action |
|---------|--------|
| PASS | Invoke `/dgsf_state_update` with type=complete |
| FAIL | Invoke `/dgsf_diagnose` to find root cause |
| INCONCLUSIVE | Gather more evidence or clarify claim |

## OUTPUT FORMAT

```markdown
## Verification: {claim in measurable terms}

**Source**: {file path}
**Accessed**: {timestamp}

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| {item} | {value} | {value} | ✅/❌ |

**Verdict**: PASS / FAIL / INCONCLUSIVE
**Reason**: {one sentence}
```

## EXAMPLE: Successful Verification

```markdown
## Verification: t4_final achieved OOS Sharpe ≥ 1.5

**Source**: projects/dgsf/experiments/t4_final/results.json
**Accessed**: 2026-02-04 14:32:00

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| File exists | Yes | Yes | ✅ |
| oos_sharpe | ≥ 1.5 | 1.67 | ✅ |
| oos_is_ratio | ≥ 0.9 | 0.94 | ✅ |

**Verdict**: PASS
**Reason**: All metrics meet or exceed thresholds.
```

## EXAMPLE: Failed Verification

```markdown
## Verification: Dropout improves OOS performance

**Source**: experiments/t4_dropout/results.json
**Accessed**: 2026-02-04 14:35:00

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| File exists | Yes | NO | ❌ |

**Verdict**: FAIL
**Reason**: Evidence file does not exist. Run experiment first.
```

## DGSF SUCCESS THRESHOLDS (from Kernel)

When verifying DGSF experiments, ALWAYS check against these thresholds:

| Metric | Threshold | Auto-check |
|--------|-----------|------------|
| OOS Sharpe | ≥ 1.5 | `results.oos_sharpe >= 1.5` |
| OOS/IS Ratio | ≥ 0.9 | `results.oos_sharpe / results.is_sharpe >= 0.9` |
| Max Drawdown | ≤ 20% | `results.max_drawdown <= 0.20` |
| Turnover | ≤ 200% | `results.annual_turnover <= 2.0` |

**If ANY threshold is breached, verdict = FAIL.**

(Get-Item "experiments/t4_final/results.json").LastWriteTime

# Verify checksum
Get-FileHash "data/processed/features.parquet" -Algorithm SHA256
```

## BOUNDARIES

- ONLY accept primary sources (actual files, not memory)
- NO interpolation or estimation of missing values
- MUST report exact values from source
- MUST include timestamp of access
