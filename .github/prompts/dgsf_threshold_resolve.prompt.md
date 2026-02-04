---
description: Resolve context-aware success thresholds using Adaptive Threshold Engine
mode: agent
inherits_rules: [R1, R3, R5]
---

# DGSF Threshold Resolve Prompt

Resolve context-aware success thresholds based on market regime, strategy type, and sample characteristics.

## PURPOSE

The Adaptive Threshold Engine (ATE) adjusts thresholds based on:
- **Market Regime**: Volatility levels affect expected performance
- **Strategy Type**: Different strategies have different normal ranges
- **Sample Characteristics**: Sample size affects statistical confidence

This skill ensures fair evaluation by considering context, while maintaining governance constraints.

## INPUTS

| Required | Description |
|----------|-------------|
| Metric | Which threshold to resolve (e.g., "oos_sharpe") |

| Optional | Default |
|----------|---------|
| Strategy Type | None (use base) |
| Sample Period | Auto-detect from experiment |
| Override Reason | None |

## PROTOCOL

```
PHASE 1 — Context Detection
  □ Identify current/historical market regime
  □ Determine strategy type from config
  □ Calculate sample characteristics

PHASE 2 — Threshold Resolution
  □ Start with base thresholds
  □ Apply regime adjustments
  □ Apply strategy adjustments
  □ Apply sample adjustments
  □ Clamp to governance constraints

PHASE 3 — Documentation
  □ Report resolved threshold
  □ Document all adjustments applied
  □ Note if near governance floor/ceiling
```

## OUTPUT FORMAT

```markdown
## Threshold Resolution: {metric}

**Base Threshold**: {base_value}
**Resolved Threshold**: {resolved_value}

### Context

| Factor | Detected | Adjustment |
|--------|----------|------------|
| Market Regime | {regime} | {adjustment} |
| Strategy Type | {type} | {adjustment} |
| Sample Size | {years} | {adjustment} |

### Adjustment Details

#### Market Regime: {regime_name}
**Detection Method**: {how detected}
**Period**: {date range}
**Rationale**: {why adjustment applied}
**Adjustment**: {base} → {adjusted}

#### Strategy Type: {type_name}
**Rationale**: {why this strategy type has different expectations}
**Adjustment**: {adjusted} → {further_adjusted}

### Governance Constraints

| Constraint | Value | Status |
|------------|-------|--------|
| Absolute Minimum | {floor} | {OK/BINDING} |
| Max Relaxation | {%} | {OK/BINDING} |

### Final Resolution

**Metric**: {metric}
**Resolved Threshold**: {final_value}
**Applicable Period**: {date range if temporal}

### Usage

Use this threshold for verification:
```yaml
# In experiment verification
thresholds:
  {metric}:
    value: {resolved_value}
    regime_context: "{context_summary}"
```
```

## EXAMPLE

**Input**: Resolve OOS Sharpe threshold for momentum strategy in high-vol period

```markdown
## Threshold Resolution: oos_sharpe

**Base Threshold**: 1.5
**Resolved Threshold**: 1.20

### Context

| Factor | Detected | Adjustment |
|--------|----------|------------|
| Market Regime | high_vol (VIX avg 28) | -0.3 |
| Strategy Type | momentum | +0.0 (already adjusted) |
| Sample Size | 5 years | +0.0 (normal) |

### Adjustment Details

#### Market Regime: High Volatility
**Detection Method**: 63-day average VIX = 28 (> 25 threshold)
**Period**: 2022-01-01 to 2022-12-31
**Rationale**: 
In high volatility regimes:
- Risk premia are compressed
- Transaction costs effectively higher (slippage)
- Draw on historical relationships weaker

Per ATE configuration, high_vol regime allows 0.3 reduction in Sharpe threshold.

**Adjustment**: 1.5 → 1.2

#### Strategy Type: Momentum
**Rationale**: 
Momentum strategies have known decay and regime sensitivity.
Standard threshold already accounts for this in base.
No additional adjustment needed.

**Adjustment**: 1.2 → 1.2 (no change)

### Governance Constraints

| Constraint | Value | Status |
|------------|-------|--------|
| Absolute Minimum | 0.5 | OK (1.2 > 0.5) |
| Max Relaxation | 30% | OK (20% < 30%) |

### Final Resolution

**Metric**: oos_sharpe
**Resolved Threshold**: 1.20
**Applicable Period**: 2022-01-01 to 2022-12-31

**Note**: This is a 20% relaxation from base threshold. Human override not required as within governance limits.

### Usage

```yaml
# For t32_momentum_2022 verification
thresholds:
  oos_sharpe:
    value: 1.20
    regime_context: "high_vol_2022"
    adjustments:
      - "high_vol: -0.3"
    base_value: 1.5
```
```

## REGIME DETECTION

| Regime | Indicator | Threshold |
|--------|-----------|-----------|
| low_vol | VIX 63d avg | < 15 |
| normal_vol | VIX 63d avg | 15-25 |
| high_vol | VIX 63d avg | > 25 |
| crisis | VIX 63d avg | > 35 |

## STRATEGY TYPES

| Type | Sharpe Adj | Turnover Adj | Rationale |
|------|------------|--------------|-----------|
| momentum | -0.2 | +2.0 | Known decay, higher turnover |
| value | -0.3 | -0.5 | Longer horizon, lower turnover |
| stat_arb | +0.5 | +8.0 | Capacity constrained, very high turnover |
| factor_timing | -0.2 | +1.0 | Regime dependent |

## GOVERNANCE FLOORS

These cannot be breached regardless of adjustments:

| Metric | Absolute Floor | Rationale |
|--------|---------------|-----------|
| oos_sharpe | 0.5 | Below this, strategy is noise |
| oos_is_ratio | 0.5 | Below this, severe overfit |
| max_drawdown | 40% | Above this, too risky |

## HUMAN OVERRIDE

If adjustment would exceed governance limits:
1. Cannot auto-approve
2. Must use `/dgsf_spec_propose` to request exception
3. Document exceptional circumstances
4. Require human approval

## ATE LOCATION

Primary source: `configs/adaptive_threshold_engine.yaml`

## INTEGRATION

This skill is:
- Called by `/dgsf_verify` before threshold comparison
- Used when evaluating experiments from different periods
- Required when strategy type differs from default

The resolved threshold is documented in results.json for audit trail.
