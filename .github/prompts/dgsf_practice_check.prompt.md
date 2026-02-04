---
description: Verify code against the Code Practice Registry (CPR)
mode: agent
inherits_rules: [R1, R3, R5]
---

# DGSF Practice Check Prompt

Verify that code adheres to quantitative best practices defined in the Code Practice Registry.

## PURPOSE

The Code Practice Registry (CPR) defines:
- **Critical practices** (DH-*, BT-*): Violations block work
- **High practices** (PF-*, MD-*): Violations require justification
- **Medium practices**: Violations flagged for review

This skill systematically checks code against these practices.

## INPUTS

| Required | Description |
|----------|-------------|
| Target | File path or code snippet to check |

| Optional | Default |
|----------|---------|
| Practices | "all" (or specific IDs like "DH-01,BT-02") |
| Severity | "critical,high" |

## PROTOCOL

```
PHASE 1 — Identify Applicable Practices
  □ Determine code type (data handling, backtesting, modeling, etc.)
  □ Load relevant practices from CPR
  □ Prioritize by severity

PHASE 2 — Pattern Matching
  □ Check for anti-patterns in code
  □ Look for missing correct patterns
  □ Note any violations

PHASE 3 — Report Generation
  □ List all violations with severity
  □ Provide correct patterns for each
  □ Suggest fixes
```

## OUTPUT FORMAT

```markdown
## Practice Check: {target file/snippet}

**Practices Checked**: {count}
**Violations Found**: {count}

### Critical Violations ❌

#### {Practice ID}: {Practice Name}
**Severity**: Critical
**Location**: {file:line}

**Violation**:
```python
{violating code}
```

**Issue**: {description of what's wrong}

**Correct Pattern**:
```python
{correct code}
```

**Reference**: {CPR reference}

---

### High Severity Warnings ⚠️

{Same format as above}

---

### Compliant Practices ✅

| Practice | Status | Notes |
|----------|--------|-------|
| DH-01 Point-in-Time | ✅ | Shifts use positive lags |
| DH-02 Survivorship | ✅ | Historical universe used |
| BT-02 Walk-Forward | ✅ | TimeSeriesSplit used |

---

## Summary

**Pass/Fail**: {PASS if no critical violations, else FAIL}

**Recommended Actions**:
1. {Fix critical violation 1}
2. {Fix critical violation 2}

**Next Step**: {/dgsf_execute to fix OR /dgsf_verify if passed}
```

## EXAMPLE CHECK

**Input**: Check `projects/dgsf/repo/src/dgsf/features/builder.py`

```markdown
## Practice Check: projects/dgsf/repo/src/dgsf/features/builder.py

**Practices Checked**: 8
**Violations Found**: 2

### Critical Violations ❌

#### DH-01: Point-in-Time Correctness
**Severity**: Critical
**Location**: features/builder.py:45

**Violation**:
```python
df['ma_20'] = df['price'].rolling(20).mean()
# Missing shift - uses today's price in today's feature
```

**Issue**: Rolling mean includes current observation, creating lookahead bias.

**Correct Pattern**:
```python
df['ma_20'] = df['price'].shift(1).rolling(20).mean()
# Shift(1) ensures we only use yesterday's data
```

**Reference**: configs/code_practice_registry.yaml#DH-01

---

#### DH-03: Data Leakage Prevention
**Severity**: Critical
**Location**: features/builder.py:78

**Violation**:
```python
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)  # Before train-test split
```

**Issue**: Scaler fitted on full data including future test data.

**Correct Pattern**:
```python
# Use pipeline or fit on train only
scaler.fit(df_train)
df_test_scaled = scaler.transform(df_test)
```

**Reference**: configs/code_practice_registry.yaml#DH-03

---

### Compliant Practices ✅

| Practice | Status | Notes |
|----------|--------|-------|
| DH-02 Survivorship | ✅ | Uses historical_constituents() |
| BT-01 Transaction Costs | N/A | Not in this file |
| BT-02 Walk-Forward | N/A | Not in this file |
| PF-01 Vectorized | ✅ | No row-wise loops found |

---

## Summary

**Pass/Fail**: FAIL (2 critical violations)

**Recommended Actions**:
1. Add shift(1) to rolling calculations (line 45)
2. Move scaler fitting inside CV loop (line 78)

**Next Step**: /dgsf_execute to fix violations
```

## ENFORCEMENT LEVELS

| Severity | Action on Violation |
|----------|---------------------|
| Critical | BLOCK - must fix before proceed |
| High | WARN - require documented justification |
| Medium | NOTE - flag for code review |

## CPR LOCATION

Primary source: `configs/code_practice_registry.yaml`

## INTEGRATION

This skill is auto-triggered when:
- Code review requested
- Before experiment execution
- New file created in src/ directories

Cross-reference with QKB for rationale behind practices.
