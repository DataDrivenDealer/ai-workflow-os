# Compliance Metrics

> 治理合规性指标追踪

**Last Updated**: 2026-02-05T00:00:00Z

---

## Gate-E4.5 Compliance (Pair Programming Review)

> 目标: 100% 代码变更有审核 artifact，Bypass 率 < 5%

### Weekly Summary

| Week | Tasks Executed | With Review | Bypassed | Compliance % | Notes |
|------|----------------|-------------|----------|--------------|-------|
| 2026-W06 | 0 | 0 | 0 | N/A | Gate introduced |

### Bypass Log

| Date | Task ID | Commit | User | Reason | Remediated |
|------|---------|--------|------|--------|------------|
| - | - | - | - | - | - |

**Remediation Target**: All bypasses must be remediated within 48 hours.

---

## Daily Refactor Cadence

> 目标: 每周至少运行 3 次

### Weekly Summary

| Week | Runs | Files Changed | Safe Transforms | Moderate | Status |
|------|------|---------------|-----------------|----------|--------|
| 2026-W06 | 0 | 0 | 0 | 0 | 🆕 Introduced |

### Recent Runs

| Date | Files | Safe | Moderate | Risky | Report Path | Status |
|------|-------|------|----------|-------|-------------|--------|
| - | - | - | - | - | - | - |

---

## Gate Status Overview

| Gate | Purpose | Enforcement | Last Check |
|------|---------|-------------|------------|
| Gate-E0 | Pre-execution Subagent Check | Hard | - |
| Gate-E4.5 | Pair Programming Review | Hard | 2026-02-05 (introduced) |
| Gate-E5 | Risk Review (quant_risk_review) | Hard | - |

---

## Audit Trail

### Recent Bypass Events

```
# From docs/audits/bypasses.log
(No bypasses recorded yet)
```

### Review Artifacts Created

| Date | Task ID | Rounds | Final Verdict | Duration |
|------|---------|--------|---------------|----------|
| - | - | - | - | - |

---

## Metrics Definitions

### Compliance Rate

```
Compliance % = (Tasks with APPROVED REVIEW_2.md / Total Tasks with Code Changes) × 100
```

### Bypass Rate

```
Bypass % = (Commits with --no-verify on code files / Total Code Commits) × 100
```

### Refactor Cadence

```
Cadence Score = Runs per Week / Target (3)
```

---

## Collection Methods

| Metric | Source | Frequency |
|--------|--------|-----------|
| Review Compliance | `docs/reviews/*/REVIEW_*.md` | Per task |
| Bypasses | `docs/audits/bypasses.log` | Per commit |
| Refactor Runs | `docs/refactor/*/REPORT.md` | Per run |

---

*Updated automatically after each EXECUTE MODE completion and Daily Refactor run.*
