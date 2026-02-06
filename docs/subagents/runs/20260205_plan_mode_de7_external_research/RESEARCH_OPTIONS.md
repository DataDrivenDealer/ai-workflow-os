# external_research — DE7 Upgrade Options (2026-02-05)

## Option 1: “Surgical v2” (recommended near-term)
- Replace any cartesian PIT merge patterns with `merge_asof`-style as-of joins.
- Add schema normalization for mixed upstreams (full vs A0 interim).
- Add fail-fast QA gates (100% NaN guardrails; key uniqueness; month coverage; dtype checks).

Pros: minimal disruption, strong perf win, keeps DE7 structure.
Cons: requires strict sorting/types and deterministic tie-breaks.

## Option 2: Upstream canonical monthly PIT panels
- Push as-of selection into DE3/DE4 outputs (monthly effective panels keyed by `date`), so DE7 becomes simple merges.

Pros: best long-term causality/contract story; fastest joins.
Cons: larger migration surface; touches DE3/DE4 contracts.

## Option 3: Two-path migration with caching
- Precompute aligned as-of snapshots once per run; reuse across factor blocks.

Pros: major runtime reduction without redesign.
Cons: higher peak memory if not column-disciplined.

## Migration strategy
- Version outputs side-by-side (keep frozen v1; publish v2 as `de7_factor_panel_v2.parquet`).
- Shadow-compare key stats (rows, duplicates, missingness) before switching downstream.
- Switch one consumer behind a config flag, then promote.

## Suggested acceptance criteria for the upgrade
- PIT correctness: chosen statement row satisfies `effective_date <= month_end` for sampled keys.
- Robustness: succeeds on both full-style and A0-style upstream schemas.
- QA guardrails: block if key factors are 100% NaN, warn if missingness exceeds defined thresholds.
- Performance: no intermediate cartesian blowups; peak memory roughly linear in input sizes.
