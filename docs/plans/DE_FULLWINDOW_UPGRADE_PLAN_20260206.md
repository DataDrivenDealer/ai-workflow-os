# DataEng 全量数据升级规划（PLAN MODE）

**Date**: 2026-02-06  
**Scope**: DE1→DE10 全链路升级，对齐 201501-202601 全量 A 股 Raw Data  
**Target**: 使 PANELTREE 和 SDF 层能够获得完整的 21 因子输入

## 0) 现状诊断总结

### 0.1 数据覆盖现状

| 层级 | 数据文件 | 覆盖范围 | 状态 |
|------|----------|----------|------|
| **Raw** | daily_basic, fina_raw | 20150101-20260131 | ✅ 已升级 |
| **DE1** | de1_canonical_daily/monthly | 201501-202601 | ✅ 已升级 |
| **DE3** | de3_fina_*_eff.parquet | 20150401-20191229 | ❌ 旧版本 |
| **DE4** | de4_fina_indicator_eff.parquet | 20150430-20191231 | ❌ 旧版本 |
| **DE5** | de5_market_micro_monthly.parquet | 需验证 | ⚠️ |
| **DE6** | universe_mask.parquet | 需验证 | ⚠️ |
| **DE7** | de7_factor_panel.parquet | 201501-202601 | ⚠️ fundamentals 2020+ = 100% NaN |

### 0.2 根因分析

```
Raw Data (201501-202601 ✅) ──────────► DE1 ✅
                                          │
              ┌───────────────────────────┘
              │
              ├─► DE3 A0 Eff Builder (config: end_date=2019-12-31) ❌
              │         │
              │         └─► DE4 A0 Fina Indicators ❌ (2019-12-31截止)
              │                    │
              └─► DE5 ─┬─► DE6 ────┴─► DE7 ⚠️
                       │              (fundamentals 依赖 DE3/DE4，2020+ 全 NaN)
                       │
                       └─► DE7.DY 因子 (股息率) 100% NaN
```

**问题根因**:
1. `configs/de3_a0.yaml` 硬编码 `end_date: "2019-12-31"`
2. DE3/DE4 builder 未随 raw data 升级重新运行
3. DE7 的 DY (股息率) 计算可能缺少必要的上游数据

### 0.3 DE7 因子缺失详情（按年）

| 因子 | 2015 | 2016-2019 | 2020-2026 | 问题 |
|------|------|-----------|-----------|------|
| ep/bm/cfp | ~75% | ~18-50% | ~18% | 正常（市值数据延迟） |
| dy | 100% | 100% | 100% | 计算公式缺失或上游缺失 |
| roe/roa/net_margin | ~70% | ~26-35% | **100%** | DE4 截止 2019 |

## 1) PANELTREE 输入需求规范

来源: `projects/dgsf/repo/src/dgsf/paneltree/a0_runner.py`

### 1.1 必需输入

| 输入 | Shape | 来源 | 列/字段 |
|------|-------|------|---------|
| X_factor_clean | (T, N, K=21) | de7_factor_panel | 21 个 A0 因子 |
| returns | (T, N) | de1_canonical_monthly | pct_chg_month / 100 |
| universe_mask | (T, N) bool | de6_universe_mask | in_index |
| float_mktcap | (T, N) | de1_canonical_monthly | circ_mv_month_end |

### 1.2 A0 因子清单（21 个）

```
Value:        EP, BM, DY, CFP
Profitability: ROE, ROA, net_margin
Growth:       RevG, EarnG, AG, InvG
Leverage:     D_E, D_A
Momentum:     Mom12, Reversal
Risk:         Beta, IVOL
Liquidity:    Turnover, ILLIQ
Size:         Size, FloatSize
```

### 1.3 raw61 ↔ A0 Mapping

| A0 | raw61 | 状态 |
|----|-------|------|
| EP | ep | ✅ |
| BM | bm | ✅ |
| DY | dy | ❌ 100% NaN |
| CFP | cfp | ✅ |
| ROE | roe | ❌ 2020+ 100% NaN |
| ROA | roa | ❌ 2020+ 100% NaN |
| net_margin | net_margin | ❌ 2020+ 100% NaN |
| ... | ... | ✅ |

## 2) SDF 输入需求规范

来源: `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml`

### 2.1 StateEngine 输入

| 输入 | Shape | 约束 |
|------|-------|------|
| returns | [T, K] | T >= 252, K >= 10 |
| turnover | [T, K] | Non-negative |

### 2.2 Trainer 输入

| 输入 | Shape | 约束 |
|------|-------|------|
| R_leaf | [T, N] | From PANELTREE leaf_portfolios |
| Info_t | [T, D] | t-observable only (no leak) |
| phi | [T, J] | J=4 baseline or J=5 with crowd |

## 3) 执行计划（升级任务序列）

### 3.1 Phase 1: 配置升级（阻塞）

| 任务 | 目标 | 预估工时 |
|------|------|----------|
| DE.CFG.1 | 更新 de3_a0.yaml: end_date → 2026-01-31 | 10min |
| DE.CFG.2 | 更新 de4_a0.yaml: end_date → 2026-01-31 | 10min |
| DE.CFG.3 | 验证 de5/de6/de7 configs 时间窗口 | 15min |

### 3.2 Phase 2: 上游重建（DE3/DE4）

| 任务 | 目标 | 依赖 | 预估工时 |
|------|------|------|----------|
| DE.3.REBUILD | 重建 de3_fina_*_eff.parquet (201501-202601) | DE.CFG | 2-4h |
| DE.4.REBUILD | 重建 de4_fina_indicator_eff.parquet | DE.3 | 1-2h |

### 3.3 Phase 3: 中游验证/重建（DE5/DE6）

| 任务 | 目标 | 依赖 | 预估工时 |
|------|------|------|----------|
| DE.5.VALIDATE | 验证 de5_market_micro_monthly 覆盖 202601 | DE.1 | 30min |
| DE.5.REBUILD | 如验证失败，重建 DE5 | DE.5.VAL | 1-2h |
| DE.6.VALIDATE | 验证 de6_universe_mask 覆盖 202601 | DE.1 | 30min |
| DE.6.REBUILD | 如验证失败，重建 DE6 | DE.6.VAL | 1-2h |

### 3.4 Phase 4: DE7 重建与 DY 修复

| 任务 | 目标 | 依赖 | 预估工时 |
|------|------|------|----------|
| DE.7.FULLWINDOW | 重建 de7_factor_panel_v3.parquet | DE.3, DE.4, DE.5, DE.6 | 2-4h |
| DE.7.DY_INVEST | 诊断 DY 因子 100% NaN 根因 | DE.7.FULL | 1h |
| DE.7.DY_FIX | 修复 DY 因子计算 | DE.7.DY_INV | 1-2h |
| DE.7.A0 | 重建 de7_a0_factor_panel.parquet | DE.7.FULL | 1h |

### 3.5 Phase 5: QA & 下游验证

| 任务 | 目标 | 依赖 | 预估工时 |
|------|------|------|----------|
| DE.7.QA | 运行 qa_de7_runner.py 验证产物 | DE.7.A0 | 30min |
| PT.VALIDATE | 验证 PANELTREE a0_runner 输入可用 | DE.7.QA | 1h |
| SDF.VALIDATE | 验证 SDF 模块可运行 | PT.VAL | 1h |

## 4) 升级策略

### 4.1 路径规范（不覆盖 frozen）

| 产物 | Frozen 路径 | 升级路径 (v3) |
|------|-------------|---------------|
| DE3 Eff | data/a0/interim/de3_fina_*_eff.parquet | data/a0/interim_v3/de3_fina_*_eff.parquet |
| DE4 Eff | data/a0/interim/de4_fina_indicator_eff.parquet | data/a0/interim_v3/de4_fina_indicator_eff.parquet |
| DE7 Full | data/full/de7_factor_panel.parquet | data/full/de7_factor_panel_v3.parquet |
| DE7 A0 | data/interim/de7_a0/factor_panel_a0.parquet | data/interim/de7_a0_v3/factor_panel_a0.parquet |

### 4.2 验收阈值

| 指标 | 阈值 | 验证方式 |
|------|------|----------|
| 时间覆盖 | 201501-202601 | date/month_end min/max |
| (ts_code, date) 唯一 | 无重复 | duplicated() == 0 |
| 关键因子组 NaN | < 80% (非 100%) | isna().mean() |
| fundamentals 2020+ | < 60% NaN | 分年统计 |
| DY 因子 | < 50% NaN | isna().mean() |

### 4.3 回滚策略

- 保留所有 frozen 产物不覆盖
- v3 产物验证失败时，下游继续使用 v2/v1
- 切换使用 v3 时需显式更新下游配置

## 5) 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| DE3 重建耗时过长 | 阻塞下游 | 分批处理 (income → balance → cashflow) |
| DY 上游数据缺失 | 无法修复 | 使用 placeholder (stub NaN) 或替代定义 |
| DE7 QA 不通过 | 阻塞 PANELTREE | 放宽 missingness 阈值，记录 debt |
| 下游接口变更 | PANELTREE/SDF 需适配 | 保持 time key 一致，仅更新路径 |

## 6) 证据与审计

### 当前诊断命令
```python
# Data coverage check
python -c "
import pandas as pd
files = [
    ('data/a0/interim/de3_fina_income_eff.parquet', 'effective_date'),
    ('data/a0/interim/de4_fina_indicator_eff.parquet', 'month_end'),
    ('data/full/de7_factor_panel.parquet', 'date'),
]
for f, col in files:
    df = pd.read_parquet(f)
    print(f'{f}: {col} = {df[col].min()} - {df[col].max()}')
"
```

### DE7 Missingness Check
```python
# Fundamentals missingness by year
de7 = pd.read_parquet('data/full/de7_factor_panel.parquet')
de7['year'] = de7['date'] // 100
for col in ['dy', 'roe', 'roa', 'net_margin']:
    print(f"{col}: {de7.groupby('year')[col].apply(lambda x: x.isna().mean()).to_dict()}")
```

---

*PLAN MODE 规划文档 v1.0 — 2026-02-06*
