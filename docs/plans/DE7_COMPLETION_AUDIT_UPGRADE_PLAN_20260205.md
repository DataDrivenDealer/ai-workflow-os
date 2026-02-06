# DE7 补完 / 审计 / 升级调研计划（PLAN MODE）

**Date**: 2026-02-05  
**Scope**: DGSF DataEng DE7 factor panel（Fullwindow + A0）、DE7 style spreads、QA/契约治理

## 0) 当前结论（已知现状）
- PIT 合规性：已完成并有报告 `projects/dgsf/reports/POINT_IN_TIME_COMPLIANCE_REPORT.md`（针对 A0 builder）。
- DE7 产物已在 repo 严格契约路径下产出并通过 QA（见下方“执行状态”）。

### 0.1 执行状态（2026-02-05）
- DE7_RAW_61：已产出并同步到契约默认路径
  - `projects/dgsf/repo/data/processed/factor_raw_panel.parquet`（默认消费/QA）
  - `projects/dgsf/repo/data/full/de7_factor_panel.parquet`（canonical）
  - `projects/dgsf/repo/data/full/de7_factor_panel_v2.parquet`（v2 备份）
- DE7_A0_21：已产出
  - `projects/dgsf/repo/data/interim/de7_a0/factor_panel_a0.parquet`
- QA 报告：
  - `projects/dgsf/reports/de7_qa_both_20260205_1452.json`（显式指定路径运行）
  - `projects/dgsf/reports/de7_qa_both_defaults_20260205_1456.json`（使用默认契约路径运行）
- `projects/dgsf/reports/de7_qa_both_defaults_20260205_1512.json`（更新 A0 DY/InvG 计算后的默认路径 QA）
- 备注（coverage）：在当前数据快照下，A0 `DY`（由 DE3 cashflow TTM proxy 回退）覆盖率约 22%，`InvG`（由 DE3 balance 资产代理 YoY 回退）覆盖率约 23%。

## 1) 明确两类“DE7 产物契约”（必须补完）
为避免“DE7 factor panel”指代不清，强制区分两个契约：

1) **DE7_RAW_61（Fullwindow 原始 61 因子面板）**
- Producer: `projects/dgsf/repo/src/dgsf/dataeng/de7_factor_panel.py`
- Time key: `date`（int YYYYMM，内部约定）
- Schema: `FACTOR_RAW_PANEL_SCHEMA_V42`（见 `projects/dgsf/repo/src/dgsf/config/schema.py`）
- Output (configurable default): `data/processed/factor_raw_panel.parquet`
- Canonical schema path: `data/full/de7_factor_panel.parquet`

2) **DE7_A0_21（PanelTree A0 因子面板）**
- Producer: `projects/dgsf/repo/src/dgsf/dataeng/de7_factor_panel_a0_builder.py`
- Time key: `month_end`（datetime64[ns]，外部 canonical）
- Factors: A0 builder 列表（mixed-case；包含 2 个 stub）
- Output: `data/interim/de7_a0/factor_panel_a0.parquet`

必须补齐：**mapping 表（raw61 ↔ a0_21）**，以及谁负责 rename/alias。

### raw61 ↔ A0 mapping（覆盖 ≥21 个 A0 因子）

| A0 列名 | raw61 列名 | 备注 |
|---|---|---|
| EP | ep | 估值：E/P |
| BM | bm | 估值：B/M |
| DY | dy | 股息率 |
| CFP | cfp | 现金流/市值 |
| ROE | roe | 盈利能力 |
| ROA | roa | 盈利能力 |
| gross_margin | gross_margin | A0 stub（NaN） |
| oper_margin | oper_margin | A0 stub（NaN） |
| net_margin | net_margin | 盈利能力 |
| RevG | rev_growth | 增长 |
| EarnG | earn_growth | 增长 |
| AG | asset_growth | 增长 |
| InvG | investment_growth | 投资增长 |
| D_E | de_ratio | 杠杆 |
| D_A | da_ratio | 杠杆 |
| Mom12 | mom_12m | 动量 |
| Reversal | rev_1m | 反转（1M） |
| Beta | beta_12m | 风险 |
| IVOL | idio_vol_12m | 风险 |
| Turnover | turnover_ratio_1m | 流动性（定义需对齐） |
| ILLIQ | illiq_proxy | 流动性 |
| Size | log_mktcap | 规模 |
| FloatSize | log_float_mktcap | 规模 |

> 注：A0 builder 当前输出 23 个因子列，其中 `gross_margin`、`oper_margin` 为 stub（NaN），其余 21 为有效计算项。

## 2) 审计（QA）计划
目标：把“能跑”升级为“可复现、可审计、能阻断劣化”。

### 2.1 QA runner 补完
- 现状：`projects/dgsf/repo/tools/qa_de7_runner.py` 已可用，支持 `--panel {a0,raw,both}` 与 `--report` 落盘。
- 行动：如需进一步治理，可补充 QA 文档对默认路径与 v2 shadow compare 的说明。

### 2.2 QA guardrails（阻断 100% NaN 等灾难）
- 增加“关键因子组 missingness”阈值：
  - block: valuation/profitability/growth 关键字段出现 100% NaN
  - warn: missingness 超过配置阈值（阈值写入配置或报告中）
- 对 key/coverage 的强校验：
  - `(ts_code, month_end)` / `(ts_code, date)` 唯一性
  - 月度覆盖连续性
  - time key 类型一致性

### 2.3 PIT 单元测试（显式）
在 tests 层增加“PIT 不可回归”测试：验证 t 时点特征仅使用 ≤t 的信息（尤其是 shift(1) 与 effective_date as-of 选择）。

## 3) 升级调研（v2）计划
目标：给出可执行的升级路线（并能灰度迁移）。

### 3.1 选型（DRS）
候选：
- Option 1：Surgical v2（`merge_asof` + schema normalization + QA gates）— **优先推荐短期**
- Option 2：上游生成“月度 PIT 对齐”的 canonical panels（DE3/DE4 月度 eff）— **长期最干净**
- Option 3：DE7 内部 cache as-of snapshots（减少重复 join）

**结论（v2 路线）**：
- **短期执行**：Option 1（Surgical v2）
  - 依据：已有 v2 配置（`configs/de7_fullwindow_v2.yaml`）指向 A0 interim，上游 schema 已修复，且不覆盖 frozen 产物。
  - 目标：修复 100% NaN fundamentals，快速恢复可用性。
- **中长期演进**：Option 2（上游 monthly PIT canonical panels）
  - 依据：从根源保证 effective_date 与字段命名一致，减少适配分支。
  - 目标：降低 ROE/ROA/NET_MARGIN 缺失并提升覆盖率。

输出：一份决策记录（选择路径 + 风险 + 回滚策略）。

### 3.2 迁移策略（不破坏下游）
- 冻结旧产物不覆盖：保留 `data/full/de7_factor_panel.parquet`（frozen），新增 `data/full/de7_factor_panel_v2.parquet`。
- Shadow compare：行数/重复键/missingness/基本分布对比通过后，再切换下游消费路径。

### 3.3 验收阈值（v2）
**硬性要求（fail-fast）**：
- 关键因子组 100% NaN 直接失败（valuation/profitability/growth/leverage/momentum/risk/liquidity/size）。
- `(ts_code, date)` 唯一键必须无重复。

**missingness 目标（来自 v2 config + hardening report）**：
- EP/BM/CFP：~6.9% missing
- DY：~26% missing
- ROE/ROA/NET_MARGIN：~54% missing

**告警阈值（warn）**：
- fundamentals missingness > 80% 触发硬告警
- 月度覆盖出现断档（非预期）触发告警

### 3.4 决策记录模板（需人工创建）
> 由于治理规则要求“重要决策记录由用户手动创建”，请在完成确认后创建：
> `decisions/20260205_de7_v2_route_selection.md`
>
> 建议内容：
> - 选择：Option 1（短期）+ Option 2（长期）
> - 风险：A0 interim 覆盖有限导致 ROE/ROA/NET_MARGIN 仍较高缺失
> - 回滚：保留 frozen v1，切换回旧路径
> - 迁移：shadow compare 通过后切换下游

## 4) Execute Mode 任务化（建议写入 execution_queue）
- DE7.1（补完）：契约拆分 + mapping + 文档修订（含 QA loop 修订）
- DE7.2（审计）：补齐 QA runner + guardrails + PIT 单测
- DE7.R1（升级调研）：v2 选型决策 + 阈值/验收标准 + 迁移策略

## 5) 证据与子代理输出
- Subagent artifacts:
  - `docs/subagents/runs/20260205_plan_mode_de7_repo_specs_retrieval/`
  - `docs/subagents/runs/20260205_plan_mode_de7_spec_drift/`
  - `docs/subagents/runs/20260205_plan_mode_de7_external_research/`
