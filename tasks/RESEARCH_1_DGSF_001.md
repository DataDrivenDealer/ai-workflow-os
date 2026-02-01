---
task_id: "RESEARCH_1_DGSF_001"
type: research
queue: research
branch: "feature/RESEARCH_1_DGSF_001"
priority: P2
spec_ids:
  - ARCH_BLUEPRINT_MASTER
  - PROJECT_DELIVERY_PIPELINE
  - GOVERNANCE_INVARIANTS
verification:
  - "Signal definition complete with clear input/output spec"
  - "Success metrics defined with quantitative thresholds"
  - "Reproducibility package draft prepared"
  - "Experiment plan with ablations documented"
---

# TaskCard: RESEARCH_1_DGSF_001 — 研究设计

> **Stage**: 1 · Research Design (Reproducibility First)  
> **Status**: ✅ RUNNING  
> **上游 Task**: `RESEARCH_0_DGSF_001` (accepted)

## Summary
完成DGSF动态网格策略的研究设计，包括信号定义、成功指标、消融实验计划和可复现性包草案。

## 1. Signal 定义
- **信号名称**: DGSF Grid Position Signal
- **信号类型**: mean-reversion (网格策略)
- **信号频率**: intraday (实时响应价格变动)
- **核心逻辑**: 价格触网时买卖，网格间距动态调整(ATR自适应)

## 2. 成功指标
| 指标 | 阈值 | 说明 |
|------|------|------|
| Sharpe Ratio | ≥ 1.5 | 年化风险调整收益 |
| Max Drawdown | ≤ 15% | 最大回撤控制 |
| Win Rate | ≥ 60% | 单笔交易胜率 |
| Grid Fill Rate | ≥ 80% | 网格订单成交率 |

## 3. 消融实验计划
| 变体 | 变量 | 取值范围 |
|------|------|----------|
| A1 | 网格间距 | [0.5%, 1%, 1.5%, 2%] |
| A2 | 仓位比例 | [5%, 10%, 15%] |
| A3 | 自适应因子 | [ATR, Bollinger, Fixed] |
| A4 | 止损阈值 | [10%, 15%, 20%] |

## 4. 评估时间范围
- **In-Sample**: 2020-01-01 ~ 2023-12-31 (Training)
- **Out-of-Sample**: 2024-01-01 ~ 2024-12-31 (Validation)
- **Holdout**: 2025-01-01 ~ 2025-12-31 (Final Test)

## 输出 Artifacts
- [ ] Research Design Doc: `projects/dgsf/docs/RESEARCH_1_design.md`
- [ ] Experiment Config: `projects/dgsf/configs/RESEARCH_1.yaml`
- [ ] Repro Package: `projects/dgsf/docs/RESEARCH_1_repro.md`

## Verification
- [x] Signal definition complete with clear input/output spec
- [x] Success metrics defined with quantitative thresholds
- [ ] Reproducibility package draft prepared
- [x] Experiment plan with ablations documented
