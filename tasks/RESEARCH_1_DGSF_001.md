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

# TaskCard: RESEARCH_1_DGSF_001

> **Stage**: 1 · Research Design (Reproducibility First)  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `RESEARCH_1_DGSF_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `architect` / `planner` |
| **Authority** | `speculative` |
| **Authorized By** | Project Owner |
| **上游 Task** | `RESEARCH_0_DGSF_001` |

---

## 1. 假设与目标

### 1.1 Signal 定义
- **信号名称**: DGSF Grid Position Signal
- **信号类型**: mean-reversion (网格策略)
- **信号频率**: intraday (实时响应价格变动)
- **信号范围**: time-series (单标的动态网格)

**核心逻辑**:
```
当价格触及网格线时:
  - 下跌触网: 买入固定数量
  - 上涨触网: 卖出固定数量
  - 网格间距: 动态调整 (波动率自适应)
```

### 1.2 预期适用 Regime
- **适用环境**: 震荡市、区间波动、低趋势性行情
- **失效环境**: 强趋势行情、单边快速移动
- **Regime 判别指标**: 
  - ATR (Average True Range) 相对稳定
  - ADX < 25 (低趋势强度)
  - 价格在布林带内震荡

### 1.3 成功指标 (Success Metrics)
| 指标 | 阈值 | 说明 |
|------|------|------|
| Sharpe Ratio | ≥ 1.5 | 年化风险调整收益 |
| Max Drawdown | ≤ 15% | 最大回撤控制 |
| Win Rate | ≥ 60% | 单笔交易胜率 |
| Daily PnL Volatility | ≤ 2% | 日收益波动控制 |
| Grid Fill Rate | ≥ 80% | 网格订单成交率 |

---

## 2. 实验计划

### 2.1 消融实验 (Ablations)
| 变体 ID | 变量 | 取值范围 | 目的 |
|---------|------|----------|------|
| A1 | 网格间距 | [0.5%, 1%, 1.5%, 2%] | 找最优间距 |
| A2 | 仓位比例 | [5%, 10%, 15%] | 单格仓位敏感性 |
| A3 | 动态调整因子 | [ATR, Bollinger Width, Fixed] | 自适应方法对比 |
| A4 | 止损阈值 | [10%, 15%, 20%] | 风控参数优化 |

### 2.2 基线比较 (Baselines)
| 基线 ID | 描述 | 预期表现 |
|---------|------|----------|
| B0 | Buy-and-Hold | 基准对比 |
| B1 | Fixed Grid (1% spacing) | 静态网格 |
| B2 | Simple Mean-Reversion (RSI) | 传统均值回归 |
| B3 | DCA (Dollar Cost Averaging) | 定投策略 |

### 2.3 评估时间范围
| 区间 | 开始 | 结束 | 用途 |
|------|------|------|------|
| In-Sample | 2020-01-01 | 2023-12-31 | Training & Optimization |
| Out-of-Sample | 2024-01-01 | 2024-12-31 | Validation |
| Holdout | 2025-01-01 | 2025-12-31 | Final Test (不参与优化) |

---

## 3. Reproducibility Package (Draft)

### 3.1 配置规范
```yaml
# 配置文件路径: projects/dgsf/configs/RESEARCH_1_DGSF_001.yaml
experiment:
  name: RESEARCH_1_DGSF_001
  seed: 42
  version: "1.0.0"
  
data:
  universe: ["BTC/USDT", "ETH/USDT"]  # 初始测试标的
  source: "exchange_api"              # 数据来源
  frequency: "1h"                     # 采样频率
  start_date: "2020-01-01"
  end_date: "2025-12-31"
  
model:
  type: "dynamic_grid"
  parameters:
    base_grid_spacing: 0.01           # 基础网格间距 1%
    position_per_grid: 0.10           # 单格仓位 10%
    max_position: 1.0                 # 最大总仓位 100%
    adaptive_method: "atr"            # ATR自适应
    stop_loss: 0.15                   # 15% 止损
    
evaluation:
  metrics: [sharpe, max_dd, win_rate, grid_fill_rate]
  rebalance_freq: "continuous"        # 实时响应
  slippage: 0.001                     # 滑点假设 0.1%
  commission: 0.001                   # 手续费 0.1%
```

### 3.2 脚本/Notebook 映射
| 文件 | 用途 | 依赖 |
|------|------|------|
| `01_data_fetch.py` | 获取历史数据 | exchange API |
| `02_grid_signal.py` | 网格信号生成 | 01 |
| `03_backtest.py` | 回测框架 | 01, 02 |
| `04_ablation.py` | 消融实验 | 01, 02, 03 |
| `05_analysis.ipynb` | 结果分析 | 04 |

### 3.3 环境要求
```
# requirements.txt 新增依赖 (projects/dgsf/requirements.txt)
pandas>=2.0
numpy>=1.24
ccxt>=4.0          # 交易所API
vectorbt>=0.26     # 回测框架
plotly>=5.18       # 可视化
```

---

## 4. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Research Design Doc | `projects/dgsf/docs/RESEARCH_1_design.md` | ✅ `complete` |
| Experiment Config | `projects/dgsf/configs/RESEARCH_1.yaml` | ✅ `complete` |
| Repro Package Manifest | `projects/dgsf/docs/RESEARCH_1_repro.md` | ✅ `complete` |
| Ablation Results | `projects/dgsf/results/ablation_report.md` | `pending` (Stage 4) |

---

## 5. Gate & 下游依赖

- **Gate**: 无 (Stage 1 无 Gate)
- **后续 TaskCard**: `DATA_2_DGSF_001`
- **Stage 2 需要**: 
  - Finalized data requirements (OHLCV + Volume)
  - Universe definition (BTC/USDT, ETH/USDT)
  - Date range specification (2020-2025)
  - Data quality criteria

---

## 6. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: Project Owner
  scope: research_design
  expires: 2026-03-31
  
# 注意：Research Design 需要 accept 后方可进入 Data Engineering
# Design 变更需要重新审批
```

---

## 7. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T21:00:00Z | liu_pm | `task_created` | From RESEARCH_0_DGSF_001 |
| 2026-02-01T21:00:00Z | liu_pm | `design_draft` | Initial research design |

---

*Template: TASKCARD_RESEARCH_1 v1.0.0*  
*Created by: 刘PM (Project Manager)*
