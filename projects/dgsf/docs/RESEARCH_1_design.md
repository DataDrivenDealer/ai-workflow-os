# RESEARCH_1_DGSF_001 — 研究设计文档

**文档 ID**: RESEARCH_1_design  
**Task**: RESEARCH_1_DGSF_001  
**Stage**: 1 · Research Design (Reproducibility First)  
**作者**: 陈研究 (Quant Researcher)  
**创建日期**: 2026-02-01  
**状态**: Active

---

## 0. 摘要

本文档定义 DGSF（Dynamic Grid Strategy Framework）动态网格策略的研究设计，包括信号定义、评估指标、消融实验计划和数据需求。

---

## 1. 研究假设

### 1.1 核心假设
> 通过动态调整网格间距（基于ATR自适应），可以在波动市场中实现稳定的均值回归收益，同时控制最大回撤在可接受范围内。

### 1.2 子假设
- **H1**: ATR自适应间距优于固定间距
- **H2**: 网格策略在震荡市优于趋势市
- **H3**: 仓位比例与风险收益比存在最优解

---

## 2. 信号定义

### 2.1 信号概述
| 属性 | 值 |
|------|-----|
| **信号名称** | DGSF Grid Position Signal |
| **信号类型** | mean-reversion |
| **频率** | intraday (1H) |
| **范围** | time-series (单标的) |

### 2.2 信号逻辑
```
输入:
  - price: 当前价格
  - grid_levels: 网格价位列表
  - positions: 当前持仓状态
  
处理:
  1. 检测价格是否触及网格线
  2. 若下穿网格: 生成买入信号
  3. 若上穿网格: 生成卖出信号
  4. 更新网格间距 (ATR自适应)
  
输出:
  - signal: {-1: 卖出, 0: 持有, 1: 买入}
  - size: 交易数量
  - grid_update: 新网格参数
```

### 2.3 网格间距计算
```python
def calculate_grid_spacing(atr: float, base_spacing: float = 0.01) -> float:
    """
    基于ATR动态调整网格间距
    
    - 高波动 (ATR高): 增大间距，减少触发频率
    - 低波动 (ATR低): 减小间距，增加交易机会
    """
    volatility_factor = atr / atr_baseline
    adjusted_spacing = base_spacing * volatility_factor
    return np.clip(adjusted_spacing, 0.005, 0.03)  # 0.5% - 3%
```

---

## 3. 成功指标

### 3.1 主要指标
| 指标 | 阈值 | 说明 | 验证方法 |
|------|------|------|----------|
| **Sharpe Ratio** | ≥ 1.5 | 年化风险调整收益 | 回测计算 |
| **Max Drawdown** | ≤ 15% | 最大回撤控制 | 回测计算 |
| **Win Rate** | ≥ 60% | 单笔交易胜率 | 交易统计 |
| **Grid Fill Rate** | ≥ 80% | 网格订单成交率 | 回测模拟 |

### 3.2 次要指标
| 指标 | 目标 | 说明 |
|------|------|------|
| Calmar Ratio | ≥ 2.0 | 收益/最大回撤 |
| Recovery Time | ≤ 30 days | 回撤恢复时间 |
| Sortino Ratio | ≥ 2.0 | 下行风险调整收益 |

---

## 4. 消融实验计划

### 4.1 实验矩阵

| 变体ID | 变量 | 值范围 | 目的 |
|--------|------|--------|------|
| A1 | grid_spacing | [0.5%, 1%, 1.5%, 2%] | 确定最优间距 |
| A2 | position_ratio | [5%, 10%, 15%] | 确定最优仓位 |
| A3 | adaptive_factor | [ATR, Bollinger, Fixed] | 比较自适应方法 |
| A4 | stop_loss | [10%, 15%, 20%] | 优化止损阈值 |

### 4.2 基线配置
```yaml
baseline:
  grid_spacing: 1%
  position_ratio: 10%
  adaptive_factor: ATR
  stop_loss: 15%
```

### 4.3 实验运行规则
1. 每个变体独立运行
2. 其他参数保持基线值
3. 使用相同随机种子
4. 记录完整指标集

---

## 5. 数据需求

### 5.1 Universe
- BTC/USDT
- ETH/USDT

### 5.2 时间范围
| 数据集 | 范围 | 用途 |
|--------|------|------|
| In-Sample | 2020-01-01 ~ 2023-12-31 | 训练/调参 |
| Out-of-Sample | 2024-01-01 ~ 2024-12-31 | 验证 |
| Holdout | 2025-01-01 ~ 2025-12-31 | 最终测试 |

### 5.3 数据字段
- timestamp (UTC)
- open, high, low, close
- volume, quote_volume

---

## 6. 风险控制

### 6.1 单笔风险
- 最大单笔亏损: 2% of capital
- 止损触发: 强制平仓

### 6.2 组合风险
- 最大持仓: 10个网格层级
- 最大杠杆: 1x (无杠杆)
- 最大回撤警告: 10%
- 最大回撤熔断: 15%

---

## 7. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| 研究设计文档 | `projects/dgsf/docs/RESEARCH_1_design.md` | ✅ Complete |
| 实验配置 | `projects/dgsf/configs/RESEARCH_1.yaml` | ✅ Complete |
| 可复现性包 | `projects/dgsf/docs/RESEARCH_1_repro.md` | ✅ Complete |

---

## 8. 下游依赖

- **后续任务**: `DATA_2_DGSF_001` (数据工程)
- **Gate要求**: Stage 1 无强制Gate，但需完成所有Verification项

---

*本文档遵循 AI Workflow OS 的 PROJECT_DELIVERY_PIPELINE Stage 1 规范*
