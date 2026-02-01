# TaskCard: [EVAL_4_XXX]

> **Stage**: 4 · Backtest & Evaluation  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `EVAL_4_XXX` |
| **创建日期** | YYYY-MM-DD |
| **Role Mode** | `executor` |
| **Authority** | `speculative` |
| **Authorized By** | [Project Owner ID] |
| **上游 Task** | `DEV_3_XXX` |
| **Strategy Package** | `SP_XXX` |
| **Data Snapshot** | `DS_XXX` |

---

## 1. 输入锁定

### 1.1 版本锁定
| 依赖 | ID | Checksum | 锁定时间 |
|------|----|----------|----------|
| Strategy Package | `SP_XXX` | | |
| Data Snapshot | `DS_XXX` | | |
| Config | | | |

### 1.2 评估参数
```yaml
evaluation:
  seed: 42
  # 继承自 RESEARCH_1
  in_sample: [start, end]
  out_of_sample: [start, end]
  holdout: [start, end]
```

---

## 2. Backtest Run

### 2.1 方法论
- [ ] Walk-forward
- [ ] Rolling window
- [ ] Expanding window
- [ ] Fixed split

### 2.2 配置
```yaml
backtest:
  method: walk_forward
  train_window: 252  # trading days
  test_window: 63
  step_size: 21
  embargo_days: 5
  
  rebalance:
    frequency: daily
    timing: close
```

### 2.3 运行记录
| Run ID | 开始时间 | 结束时间 | 状态 |
|--------|----------|----------|------|
| | | | |

---

## 3. Performance Report

### 3.1 收益指标
| 指标 | In-Sample | Out-of-Sample | Holdout | 阈值 |
|------|-----------|---------------|---------|------|
| Annual Return | | | | |
| Sharpe Ratio | | | | ≥ 1.0 |
| Sortino Ratio | | | | |
| Max Drawdown | | | | ≤ 20% |
| Hit Rate | | | | |
| Win/Loss Ratio | | | | |
| Information Ratio | | | | |

### 3.2 收益曲线
<!-- 附上或链接收益曲线图 -->
```
图表路径: reports/EVAL_4_XXX/equity_curve.png
```

### 3.3 月度收益表
<!-- 附上或链接月度收益热力图 -->

---

## 4. Risk Attribution

### 4.1 因子暴露
| Factor | 平均暴露 | 暴露范围 | 贡献占比 |
|--------|----------|----------|----------|
| Market | | | |
| Size | | | |
| Value | | | |
| Momentum | | | |
| Volatility | | | |

### 4.2 Regime 敏感性
| Regime | 样本量 | 年化收益 | Sharpe | 说明 |
|--------|--------|----------|--------|------|
| Bull | | | | |
| Bear | | | | |
| High Vol | | | | |
| Low Vol | | | | |

### 4.3 尾部行为
| 指标 | 值 | 说明 |
|------|-----|------|
| VaR 95% | | |
| CVaR 95% | | |
| Max Daily Loss | | |
| 最大连续亏损天数 | | |

---

## 5. Robustness Tests

### 5.1 参数敏感性
| 参数 | 基准值 | 测试范围 | Sharpe 变化 | 稳定? |
|------|--------|----------|-------------|-------|
| | | | | |

### 5.2 子期间稳定性
| 子期间 | Sharpe | DD | 与整体差异 |
|--------|--------|-----|-----------|
| 2019 | | | |
| 2020 | | | |
| 2021 | | | |
| 2022 | | | |
| 2023 | | | |

### 5.3 压力测试
| 场景 | 日期范围 | 策略收益 | 基准收益 | 说明 |
|------|----------|----------|----------|------|
| COVID Crash | 2020-02 to 2020-03 | | | |
| 2022 Bear | 2022-01 to 2022-12 | | | |

---

## 6. Gate G3: Performance & Robustness

### 6.1 必须通过的检查
| 检查项 | 状态 | 实际值 | 阈值 | 通过? |
|--------|------|--------|------|-------|
| OOS Sharpe | `pending` | | ≥ 0.5 | |
| OOS Sharpe vs IS | `pending` | | 衰减 < 50% | |
| Max DD | `pending` | | ≤ 25% | |
| 子期间稳定性 | `pending` | | 全部 > 0 | |
| 参数敏感性 | `pending` | | 低敏感 | |
| 压力测试存活 | `pending` | | 无爆仓 | |

### 6.2 Gate 结果
- [ ] **PASS** → 进入 Stage 5
- [ ] **CONDITIONAL PASS** → 附条件通过，记录风险
- [ ] **FAIL** → 返回修改

### 6.3 Gate 证据
```
Gate G3 验证报告: reports/EVAL_4_XXX/gate_g3_report.md
Performance Report: reports/EVAL_4_XXX/performance.md
Robustness Report: reports/EVAL_4_XXX/robustness.md
```

---

## 7. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Evaluation Report | `reports/EVAL_4_XXX/` | `pending` |
| Backtest Results | `results/EVAL_4_XXX/` | `pending` |
| Risk Attribution | `reports/EVAL_4_XXX/risk.md` | `pending` |
| Gate G3 Report | `reports/EVAL_4_XXX/gate_g3.md` | `pending` |

---

## 8. 下游依赖

- **后续 TaskCard**: `RELEASE_5_XXX`
- **Stage 5 需要**: 
  - Complete Evaluation Report
  - Gate G3 Pass evidence
  - Identified failure modes

---

## 9. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: [Project Owner ID]
  scope: evaluation
  expires: YYYY-MM-DD
  
# Evaluation 结果是 Release Decision 的关键依据
# 任何数据或代码变更需重新 Eval
```

---

## 10. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| | | `task_created` | From DEV_3_XXX |
| | | `backtest_started` | |
| | | `backtest_completed` | |
| | | `gate_g3_evaluated` | |

---

*Template: TASKCARD_EVAL_4 v1.0.0*
