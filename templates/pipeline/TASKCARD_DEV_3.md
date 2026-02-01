# TaskCard: [DEV_3_XXX]

> **Stage**: 3 · Model/Strategy Build (Modular)  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `DEV_3_XXX` |
| **创建日期** | YYYY-MM-DD |
| **Role Mode** | `executor` / `builder` |
| **Authority** | `speculative` |
| **Authorized By** | [Project Owner ID] |
| **上游 Task** | `DATA_2_XXX` |
| **Data Snapshot** | `DS_XXX` |

---

## 1. 输入数据

### 1.1 数据依赖
| 依赖 | Snapshot ID | Checksum |
|------|-------------|----------|
| Data Snapshot | `DS_XXX` | `sha256:...` |
| Factor Library | `FL_XXX` | `sha256:...` |

### 1.2 配置继承
```yaml
# 继承自 RESEARCH_1_XXX
config_ref: configs/RESEARCH_1_XXX.yaml
```

---

## 2. Signal Module

### 2.1 Factor Transforms
| Transform ID | 输入 Factor | 输出 Signal | 方法 |
|--------------|-------------|-------------|------|
| T001 | | | zscore |
| T002 | | | rank |
| T003 | | | winsorize |

### 2.2 Model Inference (如适用)
```yaml
model:
  type: [linear / tree / neural / rule-based]
  framework: 
  architecture:
    
  training:
    method: 
    validation: 
```

### 2.3 Signal 输出规范
```yaml
signal_output:
  name: 
  range: [-1, 1]  # or [0, 1]
  frequency: daily
  type: cross-sectional
```

---

## 3. Portfolio Construction

### 3.1 构建方法
- [ ] Equal-weight
- [ ] Risk-parity
- [ ] Mean-variance
- [ ] Signal-weighted
- [ ] 其他: _______________

### 3.2 约束条件
| 约束 | 类型 | 值 |
|------|------|-----|
| Max single position | upper | 5% |
| Min position | lower | 0% |
| Sector neutrality | equality | |
| Beta exposure | range | [-0.1, 0.1] |
| Leverage | upper | 1.0 |

### 3.3 Turnover Control
```yaml
turnover:
  max_daily: 0.10  # 10% max
  penalty_factor: 0.001
  rebalance_frequency: daily
```

---

## 4. Cost Model

### 4.1 交易成本
| 成本项 | 模型 | 参数 |
|--------|------|------|
| Commission | fixed | bps |
| Slippage | sqrt-volume | |
| Market impact | linear | |

### 4.2 执行假设
```yaml
execution:
  latency_ms: 100
  partial_fill_model: 
  liquidity_threshold: 
```

---

## 5. Strategy Package

### 5.1 Package 结构
```
strategy_package/
├── config.yaml          # 完整配置
├── signal_module.py     # 信号生成
├── portfolio_module.py  # 组合构建
├── cost_model.py        # 成本模型
├── requirements.txt     # 依赖
└── checksums.yaml       # 校验和
```

### 5.2 版本信息
| 字段 | 值 |
|------|-----|
| **Package ID** | `SP_XXX` |
| **Git Commit** | |
| **Data Snapshot** | `DS_XXX` |
| **Config Hash** | |

---

## 6. Gate G2: Sanity Checks

### 6.1 必须通过的检查
| 检查项 | 状态 | 说明 |
|--------|------|------|
| Unit Tests Pass | `pending` | All tests green |
| No Look-ahead | `pending` | Leakage check |
| Cost Assumptions Valid | `pending` | Within bounds |
| Signal Range Valid | `pending` | Expected distribution |
| Reproducibility | `pending` | Same seed → same output |

### 6.2 Unit Test 清单
- [ ] `test_signal_module.py`
- [ ] `test_portfolio_construction.py`
- [ ] `test_cost_model.py`
- [ ] `test_no_lookahead.py`

### 6.3 Gate 结果
- [ ] **PASS** → 进入 Stage 4
- [ ] **FAIL** → 修复后重新验证

### 6.4 Gate 证据
```
Gate G2 验证报告: [路径]
pytest output: [路径]
```

---

## 7. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Strategy Package | `packages/SP_XXX/` | `pending` |
| Config (frozen) | `packages/SP_XXX/config.yaml` | `pending` |
| Test Report | `reports/DEV_3_XXX_tests.md` | `pending` |

---

## 8. 下游依赖

- **后续 TaskCard**: `EVAL_4_XXX`
- **Stage 4 需要**: 
  - Strategy Package ID
  - Frozen config
  - Data Snapshot ID

---

## 9. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: [Project Owner ID]
  scope: strategy_development
  expires: YYYY-MM-DD
  
# Strategy Package 一旦 freeze 后不可修改
# Bug fix 需要创建新版本
```

---

## 10. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| | | `task_created` | From DATA_2_XXX |
| | | `dev_started` | |
| | | `gate_g2_passed` | |

---

*Template: TASKCARD_DEV_3 v1.0.0*
