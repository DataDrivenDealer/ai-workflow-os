# TaskCard: [RESEARCH_1_XXX]

> **Stage**: 1 · Research Design (Reproducibility First)  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `RESEARCH_1_XXX` |
| **创建日期** | YYYY-MM-DD |
| **Role Mode** | `architect` / `planner` |
| **Authority** | `speculative` |
| **Authorized By** | [Project Owner ID] |
| **上游 Task** | `RESEARCH_0_XXX` |

---

## 1. 假设与目标

### 1.1 Signal 定义
<!-- 精确描述要构建的信号 -->
- **信号名称**: 
- **信号类型**: (momentum / mean-reversion / fundamental / ...)
- **信号频率**: (daily / intraday / weekly)
- **信号范围**: (cross-sectional / time-series)

### 1.2 预期适用 Regime
<!-- 描述信号预期在何种市场状态下有效 -->
- **适用环境**: 
- **失效环境**: 
- **Regime 判别指标**: 

### 1.3 成功指标 (Success Metrics)
| 指标 | 阈值 | 说明 |
|------|------|------|
| Sharpe Ratio | ≥ | |
| Max Drawdown | ≤ | |
| Hit Rate | ≥ | |
| Information Ratio | ≥ | |
| 其他: | | |

---

## 2. 实验计划

### 2.1 消融实验 (Ablations)
<!-- 列出需要测试的变体 -->
| 变体 ID | 变量 | 取值范围 | 目的 |
|---------|------|----------|------|
| A1 | | | |
| A2 | | | |

### 2.2 基线比较 (Baselines)
<!-- 列出用于比较的基准 -->
| 基线 ID | 描述 | 预期表现 |
|---------|------|----------|
| B0 | Equal-weighted benchmark | |
| B1 | Simple momentum | |
| B2 | | |

### 2.3 评估时间范围
| 区间 | 开始 | 结束 | 用途 |
|------|------|------|------|
| In-Sample | | | Training |
| Out-of-Sample | | | Validation |
| Holdout | | | Final Test |

---

## 3. Reproducibility Package (Draft)

### 3.1 配置规范
```yaml
# 配置文件路径: projects/[project]/configs/[config].yaml
experiment:
  name: RESEARCH_1_XXX
  seed: 42
  
data:
  universe: 
  start_date: 
  end_date: 
  
model:
  type: 
  parameters:
    
evaluation:
  metrics: [sharpe, max_dd, hit_rate]
  rebalance_freq: 
```

### 3.2 脚本/Notebook 映射
| 文件 | 用途 | 依赖 |
|------|------|------|
| `01_data_prep.py` | 数据准备 | |
| `02_signal_build.py` | 信号构建 | 01 |
| `03_backtest.py` | 回测 | 01, 02 |
| `04_analysis.ipynb` | 分析 | 03 |

### 3.3 环境要求
```
# requirements.txt 新增依赖
```

---

## 4. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Research Design Doc | `docs/research/RESEARCH_1_XXX_design.md` | `pending` |
| Experiment Config | `configs/RESEARCH_1_XXX.yaml` | `pending` |
| Repro Package Manifest | `docs/research/RESEARCH_1_XXX_repro.md` | `pending` |

---

## 5. Gate & 下游依赖

- **Gate**: 无 (Stage 1 无 Gate)
- **后续 TaskCard**: `DATA_2_XXX`
- **Stage 2 需要**: 
  - Finalized data requirements
  - Universe definition
  - Date range specification

---

## 6. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: [Project Owner ID]
  scope: research_design
  expires: YYYY-MM-DD
  
# 注意：Research Design 需要 accept 后方可进入 Data Engineering
# Design 变更需要重新审批
```

---

## 7. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| | | `task_created` | From RESEARCH_0_XXX |

---

*Template: TASKCARD_RESEARCH_1 v1.0.0*
