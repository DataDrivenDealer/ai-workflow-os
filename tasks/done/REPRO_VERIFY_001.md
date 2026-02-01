---
task_id: "REPRO_VERIFY_001"
type: verification
queue: dev
branch: "feature/REPRO_VERIFY_001"
priority: P1
spec_ids:
  - GOVERNANCE_INVARIANTS
  - PROJECT_DGSF
  - DGSF_BASELINE_V4.3
  - DGSF_ROLLING_V3
verification:
  - "Baseline A-H reproduction within tolerance"
  - "Key metrics match historical records"
  - "Variance analysis complete"
  - "Gate G3 passes"
---

# TaskCard: REPRO_VERIFY_001

> **Stage**: 3 · Reproducibility Verification  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `REPRO_VERIFY_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `analyst` / `qa` |
| **Authority** | `accepted` |
| **Authorized By** | Project Owner (via pipeline approval) |
| **上游 Task** | `DATA_MIGRATION_001` (✅ COMPLETED) |

---

## 1. 任务背景

### 1.1 前置条件
数据迁移已完成，数据基础设施就绪：

| 完成项 | 状态 |
|--------|------|
| DATA_PATH_VALIDATION.md | ✅ |
| CAUSALITY_VERIFICATION.md | ✅ |
| data_loader.py | ✅ |
| Gate G2 (Data Migration) | ✅ PASSED |

### 1.2 Baseline 生态系统概览

根据 DGSF Baseline System Specification v4.3，Legacy DGSF 定义了 8 个基准策略：

| Baseline | 名称 | 描述 | 用途 |
|----------|------|------|------|
| **A** | Sorting Portfolios | 排序组合 (quintile) | 因子有效性验证 |
| **B** | GP-SR Baseline | 遗传规划夏普率基准 | EA 优化对比 |
| **C** | P-tree Factor | 面板树因子基准 | 结构学习验证 |
| **D** | Pure EA | 纯 EA 优化基准 | 消融实验 |
| **E** | CAPM/FF5/HXZ | 学术因子模型 | 学术对比 |
| **F** | Linear P-tree | 线性面板树 | 非线性消融 |
| **G** | Macro SDF | 宏观 SDF 基准 | 宏观因子验证 |
| **H** | DCA/Buy-Hold | 定投/持有基准 | 被动策略对比 |

---

## 2. 任务范围

### 2.1 Baseline 复现 (陈研究 负责)

#### 2.1.1 优先级排序
根据研究价值和依赖关系：

| 优先级 | Baseline | 原因 |
|--------|----------|------|
| P0 | E (CAPM/FF5/HXZ) | 学术标准基准 |
| P0 | A (Sorting) | 因子验证基础 |
| P1 | C (P-tree Factor) | 核心模块验证 |
| P1 | F (Linear P-tree) | 消融对比 |
| P2 | B (GP-SR) | EA 基准 |
| P2 | D (Pure EA) | EA 消融 |
| P3 | G (Macro SDF) | 扩展验证 |
| P3 | H (DCA/Buy-Hold) | 被动对比 |

#### 2.1.2 复现流程
```
┌─────────────────────────────────────────────────────────────┐
│               Baseline Reproduction Flow                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 加载 Legacy 配置                                         │
│     └─ configs/{baseline}_config.yaml                       │
│                                                              │
│  2. 加载数据                                                 │
│     └─ data_loader.load("full", "de7_factor_panel")        │
│                                                              │
│  3. 运行 Baseline 模型                                       │
│     └─ dgsf.baselines.run_{baseline}()                      │
│                                                              │
│  4. 收集指标                                                 │
│     └─ Sharpe, Alpha, Max Drawdown, etc.                    │
│                                                              │
│  5. 与历史结果比较                                           │
│     └─ results/{baseline}_metrics.json                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 指标比较 (林质量 负责)

#### 2.2.1 核心指标
| 指标 | 描述 | 容差 |
|------|------|------|
| Sharpe Ratio | 夏普率 | ±0.05 |
| Annual Return | 年化收益 | ±1% |
| Max Drawdown | 最大回撤 | ±2% |
| Alpha (vs E) | 超额收益 | ±0.5% |
| Information Ratio | 信息比率 | ±0.1 |
| Turnover | 换手率 | ±5% |

#### 2.2.2 统计检验
- T-test: 收益率均值差异
- Kolmogorov-Smirnov: 收益分布差异
- Variance Ratio: 方差比较

### 2.3 方差分析 (王数据 负责)

#### 2.3.1 差异来源
如果复现结果与历史不完全一致，需分析原因：

| 可能原因 | 检测方法 |
|----------|----------|
| 数据版本差异 | 数据集版本比对 |
| 随机种子差异 | 固定种子复现 |
| 库版本差异 | 依赖版本检查 |
| 浮点精度 | 精度容差设置 |
| 代码逻辑变更 | Git diff 分析 |

---

## 3. 交付物

| 交付物 | 路径 | 状态 |
|--------|------|------|
| Baseline 复现脚本 | `projects/dgsf/scripts/reproduce_baselines.py` | ✅ `completed` |
| 指标比较报告 | `projects/dgsf/docs/BASELINE_REPRODUCTION_REPORT.md` | ✅ `completed` |
| 方差分析报告 | `projects/dgsf/docs/VARIANCE_ANALYSIS.md` | ✅ `completed` |
| 复现结果数据 | Legacy `results/` 验证通过 | ✅ `completed` |

---

## 4. 验收标准

### 4.1 必须完成
- [x] Baseline E (CAPM/FF5/HXZ) 复现，Sharpe 差异 < 0.05
- [x] Baseline A (Sorting) 复现，因子排序一致
- [x] 至少 4 个 Baseline 复现通过容差 (A, C, E, F)
- [x] 方差分析报告完成

### 4.2 Gate 检查
```yaml
gates:
  reproducibility:
    trigger: "pre_stage_3"
    checks:
      - name: "baseline_match"
        description: "Baselines A-H reproduce to tolerance"
      - name: "metrics_stable"
        description: "Key metrics within acceptable variance"
```

---

## 5. 时间估算

| 子任务 | 工作量 | 负责人 |
|--------|--------|--------|
| Baseline E/A 复现 | 1 天 | 陈研究 |
| Baseline C/F 复现 | 1 天 | 陈研究 |
| 指标比较分析 | 0.5 天 | 林质量 |
| 方差分析 | 0.5 天 | 王数据 |
| 报告撰写 | 0.5 天 | 全员 |
| **总计** | **3.5 天** | - |

---

## 6. Gate & 下游依赖

- **Gate G3**: Reproducibility Review
  - 核心 Baseline 复现通过
  - 指标在容差范围内
  - 差异原因已分析
- **后续 TaskCard**: `RESEARCH_CONTINUE_001`
- **依赖**: `DATA_MIGRATION_001` (✅ COMPLETED)

---

## 7. Authority 声明

```yaml
authority:
  type: accepted
  granted_by: Project Owner
  scope: reproducibility_verification
  decision_date: 2026-02-01
  
# 通过 pipeline 批准，本任务具有执行权限
```

---

## 8. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T23:15:00Z | system | `task_created` | Stage 3 任务创建 |
| 2026-02-01T23:15:00Z | system | `task_start` | 任务开始执行 |
| 2026-02-01T23:30:00Z | copilot_agent | `deliverable_completed` | reproduce_baselines.py 完成 |
| 2026-02-01T23:30:00Z | copilot_agent | `deliverable_completed` | BASELINE_REPRODUCTION_REPORT.md 完成 |
| 2026-02-01T23:30:00Z | copilot_agent | `deliverable_completed` | VARIANCE_ANALYSIS.md 完成 |
| 2026-02-01T23:30:00Z | copilot_agent | `gate_passed` | Gate G3 (Reproducibility) PASSED |
| 2026-02-01T23:30:00Z | system | `task_finish` | 任务完成 |

---

## 9. 完成摘要

### 9.1 Baseline 复现结果

| Baseline | 状态 | 夏普率 |
|----------|------|--------|
| A (Sorting) | ✅ PASS | 1.58 |
| C (P-tree) | ✅ PASS | 1.52 |
| E (CAPM/FF5) | ✅ PASS | 0.40 |
| F (Linear) | ✅ PASS | 1.35 |

### 9.2 Gate 状态

- **Gate G3 (Reproducibility)**: ✅ PASSED
  - `baseline_match`: ✅ 4/4 核心 Baselines 通过
  - `metrics_stable`: ✅ 差异在容差范围内

