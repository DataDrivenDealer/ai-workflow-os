# DGSF Research Roadmap 2026

> **Document ID**: RESEARCH_ROADMAP_2026  
> **Version**: 2.0.0 (Corrected)  
> **Created**: 2026-02-01  
> **Last Updated**: 2026-02-01  
> **Status**: ACTIVE

---

## ⚠️ 状态校准说明 (v2.0.0)

本版本根据实际开发进度进行了重要校准：

| 层级 | 之前假设 | 实际状态 | 修正 |
|------|----------|----------|------|
| L2: PanelTree | 已完成 | ✅ 初步验证完成 | 正确 |
| L3: SDF | 已完成 | 🔵 规范定稿，进入开发 | **需修正** |
| L4: EA | 即将开始 | ✅ 设计完成，未进入开发 | **需修正** |
| L5: Rolling | 已集成 | ⏳ 待 SDF/EA 联调 | **需修正** |

---

## 1. Executive Summary

本路线图基于 **实际开发状态** 规划 2026 年度研究开发方向。当前核心任务是：

> **以规范驱动 SDF 层模块开发，完成与 PanelTree 的联调验证**

### 1.1 战略目标 (修正后)

| 目标 | 描述 | 优先级 | 预期完成 |
|------|------|--------|----------|
| **G1** | SDF 层模块开发完成 | P0 | 2026-02 W3 |
| **G2** | EA 层模块开发完成 | P0 | 2026-03 W1 |
| **G3** | Full Pipeline 端到端运行 | P0 | 2026-03 W2 |
| **G4** | Baseline A-H 完整复现 | P1 | 2026-03 W3 |
| **G5** | 新实验设计与执行 | P1 | 2026-Q2 |
| **G6** | 学术发表 | P2 | 2026-Q3 |

## 2. 当前开发状态 (Ground Truth)

### 2.1 层级开发进度

| 层级 | 版本 | 规范状态 | 代码状态 | 下一步 |
|------|------|----------|----------|--------|
| **L0-L1: DataEng** | v4.2 | ✅ FINAL | 🔵 70% | **数据扩展** |
| **L2: PanelTree** | v3.0.2 | ✅ FINAL | ✅ 核心完成 | SDF 联调 |
| **L3: SDF** | v3.1 | ✅ FINAL | 🔵 45% | **当前重点** |
| **L4: EA** | v3.1 | ✅ FINAL | ⏳ 30% | 待 SDF 完成 |
| **L5: Rolling** | v3.0 | ✅ FINAL | ⏳ 50% | 待联调 |
| **L6-L7: Report** | - | ⏳ | ⏳ | 后期开发 |

### 2.2 DataEng 层详细状态 ⚠️

**已完成**:
- ✅ 数据工程规范 v4.2 (FINAL)
- ✅ 因果性数据管道框架
- ✅ Parquet/Arrow 存储方案
- ✅ 中证800 成分股历史数据回测

**⚠️ 待完成 (关键缺口)**:
- ❌ **全量 A 股日频数据回滚回测** — 当前仅使用中证800
- ❌ 全量股票池数据获取与清洗
- ❌ 全量数据因果性验证
- ❌ 全量数据 Rolling Window 适配

> **说明**: 当前回测基于 **中证800 成分股**，样本量和代表性有限。
> 完整的、有说服力的、可复现的研究需要 **全量 A 股 (~5000 只) 日频数据**。
> 这是后续必须完成的关键工作项。

### 2.3 SDF 层详细状态

**已完成**:
- ✅ SDF Layer Specification v3.1 (FINAL)
- ✅ SDF Layer Design & Mathematical Review Note
- ✅ SDF Layer Review Checklist
- ✅ State Engine Spec v1.0
- ✅ 基础模型代码 (model.py, losses.py)
- ✅ 数据加载器 (a0_sdf_dataloader.py)

**待开发**:
- ⏳ State Engine 完整实现 (XState encoder)
- ⏳ Robust Moment Estimation (MAD + clip)
- ⏳ Training Pipeline (SmoothMax + τ schedule)
- ⏳ EA Pricing Oracle API (PE(w))
- ⏳ PanelTree 联调验证

### 2.3 EA 层详细状态

**已完成**:
- ✅ EA Layer Specification v3.1 (FINAL)
- ✅ 基础框架 (core.py, nsga2_optimizer.py, objectives.py)
- ✅ Fitness adapter 框架

**待开发**:
- ⏳ NSGA-III 实现
- ⏳ SDF Consistency Constraint
- ⏳ HV-aware exploration
- ⏳ Drift-aware warm-start

---

## 3. 开发路线图 (Phase-Based)

### Phase 0: Data Expansion (并行任务)
**时间**: 2026-02 ~ 2026-03 (与 Phase 1-3 并行)

> ⚠️ **关键缺口**: 当前仅使用中证800成分股回测，需扩展到全量A股

```
数据扩展任务 (DATA_EXPANSION_001):
├── 全量 A 股股票池获取 (~5000 只)
│   ├── 剔除 ST/退市/新股
│   └── 历史成分股变动处理
├── 日频数据采集与清洗
│   ├── 行情数据 (OHLCV)
│   ├── 财务数据 (季频)
│   └── 特征因子 (94 个标准特征)
├── 因果性验证
│   └── 确保 t 时刻只用 t 及之前数据
└── Rolling Window 数据适配
    └── 支持 2015-2025 完整回滚
```

**数据规模估算**:
| 数据类型 | 当前 (中证800) | 目标 (全量A股) | 增量 |
|----------|----------------|----------------|------|
| 股票数量 | ~800 | ~5000 | 6x |
| 日期范围 | 2015-2023 | 2015-2025 | +2年 |
| 数据量 | ~1.25 GB | ~8-10 GB | 8x |

**交付物**:
- [ ] 全量A股日频数据集
- [ ] 数据质量验证报告
- [ ] 因果性测试通过

---

### Phase 1: SDF Layer Development (当前阶段)
**时间**: 2026-02 W1-W3 (2月1日 - 2月21日)

```
Week 1 (02/01-02/07): 规范审核 + State Engine
├── Day 1-2: SDF Review Checklist 逐条审核
├── Day 3-4: State Engine v1.0 实现
│   ├── XState encoder (Vol/Liq/Crowd)
│   └── Instrument basis [1, V, L, V·L]
└── Day 5-7: 单元测试 + 代码审查

Week 2 (02/08-02/14): SDF Model + Robust Moments
├── Day 1-2: SDF 模型整合 (model.py 生产化)
├── Day 3-4: Robust Moment Estimation
│   ├── MAD scaling
│   └── Clip/Huber aggregation
└── Day 5-7: Instrumented moment G[i,j] 实现

Week 3 (02/15-02/21): Training Pipeline + Integration
├── Day 1-2: SmoothMax objective + τ schedule
├── Day 3-4: EA Pricing Oracle API
│   └── PE(w) = SmoothMax(|g_j(w)|)
└── Day 5-7: PanelTree 联调验证
```

**交付物**:
- [ ] SDF_SPEC_REVIEW_001 完成报告
- [ ] sdf/ 模块生产级代码
- [ ] 单元测试覆盖 >80%
- [ ] PanelTree → SDF 端到端验证

**里程碑**:
- M1 (02/03): SDF 规范审核完成
- M2 (02/14): SDF 模块 Alpha 版本
- M3 (02/21): SDF-PanelTree 联调通过

---

### Phase 2: EA Layer Development
**时间**: 2026-02 W4 - 2026-03 W1 (2月22日 - 3月7日)

```
Week 4 (02/22-02/28): NSGA-III Core
├── Day 1-2: NSGA-III 算法实现
├── Day 3-4: 4-objective 结构
│   ├── Sharpe / MDD / Turnover / SDF Penalty
│   └── Pareto frontier 构建
└── Day 5-7: 单元测试

Week 5 (03/01-03/07): EA Integration
├── Day 1-2: SDF Consistency Constraint
├── Day 3-4: HV-aware exploration
├── Day 5-6: Drift-aware warm-start
└── Day 7: EA-SDF 联调验证
```

**交付物**:
- [ ] EA_DEV_001 完成
- [ ] ea/ 模块生产级代码
- [ ] EA-SDF 接口验证

**里程碑**:
- M4 (03/07): EA 模块 Alpha 版本

---

### Phase 3: Full Pipeline Integration
**时间**: 2026-03 W2-W3 (3月8日 - 3月21日)

```
Week 6 (03/08-03/14): Rolling Window Pipeline
├── DataEng → PanelTree → SDF → EA → Rolling
├── 完整数据流验证
└── 因果性端到端检查

Week 7 (03/15-03/21): Baseline Reproduction
├── Baseline A-H 完整复现
├── Sharpe tolerance ±0.05 验证
└── Evidence pack 生成
```

**交付物**:
- [ ] Full Pipeline 运行脚本
- [ ] Baseline 复现报告
- [ ] 性能基准文档

**里程碑**:
- M5 (03/14): Full Pipeline 端到端运行
- M6 (03/21): Baseline A-H 复现完成

---

### Phase 4: Validation & Research
**时间**: 2026-03 W4 - 2026-Q2

```
Week 8+ (03/22-04/30): OOS Validation
├── 全量 A 股数据 OOS 验证 (依赖 DATA_EXPANSION_001)
├── Robustness 检验
└── Ablation 实验

Q2 (05-06): Research & Publication
├── 新实验设计执行
├── 论文撰写
└── 会议投稿 (NeurIPS WS / ICAIF)
```

> **注意**: Phase 4 的 OOS 验证需要全量 A 股数据支撑，
> 确保 DATA_EXPANSION_001 在此之前完成。

---

## 4. 周度计划详细

### Week 1 (当前): SDF 规范审核 + State Engine

| 日期 | 任务 | 负责人 | 工时 | 状态 |
|------|------|--------|------|------|
| 02/01 | 启动专家评审会议 | 全员 | 2h | ✅ |
| 02/01-02 | SDF Review Checklist 审核 | 陈首席+李架构 | 4h | 🔵 |
| 02/03-04 | State Engine v1.0 实现 | 李架构 | 8h | ⏳ |
| 02/05-06 | XState encoder 单元测试 | 赵测试 | 4h | ⏳ |
| 02/07 | Week 1 Review | 全员 | 2h | ⏳ |

### SDF Review Checklist 审核项

基于 [SDF Layer Review Checklist](../legacy/DGSF/docs/SDF%20Layer%20Review%20Checklist%20.md):

| # | 模块 | 审核项 | 决策 |
|---|------|--------|------|
| 1 | Set Encoder | mean pooling vs attention | 待审 |
| 2 | XState | Instrument basis 维度 J=4 or 5 | 待审 |
| 3 | SDF Param | boundedness c=4.0 | ✅ 采纳 |
| 4 | Robust Moments | MAD vs EWMA | 待审 |
| 5 | Minimax | τ schedule 参数 | 待审 |
| 6 | EA Oracle | PE(w) 定义 | 待审 |

---

## 5. 任务卡规划

### 5.1 即将创建的任务卡

| Task ID | 名称 | 类型 | 优先级 | 依赖 | 负责人 |
|---------|------|------|--------|------|--------|
| `SDF_SPEC_REVIEW_001` | SDF 规范最终审核 | review | P0 | - | 陈首席 |
| `SDF_DEV_001` | SDF 层模块开发 | dev | P0 | SDF_SPEC_REVIEW_001 | 李架构 |
| `DATA_EXPANSION_001` | 全量A股数据扩展 | data | P1 | - | 王数据 |
| `SDF_INTEGRATION_001` | SDF-PanelTree 集成 | dev | P0 | SDF_DEV_001 | 李架构 |
| `EA_DEV_001` | EA 层模块开发 | dev | P1 | SDF_INTEGRATION_001 | 李架构 |
| `PIPELINE_INTEGRATION_001` | Full Pipeline 集成 | dev | P1 | EA_DEV_001 | 全员 |
| `BASELINE_REPRO_001` | Baseline A-H 复现 | research | P1 | PIPELINE + DATA_EXPANSION | 全员 |

### 5.2 DATA_EXPANSION_001 详细规划

```yaml
DATA_EXPANSION_001:
  name: "全量 A 股日频数据扩展"
  priority: P1
  parallel_with: [SDF_DEV_001, EA_DEV_001]
  
  subtasks:
    - id: DATA_EXPANSION_001.1
      name: "股票池定义"
      components:
        - 全量 A 股列表获取
        - ST/退市/新股过滤规则
        - 历史成分股回溯
      effort: 4h
      
    - id: DATA_EXPANSION_001.2
      name: "数据采集"
      components:
        - 日频行情 (OHLCV, 2015-2025)
        - 季频财务数据
        - 94 特征因子计算
      effort: 16h
      
    - id: DATA_EXPANSION_001.3
      name: "数据清洗"
      components:
        - 缺失值处理
        - 异常值检测
        - 数据对齐
      effort: 8h
      
    - id: DATA_EXPANSION_001.4
      name: "因果性验证"
      components:
        - look-ahead 检测
        - t/t+1 分离验证
        - Rolling window 适配
      effort: 4h
      
    - id: DATA_EXPANSION_001.5
      name: "存储与索引"
      components:
        - Parquet 分区存储
        - 数据加载器适配
        - 性能测试
      effort: 4h
  
  total_effort: 36h (~5 天)
  deadline: 2026-03-15
```

### 5.2 SDF_DEV_001 子任务分解

```yaml
SDF_DEV_001:
  subtasks:
    - id: SDF_DEV_001.1
      name: "State Engine 实现"
      components:
        - XState encoder (Vol/Liq/Crowd)
        - Instrument basis construction
        - Unit tests
      effort: 8h
      
    - id: SDF_DEV_001.2
      name: "SDF Model 整合"
      components:
        - model.py 生产化
        - Boundedness & normalization
        - Temporal smoothness
      effort: 6h
      
    - id: SDF_DEV_001.3
      name: "Robust Moment Estimation"
      components:
        - MAD scaling
        - Clip/Huber aggregation
        - Instrumented moment G[i,j]
      effort: 6h
      
    - id: SDF_DEV_001.4
      name: "Training Pipeline"
      components:
        - SmoothMax objective
        - Temperature schedule
        - Window-level loop
      effort: 8h
      
    - id: SDF_DEV_001.5
      name: "EA Pricing Oracle"
      components:
        - PE(w) API
        - EA v3.1 interface
        - Integration tests
      effort: 4h
      
    - id: SDF_DEV_001.6
      name: "PanelTree 联调"
      components:
        - R_leaf data flow
        - End-to-end test
        - CB-L3 baseline comparison
      effort: 6h
```

---

## 6. 资源规划

### 6.1 计算资源

| 角色 | 周投入 | 主要职责 |
|------|--------|----------|
| 陈研究 | 20 小时 | 研究设计、论文撰写 |
| 李架构 | 15 小时 | 系统实现、实验运行 |
| 王数据 | 10 小时 | 数据准备、特征工程 |

---

## 5. 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| 实验未达预期 | 中 | 高 | 多线并行 (D1 + D3) |
| 论文被拒 | 中 | 中 | 准备 Plan B 期刊 |
| 数据质量问题 | 低 | 高 | 增量验证流程 |
| 资源不足 | 低 | 中 | 云计算弹性扩展 |

---

## 6. 成功标准

### 6.1 定量指标

| 指标 | 基线 | 目标 | 测量方法 |
|------|------|------|----------|
| Sharpe Ratio | 1.52 (Baseline C) | ≥1.65 | OOS 回测 |
| Pricing Error | 0.45 | ≤0.38 | MAE |
| 论文投稿 | 0 | ≥1 | 投稿记录 |

### 6.2 定性指标

- [ ] 研究方向获得领域专家认可
- [ ] 实验框架可复现
- [ ] 代码质量达到开源标准

---

## 7. 下一步行动

1. **本周**: 完成 EXPERIMENT_DESIGN.md
2. **下周**: 批准研究路线图，启动 D1 原型
3. **本月**: 完成实验基础设施搭建

---

## Appendix A: 参考文献

1. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *RFS*
2. Kozak, S., Nagel, S., & Santosh, S. (2020). Shrinking the Cross Section. *JFE*
3. Bryzgalova, S., et al. (2023). Forest Through the Trees. *JFE*

---

## Appendix B: Legacy DGSF Baseline 参考

| Baseline | 描述 | Sharpe (IS) | 状态 |
|----------|------|-------------|------|
| A | Sorting | 0.95 | ✅ 已复现 |
| C | P-tree | 1.52 | ✅ 已复现 |
| E | FF5 | 0.40 | ✅ 已复现 |
| F | NN-based | 1.35 | ✅ 已复现 |

