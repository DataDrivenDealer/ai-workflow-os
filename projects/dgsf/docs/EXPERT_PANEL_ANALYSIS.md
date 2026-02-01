# DGSF 开发状态专家评审报告

> **Document ID**: EXPERT_PANEL_ANALYSIS  
> **Version**: 1.0.0  
> **Date**: 2026-02-01  
> **Status**: ACTIVE

---

## 专家团队构成

| 角色 | 专业领域 | 职责 |
|------|----------|------|
| 🧠 **陈首席** | 量化金融 + 资产定价 | 学术方向把控、SDF 理论审核 |
| 🔧 **李架构** | 系统架构 + ML Engineering | 模块设计、接口规范 |
| 📊 **王数据** | 数据工程 + 因果推断 | 数据流验证、因果性保障 |
| 🧪 **赵测试** | 测试工程 + 质量保证 | 验证框架、回归测试 |
| 📋 **周项目** | 项目管理 + 敏捷开发 | 时间规划、里程碑追踪 |

---

## 1. 当前状态审核 (Status Audit)

### 1.1 层级开发进度

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DGSF 8-Layer Development Status                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  L0-L1: DataEng (v4.2)     ██████████████████████░░░░░░░░░░ 70%     │
│         ⚠️ 仅完成中证800回测，全量A股数据待完成                       │
│                                                                      │
│  L2: PanelTree (v3.0.2)    ██████████████████████████░░░░░░ 85%     │
│      ✅ 规范定稿 (v3.0.2 FINAL)                                      │
│      ✅ 核心代码实现 (api.py, builder.py, core.py, tree.py)          │
│      ✅ Rolling 集成 (rolling.py, rolling_runner.py)                 │
│      🔵 初步验证完成                                                 │
│      ⏳ 待: 与 SDF 层联调                                            │
│                                                                      │
│  L3: SDF (v3.1)            ██████████████░░░░░░░░░░░░░░░░░░ 45%     │
│      ✅ 规范定稿 (v3.1 FINAL + Design Review Note)                   │
│      ✅ 基础模型代码 (model.py, losses.py)                           │
│      ✅ 数据加载器 (a0_sdf_dataloader.py)                            │
│      🔵 当前阶段: 规范最终审核 + 模块开发                            │
│      ⏳ 待: Trainer 集成、EA 接口、Rolling 集成                       │
│                                                                      │
│  L4: EA (v3.1)             ██████████░░░░░░░░░░░░░░░░░░░░░░ 30%     │
│      ✅ 规范定稿 (v3.1 FINAL)                                        │
│      ✅ 基础框架代码 (core.py, nsga2_optimizer.py)                   │
│      ⏳ 待: SDF 接口、NSGA-III、HV-aware 实现                        │
│                                                                      │
│  L5: Rolling (v3.0)        ████████████████░░░░░░░░░░░░░░░░ 50%     │
│      ✅ 规范定稿 (v3.0 + Execution Framework v3.1)                   │
│      ✅ 基础框架代码                                                 │
│      ⏳ 待: 与 SDF/EA 联调                                           │
│                                                                      │
│  L6-L7: Report/Telemetry   ████████░░░░░░░░░░░░░░░░░░░░░░░░ 25%     │
│      ⏳ 待: 上游层稳定后开发                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 专家共识

**陈首席** (学术):
> "SDF 层是整个框架的理论核心。v3.1 规范已经非常完整，包含了 State Engine、Instrument Basis、Minimax Objective 等关键创新点。当前最重要的是将规范落地为高质量实现，并与 PanelTree 完成联调验证。"

**李架构** (工程):
> "Legacy 代码库中 SDF 模块已有 model.py, losses.py, training.py 等核心文件，但分散且部分是 a0/dev 前缀的实验代码。需要整合为符合 v3.1 规范的生产级模块，明确接口契约。"

**王数据** (数据):
> "SDF 层的输入来自 PanelTree 的 leaf portfolio returns，输出是 pricing error oracle 给 EA。数据流已经清晰，但需要验证 m_t 和 g_k 的计算是否严格遵循因果性。"

**赵测试** (测试):
> "当前 SDF 测试覆盖不足。需要建立: (1) 单元测试 - 各子模块；(2) 集成测试 - PanelTree → SDF 数据流；(3) 对比测试 - vs CB-L3 线性基线。"

**周项目** (管理):
> "SDF 开发预计 2-3 周，EA 开发预计 2 周。建议串行执行以确保接口稳定，总计 4-5 周完成核心开发。"

---

## 2. SDF 层开发规划

### 2.1 SDF 规范最终审核清单

基于 [SDF Layer Review Checklist](../legacy/DGSF/docs/SDF%20Layer%20Review%20Checklist%20.md):

| 模块 | 审核项 | 当前状态 | 行动项 |
|------|--------|----------|--------|
| **Set Encoder** | DeepSets mean pooling | ✅ 已实现 | 验证 tail 信息捕捉 |
| **XState + Instrument** | [1, V, L, V·L] basis | ✅ 规范明确 | 实现 State Engine v1.0 |
| **SDF Parameterization** | log m = c·tanh(h(·)) | ✅ model.py 有实现 | 验证 boundedness |
| **Robust Moments** | MAD scaling + clip | ⏳ 部分实现 | 完善 robust 估计 |
| **Minimax Objective** | SmoothMax + τ schedule | ⏳ losses.py 有基础 | 实现完整训练循环 |
| **EA Oracle** | PE(w) API | ⏳ 待实现 | 定义标准接口 |

### 2.2 SDF 模块开发任务分解

```
SDF_DEV_001: SDF 层模块开发
├── SDF_DEV_001.1: State Engine 实现
│   ├── XState encoder (Vol/Liq/Crowd)
│   ├── Instrument basis construction
│   └── Unit tests
│
├── SDF_DEV_001.2: SDF Model 整合
│   ├── 整合 model.py 为生产级代码
│   ├── Boundedness & normalization
│   └── Temporal smoothness
│
├── SDF_DEV_001.3: Robust Moment Estimation
│   ├── MAD scaling 实现
│   ├── Clip/Huber robust aggregation
│   └── Instrumented moment G[i,j]
│
├── SDF_DEV_001.4: Training Pipeline
│   ├── SmoothMax objective
│   ├── Temperature schedule
│   └── Window-level training loop
│
├── SDF_DEV_001.5: EA Pricing Oracle
│   ├── PE(w) API 定义
│   ├── 与 EA v3.1 接口对齐
│   └── Integration tests
│
└── SDF_DEV_001.6: PanelTree 联调
    ├── R_leaf[t+1, i] 数据流验证
    ├── 端到端测试
    └── Baseline 对比 (vs CB-L3)
```

---

## 3. 修正后的开发路线图

### 3.1 阶段划分

```
┌─────────────────────────────────────────────────────────────────────┐
│                  DGSF Development Roadmap (Corrected)                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: SDF Layer Development (当前)         2026-02 W1-W3        │
│  ═══════════════════════════════════════                            │
│  ├─ W1: 规范审核 + State Engine                                     │
│  ├─ W2: SDF Model + Robust Moments                                  │
│  └─ W3: Training Pipeline + EA Oracle + PanelTree 联调              │
│                                                                      │
│  Phase 2: EA Layer Development                 2026-02 W4 - 03 W1   │
│  ═══════════════════════════════════════                            │
│  ├─ W4: NSGA-III 核心实现                                           │
│  ├─ W5: SDF Consistency + HV-aware                                  │
│  └─ W1: Drift-aware warm-start + Integration                        │
│                                                                      │
│  Phase 3: Full Pipeline Integration            2026-03 W2-W3        │
│  ═══════════════════════════════════════                            │
│  ├─ W2: Rolling Window 完整流程                                     │
│  └─ W3: Baseline A-H 完整复现                                       │
│                                                                      │
│  Phase 4: Validation & Research                2026-03 W4 - 04      │
│  ═══════════════════════════════════════                            │
│  ├─ W4: OOS 验证 (2024-2025)                                        │
│  └─ Q2: 新实验 + 论文准备                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 里程碑定义

| ID | 里程碑 | 目标日期 | 验收标准 |
|----|--------|----------|----------|
| M1 | SDF 规范审核完成 | 2026-02-03 | Review Checklist 全通过 |
| M2 | SDF 模块 Alpha | 2026-02-10 | 单元测试通过 |
| M3 | SDF-PanelTree 联调 | 2026-02-17 | 端到端数据流验证 |
| M4 | EA 模块 Alpha | 2026-02-28 | NSGA-III + SDF Oracle |
| M5 | Full Pipeline | 2026-03-10 | Rolling 完整流程运行 |
| M6 | Baseline 复现 | 2026-03-17 | A-H 全复现 |

---

## 4. 下一步行动计划

### 4.1 本周任务 (W1: 2026-02-01 ~ 02-07)

| 任务 | 负责人 | 工时 | 优先级 |
|------|--------|------|--------|
| SDF Review Checklist 逐条审核 | 陈首席 + 李架构 | 4h | P0 |
| State Engine v1.0 实现 | 李架构 | 8h | P0 |
| XState encoder 单元测试 | 赵测试 | 4h | P1 |
| 数据流因果性验证 | 王数据 | 4h | P1 |

### 4.2 任务卡创建计划

需要创建以下 TaskCards:

1. **SDF_SPEC_REVIEW_001**: SDF 规范最终审核
2. **SDF_DEV_001**: SDF 层模块开发 (包含 6 个子任务)
3. **SDF_INTEGRATION_001**: SDF-PanelTree 集成测试

---

## 5. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| SDF 训练不稳定 | 中 | 高 | τ schedule + gradient clip |
| EA 接口不兼容 | 低 | 高 | 提前定义 PE(w) 契约 |
| 数据因果性泄漏 | 低 | 极高 | 严格 t/t+1 分离验证 |
| 工期延误 | 中 | 中 | 缓冲时间 + 并行任务 |

---

## 6. 专家建议汇总

### 陈首席 (学术方向):
> "优先验证 SDF 的经济学含义：m_t 应该在高波动/低流动性时期偏高。建议在开发过程中加入 SDF diagnostics 可视化。"

### 李架构 (技术实现):
> "建议将 a0_ 和 dev_ 前缀代码整合为单一 sdf/ 模块。接口设计参考 PyTorch Lightning 风格，便于后续扩展。"

### 王数据 (数据质量):
> "R_leaf 的计算必须严格使用 t 时刻信息，返回 t+1 收益。建议在数据加载阶段加入自动因果性检查。**另外，当前回测仅使用中证800成分股，样本代表性不足。后续必须扩展到全量A股 (~5000只) 日频数据，才能支撑可复现的学术发表。**"

### 赵测试 (质量保证):
> "建立三层测试：单元 → 集成 → 系统。重点测试 SDF-EA 接口的边界情况。"

### 周项目 (进度管理):
> "SDF 是关键路径。建议每日 standup 追踪进度，周末进行 Sprint Review。**数据扩展任务可与 SDF 开发并行，由王数据负责。**"

---

**下一步**: 创建 SDF_SPEC_REVIEW_001 和 SDF_DEV_001 任务卡，开始执行。
