---
task_id: "SDF_SPEC_REVIEW_001"
type: review
queue: research
branch: "feature/SDF_SPEC_REVIEW_001"
priority: P0
spec_ids:
  - "DGSF_SDF_V3.1"
  - "STATE_ENGINE_V1.0"
  - "GOVERNANCE_INVARIANTS"
verification:
  - "All Review Checklist items addressed"
  - "Design decisions documented"
  - "Implementation guidance finalized"
---

# TaskCard: SDF_SPEC_REVIEW_001

> **Phase**: 1 · SDF Layer Development  
> **Pipeline**: DGSF Development Pipeline  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `SDF_SPEC_REVIEW_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `architect` / `researcher` |
| **Authority** | `accepted` |
| **Authorized By** | Project Owner |
| **下游 Task** | `SDF_DEV_001` |

---

## 1. 任务背景

### 1.1 开发状态校准

| 层级 | 规范状态 | 代码状态 | 当前阶段 |
|------|----------|----------|----------|
| L2: PanelTree | ✅ v3.0.2 FINAL | ✅ 初步验证 | 待 SDF 联调 |
| **L3: SDF** | ✅ v3.1 FINAL | 🔵 45% | **规范审核 → 开发** |
| L4: EA | ✅ v3.1 FINAL | ⏳ 30% | 待 SDF 完成 |

### 1.2 可用规范文档

| 文档 | 路径 | 状态 |
|------|------|------|
| SDF Layer Specification v3.1 | `legacy/DGSF/docs/specs_v3/DGSF SDF Layer Specification v3.1.md` | ✅ FINAL |
| SDF Layer Final Spec v1.0 | `legacy/DGSF/docs/SDF Layer Final Spec v1.0.txt` | ✅ Frozen |
| SDF Layer Design Note | `legacy/DGSF/docs/SDF Layer Design & Mathematical Review Note.md` | ✅ |
| SDF Layer Review Checklist | `legacy/DGSF/docs/SDF Layer Review Checklist .md` | ✅ |
| State Engine Spec v1.0 | `legacy/DGSF/docs/State Engine Spec v1.0.txt` | ✅ Frozen |

---

## 2. 任务范围

### 2.1 Review Checklist 逐条审核

基于 `SDF Layer Review Checklist .md`，需要对以下 6 个模块进行最终决策：

---

#### Module 1: Set Encoder (Market Representation)

| 审核维度 | 当前设计 | 待决策 |
|----------|----------|--------|
| 数学目标 | SDF 依赖市场整体结构 | ✅ 合理 |
| 数学方法 | DeepSets: MLP + mean pooling | ✅ **Keep Mean Pooling** |
| 数学风险 | mean pooling 可能忽略 tail | ✅ **Defer (vNext)** |
| 工程实现 | 标准 DeepSets | ✅ 采纳 |

**决策项**:
- [x] Mean pooling vs Attention-based: **Mean Pooling** (复杂度/数据量权衡)
- [x] Tail/dispersion 编码: **否** (Defer to vNext)

---

#### Module 2: XState + Instrument Basis

| 审核维度 | 当前设计 | 待决策 |
|----------|----------|--------|
| 数学目标 | Conditional no-arbitrage | ✅ 合理 |
| Instrument 选择 | [1, V, L, V·L] (J=4) | ✅ **J=4 Baseline** |
| 数学风险 | Instrument bias | ✅ **不正则化** |
| 工程实现 | 显式 feature expansion | ✅ 采纳 |

**决策项**:
- [x] Instrument 维度: **J=4** [1, V, L, V·L] (Baseline frozen)
- [x] Instrument 正则化: **否** (低维不需要)

---

#### Module 3: SDF Parameterization

| 审核维度 | 当前设计 | 待决策 |
|----------|----------|--------|
| 模型形式 | log m = c·tanh(h(·)) | ✅ 采纳 |
| Boundedness | c = 4.0 | ✅ 冻结 |
| 归一化 | E[m] = 1 | ✅ 冻结 |
| Temporal smoothness | λ = 10^-3 | ✅ **λ = 10⁻³ Frozen** |

**决策项**:
- [x] Temporal smoothness λ: **10⁻³** (保持 Frozen 值)

---

#### Module 4: Robust Moment Estimation

| 审核维度 | 当前设计 | 待决策 |
|----------|----------|--------|
| Return scaling | MAD (baseline) | ✅ **MAD Frozen** |
| Robust aggregation | clip ±c_y, c_y=3.0 | ✅ **c_y=3.0 Frozen** |
| 数学风险 | Clipping 引入 bias | 可接受 |

**决策项**:
- [x] Scaling 方法: **MAD** (稳健于 outliers)
- [x] Clip bound c_y: **3.0** (标准稳健统计)

---

#### Module 5: Minimax Objective

| 审核维度 | 当前设计 | 待决策 |
|----------|----------|--------|
| 目标函数 | SmoothMax(|g_{i,j}|) | ✅ 采纳 |
| τ schedule | τ: 5 → 20, warmup=10 | ✅ **Frozen** |
| 工程实现 | log-sum-exp | ✅ 数值稳定 |

**决策项**:
- [x] τ_start: **5** (Frozen)
- [x] τ_end: **20** (Frozen)
- [x] warmup epochs: **10** (Frozen)

---

#### Module 6: EA Pricing Oracle

| 审核维度 | 当前设计 | 待决策 |
|----------|----------|--------|
| Oracle 定义 | PE(w) = SmoothMax(|g_j(w)|) | ✅ 采纳 |
| 一致性 | 与 SDF 训练同口径 | ✅ 必须 |
| EA objectives | Sharpe, MDD, Turnover, PE | ✅ 冻结 |

**决策项**:
- [x] PE(w) 是否需要额外 normalization: **否** (保持原始量纲)

---

### 2.2 输出: 实现指导文档

完成审核后，输出 `SDF_IMPLEMENTATION_GUIDE.md`:

```
SDF_IMPLEMENTATION_GUIDE.md
├── 1. Design Decisions Summary
│   └── 所有审核决策记录
├── 2. Module Interface Contracts
│   └── 输入/输出数据格式
├── 3. Implementation Priorities
│   └── 开发顺序建议
├── 4. Testing Requirements
│   └── 单元测试 + 集成测试要求
└── 5. Known Risks & Mitigations
    └── 风险与应对
```

---

## 3. 交付物

| 交付物 | 路径 | 状态 |
|--------|------|------|
| Review Checklist 决策记录 | `projects/dgsf/docs/SDF_REVIEW_DECISIONS.md` | ✅ `completed` |
| 实现指导文档 | `projects/dgsf/docs/SDF_IMPLEMENTATION_GUIDE.md` | ✅ `completed` |
| 接口契约定义 | `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml` | ✅ `completed` |

---

## 4. 验收标准

### 4.1 必须完成
- [x] 6 个模块所有决策项已确定
- [x] 决策记录文档化
- [x] 实现指导文档完成
- [x] 接口契约定义

### 4.2 质量要求
- [x] 决策与 v3.1 规范一致
- [x] 与 EA v3.1 接口对齐
- [x] 遵循因果性要求

---

## 5. 时间估算

| 子任务 | 工作量 | 负责人 |
|--------|--------|--------|
| Set Encoder 审核 | 0.5h | 陈首席 |
| XState 审核 | 0.5h | 陈首席 |
| SDF Param 审核 | 0.5h | 李架构 |
| Robust Moments 审核 | 0.5h | 李架构 |
| Minimax 审核 | 0.5h | 李架构 |
| EA Oracle 审核 | 0.5h | 全员 |
| 文档输出 | 2h | 李架构 |
| **总计** | **5h** | - |

---

## 6. Gate & 下游依赖

- **Gate**: 所有决策项完成
- **后续 TaskCard**: `SDF_DEV_001` (依赖本任务)
- **依赖**: 无 (起始任务)

---

## 7. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T23:45:00Z | system | `task_created` | 任务创建 |
| 2026-02-01T23:50:00Z | system | `task_start` | 专家评审启动 |
| 2026-02-02T00:00:00Z | Expert Panel | `review_complete` | 6 模块审核完成 |
| 2026-02-02T00:00:00Z | Project Owner | `approve` | 决策批准 |
| 2026-02-02T00:00:00Z | system | `task_release` | 任务发布完成 |

---

## 8. Task Completion Summary

✅ **SDF_SPEC_REVIEW_001 COMPLETED**

所有 6 个模块的待决策项已由专家团队完成裁决：

| 模块 | 决策项 | 最终决策 |
|------|--------|----------|
| Set Encoder | Pooling | Mean Pooling |
| Set Encoder | Tail encoding | Defer |
| XState | J dimension | J=4 |
| XState | Regularization | 否 |
| SDF Param | λ_smooth | 10⁻³ |
| Robust | Scaling | MAD |
| Robust | c_y | 3.0 |
| Minimax | τ schedule | 5→20, warmup=10 |
| EA Oracle | Normalization | 否 |

**下游任务 `SDF_DEV_001` 已解除阻塞，可以启动开发。**

