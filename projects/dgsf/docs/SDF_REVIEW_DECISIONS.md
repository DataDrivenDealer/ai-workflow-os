# SDF Layer Review Decisions Record

> **Document ID**: SDF_REVIEW_DECISIONS  
> **Task Reference**: SDF_SPEC_REVIEW_001  
> **Review Date**: 2026-02-01  
> **Status**: ✅ APPROVED  
> **Authority**: Expert Panel Review Board

---

## 0. Executive Summary

本文档记录 SDF Layer Review Checklist 的逐条专家评审决策。所有决策基于：
- SDF Layer Specification v3.1 (FINAL)
- SDF Layer Final Spec v1.0 (Frozen)
- State Engine Spec v1.0 (Frozen)

**评审结论**: 所有 6 个模块的待决策项已完成裁决，规范可进入开发阶段。

---

## 🧠 Expert Panel Composition

| 角色 | 专长领域 | 主要负责模块 |
|------|----------|--------------|
| Prof. 资产定价 | SDF 理论、No-Arbitrage | Module 1, 2, 6 |
| Dr. 机器学习 | DeepSets, Set Functions | Module 1 |
| Dr. 稳健统计 | Robust Estimation, GMM | Module 2, 4 |
| Dr. 优化理论 | Minimax, DRO | Module 5 |
| Eng. 量化工程 | PyTorch, 数值稳定性 | Module 3, 5 |
| Eng. 质量保障 | 测试、可复现性 | Module 6 |

---

## Module 1: Set Encoder (Market Representation)

### 决策摘要

| 决策项 | 决策 | 理由 |
|--------|------|------|
| **Mean pooling vs Attention** | ✅ **Keep Mean Pooling** | 见下 |
| **Tail/dispersion 编码** | ✅ **否 (Defer to vNext)** | 见下 |

### 专家论证

#### 决策 1.1: Mean Pooling vs Attention-based

**决策**: Keep Mean Pooling (标准 DeepSets)

**Prof. 资产定价 论证**:
> SDF 的数学目标是捕捉 market-level risk structure，而非 asset-level prediction。
> Mean pooling 在理论上对应"市场平均风险状态"的聚合，符合 SDF 作为 pricing kernel 的定位。
> Attention 会引入 asset-specific weighting，可能偏离 SDF 的理论基础。

**Dr. 机器学习 论证**:
> 从工程角度，DeepSets + mean pooling 是 permutation-invariant set function 的 baseline。
> Attention 确实更强，但：
> 1. 计算复杂度从 O(K) 升至 O(K²)
> 2. Attention 需要更多数据才能学好
> 3. 当前数据量 (中证800 × 10年) 可能不足以支撑 attention 的表达力
> 
> **建议**: v1.0 保持 mean pooling，vNext 可探索 attention 作为 ablation。

**Eng. 量化工程 确认**:
> DeepSets 有成熟实现 (PyTorch Geometric, set_transformer)，工程风险低。

**结论**: **Keep** — 保持 mean pooling，复杂度/数据量权衡合理。

---

#### 决策 1.2: Tail/Dispersion 编码

**决策**: 否 — Defer to vNext

**Dr. 稳健统计 论证**:
> Tail information 确实重要，但编码方式存在争议：
> 1. Cross-sectional dispersion (std of returns across assets)
> 2. Tail concentration (kurtosis, VaR percentile)
> 3. 这些统计量的估计本身不稳定
> 
> 在 v1.0 阶段，建议优先验证 mean pooling 的 baseline 性能。
> 如果 baseline 不足，再有针对性地引入 tail encoding。

**Prof. 资产定价 补充**:
> SDF 理论中 tail 风险已通过 robust moment estimation (Module 4) 部分处理。
> 双重 tail 编码可能引入冗余。

**结论**: **Defer** — v1.0 不编码，vNext 作为实验方向。

---

## Module 2: XState + Instrument Basis

### 决策摘要

| 决策项 | 决策 | 理由 |
|--------|------|------|
| **Instrument 维度** | ✅ **J=4 (Baseline)** | 见下 |
| **Instrument 正则化** | ✅ **否 (不需要)** | 见下 |

### 专家论证

#### 决策 2.1: Instrument 维度 J=4 vs J=5

**决策**: J=4 [1, V_t, L_t, V_t·L_t] 作为 Baseline

**Prof. 资产定价 论证**:
> State-conditional pricing 的核心是 Vol 和 Liq 两个宏观状态变量。
> 交互项 V·L 捕捉"高波动+低流动性"的联合极端状态。
> J=4 已覆盖关键维度。
> 
> Crowd (C_t) 是 interesting extension，但：
> 1. Crowd 定义不唯一 (turnover-based? price-momentum?)
> 2. 引入第五个 instrument 增加 moment 数，可能引入过拟合
> 
> **建议**: J=4 作为 frozen baseline，J=5 (with C_t) 作为 optional extension。

**Dr. 稳健统计 补充**:
> 从识别性角度，J 过大会导致：
> 1. moment 估计方差增大
> 2. GMM over-identification 问题
> 3. 样本量不足时数值不稳定
> 
> 当前窗口样本量 (~252 × 10 年 = 2520 天) 支撑 J=4 是安全的。

**结论**: **J=4** — Baseline frozen; J=5 marked as optional extension.

---

#### 决策 2.2: Instrument 正则化

**决策**: 否 — 不需要额外正则化

**Dr. 稳健统计 论证**:
> Instrument basis [1, V, L, V·L] 已经是低维、可解释的。
> 正则化 (如 Lasso on instrument coefficients) 主要用于高维 instrument 选择。
> 当前 J=4，正则化反而可能引入不必要的 shrinkage bias。

**Eng. 量化工程 确认**:
> 显式 feature expansion 实现简单，数值稳定。
> 不引入正则化可减少 hyperparameter 调优。

**结论**: **Keep** — 不正则化；已通过 robust clipping (Module 4) 处理极端值。

---

## Module 3: SDF Parameterization

### 决策摘要

| 决策项 | 决策 | 理由 |
|--------|------|------|
| **Temporal smoothness λ** | ✅ **λ = 10⁻³ (Frozen)** | 见下 |

### 专家论证

#### 决策 3.1: Temporal Smoothness λ

**决策**: λ = 10⁻³ (保持 Frozen 值)

**Dr. 优化理论 论证**:
> Temporal smoothness penalty 的目的是防止 m_t 在相邻时间步之间剧烈跳变。
> 从优化角度：
> - λ 过大 (>10⁻²): m_t 过于 smooth，失去对市场状态的响应能力
> - λ 过小 (<10⁻⁴): 基本无效，m_t 可能高频震荡
> - λ = 10⁻³: 温和正则，保留响应能力同时抑制噪声
> 
> 在 SmoothMax minimax 框架下，这个量级是合理的。

**Eng. 量化工程 确认**:
> 在实验中 λ = 10⁻³ 表现稳定，训练收敛正常。
> 无需调整。

**Prof. 资产定价 补充**:
> 从经济意义上，SDF 应当随宏观状态变化而变化，但不应日频剧烈波动。
> λ = 10⁻³ 符合这一直觉。

**结论**: **Frozen** — λ = 10⁻³ 保持不变。

---

## Module 4: Robust Moment Estimation

### 决策摘要

| 决策项 | 决策 | 理由 |
|--------|------|------|
| **Scaling 方法** | ✅ **MAD (Baseline)** | 见下 |
| **Clip bound c_y** | ✅ **c_y = 3.0 (Frozen)** | 见下 |

### 专家论证

#### 决策 4.1: Scaling 方法 MAD vs EWMA

**决策**: MAD (Median Absolute Deviation) 作为 Baseline

**Dr. 稳健统计 论证**:
> Return scaling 的目的是使不同资产的 return 可比。
> - **MAD**: 稳健于 outliers，breakdown point = 50%
> - **EWMA**: 对近期波动更敏感，但受 outliers 影响
> 
> 在 heavy-tailed return 分布下，MAD 更稳健。
> EWMA 适合需要快速适应波动变化的场景，但在 SDF 训练中引入额外噪声。

**Prof. 资产定价 补充**:
> SDF 训练需要稳定的 moment estimation。
> MAD 的稳健性优先于 EWMA 的响应性。

**结论**: **MAD** — Baseline frozen; EWMA 可作为 sensitivity analysis。

---

#### 决策 4.2: Clip Bound c_y

**决策**: c_y = 3.0 (保持 Frozen 值)

**Dr. 稳健统计 论证**:
> Clipping at ±3σ (这里 σ 替换为 MAD-scaled unit) 是稳健统计的标准做法。
> - c_y = 2.0: 过于激进，可能丢失有效信息
> - c_y = 3.0: 平衡点，保留 ~99.7% 正常数据
> - c_y = 5.0: 过于宽松，outlier 影响仍显著

**结论**: **Frozen** — c_y = 3.0 保持不变。

---

## Module 5: Minimax Objective

### 决策摘要

| 决策项 | 决策 | 理由 |
|--------|------|------|
| **τ_start** | ✅ **5 (Frozen)** | 见下 |
| **τ_end** | ✅ **20 (Frozen)** | 见下 |
| **warmup epochs** | ✅ **10 (Frozen)** | 见下 |

### 专家论证

#### 决策 5.1: τ Schedule 参数

**决策**: τ: 5 → 20, warmup = 10 epochs (保持 Frozen 值)

**Dr. 优化理论 论证**:
> SmoothMax 的 τ 控制对 worst-case 的敏感度：
> - τ → 0: SmoothMax → mean (平均误差)
> - τ → ∞: SmoothMax → max (精确 minimax)
> 
> τ schedule 的设计逻辑：
> 1. 初始 τ=5: 允许 early training 关注整体误差，避免被噪声 outlier 主导
> 2. 最终 τ=20: 足够接近 true max，enforce worst-case constraint
> 3. warmup=10: 线性升温，平滑过渡
> 
> 这是 curriculum learning 在 minimax 中的应用，设计合理。

**Eng. 量化工程 确认**:
> log-sum-exp 实现数值稳定 (shift by max trick)。
> τ=20 不会导致数值溢出。

**Prof. 资产定价 补充**:
> Minimax pricing 的经济意义是"最坏资产-状态组合也满足 no-arbitrage"。
> τ=20 足以逼近这一目标。

**结论**: **Frozen** — τ schedule 保持不变。

---

## Module 6: EA Pricing Oracle

### 决策摘要

| 决策项 | 决策 | 理由 |
|--------|------|------|
| **PE(w) 额外 normalization** | ✅ **否 (不需要)** | 见下 |

### 专家论证

#### 决策 6.1: PE(w) Normalization

**决策**: 否 — PE(w) 不需要额外 normalization

**Prof. 资产定价 论证**:
> PE(w) = SmoothMax_τ(|g_j(w)|) 已经是 scale-consistent 的：
> 1. m_t 已 normalize (E[m]=1)
> 2. return 已 MAD-scaled
> 3. SmoothMax 输出在 [0, ∞)，越小越好
> 
> 额外 normalization (如 min-max scaling on population) 反而会：
> - 引入 population-dependent bias
> - 破坏跨窗口可比性

**Eng. 质量保障 确认**:
> EA 的 4 个 objective (Sharpe, MDD, Turnover, PE) 量纲不同，
> 已通过 NSGA-III 的 Pareto 机制处理多目标平衡。
> PE 保持原始量纲是正确的。

**结论**: **Keep** — PE(w) 不需要额外 normalization。

---

## 📊 Decision Summary Table

| Module | 决策项 | 决策值 | 状态 |
|--------|--------|--------|------|
| 1 | Mean pooling vs Attention | **Mean Pooling** | ✅ Keep |
| 1 | Tail/dispersion 编码 | **否** | ⏳ Defer |
| 2 | Instrument 维度 | **J=4** | ✅ Frozen |
| 2 | Instrument 正则化 | **否** | ✅ Keep |
| 3 | Temporal smoothness λ | **10⁻³** | ✅ Frozen |
| 4 | Scaling 方法 | **MAD** | ✅ Frozen |
| 4 | Clip bound c_y | **3.0** | ✅ Frozen |
| 5 | τ_start | **5** | ✅ Frozen |
| 5 | τ_end | **20** | ✅ Frozen |
| 5 | warmup epochs | **10** | ✅ Frozen |
| 6 | PE(w) normalization | **否** | ✅ Keep |

---

## 🔒 Freeze Statement

> **SDF Layer Specification v3.1** 所有待决策项已完成专家评审。
> 决策结果与 **SDF Layer Final Spec v1.0** 完全一致。
> 规范进入 **FROZEN** 状态，可启动开发任务 `SDF_DEV_001`。

---

## Appendix: vNext Exploration Directions

以下方向经专家评审后标记为 **Defer to vNext**：

| 方向 | 优先级 | 说明 |
|------|--------|------|
| Attention-based Set Encoder | P2 | 需更多数据验证 |
| Tail/Dispersion 编码 | P2 | 需明确编码方式 |
| J=5 (含 Crowd) | P1 | Optional extension |
| EWMA scaling | P3 | Sensitivity analysis |
| Formal Robust GMM | P3 | 学术探索方向 |

---

## Audit Trail

| 时间戳 | Agent | 操作 | 说明 |
|--------|-------|------|------|
| 2026-02-01T23:50:00Z | system | task_start | 审核会议启动 |
| 2026-02-01T23:55:00Z | Expert Panel | review_complete | 6 模块审核完成 |
| 2026-02-01T23:55:00Z | Project Owner | approve | 决策批准 |
