# DGSF Experiment Design Document

> **Document ID**: EXPERIMENT_DESIGN  
> **Version**: 1.0.0  
> **Created**: 2026-02-01  
> **Status**: DRAFT

---

## 1. Overview

本文档定义 DGSF 研究项目的新实验设计，基于 Legacy DGSF 验证结果和研究路线图规划。

### 1.1 实验目标

| 实验 ID | 名称 | 假设 | 优先级 |
|---------|------|------|--------|
| EXP_DEEP_SDF_001 | Deep SDF Architecture | 深度网络提升 SDF 精度 | P0 |
| EXP_TEMPORAL_PTREE_001 | Temporal PanelTree | 时变结构捕捉状态转换 | P1 |

### 1.2 Baseline 参照

所有新实验需与以下 Baseline 比较：

| Baseline | 角色 | Sharpe (复现) |
|----------|------|---------------|
| A: Sorting | 传统基准 | 0.95 |
| C: P-tree | 当前最佳 | 1.52 |
| E: FF5 | 学术基准 | 0.40 |
| F: NN-based | ML 基准 | 1.35 |

---

## 2. EXP_DEEP_SDF_001: Deep SDF Architecture

### 2.1 研究假设

**H1**: 深度神经网络可以学习到更准确的随机贴现因子 (SDF) 表示

**理论依据**:
- SDF 是高维非线性函数 (Hansen & Jagannathan, 1991)
- 深度网络在函数逼近方面具有理论优势 (Universal Approximation)
- 近期资产定价文献支持深度学习方法 (Gu, Kelly & Xiu, 2020)

### 2.2 方法设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    Deep SDF Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│   │  Firm      │    │  PanelTree │    │  Temporal  │            │
│   │  Features  │───▶│  Encoder   │───▶│  Attention │            │
│   │  (F_it)    │    │            │    │            │            │
│   └────────────┘    └────────────┘    └─────┬──────┘            │
│                                             │                    │
│   ┌────────────┐    ┌────────────┐    ┌─────▼──────┐            │
│   │  Market    │    │  Factor    │    │  SDF       │            │
│   │  Returns   │───▶│  Network   │───▶│  Estimator │───▶ M_t    │
│   │  (R_t)     │    │            │    │            │            │
│   └────────────┘    └────────────┘    └────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2.1 模型组件

**A. PanelTree Encoder**
- 输入: 企业特征 $F_{it} \in \mathbb{R}^{d}$
- 结构: 基于 Legacy PanelTree 的嵌入层
- 输出: 树结构嵌入 $z_{it} \in \mathbb{R}^{h}$

```python
class PanelTreeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_trees):
        super().__init__()
        self.trees = nn.ModuleList([
            TreeNet(input_dim, hidden_dim) 
            for _ in range(num_trees)
        ])
        self.aggregator = nn.Linear(num_trees * hidden_dim, hidden_dim)
    
    def forward(self, features):
        tree_outputs = [tree(features) for tree in self.trees]
        combined = torch.cat(tree_outputs, dim=-1)
        return self.aggregator(combined)
```

**B. Temporal Attention**
- 捕捉时间序列依赖
- 使用多头自注意力机制

**C. SDF Estimator**
- 输出: $M_t = \exp(-\gamma \cdot g(z_t))$
- 约束: $E[M_t R_t] = 1$ (Euler 方程)

#### 2.2.2 训练目标

$$
\mathcal{L} = \mathcal{L}_{pricing} + \lambda_1 \mathcal{L}_{HJ} + \lambda_2 \mathcal{L}_{reg}
$$

其中:
- $\mathcal{L}_{pricing}$: 定价误差 (预测收益 vs 实际收益)
- $\mathcal{L}_{HJ}$: Hansen-Jagannathan 距离
- $\mathcal{L}_{reg}$: 正则化项

### 2.3 实验配置

```yaml
# experiments/EXP_DEEP_SDF_001/config.yaml
experiment:
  id: "EXP_DEEP_SDF_001"
  name: "Deep SDF Architecture"
  version: "0.1.0"
  
model:
  architecture: "deep_sdf"
  encoder:
    type: "panel_tree"
    hidden_dim: 128
    num_trees: 8
  attention:
    type: "multi_head"
    num_heads: 4
    dropout: 0.1
  sdf_estimator:
    layers: [128, 64, 32]
    activation: "relu"
    
training:
  optimizer: "adam"
  learning_rate: 1e-4
  batch_size: 256
  epochs: 100
  early_stopping:
    patience: 10
    metric: "sharpe_ratio"
    
data:
  train_period: ["2015-01-01", "2021-12-31"]
  val_period: ["2022-01-01", "2022-12-31"]
  test_period: ["2023-01-01", "2023-12-31"]
  features: "standard_94"
  
evaluation:
  metrics:
    - sharpe_ratio
    - pricing_error
    - alpha
    - max_drawdown
  baselines:
    - "A"  # Sorting
    - "C"  # P-tree
    - "E"  # FF5
    - "F"  # NN-based
```

### 2.4 预期结果

| 指标 | Baseline C | 目标 | 改进幅度 |
|------|------------|------|----------|
| Sharpe Ratio | 1.52 | ≥1.65 | +8.5% |
| Pricing Error | 0.45 | ≤0.38 | -15.5% |
| Alpha (ann.) | 12.3% | ≥14% | +13.8% |

### 2.5 风险分析

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 过拟合 | 高 | 高 | Dropout, Early stopping, 正则化 |
| 训练不稳定 | 中 | 中 | Gradient clipping, 学习率调度 |
| 计算成本高 | 中 | 中 | 混合精度训练, 模型剪枝 |

---

## 3. EXP_TEMPORAL_PTREE_001: Temporal PanelTree

### 3.1 研究假设

**H2**: 时变树结构可以更好地捕捉市场状态转换，提升策略稳定性

**理论依据**:
- 市场存在不同状态 (牛市/熊市/震荡)
- 静态结构难以适应状态变化
- 动态模型在金融应用中表现更好

### 3.2 方法设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    Temporal PanelTree                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────┐                                                 │
│   │   Regime   │──────────────────────────┐                     │
│   │  Detector  │                          │                     │
│   └─────┬──────┘                          │                     │
│         │                                 ▼                     │
│   ┌─────▼──────┐    ┌────────────┐  ┌────────────┐             │
│   │  Market    │    │  Dynamic   │  │  Tree      │             │
│   │  State     │───▶│  Split     │──│  Structure │──▶ Factors  │
│   │  (s_t)     │    │  Rules     │  │  (T_t)     │             │
│   └────────────┘    └────────────┘  └────────────┘             │
│                                                                  │
│   Time: ─────────▶  t-2    t-1     t      t+1                   │
│   State:            bull   bull   bear   bear                    │
│   Tree:             T_bull ─────▶ T_bear ─────▶                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.1 Regime Detection

使用隐马尔可夫模型 (HMM) 检测市场状态:

```python
class RegimeDetector:
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.hmm = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100
        )
    
    def fit(self, returns):
        """Fit HMM to market returns."""
        self.hmm.fit(returns.reshape(-1, 1))
        return self
    
    def predict(self, returns):
        """Predict current regime."""
        return self.hmm.predict(returns.reshape(-1, 1))
```

#### 3.2.2 Dynamic Split Rules

树分裂规则根据状态动态调整:

$$
\text{Split}(x; s_t) = 
\begin{cases}
\text{Split}_{bull}(x) & \text{if } s_t = \text{bull} \\
\text{Split}_{bear}(x) & \text{if } s_t = \text{bear} \\
\text{Split}_{neutral}(x) & \text{if } s_t = \text{neutral}
\end{cases}
$$

### 3.3 实验配置

```yaml
# experiments/EXP_TEMPORAL_PTREE_001/config.yaml
experiment:
  id: "EXP_TEMPORAL_PTREE_001"
  name: "Temporal PanelTree"
  version: "0.1.0"
  
model:
  architecture: "temporal_panel_tree"
  regime_detector:
    type: "hmm"
    n_regimes: 3
    covariance: "full"
  tree:
    max_depth: 6
    min_samples_split: 50
    state_specific: true
    
training:
  method: "regime_aware"
  regime_warmup: 252  # 1 year of data for regime detection
  tree_update_frequency: "monthly"
  
data:
  train_period: ["2015-01-01", "2021-12-31"]
  val_period: ["2022-01-01", "2022-12-31"]
  test_period: ["2023-01-01", "2023-12-31"]
  
evaluation:
  metrics:
    - sharpe_ratio
    - sharpe_ratio_bear  # Bear market specific
    - regime_stability
    - turnover
  baselines:
    - "C"  # Static P-tree
```

### 3.4 预期结果

| 指标 | Baseline C | 目标 | 改进幅度 |
|------|------------|------|----------|
| Sharpe (全期) | 1.52 | ≥1.55 | +2% |
| Sharpe (熊市) | 0.85 | ≥1.10 | +29% |
| Max Drawdown | -18% | ≤-14% | +22% |
| Turnover | 0.45 | ≤0.50 | 可控 |

---

## 4. 评估协议

### 4.1 标准评估流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation Protocol                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. In-Sample (IS)                                               │
│     └─ Train: 2015-2021                                          │
│                                                                  │
│  2. Out-of-Sample (OOS)                                          │
│     └─ Test: 2022-2023                                           │
│                                                                  │
│  3. Robustness Checks                                            │
│     ├─ Rolling Window                                            │
│     ├─ Different Markets (Optional)                              │
│     └─ Feature Ablation                                          │
│                                                                  │
│  4. Statistical Tests                                            │
│     ├─ Sharpe Ratio Difference Test                              │
│     ├─ Spanning Test                                             │
│     └─ Bootstrap Confidence Intervals                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| Sharpe Ratio | $SR = \frac{E[R_p - R_f]}{\sigma_p} \times \sqrt{252}$ | 年化 |
| Pricing Error | $PE = \frac{1}{N}\sum_i |E[M \cdot R_i] - 1|$ | MAE |
| Alpha | $\alpha = R_p - \beta R_m$ | CAPM Alpha |
| Max Drawdown | $MDD = \min_t \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}$ | 最大回撤 |

### 4.3 统计显著性

使用 Ledoit-Wolf (2008) 方法检验 Sharpe Ratio 差异:

$$
H_0: SR_{new} = SR_{baseline}
$$
$$
H_1: SR_{new} > SR_{baseline}
$$

显著性水平: $\alpha = 0.05$

---

## 5. 实验管理

### 5.1 目录结构

```
projects/dgsf/experiments/
├── templates/
│   ├── experiment_config.yaml
│   └── evaluation_protocol.yaml
├── EXP_DEEP_SDF_001/
│   ├── config.yaml
│   ├── README.md
│   ├── src/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── notebooks/
│   │   └── analysis.ipynb
│   └── results/
│       ├── metrics.json
│       └── figures/
└── EXP_TEMPORAL_PTREE_001/
    ├── config.yaml
    ├── README.md
    ├── src/
    └── results/
```

### 5.2 版本控制

- 每次实验运行生成唯一 Run ID
- 配置、代码、数据版本锁定
- 结果可复现

### 5.3 审计集成

实验与 AI Workflow OS 审计系统集成:

```python
from projects.dgsf.adapter import DGSFAuditBridge

audit = DGSFAuditBridge()
audit.log_event(
    event_type="experiment_start",
    experiment_id="EXP_DEEP_SDF_001",
    config=config,
    timestamp=datetime.now()
)
```

---

## 6. 时间线

| 周次 | EXP_DEEP_SDF_001 | EXP_TEMPORAL_PTREE_001 |
|------|------------------|------------------------|
| W5 | 架构设计 | - |
| W6 | 编码实现 | - |
| W7 | 调试训练 | - |
| W8 | IS 实验 | - |
| W9 | OOS 验证 | - |
| W10 | 结果分析 | 架构设计 |
| W11 | - | 编码实现 |
| W12 | - | IS/OOS 实验 |

---

## Appendix: 实验配置模板

```yaml
# templates/experiment_config.yaml
experiment:
  id: "${EXPERIMENT_ID}"
  name: "${EXPERIMENT_NAME}"
  version: "0.1.0"
  created: "${DATE}"
  
model:
  architecture: "${ARCHITECTURE}"
  # Model-specific parameters
  
training:
  optimizer: "adam"
  learning_rate: 1e-4
  batch_size: 256
  epochs: 100
  
data:
  train_period: ["2015-01-01", "2021-12-31"]
  val_period: ["2022-01-01", "2022-12-31"]  
  test_period: ["2023-01-01", "2023-12-31"]
  
evaluation:
  metrics:
    - sharpe_ratio
    - pricing_error
    - alpha
  baselines:
    - "C"
```
