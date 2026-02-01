# SDF Layer Implementation Guide

> **Document ID**: SDF_IMPLEMENTATION_GUIDE  
> **Version**: 1.0.0  
> **Created**: 2026-02-01  
> **Status**: ✅ ACTIVE  
> **Upstream**: SDF_SPEC_REVIEW_001 (Completed)  
> **Downstream**: SDF_DEV_001 (Ready to Start)

---

## 0. Executive Summary

本文档基于 `SDF_REVIEW_DECISIONS.md` 的专家评审结论，为 SDF Layer 开发提供实现指导。

### 开发目标
将 SDF Layer Specification v3.1 落地为生产级 Python 模块，实现：
- State Engine (XState encoder)
- SDF Model (Generative SDF)
- Robust Moment Estimation
- Training Pipeline (Minimax + SmoothMax)
- EA Pricing Oracle API

### 关键约束
- **Boundedness**: c = 4.0 (frozen)
- **Normalization**: E[m] = 1 (frozen)
- **Instruments**: J = 4 [1, V, L, V·L] (baseline)
- **Robust**: MAD scaling + clip ±3.0

---

## 1. Design Decisions Summary

### 1.1 Frozen Parameters (不可修改)

| 参数 | 值 | 来源 |
|------|-----|------|
| `c` (SDF bound) | 4.0 | SDF Final Spec v1.0 §2.2 |
| `E[m]` (normalization) | 1.0 | SDF Final Spec v1.0 §2.2 |
| `λ_smooth` | 10⁻³ | SDF Final Spec v1.0 §2.2 |
| `c_y` (clip bound) | 3.0 | SDF Final Spec v1.0 §3.2 |
| `J` (instruments) | 4 | SDF Final Spec v1.0 §7 |
| `τ_start` | 5 | SDF Final Spec v1.0 §4.2 |
| `τ_end` | 20 | SDF Final Spec v1.0 §4.2 |
| `warmup` | 10 epochs | SDF Final Spec v1.0 §4.2 |

### 1.2 Architecture Decisions

| 模块 | 决策 | 理由 |
|------|------|------|
| Set Encoder | Mean pooling (DeepSets) | 复杂度低，数据量匹配 |
| Scaling | MAD | 稳健于 outliers |
| Instrument | [1, V, L, V·L] | 低维可解释 |
| Objective | SmoothMax minimax | 逼近 worst-case no-arbitrage |

---

## 2. Module Interface Contracts

### 2.1 State Engine (`state_engine.py`)

```python
class StateEngine:
    """
    State Engine v1.0 - Construct instrument basis from market state.
    
    Spec Reference: State Engine Spec v1.0
    """
    
    def __init__(
        self,
        vol_lookback: int = 12,      # Frozen
        std_window: int = 36,         # Frozen
        lambda_ewma: float = 0.8,     # Frozen
        include_crowd: bool = False   # Optional (J=5)
    ):
        ...
    
    @property
    def J(self) -> int:
        """Instrument dimension: 4 (baseline) or 5 (with crowd)"""
        return 5 if self.include_crowd else 4
    
    def compute_volatility_state(
        self, 
        returns: np.ndarray  # Shape: [T, K]
    ) -> np.ndarray:         # Shape: [T]
        """
        Compute market volatility state V_t.
        
        Method: Rolling std → EWMA smoothing → tanh normalization
        Output range: [-1, 1]
        """
        ...
    
    def compute_liquidity_state(
        self,
        turnover: np.ndarray  # Shape: [T, K]
    ) -> np.ndarray:          # Shape: [T]
        """
        Compute market liquidity state L_t.
        
        Method: Cross-sectional median → EWMA → tanh normalization
        Output range: [-1, 1]
        """
        ...
    
    def construct_basis(
        self,
        V_t: np.ndarray,              # Shape: [T]
        L_t: np.ndarray,              # Shape: [T]
        C_t: Optional[np.ndarray] = None  # Shape: [T] (optional)
    ) -> np.ndarray:                  # Shape: [T, J]
        """
        Construct instrument basis phi_t.
        
        Baseline (J=4): phi = [1, V_t, L_t, V_t * L_t]
        Extended (J=5): phi = [1, V_t, L_t, V_t * L_t, C_t]
        
        Invariant: All elements in [-1, 1] (except constant 1)
        """
        ...
```

**Input/Output Contract**:

| 输入 | 类型 | 约束 |
|------|------|------|
| `returns` | `np.ndarray[T, K]` | K = asset 数, T = 时间步 |
| `turnover` | `np.ndarray[T, K]` | 换手率 |

| 输出 | 类型 | 约束 |
|------|------|------|
| `phi` | `np.ndarray[T, J]` | J=4 或 5, 元素 ∈ [-1, 1] |

---

### 2.2 SDF Model (`model.py`)

```python
class GenerativeSDF(nn.Module):
    """
    Generative SDF Model v3.1
    
    Architecture:
        log m_t = c · tanh(h_θ(Info_t))
        m_t = exp(log m_t)
    
    Spec Reference: SDF Layer Specification v3.1 §2
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        c: float = 4.0,           # Frozen
        lambda_smooth: float = 1e-3  # Frozen
    ):
        ...
    
    def forward(
        self,
        info_t: torch.Tensor  # Shape: [T, input_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            m_t: SDF values, shape [T], positive
            log_m_t: Log SDF values, shape [T], bounded by [-c, c]
        
        Invariant: mean(m_t) ≈ 1 (enforced by normalization)
        """
        ...
    
    def normalize(
        self,
        m_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce E[m_t] = 1 normalization.
        """
        return m_t / m_t.mean()
```

**Constraints**:

| 约束 | 公式 | 验证方式 |
|------|------|----------|
| Boundedness | `log m ∈ [-4, 4]` | `assert log_m.abs().max() <= c` |
| Positivity | `m > 0` | `assert m.min() > 0` |
| Normalization | `E[m] = 1` | `assert abs(m.mean() - 1) < 1e-6` |

---

### 2.3 Robust Moment Estimator (`moments.py`)

```python
class RobustMomentEstimator:
    """
    Robust Moment Estimation with MAD scaling and clipping.
    
    Spec Reference: SDF Layer Final Spec v1.0 §3
    """
    
    def __init__(
        self,
        scale_method: str = "MAD",  # Frozen
        clip_bound: float = 3.0,     # Frozen
        epsilon: float = 1e-8
    ):
        ...
    
    def compute_scale(
        self,
        returns: np.ndarray  # Shape: [T, N]
    ) -> np.ndarray:         # Shape: [N]
        """
        Compute MAD scale per asset.
        
        MAD_i = median(|r_i - median(r_i)|)
        """
        ...
    
    def scale_returns(
        self,
        returns: np.ndarray,  # Shape: [T, N]
        scale: np.ndarray     # Shape: [N]
    ) -> np.ndarray:          # Shape: [T, N]
        """
        Scale returns: r_tilde = r / (scale + epsilon)
        """
        ...
    
    def robust_aggregate(
        self,
        m_t: np.ndarray,       # Shape: [T]
        r_scaled: np.ndarray,  # Shape: [T, N]
    ) -> np.ndarray:           # Shape: [T, N]
        """
        Compute clipped y = clip(m * r, ±c_y)
        """
        y = m_t[:, None] * r_scaled
        return np.clip(y, -self.clip_bound, self.clip_bound)
    
    def compute_instrumented_moments(
        self,
        y_bar: np.ndarray,  # Shape: [T, N]
        phi: np.ndarray     # Shape: [T, J]
    ) -> np.ndarray:        # Shape: [N, J]
        """
        Compute G[i,j] = (1/T) * sum_t y_bar[t,i] * phi[t,j]
        """
        T = y_bar.shape[0]
        return (y_bar.T @ phi) / T  # [N, J]
```

---

### 2.4 Training Pipeline (`trainer.py`)

```python
class SDFTrainer:
    """
    SDF Training Pipeline with SmoothMax minimax objective.
    
    Spec Reference: SDF Layer Final Spec v1.0 §4-5
    """
    
    def __init__(
        self,
        model: GenerativeSDF,
        moment_estimator: RobustMomentEstimator,
        tau_start: float = 5.0,    # Frozen
        tau_end: float = 20.0,     # Frozen
        warmup_epochs: int = 10,   # Frozen
        total_epochs: int = 100
    ):
        ...
    
    def smooth_max(
        self,
        G: torch.Tensor,  # Shape: [N, J]
        tau: float
    ) -> torch.Tensor:    # Scalar
        """
        SmoothMax_τ(|g_{i,j}|) = (1/τ) * log(sum exp(τ * |g|))
        
        Numerical stability: subtract max before exp.
        """
        G_abs = G.abs().flatten()
        G_max = G_abs.max()
        return G_max + (1/tau) * torch.log(
            torch.exp(tau * (G_abs - G_max)).sum()
        )
    
    def get_tau(self, epoch: int) -> float:
        """
        Temperature schedule: linear warmup from τ_start to τ_end.
        """
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            return self.tau_start + alpha * (self.tau_end - self.tau_start)
        return self.tau_end
    
    def train_window(
        self,
        R_leaf: np.ndarray,  # Shape: [T, N] - 下一期回报 (t+1)
        Info_t: np.ndarray,  # Shape: [T, D] - t 时刻信息
        phi: np.ndarray      # Shape: [T, J] - t 时刻 instrument
    ) -> Dict[str, Any]:
        """
        Train SDF for one rolling window.
        
        Returns:
            - theta_star: Trained model parameters
            - m_t: Final SDF series
            - G_final: Final moment matrix
            - diagnostics: Training metrics
        """
        ...
```

---

### 2.5 EA Pricing Oracle (`oracle.py`)

```python
class PricingErrorOracle:
    """
    Pricing Error Oracle for EA integration.
    
    Spec Reference: SDF Layer Final Spec v1.0 §6
    """
    
    def __init__(
        self,
        m_t: np.ndarray,              # Shape: [T] - frozen SDF
        R_leaf: np.ndarray,           # Shape: [T, N]
        phi: np.ndarray,              # Shape: [T, J]
        moment_estimator: RobustMomentEstimator,
        tau: float = 20.0             # Use τ_end
    ):
        ...
    
    def compute_portfolio_return(
        self,
        w: np.ndarray  # Shape: [N]
    ) -> np.ndarray:   # Shape: [T]
        """
        Compute portfolio return: r_p = sum_i w_i * r_i
        """
        return self.R_leaf @ w
    
    def compute_pricing_error(
        self,
        w: np.ndarray  # Shape: [N]
    ) -> float:
        """
        Compute PE(w) = SmoothMax_τ(|g_j(w)|)
        
        EA objective: minimize PE(w)
        """
        r_p = self.compute_portfolio_return(w)
        r_p_scaled = r_p / (self.scale_p + self.eps)
        y_bar = np.clip(self.m_t * r_p_scaled, 
                       -self.clip_bound, self.clip_bound)
        g_j = (y_bar @ self.phi) / self.T  # [J]
        
        # SmoothMax
        g_abs = np.abs(g_j)
        g_max = g_abs.max()
        pe = g_max + (1/self.tau) * np.log(
            np.exp(self.tau * (g_abs - g_max)).sum()
        )
        return float(pe)
    
    def __call__(self, w: np.ndarray) -> float:
        """Callable interface for EA."""
        return self.compute_pricing_error(w)
```

---

## 3. Implementation Priorities

### 3.1 Phase 1 (Week 1): Core Modules

| 任务 ID | 模块 | 工作量 | 依赖 |
|---------|------|--------|------|
| SDF_DEV_001.1 | State Engine | 8h | None |
| SDF_DEV_001.2 | SDF Model | 6h | None |
| SDF_DEV_001.3 | Robust Moments | 6h | None |

### 3.2 Phase 2 (Week 2): Training & Integration

| 任务 ID | 模块 | 工作量 | 依赖 |
|---------|------|--------|------|
| SDF_DEV_001.4 | Training Pipeline | 10h | .1, .2, .3 |
| SDF_DEV_001.5 | EA Oracle | 6h | .3, .4 |
| SDF_DEV_001.6 | PanelTree Integration | 8h | .4 |

### 3.3 Dependency Graph

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ State Engine    │     │ SDF Model       │     │ Robust Moments  │
│ (SDF_DEV_001.1) │     │ (SDF_DEV_001.2) │     │ (SDF_DEV_001.3) │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Training Pipeline      │
                    │ (SDF_DEV_001.4)        │
                    └────────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                                     │
              ▼                                     ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│ EA Oracle               │          │ PanelTree Integration   │
│ (SDF_DEV_001.5)         │          │ (SDF_DEV_001.6)         │
└─────────────────────────┘          └─────────────────────────┘
```

---

## 4. Testing Requirements

### 4.1 P0 Tests (必须通过)

| 测试类别 | 测试项 | 验证条件 |
|----------|--------|----------|
| **No-Leakage** | phi[t] 不依赖 r[t+1] | 时间戳断言 |
| **Boundedness** | phi ∈ [-1, 1] | 范围检查 |
| **Determinism** | 相同输入 → 相同输出 | 重复运行比对 |
| **Consistency** | 训练 loss 与 oracle 同口径 | G 矩阵比对 |
| **Normalization** | E[m] = 1 | 均值断言 |
| **SDF Bound** | log m ∈ [-4, 4] | 范围检查 |

### 4.2 P1 Tests (推荐)

| 测试类别 | 测试项 | 验证条件 |
|----------|--------|----------|
| **Stability** | 相邻窗口 phi 无异常跳变 | Δphi < threshold |
| **Robustness** | 极端样本不主导 SmoothMax | G_max 分布检查 |
| **Discrimination** | PE(w) 对不同 w 有区分度 | 方差检查 |

### 4.3 Test Coverage Target

```
Module           Target Coverage
─────────────────────────────────
state_engine.py       90%
model.py              90%
moments.py            85%
trainer.py            80%
oracle.py             85%
─────────────────────────────────
Overall               85%
```

---

## 5. Known Risks & Mitigations

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SmoothMax 数值溢出 | 训练失败 | max-shift trick |
| MAD = 0 (低波动资产) | 除零错误 | epsilon 保护 |
| m_t 退化为常数 | SDF 无效 | 监控 m_t 分布 |
| τ warmup 过快 | 早期过拟合 | 严格遵循 schedule |
| phi 跨窗口跳变 | 不稳定 | EWMA 平滑 (已内置) |

---

## 6. File Structure

```
projects/dgsf/repo/src/dgsf/sdf/
├── __init__.py
├── state_engine.py      # StateEngine class
├── model.py             # GenerativeSDF class
├── moments.py           # RobustMomentEstimator class
├── trainer.py           # SDFTrainer class
├── oracle.py            # PricingErrorOracle class
├── config.py            # Frozen hyperparameters
└── utils.py             # Helper functions

projects/dgsf/repo/tests/sdf/
├── __init__.py
├── test_state_engine.py
├── test_model.py
├── test_moments.py
├── test_trainer.py
├── test_oracle.py
└── test_integration.py
```

---

## 7. Configuration File

```yaml
# projects/dgsf/configs/sdf_config.yaml

# SDF Model (Frozen)
sdf:
  c: 4.0
  lambda_smooth: 1.0e-3
  hidden_dim: 64
  num_layers: 2

# Robust Moments (Frozen)
moments:
  scale_method: "MAD"
  clip_bound: 3.0
  epsilon: 1.0e-8

# State Engine (Frozen)
state_engine:
  vol_lookback: 12
  std_window: 36
  lambda_ewma: 0.8
  include_crowd: false  # J=4

# Training (Frozen)
training:
  tau_start: 5.0
  tau_end: 20.0
  warmup_epochs: 10
  total_epochs: 100
  learning_rate: 1.0e-3
  optimizer: "Adam"

# Instruments (Frozen)
instruments:
  J: 4
  basis: ["1", "V", "L", "V*L"]
```

---

## Appendix A: Mathematical Reference

### A.1 SDF Parameterization
$$
\log m_t = c \cdot \tanh(h_\theta(\text{Info}_t))
$$
$$
m_t = \exp(\log m_t), \quad \text{normalized: } E[m_t] = 1
$$

### A.2 Instrumented Moments
$$
g_{i,j} = \frac{1}{T} \sum_{t=1}^{T} \bar{y}_{t,i} \cdot \phi_{t,j}
$$

Where:
$$
\bar{y}_{t,i} = \text{clip}(m_t \cdot \tilde{r}_{t+1,i}, \pm c_y)
$$

### A.3 SmoothMax Objective
$$
\mathcal{L}(\theta) = \text{SmoothMax}_\tau(|g_{i,j}|) = \frac{1}{\tau} \log \sum_{i,j} \exp(\tau |g_{i,j}|)
$$

### A.4 EA Pricing Error Oracle
$$
\text{PE}(w) = \text{SmoothMax}_\tau(|g_j(w)|)
$$

Where:
$$
g_j(w) = \frac{1}{T} \sum_t \text{clip}(m_t \cdot \tilde{r}_{p,t+1}(w), \pm c_y) \cdot \phi_{t,j}
$$

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-01 | Expert Panel | Initial release |
