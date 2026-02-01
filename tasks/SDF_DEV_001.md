---
task_id: "SDF_DEV_001"
type: dev
queue: dev
branch: "feature/SDF_DEV_001"
priority: P0
spec_ids:
  - "DGSF_SDF_V3.1"
  - "STATE_ENGINE_V1.0"
  - "SDF_IMPLEMENTATION_GUIDE"
verification:
  - "All modules implemented and tested"
  - "Unit test coverage >80%"
  - "PanelTree integration verified"
---

# TaskCard: SDF_DEV_001

> **Phase**: 1 · SDF Layer Development  
> **Pipeline**: DGSF Development Pipeline  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `SDF_DEV_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `developer` |
| **Authority** | `accepted` |
| **Authorized By** | Project Owner |
| **上游 Task** | `SDF_SPEC_REVIEW_001` |
| **下游 Task** | `SDF_INTEGRATION_001` |

---

## 1. 任务背景

### 1.1 目标

将 SDF Layer Specification v3.1 落地为生产级 Python 模块，完成：
- State Engine 实现
- SDF Model 整合
- Robust Moment Estimation
- Training Pipeline
- EA Pricing Oracle API

### 1.2 现有代码资产

| 文件 | 功能 | 状态 | 行动 |
|------|------|------|------|
| `model.py` | GenerativeSDF 模型 | ✅ 可用 | 整合 |
| `losses.py` | 损失函数 | ⏳ 部分 | 补全 |
| `training.py` | 训练框架 | ⏳ 基础 | 重构 |
| `a0_sdf_dataloader.py` | 数据加载 | ✅ 可用 | 整合 |
| `a0_sdf_trainer.py` | 训练器 | ⏳ 实验性 | 重构 |
| `features.py` | 特征工程 | ✅ 可用 | 整合 |

---

## 2. 子任务分解

### 2.1 SDF_DEV_001.1: State Engine 实现

**目标**: 实现 State Engine v1.0，构造 instrument basis

**规范参考**: 
- `State Engine Spec v1.0.txt`
- `SDF Layer Final Spec v1.0.txt` §1.1

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/sdf/state_engine.py

class StateEngine:
    """
    State Engine v1.0 - Construct instrument basis from market state.
    
    Inputs:
        - V_t: Volatility state (from market data)
        - L_t: Liquidity state (from market data)
        - C_t: Crowd state (optional)
    
    Outputs:
        - phi_t: Instrument basis [1, V_t, L_t, V_t·L_t] (J=4)
        - Or [1, V_t, L_t, V_t·L_t, C_t] (J=5)
    """
    
    def __init__(self, include_crowd: bool = False):
        self.include_crowd = include_crowd
        self.J = 5 if include_crowd else 4
    
    def compute_volatility_state(self, returns: np.ndarray) -> np.ndarray:
        """Compute volatility state from returns."""
        # Rolling std, EWMA, or realized vol
        pass
    
    def compute_liquidity_state(self, turnover: np.ndarray) -> np.ndarray:
        """Compute liquidity state from turnover."""
        pass
    
    def construct_basis(self, V_t: np.ndarray, L_t: np.ndarray, 
                       C_t: Optional[np.ndarray] = None) -> np.ndarray:
        """Construct instrument basis phi_t."""
        # phi = [1, V_t, L_t, V_t * L_t, (C_t)]
        pass
```

**验收标准**:
- [ ] StateEngine 类实现完整
- [ ] 支持 J=4 和 J=5 配置
- [ ] 单元测试覆盖

**工作量**: 8h

---

### 2.2 SDF_DEV_001.2: SDF Model 整合

**目标**: 将 `model.py` 整合为生产级代码，确保 boundedness 和 normalization

**规范参考**:
- `SDF Layer Specification v3.1.md` §2
- `SDF Layer Final Spec v1.0.txt` §2

**实现内容**:

```python
# 整合到 projects/dgsf/repo/src/dgsf/sdf/model.py

class GenerativeSDF(nn.Module):
    """
    Generative SDF Model v3.1
    
    Architecture:
        log m_t = c · tanh(h_θ(Info_t))
        m_t = exp(log m_t)
    
    Constraints:
        - Boundedness: c = 4.0 (frozen)
        - Normalization: E[m_t] = 1
        - Temporal smoothness (optional)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, c: float = 4.0):
        super().__init__()
        self.c = c  # Boundedness parameter (frozen)
        # ... network layers
    
    def forward(self, info_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            m_t: SDF values (T,), strictly positive
            z_t: Hidden embeddings (T, H)
        """
        pass
    
    def normalize(self, m_t: torch.Tensor) -> torch.Tensor:
        """Enforce E[m_t] = 1 normalization."""
        return m_t / m_t.mean()
```

**验收标准**:
- [ ] Boundedness c=4.0 硬编码
- [ ] Mean normalization 实现
- [ ] Temporal smoothness 可选
- [ ] 单元测试覆盖

**工作量**: 6h

---

### 2.3 SDF_DEV_001.3: Robust Moment Estimation

**目标**: 实现 heavy-tail robust 的 moment 估计

**规范参考**:
- `SDF Layer Final Spec v1.0.txt` §3

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/sdf/moments.py

class RobustMomentEstimator:
    """
    Robust Moment Estimation for SDF training.
    
    Steps:
        1. Scale returns: r̃ = r / (MAD + ε)
        2. Robust aggregation: ȳ = clip(m · r̃, ±c_y)
        3. Instrumented moments: g_{i,j} = mean(ȳ · φ_j)
    """
    
    def __init__(self, scaling: str = "mad", c_y: float = 3.0):
        self.scaling = scaling
        self.c_y = c_y
    
    def scale_returns(self, returns: np.ndarray) -> np.ndarray:
        """Scale returns by MAD or EWMA."""
        if self.scaling == "mad":
            scale = np.median(np.abs(returns - np.median(returns)))
        elif self.scaling == "ewma":
            # EWMA implementation
            pass
        return returns / (scale + 1e-8)
    
    def robust_aggregate(self, m_t: np.ndarray, 
                         r_tilde: np.ndarray) -> np.ndarray:
        """Robust time aggregation with clipping."""
        y = m_t * r_tilde
        return np.clip(y, -self.c_y, self.c_y)
    
    def compute_instrumented_moments(self, y_bar: np.ndarray,
                                     phi: np.ndarray) -> np.ndarray:
        """
        Compute G[i,j] = (1/T) Σ_t ȳ_{t,i} · φ_{t,j}
        
        Returns:
            G: (N, J) instrumented moment matrix
        """
        pass
```

**验收标准**:
- [ ] MAD scaling 实现
- [ ] Clipping c_y=3.0 实现
- [ ] G[i,j] 矩阵计算正确
- [ ] 单元测试覆盖

**工作量**: 6h

---

### 2.4 SDF_DEV_001.4: Training Pipeline

**目标**: 实现 SmoothMax 目标函数和完整训练循环

**规范参考**:
- `SDF Layer Final Spec v1.0.txt` §4, §5

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/sdf/trainer.py

class SDFTrainer:
    """
    SDF Training Pipeline v3.1
    
    Objective:
        L(θ) = SmoothMax_τ(|g_{i,j}|)
             = (1/τ) log Σ_{i,j} exp(τ |g_{i,j}|)
    
    Temperature Schedule:
        τ: τ_start → τ_end over warmup epochs
    """
    
    def __init__(self, model: GenerativeSDF,
                 state_engine: StateEngine,
                 moment_estimator: RobustMomentEstimator,
                 tau_start: float = 5.0,
                 tau_end: float = 20.0,
                 warmup_epochs: int = 10):
        self.model = model
        self.state_engine = state_engine
        self.moment_estimator = moment_estimator
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.warmup_epochs = warmup_epochs
    
    def smooth_max(self, g: torch.Tensor, tau: float) -> torch.Tensor:
        """SmoothMax aggregation (log-sum-exp)."""
        g_abs = torch.abs(g.flatten())
        return (1/tau) * torch.logsumexp(tau * g_abs, dim=0)
    
    def get_tau(self, epoch: int) -> float:
        """Get temperature for current epoch."""
        if epoch < self.warmup_epochs:
            return self.tau_start + (self.tau_end - self.tau_start) * (epoch / self.warmup_epochs)
        return self.tau_end
    
    def train_window(self, R_leaf: np.ndarray, Info: np.ndarray,
                     num_epochs: int = 100) -> Dict:
        """
        Train SDF for a single rolling window.
        
        Args:
            R_leaf: (T, K) leaf portfolio returns (t+1)
            Info: (T, D) information at time t
        
        Returns:
            Trained model state and diagnostics
        """
        pass
```

**验收标准**:
- [ ] SmoothMax 实现 (数值稳定)
- [ ] τ schedule 实现
- [ ] Window-level 训练循环
- [ ] 训练诊断输出
- [ ] 单元测试覆盖

**工作量**: 8h

---

### 2.5 SDF_DEV_001.5: EA Pricing Oracle

**目标**: 实现 EA 调用的 pricing error oracle API

**规范参考**:
- `SDF Layer Final Spec v1.0.txt` §6
- `EA Layer Specification v3.1.md` §1.3

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/sdf/oracle.py

class PricingErrorOracle:
    """
    EA Pricing Error Oracle v3.1
    
    Provides PE(w) for any candidate portfolio weights.
    
    Definition:
        r_p(w) = Σ_i w_i · r_i
        g_j(w) = (1/T) Σ_t clip(m_t · r̃_p, ±c_y) · φ_{t,j}
        PE(w) = SmoothMax_τ(|g_j(w)|)
    """
    
    def __init__(self, trained_sdf: GenerativeSDF,
                 state_engine: StateEngine,
                 moment_estimator: RobustMomentEstimator,
                 tau: float = 20.0):
        self.sdf = trained_sdf
        self.state_engine = state_engine
        self.moment_estimator = moment_estimator
        self.tau = tau
        
        # Cache for efficiency
        self._m_t = None
        self._phi_t = None
    
    def precompute(self, Info: np.ndarray) -> None:
        """Precompute m_t and φ_t for the window."""
        with torch.no_grad():
            self._m_t, _ = self.sdf(torch.tensor(Info))
            self._phi_t = self.state_engine.construct_basis(...)
    
    def compute_pe(self, w: np.ndarray, R_leaf: np.ndarray) -> float:
        """
        Compute pricing error for candidate weights.
        
        Args:
            w: (K,) portfolio weights over leaf portfolios
            R_leaf: (T, K) leaf portfolio returns
        
        Returns:
            PE(w): scalar pricing error
        """
        # r_p = R_leaf @ w
        # g_j = instrumented moments for r_p
        # PE = smooth_max(|g_j|)
        pass
    
    def __call__(self, w: np.ndarray, R_leaf: np.ndarray) -> float:
        """Callable interface for EA."""
        return self.compute_pe(w, R_leaf)
```

**验收标准**:
- [ ] PE(w) API 与 EA v3.1 对齐
- [ ] Precompute 缓存机制
- [ ] 与训练目标同口径
- [ ] 性能测试 (EA 大 population 场景)

**工作量**: 4h

---

### 2.6 SDF_DEV_001.6: PanelTree 联调

**目标**: 验证 SDF 与 PanelTree 的端到端数据流

**实现内容**:

```python
# projects/dgsf/repo/tests/integration/test_sdf_paneltree_integration.py

def test_sdf_paneltree_dataflow():
    """
    End-to-end test: PanelTree → SDF
    
    1. Load PanelTree outputs (R_leaf)
    2. Verify R_leaf is (T, K) with proper dates
    3. Train SDF on R_leaf
    4. Verify m_t, G[i,j], PE(w) outputs
    5. Compare vs CB-L3 linear baseline
    """
    pass

def test_causality_preservation():
    """
    Verify no look-ahead leakage.
    
    - Info_t only uses t and before
    - R_leaf[t+1] is t+1 return
    """
    pass

def test_vs_cb_l3_baseline():
    """
    Compare full SDF vs CB-L3 linear baseline.
    
    Full SDF must outperform linear baseline (Sharpe).
    """
    pass
```

**验收标准**:
- [ ] 端到端数据流验证
- [ ] 因果性保持验证
- [ ] vs CB-L3 baseline 对比
- [ ] 集成测试通过

**工作量**: 6h

---

## 3. 交付物

| 交付物 | 路径 | 状态 |
|--------|------|------|
| State Engine 模块 | `src/dgsf/sdf/state_engine.py` | `pending` |
| SDF Model 模块 | `src/dgsf/sdf/model.py` | `pending` |
| Robust Moments 模块 | `src/dgsf/sdf/moments.py` | `pending` |
| Trainer 模块 | `src/dgsf/sdf/trainer.py` | `pending` |
| Oracle 模块 | `src/dgsf/sdf/oracle.py` | `pending` |
| 单元测试 | `tests/unit/test_sdf_*.py` | `pending` |
| 集成测试 | `tests/integration/test_sdf_*.py` | `pending` |

---

## 4. 验收标准

### 4.1 功能验收
- [ ] 所有 6 个子模块实现完成
- [ ] 输入/输出契约符合 v3.1 规范
- [ ] EA Oracle API 可被 EA 调用

### 4.2 质量验收
- [ ] 单元测试覆盖 >80%
- [ ] 集成测试通过
- [ ] vs CB-L3 baseline 性能优于线性

### 4.3 因果性验收
- [ ] 无 look-ahead leakage
- [ ] t/t+1 分离正确

---

## 5. 时间估算

| 子任务 | 工作量 | 负责人 | 预计完成 |
|--------|--------|--------|----------|
| SDF_DEV_001.1 State Engine | 8h | 李架构 | 02/04 |
| SDF_DEV_001.2 SDF Model | 6h | 李架构 | 02/07 |
| SDF_DEV_001.3 Robust Moments | 6h | 李架构 | 02/10 |
| SDF_DEV_001.4 Training Pipeline | 8h | 李架构 | 02/13 |
| SDF_DEV_001.5 EA Oracle | 4h | 李架构 | 02/15 |
| SDF_DEV_001.6 Integration | 6h | 全员 | 02/17 |
| **总计** | **38h (~5 天)** | - | - |

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 训练不稳定 | 中 | 高 | τ schedule + gradient clip |
| 接口不兼容 | 低 | 高 | 提前定义契约 |
| 因果性泄漏 | 低 | 极高 | 严格 t/t+1 分离 |
| 性能不足 | 中 | 中 | 缓存 + 批处理 |

---

## 7. Gate & 下游依赖

- **Gate**: 所有验收标准通过
- **后续 TaskCard**: `SDF_INTEGRATION_001`, `EA_DEV_001`
- **依赖**: `SDF_SPEC_REVIEW_001` (规范审核完成)

---

## 8. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T23:45:00Z | system | `task_created` | 任务创建 |

