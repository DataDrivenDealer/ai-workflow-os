---
task_id: "EA_DEV_001"
type: dev
queue: dev
branch: "feature/EA_DEV_001"
priority: P1
spec_ids:
  - "EA_LAYER_SPEC_V3.1"
  - "SDF_INTERFACE_CONTRACT"
  - "PROJECT_DGSF"
verification:
  - "EA optimizer implemented and tested"
  - "Integration with SDF Oracle verified"
  - "Unit test coverage >80%"
  - "OOS portfolio performance validated"
---

# TaskCard: EA_DEV_001

> **Phase**: 2 · EA Layer Development  
> **Pipeline**: DGSF Development Pipeline  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `EA_DEV_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `developer` |
| **Authority** | `accepted` |
| **Authorized By** | Project Owner |
| **上游 Task** | `SDF_DEV_001` |
| **下游 Task** | `BASELINE_REPRO_001`, `FULL_BACKTEST_001` |
| **并行任务** | `DATA_EXPANSION_001` |

---

## 1. 任务背景

### 1.1 目标

实现 Evolutionary Algorithm (EA) Layer，用于：
- 基于 SDF Pricing Oracle 优化投资组合权重
- 最小化 Pricing Error PE(w) 来寻找高效前沿组合
- 与 PanelTree Leaf Portfolios 集成

### 1.2 EA Layer 职责

```
┌─────────────────────────────────────────────────────────────────┐
│                        EA Layer v3.1                             │
├─────────────────────────────────────────────────────────────────┤
│  Input:                                                         │
│    - R_leaf[T, N]: Leaf portfolio returns from PanelTree        │
│    - oracle(w) → PE: Pricing Error function from SDF            │
│                                                                 │
│  Optimization:                                                  │
│    min_w PE(w)  subject to constraints                          │
│                                                                 │
│  Output:                                                        │
│    - w_star[N]: Optimal portfolio weights                       │
│    - r_portfolio = R_leaf @ w_star: Combined returns            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 依赖关系

| 依赖项 | 提供方 | 接口 |
|--------|--------|------|
| `PricingErrorOracle` | SDF_DEV_001.5 | `oracle(w) → float` |
| `R_leaf` | PanelTree | `[T, N] np.ndarray` |
| `phi_t` | StateEngine | `[T, J] np.ndarray` |

---

## 2. 子任务分解

### 2.1 EA_DEV_001.1: EA 核心优化器

**目标**: 实现基于进化算法的权重优化器

**规范参考**:
- `EA Layer Specification v3.1.md` §1-2

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/ea/optimizer.py

from typing import Callable, Optional, Dict, List
import numpy as np

class EAOptimizer:
    """
    Evolutionary Algorithm Portfolio Optimizer v3.1
    
    Objective:
        min_w PE(w) = SmoothMax_τ(|g_j(w)|)
    
    Algorithm:
        - CMA-ES or Differential Evolution
        - Population-based optimization
        - Constraint handling via penalty
    """
    
    def __init__(
        self,
        n_assets: int,
        oracle: Callable[[np.ndarray], float],
        population_size: int = 50,
        max_generations: int = 100,
        algorithm: str = "cma-es"
    ):
        self.n_assets = n_assets
        self.oracle = oracle
        self.population_size = population_size
        self.max_generations = max_generations
        self.algorithm = algorithm
    
    def optimize(
        self,
        w_init: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Run EA optimization.
        
        Args:
            w_init: Initial weights [N] (default: equal weight)
            constraints: Optional constraints dict
                - "long_only": bool (w >= 0)
                - "sum_to_one": bool (sum(w) = 1)
                - "leverage_limit": float (sum(|w|) <= L)
        
        Returns:
            Dict with:
                - w_star: Optimal weights [N]
                - pe_star: Final pricing error
                - history: Optimization history
                - diagnostics: Algorithm diagnostics
        """
        pass
    
    def _fitness(self, w: np.ndarray, constraints: Dict) -> float:
        """Compute fitness = PE + penalty."""
        pass
    
    def _apply_constraints(self, population: np.ndarray) -> np.ndarray:
        """Project population to feasible region."""
        pass
```

**验收标准**:
- [ ] CMA-ES 算法实现
- [ ] 约束处理机制
- [ ] 支持 long-only 和 long-short
- [ ] 单元测试覆盖

**工作量**: 10h

---

### 2.2 EA_DEV_001.2: 约束系统

**目标**: 实现灵活的约束处理系统

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/ea/constraints.py

class ConstraintHandler:
    """
    Portfolio Constraint Handler
    
    Supported constraints:
        - Budget: sum(w) = 1 or sum(|w|) <= L
        - Long-only: w >= 0
        - Box: w_min <= w <= w_max
        - Sector exposure: sum(w[sector]) <= limit
    """
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.constraints = []
    
    def add_budget_constraint(self, target: float = 1.0):
        """Add budget constraint sum(w) = target."""
        pass
    
    def add_leverage_constraint(self, max_leverage: float = 2.0):
        """Add leverage constraint sum(|w|) <= L."""
        pass
    
    def add_long_only(self):
        """Add long-only constraint w >= 0."""
        pass
    
    def add_box_constraint(self, w_min: float, w_max: float):
        """Add box constraint w_min <= w <= w_max."""
        pass
    
    def project(self, w: np.ndarray) -> np.ndarray:
        """Project weights to feasible region."""
        pass
    
    def penalty(self, w: np.ndarray) -> float:
        """Compute constraint violation penalty."""
        pass
```

**验收标准**:
- [ ] Budget constraint 实现
- [ ] Leverage constraint 实现
- [ ] 投影算法正确
- [ ] 单元测试覆盖

**工作量**: 6h

---

### 2.3 EA_DEV_001.3: Oracle 集成适配器

**目标**: 实现 EA 与 SDF Oracle 的集成层

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/ea/oracle_adapter.py

class OracleAdapter:
    """
    Adapter between EA Optimizer and SDF Pricing Oracle.
    
    Handles:
        - Data alignment (T, N dimensions)
        - Caching for repeated oracle calls
        - Batch evaluation for population
    """
    
    def __init__(
        self,
        pricing_oracle,  # PricingErrorOracle from SDF
        R_leaf: np.ndarray,
        cache_enabled: bool = True
    ):
        self.oracle = pricing_oracle
        self.R_leaf = R_leaf
        self.cache = {} if cache_enabled else None
        self._call_count = 0
    
    def __call__(self, w: np.ndarray) -> float:
        """Callable interface for EA."""
        self._call_count += 1
        
        # Cache lookup
        w_key = tuple(w.round(6))
        if self.cache is not None and w_key in self.cache:
            return self.cache[w_key]
        
        # Oracle call
        pe = self.oracle.compute_pe(w, self.R_leaf)
        
        # Cache store
        if self.cache is not None:
            self.cache[w_key] = pe
        
        return pe
    
    def batch_evaluate(self, population: np.ndarray) -> np.ndarray:
        """Evaluate PE for entire population."""
        return np.array([self(w) for w in population])
    
    @property
    def call_count(self) -> int:
        return self._call_count
```

**验收标准**:
- [ ] Oracle 调用正确
- [ ] 缓存机制有效
- [ ] 批量评估性能优化
- [ ] 单元测试覆盖

**工作量**: 4h

---

### 2.4 EA_DEV_001.4: 回测集成

**目标**: 将 EA 优化结果集成到回测框架

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/ea/backtest_integration.py

class EABacktestRunner:
    """
    Run EA-based portfolio in rolling backtest.
    
    Rolling procedure:
        1. For each window [t-W, t]:
           - Train SDF
           - Get Oracle
           - Run EA optimization
           - Get w_star
        2. Apply w_star to [t, t+H] returns
        3. Compute performance metrics
    """
    
    def __init__(
        self,
        ea_optimizer: EAOptimizer,
        sdf_trainer,  # SDFTrainer
        window_size: int = 252,
        rebalance_freq: int = 21,
        holding_period: int = 21
    ):
        self.ea = ea_optimizer
        self.sdf = sdf_trainer
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        self.holding_period = holding_period
    
    def run_rolling(
        self,
        R_leaf: np.ndarray,
        Info: np.ndarray,
        dates: np.ndarray
    ) -> Dict:
        """
        Run rolling backtest.
        
        Returns:
            Dict with:
                - portfolio_returns: [T_oos] returns
                - weights_history: [N_rebal, N_assets]
                - pe_history: [N_rebal] pricing errors
                - metrics: Sharpe, MaxDD, Turnover
        """
        pass
```

**验收标准**:
- [ ] Rolling 框架正确
- [ ] 无 look-ahead leakage
- [ ] 性能指标计算
- [ ] 集成测试覆盖

**工作量**: 8h

---

### 2.5 EA_DEV_001.5: 基准对比

**目标**: 实现与基准策略的对比框架

**实现内容**:

```python
# projects/dgsf/repo/src/dgsf/ea/baselines.py

class BaselineComparison:
    """
    Compare EA portfolio against baselines.
    
    Baselines:
        - Equal Weight (1/N)
        - Mean-Variance (Markowitz)
        - Risk Parity
        - CB-L3 Linear SDF baseline
    """
    
    def __init__(self, R_leaf: np.ndarray):
        self.R_leaf = R_leaf
    
    def equal_weight(self) -> np.ndarray:
        """1/N portfolio."""
        N = self.R_leaf.shape[1]
        return np.ones(N) / N
    
    def mean_variance(self, gamma: float = 1.0) -> np.ndarray:
        """Mean-variance optimal."""
        pass
    
    def risk_parity(self) -> np.ndarray:
        """Risk parity weights."""
        pass
    
    def compare_all(
        self,
        w_ea: np.ndarray,
        metrics: List[str] = ["sharpe", "maxdd", "calmar"]
    ) -> pd.DataFrame:
        """Compare EA vs all baselines."""
        pass
```

**验收标准**:
- [ ] 三个基准实现
- [ ] 指标计算正确
- [ ] 对比报告生成
- [ ] 单元测试覆盖

**工作量**: 6h

---

## 3. 交付物

| 交付物 | 路径 | 状态 |
|--------|------|------|
| EA Optimizer 模块 | `src/dgsf/ea/optimizer.py` | `pending` |
| Constraint Handler | `src/dgsf/ea/constraints.py` | `pending` |
| Oracle Adapter | `src/dgsf/ea/oracle_adapter.py` | `pending` |
| Backtest Integration | `src/dgsf/ea/backtest_integration.py` | `pending` |
| Baselines | `src/dgsf/ea/baselines.py` | `pending` |
| 单元测试 | `tests/unit/test_ea_*.py` | `pending` |
| 集成测试 | `tests/integration/test_ea_*.py` | `pending` |

---

## 4. 验收标准

### 4.1 功能验收
- [ ] EA 优化器能找到低 PE 的组合
- [ ] 约束系统正确处理各类约束
- [ ] Oracle 集成无缝衔接

### 4.2 质量验收
- [ ] 单元测试覆盖 >80%
- [ ] 集成测试通过
- [ ] vs Baseline 性能可比较

### 4.3 因果性验收
- [ ] Rolling backtest 无 look-ahead
- [ ] 权重仅基于历史信息

---

## 5. 时间估算

| 子任务 | 工作量 | 负责人 | 预计完成 |
|--------|--------|--------|----------|
| EA_DEV_001.1 EA 核心 | 10h | Dr. Quant | 02/18 |
| EA_DEV_001.2 约束系统 | 6h | Dr. Quant | 02/19 |
| EA_DEV_001.3 Oracle 适配 | 4h | 李架构 | 02/20 |
| EA_DEV_001.4 回测集成 | 8h | Dr. Quant | 02/22 |
| EA_DEV_001.5 基准对比 | 6h | 李架构 | 02/23 |
| **总计** | **34h (~4.5 天)** | - | - |

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| CMA-ES 收敛慢 | 中 | 中 | 调参 + 早停 |
| Oracle 调用开销大 | 高 | 中 | 缓存 + 批量 |
| 约束投影复杂 | 低 | 中 | 使用成熟库 |
| OOS 表现差 | 中 | 高 | 多基准对比 |

---

## 7. Gate & 下游依赖

- **前置 Gate**: SDF_DEV_001 完成
- **本任务 Gate**: 所有验收标准通过
- **后续 TaskCard**: `BASELINE_REPRO_001`, `FULL_BACKTEST_001`

---

## 8. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T00:00:00Z | PM_Chen | `task_created` | TaskCard 创建 |

