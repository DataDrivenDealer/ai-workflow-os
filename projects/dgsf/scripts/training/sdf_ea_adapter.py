"""
SDF-EA Interface Adapter for DGSF Stage 5.

This module provides the bridge between Stage 4 (SDF Layer) and Stage 5 (EA Optimizer).
It implements the interface contract defined in SDF_INTERFACE_CONTRACT.yaml Section 6.

Key responsibilities:
1. Load trained SDF model and provide frozen m_t
2. Expose pricing_error_oracle(w) callable for EA objective f4
3. Ensure NO leakage (SDF frozen per window)

Author: DGSF Pipeline
Date: 2026-02-04
Stage: 5 (EA Optimizer Development)
Task: EA_DEV_001 P0-16
"""

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Callable

# Ensure legacy DGSF src is importable
LEGACY_SRC = Path(__file__).parent.parent / "legacy" / "DGSF" / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))


class SDFEAAdapter:
    """
    Adapter connecting SDF Layer outputs to EA Layer inputs.
    
    Implements the contract from SDF_INTERFACE_CONTRACT.yaml:
    - sdf_provides: m_t, phi, pricing_error_oracle
    - ea_uses: 4-objective fitness (Sharpe, MDD, Turnover, PE)
    
    Invariants (enforced):
    - EA does NOT modify m_t (SDF is environment)
    - EA does NOT retrain SDF (frozen per window)
    - PE(w) uses SAME moment formula as training
    
    Parameters
    ----------
    sdf_model : torch.nn.Module
        Trained SDF model (frozen, requires_grad=False)
    X_sdf : torch.Tensor
        SDF input features [T, D] - deterministic phi basis
    leaf_returns : torch.Tensor
        Leaf portfolio returns [T, K] from PanelTree
    device : str
        Compute device ('cuda' or 'cpu')
    
    Examples
    --------
    >>> adapter = SDFEAAdapter(sdf_model, X_sdf, leaf_returns)
    >>> pe = adapter.pricing_error_oracle(weights)  # PE(w) for f4
    >>> fitness = adapter.compute_full_fitness(weights)  # 4-objective vector
    """
    
    def __init__(
        self,
        sdf_model: torch.nn.Module,
        X_sdf: torch.Tensor,
        leaf_returns: torch.Tensor,
        device: str = "cpu",
    ):
        """Initialize the SDF-EA adapter with frozen SDF model."""
        self.device = torch.device(device)
        
        # Freeze SDF model - CRITICAL: EA must NOT modify SDF
        self.sdf_model = sdf_model.to(self.device)
        self.sdf_model.eval()
        for param in self.sdf_model.parameters():
            param.requires_grad = False
        
        # Store data tensors
        self.X_sdf = X_sdf.to(self.device, dtype=torch.float32)
        self.leaf_returns = leaf_returns.to(self.device, dtype=torch.float32)
        
        # Compute and cache m_t (SDF series) - frozen for this window
        with torch.no_grad():
            output = self.sdf_model(self.X_sdf)
            # GenerativeSDF.forward() returns (m, z) tuple
            if isinstance(output, tuple):
                self.m_t = output[0]  # m is first element
            else:
                self.m_t = output
            
            # Ensure 1D: [T]
            if self.m_t.ndim == 2:
                self.m_t = self.m_t.squeeze(-1)  # [T]
        
        # Validate shapes
        T_sdf, D = self.X_sdf.shape
        T_ret, K = self.leaf_returns.shape
        assert T_sdf == T_ret, f"Time mismatch: X_sdf has {T_sdf}, leaf_returns has {T_ret}"
        
        self.T = T_sdf
        self.K = K
        self.D = D
        
        print(f"[SDFEAAdapter] Initialized: T={self.T}, K={self.K}, D={self.D}")
        print(f"[SDFEAAdapter] m_t range: [{self.m_t.min():.4f}, {self.m_t.max():.4f}]")
    
    def pricing_error_oracle(self, w: np.ndarray) -> float:
        """
        Compute pricing error PE(w) = |E[m_t * (w^T R_t)]|.
        
        This is the f4 objective in EA optimization.
        Uses the SAME moment formula as SDF training.
        
        Parameters
        ----------
        w : np.ndarray
            Portfolio weights [K], must sum to 1.
        
        Returns
        -------
        pe : float
            Absolute pricing error |g^(w)|
        
        Notes
        -----
        Contract: PE normalization = NONE (keep raw scale)
        """
        w_tensor = torch.tensor(w, dtype=torch.float32, device=self.device)
        
        # Strategy returns: R_w = w^T * R_leaf  [T]
        R_w = torch.matmul(self.leaf_returns, w_tensor)  # [T]
        
        # Pricing error: E[m_t * R_w]
        moment = self.m_t * R_w  # [T]
        pe = torch.abs(moment.mean()).item()
        
        return pe
    
    def compute_sharpe(self, w: np.ndarray, periods_per_year: int = 12) -> float:
        """
        Compute annualized Sharpe ratio for portfolio weights.
        
        Parameters
        ----------
        w : np.ndarray
            Portfolio weights [K]
        periods_per_year : int
            Annualization factor (12 for monthly)
        
        Returns
        -------
        sharpe : float
            Annualized Sharpe ratio
        """
        w_tensor = torch.tensor(w, dtype=torch.float32, device=self.device)
        R_w = torch.matmul(self.leaf_returns, w_tensor)  # [T]
        
        mean_ret = R_w.mean().item()
        std_ret = R_w.std().item()
        
        if std_ret < 1e-8:
            return 0.0
        
        sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
        return sharpe
    
    def compute_max_drawdown(self, w: np.ndarray) -> float:
        """
        Compute maximum drawdown for portfolio weights.
        
        Parameters
        ----------
        w : np.ndarray
            Portfolio weights [K]
        
        Returns
        -------
        mdd : float
            Maximum drawdown (positive value, to minimize)
        """
        w_tensor = torch.tensor(w, dtype=torch.float32, device=self.device)
        R_w = torch.matmul(self.leaf_returns, w_tensor)  # [T]
        
        # Cumulative returns
        cumulative = torch.cumprod(1 + R_w, dim=0)
        running_max = torch.cummax(cumulative, dim=0)[0]
        drawdown = (running_max - cumulative) / running_max
        mdd = drawdown.max().item()
        
        return mdd
    
    def compute_turnover(self, w: np.ndarray, w_prev: Optional[np.ndarray] = None) -> float:
        """
        Compute portfolio turnover.
        
        Parameters
        ----------
        w : np.ndarray
            Current portfolio weights [K]
        w_prev : np.ndarray or None
            Previous portfolio weights [K]
        
        Returns
        -------
        turnover : float
            Sum of absolute weight changes (0 if w_prev is None)
        """
        if w_prev is None:
            return 0.0
        
        return float(np.abs(w - w_prev).sum())
    
    def compute_full_fitness(
        self,
        w: np.ndarray,
        w_prev: Optional[np.ndarray] = None,
        periods_per_year: int = 12,
    ) -> np.ndarray:
        """
        Compute full 4-objective fitness vector for EA.
        
        Parameters
        ----------
        w : np.ndarray
            Portfolio weights [K]
        w_prev : np.ndarray or None
            Previous portfolio weights for turnover
        periods_per_year : int
            Annualization factor
        
        Returns
        -------
        fitness : np.ndarray
            4D vector [f1, f2, f3, f4] where:
            - f1 = -Sharpe (minimize → maximize Sharpe)
            - f2 = MDD (minimize)
            - f3 = Turnover (minimize)
            - f4 = PE(w) (minimize)
        
        All objectives are to be MINIMIZED.
        """
        sharpe = self.compute_sharpe(w, periods_per_year)
        mdd = self.compute_max_drawdown(w)
        turnover = self.compute_turnover(w, w_prev)
        pe = self.pricing_error_oracle(w)
        
        fitness = np.array([
            -sharpe,  # f1: minimize negative Sharpe
            mdd,      # f2: minimize MDD
            turnover, # f3: minimize turnover
            pe,       # f4: minimize pricing error
        ], dtype=np.float64)
        
        return fitness
    
    def get_sdf_summary(self) -> Dict:
        """Return summary of frozen SDF state."""
        return {
            "m_t_mean": float(self.m_t.mean()),
            "m_t_std": float(self.m_t.std()),
            "m_t_min": float(self.m_t.min()),
            "m_t_max": float(self.m_t.max()),
            "T": self.T,
            "K": self.K,
            "D": self.D,
            "device": str(self.device),
        }


def create_adapter_from_data(
    X_sdf: np.ndarray,
    leaf_returns: np.ndarray,
    sdf_hidden_dim: int = 32,
    sdf_num_layers: int = 2,
    device: str = "cpu",
) -> SDFEAAdapter:
    """
    Factory function to create SDFEAAdapter with synthetic SDF model.
    
    For Stage 5 development/testing before real trained SDF is available.
    
    Parameters
    ----------
    X_sdf : np.ndarray
        SDF features [T, D]
    leaf_returns : np.ndarray
        Leaf returns [T, K]
    sdf_hidden_dim : int
        Hidden dimension for synthetic SDF
    sdf_num_layers : int
        Number of hidden layers
    device : str
        Compute device
    
    Returns
    -------
    adapter : SDFEAAdapter
        Initialized adapter with synthetic SDF model
    """
    from dgsf.sdf import GenerativeSDF
    
    D = X_sdf.shape[1]
    
    # Create synthetic SDF model
    sdf_model = GenerativeSDF(
        input_dim=D,
        hidden_dim=sdf_hidden_dim,
        num_hidden_layers=sdf_num_layers,
        activation="tanh",
        output_activation="softplus",
    )
    
    # Convert to tensors
    X_tensor = torch.tensor(X_sdf, dtype=torch.float32)
    R_tensor = torch.tensor(leaf_returns, dtype=torch.float32)
    
    return SDFEAAdapter(sdf_model, X_tensor, R_tensor, device)


def smoke_test():
    """Smoke test for SDFEAAdapter."""
    print("=" * 60)
    print("SDFEAAdapter Smoke Test")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    T, K, D = 50, 10, 20
    
    X_sdf = np.random.randn(T, D)
    leaf_returns = np.random.randn(T, K) * 0.02  # ~2% monthly vol
    
    # Create adapter
    adapter = create_adapter_from_data(X_sdf, leaf_returns)
    
    # Test with random weights
    w = np.abs(np.random.randn(K))
    w = w / w.sum()  # Normalize to sum to 1
    
    print(f"\nTest weights: {w[:3]}... (sum={w.sum():.4f})")
    
    # Test individual metrics
    pe = adapter.pricing_error_oracle(w)
    sharpe = adapter.compute_sharpe(w)
    mdd = adapter.compute_max_drawdown(w)
    
    print(f"\n[Metrics]")
    print(f"  Pricing Error (f4): {pe:.6f}")
    print(f"  Sharpe Ratio:       {sharpe:.4f}")
    print(f"  Max Drawdown:       {mdd:.4f}")
    
    # Test full fitness
    fitness = adapter.compute_full_fitness(w)
    print(f"\n[4-Objective Fitness]")
    print(f"  f1 (-Sharpe):  {fitness[0]:.4f}")
    print(f"  f2 (MDD):      {fitness[1]:.4f}")
    print(f"  f3 (Turnover): {fitness[2]:.4f}")
    print(f"  f4 (PE):       {fitness[3]:.6f}")
    
    # Test turnover with prev weights
    w_prev = np.ones(K) / K  # Equal weight
    fitness_with_turnover = adapter.compute_full_fitness(w, w_prev)
    print(f"\n[With Turnover from EW]")
    print(f"  f3 (Turnover): {fitness_with_turnover[2]:.4f}")
    
    # SDF summary
    summary = adapter.get_sdf_summary()
    print(f"\n[SDF Summary]")
    print(f"  m_t mean: {summary['m_t_mean']:.4f}")
    print(f"  m_t std:  {summary['m_t_std']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ SDFEAAdapter Smoke Test PASSED")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    smoke_test()
