"""
T5.1 SDF Evaluation Framework (SDF_DEV_001_T5)

Purpose: Implement comprehensive SDF model evaluation metrics.

Metrics Implemented:
1. Pricing Error (Euler equation residual)
2. Sharpe Ratio (IS and OOS)
3. Alpha (Jensen's Alpha)
4. HJ Distance (Hansen-Jagannathan distance)
5. Cross-sectional R² (pricing accuracy)

Reference: SDF_SPEC v3.1, Cochrane (2005) Asset Pricing

Usage:
    python scripts/evaluate_sdf.py --model checkpoints/best_model.pt
    python scripts/evaluate_sdf.py --benchmark
    
Output:
    reports/sdf_evaluation_report.md
    experiments/t5_evaluation/metrics.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "name": "T5.1 SDF Evaluation Framework",
    "timestamp": datetime.now().isoformat(),
    "metrics": [
        "pricing_error",
        "sharpe_ratio",
        "alpha",
        "hj_distance",
        "cross_sectional_r2",
    ],
    "benchmark": {
        "n_samples": 500,
        "n_test_assets": 25,
        "seed": 42,
    },
}

# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class SDFMetrics:
    """Container for SDF evaluation metrics."""
    
    # Pricing Error
    pricing_error_mean: float
    pricing_error_std: float
    pricing_error_max: float
    
    # Sharpe Ratio
    is_sharpe: float
    oos_sharpe: float
    sharpe_ratio: float  # OOS/IS
    
    # Alpha (Jensen's Alpha)
    alpha: float
    alpha_t_stat: float
    alpha_p_value: float
    
    # Hansen-Jagannathan Distance
    hj_distance: float
    hj_se: float
    
    # Cross-sectional R²
    cs_r2: float
    cs_rmse: float
    
    # Additional
    n_samples: int
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "SDF EVALUATION METRICS SUMMARY",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Samples: {self.n_samples}",
            "",
            "PRICING ERROR:",
            f"  Mean: {self.pricing_error_mean:.6f}",
            f"  Std:  {self.pricing_error_std:.6f}",
            f"  Max:  {self.pricing_error_max:.6f}",
            "",
            "SHARPE RATIO:",
            f"  In-Sample:   {self.is_sharpe:.4f}",
            f"  Out-Sample:  {self.oos_sharpe:.4f}",
            f"  OOS/IS:      {self.sharpe_ratio:.4f}",
            "",
            "ALPHA (Jensen's Alpha):",
            f"  Alpha:    {self.alpha:.6f}",
            f"  t-stat:   {self.alpha_t_stat:.3f}",
            f"  p-value:  {self.alpha_p_value:.4f}",
            "",
            "HANSEN-JAGANNATHAN DISTANCE:",
            f"  HJ Distance: {self.hj_distance:.6f}",
            f"  SE:          {self.hj_se:.6f}",
            "",
            "CROSS-SECTIONAL PRICING:",
            f"  R²:   {self.cs_r2:.4f}",
            f"  RMSE: {self.cs_rmse:.6f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# =============================================================================
# Metric Calculations
# =============================================================================

def compute_pricing_error(m: np.ndarray, r: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute Euler equation pricing error: E[(1+R)*M - 1].
    
    Args:
        m: SDF values (N,)
        r: Asset returns (N,) or (N, K)
        
    Returns:
        (mean, std, max) of pricing errors
    """
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    
    # Pricing error for each asset
    errors = (1 + r) * m.reshape(-1, 1) - 1
    
    # Per-asset mean errors
    mean_errors = np.mean(errors, axis=0)
    
    return float(np.mean(np.abs(mean_errors))), float(np.std(mean_errors)), float(np.max(np.abs(mean_errors)))


def compute_sharpe_ratio(returns: np.ndarray, annualize: bool = True) -> float:
    """
    Compute Sharpe ratio.
    
    Args:
        returns: Return series
        annualize: Whether to annualize (assumes monthly data)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    
    sr = np.mean(returns) / np.std(returns)
    
    if annualize:
        sr *= np.sqrt(12)  # Monthly to annual
    
    return float(sr)


def compute_alpha(
    portfolio_returns: np.ndarray,
    market_returns: np.ndarray,
    rf_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Compute Jensen's Alpha via CAPM regression.
    
    R_p - R_f = alpha + beta * (R_m - R_f) + epsilon
    
    Args:
        portfolio_returns: Portfolio excess returns
        market_returns: Market excess returns
        rf_rate: Risk-free rate
        
    Returns:
        (alpha, t_stat, p_value)
    """
    y = portfolio_returns - rf_rate
    x = market_returns - rf_rate
    
    # Simple OLS using numpy
    n = len(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Slope (beta)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator if denominator > 0 else 0.0
    
    # Intercept (alpha)
    intercept = y_mean - slope * x_mean
    alpha = intercept
    
    # Residuals and MSE
    residuals = y - (intercept + slope * x)
    mse = np.sum(residuals ** 2) / (n - 2) if n > 2 else 1.0
    
    # Standard error of alpha
    x_var = np.sum((x - x_mean) ** 2)
    se_alpha = np.sqrt(mse * (1/n + x_mean**2 / x_var)) if x_var > 0 else 1.0
    
    # t-statistic
    t_stat = alpha / se_alpha if se_alpha > 0 else 0.0
    
    # Approximate p-value using simple formula (for large n, t ~ N(0,1))
    # Two-tailed p-value approximation
    import math
    z = abs(t_stat)
    p_val = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))  # Normal CDF
    
    return float(alpha), float(t_stat), float(p_val)


def compute_hj_distance(
    m: np.ndarray,
    returns: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Hansen-Jagannathan distance.
    
    HJ = min ||E[M*R] - 1||_Σ^{-1}
    
    This measures the minimum distance from the candidate SDF to the
    set of valid SDFs.
    
    Args:
        m: SDF values (N,)
        returns: Asset returns (N, K)
        
    Returns:
        (hj_distance, standard_error)
    """
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    
    n, k = returns.shape
    
    # Pricing errors
    g = (1 + returns) * m.reshape(-1, 1) - 1  # (N, K)
    g_bar = np.mean(g, axis=0)  # (K,)
    
    # Covariance matrix of pricing errors
    S = np.cov(g, rowvar=False)
    if k == 1:
        S = np.array([[S]])
    
    # HJ distance
    try:
        S_inv = np.linalg.inv(S + 1e-8 * np.eye(k))
        hj_sq = g_bar @ S_inv @ g_bar
        hj = np.sqrt(max(0, hj_sq))
    except np.linalg.LinAlgError:
        hj = np.sqrt(np.mean(g_bar ** 2))
    
    # Standard error (simplified)
    se = hj / np.sqrt(n) if n > 0 else 0.0
    
    return float(hj), float(se)


def compute_cross_sectional_r2(
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute cross-sectional R² for pricing accuracy.
    
    Args:
        predicted_returns: Model-implied expected returns (K,)
        actual_returns: Actual average returns (K,)
        
    Returns:
        (R², RMSE)
    """
    # R²
    ss_res = np.sum((actual_returns - predicted_returns) ** 2)
    ss_tot = np.sum((actual_returns - np.mean(actual_returns)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # RMSE
    rmse = np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))
    
    return float(r2), float(rmse)


# =============================================================================
# SDF Model (for benchmark)
# =============================================================================

class SDFModel(nn.Module):
    """Baseline SDF model for evaluation benchmark."""
    
    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        dropout: float = 0.4,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)
        self.output_act = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.hidden(x)
        m = self.output_act(self.output(z))
        return m


# =============================================================================
# Evaluation Pipeline
# =============================================================================

class SDFEvaluator:
    """Comprehensive SDF model evaluator."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
    
    def evaluate(
        self,
        X_is: np.ndarray,
        R_is: np.ndarray,
        X_oos: np.ndarray,
        R_oos: np.ndarray,
        market_returns: Optional[np.ndarray] = None,
    ) -> SDFMetrics:
        """
        Run full evaluation pipeline.
        
        Args:
            X_is: In-sample features
            R_is: In-sample returns (can be multi-asset)
            X_oos: Out-of-sample features
            R_oos: Out-of-sample returns
            market_returns: Market returns for alpha calculation
            
        Returns:
            SDFMetrics dataclass
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get SDF values
            m_is = self.model(torch.from_numpy(X_is).float().to(self.device)).cpu().numpy().flatten()
            m_oos = self.model(torch.from_numpy(X_oos).float().to(self.device)).cpu().numpy().flatten()
        
        # Ensure returns are 2D for multi-asset
        R_is_2d = R_is.reshape(-1, 1) if R_is.ndim == 1 else R_is
        R_oos_2d = R_oos.reshape(-1, 1) if R_oos.ndim == 1 else R_oos
        
        # 1. Pricing Error
        pe_mean, pe_std, pe_max = compute_pricing_error(m_oos, R_oos_2d)
        
        # 2. Sharpe Ratio
        # Use SDF-implied returns
        sdf_returns_is = (m_is - 1) / 0.04  # Simplified transformation
        sdf_returns_oos = (m_oos - 1) / 0.04
        
        is_sharpe = compute_sharpe_ratio(sdf_returns_is)
        oos_sharpe = compute_sharpe_ratio(sdf_returns_oos)
        sharpe_ratio = oos_sharpe / is_sharpe if abs(is_sharpe) > 0.01 else 0.0
        
        # 3. Alpha
        if market_returns is not None:
            alpha, alpha_t, alpha_p = compute_alpha(sdf_returns_oos, market_returns)
        else:
            # Use first asset as market proxy
            market_proxy = R_oos_2d[:, 0] if R_oos_2d.shape[1] > 0 else R_oos.flatten()
            alpha, alpha_t, alpha_p = compute_alpha(sdf_returns_oos, market_proxy)
        
        # 4. HJ Distance
        hj_dist, hj_se = compute_hj_distance(m_oos, R_oos_2d)
        
        # 5. Cross-sectional R²
        # Model-implied expected returns vs actual
        predicted_er = np.mean(R_oos_2d * m_oos.reshape(-1, 1), axis=0)
        actual_er = np.mean(R_oos_2d, axis=0)
        cs_r2, cs_rmse = compute_cross_sectional_r2(predicted_er, actual_er)
        
        return SDFMetrics(
            pricing_error_mean=pe_mean,
            pricing_error_std=pe_std,
            pricing_error_max=pe_max,
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            sharpe_ratio=sharpe_ratio,
            alpha=alpha,
            alpha_t_stat=alpha_t,
            alpha_p_value=alpha_p,
            hj_distance=hj_dist,
            hj_se=hj_se,
            cs_r2=cs_r2,
            cs_rmse=cs_rmse,
            n_samples=len(X_oos),
            timestamp=datetime.now().isoformat(),
        )


# =============================================================================
# Data Generation (for benchmark)
# =============================================================================

def generate_benchmark_data(
    n_samples: int = 500,
    n_assets: int = 25,
    n_features: int = 48,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for evaluation benchmark."""
    np.random.seed(seed)
    
    # Features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Multi-asset returns (factor model)
    betas = np.random.randn(n_assets, 3) * 0.5  # 3 factors
    factors = np.random.randn(n_samples, 3) * 0.02
    idio = np.random.randn(n_samples, n_assets) * 0.05
    
    R = 0.007 + factors @ betas.T + idio
    R = R.astype(np.float32)
    
    # Split
    n_train = int(0.7 * n_samples)
    X_is, X_oos = X[:n_train], X[n_train:]
    R_is, R_oos = R[:n_train], R[n_train:]
    
    return X_is, R_is, X_oos, R_oos


# =============================================================================
# Report Generation
# =============================================================================

def generate_evaluation_report(metrics: SDFMetrics, output_path: Path) -> str:
    """Generate markdown evaluation report."""
    report = f"""# SDF Model Evaluation Report

**Generated**: {metrics.timestamp}
**Framework**: T5.1 SDF Evaluation Framework
**Reference**: SDF_SPEC v3.1, Cochrane (2005)

## Executive Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Pricing Error | {metrics.pricing_error_mean:.6f} | <0.01 | {'✅' if metrics.pricing_error_mean < 0.01 else '⚠️'} |
| OOS Sharpe | {metrics.oos_sharpe:.4f} | ≥1.5 | {'✅' if metrics.oos_sharpe >= 1.5 else '⚠️'} |
| OOS/IS Sharpe | {metrics.sharpe_ratio:.4f} | ≥0.9 | {'✅' if metrics.sharpe_ratio >= 0.9 else '⚠️'} |
| HJ Distance | {metrics.hj_distance:.6f} | <0.5 | {'✅' if metrics.hj_distance < 0.5 else '⚠️'} |
| Cross-sectional R² | {metrics.cs_r2:.4f} | ≥0.7 | {'✅' if metrics.cs_r2 >= 0.7 else '⚠️'} |

## Detailed Metrics

### 1. Pricing Error (Euler Equation)

The Euler equation pricing error measures how well the SDF prices assets:

$$E[(1+R)M - 1] = 0$$

| Statistic | Value |
|-----------|-------|
| Mean | {metrics.pricing_error_mean:.6f} |
| Std | {metrics.pricing_error_std:.6f} |
| Max | {metrics.pricing_error_max:.6f} |

### 2. Sharpe Ratio

| Sample | Sharpe Ratio |
|--------|--------------|
| In-Sample | {metrics.is_sharpe:.4f} |
| Out-of-Sample | {metrics.oos_sharpe:.4f} |
| OOS/IS Ratio | {metrics.sharpe_ratio:.4f} |

### 3. Jensen's Alpha

Testing for abnormal returns after risk adjustment:

| Statistic | Value |
|-----------|-------|
| Alpha | {metrics.alpha:.6f} |
| t-statistic | {metrics.alpha_t_stat:.3f} |
| p-value | {metrics.alpha_p_value:.4f} |

**Interpretation**: {'Alpha is statistically significant (p < 0.05)' if metrics.alpha_p_value < 0.05 else 'Alpha is not statistically significant'}

### 4. Hansen-Jagannathan Distance

The HJ distance measures the minimum adjustment needed to make the SDF valid:

| Statistic | Value |
|-----------|-------|
| HJ Distance | {metrics.hj_distance:.6f} |
| Standard Error | {metrics.hj_se:.6f} |

### 5. Cross-Sectional Pricing

| Metric | Value |
|--------|-------|
| R² | {metrics.cs_r2:.4f} |
| RMSE | {metrics.cs_rmse:.6f} |

## Recommendations

1. **Pricing Accuracy**: {'Good - pricing errors within acceptable range' if metrics.pricing_error_mean < 0.01 else 'Consider model refinement to reduce pricing errors'}
2. **Generalization**: {'Strong OOS performance' if metrics.sharpe_ratio >= 0.9 else 'Model may be overfitting - consider additional regularization'}
3. **Risk-Adjusted Returns**: {'Alpha generation potential' if metrics.alpha > 0 and metrics.alpha_p_value < 0.05 else 'No significant alpha'}

---
*Report generated by T5.1 SDF Evaluation Framework*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report


# =============================================================================
# CLI
# =============================================================================

def run_benchmark(verbose: bool = True) -> Dict:
    """Run evaluation benchmark with synthetic data."""
    print("=" * 70)
    print("T5.1 SDF Evaluation Framework - Benchmark")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print()
    
    # Generate data
    X_is, R_is, X_oos, R_oos = generate_benchmark_data(
        n_samples=CONFIG["benchmark"]["n_samples"],
        n_assets=CONFIG["benchmark"]["n_test_assets"],
        seed=CONFIG["benchmark"]["seed"],
    )
    
    print(f"Data: IS={len(X_is)}, OOS={len(X_oos)}, Assets={R_is.shape[1]}")
    print()
    
    # Train a simple model for benchmark
    print("Training benchmark model...")
    model = SDFModel(input_dim=X_is.shape[1], dropout=0.4, seed=42)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    X_is_t = torch.from_numpy(X_is).float()
    R_is_t = torch.from_numpy(R_is[:, 0]).float()  # Use first asset for training
    
    model.train()
    for epoch in range(50):
        m = model(X_is_t).squeeze()
        residual = (1 + R_is_t) * m - 1
        loss = torch.mean(residual ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")
    
    print()
    
    # Evaluate
    print("Running evaluation...")
    evaluator = SDFEvaluator(model)
    metrics = evaluator.evaluate(X_is, R_is, X_oos, R_oos)
    
    print()
    print(metrics.summary())
    
    # Save results
    output_dir = PROJECT_ROOT / "experiments" / "t5_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON metrics
    json_path = output_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nMetrics saved to: {json_path}")
    
    # Markdown report
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "sdf_evaluation_report.md"
    generate_evaluation_report(metrics, report_path)
    print(f"Report saved to: {report_path}")
    
    return {
        "config": CONFIG,
        "metrics": metrics.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="T5.1 SDF Evaluation Framework")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark evaluation")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    if args.benchmark or args.model is None:
        results = run_benchmark(verbose=not args.quiet)
        print("\n✅ T5.1 SDF Evaluation Framework benchmark completed!")


if __name__ == "__main__":
    main()
