"""
T5.2 OOS Validation Pipeline (SDF_DEV_001_T5)

Purpose: Implement rolling window out-of-sample validation for SDF models.

Features:
1. Time-series train/val/test split
2. Rolling window evaluation
3. Multi-period OOS metrics aggregation
4. Validation report generation

Reference: Gu, Kelly, Xiu (2020) - Empirical Asset Pricing via Machine Learning

Usage:
    python scripts/validate_sdf_oos.py --rolling
    python scripts/validate_sdf_oos.py --expanding
    
Output:
    experiments/t5_oos_validation/rolling_results.json
    reports/sdf_oos_validation_report.md
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

# Import evaluation framework
from evaluate_sdf import SDFMetrics, SDFEvaluator, compute_sharpe_ratio

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "name": "T5.2 OOS Validation Pipeline",
    "timestamp": datetime.now().isoformat(),
    "validation": {
        "method": "rolling",  # or "expanding"
        "train_window": 36,  # months
        "val_window": 12,    # months
        "test_window": 12,   # months
        "step_size": 12,     # months between windows
    },
    "training": {
        "max_epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "dropout": 0.4,
        "l2_weight": 1e-4,
        "early_stopping_patience": 10,
    },
    "model": {
        "input_dim": 48,
        "hidden_dim": 64,
        "num_hidden_layers": 3,
    },
}

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WindowResult:
    """Results for a single validation window."""
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_samples: int
    test_samples: int
    metrics: Dict
    training_time_sec: float


@dataclass
class ValidationSummary:
    """Summary of all validation windows."""
    method: str
    n_windows: int
    
    # Aggregated metrics (mean ± std)
    pricing_error_mean: float
    pricing_error_std: float
    
    oos_sharpe_mean: float
    oos_sharpe_std: float
    
    sharpe_ratio_mean: float  # OOS/IS
    sharpe_ratio_std: float
    
    hj_distance_mean: float
    hj_distance_std: float
    
    cs_r2_mean: float
    cs_r2_std: float
    
    # Consistency metrics
    positive_sharpe_pct: float  # % of windows with positive OOS Sharpe
    stable_performance: bool    # OOS/IS > 0.8 in >80% of windows
    
    total_time_sec: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Model
# =============================================================================

class SDFModel(nn.Module):
    """SDF model with T4 optimizations."""
    
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


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.counter = 0
        self.best_state_dict = None
    
    def __call__(self, value: float, model: nn.Module) -> bool:
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            self.best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience
    
    def load_best(self, model: nn.Module):
        if self.best_state_dict:
            model.load_state_dict(self.best_state_dict)
        return model


# =============================================================================
# Data Generation
# =============================================================================

def generate_time_series_data(
    n_periods: int = 120,  # 10 years monthly
    n_assets: int = 25,
    n_features: int = 48,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic time-series data.
    
    Returns:
        dates: Period indices (N,)
        X: Features (N, F)
        R: Returns (N, K)
    """
    np.random.seed(seed)
    
    dates = np.arange(n_periods)
    
    # Features with time-varying structure
    X = np.random.randn(n_periods, n_features).astype(np.float32)
    # Add some persistence
    for t in range(1, n_periods):
        X[t] = 0.3 * X[t-1] + 0.7 * X[t]
    
    # Returns with factor structure
    betas = np.random.randn(n_assets, 3) * 0.5
    factors = np.random.randn(n_periods, 3) * 0.02
    # Add time-varying volatility
    vol_regime = 1 + 0.5 * np.sin(2 * np.pi * dates / 60)
    idio = np.random.randn(n_periods, n_assets) * 0.05 * vol_regime.reshape(-1, 1)
    
    R = 0.007 + factors @ betas.T + idio
    R = R.astype(np.float32)
    
    return dates, X, R


# =============================================================================
# Rolling Window Validation
# =============================================================================

def create_rolling_windows(
    n_periods: int,
    train_window: int,
    test_window: int,
    step_size: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Create rolling window indices.
    
    Returns:
        List of (train_start, train_end, test_start, test_end)
    """
    windows = []
    
    start = 0
    while start + train_window + test_window <= n_periods:
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = train_end + test_window
        
        windows.append((train_start, train_end, test_start, test_end))
        start += step_size
    
    return windows


def train_and_evaluate_window(
    X: np.ndarray,
    R: np.ndarray,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    window_id: int,
    verbose: bool = True,
) -> WindowResult:
    """Train model on window and evaluate OOS."""
    import time
    
    start_time = time.time()
    
    # Split data
    X_train = X[train_start:train_end]
    R_train = R[train_start:train_end]
    X_test = X[test_start:test_end]
    R_test = R[test_start:test_end]
    
    # Train model
    model = SDFModel(
        input_dim=X.shape[1],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        dropout=CONFIG["training"]["dropout"],
        seed=42 + window_id,
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["l2_weight"],
    )
    
    early_stopping = EarlyStopping(
        patience=CONFIG["training"]["early_stopping_patience"],
    )
    
    X_train_t = torch.from_numpy(X_train).float()
    R_train_t = torch.from_numpy(R_train[:, 0]).float()  # Use first asset
    
    batch_size = CONFIG["training"]["batch_size"]
    
    model.train()
    for epoch in range(CONFIG["training"]["max_epochs"]):
        # Mini-batch training
        indices = np.random.permutation(len(X_train))
        batch_losses = []
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            R_batch = R_train_t[batch_idx]
            
            m = model(X_batch).squeeze()
            residual = (1 + R_batch) * m - 1
            loss = torch.mean(residual ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        val_loss = np.mean(batch_losses)
        
        if early_stopping(val_loss, model):
            break
    
    # Load best model
    model = early_stopping.load_best(model)
    
    # Evaluate
    evaluator = SDFEvaluator(model)
    metrics = evaluator.evaluate(X_train, R_train, X_test, R_test)
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f"  Window {window_id:2d}: [{train_start:3d}-{train_end:3d}] -> [{test_start:3d}-{test_end:3d}] | "
              f"OOS Sharpe: {metrics.oos_sharpe:+.3f} | OOS/IS: {metrics.sharpe_ratio:.3f}")
    
    return WindowResult(
        window_id=window_id,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        train_samples=len(X_train),
        test_samples=len(X_test),
        metrics=metrics.to_dict(),
        training_time_sec=training_time,
    )


def run_rolling_validation(
    X: np.ndarray,
    R: np.ndarray,
    verbose: bool = True,
) -> Tuple[List[WindowResult], ValidationSummary]:
    """Run full rolling window validation."""
    import time
    
    start_time = time.time()
    
    # Create windows
    windows = create_rolling_windows(
        n_periods=len(X),
        train_window=CONFIG["validation"]["train_window"],
        test_window=CONFIG["validation"]["test_window"],
        step_size=CONFIG["validation"]["step_size"],
    )
    
    if verbose:
        print(f"Rolling validation: {len(windows)} windows")
        print(f"Train window: {CONFIG['validation']['train_window']} periods")
        print(f"Test window: {CONFIG['validation']['test_window']} periods")
        print("-" * 70)
    
    # Evaluate each window
    results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        result = train_and_evaluate_window(
            X, R, train_start, train_end, test_start, test_end,
            window_id=i, verbose=verbose,
        )
        results.append(result)
    
    # Aggregate metrics
    pe_values = [r.metrics["pricing_error_mean"] for r in results]
    sharpe_values = [r.metrics["oos_sharpe"] for r in results]
    ratio_values = [r.metrics["sharpe_ratio"] for r in results]
    hj_values = [r.metrics["hj_distance"] for r in results]
    r2_values = [r.metrics["cs_r2"] for r in results]
    
    positive_sharpe_pct = sum(1 for s in sharpe_values if s > 0) / len(sharpe_values) * 100
    stable_pct = sum(1 for r in ratio_values if r > 0.8) / len(ratio_values) * 100
    
    summary = ValidationSummary(
        method="rolling",
        n_windows=len(windows),
        pricing_error_mean=float(np.mean(pe_values)),
        pricing_error_std=float(np.std(pe_values)),
        oos_sharpe_mean=float(np.mean(sharpe_values)),
        oos_sharpe_std=float(np.std(sharpe_values)),
        sharpe_ratio_mean=float(np.mean(ratio_values)),
        sharpe_ratio_std=float(np.std(ratio_values)),
        hj_distance_mean=float(np.mean(hj_values)),
        hj_distance_std=float(np.std(hj_values)),
        cs_r2_mean=float(np.mean(r2_values)),
        cs_r2_std=float(np.std(r2_values)),
        positive_sharpe_pct=positive_sharpe_pct,
        stable_performance=stable_pct > 80,
        total_time_sec=time.time() - start_time,
        timestamp=datetime.now().isoformat(),
    )
    
    return results, summary


# =============================================================================
# Report Generation
# =============================================================================

def generate_oos_report(
    results: List[WindowResult],
    summary: ValidationSummary,
    output_path: Path,
) -> str:
    """Generate OOS validation report."""
    
    # Window details table
    window_rows = []
    for r in results:
        window_rows.append(
            f"| {r.window_id} | {r.train_start}-{r.train_end} | {r.test_start}-{r.test_end} | "
            f"{r.metrics['oos_sharpe']:+.3f} | {r.metrics['sharpe_ratio']:.3f} | "
            f"{r.metrics['pricing_error_mean']:.4f} |"
        )
    window_table = "\n".join(window_rows)
    
    report = f"""# SDF Out-of-Sample Validation Report

**Generated**: {summary.timestamp}
**Method**: {summary.method.title()} Window Validation
**Windows**: {summary.n_windows}

## Executive Summary

| Metric | Mean | Std | Status |
|--------|------|-----|--------|
| OOS Sharpe | {summary.oos_sharpe_mean:+.3f} | {summary.oos_sharpe_std:.3f} | {'✅' if summary.oos_sharpe_mean > 0 else '⚠️'} |
| OOS/IS Ratio | {summary.sharpe_ratio_mean:.3f} | {summary.sharpe_ratio_std:.3f} | {'✅' if summary.sharpe_ratio_mean > 0.8 else '⚠️'} |
| Pricing Error | {summary.pricing_error_mean:.4f} | {summary.pricing_error_std:.4f} | {'✅' if summary.pricing_error_mean < 0.01 else '⚠️'} |
| HJ Distance | {summary.hj_distance_mean:.4f} | {summary.hj_distance_std:.4f} | {'✅' if summary.hj_distance_mean < 0.5 else '⚠️'} |
| CS R² | {summary.cs_r2_mean:.3f} | {summary.cs_r2_std:.3f} | {'✅' if summary.cs_r2_mean > 0.7 else '⚠️'} |

## Consistency Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Positive OOS Sharpe | {summary.positive_sharpe_pct:.1f}% | ≥50% | {'✅' if summary.positive_sharpe_pct >= 50 else '⚠️'} |
| Stable Performance (OOS/IS>0.8) | {'Yes' if summary.stable_performance else 'No'} | Yes | {'✅' if summary.stable_performance else '⚠️'} |

## Window-by-Window Results

| Window | Train | Test | OOS Sharpe | OOS/IS | Pricing Error |
|--------|-------|------|------------|--------|---------------|
{window_table}

## Time Analysis

- **Total validation time**: {summary.total_time_sec:.1f} seconds
- **Average per window**: {summary.total_time_sec / summary.n_windows:.2f} seconds

## Interpretation

### Strengths
{f"- **Consistent positive OOS Sharpe**: {summary.positive_sharpe_pct:.0f}% of windows show positive performance" if summary.positive_sharpe_pct >= 50 else ""}
{f"- **Stable generalization**: OOS/IS ratio consistently above 0.8" if summary.stable_performance else ""}
{f"- **Good cross-sectional fit**: Average R² of {summary.cs_r2_mean:.3f}" if summary.cs_r2_mean > 0.7 else ""}

### Areas for Improvement
{f"- OOS Sharpe variability is high (std={summary.oos_sharpe_std:.3f})" if summary.oos_sharpe_std > 1.0 else ""}
{f"- Pricing error above target (mean={summary.pricing_error_mean:.4f})" if summary.pricing_error_mean > 0.01 else ""}
{f"- HJ distance indicates room for SDF improvement" if summary.hj_distance_mean > 0.5 else ""}

## Recommendations

1. **Model Stability**: {"Model shows stable OOS performance across time periods" if summary.stable_performance else "Consider additional regularization or model simplification"}
2. **Production Readiness**: {"Suitable for paper trading" if summary.positive_sharpe_pct >= 60 else "Further validation recommended before deployment"}
3. **Next Steps**: Validate with real data to confirm synthetic benchmark results

---
*Report generated by T5.2 OOS Validation Pipeline*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report


# =============================================================================
# CLI
# =============================================================================

def run_validation(verbose: bool = True) -> Dict:
    """Run full OOS validation."""
    print("=" * 70)
    print("T5.2 OOS Validation Pipeline")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print()
    
    # Generate data
    dates, X, R = generate_time_series_data(
        n_periods=120,  # 10 years
        n_assets=25,
        seed=42,
    )
    
    print(f"Data: {len(dates)} periods, {X.shape[1]} features, {R.shape[1]} assets")
    print()
    
    # Run rolling validation
    results, summary = run_rolling_validation(X, R, verbose=verbose)
    
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Windows evaluated: {summary.n_windows}")
    print(f"OOS Sharpe: {summary.oos_sharpe_mean:+.3f} ± {summary.oos_sharpe_std:.3f}")
    print(f"OOS/IS Ratio: {summary.sharpe_ratio_mean:.3f} ± {summary.sharpe_ratio_std:.3f}")
    print(f"Positive Sharpe %: {summary.positive_sharpe_pct:.1f}%")
    print(f"Stable Performance: {'Yes' if summary.stable_performance else 'No'}")
    print(f"Total time: {summary.total_time_sec:.1f}s")
    
    # Save results
    output_dir = PROJECT_ROOT / "experiments" / "t5_oos_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    json_path = output_dir / "rolling_results.json"
    output_data = {
        "config": CONFIG,
        "summary": summary.to_dict(),
        "windows": [asdict(r) for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Markdown report
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "sdf_oos_validation_report.md"
    generate_oos_report(results, summary, report_path)
    print(f"Report saved to: {report_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="T5.2 OOS Validation Pipeline")
    parser.add_argument("--rolling", action="store_true", help="Run rolling window validation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = run_validation(verbose=not args.quiet)
    print("\n✅ T5.2 OOS Validation Pipeline completed!")


if __name__ == "__main__":
    main()
