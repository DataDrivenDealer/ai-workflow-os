"""
T4.5 Regularization Grid Search (T4-STR-4)

Purpose: Find optimal L2 weight decay and dropout configuration to reduce overfitting.

Expected Impact: 
- Reduce OOS/IS gap (T4-OBJ-3: target ≥0.9)
- Improve model generalization

Grid:
- L2 weight decay: [1e-5, 1e-4, 1e-3]
- Dropout: [0.1, 0.2, 0.3, 0.4, 0.5]
- Total: 3 × 5 = 15 configurations

Usage:
    python scripts/t4_regularization.py --benchmark
    
Output:
    experiments/t4_regularization/results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

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
    "name": "T4.5 Regularization Grid Search",
    "timestamp": datetime.now().isoformat(),
    "training": {
        "max_epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": 42,
        "device": "cpu",
    },
    "model": {
        "input_dim": 48,
        "hidden_dim": 64,
        "num_hidden_layers": 3,
    },
    "early_stopping": {
        "patience": 10,
        "min_delta": 0.0001,
    },
    "grid_search": {
        "l2_values": [1e-5, 1e-4, 1e-3],
        "dropout_values": [0.1, 0.2, 0.3, 0.4, 0.5],
    },
    "cv": {
        "n_folds": 3,  # Time-series CV
    },
}

# =============================================================================
# Model with Dropout
# =============================================================================

class RegularizedSDF(nn.Module):
    """SDF model with dropout regularization."""
    
    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        
        layers = []
        in_dim = input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)
        self.output_act = nn.Softplus()
        self.dropout_rate = dropout
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.hidden(x)
        m = self.output_act(self.output(z))
        return m, z


def compute_pricing_error(m: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Euler equation pricing error."""
    if r.dim() == 1:
        r = r.unsqueeze(1)
    residual = (1.0 + r) * m - 1.0
    return torch.mean(residual ** 2)


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None
    
    def __call__(self, epoch: int, value: float, model: nn.Module) -> bool:
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            self.best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False
    
    def load_best(self, model: nn.Module):
        if self.best_state_dict:
            model.load_state_dict(self.best_state_dict)
        return model


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, CONFIG["model"]["input_dim"]).astype(np.float32)
    returns = (0.007 + 0.04 * np.random.randn(n_samples)).astype(np.float32)
    return X, returns


def time_series_cv_split(X: np.ndarray, R: np.ndarray, n_folds: int = 3) -> List[Tuple]:
    """
    Time-series cross-validation split.
    Each fold uses all prior data as training.
    """
    n = len(X)
    fold_size = n // (n_folds + 1)
    
    splits = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        val_start = train_end
        val_end = fold_size * (i + 2)
        
        X_train, R_train = X[:train_end], R[:train_end]
        X_val, R_val = X[val_start:val_end], R[val_start:val_end]
        
        splits.append((X_train, R_train, X_val, R_val))
    
    return splits


# =============================================================================
# Training Function
# =============================================================================

def train_single_config(
    X_train: np.ndarray,
    R_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    l2_weight: float,
    dropout: float,
) -> Dict:
    """Train with single configuration."""
    device = torch.device(CONFIG["training"]["device"])
    seed = CONFIG["training"]["seed"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = RegularizedSDF(
        input_dim=CONFIG["model"]["input_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        dropout=dropout,
        seed=seed,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=l2_weight,  # L2 regularization
    )
    
    early_stopping = EarlyStopping(
        patience=CONFIG["early_stopping"]["patience"],
        min_delta=CONFIG["early_stopping"]["min_delta"],
    )
    
    X_train_t = torch.from_numpy(X_train).to(device)
    R_train_t = torch.from_numpy(R_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    
    batch_size = CONFIG["training"]["batch_size"]
    
    for epoch in range(CONFIG["training"]["max_epochs"]):
        model.train()
        batch_losses = []
        
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            R_batch = R_train_t[batch_idx]
            
            m, _ = model(X_batch)
            loss = compute_pricing_error(m, R_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        
        model.eval()
        with torch.no_grad():
            m_val, _ = model(X_val_t)
            val_loss = compute_pricing_error(m_val, R_val_t).item()
        
        if early_stopping(epoch, val_loss, model):
            break
    
    # Load best model
    model = early_stopping.load_best(model)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        m_train, _ = model(X_train_t)
        final_train_loss = compute_pricing_error(m_train, R_train_t).item()
        
        m_val, _ = model(X_val_t)
        final_val_loss = compute_pricing_error(m_val, R_val_t).item()
    
    return {
        "l2_weight": l2_weight,
        "dropout": dropout,
        "best_val_loss": early_stopping.best_value,
        "best_epoch": early_stopping.best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "oos_is_ratio": final_val_loss / final_train_loss if final_train_loss > 0 else float("inf"),
    }


def run_cv_for_config(
    cv_splits: List[Tuple],
    l2_weight: float,
    dropout: float,
) -> Dict:
    """Run cross-validation for a single configuration."""
    fold_results = []
    
    for fold_idx, (X_train, R_train, X_val, R_val) in enumerate(cv_splits):
        result = train_single_config(X_train, R_train, X_val, R_val, l2_weight, dropout)
        fold_results.append(result)
    
    # Aggregate across folds
    avg_val_loss = np.mean([r["final_val_loss"] for r in fold_results])
    avg_train_loss = np.mean([r["final_train_loss"] for r in fold_results])
    avg_oos_is = np.mean([r["oos_is_ratio"] for r in fold_results])
    std_val_loss = np.std([r["final_val_loss"] for r in fold_results])
    
    return {
        "l2_weight": l2_weight,
        "dropout": dropout,
        "avg_val_loss": avg_val_loss,
        "std_val_loss": std_val_loss,
        "avg_train_loss": avg_train_loss,
        "avg_oos_is_ratio": avg_oos_is,
        "fold_results": fold_results,
    }


# =============================================================================
# Grid Search
# =============================================================================

def run_grid_search(cv_splits: List[Tuple], verbose: bool = True) -> List[Dict]:
    """Run full grid search."""
    l2_values = CONFIG["grid_search"]["l2_values"]
    dropout_values = CONFIG["grid_search"]["dropout_values"]
    
    total_configs = len(l2_values) * len(dropout_values)
    results = []
    
    print(f"Running {total_configs} configurations ({len(l2_values)} L2 × {len(dropout_values)} Dropout)")
    print("-" * 70)
    
    for idx, (l2, dropout) in enumerate(product(l2_values, dropout_values)):
        if verbose:
            print(f"[{idx+1:2d}/{total_configs}] L2={l2:.0e}, Dropout={dropout:.1f}...", end=" ")
        
        start_time = time.time()
        result = run_cv_for_config(cv_splits, l2, dropout)
        elapsed = time.time() - start_time
        
        result["train_time_sec"] = elapsed
        results.append(result)
        
        if verbose:
            print(f"Val Loss: {result['avg_val_loss']:.6f} ± {result['std_val_loss']:.6f}, OOS/IS: {result['avg_oos_is_ratio']:.3f}")
    
    return results


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(verbose: bool = True) -> Dict:
    """Run regularization grid search benchmark."""
    print("=" * 70)
    print("T4.5 Regularization Grid Search (T4-STR-4)")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print(f"L2 values: {CONFIG['grid_search']['l2_values']}")
    print(f"Dropout values: {CONFIG['grid_search']['dropout_values']}")
    print(f"CV Folds: {CONFIG['cv']['n_folds']}")
    print()
    
    # Generate data
    X, returns = generate_data(n_samples=500)
    
    # Create time-series CV splits
    cv_splits = time_series_cv_split(X, returns, n_folds=CONFIG["cv"]["n_folds"])
    
    print(f"Data: {len(X)} samples")
    print(f"CV Splits: {len(cv_splits)} folds")
    for i, (X_tr, _, X_va, _) in enumerate(cv_splits):
        print(f"  Fold {i+1}: Train={len(X_tr)}, Val={len(X_va)}")
    print()
    
    # Run grid search
    start_time = time.time()
    results = run_grid_search(cv_splits, verbose=verbose)
    total_time = time.time() - start_time
    
    # Find best configuration
    best_result = min(results, key=lambda r: r["avg_val_loss"])
    
    # Baseline (no regularization)
    baseline = next((r for r in results if r["l2_weight"] == 1e-5 and r["dropout"] == 0.1), results[0])
    
    print()
    print("=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    
    # Sort by validation loss
    sorted_results = sorted(results, key=lambda r: r["avg_val_loss"])
    
    print(f"{'Rank':<5} {'L2':<10} {'Dropout':<10} {'Val Loss':<15} {'OOS/IS':<10}")
    print("-" * 50)
    for rank, r in enumerate(sorted_results[:5], 1):
        print(f"{rank:<5} {r['l2_weight']:<10.0e} {r['dropout']:<10.1f} {r['avg_val_loss']:<15.6f} {r['avg_oos_is_ratio']:<10.3f}")
    print("...")
    for rank, r in enumerate(sorted_results[-2:], len(sorted_results)-1):
        print(f"{rank:<5} {r['l2_weight']:<10.0e} {r['dropout']:<10.1f} {r['avg_val_loss']:<15.6f} {r['avg_oos_is_ratio']:<10.3f}")
    
    print()
    print("=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"L2 Weight:     {best_result['l2_weight']:.0e}")
    print(f"Dropout:       {best_result['dropout']:.1f}")
    print(f"Avg Val Loss:  {best_result['avg_val_loss']:.6f}")
    print(f"Avg OOS/IS:    {best_result['avg_oos_is_ratio']:.3f}")
    
    # Calculate improvement vs baseline
    baseline_oos_is = baseline["avg_oos_is_ratio"]
    best_oos_is = best_result["avg_oos_is_ratio"]
    oos_is_improvement = (baseline_oos_is - best_oos_is) / baseline_oos_is * 100 if baseline_oos_is > 0 else 0
    
    print()
    print("COMPARISON VS BASELINE (L2=1e-5, Dropout=0.1):")
    print(f"  Baseline OOS/IS:   {baseline_oos_is:.3f}")
    print(f"  Best OOS/IS:       {best_oos_is:.3f}")
    print(f"  Improvement:       {oos_is_improvement:.1f}%")
    
    # Assessment
    target_improvement = 5  # ≥5% OOS/IS gap reduction
    
    print()
    if oos_is_improvement >= target_improvement:
        print(f"✅ OOS/IS gap reduced by {oos_is_improvement:.1f}% (target: ≥{target_improvement}%)")
        assessment = "TARGET_MET"
    elif oos_is_improvement > 0:
        print(f"⚠️ OOS/IS gap reduced by {oos_is_improvement:.1f}% (target: ≥{target_improvement}%)")
        assessment = "PARTIAL_IMPROVEMENT"
    else:
        print(f"⚠️ No OOS/IS improvement observed")
        assessment = "NO_IMPROVEMENT"
    
    output = {
        "config": CONFIG,
        "total_configurations": len(results),
        "total_time_sec": total_time,
        "all_results": [
            {k: v for k, v in r.items() if k != "fold_results"}
            for r in results
        ],
        "best_configuration": {
            "l2_weight": best_result["l2_weight"],
            "dropout": best_result["dropout"],
            "avg_val_loss": best_result["avg_val_loss"],
            "avg_oos_is_ratio": best_result["avg_oos_is_ratio"],
        },
        "comparison": {
            "baseline_oos_is": baseline_oos_is,
            "best_oos_is": best_oos_is,
            "improvement_pct": oos_is_improvement,
            "assessment": assessment,
        },
        "recommendation": {
            "l2_weight": best_result["l2_weight"],
            "dropout": best_result["dropout"],
            "rationale": f"Best validation loss with {oos_is_improvement:.1f}% OOS/IS improvement",
        },
    }
    
    return output


def save_results(results: Dict, output_path: str = None):
    """Save benchmark results."""
    if output_path is None:
        output_dir = PROJECT_ROOT / "experiments" / "t4_regularization"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "results.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="T4.5 Regularization Grid Search")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = run_benchmark(verbose=not args.quiet)
    save_results(results)
    print("\n✅ T4.5 Regularization Grid Search completed!")


if __name__ == "__main__":
    main()
