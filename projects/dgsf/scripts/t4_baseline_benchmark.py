"""
T4.1 Baseline Benchmark - SDF Training Performance Measurement

Purpose: Measure current SDF training performance to establish baselines for T4 optimization.

Metrics Collected:
1. Wall-clock time per epoch
2. Total training time to convergence
3. GPU utilization (if applicable)
4. In-sample vs Out-of-sample Sharpe ratio
5. Loss convergence curve

Usage:
    python scripts/t4_baseline_benchmark.py
    
Output:
    experiments/t4_baseline/baseline_metrics.json
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add repo src to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent / "repo"
LEGACY_ROOT = SCRIPT_DIR.parent / "legacy" / "DGSF"

sys.path.insert(0, str(REPO_ROOT / "src"))

# =============================================================================
# Configuration
# =============================================================================

BASELINE_CONFIG = {
    "name": "T4 Baseline Benchmark",
    "timestamp": datetime.now().isoformat(),
    "training": {
        "num_epochs": 20,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "model": {
        "input_dim": 48,
        "hidden_dim": 64,
        "num_hidden_layers": 3,
        "activation": "tanh",
        "output_activation": "softplus",
    },
    "data": {
        "xstate_path": str(LEGACY_ROOT / "data" / "full" / "xstate_monthly_final.parquet"),
        "returns_path": str(LEGACY_ROOT / "data" / "final" / "monthly_returns.parquet"),
    },
}

# =============================================================================
# Minimal SDF Model (for baseline measurement)
# =============================================================================

class BaselineSDF(nn.Module):
    """Minimal SDF model for baseline benchmark."""
    
    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        activation: str = "tanh",
        output_activation: str = "softplus",
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        
        # Build hidden layers
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)
        
        # Output activation
        if output_activation == "softplus":
            self.output_act = nn.Softplus()
        elif output_activation == "exp":
            self.output_act = lambda x: torch.exp(x)
        else:
            self.output_act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            m: SDF values (batch_size, 1)
            z: Hidden state (batch_size, hidden_dim)
        """
        z = self.hidden(x)
        m = self.output_act(self.output(z))
        return m, z


def compute_pricing_error(m: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Euler equation pricing error: E[(1 + R) * m] = 1
    """
    if r.dim() == 1:
        r = r.unsqueeze(1)
    residual = (1.0 + r) * m - 1.0
    return torch.mean(residual ** 2)


def compute_sharpe_ratio(returns: np.ndarray) -> float:
    """Annualized Sharpe ratio (monthly data)."""
    if len(returns) == 0:
        return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(12)  # Annualize


# =============================================================================
# Synthetic Data Generator (fallback if real data unavailable)
# =============================================================================

def generate_synthetic_data(
    n_samples: int = 500,
    n_features: int = 48,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic X_state and returns for benchmark."""
    np.random.seed(seed)
    
    # Generate X_state with realistic distributions
    X = np.random.randn(n_samples, n_features)
    
    # Generate returns with realistic mean and volatility
    returns = 0.007 + 0.04 * np.random.randn(n_samples)  # ~7% annual return, 14% vol
    
    return X.astype(np.float32), returns.astype(np.float32)


def load_data() -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Attempt to load real data, fall back to synthetic if unavailable.
    
    DATA-001 Fix: Handle xstate nested x_state_vector column and panel returns.
    
    Data formats:
    - xstate_monthly_final.parquet: (n_months, 2) with columns ['date', 'x_state_vector']
      where x_state_vector is a list/array of ~48 features per month
    - monthly_returns.parquet: (n_obs, 3) panel data ['month_end', 'ts_code', 'ret']
      needs to be aggregated to monthly cross-sectional mean returns
    
    Returns:
        X: Features (n_samples, n_features) - monthly state features
        returns: Returns (n_samples,) - monthly cross-sectional mean returns
        is_real_data: True if real data loaded successfully
    """
    try:
        import pandas as pd
        
        xstate_path = BASELINE_CONFIG["data"]["xstate_path"]
        returns_path = BASELINE_CONFIG["data"]["returns_path"]
        
        if Path(xstate_path).exists() and Path(returns_path).exists():
            print(f"Loading real data from:")
            print(f"  X_state: {xstate_path}")
            print(f"  Returns: {returns_path}")
            
            # Load xstate - contains nested x_state_vector column
            X_df = pd.read_parquet(xstate_path)
            print(f"  xstate raw shape: {X_df.shape}, columns: {X_df.columns.tolist()}")
            
            # DATA-001 FIX: Expand the nested x_state_vector column
            if 'x_state_vector' in X_df.columns:
                # Extract the list/array column and stack into matrix
                x_vectors = X_df['x_state_vector'].tolist()
                X = np.array(x_vectors, dtype=np.float32)
                
                # Use date as index for alignment
                if 'date' in X_df.columns:
                    dates = X_df['date'].astype(str).values
                else:
                    dates = X_df.index.astype(str).values
                    
                print(f"  xstate expanded shape: {X.shape} ({X.shape[0]} months, {X.shape[1]} features)")
            else:
                # Fallback: assume standard columnar format
                X = X_df.values.astype(np.float32)
                dates = X_df.index.astype(str).values
            
            # Load returns - panel data (month × firm)
            R_df = pd.read_parquet(returns_path)
            print(f"  returns raw shape: {R_df.shape}, columns: {R_df.columns.tolist()}")
            
            # DATA-001 FIX: Aggregate panel returns to monthly cross-sectional mean
            if 'month_end' in R_df.columns and 'ret' in R_df.columns:
                # Group by month and compute cross-sectional mean return
                R_monthly = R_df.groupby('month_end')['ret'].mean()
                R_dates = R_monthly.index.astype(str).values
                R_values = R_monthly.values.astype(np.float32)
                print(f"  returns aggregated: {len(R_values)} months")
            elif 'ret' in R_df.columns:
                R_values = R_df['ret'].values.astype(np.float32)
                R_dates = R_df.index.astype(str).values
            else:
                R_values = R_df.iloc[:, -1].values.astype(np.float32)
                R_dates = R_df.index.astype(str).values
            
            # Align X_state dates with returns dates
            # Convert date formats (xstate: YYYYMM, returns: YYYYMMDD)
            xstate_dates_norm = [d[:6] if len(d) >= 6 else d for d in dates]
            returns_dates_norm = [d[:6] if len(d) >= 6 else d for d in R_dates]
            
            # Find common months
            xstate_date_set = set(xstate_dates_norm)
            common_dates = [d for d in returns_dates_norm if d in xstate_date_set]
            
            if len(common_dates) == 0:
                raise ValueError(f"No overlapping dates. xstate: {dates[:5]}, returns: {R_dates[:5]}")
            
            # Build aligned arrays
            xstate_date_to_idx = {d: i for i, d in enumerate(xstate_dates_norm)}
            returns_date_to_idx = {d: i for i, d in enumerate(returns_dates_norm)}
            
            aligned_X = []
            aligned_R = []
            for d in common_dates:
                if d in xstate_date_to_idx and d in returns_date_to_idx:
                    aligned_X.append(X[xstate_date_to_idx[d]])
                    aligned_R.append(R_values[returns_date_to_idx[d]])
            
            X = np.array(aligned_X, dtype=np.float32)
            R = np.array(aligned_R, dtype=np.float32)
            
            print(f"[OK] Loaded {len(X)} aligned samples with {X.shape[1]} features")
            print(f"   Date range: {common_dates[0]} to {common_dates[-1]}")
            return X, R, True
        
    except Exception as e:
        print(f"Failed to load real data: {e}")
        import traceback
        traceback.print_exc()
    
    # Fall back to synthetic
    print("[WARN] Using synthetic data for baseline benchmark")
    X, R = generate_synthetic_data(n_samples=500, n_features=48)
    return X, R, False


# =============================================================================
# Baseline Benchmark
# =============================================================================

def run_baseline_benchmark() -> Dict:
    """
    Run baseline benchmark and collect metrics.
    
    Returns:
        metrics: Dictionary of benchmark metrics
    """
    print("=" * 70)
    print("T4.1 Baseline Benchmark - SDF Training Performance")
    print("=" * 70)
    print(f"Timestamp: {BASELINE_CONFIG['timestamp']}")
    print(f"Device: {BASELINE_CONFIG['training']['device']}")
    print()
    
    # Load data
    X, returns, is_real_data = load_data()
    n_samples = len(X)
    
    # Split into train/val/test (70/15/15)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    R_train, R_val, R_test = returns[:n_train], returns[n_train:n_train+n_val], returns[n_train+n_val:]
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Initialize model
    device = torch.device(BASELINE_CONFIG["training"]["device"])
    model = BaselineSDF(
        input_dim=BASELINE_CONFIG["model"]["input_dim"],
        hidden_dim=BASELINE_CONFIG["model"]["hidden_dim"],
        num_hidden_layers=BASELINE_CONFIG["model"]["num_hidden_layers"],
        activation=BASELINE_CONFIG["model"]["activation"],
        output_activation=BASELINE_CONFIG["model"]["output_activation"],
        seed=BASELINE_CONFIG["training"]["seed"],
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=BASELINE_CONFIG["training"]["learning_rate"]
    )
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).to(device)
    R_train_t = torch.from_numpy(R_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    R_test_t = torch.from_numpy(R_test).to(device)
    
    # Training metrics
    epoch_times = []
    train_losses = []
    val_losses = []
    
    batch_size = BASELINE_CONFIG["training"]["batch_size"]
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    
    print()
    print("Training...")
    print("-" * 70)
    
    total_start = time.time()
    
    for epoch in range(BASELINE_CONFIG["training"]["num_epochs"]):
        epoch_start = time.time()
        
        # Training
        model.train()
        batch_losses = []
        
        # Shuffle indices
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            R_batch = R_train_t[batch_idx]
            
            m, z = model(X_batch)
            loss = compute_pricing_error(m, R_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        
        # Validation
        model.eval()
        with torch.no_grad():
            m_val, _ = model(X_val_t)
            val_loss = compute_pricing_error(m_val, R_val_t).item()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:2d}/{BASELINE_CONFIG['training']['num_epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {epoch_time:.2f}s")
    
    total_time = time.time() - total_start
    
    print("-" * 70)
    print(f"Total training time: {total_time:.2f}s")
    print()
    
    # Final evaluation
    print("Evaluating final model...")
    model.eval()
    with torch.no_grad():
        # In-sample (train)
        m_train, _ = model(X_train_t)
        is_loss = compute_pricing_error(m_train, R_train_t).item()
        
        # Out-of-sample (test)
        m_test, _ = model(X_test_t)
        oos_loss = compute_pricing_error(m_test, R_test_t).item()
        
        # Compute portfolio returns (simplified: weight by SDF)
        m_test_np = m_test.cpu().numpy().flatten()
        weights = m_test_np / m_test_np.sum()
        portfolio_returns = R_test * weights * len(weights)
    
    is_sharpe = compute_sharpe_ratio(R_train)
    oos_sharpe = compute_sharpe_ratio(portfolio_returns)
    
    # Compile metrics
    metrics = {
        "config": BASELINE_CONFIG,
        "data": {
            "is_real_data": is_real_data,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
        },
        "model": {
            "n_parameters": n_params,
        },
        "timing": {
            "total_training_time_sec": total_time,
            "avg_epoch_time_sec": np.mean(epoch_times),
            "min_epoch_time_sec": np.min(epoch_times),
            "max_epoch_time_sec": np.max(epoch_times),
            "epoch_times": epoch_times,
        },
        "convergence": {
            "initial_train_loss": train_losses[0],
            "final_train_loss": train_losses[-1],
            "initial_val_loss": val_losses[0],
            "final_val_loss": val_losses[-1],
            "loss_reduction_pct": (1 - train_losses[-1] / train_losses[0]) * 100,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        "performance": {
            "in_sample_loss": is_loss,
            "out_of_sample_loss": oos_loss,
            "oos_is_loss_ratio": oos_loss / is_loss if is_loss > 0 else float("inf"),
            "in_sample_sharpe": is_sharpe,
            "out_of_sample_sharpe": oos_sharpe,
            "oos_is_sharpe_ratio": oos_sharpe / is_sharpe if is_sharpe > 0 else float("inf"),
        },
        "gpu": {
            "device": BASELINE_CONFIG["training"]["device"],
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    
    # Print summary
    print("=" * 70)
    print("Baseline Benchmark Results")
    print("=" * 70)
    print(f"Training Time:     {total_time:.2f}s ({np.mean(epoch_times):.2f}s/epoch)")
    print(f"Final Train Loss:  {train_losses[-1]:.6f}")
    print(f"Final Val Loss:    {val_losses[-1]:.6f}")
    print(f"OOS/IS Loss Ratio: {metrics['performance']['oos_is_loss_ratio']:.4f}")
    print(f"IS Sharpe:         {is_sharpe:.4f}")
    print(f"OOS Sharpe:        {oos_sharpe:.4f}")
    print(f"OOS/IS Sharpe:     {metrics['performance']['oos_is_sharpe_ratio']:.4f}")
    print("=" * 70)
    
    return metrics


def save_metrics(metrics: Dict, output_path: Optional[str] = None):
    """Save metrics to JSON file."""
    if output_path is None:
        output_dir = SCRIPT_DIR.parent / "experiments" / "t4_baseline"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "baseline_metrics.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"\nMetrics saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    metrics = run_baseline_benchmark()
    save_metrics(metrics)
    print("\n✅ T4.1 Baseline Benchmark completed!")
