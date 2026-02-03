"""
T4.3 Mixed Precision Training Benchmark (T4-STR-2)

Purpose: Test FP16 mixed precision training for potential speedup.

Expected Impact: 40-50% speedup on GPU (minimal impact on CPU)

Note: Mixed precision primarily benefits GPU training. On CPU, the benefit
is limited or may even cause slight slowdown due to conversion overhead.

Usage:
    python scripts/t4_mixed_precision.py --benchmark
    
Output:
    experiments/t4_mixed_precision/results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# Check AMP availability
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = torch.cuda.is_available()
except ImportError:
    AMP_AVAILABLE = False
    GradScaler = None
    autocast = None

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "name": "T4.3 Mixed Precision Benchmark",
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
    },
}

# =============================================================================
# Model
# =============================================================================

class BaselineSDF(nn.Module):
    """Minimal SDF model for benchmark."""
    
    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)
        self.output_act = nn.Softplus()
    
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
# Data Generation
# =============================================================================

def generate_data(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, CONFIG["model"]["input_dim"]).astype(np.float32)
    returns = (0.007 + 0.04 * np.random.randn(n_samples)).astype(np.float32)
    return X, returns


# =============================================================================
# Training Functions
# =============================================================================

def train_fp32(
    X_train: np.ndarray,
    R_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """Train with standard FP32 precision."""
    device = torch.device(CONFIG["training"]["device"])
    seed = CONFIG["training"]["seed"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = BaselineSDF(
        input_dim=CONFIG["model"]["input_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        seed=seed,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["training"]["learning_rate"])
    
    X_train_t = torch.from_numpy(X_train).to(device)
    R_train_t = torch.from_numpy(R_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    
    batch_size = CONFIG["training"]["batch_size"]
    train_losses = []
    val_losses = []
    epoch_times = []
    
    start_time = time.time()
    
    for epoch in range(CONFIG["training"]["num_epochs"]):
        epoch_start = time.time()
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
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(time.time() - epoch_start)
        
        if verbose:
            print(f"  Epoch {epoch+1:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    total_time = time.time() - start_time
    
    # Final predictions for comparison
    model.eval()
    with torch.no_grad():
        m_final, _ = model(X_val_t)
        final_predictions = m_final.cpu().numpy()
    
    return {
        "precision": "FP32",
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "total_time_sec": total_time,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "avg_epoch_time": np.mean(epoch_times),
        "final_predictions": final_predictions.tolist()[:10],  # First 10 for comparison
    }


def train_fp16(
    X_train: np.ndarray,
    R_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """Train with mixed precision (FP16) using AMP."""
    device = torch.device(CONFIG["training"]["device"])
    seed = CONFIG["training"]["seed"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = BaselineSDF(
        input_dim=CONFIG["model"]["input_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        seed=seed,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["training"]["learning_rate"])
    
    # Initialize GradScaler for mixed precision
    use_amp = AMP_AVAILABLE and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    
    X_train_t = torch.from_numpy(X_train).to(device)
    R_train_t = torch.from_numpy(R_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    
    batch_size = CONFIG["training"]["batch_size"]
    train_losses = []
    val_losses = []
    epoch_times = []
    
    start_time = time.time()
    
    for epoch in range(CONFIG["training"]["num_epochs"]):
        epoch_start = time.time()
        model.train()
        batch_losses = []
        
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            R_batch = R_train_t[batch_idx]
            
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision forward pass
                with autocast():
                    m, _ = model(X_batch)
                    loss = compute_pricing_error(m, R_batch)
                
                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward/backward (CPU fallback)
                m, _ = model(X_batch)
                loss = compute_pricing_error(m, R_batch)
                loss.backward()
                optimizer.step()
            
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        
        model.eval()
        with torch.no_grad():
            if use_amp:
                with autocast():
                    m_val, _ = model(X_val_t)
                    val_loss = compute_pricing_error(m_val, R_val_t).item()
            else:
                m_val, _ = model(X_val_t)
                val_loss = compute_pricing_error(m_val, R_val_t).item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(time.time() - epoch_start)
        
        if verbose:
            print(f"  Epoch {epoch+1:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    total_time = time.time() - start_time
    
    # Final predictions for comparison
    model.eval()
    with torch.no_grad():
        m_final, _ = model(X_val_t)
        final_predictions = m_final.cpu().numpy()
    
    return {
        "precision": "FP16 (AMP)" if use_amp else "FP16 (CPU fallback to FP32)",
        "amp_used": use_amp,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "total_time_sec": total_time,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "avg_epoch_time": np.mean(epoch_times),
        "final_predictions": final_predictions.tolist()[:10],
    }


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(verbose: bool = True) -> Dict:
    """Run FP32 vs FP16 benchmark."""
    print("=" * 70)
    print("T4.3 Mixed Precision Benchmark (T4-STR-2)")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print(f"Device: {CONFIG['training']['device']}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"AMP Available: {AMP_AVAILABLE}")
    print()
    
    # Generate data
    X, returns = generate_data(n_samples=500)
    n_train = int(0.7 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    R_train, R_val = returns[:n_train], returns[n_train:]
    
    print(f"Data: Train={len(X_train)}, Val={len(X_val)}")
    print()
    
    # FP32 baseline
    print("[FP32] Training with standard precision...")
    print("-" * 50)
    fp32_results = train_fp32(X_train, R_train, X_val, R_val, verbose=verbose)
    print()
    
    # FP16 (or CPU fallback)
    print("[FP16] Training with mixed precision...")
    print("-" * 50)
    fp16_results = train_fp16(X_train, R_train, X_val, R_val, verbose=verbose)
    print()
    
    # Comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    speedup = fp32_results["total_time_sec"] / fp16_results["total_time_sec"]
    loss_diff = abs(fp32_results["final_val_loss"] - fp16_results["final_val_loss"])
    loss_diff_pct = loss_diff / fp32_results["final_val_loss"] * 100
    
    # Prediction difference
    fp32_preds = np.array(fp32_results["final_predictions"])
    fp16_preds = np.array(fp16_results["final_predictions"])
    pred_diff = np.mean(np.abs(fp32_preds - fp16_preds))
    pred_diff_pct = pred_diff / np.mean(np.abs(fp32_preds)) * 100
    
    print(f"{'Metric':<25} {'FP32':<15} {'FP16':<15} {'Diff':<15}")
    print("-" * 70)
    print(f"{'Total Time (s)':<25} {fp32_results['total_time_sec']:<15.3f} {fp16_results['total_time_sec']:<15.3f} {speedup:.2f}x")
    print(f"{'Avg Epoch Time (s)':<25} {fp32_results['avg_epoch_time']:<15.4f} {fp16_results['avg_epoch_time']:<15.4f} -")
    print(f"{'Final Train Loss':<25} {fp32_results['final_train_loss']:<15.6f} {fp16_results['final_train_loss']:<15.6f} -")
    print(f"{'Final Val Loss':<25} {fp32_results['final_val_loss']:<15.6f} {fp16_results['final_val_loss']:<15.6f} {loss_diff_pct:.2f}%")
    print(f"{'Prediction Diff':<25} {'-':<15} {'-':<15} {pred_diff_pct:.4f}%")
    
    # Assessment
    print()
    if not AMP_AVAILABLE:
        print("⚠️ Note: CUDA not available. FP16 falls back to FP32 on CPU.")
        print("   Mixed precision benefit requires GPU.")
        assessment = "CPU_ONLY_NO_BENEFIT"
    elif loss_diff_pct > 1.0:
        print("⚠️ Warning: Precision difference > 1%. FP16 may not be suitable.")
        assessment = "PRECISION_CONCERN"
    elif speedup < 1.1:
        print("ℹ️ Minimal speedup observed. Consider for GPU-bound workloads.")
        assessment = "MINIMAL_BENEFIT"
    else:
        print(f"✅ FP16 provides {speedup:.1f}x speedup with acceptable precision loss.")
        assessment = "RECOMMENDED"
    
    output = {
        "config": CONFIG,
        "fp32_results": fp32_results,
        "fp16_results": fp16_results,
        "comparison": {
            "speedup": speedup,
            "loss_diff_pct": loss_diff_pct,
            "prediction_diff_pct": pred_diff_pct,
            "assessment": assessment,
        },
        "recommendation": {
            "use_fp16": assessment in ["RECOMMENDED", "MINIMAL_BENEFIT"],
            "rationale": "Enable for GPU training; minimal benefit on CPU" if not AMP_AVAILABLE else f"Speedup: {speedup:.1f}x",
        },
    }
    
    return output


def save_results(results: Dict, output_path: str = None):
    """Save benchmark results."""
    if output_path is None:
        output_dir = PROJECT_ROOT / "experiments" / "t4_mixed_precision"
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
    parser = argparse.ArgumentParser(description="T4.3 Mixed Precision Benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = run_benchmark(verbose=not args.quiet)
    save_results(results)
    print("\n✅ T4.3 Mixed Precision Benchmark completed!")


if __name__ == "__main__":
    main()
