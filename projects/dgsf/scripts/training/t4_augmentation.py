"""
T4.6 Data Augmentation (T4-STR-5)

Purpose: Test data augmentation strategies for improving model robustness.

Strategies:
1. Temporal Jittering: Random shift of time indices
2. Gaussian Noise Injection: Add small noise to features
3. Feature Masking: Random zeroing of features

Expected Impact: 
- Improve model robustness
- Potentially improve OOS performance (or identify as "ineffective")

Usage:
    python scripts/t4_augmentation.py --benchmark
    
Output:
    experiments/t4_augmentation/results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

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
    "name": "T4.6 Data Augmentation Benchmark",
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
        "dropout": 0.4,  # Best from T4.5
        "l2_weight": 1e-4,  # Best from T4.5
    },
    "early_stopping": {
        "patience": 10,
        "min_delta": 0.0001,
    },
    "augmentation": {
        "noise_std": [0.01, 0.05, 0.1],  # Gaussian noise standard deviation
        "feature_mask_prob": [0.05, 0.1, 0.2],  # Probability of masking features
        "augment_prob": 0.5,  # Probability of applying augmentation to each sample
    },
}

# =============================================================================
# Data Augmentation Functions
# =============================================================================

def augment_none(X: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """No augmentation (baseline)."""
    return X, R


def augment_gaussian_noise(X: np.ndarray, R: np.ndarray, std: float = 0.05, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to features."""
    mask = np.random.random(len(X)) < prob
    X_aug = X.copy()
    X_aug[mask] = X_aug[mask] + np.random.randn(mask.sum(), X.shape[1]).astype(np.float32) * std
    return X_aug, R


def augment_feature_masking(X: np.ndarray, R: np.ndarray, mask_prob: float = 0.1, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly mask (zero out) features."""
    mask = np.random.random(len(X)) < prob
    X_aug = X.copy()
    feature_mask = np.random.random((mask.sum(), X.shape[1])) < mask_prob
    X_aug[mask] = X_aug[mask] * (1 - feature_mask.astype(np.float32))
    return X_aug, R


def augment_temporal_jitter(X: np.ndarray, R: np.ndarray, shift_range: int = 1, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate temporal jittering by mixing adjacent samples."""
    n = len(X)
    mask = np.random.random(n) < prob
    
    X_aug = X.copy()
    R_aug = R.copy()
    
    for i in np.where(mask)[0]:
        shift = np.random.randint(-shift_range, shift_range + 1)
        if 0 <= i + shift < n:
            # Mix with shifted sample
            alpha = np.random.uniform(0.8, 1.0)
            X_aug[i] = alpha * X[i] + (1 - alpha) * X[i + shift]
            R_aug[i] = alpha * R[i] + (1 - alpha) * R[i + shift]
    
    return X_aug, R_aug


# =============================================================================
# Model (with best regularization from T4.5)
# =============================================================================

class RegularizedSDF(nn.Module):
    """SDF model with dropout regularization."""
    
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
        self.best_state_dict = None
    
    def __call__(self, epoch: int, value: float, model: nn.Module) -> bool:
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = value
            self.best_epoch = epoch
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

def generate_data(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, CONFIG["model"]["input_dim"]).astype(np.float32)
    returns = (0.007 + 0.04 * np.random.randn(n_samples)).astype(np.float32)
    return X, returns


# =============================================================================
# Training Function
# =============================================================================

def train_with_augmentation(
    X_train: np.ndarray,
    R_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    augment_fn: Callable,
    augment_name: str,
) -> Dict:
    """Train with data augmentation."""
    device = torch.device(CONFIG["training"]["device"])
    seed = CONFIG["training"]["seed"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = RegularizedSDF(
        input_dim=CONFIG["model"]["input_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        dropout=CONFIG["model"]["dropout"],
        seed=seed,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["model"]["l2_weight"],
    )
    
    early_stopping = EarlyStopping(
        patience=CONFIG["early_stopping"]["patience"],
        min_delta=CONFIG["early_stopping"]["min_delta"],
    )
    
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    
    batch_size = CONFIG["training"]["batch_size"]
    train_losses = []
    val_losses = []
    epochs_run = 0
    
    start_time = time.time()
    
    for epoch in range(CONFIG["training"]["max_epochs"]):
        # Apply augmentation each epoch
        X_aug, R_aug = augment_fn(X_train, R_train)
        X_train_t = torch.from_numpy(X_aug).to(device)
        R_train_t = torch.from_numpy(R_aug).to(device)
        
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
        epochs_run = epoch + 1
        
        if early_stopping(epoch, val_loss, model):
            break
    
    total_time = time.time() - start_time
    
    # Load best model
    model = early_stopping.load_best(model)
    
    # Final evaluation on original (non-augmented) data
    X_train_orig_t = torch.from_numpy(X_train).to(device)
    R_train_orig_t = torch.from_numpy(R_train).to(device)
    
    model.eval()
    with torch.no_grad():
        m_train, _ = model(X_train_orig_t)
        final_train_loss = compute_pricing_error(m_train, R_train_orig_t).item()
        
        m_val, _ = model(X_val_t)
        final_val_loss = compute_pricing_error(m_val, R_val_t).item()
    
    return {
        "augmentation": augment_name,
        "epochs_run": epochs_run,
        "best_epoch": early_stopping.best_epoch + 1,
        "best_val_loss": early_stopping.best_value,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "oos_is_ratio": final_val_loss / final_train_loss if final_train_loss > 0 else float("inf"),
        "total_time_sec": total_time,
    }


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(verbose: bool = True) -> Dict:
    """Run data augmentation benchmark."""
    print("=" * 70)
    print("T4.6 Data Augmentation Benchmark (T4-STR-5)")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print(f"Model: Dropout={CONFIG['model']['dropout']}, L2={CONFIG['model']['l2_weight']}")
    print()
    
    # Generate data
    X, returns = generate_data(n_samples=500)
    n_train = int(0.7 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    R_train, R_val = returns[:n_train], returns[n_train:]
    
    print(f"Data: Train={len(X_train)}, Val={len(X_val)}")
    print()
    
    # Define augmentation strategies
    augmentations = [
        ("none", lambda X, R: augment_none(X, R)),
    ]
    
    # Gaussian noise variants
    for std in CONFIG["augmentation"]["noise_std"]:
        name = f"gaussian_noise_std={std}"
        augmentations.append((name, lambda X, R, s=std: augment_gaussian_noise(X, R, std=s)))
    
    # Feature masking variants
    for prob in CONFIG["augmentation"]["feature_mask_prob"]:
        name = f"feature_mask_prob={prob}"
        augmentations.append((name, lambda X, R, p=prob: augment_feature_masking(X, R, mask_prob=p)))
    
    # Temporal jitter
    augmentations.append(("temporal_jitter", lambda X, R: augment_temporal_jitter(X, R)))
    
    # Run benchmarks
    results = []
    
    print(f"Running {len(augmentations)} augmentation strategies...")
    print("-" * 70)
    
    for idx, (name, augment_fn) in enumerate(augmentations):
        print(f"[{idx+1:2d}/{len(augmentations)}] {name}...", end=" ")
        
        result = train_with_augmentation(
            X_train, R_train, X_val, R_val,
            augment_fn, name,
        )
        results.append(result)
        
        print(f"Val Loss: {result['final_val_loss']:.6f}, OOS/IS: {result['oos_is_ratio']:.3f}")
    
    # Analysis
    print()
    print("=" * 70)
    print("AUGMENTATION RESULTS")
    print("=" * 70)
    
    baseline = results[0]  # "none"
    
    print(f"{'Strategy':<30} {'Val Loss':<15} {'OOS/IS':<10} {'vs Baseline':<15}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x["final_val_loss"]):
        delta = (r["final_val_loss"] - baseline["final_val_loss"]) / baseline["final_val_loss"] * 100
        delta_str = f"{delta:+.1f}%" if r["augmentation"] != "none" else "-"
        print(f"{r['augmentation']:<30} {r['final_val_loss']:<15.6f} {r['oos_is_ratio']:<10.3f} {delta_str:<15}")
    
    # Find best augmentation
    best = min(results, key=lambda x: x["final_val_loss"])
    
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Check if any augmentation improves over baseline
    improvements = [r for r in results if r["final_val_loss"] < baseline["final_val_loss"]]
    
    if improvements:
        best_improvement = min(improvements, key=lambda x: x["final_val_loss"])
        improvement_pct = (baseline["final_val_loss"] - best_improvement["final_val_loss"]) / baseline["final_val_loss"] * 100
        print(f"✅ Best augmentation: {best_improvement['augmentation']}")
        print(f"   Improvement: {improvement_pct:.1f}% lower validation loss")
        assessment = "EFFECTIVE"
        recommendation = {
            "use_augmentation": True,
            "strategy": best_improvement["augmentation"],
            "rationale": f"{improvement_pct:.1f}% improvement in validation loss",
        }
    else:
        print("⚠️ No augmentation improved over baseline")
        print("   Recommendation: DISABLE augmentation for this model/data")
        assessment = "INEFFECTIVE"
        recommendation = {
            "use_augmentation": False,
            "strategy": "none",
            "rationale": "No augmentation improved OOS performance",
        }
    
    # OOS performance check
    print()
    if best["oos_is_ratio"] < baseline["oos_is_ratio"]:
        oos_improvement = (baseline["oos_is_ratio"] - best["oos_is_ratio"]) / baseline["oos_is_ratio"] * 100
        print(f"✅ OOS/IS ratio improved by {oos_improvement:.1f}%")
    else:
        print("ℹ️ OOS/IS ratio did not improve with augmentation")
    
    output = {
        "config": CONFIG,
        "results": results,
        "baseline": {
            "final_val_loss": baseline["final_val_loss"],
            "oos_is_ratio": baseline["oos_is_ratio"],
        },
        "best": {
            "strategy": best["augmentation"],
            "final_val_loss": best["final_val_loss"],
            "oos_is_ratio": best["oos_is_ratio"],
        },
        "assessment": assessment,
        "recommendation": recommendation,
    }
    
    return output


def save_results(results: Dict, output_path: str = None):
    """Save benchmark results."""
    if output_path is None:
        output_dir = PROJECT_ROOT / "experiments" / "t4_augmentation"
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
    parser = argparse.ArgumentParser(description="T4.6 Data Augmentation Benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = run_benchmark(verbose=not args.quiet)
    save_results(results)
    print("\n✅ T4.6 Data Augmentation Benchmark completed!")


if __name__ == "__main__":
    main()
