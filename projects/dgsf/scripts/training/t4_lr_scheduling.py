"""
T4.2 Learning Rate Scheduling Benchmark (T4-STR-1)

Purpose: Test 3 LR schedulers and select the best for SDF training.

Schedulers:
1. CosineAnnealingLR - Smooth cosine decay
2. ReduceLROnPlateau - Adaptive reduction on plateau
3. OneCycleLR - Super-convergence schedule

Expected Impact: 15-20% speedup in convergence

Usage:
    python scripts/t4_lr_scheduling.py --benchmark
    python scripts/t4_lr_scheduling.py --scheduler cosine
    
Output:
    experiments/t4_lr_scheduling/results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "name": "T4.2 LR Scheduling Benchmark",
    "timestamp": datetime.now().isoformat(),
    "training": {
        "num_epochs": 30,  # More epochs to see scheduler effects
        "base_learning_rate": 1e-3,
        "batch_size": 32,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "model": {
        "input_dim": 48,
        "hidden_dim": 64,
        "num_hidden_layers": 3,
    },
    "schedulers": {
        "none": {"type": "none"},
        "cosine": {
            "type": "CosineAnnealingLR",
            "T_max": 30,
            "eta_min": 1e-6,
        },
        "plateau": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
        },
        "onecycle": {
            "type": "OneCycleLR",
            "max_lr": 1e-2,
            "pct_start": 0.3,
            "anneal_strategy": "cos",
        },
    },
    "convergence_target": 0.002,  # Target loss to measure epochs-to-convergence
}

# =============================================================================
# Model (same as baseline)
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
# Data Generation (synthetic)
# =============================================================================

def generate_data(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, CONFIG["model"]["input_dim"]).astype(np.float32)
    returns = (0.007 + 0.04 * np.random.randn(n_samples)).astype(np.float32)
    return X, returns


# =============================================================================
# Scheduler Factory
# =============================================================================

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    n_batches_per_epoch: int,
) -> Optional[object]:
    """Create learning rate scheduler."""
    cfg = CONFIG["schedulers"][scheduler_name]
    
    if cfg["type"] == "none":
        return None
    elif cfg["type"] == "CosineAnnealingLR":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg["T_max"],
            eta_min=cfg["eta_min"],
        )
    elif cfg["type"] == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg["mode"],
            factor=cfg["factor"],
            patience=cfg["patience"],
            min_lr=cfg["min_lr"],
        )
    elif cfg["type"] == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=cfg["max_lr"],
            epochs=CONFIG["training"]["num_epochs"],
            steps_per_epoch=n_batches_per_epoch,
            pct_start=cfg["pct_start"],
            anneal_strategy=cfg["anneal_strategy"],
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg['type']}")


# =============================================================================
# Training Function
# =============================================================================

def train_with_scheduler(
    scheduler_name: str,
    X_train: np.ndarray,
    R_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """Train model with specified scheduler and return metrics."""
    
    device = torch.device(CONFIG["training"]["device"])
    seed = CONFIG["training"]["seed"]
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize model
    model = BaselineSDF(
        input_dim=CONFIG["model"]["input_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        seed=seed,
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["training"]["base_learning_rate"],
    )
    
    # Create scheduler
    batch_size = CONFIG["training"]["batch_size"]
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    scheduler = create_scheduler(optimizer, scheduler_name, n_batches)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).to(device)
    R_train_t = torch.from_numpy(R_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    
    # Training metrics
    train_losses = []
    val_losses = []
    learning_rates = []
    epoch_times = []
    epochs_to_target = None
    
    start_time = time.time()
    
    for epoch in range(CONFIG["training"]["num_epochs"]):
        epoch_start = time.time()
        model.train()
        batch_losses = []
        
        # Shuffle
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
            
            # OneCycleLR steps per batch
            if scheduler_name == "onecycle" and scheduler is not None:
                scheduler.step()
        
        train_loss = np.mean(batch_losses)
        
        # Validation
        model.eval()
        with torch.no_grad():
            m_val, _ = model(X_val_t)
            val_loss = compute_pricing_error(m_val, R_val_t).item()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)
        epoch_times.append(time.time() - epoch_start)
        
        # Check convergence target
        if epochs_to_target is None and train_loss <= CONFIG["convergence_target"]:
            epochs_to_target = epoch + 1
        
        # Step scheduler (except OneCycleLR which steps per batch)
        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(val_loss)
            elif scheduler_name != "onecycle":
                scheduler.step()
        
        if verbose:
            print(f"  Epoch {epoch+1:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")
    
    total_time = time.time() - start_time
    
    return {
        "scheduler": scheduler_name,
        "scheduler_config": CONFIG["schedulers"][scheduler_name],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rates": learning_rates,
        "epoch_times": epoch_times,
        "total_time_sec": total_time,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "epochs_to_target": epochs_to_target,
        "convergence_target": CONFIG["convergence_target"],
        "best_val_loss": min(val_losses),
        "best_val_epoch": val_losses.index(min(val_losses)) + 1,
    }


# =============================================================================
# Benchmark All Schedulers
# =============================================================================

def run_benchmark(verbose: bool = True) -> Dict:
    """Run benchmark for all schedulers."""
    print("=" * 70)
    print("T4.2 LR Scheduling Benchmark (T4-STR-1)")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print(f"Device: {CONFIG['training']['device']}")
    print(f"Epochs: {CONFIG['training']['num_epochs']}")
    print(f"Convergence Target: {CONFIG['convergence_target']}")
    print()
    
    # Generate data
    X, returns = generate_data(n_samples=500)
    n_train = int(0.7 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    R_train, R_val = returns[:n_train], returns[n_train:]
    
    print(f"Data: Train={len(X_train)}, Val={len(X_val)}")
    print()
    
    results = {}
    scheduler_names = ["none", "cosine", "plateau", "onecycle"]
    
    for name in scheduler_names:
        print(f"[{name.upper()}] Training with {CONFIG['schedulers'][name]['type']}...")
        print("-" * 50)
        
        result = train_with_scheduler(
            scheduler_name=name,
            X_train=X_train,
            R_train=R_train,
            X_val=X_val,
            R_val=R_val,
            verbose=verbose,
        )
        results[name] = result
        print()
    
    # Comparison summary
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Scheduler':<12} {'Final Train':<12} {'Final Val':<12} {'Best Val':<12} {'Epochs→Target':<15} {'Time (s)':<10}")
    print("-" * 70)
    
    for name, r in results.items():
        epochs_str = str(r["epochs_to_target"]) if r["epochs_to_target"] else "N/A"
        print(f"{name:<12} {r['final_train_loss']:<12.6f} {r['final_val_loss']:<12.6f} {r['best_val_loss']:<12.6f} {epochs_str:<15} {r['total_time_sec']:<10.2f}")
    
    # Select best scheduler
    best_scheduler = min(
        results.keys(),
        key=lambda k: (
            results[k]["epochs_to_target"] or float("inf"),
            results[k]["best_val_loss"],
        )
    )
    
    print()
    print(f"🏆 BEST SCHEDULER: {best_scheduler.upper()}")
    print(f"   - Epochs to target: {results[best_scheduler]['epochs_to_target']}")
    print(f"   - Best val loss: {results[best_scheduler]['best_val_loss']:.6f}")
    
    # Compile final output
    output = {
        "config": CONFIG,
        "results": results,
        "comparison": {
            "best_scheduler": best_scheduler,
            "ranking": sorted(
                scheduler_names,
                key=lambda k: (
                    results[k]["epochs_to_target"] or float("inf"),
                    results[k]["best_val_loss"],
                )
            ),
        },
        "recommendation": {
            "scheduler": best_scheduler,
            "config": CONFIG["schedulers"][best_scheduler],
            "rationale": f"Fastest convergence with {results[best_scheduler]['epochs_to_target']} epochs to target loss",
        },
    }
    
    return output


def save_results(results: Dict, output_path: Optional[str] = None):
    """Save benchmark results to JSON."""
    if output_path is None:
        output_dir = PROJECT_ROOT / "experiments" / "t4_lr_scheduling"
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
    parser = argparse.ArgumentParser(description="T4.2 LR Scheduling Benchmark")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark of all schedulers",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine", "plateau", "onecycle"],
        help="Test specific scheduler only",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress epoch-level output",
    )
    
    args = parser.parse_args()
    
    if args.benchmark or (not args.scheduler):
        results = run_benchmark(verbose=not args.quiet)
        save_results(results)
        print("\n✅ T4.2 LR Scheduling Benchmark completed!")
    elif args.scheduler:
        print(f"Testing scheduler: {args.scheduler}")
        X, returns = generate_data()
        n_train = int(0.7 * len(X))
        result = train_with_scheduler(
            scheduler_name=args.scheduler,
            X_train=X[:n_train],
            R_train=returns[:n_train],
            X_val=X[n_train:],
            R_val=returns[n_train:],
            verbose=not args.quiet,
        )
        print(f"\nFinal train loss: {result['final_train_loss']:.6f}")
        print(f"Final val loss: {result['final_val_loss']:.6f}")


if __name__ == "__main__":
    main()
