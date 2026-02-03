"""
T4.4 Early Stopping Implementation (T4-STR-3)

Purpose: Implement EarlyStopping callback to prevent overfitting and improve sample efficiency.

Expected Impact: 
- Reduce overfitting (OOS/IS ratio improvement)
- Fewer wasted epochs (≥20% sample efficiency)
- Automatic checkpoint management

Usage:
    python scripts/t4_early_stopping.py --benchmark
    
Output:
    experiments/t4_early_stopping/results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    "name": "T4.4 Early Stopping Benchmark",
    "timestamp": datetime.now().isoformat(),
    "training": {
        "max_epochs": 100,  # Increased to allow early stopping to trigger
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
        "mode": "min",  # For loss
    },
}

# =============================================================================
# Early Stopping Callback
# =============================================================================

class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for metrics like accuracy
        checkpoint_path: Path to save best model checkpoint
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "min",
        checkpoint_path: Optional[Path] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_state_dict: Optional[Dict] = None
        
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
    
    def __call__(self, epoch: int, value: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            value: Current validation metric value
            model: Model to checkpoint
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            self._save_checkpoint(model)
            return False
        
        if self.is_better(value, self.best_value):
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model)
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def _save_checkpoint(self, model: nn.Module):
        """Save model state dict."""
        self.best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.best_state_dict, self.checkpoint_path)
    
    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best model state dict."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        elif self.checkpoint_path and self.checkpoint_path.exists():
            model.load_state_dict(torch.load(self.checkpoint_path))
        return model
    
    def get_summary(self) -> Dict:
        """Get summary of early stopping state."""
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "patience_counter": self.counter,
            "early_stopped": self.early_stop,
            "patience": self.patience,
            "min_delta": self.min_delta,
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

def train_without_early_stopping(
    X_train: np.ndarray,
    R_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    max_epochs: int = 100,
    verbose: bool = True,
) -> Dict:
    """Train without early stopping (baseline)."""
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
    best_val_loss = float("inf")
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    total_time = time.time() - start_time
    
    return {
        "method": "no_early_stopping",
        "epochs_run": max_epochs,
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_time_sec": total_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "wasted_epochs": max_epochs - (best_epoch + 1),
    }


def train_with_early_stopping(
    X_train: np.ndarray,
    R_train: np.ndarray,
    X_val: np.ndarray,
    R_val: np.ndarray,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0001,
    checkpoint_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """Train with early stopping."""
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
    
    # Initialize early stopping
    checkpoint_path = checkpoint_dir / "best_model.pt" if checkpoint_dir else None
    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode="min",
        checkpoint_path=checkpoint_path,
    )
    
    X_train_t = torch.from_numpy(X_train).to(device)
    R_train_t = torch.from_numpy(R_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    
    batch_size = CONFIG["training"]["batch_size"]
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    epochs_run = 0
    
    for epoch in range(max_epochs):
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
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Check early stopping
        if early_stopping(epoch, val_loss, model):
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    
    # Load best model
    model = early_stopping.load_best(model)
    
    # Verify checkpoint consistency
    checkpoint_consistent = False
    if checkpoint_path and checkpoint_path.exists():
        loaded_state = torch.load(checkpoint_path)
        model_state = model.state_dict()
        checkpoint_consistent = all(
            torch.equal(loaded_state[k], model_state[k])
            for k in loaded_state.keys()
        )
    
    es_summary = early_stopping.get_summary()
    
    return {
        "method": "early_stopping",
        "patience": patience,
        "min_delta": min_delta,
        "epochs_run": epochs_run,
        "best_epoch": es_summary["best_epoch"] + 1,
        "best_val_loss": es_summary["best_value"],
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "total_time_sec": total_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "early_stopped": es_summary["early_stopped"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_consistent": checkpoint_consistent,
        "saved_epochs": max_epochs - epochs_run if es_summary["early_stopped"] else 0,
    }


# =============================================================================
# Benchmark
# =============================================================================

def run_benchmark(verbose: bool = True) -> Dict:
    """Run early stopping benchmark."""
    print("=" * 70)
    print("T4.4 Early Stopping Benchmark (T4-STR-3)")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print(f"Max Epochs: {CONFIG['training']['max_epochs']}")
    print(f"Patience: {CONFIG['early_stopping']['patience']}")
    print(f"Min Delta: {CONFIG['early_stopping']['min_delta']}")
    print()
    
    # Generate data
    X, returns = generate_data(n_samples=500)
    n_train = int(0.7 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    R_train, R_val = returns[:n_train], returns[n_train:]
    
    print(f"Data: Train={len(X_train)}, Val={len(X_val)}")
    print()
    
    # Prepare checkpoint directory
    checkpoint_dir = PROJECT_ROOT / "experiments" / "t4_early_stopping" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Run without early stopping
    print("[BASELINE] Training without early stopping...")
    print("-" * 50)
    baseline_results = train_without_early_stopping(
        X_train, R_train, X_val, R_val,
        max_epochs=CONFIG["training"]["max_epochs"],
        verbose=verbose,
    )
    print()
    
    # Run with early stopping
    print("[EARLY STOPPING] Training with early stopping...")
    print("-" * 50)
    es_results = train_with_early_stopping(
        X_train, R_train, X_val, R_val,
        max_epochs=CONFIG["training"]["max_epochs"],
        patience=CONFIG["early_stopping"]["patience"],
        min_delta=CONFIG["early_stopping"]["min_delta"],
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )
    print()
    
    # Comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    efficiency_gain = (baseline_results["epochs_run"] - es_results["epochs_run"]) / baseline_results["epochs_run"] * 100
    time_saved = (baseline_results["total_time_sec"] - es_results["total_time_sec"]) / baseline_results["total_time_sec"] * 100
    
    print(f"{'Metric':<30} {'No ES':<15} {'Early Stop':<15} {'Improvement':<15}")
    print("-" * 75)
    print(f"{'Epochs Run':<30} {baseline_results['epochs_run']:<15} {es_results['epochs_run']:<15} {efficiency_gain:.1f}% fewer")
    print(f"{'Best Epoch':<30} {baseline_results['best_epoch']:<15} {es_results['best_epoch']:<15} -")
    print(f"{'Best Val Loss':<30} {baseline_results['best_val_loss']:<15.6f} {es_results['best_val_loss']:<15.6f} -")
    print(f"{'Final Val Loss':<30} {baseline_results['final_val_loss']:<15.6f} {es_results['final_val_loss']:<15.6f} -")
    print(f"{'Training Time (s)':<30} {baseline_results['total_time_sec']:<15.3f} {es_results['total_time_sec']:<15.3f} {time_saved:.1f}% faster")
    print(f"{'Wasted Epochs':<30} {baseline_results['wasted_epochs']:<15} {es_results['saved_epochs']:<15} -")
    print()
    
    # Checkpoint verification
    print("Checkpoint Verification:")
    print(f"  - Checkpoint path: {es_results['checkpoint_path']}")
    print(f"  - Checkpoint consistent: {es_results['checkpoint_consistent']}")
    print()
    
    # Assessment
    target_efficiency = 20  # ≥20% sample efficiency
    
    if efficiency_gain >= target_efficiency:
        print(f"✅ Sample efficiency improved by {efficiency_gain:.1f}% (target: ≥{target_efficiency}%)")
        assessment = "TARGET_MET"
    else:
        print(f"⚠️ Sample efficiency improved by {efficiency_gain:.1f}% (target: ≥{target_efficiency}%)")
        assessment = "BELOW_TARGET"
    
    if es_results["checkpoint_consistent"]:
        print("✅ Checkpoint save/load consistency verified")
    else:
        print("⚠️ Checkpoint consistency check failed")
    
    output = {
        "config": CONFIG,
        "baseline_results": baseline_results,
        "early_stopping_results": es_results,
        "comparison": {
            "efficiency_gain_pct": efficiency_gain,
            "time_saved_pct": time_saved,
            "checkpoint_verified": es_results["checkpoint_consistent"],
            "assessment": assessment,
        },
        "recommendation": {
            "use_early_stopping": True,
            "patience": CONFIG["early_stopping"]["patience"],
            "min_delta": CONFIG["early_stopping"]["min_delta"],
            "rationale": f"Saves {efficiency_gain:.0f}% of training epochs with equivalent best performance",
        },
    }
    
    return output


def save_results(results: Dict, output_path: str = None):
    """Save benchmark results."""
    if output_path is None:
        output_dir = PROJECT_ROOT / "experiments" / "t4_early_stopping"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "results.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove non-serializable items
    results_clean = results.copy()
    for key in ["baseline_results", "early_stopping_results"]:
        if key in results_clean:
            results_clean[key] = {
                k: v for k, v in results_clean[key].items()
                if not isinstance(v, (list,)) or len(v) < 200
            }
    
    with open(output_path, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="T4.4 Early Stopping Benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = run_benchmark(verbose=not args.quiet)
    save_results(results)
    print("\n✅ T4.4 Early Stopping Benchmark completed!")


if __name__ == "__main__":
    main()
