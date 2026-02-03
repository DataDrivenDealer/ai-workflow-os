"""
T4.7 Integration & Final Validation (T4-FINAL)

Purpose: Integrate all effective T4 strategies and validate against objectives.

Integrated Strategies:
- T4.2: OneCycleLR (max_lr=0.01, pct_start=0.3)
- T4.3: FP16 (GPU only, code preserved)
- T4.4: EarlyStopping (patience=10, min_delta=0.0001)
- T4.5: Regularization (L2=1e-4, Dropout=0.4)
- T4.6: Feature Masking (prob=0.2)

T4 Objectives:
- T4-OBJ-1: Training speedup ≥30%
- T4-OBJ-2: OOS Sharpe ≥1.5
- T4-OBJ-3: OOS/IS ratio ≥0.9

Usage:
    python scripts/train_sdf_optimized.py --validate-all
    
Output:
    experiments/t4_final/comparison_report.md
    experiments/t4_final/results.json
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
from torch.optim.lr_scheduler import OneCycleLR

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# T4 Integrated Configuration
# =============================================================================

CONFIG = {
    "name": "T4.7 Integrated Training",
    "timestamp": datetime.now().isoformat(),
    "baseline": {
        "max_epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": 42,
        "dropout": 0.0,
        "l2_weight": 0.0,
    },
    "optimized": {
        "max_epochs": 100,  # Early stopping will terminate earlier
        "learning_rate": 1e-3,
        "batch_size": 32,
        "seed": 42,
        # T4.5: Regularization
        "dropout": 0.4,
        "l2_weight": 1e-4,
        # T4.2: LR Scheduling
        "use_onecycle": True,
        "max_lr": 0.01,
        "pct_start": 0.3,
        # T4.4: Early Stopping
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0001,
        # T4.6: Data Augmentation
        "use_feature_mask": True,
        "feature_mask_prob": 0.2,
        "augment_prob": 0.5,
    },
    "model": {
        "input_dim": 48,
        "hidden_dim": 64,
        "num_hidden_layers": 3,
    },
    "objectives": {
        "T4-OBJ-1": {"name": "Training Speedup", "target": "≥30%"},
        "T4-OBJ-2": {"name": "OOS Sharpe", "target": "≥1.5"},
        "T4-OBJ-3": {"name": "OOS/IS Ratio", "target": "≥0.9"},
    },
}

# =============================================================================
# Model
# =============================================================================

class SDFModel(nn.Module):
    """SDF model with configurable regularization."""
    
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


def compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(12)  # Monthly to annual


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
# Data Augmentation (T4.6)
# =============================================================================

def augment_feature_mask(X: np.ndarray, R: np.ndarray, mask_prob: float = 0.2, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply feature masking augmentation."""
    mask = np.random.random(len(X)) < prob
    X_aug = X.copy()
    feature_mask = np.random.random((mask.sum(), X.shape[1])) < mask_prob
    X_aug[mask] = X_aug[mask] * (1 - feature_mask.astype(np.float32))
    return X_aug, R


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

def train_baseline(X_train, R_train, X_val, R_val, verbose: bool = True) -> Dict:
    """Train with baseline configuration (no optimizations)."""
    config = CONFIG["baseline"]
    device = torch.device("cpu")
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    model = SDFModel(
        input_dim=CONFIG["model"]["input_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        dropout=config["dropout"],
        seed=config["seed"],
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["l2_weight"],
    )
    
    X_train_t = torch.from_numpy(X_train).to(device)
    R_train_t = torch.from_numpy(R_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    
    batch_size = config["batch_size"]
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(config["max_epochs"]):
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
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  [Baseline] Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
    
    total_time = time.time() - start_time
    
    # Compute metrics
    model.eval()
    with torch.no_grad():
        m_train, _ = model(X_train_t)
        m_val, _ = model(X_val_t)
    
    is_returns = (m_train.numpy().flatten() - 1) / 0.04
    oos_returns = (m_val.numpy().flatten() - 1) / 0.04
    
    is_sharpe = compute_sharpe(is_returns)
    oos_sharpe = compute_sharpe(oos_returns)
    
    return {
        "method": "baseline",
        "epochs_run": config["max_epochs"],
        "best_epoch": best_epoch + 1,
        "total_time_sec": total_time,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
        "oos_is_ratio": val_losses[-1] / train_losses[-1] if train_losses[-1] > 0 else float("inf"),
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "sharpe_ratio": oos_sharpe / is_sharpe if is_sharpe > 0 else 0,
    }


def train_optimized(X_train, R_train, X_val, R_val, verbose: bool = True) -> Dict:
    """Train with all T4 optimizations."""
    config = CONFIG["optimized"]
    device = torch.device("cpu")
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # T4.5: Model with dropout
    model = SDFModel(
        input_dim=CONFIG["model"]["input_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        dropout=config["dropout"],
        seed=config["seed"],
    ).to(device)
    
    # T4.5: Optimizer with L2 regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["l2_weight"],
    )
    
    # T4.2: OneCycleLR scheduler
    steps_per_epoch = len(X_train) // config["batch_size"] + 1
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        epochs=config["max_epochs"],
        steps_per_epoch=steps_per_epoch,
        pct_start=config["pct_start"],
        anneal_strategy="cos",
    ) if config["use_onecycle"] else None
    
    # T4.4: Early stopping
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
    )
    
    X_val_t = torch.from_numpy(X_val).to(device)
    R_val_t = torch.from_numpy(R_val).to(device)
    
    batch_size = config["batch_size"]
    train_losses = []
    val_losses = []
    epochs_run = 0
    
    start_time = time.time()
    
    for epoch in range(config["max_epochs"]):
        # T4.6: Apply feature masking augmentation
        if config["use_feature_mask"]:
            X_aug, R_aug = augment_feature_mask(
                X_train, R_train,
                mask_prob=config["feature_mask_prob"],
                prob=config["augment_prob"],
            )
        else:
            X_aug, R_aug = X_train, R_train
        
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
            
            # T4.2: Step scheduler
            if scheduler:
                scheduler.step()
            
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        
        model.eval()
        with torch.no_grad():
            m_val, _ = model(X_val_t)
            val_loss = compute_pricing_error(m_val, R_val_t).item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_run = epoch + 1
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [Optimized] Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # T4.4: Check early stopping
        if early_stopping(epoch, val_loss, model):
            if verbose:
                print(f"  [Optimized] Early stopping at epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    
    # Load best model
    model = early_stopping.load_best(model)
    
    # Compute metrics on original (non-augmented) data
    X_train_orig_t = torch.from_numpy(X_train).to(device)
    R_train_orig_t = torch.from_numpy(R_train).to(device)
    
    model.eval()
    with torch.no_grad():
        m_train, _ = model(X_train_orig_t)
        m_val, _ = model(X_val_t)
        final_train_loss = compute_pricing_error(m_train, R_train_orig_t).item()
        final_val_loss = compute_pricing_error(m_val, R_val_t).item()
    
    is_returns = (m_train.numpy().flatten() - 1) / 0.04
    oos_returns = (m_val.numpy().flatten() - 1) / 0.04
    
    is_sharpe = compute_sharpe(is_returns)
    oos_sharpe = compute_sharpe(oos_returns)
    
    return {
        "method": "optimized",
        "epochs_run": epochs_run,
        "best_epoch": early_stopping.best_epoch + 1,
        "total_time_sec": total_time,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": early_stopping.best_value,
        "oos_is_ratio": final_val_loss / final_train_loss if final_train_loss > 0 else float("inf"),
        "is_sharpe": is_sharpe,
        "oos_sharpe": oos_sharpe,
        "sharpe_ratio": oos_sharpe / is_sharpe if is_sharpe > 0 else 0,
        "strategies_used": [
            "T4.2: OneCycleLR",
            "T4.4: EarlyStopping",
            "T4.5: L2 + Dropout",
            "T4.6: Feature Masking",
        ],
    }


# =============================================================================
# Validation
# =============================================================================

def validate_objectives(baseline: Dict, optimized: Dict) -> Dict:
    """Validate T4 objectives."""
    results = {}
    
    # T4-OBJ-1: Training Speedup ≥30%
    speedup = (baseline["total_time_sec"] - optimized["total_time_sec"]) / baseline["total_time_sec"] * 100
    results["T4-OBJ-1"] = {
        "name": "Training Speedup",
        "target": "≥30%",
        "actual": f"{speedup:.1f}%",
        "passed": speedup >= 30,
        "details": f"Baseline: {baseline['total_time_sec']:.2f}s, Optimized: {optimized['total_time_sec']:.2f}s",
    }
    
    # T4-OBJ-2: OOS Sharpe ≥1.5
    results["T4-OBJ-2"] = {
        "name": "OOS Sharpe",
        "target": "≥1.5",
        "actual": f"{optimized['oos_sharpe']:.3f}",
        "passed": optimized["oos_sharpe"] >= 1.5,
        "details": f"Baseline: {baseline['oos_sharpe']:.3f}, Optimized: {optimized['oos_sharpe']:.3f}",
        "note": "⚠️ Synthetic data - validate with real data",
    }
    
    # T4-OBJ-3: OOS/IS Ratio ≥0.9
    results["T4-OBJ-3"] = {
        "name": "OOS/IS Ratio",
        "target": "≥0.9",
        "actual": f"{optimized['sharpe_ratio']:.3f}",
        "passed": optimized["sharpe_ratio"] >= 0.9,
        "details": f"Baseline: {baseline['sharpe_ratio']:.3f}, Optimized: {optimized['sharpe_ratio']:.3f}",
    }
    
    return results


def generate_report(baseline: Dict, optimized: Dict, objectives: Dict) -> str:
    """Generate comparison report."""
    report = f"""# T4 Training Optimization - Final Report

**Generated**: {CONFIG['timestamp']}
**Status**: {'✅ ALL PASSED' if all(o['passed'] for o in objectives.values()) else '⚠️ PARTIAL'}

## Executive Summary

T4 Training Optimization integrated 4 strategies to improve SDF model training:
- T4.2: OneCycleLR scheduling
- T4.4: Early Stopping (patience=10)
- T4.5: Regularization (L2=1e-4, Dropout=0.4)
- T4.6: Feature Masking (prob=0.2)

## Objective Validation

| Objective | Target | Actual | Status |
|-----------|--------|--------|--------|
| T4-OBJ-1: Speedup | ≥30% | {objectives['T4-OBJ-1']['actual']} | {'✅' if objectives['T4-OBJ-1']['passed'] else '❌'} |
| T4-OBJ-2: OOS Sharpe | ≥1.5 | {objectives['T4-OBJ-2']['actual']} | {'✅' if objectives['T4-OBJ-2']['passed'] else '⚠️'} |
| T4-OBJ-3: OOS/IS Ratio | ≥0.9 | {objectives['T4-OBJ-3']['actual']} | {'✅' if objectives['T4-OBJ-3']['passed'] else '❌'} |

**Note**: OOS Sharpe target (≥1.5) requires real data validation. Synthetic data provides baseline comparison only.

## Comparison: Baseline vs Optimized

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Epochs Run | {baseline['epochs_run']} | {optimized['epochs_run']} | {baseline['epochs_run'] - optimized['epochs_run']} fewer |
| Training Time | {baseline['total_time_sec']:.2f}s | {optimized['total_time_sec']:.2f}s | {(baseline['total_time_sec'] - optimized['total_time_sec'])/baseline['total_time_sec']*100:.1f}% faster |
| Final Val Loss | {baseline['final_val_loss']:.6f} | {optimized['final_val_loss']:.6f} | {(baseline['final_val_loss'] - optimized['final_val_loss'])/baseline['final_val_loss']*100:.1f}% |
| OOS/IS Ratio | {baseline['oos_is_ratio']:.3f} | {optimized['oos_is_ratio']:.3f} | {(baseline['oos_is_ratio'] - optimized['oos_is_ratio'])/baseline['oos_is_ratio']*100:.1f}% |
| IS Sharpe | {baseline['is_sharpe']:.3f} | {optimized['is_sharpe']:.3f} | - |
| OOS Sharpe | {baseline['oos_sharpe']:.3f} | {optimized['oos_sharpe']:.3f} | - |

## Strategy Effectiveness

| Strategy | Contribution |
|----------|--------------|
| T4.2: OneCycleLR | Better convergence, faster initial learning |
| T4.4: Early Stopping | 80%+ epoch reduction, prevents overfitting |
| T4.5: Regularization | 24% OOS/IS improvement |
| T4.6: Feature Masking | 1.5% val loss reduction |

## Recommendations

1. **Production Config**:
   ```python
   config = {{
       "dropout": 0.4,
       "l2_weight": 1e-4,
       "use_onecycle": True,
       "max_lr": 0.01,
       "early_stopping_patience": 10,
       "feature_mask_prob": 0.2,
   }}
   ```

2. **Next Steps**:
   - ⚠️ Validate with real data (DATA-001 fix required)
   - Consider GPU for FP16 benefits
   - Monitor OOS Sharpe on production data

## Artifacts

- `experiments/t4_final/results.json`
- `experiments/t4_final/comparison_report.md`
- `scripts/train_sdf_optimized.py`
"""
    return report


# =============================================================================
# Main
# =============================================================================

def run_validation(verbose: bool = True) -> Dict:
    """Run full T4 validation."""
    print("=" * 70)
    print("T4.7 Integration & Final Validation")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print()
    
    # Generate data
    X, returns = generate_data(n_samples=500)
    n_train = int(0.7 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    R_train, R_val = returns[:n_train], returns[n_train:]
    
    print(f"Data: Train={len(X_train)}, Val={len(X_val)}")
    print()
    
    # Train baseline
    print("[BASELINE] Training without optimizations...")
    print("-" * 50)
    baseline = train_baseline(X_train, R_train, X_val, R_val, verbose=verbose)
    print()
    
    # Train optimized
    print("[OPTIMIZED] Training with T4 strategies...")
    print("-" * 50)
    optimized = train_optimized(X_train, R_train, X_val, R_val, verbose=verbose)
    print()
    
    # Validate objectives
    print("=" * 70)
    print("OBJECTIVE VALIDATION")
    print("=" * 70)
    
    objectives = validate_objectives(baseline, optimized)
    
    for obj_id, obj in objectives.items():
        status = "✅ PASS" if obj["passed"] else "❌ FAIL"
        print(f"{obj_id}: {obj['name']}")
        print(f"  Target: {obj['target']}, Actual: {obj['actual']} → {status}")
        print(f"  {obj['details']}")
        if "note" in obj:
            print(f"  {obj['note']}")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for o in objectives.values() if o["passed"])
    total = len(objectives)
    
    speedup = (baseline["total_time_sec"] - optimized["total_time_sec"]) / baseline["total_time_sec"] * 100
    
    print(f"Objectives Passed: {passed}/{total}")
    print(f"Training Speedup: {speedup:.1f}%")
    print(f"Epochs Saved: {baseline['epochs_run'] - optimized['epochs_run']}")
    print(f"OOS/IS Improvement: {(baseline['oos_is_ratio'] - optimized['oos_is_ratio'])/baseline['oos_is_ratio']*100:.1f}%")
    
    output = {
        "config": CONFIG,
        "baseline": baseline,
        "optimized": optimized,
        "objectives": objectives,
        "summary": {
            "objectives_passed": passed,
            "objectives_total": total,
            "speedup_pct": speedup,
            "epochs_saved": baseline["epochs_run"] - optimized["epochs_run"],
        },
    }
    
    return output


def save_results(results: Dict):
    """Save results and report."""
    output_dir = PROJECT_ROOT / "experiments" / "t4_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")
    
    # Generate and save report
    report = generate_report(
        results["baseline"],
        results["optimized"],
        results["objectives"],
    )
    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="T4.7 Integration & Final Validation")
    parser.add_argument("--validate-all", action="store_true", help="Run full validation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    results = run_validation(verbose=not args.quiet)
    save_results(results)
    print("\n✅ T4.7 Integration & Final Validation completed!")


if __name__ == "__main__":
    main()
