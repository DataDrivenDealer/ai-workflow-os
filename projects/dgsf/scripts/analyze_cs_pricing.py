"""
T5.3 Cross-sectional Pricing Analysis (SDF_DEV_001_T5)

Purpose: Analyze cross-sectional pricing accuracy with Fama-MacBeth regression
and feature importance analysis.

Features:
1. Fama-MacBeth two-pass regression
2. Feature importance via permutation
3. Per-asset pricing error decomposition
4. Risk premium estimation

Reference: Fama & MacBeth (1973), Gu, Kelly, Xiu (2020)

Usage:
    python scripts/analyze_cs_pricing.py --fama-macbeth
    python scripts/analyze_cs_pricing.py --feature-importance
    python scripts/analyze_cs_pricing.py --full
    
Output:
    experiments/t5_cs_pricing/fama_macbeth_results.json
    experiments/t5_cs_pricing/feature_importance.json
    reports/sdf_cs_pricing_report.md
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

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
    "name": "T5.3 Cross-sectional Pricing Analysis",
    "timestamp": datetime.now().isoformat(),
    "data": {
        "n_periods": 120,
        "n_assets": 50,
        "n_features": 48,
        "seed": 42,
    },
    "model": {
        "input_dim": 48,
        "hidden_dim": 64,
        "num_hidden_layers": 3,
        "dropout": 0.4,
    },
    "fama_macbeth": {
        "min_periods": 24,  # Minimum periods for beta estimation
        "newey_west_lags": 6,  # Lags for Newey-West SE
    },
    "feature_importance": {
        "n_permutations": 10,
        "top_k": 10,
    },
}

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FamaMacBethResult:
    """Fama-MacBeth regression results."""
    # Risk premia estimates
    lambda_mean: List[float]
    lambda_std: List[float]
    lambda_tstat: List[float]
    
    # Model fit
    avg_r2: float
    avg_r2_adj: float
    
    # Time-series of cross-sectional R²
    r2_by_period: List[float]
    
    # Pricing errors
    avg_pricing_error: float
    pricing_error_by_asset: List[float]
    
    # Diagnostics
    n_periods: int
    n_assets: int
    n_factors: int
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeatureImportance:
    """Feature importance results."""
    feature_names: List[str]
    importance_scores: List[float]
    importance_std: List[float]
    ranking: List[int]
    
    # Top features
    top_k_features: List[str]
    top_k_scores: List[float]
    
    # Cumulative importance
    cumulative_importance: List[float]
    
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CSPricingAnalysis:
    """Complete cross-sectional pricing analysis."""
    fama_macbeth: Optional[FamaMacBethResult]
    feature_importance: Optional[FeatureImportance]
    
    # Summary metrics
    overall_r2: float
    overall_rmse: float
    significant_factors: int
    
    timestamp: str
    
    def to_dict(self) -> Dict:
        result = {
            "overall_r2": self.overall_r2,
            "overall_rmse": self.overall_rmse,
            "significant_factors": self.significant_factors,
            "timestamp": self.timestamp,
        }
        if self.fama_macbeth:
            result["fama_macbeth"] = self.fama_macbeth.to_dict()
        if self.feature_importance:
            result["feature_importance"] = self.feature_importance.to_dict()
        return result


# =============================================================================
# Model
# =============================================================================

class SDFModel(nn.Module):
    """SDF model for cross-sectional pricing."""
    
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
        for i in range(num_hidden_layers):
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
# Data Generation
# =============================================================================

def generate_factor_data(
    n_periods: int = 120,
    n_assets: int = 50,
    n_features: int = 48,
    n_factors: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data with known factor structure.
    
    Returns:
        X: Features (T, F)
        R: Returns (T, N)
        betas: True factor loadings (N, K)
        factors: Factor returns (T, K)
    """
    np.random.seed(seed)
    
    # Features with some structure
    X = np.random.randn(n_periods, n_features).astype(np.float32)
    for t in range(1, n_periods):
        X[t] = 0.3 * X[t-1] + 0.7 * X[t]
    
    # True factor loadings (betas)
    betas = np.random.randn(n_assets, n_factors).astype(np.float32)
    betas = betas * np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # Decreasing importance
    
    # Factor returns with risk premia
    risk_premia = np.array([0.008, 0.005, 0.003, 0.002, 0.001])  # Monthly
    factors = np.random.randn(n_periods, n_factors) * 0.03 + risk_premia
    factors = factors.astype(np.float32)
    
    # Returns = betas @ factors.T + idiosyncratic
    idio_vol = 0.05
    idio = np.random.randn(n_periods, n_assets) * idio_vol
    R = factors @ betas.T + idio
    R = R.astype(np.float32)
    
    return X, R, betas, factors


def get_feature_names(n_features: int = 48) -> List[str]:
    """Generate feature names."""
    categories = [
        ("momentum", 6),
        ("value", 6),
        ("size", 4),
        ("volatility", 6),
        ("liquidity", 6),
        ("quality", 6),
        ("sentiment", 4),
        ("macro", 5),
        ("technical", 5),
    ]
    
    names = []
    for cat, count in categories:
        for i in range(count):
            names.append(f"{cat}_{i+1}")
    
    # Pad if needed
    while len(names) < n_features:
        names.append(f"feature_{len(names)+1}")
    
    return names[:n_features]


# =============================================================================
# Fama-MacBeth Regression
# =============================================================================

def ols_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Simple OLS regression."""
    X_aug = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(X_aug.shape[1])
    
    y_pred = X_aug @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return beta, max(0, min(1, r2))


def newey_west_se(
    residuals: np.ndarray,
    X: np.ndarray,
    lags: int = 6,
) -> np.ndarray:
    """Compute Newey-West standard errors."""
    T = len(residuals)
    k = X.shape[1]
    
    # Compute meat of sandwich
    S = np.zeros((k, k))
    
    for l in range(lags + 1):
        weight = 1 - l / (lags + 1) if l > 0 else 1
        
        for t in range(l, T):
            x_t = X[t].reshape(-1, 1)
            x_tl = X[t-l].reshape(-1, 1)
            S += weight * residuals[t] * residuals[t-l] * (x_t @ x_tl.T + x_tl @ x_t.T) / 2
    
    S = S / T
    
    # Bread
    XtX = X.T @ X / T
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.eye(k)
    
    # Sandwich
    var_beta = XtX_inv @ S @ XtX_inv / T
    se = np.sqrt(np.diag(var_beta))
    
    return se


def run_fama_macbeth(
    R: np.ndarray,
    betas: np.ndarray,
    verbose: bool = True,
) -> FamaMacBethResult:
    """
    Run Fama-MacBeth two-pass regression.
    
    Pass 1: Estimate betas (given in this synthetic case)
    Pass 2: Cross-sectional regression each period
    
    Args:
        R: Returns (T, N)
        betas: Factor loadings (N, K)
    
    Returns:
        FamaMacBethResult
    """
    T, N = R.shape
    K = betas.shape[1]
    
    if verbose:
        print(f"Fama-MacBeth: {T} periods, {N} assets, {K} factors")
    
    # Pass 2: Cross-sectional regressions
    lambdas = []
    r2_list = []
    
    for t in range(T):
        r_t = R[t]  # N returns
        
        # Regress returns on betas
        beta_aug = np.column_stack([np.ones(N), betas])
        lambda_t, r2_t = ols_regression(betas, r_t)
        
        # lambda_t[0] is intercept (alpha), lambda_t[1:] are risk premia
        lambdas.append(lambda_t)
        r2_list.append(r2_t)
    
    lambdas = np.array(lambdas)  # (T, K+1)
    
    # Time-series averages
    lambda_mean = np.mean(lambdas, axis=0)
    lambda_std = np.std(lambdas, axis=0, ddof=1)
    
    # t-statistics (simple, could use Newey-West)
    lambda_tstat = lambda_mean / (lambda_std / np.sqrt(T) + 1e-10)
    
    # Pricing errors
    avg_r2 = np.mean(r2_list)
    avg_r2_adj = 1 - (1 - avg_r2) * (N - 1) / (N - K - 1)
    
    # Per-asset pricing error
    mean_r = np.mean(R, axis=0)
    predicted_r = betas @ lambda_mean[1:]  # Exclude intercept
    pricing_errors = mean_r - predicted_r
    avg_pricing_error = np.mean(np.abs(pricing_errors))
    
    if verbose:
        print(f"\nRisk Premia Estimates:")
        print(f"{'Factor':<12} {'Lambda':>10} {'Std':>10} {'t-stat':>10}")
        print("-" * 44)
        print(f"{'Intercept':<12} {lambda_mean[0]:>10.4f} {lambda_std[0]:>10.4f} {lambda_tstat[0]:>10.2f}")
        for k in range(K):
            sig = "***" if abs(lambda_tstat[k+1]) > 2.58 else "**" if abs(lambda_tstat[k+1]) > 1.96 else "*" if abs(lambda_tstat[k+1]) > 1.64 else ""
            print(f"{'Factor_'+str(k+1):<12} {lambda_mean[k+1]:>10.4f} {lambda_std[k+1]:>10.4f} {lambda_tstat[k+1]:>10.2f} {sig}")
        print(f"\nAvg Cross-sectional R²: {avg_r2:.4f}")
        print(f"Avg Pricing Error: {avg_pricing_error:.6f}")
    
    return FamaMacBethResult(
        lambda_mean=lambda_mean.tolist(),
        lambda_std=lambda_std.tolist(),
        lambda_tstat=lambda_tstat.tolist(),
        avg_r2=float(avg_r2),
        avg_r2_adj=float(avg_r2_adj),
        r2_by_period=r2_list,
        avg_pricing_error=float(avg_pricing_error),
        pricing_error_by_asset=pricing_errors.tolist(),
        n_periods=T,
        n_assets=N,
        n_factors=K,
        timestamp=datetime.now().isoformat(),
    )


# =============================================================================
# Feature Importance
# =============================================================================

def compute_feature_importance(
    model: nn.Module,
    X: np.ndarray,
    R: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 10,
    verbose: bool = True,
) -> FeatureImportance:
    """
    Compute feature importance via permutation.
    
    Args:
        model: Trained SDF model
        X: Features (T, F)
        R: Returns (T, N)
        feature_names: Feature names
        n_permutations: Number of permutations per feature
    
    Returns:
        FeatureImportance
    """
    model.eval()
    T, F = X.shape
    
    if verbose:
        print(f"Computing feature importance: {F} features, {n_permutations} permutations")
    
    # Baseline loss
    X_t = torch.from_numpy(X).float()
    R_mean = torch.from_numpy(R.mean(axis=1)).float()
    
    with torch.no_grad():
        m_base = model(X_t).squeeze()
        base_loss = torch.mean(((1 + R_mean) * m_base - 1) ** 2).item()
    
    # Permutation importance
    importance_scores = []
    importance_std = []
    
    for f in range(F):
        losses = []
        for _ in range(n_permutations):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, f])
            
            X_perm_t = torch.from_numpy(X_perm).float()
            with torch.no_grad():
                m_perm = model(X_perm_t).squeeze()
                perm_loss = torch.mean(((1 + R_mean) * m_perm - 1) ** 2).item()
            
            losses.append(perm_loss - base_loss)
        
        importance_scores.append(np.mean(losses))
        importance_std.append(np.std(losses))
    
    # Ranking (higher importance = higher rank)
    ranking = np.argsort(importance_scores)[::-1].tolist()
    
    # Top-k features
    top_k = CONFIG["feature_importance"]["top_k"]
    top_k_indices = ranking[:top_k]
    top_k_features = [feature_names[i] for i in top_k_indices]
    top_k_scores = [importance_scores[i] for i in top_k_indices]
    
    # Cumulative importance
    sorted_scores = [importance_scores[i] for i in ranking]
    total = sum(max(0, s) for s in sorted_scores)
    cumulative = []
    cum_sum = 0
    for s in sorted_scores:
        cum_sum += max(0, s)
        cumulative.append(cum_sum / total if total > 0 else 0)
    
    if verbose:
        print(f"\nTop {top_k} Features:")
        print(f"{'Rank':<6} {'Feature':<20} {'Importance':>12} {'Std':>10}")
        print("-" * 50)
        for i, (name, score) in enumerate(zip(top_k_features, top_k_scores)):
            print(f"{i+1:<6} {name:<20} {score:>12.6f} {importance_std[ranking[i]]:>10.6f}")
    
    return FeatureImportance(
        feature_names=feature_names,
        importance_scores=importance_scores,
        importance_std=importance_std,
        ranking=ranking,
        top_k_features=top_k_features,
        top_k_scores=top_k_scores,
        cumulative_importance=cumulative,
        timestamp=datetime.now().isoformat(),
    )


# =============================================================================
# Training
# =============================================================================

def train_model(
    X: np.ndarray,
    R: np.ndarray,
    epochs: int = 50,
    verbose: bool = False,
) -> nn.Module:
    """Quick training for analysis."""
    model = SDFModel(
        input_dim=X.shape[1],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        num_hidden_layers=CONFIG["model"]["num_hidden_layers"],
        dropout=CONFIG["model"]["dropout"],
        seed=42,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    X_t = torch.from_numpy(X).float()
    R_mean = torch.from_numpy(R.mean(axis=1)).float()
    
    model.train()
    for epoch in range(epochs):
        m = model(X_t).squeeze()
        loss = torch.mean(((1 + R_mean) * m - 1) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    return model


# =============================================================================
# Report Generation
# =============================================================================

def generate_cs_pricing_report(
    analysis: CSPricingAnalysis,
    output_path: Path,
) -> str:
    """Generate cross-sectional pricing report."""
    
    # Fama-MacBeth section
    fm_section = ""
    if analysis.fama_macbeth:
        fm = analysis.fama_macbeth
        
        # Risk premia table
        risk_premia_rows = []
        risk_premia_rows.append(f"| Intercept | {fm.lambda_mean[0]:.4f} | {fm.lambda_std[0]:.4f} | {fm.lambda_tstat[0]:.2f} | {'✓' if abs(fm.lambda_tstat[0]) > 1.96 else ''} |")
        for k in range(fm.n_factors):
            sig = "✓" if abs(fm.lambda_tstat[k+1]) > 1.96 else ""
            risk_premia_rows.append(
                f"| Factor_{k+1} | {fm.lambda_mean[k+1]:.4f} | {fm.lambda_std[k+1]:.4f} | {fm.lambda_tstat[k+1]:.2f} | {sig} |"
            )
        risk_premia_table = "\n".join(risk_premia_rows)
        
        fm_section = f"""
## Fama-MacBeth Regression Results

### Risk Premium Estimates

| Factor | Lambda (Mean) | Std | t-stat | Sig. (5%) |
|--------|---------------|-----|--------|-----------|
{risk_premia_table}

### Model Fit

| Metric | Value |
|--------|-------|
| Average Cross-sectional R² | {fm.avg_r2:.4f} |
| Adjusted R² | {fm.avg_r2_adj:.4f} |
| Average Pricing Error | {fm.avg_pricing_error:.6f} |
| Periods | {fm.n_periods} |
| Assets | {fm.n_assets} |
| Factors | {fm.n_factors} |

### Interpretation

The Fama-MacBeth regression decomposes expected returns into factor risk premia:

$$E[R_i] = \\alpha + \\sum_{{k=1}}^K \\beta_{{ik}} \\lambda_k$$

Where $\\lambda_k$ is the risk premium for factor $k$.

**Key Findings:**
- {sum(1 for t in fm.lambda_tstat[1:] if abs(t) > 1.96)} of {fm.n_factors} factors have significant risk premia at 5% level
- Cross-sectional R² of {fm.avg_r2:.1%} indicates {'strong' if fm.avg_r2 > 0.5 else 'moderate' if fm.avg_r2 > 0.3 else 'weak'} factor model fit
"""
    
    # Feature importance section
    fi_section = ""
    if analysis.feature_importance:
        fi = analysis.feature_importance
        
        # Top features table
        top_features_rows = []
        for i, (name, score) in enumerate(zip(fi.top_k_features[:10], fi.top_k_scores[:10])):
            top_features_rows.append(f"| {i+1} | {name} | {score:.6f} | {fi.cumulative_importance[i]:.1%} |")
        top_features_table = "\n".join(top_features_rows)
        
        fi_section = f"""
## Feature Importance Analysis

### Top 10 Features (Permutation Importance)

| Rank | Feature | Importance | Cumulative |
|------|---------|------------|------------|
{top_features_table}

### Feature Categories Distribution

Based on permutation importance, the most impactful feature categories are:
1. Features with highest individual importance indicate direct pricing relevance
2. Cumulative importance shows how quickly pricing power is captured

**Concentration**: Top 5 features explain {fi.cumulative_importance[4]:.1%} of pricing variation
"""
    
    report = f"""# SDF Cross-sectional Pricing Analysis Report

**Generated**: {analysis.timestamp}
**Method**: Fama-MacBeth Two-Pass Regression + Permutation Importance

## Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Overall R² | {analysis.overall_r2:.4f} | {'✅' if analysis.overall_r2 > 0.5 else '⚠️'} |
| Overall RMSE | {analysis.overall_rmse:.6f} | {'✅' if analysis.overall_rmse < 0.01 else '⚠️'} |
| Significant Factors | {analysis.significant_factors} | - |

{fm_section}

{fi_section}

## Recommendations

### For SDF Model Improvement
1. **Focus on significant factors**: Factors with |t-stat| > 1.96 carry pricing information
2. **Feature selection**: Top features by permutation importance should be retained
3. **Regularization**: Features with low importance may be dropped to reduce overfitting

### For Production Use
1. Re-validate with out-of-sample data
2. Monitor factor premium stability over time
3. Consider regime-dependent factor models

---
*Report generated by T5.3 Cross-sectional Pricing Analysis*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report


# =============================================================================
# Main Analysis
# =============================================================================

def run_full_analysis(
    run_fama_macbeth_flag: bool = True,
    run_feature_importance_flag: bool = True,
    verbose: bool = True,
) -> CSPricingAnalysis:
    """Run complete cross-sectional pricing analysis."""
    print("=" * 70)
    print("T5.3 Cross-sectional Pricing Analysis")
    print("=" * 70)
    print(f"Timestamp: {CONFIG['timestamp']}")
    print()
    
    # Generate data
    X, R, betas, factors = generate_factor_data(
        n_periods=CONFIG["data"]["n_periods"],
        n_assets=CONFIG["data"]["n_assets"],
        n_features=CONFIG["data"]["n_features"],
        seed=CONFIG["data"]["seed"],
    )
    feature_names = get_feature_names(CONFIG["data"]["n_features"])
    
    print(f"Data: {X.shape[0]} periods, {X.shape[1]} features, {R.shape[1]} assets")
    print(f"Factors: {betas.shape[1]} true factors")
    print()
    
    # Train model for feature importance
    if run_feature_importance_flag:
        print("Training SDF model...")
        model = train_model(X, R, epochs=50, verbose=False)
        print("Training complete.\n")
    
    # Fama-MacBeth
    fm_result = None
    if run_fama_macbeth_flag:
        print("-" * 70)
        print("FAMA-MACBETH REGRESSION")
        print("-" * 70)
        fm_result = run_fama_macbeth(R, betas, verbose=verbose)
        print()
    
    # Feature Importance
    fi_result = None
    if run_feature_importance_flag:
        print("-" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("-" * 70)
        fi_result = compute_feature_importance(
            model, X, R, feature_names,
            n_permutations=CONFIG["feature_importance"]["n_permutations"],
            verbose=verbose,
        )
        print()
    
    # Summary
    overall_r2 = fm_result.avg_r2 if fm_result else 0.0
    overall_rmse = fm_result.avg_pricing_error if fm_result else 0.0
    significant_factors = sum(1 for t in fm_result.lambda_tstat[1:] if abs(t) > 1.96) if fm_result else 0
    
    analysis = CSPricingAnalysis(
        fama_macbeth=fm_result,
        feature_importance=fi_result,
        overall_r2=overall_r2,
        overall_rmse=overall_rmse,
        significant_factors=significant_factors,
        timestamp=datetime.now().isoformat(),
    )
    
    # Save results
    output_dir = PROJECT_ROOT / "experiments" / "t5_cs_pricing"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON results
    if fm_result:
        fm_path = output_dir / "fama_macbeth_results.json"
        with open(fm_path, "w") as f:
            json.dump(fm_result.to_dict(), f, indent=2)
        print(f"Fama-MacBeth results saved to: {fm_path}")
    
    if fi_result:
        fi_path = output_dir / "feature_importance.json"
        with open(fi_path, "w") as f:
            json.dump(fi_result.to_dict(), f, indent=2)
        print(f"Feature importance saved to: {fi_path}")
    
    # Markdown report
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "sdf_cs_pricing_report.md"
    generate_cs_pricing_report(analysis, report_path)
    print(f"Report saved to: {report_path}")
    
    # Final summary
    print()
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Overall Cross-sectional R²: {overall_r2:.4f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print(f"Significant Factors: {significant_factors}")
    if fi_result:
        print(f"Top Feature: {fi_result.top_k_features[0]} (score: {fi_result.top_k_scores[0]:.6f})")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="T5.3 Cross-sectional Pricing Analysis")
    parser.add_argument("--fama-macbeth", action="store_true", help="Run Fama-MacBeth regression only")
    parser.add_argument("--feature-importance", action="store_true", help="Run feature importance only")
    parser.add_argument("--full", action="store_true", help="Run full analysis")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Default to full if no specific flag
    run_fm = args.fama_macbeth or args.full or (not args.fama_macbeth and not args.feature_importance)
    run_fi = args.feature_importance or args.full or (not args.fama_macbeth and not args.feature_importance)
    
    analysis = run_full_analysis(
        run_fama_macbeth_flag=run_fm,
        run_feature_importance_flag=run_fi,
        verbose=not args.quiet,
    )
    
    print("\n✅ T5.3 Cross-sectional Pricing Analysis completed!")


if __name__ == "__main__":
    main()
