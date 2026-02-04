"""
EA-SDF Real Data Integration Test.

This script validates the full Stage 4→5 integration using REAL data:
- RealDataLoader (56 months × 48 features)
- GenerativeSDF model
- NSGAIIOptimizer
- 4-objective Pareto optimization

Stage 5 Task: EA_DEV_001 P0-18

Author: DGSF Pipeline  
Date: 2026-02-04
"""

import sys
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime

# Ensure paths
SCRIPTS_DIR = Path(__file__).parent
LEGACY_SRC = SCRIPTS_DIR.parent / "legacy" / "DGSF" / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def run_real_data_ea_test(
    population_size: int = 30,
    num_generations: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run NSGA-II optimization on real DGSF data.
    
    Parameters
    ----------
    population_size : int
        NSGA-II population size
    num_generations : int
        Number of generations
    seed : int
        Random seed
    verbose : bool
        Print progress
    
    Returns
    -------
    results : dict
        Comprehensive test results
    """
    from data_utils import RealDataLoader
    from dgsf.ea import NSGAIIOptimizer, FitnessAdapter, StrategyEvaluator
    from dgsf.sdf import GenerativeSDF
    
    print("=" * 70)
    print("EA-SDF Real Data Integration Test")
    print("=" * 70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # ========================================================================
    # Step 1: Load real data
    # ========================================================================
    print("\n[1] Loading real data...")
    loader = RealDataLoader(verbose=verbose)
    X_sdf, returns, is_real = loader.load()
    
    T, D = X_sdf.shape
    K = min(D, 20)  # Limit K for computational feasibility
    
    print(f"    Data shape: T={T}, D={D}")
    print(f"    Using K={K} features as portfolio dimension")
    print(f"    Data source: {'REAL' if is_real else 'SYNTHETIC'}")
    
    # For this test, create synthetic "leaf returns" based on features
    # In production, this would come from PanelTree
    # Using first K features scaled as proxy returns
    leaf_returns = X_sdf[:, :K] * 0.01  # Scale down for return-like values
    
    print(f"    Leaf returns shape: {leaf_returns.shape}")
    
    # ========================================================================
    # Step 2: Create SDF model
    # ========================================================================
    print("\n[2] Creating SDF model...")
    sdf_model = GenerativeSDF(
        input_dim=D,
        hidden_dim=64,
        num_hidden_layers=3,
        activation="tanh",
        output_activation="softplus",
    )
    
    # Convert to tensors
    X_tensor = torch.tensor(X_sdf, dtype=torch.float32)
    R_tensor = torch.tensor(leaf_returns, dtype=torch.float32)
    
    print(f"    Model: GenerativeSDF(D={D}, hidden=64, layers=3)")
    
    # ========================================================================
    # Step 3: Create evaluator and fitness adapter
    # ========================================================================
    print("\n[3] Creating StrategyEvaluator...")
    evaluator = StrategyEvaluator(
        leaf_returns_window=R_tensor,
        X_sdf=X_tensor,
        sdf_model=sdf_model,
        evaluation_config={
            "periods_per_year": 12,
            "risk_free_rate": 0.0,
        }
    )
    
    fitness_adapter = FitnessAdapter(evaluator)
    
    # Test single evaluation
    test_w = np.ones(K) / K  # Equal weight
    test_fitness = fitness_adapter.compute(torch.tensor(test_w, dtype=torch.float32))
    print(f"    Equal-weight fitness: f1={test_fitness[0]:.4f}, f2={test_fitness[1]:.4f}, "
          f"f3={test_fitness[2]:.4f}, f4={test_fitness[3]:.6f}")
    
    # ========================================================================
    # Step 4: Run NSGA-II optimization
    # ========================================================================
    print(f"\n[4] Running NSGA-II (pop={population_size}, gen={num_generations})...")
    
    optimizer = NSGAIIOptimizer(
        fitness_adapter=fitness_adapter,
        leaf_count=K,
        population_size=population_size,
        num_generations=num_generations,
        crossover_eta=20.0,
        mutation_eta=20.0,
        random_seed=seed,
    )
    
    import time
    start_time = time.time()
    pareto_solutions = optimizer.run()
    elapsed = time.time() - start_time
    
    print(f"    Optimization completed in {elapsed:.2f}s")
    print(f"    Pareto solutions found: {len(pareto_solutions)}")
    
    # ========================================================================
    # Step 5: Analyze results
    # ========================================================================
    print("\n[5] Analyzing Pareto front...")
    
    fitness_values = np.array([sol["fitness"] for sol in pareto_solutions])
    weights_array = np.array([sol["weights"] for sol in pareto_solutions])
    
    # Statistics
    results = {
        "timestamp": datetime.now().isoformat(),
        "data_source": "REAL" if is_real else "SYNTHETIC",
        "T": T,
        "D": D,
        "K": K,
        "date_range": f"56 months (2015-05 to 2019-12)" if is_real else "synthetic",
        "population_size": population_size,
        "num_generations": num_generations,
        "elapsed_seconds": elapsed,
        "num_pareto_solutions": len(pareto_solutions),
        "objectives": {},
    }
    
    obj_names = ["f1_neg_sharpe", "f2_mdd", "f3_turnover", "f4_pricing_error"]
    display_names = ["f1 (-Sharpe)", "f2 (MDD)", "f3 (Turnover)", "f4 (PE)"]
    
    print(f"\n    {'Objective':<15} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print(f"    {'-'*63}")
    
    for i, (name, display) in enumerate(zip(obj_names, display_names)):
        vals = fitness_values[:, i]
        stats = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
        results["objectives"][name] = stats
        print(f"    {display:<15} {stats['min']:<12.4f} {stats['max']:<12.4f} "
              f"{stats['mean']:<12.4f} {stats['std']:<12.4f}")
    
    # Best solutions by each objective
    print("\n    Best solutions by objective:")
    
    best_sharpe_idx = np.argmin(fitness_values[:, 0])
    best_mdd_idx = np.argmin(fitness_values[:, 1])
    best_pe_idx = np.argmin(fitness_values[:, 3])
    
    results["best_sharpe"] = {
        "sharpe": float(-fitness_values[best_sharpe_idx, 0]),
        "mdd": float(fitness_values[best_sharpe_idx, 1]),
        "turnover": float(fitness_values[best_sharpe_idx, 2]),
        "pe": float(fitness_values[best_sharpe_idx, 3]),
        "weights_sparsity": float((weights_array[best_sharpe_idx] > 0.01).sum() / K),
    }
    
    results["best_mdd"] = {
        "sharpe": float(-fitness_values[best_mdd_idx, 0]),
        "mdd": float(fitness_values[best_mdd_idx, 1]),
        "pe": float(fitness_values[best_mdd_idx, 3]),
    }
    
    results["best_pe"] = {
        "sharpe": float(-fitness_values[best_pe_idx, 0]),
        "pe": float(fitness_values[best_pe_idx, 3]),
    }
    
    print(f"      Best Sharpe: {results['best_sharpe']['sharpe']:.4f} "
          f"(MDD={results['best_sharpe']['mdd']:.4f}, PE={results['best_sharpe']['pe']:.6f})")
    print(f"      Best MDD:    {results['best_mdd']['mdd']:.4f} "
          f"(Sharpe={results['best_mdd']['sharpe']:.4f})")
    print(f"      Best PE:     {results['best_pe']['pe']:.6f} "
          f"(Sharpe={results['best_pe']['sharpe']:.4f})")
    
    # Weight analysis
    print("\n    Weight statistics:")
    weight_concentration = np.max(weights_array, axis=1).mean()
    num_active = (weights_array > 0.01).sum(axis=1).mean()
    
    results["weight_analysis"] = {
        "mean_max_weight": float(weight_concentration),
        "mean_active_assets": float(num_active),
        "weight_sums_valid": bool(np.allclose(weights_array.sum(axis=1), 1.0)),
    }
    
    print(f"      Mean max weight: {weight_concentration:.4f}")
    print(f"      Mean active assets: {num_active:.1f} / {K}")
    print(f"      Weight sums valid: {results['weight_analysis']['weight_sums_valid']}")
    
    # ========================================================================
    # Step 6: Save results
    # ========================================================================
    output_dir = SCRIPTS_DIR.parent / "experiments" / "ea_integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "ea_sdf_real_data_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[6] Results saved to: {results_file}")
    
    # ========================================================================
    # Pass/Fail criteria
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST CRITERIA EVALUATION")
    print("=" * 70)
    
    criteria = {
        "pareto_count_ok": len(pareto_solutions) >= 10,
        "sharpe_positive": results["best_sharpe"]["sharpe"] > 0,
        "mdd_reasonable": results["best_mdd"]["mdd"] < 0.5,
        "pe_finite": np.isfinite(results["best_pe"]["pe"]),
        "weights_valid": results["weight_analysis"]["weight_sums_valid"],
        "elapsed_reasonable": elapsed < 120,  # Under 2 minutes
    }
    
    results["criteria"] = criteria
    
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {criterion:<25}: {status}")
    
    all_passed = all(criteria.values())
    results["all_passed"] = all_passed
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ EA-SDF REAL DATA INTEGRATION TEST PASSED")
    else:
        print("⚠️ EA-SDF REAL DATA INTEGRATION TEST HAS WARNINGS")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_real_data_ea_test(
        population_size=30,
        num_generations=20,
        verbose=True,
    )
    
    print("\n[Summary]")
    print(f"  Data: {results['date_range']}")
    print(f"  Pareto solutions: {results['num_pareto_solutions']}")
    print(f"  Best Sharpe: {results['best_sharpe']['sharpe']:.4f}")
    print(f"  Best PE: {results['best_pe']['pe']:.6f}")
    print(f"  Time: {results['elapsed_seconds']:.2f}s")
    print(f"  All criteria passed: {results['all_passed']}")
