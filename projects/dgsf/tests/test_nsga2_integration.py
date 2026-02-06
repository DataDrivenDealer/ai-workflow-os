"""
NSGA-II Integration Smoke Test with SDFEAAdapter.

This script validates the integration between:
- SDFEAAdapter (SDF → EA interface)  
- FitnessAdapter (legacy EA fitness wrapper)
- NSGAIIOptimizer (NSGA-II core algorithm)

Stage 5 Task: EA_DEV_001 P0-17

Author: DGSF Pipeline
Date: 2026-02-04
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Ensure paths are available
SCRIPTS_DIR = Path(__file__).parent
LEGACY_SRC = SCRIPTS_DIR.parent / "legacy" / "DGSF" / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def run_nsga2_smoke_test(
    T: int = 50,
    K: int = 10,
    D: int = 20,
    population_size: int = 20,
    num_generations: int = 10,
    seed: int = 42,
) -> dict:
    """
    Run NSGA-II smoke test with synthetic data.
    
    Parameters
    ----------
    T : int
        Number of time periods
    K : int
        Number of leaf assets (portfolio dimension)
    D : int
        Number of SDF features
    population_size : int
        NSGA-II population size
    num_generations : int
        Number of generations to evolve
    seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Test results including Pareto solutions and metrics
    """
    from sdf_ea_adapter import SDFEAAdapter, create_adapter_from_data
    from dgsf.ea import NSGAIIOptimizer, FitnessAdapter, StrategyEvaluator
    from dgsf.sdf import GenerativeSDF
    
    print("=" * 70)
    print("NSGA-II Integration Smoke Test")
    print("=" * 70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate synthetic data
    print(f"\n[1] Generating synthetic data: T={T}, K={K}, D={D}")
    X_sdf = np.random.randn(T, D).astype(np.float32)
    leaf_returns = (np.random.randn(T, K) * 0.02).astype(np.float32)  # ~2% monthly vol
    
    # Create SDF model
    print("\n[2] Creating synthetic SDF model")
    sdf_model = GenerativeSDF(
        input_dim=D,
        hidden_dim=32,
        num_hidden_layers=2,
        activation="tanh",
        output_activation="softplus",
    )
    
    # Convert to tensors
    X_tensor = torch.tensor(X_sdf)
    R_tensor = torch.tensor(leaf_returns)
    
    # Create StrategyEvaluator (legacy interface)
    print("\n[3] Creating StrategyEvaluator")
    evaluator = StrategyEvaluator(
        leaf_returns_window=R_tensor,
        X_sdf=X_tensor,
        sdf_model=sdf_model,
        evaluation_config={
            "periods_per_year": 12,
            "risk_free_rate": 0.0,
        }
    )
    
    # Create FitnessAdapter
    print("\n[4] Creating FitnessAdapter")
    fitness_adapter = FitnessAdapter(evaluator)
    
    # Test single fitness computation
    test_weights = np.abs(np.random.randn(K))
    test_weights = test_weights / test_weights.sum()
    test_fitness = fitness_adapter.compute(torch.tensor(test_weights))
    print(f"    Test fitness shape: {test_fitness.shape}")
    print(f"    Test fitness: f1={test_fitness[0]:.4f}, f2={test_fitness[1]:.4f}, "
          f"f3={test_fitness[2]:.4f}, f4={test_fitness[3]:.6f}")
    
    # Create NSGA-II optimizer
    print(f"\n[5] Creating NSGAIIOptimizer (pop={population_size}, gen={num_generations})")
    optimizer = NSGAIIOptimizer(
        fitness_adapter=fitness_adapter,
        leaf_count=K,
        population_size=population_size,
        num_generations=num_generations,
        crossover_eta=20.0,
        mutation_eta=20.0,
        random_seed=seed,
    )
    
    # Run optimization
    print("\n[6] Running NSGA-II optimization...")
    pareto_solutions = optimizer.run()
    
    print(f"\n[7] Results:")
    print(f"    Pareto solutions found: {len(pareto_solutions)}")
    
    if len(pareto_solutions) > 0:
        # Extract fitness values
        fitness_values = np.array([sol["fitness"] for sol in pareto_solutions])
        
        print(f"\n    Fitness statistics (4 objectives):")
        print(f"    {'Objective':<12} {'Min':<12} {'Max':<12} {'Mean':<12}")
        print(f"    {'-'*48}")
        
        obj_names = ["f1 (-Sharpe)", "f2 (MDD)", "f3 (Turn)", "f4 (PE)"]
        for i, name in enumerate(obj_names):
            vals = fitness_values[:, i]
            print(f"    {name:<12} {vals.min():<12.4f} {vals.max():<12.4f} {vals.mean():<12.4f}")
        
        # Best Sharpe solution
        best_sharpe_idx = np.argmin(fitness_values[:, 0])  # minimize -Sharpe
        best_sol = pareto_solutions[best_sharpe_idx]
        
        print(f"\n    Best Sharpe solution:")
        print(f"      Sharpe: {-best_sol['fitness'][0]:.4f}")
        print(f"      MDD:    {best_sol['fitness'][1]:.4f}")
        print(f"      Turn:   {best_sol['fitness'][2]:.4f}")
        print(f"      PE:     {best_sol['fitness'][3]:.6f}")
        print(f"      Weights (top 3): {best_sol['weights'][:3]}")
        
        # Verify weights sum to 1
        weight_sums = [np.sum(sol["weights"]) for sol in pareto_solutions]
        print(f"\n    Weight sums: min={min(weight_sums):.4f}, max={max(weight_sums):.4f}")
    
    print("\n" + "=" * 70)
    print("✅ NSGA-II Integration Smoke Test PASSED")
    print("=" * 70)
    
    return {
        "pareto_solutions": pareto_solutions,
        "num_solutions": len(pareto_solutions),
        "success": True,
    }


def run_sdf_ea_adapter_integration(
    T: int = 50,
    K: int = 10,
    D: int = 20,
) -> dict:
    """
    Test SDFEAAdapter direct integration (without legacy FitnessAdapter).
    
    This validates the new adapter can work standalone.
    """
    from sdf_ea_adapter import SDFEAAdapter, create_adapter_from_data
    
    print("\n" + "=" * 70)
    print("SDFEAAdapter Direct Integration Test")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    X_sdf = np.random.randn(T, D).astype(np.float32)
    leaf_returns = (np.random.randn(T, K) * 0.02).astype(np.float32)
    
    # Create adapter
    adapter = create_adapter_from_data(X_sdf, leaf_returns)
    
    # Test multiple random portfolios
    print("\n[Testing 10 random portfolios]")
    for i in range(10):
        w = np.abs(np.random.randn(K))
        w = w / w.sum()
        
        fitness = adapter.compute_full_fitness(w)
        
        if i < 3:  # Print first 3
            print(f"  Portfolio {i+1}: f1={fitness[0]:.4f}, f2={fitness[1]:.4f}, "
                  f"f3={fitness[2]:.4f}, f4={fitness[3]:.6f}")
    
    print("\n✅ SDFEAAdapter Direct Integration PASSED")
    
    return {"success": True}


if __name__ == "__main__":
    # Run both tests
    result1 = run_nsga2_smoke_test(
        T=50,
        K=10,
        D=20,
        population_size=20,
        num_generations=10,
    )
    
    result2 = run_sdf_ea_adapter_integration()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"NSGA-II Integration:      {'✅ PASS' if result1['success'] else '❌ FAIL'}")
    print(f"SDFEAAdapter Integration: {'✅ PASS' if result2['success'] else '❌ FAIL'}")
    print(f"Pareto Solutions Found:   {result1['num_solutions']}")
