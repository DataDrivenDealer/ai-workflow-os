"""
P0-21 Drift-Aware Warm-Start - Integration Test with Real Data

Verifies that the drift-aware warm-start module works correctly with:
1. Real Pareto front from P0-18 EA integration test
2. Simulated leaf space changes (window transitions)
3. Strategy drift calculation across windows

v3.1 Requirements Verified:
- 4.2.1 MUST: Transform and use previous Pareto solutions
- 4.2.2 MUST: Combine with random individuals
- 4.2.3 MUST: Record strategy drift
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add scripts directory to path
_scripts_dir = Path(__file__).parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from drift_aware_warmstart import (
    DriftAwareWarmStart,
    transform_pareto_to_new_leaf_space,
    compute_strategy_drift,
    compute_batch_drift,
    create_random_population,
)


def load_p018_results() -> Dict[str, Any]:
    """Load P0-18 real data test results if available."""
    results_path = Path(__file__).parent.parent / "experiments" / "ea_integration" / "p018_ea_results.json"
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    
    # If not available, create synthetic data matching P0-18 format
    print("[INFO] P0-18 results not found, using synthetic data")
    return create_synthetic_pareto_data()


def create_synthetic_pareto_data() -> Dict[str, Any]:
    """Create synthetic Pareto front data matching P0-18 format."""
    np.random.seed(42)
    n_solutions = 29  # Match P0-18
    n_leaves = 10
    
    pareto_front = []
    for i in range(n_solutions):
        # Generate random weights (Dirichlet for valid portfolio)
        weights = np.random.dirichlet(np.ones(n_leaves))
        
        solution = {
            'weights': weights.tolist(),
            'objectives': {
                'sharpe': float(np.random.uniform(0.5, 2.0)),
                'mdd': float(np.random.uniform(0.05, 0.25)),
                'turnover': float(np.random.uniform(0.05, 0.2)),
                'sdf_penalty': float(np.random.uniform(0.001, 0.005)),
            },
            'leaf_indices': list(range(n_leaves)),
        }
        pareto_front.append(solution)
    
    return {
        'pareto_front': pareto_front,
        'config': {
            'pop_size': 100,
            'n_gen': 50,
            'n_obj': 4,
        },
        'metrics': {
            'hypervolume': 2.09,
            'n_solutions': n_solutions,
        }
    }


def simulate_window_transition(n_prev_leaves: int, n_curr_leaves: int) -> Dict[int, int]:
    """
    Simulate leaf space change between rolling windows.
    
    In real DGSF, leaves can:
    - Map directly (index shift)
    - Disappear (no mapping)
    - Be new (not in previous)
    """
    np.random.seed(123)
    mapping = {}
    
    # Simulate that 80% of leaves persist with shuffled indices
    persist_count = int(0.8 * min(n_prev_leaves, n_curr_leaves))
    curr_indices = list(range(n_curr_leaves))
    np.random.shuffle(curr_indices)
    
    for i in range(n_prev_leaves):
        if i < persist_count:
            mapping[i] = curr_indices[i] if i < len(curr_indices) else None
        else:
            mapping[i] = None  # Leaf disappeared
    
    return mapping


def run_integration_test():
    """Run P0-21 integration test with real/synthetic data."""
    print("=" * 60)
    print("P0-21: Drift-Aware Warm-Start - Integration Test")
    print("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # Step 1: Load P0-18 Pareto front
    print("\n[Step 1] Load Previous Pareto Front...")
    p018_data = load_p018_results()
    pareto_front = p018_data.get('pareto_front', [])
    
    # Convert list weights to numpy arrays
    for sol in pareto_front:
        if isinstance(sol.get('weights'), list):
            sol['weights'] = np.array(sol['weights'])
    
    n_solutions = len(pareto_front)
    n_prev_leaves = len(pareto_front[0]['weights']) if pareto_front else 10
    print(f"  Pareto solutions: {n_solutions}")
    print(f"  Previous leaves: {n_prev_leaves}")
    print("  ✅ Pareto front loaded")
    
    # Step 2: Simulate window transition
    print("\n[Step 2] Simulate Window Transition...")
    n_curr_leaves = n_prev_leaves + 2  # New window has 2 more leaves
    leaf_mapping = simulate_window_transition(n_prev_leaves, n_curr_leaves)
    
    mapped_count = sum(1 for v in leaf_mapping.values() if v is not None)
    unmapped_count = n_prev_leaves - mapped_count
    print(f"  Previous leaves: {n_prev_leaves}")
    print(f"  Current leaves: {n_curr_leaves}")
    print(f"  Mapped: {mapped_count}, Unmapped: {unmapped_count}")
    print("  ✅ Window transition simulated")
    
    # Step 3: Create warm-start population
    print("\n[Step 3] Create Warm-Start Population...")
    config = {'warm_start_ratio': 0.5}
    warm_start = DriftAwareWarmStart(config=config)
    
    pop_size = 100
    population = warm_start.create_population(
        previous_pareto=pareto_front,
        leaf_mapping=leaf_mapping,
        new_leaf_count=n_curr_leaves,
        pop_size=pop_size
    )
    
    print(f"  Population size: {len(population)}")
    
    warm_count = sum(1 for ind in population if ind.get('source') == 'warm_start')
    random_count = sum(1 for ind in population if ind.get('source') == 'random')
    print(f"  Warm-start: {warm_count} ({warm_count/pop_size*100:.0f}%)")
    print(f"  Random: {random_count} ({random_count/pop_size*100:.0f}%)")
    print("  ✅ Population created")
    results['population_size'] = len(population)
    results['warm_start_ratio'] = warm_count / pop_size
    
    # Step 4: Verify warm-start ratio in valid range
    print("\n[Step 4] Verify Warm-Start Ratio (40-60%)...")
    ratio_valid = 0.4 <= results['warm_start_ratio'] <= 0.6
    print(f"  Ratio: {results['warm_start_ratio']:.2f}")
    print(f"  Valid: {ratio_valid}")
    results['ratio_valid'] = ratio_valid
    
    # Step 5: Calculate strategy drift
    print("\n[Step 5] Calculate Strategy Drift...")
    
    # Simulate 3 windows of best solution weights
    window_weights = [
        np.array([0.15, 0.25, 0.30, 0.20, 0.10]),  # Window j-2
        np.array([0.20, 0.25, 0.25, 0.20, 0.10]),  # Window j-1
        np.array([0.25, 0.20, 0.25, 0.15, 0.15]),  # Window j
    ]
    
    drifts = compute_batch_drift(window_weights)
    avg_drift = np.mean(drifts)
    print(f"  Windows: {len(window_weights)}")
    print(f"  Drifts: {[f'{d:.3f}' for d in drifts]}")
    print(f"  Average drift: {avg_drift:.3f}")
    results['avg_drift'] = avg_drift
    results['drift_computed'] = len(drifts) > 0
    print("  ✅ Strategy drift computed")
    
    # Step 6: Get telemetry
    print("\n[Step 6] Get Controller Telemetry...")
    telemetry = warm_start.get_telemetry()
    print(f"  Telemetry keys: {list(telemetry.keys())}")
    telemetry_complete = all(k in telemetry for k in [
        'warm_start_count', 'random_count', 'warm_start_ratio',
        'pareto_solutions_used', 'unmapped_leaves_count'
    ])
    results['telemetry_complete'] = telemetry_complete
    print(f"  Complete: {telemetry_complete}")
    print("  ✅ Telemetry retrieved")
    
    # Step 7: Verify baseline EA difference
    print("\n[Step 7] Verify Baseline EA (Pure Random)...")
    baseline_pop = create_random_population(pop_size=50, n_leaves=n_curr_leaves)
    
    all_random = all(ind.get('source') == 'random' for ind in baseline_pop)
    all_valid = all(np.isclose(ind['weights'].sum(), 1.0) for ind in baseline_pop)
    print(f"  Baseline population: {len(baseline_pop)}")
    print(f"  All random: {all_random}")
    print(f"  All valid weights: {all_valid}")
    results['baseline_valid'] = all_random and all_valid
    print("  ✅ Baseline EA verified")
    
    elapsed = time.time() - start_time
    results['elapsed'] = elapsed
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    criteria = {
        'population_created': results.get('population_size') == pop_size,
        'ratio_valid': results.get('ratio_valid', False),
        'drift_computed': results.get('drift_computed', False),
        'telemetry_complete': results.get('telemetry_complete', False),
        'baseline_valid': results.get('baseline_valid', False),
        'elapsed_fast': elapsed < 5.0,
    }
    
    print("\n| Criteria | Status | Detail |")
    print("|----------|--------|--------|")
    for name, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        detail = str(results.get(name.replace('_', '').replace('valid', ''), ''))[:20]
        print(f"| {name} | {status} | {detail} |")
    
    all_pass = all(criteria.values())
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ P0-21 DRIFT-AWARE WARM-START VERIFICATION PASSED")
    else:
        print("❌ P0-21 VERIFICATION FAILED")
    print("=" * 60)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "experiments" / "ea_integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "p021_warmstart_results.json"
    
    results_json = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'criteria': {k: v for k, v in criteria.items()},
        'all_pass': all_pass,
        'warm_start_ratio': results.get('warm_start_ratio'),
        'avg_drift': results.get('avg_drift'),
        'elapsed': elapsed,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {output_path}")
    
    return all_pass


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
