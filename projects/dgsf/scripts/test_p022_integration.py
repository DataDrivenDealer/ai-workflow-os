"""
P0-22 SDF-Consistency Selection - Integration Test

Verifies that the SDF-consistency selection module works correctly with:
1. Real/synthetic EA population data
2. All three selection modes (lexicographic, threshold, weighted)
3. Baseline EA comparison (no consistency filtering)

v3.1 Requirements Verified:
- 4.3.1 MUST: Set consistency threshold γ
- 4.3.2 MUST: Consistency-first selection rules
- 4.3.3: Baseline EA uses f⁴ as normal objective only
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

from sdf_consistency_selection import (
    SDFConsistencySelector,
    batch_check_consistency,
    baseline_select,
    baseline_multiobjective_rank,
)


def create_synthetic_population(n: int = 100) -> List[Dict[str, Any]]:
    """Create synthetic population with varied SDF consistency."""
    np.random.seed(42)
    population = []
    
    for i in range(n):
        # Generate correlated objectives
        sharpe = np.random.uniform(0.5, 2.5)
        mdd = np.random.uniform(0.05, 0.25)
        turnover = np.random.uniform(0.05, 0.20)
        
        # SDF penalty: mix of consistent and inconsistent
        if np.random.random() < 0.7:  # 70% consistent
            sdf_penalty = np.random.uniform(0.001, 0.008)
        else:  # 30% inconsistent
            sdf_penalty = np.random.uniform(0.012, 0.030)
        
        individual = {
            'id': i,
            'objectives': {
                'sharpe': round(sharpe, 3),
                'mdd': round(mdd, 3),
                'turnover': round(turnover, 3),
                'sdf_penalty': round(sdf_penalty, 4),
            }
        }
        population.append(individual)
    
    return population


def run_integration_test():
    """Run P0-22 integration test."""
    print("=" * 60)
    print("P0-22: SDF-Consistency Selection - Integration Test")
    print("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # Step 1: Create population
    print("\n[Step 1] Create Synthetic Population...")
    population = create_synthetic_population(100)
    print(f"  Population size: {len(population)}")
    
    gamma = 0.01
    consistency_flags = batch_check_consistency(population, gamma)
    consistent_count = sum(consistency_flags)
    inconsistent_count = len(population) - consistent_count
    
    print(f"  Gamma: {gamma}")
    print(f"  Consistent: {consistent_count}, Inconsistent: {inconsistent_count}")
    print("  ✅ Population created")
    results['consistent_count'] = consistent_count
    results['inconsistent_count'] = inconsistent_count
    
    # Step 2: Test Lexicographic Selection
    print("\n[Step 2] Lexicographic Selection...")
    selector_lex = SDFConsistencySelector(gamma=gamma, mode='lexicographic')
    elite_lex = selector_lex.select_elite(population, elite_count=10)
    
    lex_all_consistent = all(
        ind['objectives']['sdf_penalty'] <= gamma for ind in elite_lex
    )
    print(f"  Elite count: {len(elite_lex)}")
    print(f"  All consistent: {lex_all_consistent}")
    results['lex_elite_consistent'] = lex_all_consistent
    print("  ✅ Lexicographic selection working")
    
    # Step 3: Test Threshold Selection
    print("\n[Step 3] Threshold Selection...")
    selector_thr = SDFConsistencySelector(gamma=gamma, mode='threshold')
    elite_thr = selector_thr.select_elite(population, elite_count=10)
    
    thr_all_consistent = all(
        ind['objectives']['sdf_penalty'] <= gamma for ind in elite_thr
    )
    print(f"  Elite count: {len(elite_thr)}")
    print(f"  All consistent: {thr_all_consistent}")
    results['thr_elite_consistent'] = thr_all_consistent
    print("  ✅ Threshold selection working")
    
    # Step 4: Test Weighted Selection
    print("\n[Step 4] Weighted Selection...")
    selector_wgt = SDFConsistencySelector(gamma=gamma, mode='weighted', penalty_weight=50.0)
    elite_wgt = selector_wgt.select_elite(population, elite_count=10)
    
    wgt_penalized = all('penalized_fitness' in ind for ind in elite_wgt)
    print(f"  Elite count: {len(elite_wgt)}")
    print(f"  All have penalized_fitness: {wgt_penalized}")
    results['wgt_penalized'] = wgt_penalized
    print("  ✅ Weighted selection working")
    
    # Step 5: Compare with Baseline EA
    print("\n[Step 5] Compare with Baseline EA (no filter)...")
    baseline_elite = baseline_select(population, select_count=10, objective='sharpe')
    
    baseline_has_inconsistent = any(
        ind['objectives']['sdf_penalty'] > gamma for ind in baseline_elite
    )
    print(f"  Baseline elite count: {len(baseline_elite)}")
    print(f"  Contains inconsistent: {baseline_has_inconsistent}")
    results['baseline_has_inconsistent'] = baseline_has_inconsistent
    print("  ✅ Baseline EA verified (includes inconsistent)")
    
    # Step 6: Verify NSGA-II ranking for baseline
    print("\n[Step 6] Baseline NSGA-II Ranking...")
    ranked = baseline_multiobjective_rank(population)
    has_ranks = all('rank' in ind for ind in ranked)
    has_crowding = all('crowding' in ind for ind in ranked)
    print(f"  All have rank: {has_ranks}")
    print(f"  All have crowding: {has_crowding}")
    results['ranking_complete'] = has_ranks and has_crowding
    print("  ✅ NSGA-II ranking complete")
    
    # Step 7: Get Telemetry
    print("\n[Step 7] Get Selection Telemetry...")
    telemetry = selector_lex.get_telemetry()
    telemetry_keys = ['gamma', 'mode', 'consistent_count', 'inconsistent_count', 'consistency_ratio']
    telemetry_complete = all(k in telemetry for k in telemetry_keys)
    print(f"  Telemetry keys: {list(telemetry.keys())}")
    print(f"  Complete: {telemetry_complete}")
    results['telemetry_complete'] = telemetry_complete
    print("  ✅ Telemetry retrieved")
    
    elapsed = time.time() - start_time
    results['elapsed'] = elapsed
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    criteria = {
        'lex_elite_consistent': results.get('lex_elite_consistent', False),
        'thr_elite_consistent': results.get('thr_elite_consistent', False),
        'wgt_penalized': results.get('wgt_penalized', False),
        'baseline_has_inconsistent': results.get('baseline_has_inconsistent', False),
        'ranking_complete': results.get('ranking_complete', False),
        'telemetry_complete': results.get('telemetry_complete', False),
        'elapsed_fast': elapsed < 5.0,
    }
    
    print("\n| Criteria | Status | Detail |")
    print("|----------|--------|--------|")
    for name, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"| {name} | {status} | - |")
    
    all_pass = all(criteria.values())
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✅ P0-22 SDF-CONSISTENCY SELECTION VERIFICATION PASSED")
    else:
        print("❌ P0-22 VERIFICATION FAILED")
    print("=" * 60)
    
    # Save results
    output_dir = Path(__file__).parent.parent / "experiments" / "ea_integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "p022_consistency_selection_results.json"
    
    results_json = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'gamma': gamma,
        'criteria': {k: v for k, v in criteria.items()},
        'all_pass': all_pass,
        'consistent_count': results.get('consistent_count'),
        'inconsistent_count': results.get('inconsistent_count'),
        'elapsed': elapsed,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {output_path}")
    
    return all_pass


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
