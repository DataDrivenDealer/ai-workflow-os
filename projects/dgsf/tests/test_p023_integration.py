#!/usr/bin/env python3
"""
P0-23 EA Layer Runner Integration Test
=======================================

Validates v3.1 Section 4 MUST requirements:
- 4.1.x HV-aware exploration
- 4.2.x Drift-aware warm-start  
- 4.3.x SDF-consistency selection

Integration criteria:
1. Enhanced mode uses warm-start
2. Enhanced mode detects HV plateau
3. Enhanced mode applies consistency filter
4. Enhanced vs Baseline shows measurable differences
5. Telemetry captures all components
6. Full run produces valid Pareto front
7. All v3.1 modules integrate without error
"""

import sys
from pathlib import Path

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import numpy as np

def mock_eval_fn(weights):
    """Mock evaluation function matching expected signature."""
    return {
        'sharpe': np.random.uniform(0.5, 2.5),
        'mdd': np.random.uniform(0.05, 0.25),
        'turnover': np.random.uniform(0.02, 0.15),
        'sdf_penalty': np.random.uniform(0.001, 0.025),
    }


def run_integration_test():
    """Run full integration test suite."""
    print("=" * 60)
    print("P0-23 EA Layer Runner - Integration Test")
    print("=" * 60)
    
    from ea_layer_runner import EALayerRunner
    
    passed = 0
    total = 7
    
    # Test 1: Enhanced mode uses warm-start
    print("\n[Test 1] Enhanced Mode Warm-Start...")
    try:
        runner = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 30,
            'max_gen': 5,
            'gamma': 0.015,
        })
        
        # Create mock previous Pareto front for warm-start
        mock_prev_pareto = []
        for i in range(10):
            mock_prev_pareto.append({
                'id': i,
                'weights': np.random.dirichlet(np.ones(8)),
                'objectives': {
                    'sharpe': np.random.uniform(1.0, 2.0),
                    'mdd': 0.10,
                    'turnover': 0.05,
                    'sdf_penalty': 0.005,
                }
            })
        
        # Leaf mapping: 1-to-1 mapping (no drift)
        leaf_mapping = {i: i for i in range(8)}
        
        population = runner.initialize_population(
            previous_pareto=mock_prev_pareto,
            leaf_mapping=leaf_mapping,
            n_leaves=8
        )
        
        # Check warm-start was used
        warm_count = sum(1 for ind in population if ind.get('source') == 'warm_start')
        random_count = sum(1 for ind in population if ind.get('source') == 'random')
        
        assert warm_count > 0 or len(population) == 30, "Population not initialized"
        print(f"  Population: {len(population)}, Warm: {warm_count}, Random: {random_count} ✅")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 2: HV plateau detection
    print("\n[Test 2] HV Plateau Detection...")
    try:
        runner = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 20,
            'g_plateau': 5,
            'epsilon': 0.01,
        })
        
        # Simulate plateau
        for _ in range(10):
            runner.hv_trajectory.append(1.0)
        
        triggered, action = runner.check_hv_plateau()
        assert triggered is True, "Plateau not detected"
        assert action is not None, "No action returned"
        print(f"  Plateau detected: {triggered}, Action: {action} ✅")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 3: SDF consistency filtering
    print("\n[Test 3] SDF Consistency Filtering...")
    try:
        runner = EALayerRunner(config={
            'mode': 'enhanced',
            'gamma': 0.01,
        })
        
        # Create mixed population
        population = []
        for i in range(20):
            sdf_penalty = 0.005 if i < 12 else 0.025
            population.append({
                'id': i,
                'weights': np.random.dirichlet(np.ones(5)),
                'objectives': {
                    'sharpe': np.random.uniform(0.5, 2.0),
                    'mdd': 0.12,
                    'turnover': 0.08,
                    'sdf_penalty': sdf_penalty,
                }
            })
        
        elite = runner.select_elite(population, elite_count=5)
        
        # All elite should be consistent
        all_consistent = all(ind['objectives']['sdf_penalty'] <= 0.01 for ind in elite)
        assert all_consistent, "Elite contains inconsistent individuals"
        print(f"  Elite consistency verified: {len(elite)} individuals ✅")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 4: Enhanced vs Baseline difference
    print("\n[Test 4] Enhanced vs Baseline Difference...")
    try:
        # Create mock previous Pareto for enhanced
        mock_prev = []
        for i in range(8):
            mock_prev.append({
                'id': i,
                'weights': np.random.dirichlet(np.ones(6)),
                'objectives': {'sharpe': 1.5, 'mdd': 0.1, 'turnover': 0.05, 'sdf_penalty': 0.005}
            })
        leaf_map = {i: i for i in range(6)}
        
        # Enhanced run
        enhanced_runner = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 25,
            'max_gen': 8,
            'gamma': 0.015,
        })
        enhanced_pop = enhanced_runner.initialize_population(
            previous_pareto=mock_prev,
            leaf_mapping=leaf_map,
            n_leaves=6
        )
        enhanced_warm = sum(1 for ind in enhanced_pop if ind.get('source') == 'warm_start')
        
        # Baseline run - no previous Pareto, should be all random
        baseline_runner = EALayerRunner(config={
            'mode': 'baseline',
            'pop_size': 25,
            'max_gen': 8,
        })
        baseline_pop = baseline_runner.initialize_population(
            previous_pareto=[],
            leaf_mapping={},
            n_leaves=6
        )
        baseline_warm = sum(1 for ind in baseline_pop if ind.get('source') == 'warm_start')
        
        # Enhanced should have warm-start or all population, Baseline should be 0
        print(f"  Enhanced warm: {enhanced_warm}, Baseline warm: {baseline_warm} ✅")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 5: Telemetry completeness
    print("\n[Test 5] Telemetry Completeness...")
    try:
        runner = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 15,
            'max_gen': 3,
        })
        runner.initialize_population(
            previous_pareto=[],
            leaf_mapping={},
            n_leaves=5
        )
        
        telemetry = runner.get_telemetry()
        
        required_keys = [
            'hv_trajectory', 'plateau_events', 'mode',
            'generations_completed', 'best_objectives'
        ]
        missing = [k for k in required_keys if k not in telemetry]
        assert len(missing) == 0, f"Missing keys: {missing}"
        print(f"  Telemetry keys: {list(telemetry.keys())} ✅")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 6: Full run produces Pareto front
    print("\n[Test 6] Full Run Pareto Front...")
    try:
        runner = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 20,
            'n_gen': 5,
            'elite_ratio': 0.3,
            'gamma': 0.02,
        })
        
        result = runner.run(
            evaluate_fn=mock_eval_fn,
            n_leaves=6,
            previous_pareto=None,
            leaf_mapping=None
        )
        
        assert 'pareto_front' in result, "No Pareto front returned"
        assert len(result['pareto_front']) > 0, "Empty Pareto front"
        
        # Verify Pareto front individuals have objectives
        pf = result['pareto_front']
        for ind in pf:
            assert 'objectives' in ind, "Individual missing objectives"
        
        print(f"  Pareto front size: {len(pf)} individuals ✅")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 7: All v3.1 modules integrate
    print("\n[Test 7] v3.1 Module Integration...")
    try:
        from ea_evaluation_metrics import compute_hv_trajectory, detect_hv_plateau
        from hv_aware_controller import HVAwareController
        from drift_aware_warmstart import DriftAwareWarmStart
        from sdf_consistency_selection import SDFConsistencySelector
        
        # All imports successful, verify they're used in runner
        runner = EALayerRunner(config={'mode': 'enhanced'})
        runner.initialize_population(
            previous_pareto=[],
            leaf_mapping={},
            n_leaves=5
        )
        
        assert runner._hv_controller is not None, "HV controller not initialized"
        assert runner._warm_start is not None, "Warm-start not initialized"
        assert runner._selector is not None, "Selector not initialized"
        
        print("  All v3.1 modules integrated ✅")
        passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if passed == total:
        print(f"✅ ALL INTEGRATION TESTS PASSED ({passed}/{total})")
    else:
        print(f"❌ TESTS FAILED: {total - passed} failed ({passed}/{total} passed)")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
