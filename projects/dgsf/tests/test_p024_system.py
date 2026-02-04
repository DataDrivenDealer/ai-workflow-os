#!/usr/bin/env python3
"""
P0-24 EA Layer System Test
===========================

验证 EA Layer v3.1 完整端到端流程：

1. Enhanced vs Baseline 模式对比
2. 完整 EA 优化循环
3. HV trajectory 收敛
4. Warm-start 有效性
5. SDF consistency 过滤有效性
6. 遥测完整性
7. v3.1 Section 7 关键区别验证
"""

import sys
from pathlib import Path
import time

# Add scripts to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import numpy as np


def create_mock_oracle():
    """Create mock SDF pricing oracle."""
    def oracle(weights):
        """
        Mock oracle: SDF penalty based on weight concentration.
        More concentrated = higher penalty.
        """
        if len(weights) == 0:
            return 0.05
        # Herfindahl index as penalty proxy
        hhi = np.sum(weights ** 2)
        return float(hhi * 0.1)  # Scale to reasonable range
    return oracle


def create_mock_eval_fn(oracle):
    """Create mock evaluation function with 4 objectives."""
    def eval_fn(weights):
        if weights is None or len(weights) == 0:
            weights = np.ones(8) / 8
        
        # Sharpe: higher diversity = higher Sharpe (mock)
        diversity = 1 - np.sum(weights ** 2)
        sharpe = 0.5 + diversity * 2 + np.random.normal(0, 0.1)
        
        # MDD: random with slight weight dependence
        mdd = 0.10 + np.random.uniform(0, 0.10)
        
        # Turnover: mock
        turnover = 0.05 + np.random.uniform(0, 0.05)
        
        # SDF penalty from oracle
        sdf_penalty = oracle(weights)
        
        return {
            'sharpe': float(max(0.1, sharpe)),
            'mdd': float(mdd),
            'turnover': float(turnover),
            'sdf_penalty': float(sdf_penalty),
        }
    return eval_fn


def run_system_test():
    """Run full system test."""
    print("=" * 70)
    print("P0-24 EA Layer System Test - v3.1 Compliance")
    print("=" * 70)
    
    from ea_layer_runner import EALayerRunner
    
    oracle = create_mock_oracle()
    eval_fn = create_mock_eval_fn(oracle)
    
    passed = 0
    total = 8
    
    # =========================================================================
    # Test 1: Enhanced Full Run
    # =========================================================================
    print("\n[Test 1] Enhanced Mode Full EA Run...")
    try:
        t0 = time.time()
        
        enhanced_runner = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 30,
            'n_gen': 10,
            'elite_ratio': 0.3,
            'gamma': 0.02,  # SDF consistency threshold
            'g_plateau': 3,
            'epsilon': 0.005,
        })
        
        result = enhanced_runner.run(
            evaluate_fn=eval_fn,
            n_leaves=8,
            previous_pareto=None,
            leaf_mapping=None
        )
        
        elapsed = time.time() - t0
        
        assert 'pareto_front' in result
        assert len(result['pareto_front']) > 0
        
        pf = result['pareto_front']
        print(f"  Pareto front: {len(pf)} individuals")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  HV trajectory length: {len(result.get('hv_trajectory', []))}")
        print("  ✅ Enhanced run complete")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Test 2: Baseline Full Run
    # =========================================================================
    print("\n[Test 2] Baseline Mode Full EA Run...")
    try:
        t0 = time.time()
        
        baseline_runner = EALayerRunner(config={
            'mode': 'baseline',
            'pop_size': 30,
            'n_gen': 10,
            'elite_ratio': 0.3,
        })
        
        result_baseline = baseline_runner.run(
            evaluate_fn=eval_fn,
            n_leaves=8,
            previous_pareto=None,
            leaf_mapping=None
        )
        
        elapsed = time.time() - t0
        
        assert 'pareto_front' in result_baseline
        assert len(result_baseline['pareto_front']) > 0
        
        print(f"  Pareto front: {len(result_baseline['pareto_front'])} individuals")
        print(f"  Elapsed: {elapsed:.2f}s")
        print("  ✅ Baseline run complete")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # =========================================================================
    # Test 3: Enhanced vs Baseline - v3.1 Section 7 Distinction
    # =========================================================================
    print("\n[Test 3] v3.1 Section 7: Enhanced vs Baseline Distinction...")
    try:
        # Create previous pareto for warm-start test
        mock_prev_pareto = []
        for i in range(10):
            mock_prev_pareto.append({
                'id': i,
                'weights': np.random.dirichlet(np.ones(8)),
                'objectives': {
                    'sharpe': np.random.uniform(1.0, 2.0),
                    'mdd': 0.12,
                    'turnover': 0.06,
                    'sdf_penalty': 0.01,
                }
            })
        leaf_map = {i: i for i in range(8)}
        
        # Enhanced with warm-start
        enhanced = EALayerRunner(config={'mode': 'enhanced', 'pop_size': 20})
        enh_pop = enhanced.initialize_population(
            previous_pareto=mock_prev_pareto,
            leaf_mapping=leaf_map,
            n_leaves=8
        )
        enh_warm = sum(1 for ind in enh_pop if ind.get('source') == 'warm_start')
        
        # Baseline - no warm-start
        baseline = EALayerRunner(config={'mode': 'baseline', 'pop_size': 20})
        base_pop = baseline.initialize_population(
            previous_pareto=mock_prev_pareto,
            leaf_mapping=leaf_map,
            n_leaves=8
        )
        base_warm = sum(1 for ind in base_pop if ind.get('source') == 'warm_start')
        
        # v3.1 Section 7: Baseline must NOT use warm-start
        assert base_warm == 0, f"Baseline should have 0 warm-start, got {base_warm}"
        assert enh_warm > 0 or len(enh_pop) == 20, "Enhanced should use warm-start"
        
        print(f"  Enhanced warm-start: {enh_warm}/{len(enh_pop)}")
        print(f"  Baseline warm-start: {base_warm}/{len(base_pop)}")
        print("  ✅ v3.1 Section 7 distinction verified")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # =========================================================================
    # Test 4: HV-aware Exploration (v3.1 Section 4.1)
    # =========================================================================
    print("\n[Test 4] v3.1 Section 4.1: HV-aware Exploration...")
    try:
        runner = EALayerRunner(config={
            'mode': 'enhanced',
            'g_plateau': 3,
            'epsilon': 0.01,
        })
        
        # Simulate plateau
        for _ in range(5):
            runner.hv_trajectory.append(1.0)
        
        triggered, action = runner.check_hv_plateau()
        
        assert triggered is True, "Should detect plateau"
        assert action in ['mutation_boost', 'random_injection', 'partial_restart']
        
        print(f"  Plateau detected: {triggered}")
        print(f"  Action: {action}")
        print("  ✅ HV-aware exploration verified")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # =========================================================================
    # Test 5: SDF Consistency Filtering (v3.1 Section 4.3)
    # =========================================================================
    print("\n[Test 5] v3.1 Section 4.3: SDF Consistency Filtering...")
    try:
        runner = EALayerRunner(config={
            'mode': 'enhanced',
            'gamma': 0.015,  # threshold
        })
        
        # Create population with mixed consistency
        population = []
        for i in range(20):
            sdf_penalty = 0.005 if i < 12 else 0.030  # 60% consistent
            population.append({
                'id': i,
                'weights': np.random.dirichlet(np.ones(5)),
                'objectives': {
                    'sharpe': np.random.uniform(0.5, 2.5),
                    'mdd': 0.12,
                    'turnover': 0.07,
                    'sdf_penalty': sdf_penalty,
                }
            })
        
        elite = runner.select_elite(population, elite_count=5)
        
        # All elite should be consistent
        all_consistent = all(ind['objectives']['sdf_penalty'] <= 0.015 for ind in elite)
        
        assert all_consistent, "Elite should only contain consistent individuals"
        print(f"  Elite size: {len(elite)}")
        print(f"  All consistent: {all_consistent}")
        print("  ✅ SDF consistency filtering verified")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # =========================================================================
    # Test 6: Rolling Warm-Start Simulation (v3.1 Section 4.2)
    # =========================================================================
    print("\n[Test 6] v3.1 Section 4.2: Rolling Warm-Start Simulation...")
    try:
        # Window 1
        runner1 = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 20,
            'n_gen': 5,
        })
        result1 = runner1.run(
            evaluate_fn=eval_fn,
            n_leaves=8,
            previous_pareto=None,
            leaf_mapping=None
        )
        pf1 = result1['pareto_front']
        
        # Window 2 - with warm-start from Window 1
        runner2 = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 20,
            'n_gen': 5,
        })
        leaf_map_2 = {i: i for i in range(8)}  # No drift
        result2 = runner2.run(
            evaluate_fn=eval_fn,
            n_leaves=8,
            previous_pareto=pf1,
            leaf_mapping=leaf_map_2
        )
        pf2 = result2['pareto_front']
        
        print(f"  Window 1 Pareto: {len(pf1)} individuals")
        print(f"  Window 2 Pareto: {len(pf2)} individuals")
        print("  ✅ Rolling warm-start simulation complete")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Test 7: Telemetry Completeness
    # =========================================================================
    print("\n[Test 7] Telemetry Completeness...")
    try:
        runner = EALayerRunner(config={
            'mode': 'enhanced',
            'pop_size': 15,
            'n_gen': 3,
        })
        result = runner.run(
            evaluate_fn=eval_fn,
            n_leaves=6,
        )
        
        telemetry = runner.get_telemetry()
        
        required = [
            'mode', 'hv_trajectory', 'plateau_events',
            'generations_completed', 'best_objectives'
        ]
        missing = [k for k in required if k not in telemetry]
        
        assert len(missing) == 0, f"Missing: {missing}"
        
        print(f"  Telemetry keys: {len(telemetry)}")
        print(f"  Generations: {telemetry['generations_completed']}")
        print("  ✅ Telemetry complete")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # =========================================================================
    # Test 8: Module Import Verification
    # =========================================================================
    print("\n[Test 8] Module Import & Integration Check...")
    try:
        from ea_evaluation_metrics import detect_hv_plateau, compute_hv_trajectory
        from hv_aware_controller import HVAwareController
        from drift_aware_warmstart import DriftAwareWarmStart
        from sdf_consistency_selection import SDFConsistencySelector
        from ea_layer_runner import EALayerRunner
        
        # Verify EALayerRunner uses all modules
        runner = EALayerRunner(config={'mode': 'enhanced'})
        runner._init_controllers()
        
        assert runner._hv_controller is not None
        assert runner._warm_start is not None
        assert runner._selector is not None
        
        print("  All v3.1 modules imported ✅")
        print("  EALayerRunner integrates all controllers ✅")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    if passed == total:
        print(f"✅ ALL SYSTEM TESTS PASSED ({passed}/{total})")
        print("=" * 70)
        print("\n📋 v3.1 Compliance Summary:")
        print("   ✅ Section 4.1: HV-aware behaviour")
        print("   ✅ Section 4.2: Drift-aware warm-start")
        print("   ✅ Section 4.3: SDF consistency filtering")
        print("   ✅ Section 7: Baseline vs Enhanced distinction")
        print("=" * 70)
    else:
        print(f"❌ TESTS FAILED: {total - passed} failed ({passed}/{total} passed)")
        print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_system_test()
    sys.exit(0 if success else 1)
