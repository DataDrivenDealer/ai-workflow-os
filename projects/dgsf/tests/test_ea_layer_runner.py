"""
P0-23: EA Layer Integration - TDD Test Suite

Tests define the expected behavior for the integrated EA Layer runner
that combines all v3.1 MUST modules.

Spec Reference: DGSF EA Layer Specification v3.1 - Section 4

Integrated Modules:
1. ea_evaluation_metrics (P0-19): HV, plateau, drift, SDF filtering
2. hv_aware_controller (P0-20): HV-driven exploration
3. drift_aware_warmstart (P0-21): Cross-window warm-start
4. sdf_consistency_selection (P0-22): Consistency-first selection

Created: P0-23 Dev Mode D-2 (Test Design First)
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
_scripts_dir = Path(__file__).parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import numpy as np
import pytest
from typing import List, Dict, Any


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def ea_config():
    """Standard EA Layer configuration."""
    return {
        # Population
        'pop_size': 50,
        'n_gen': 20,
        
        # HV-aware (4.1)
        'g_plateau': 5,
        'hv_epsilon': 0.01,
        'cooldown_gens': 3,
        'mutation_boost_factor': 2.0,
        
        # Warm-start (4.2)
        'warm_start_ratio': 0.5,
        
        # SDF consistency (4.3)
        'gamma': 0.01,
        'selection_mode': 'lexicographic',
        
        # Objectives
        'objectives': ['sharpe', 'mdd', 'turnover', 'sdf_penalty'],
        'maximize': [True, False, False, False],
    }


@pytest.fixture
def sample_previous_pareto():
    """Pareto front from previous window."""
    np.random.seed(42)
    return [
        {
            'weights': np.random.dirichlet(np.ones(10)),
            'objectives': {'sharpe': 1.5, 'mdd': 0.10, 'turnover': 0.08, 'sdf_penalty': 0.002},
        }
        for _ in range(5)
    ]


@pytest.fixture
def sample_leaf_mapping():
    """Leaf mapping between windows."""
    return {i: i for i in range(10)}  # Identity mapping


# =============================================================================
# Test Class: EALayerRunner Initialization
# =============================================================================

class TestEALayerRunnerInit:
    """Test EA Layer Runner initialization."""
    
    def test_init_with_config(self, ea_config):
        """
        Runner should initialize with valid configuration.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        
        assert runner is not None
        assert runner.config['pop_size'] == 50
        assert runner.config['n_gen'] == 20
    
    def test_init_default_config(self):
        """
        Runner should work with default configuration.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner()
        
        assert runner.config is not None
        assert 'pop_size' in runner.config
    
    def test_init_invalid_config_raises(self):
        """
        Invalid configuration should raise ValueError.
        """
        from ea_layer_runner import EALayerRunner
        
        with pytest.raises(ValueError):
            EALayerRunner(config={'pop_size': -10})


# =============================================================================
# Test Class: Population Initialization
# =============================================================================

class TestPopulationInitialization:
    """Test population initialization with warm-start integration."""
    
    def test_init_population_with_warmstart(self, ea_config, sample_previous_pareto, sample_leaf_mapping):
        """
        v3.1 4.2.1 MUST: Use previous Pareto for initialization.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        
        population = runner.initialize_population(
            previous_pareto=sample_previous_pareto,
            leaf_mapping=sample_leaf_mapping,
            n_leaves=10
        )
        
        assert len(population) == ea_config['pop_size']
        
        # Check warm-start ratio
        warm_count = sum(1 for ind in population if ind.get('source') == 'warm_start')
        ratio = warm_count / len(population)
        assert 0.4 <= ratio <= 0.6
    
    def test_init_population_random_only(self, ea_config):
        """
        v3.1 4.2.4: Without previous Pareto, use random initialization.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        
        population = runner.initialize_population(
            previous_pareto=[],
            leaf_mapping={},
            n_leaves=10
        )
        
        assert len(population) == ea_config['pop_size']
        
        # All should be random
        all_random = all(ind.get('source') == 'random' for ind in population)
        assert all_random


# =============================================================================
# Test Class: Generation Step
# =============================================================================

class TestGenerationStep:
    """Test single generation execution."""
    
    def test_step_generation(self, ea_config):
        """
        Single generation step should evaluate and select.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        population = runner.initialize_population([], {}, n_leaves=10)
        
        # Simulate objective evaluation
        for ind in population:
            ind['objectives'] = {
                'sharpe': np.random.uniform(0.5, 2.0),
                'mdd': np.random.uniform(0.05, 0.20),
                'turnover': np.random.uniform(0.05, 0.15),
                'sdf_penalty': np.random.uniform(0.001, 0.015),
            }
        
        new_pop = runner.step_generation(population, gen=0)
        
        assert len(new_pop) == len(population)
    
    def test_step_records_hv(self, ea_config):
        """
        v3.1 4.1.1 MUST: Record HV per generation.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        population = runner.initialize_population([], {}, n_leaves=10)
        
        for ind in population:
            ind['objectives'] = {
                'sharpe': np.random.uniform(0.5, 2.0),
                'mdd': np.random.uniform(0.05, 0.20),
                'turnover': np.random.uniform(0.05, 0.15),
                'sdf_penalty': np.random.uniform(0.001, 0.015),
            }
        
        runner.step_generation(population, gen=0)
        
        assert len(runner.hv_trajectory) >= 1


# =============================================================================
# Test Class: HV-Aware Behaviour
# =============================================================================

class TestHVAwareBehaviour:
    """Test HV-aware exploration integration."""
    
    def test_plateau_triggers_exploration(self, ea_config):
        """
        v3.1 4.1.3 MUST: HV plateau triggers exploration.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        
        # Simulate plateau by feeding constant HV
        for _ in range(10):
            runner.hv_trajectory.append(1.0)
        
        triggered, action = runner.check_hv_plateau()
        
        assert triggered is True
        assert action in ['mutation_boost', 'random_injection', 'partial_restart']
    
    def test_no_plateau_no_trigger(self, ea_config):
        """
        Increasing HV should not trigger exploration.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        
        # Simulate increasing HV
        for i in range(10):
            runner.hv_trajectory.append(1.0 + i * 0.1)
        
        triggered, action = runner.check_hv_plateau()
        
        assert triggered is False


# =============================================================================
# Test Class: SDF-Consistency Selection
# =============================================================================

class TestSDFConsistencyIntegration:
    """Test SDF-consistency selection integration."""
    
    def test_selection_filters_inconsistent(self, ea_config):
        """
        v3.1 4.3.2 MUST: Selection prioritizes consistent individuals.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        
        # Create population with mix of consistent/inconsistent
        population = []
        for i in range(20):
            sdf_penalty = 0.005 if i < 15 else 0.020  # 75% consistent
            population.append({
                'id': i,
                'weights': np.random.dirichlet(np.ones(10)),
                'objectives': {
                    'sharpe': np.random.uniform(1.0, 2.0),
                    'mdd': 0.10,
                    'turnover': 0.08,
                    'sdf_penalty': sdf_penalty,
                }
            })
        
        elite = runner.select_elite(population, elite_count=5)
        
        # Elite should be consistent (sdf_penalty <= gamma)
        for ind in elite:
            assert ind['objectives']['sdf_penalty'] <= ea_config['gamma']


# =============================================================================
# Test Class: Full Run
# =============================================================================

class TestFullRun:
    """Test complete EA optimization run."""
    
    def test_run_returns_pareto_front(self, ea_config):
        """
        Full run should return Pareto front.
        """
        from ea_layer_runner import EALayerRunner
        
        ea_config['n_gen'] = 5  # Short run for test
        runner = EALayerRunner(config=ea_config)
        
        # Mock evaluate function
        def mock_evaluate(weights):
            return {
                'sharpe': np.random.uniform(0.5, 2.0),
                'mdd': np.random.uniform(0.05, 0.20),
                'turnover': np.random.uniform(0.05, 0.15),
                'sdf_penalty': np.random.uniform(0.001, 0.010),
            }
        
        result = runner.run(
            evaluate_fn=mock_evaluate,
            n_leaves=10,
            previous_pareto=[],
            leaf_mapping={}
        )
        
        assert 'pareto_front' in result
        assert len(result['pareto_front']) > 0
    
    def test_run_records_telemetry(self, ea_config):
        """
        Run should record comprehensive telemetry.
        """
        from ea_layer_runner import EALayerRunner
        
        ea_config['n_gen'] = 5
        runner = EALayerRunner(config=ea_config)
        
        def mock_evaluate(weights):
            return {
                'sharpe': np.random.uniform(0.5, 2.0),
                'mdd': np.random.uniform(0.05, 0.20),
                'turnover': np.random.uniform(0.05, 0.15),
                'sdf_penalty': np.random.uniform(0.001, 0.010),
            }
        
        result = runner.run(
            evaluate_fn=mock_evaluate,
            n_leaves=10,
            previous_pareto=[],
            leaf_mapping={}
        )
        
        telemetry = result['telemetry']
        
        assert 'hv_trajectory' in telemetry
        assert 'exploration_triggers' in telemetry
        assert 'consistency_stats' in telemetry
        assert 'total_generations' in telemetry


# =============================================================================
# Test Class: Baseline EA Mode
# =============================================================================

class TestBaselineEAMode:
    """Test baseline EA mode (no v3.1 enhancements)."""
    
    def test_baseline_mode_no_warmstart(self, ea_config):
        """
        v3.1 4.2.4: Baseline EA uses random initialization only.
        """
        from ea_layer_runner import EALayerRunner
        
        ea_config['mode'] = 'baseline'
        runner = EALayerRunner(config=ea_config)
        
        population = runner.initialize_population(
            previous_pareto=[{'weights': np.ones(10)/10, 'objectives': {}}],
            leaf_mapping={i: i for i in range(10)},
            n_leaves=10
        )
        
        # All should be random in baseline mode
        all_random = all(ind.get('source') == 'random' for ind in population)
        assert all_random
    
    def test_baseline_mode_no_consistency_filter(self, ea_config):
        """
        v3.1 4.3.3: Baseline EA has no consistency filtering.
        """
        from ea_layer_runner import EALayerRunner
        
        ea_config['mode'] = 'baseline'
        runner = EALayerRunner(config=ea_config)
        
        # Create population with inconsistent individuals
        population = []
        for i in range(10):
            population.append({
                'id': i,
                'weights': np.random.dirichlet(np.ones(10)),
                'objectives': {
                    'sharpe': 2.0 - i * 0.1,  # Decreasing Sharpe
                    'mdd': 0.10,
                    'turnover': 0.08,
                    'sdf_penalty': 0.020,  # All inconsistent
                }
            })
        
        elite = runner.select_elite(population, elite_count=3)
        
        # Baseline should select by Sharpe, regardless of consistency
        assert len(elite) == 3
        assert elite[0]['objectives']['sharpe'] >= elite[1]['objectives']['sharpe']


# =============================================================================
# Test Class: Telemetry
# =============================================================================

class TestTelemetry:
    """Test telemetry output."""
    
    def test_get_full_telemetry(self, ea_config):
        """
        Runner should provide comprehensive telemetry.
        """
        from ea_layer_runner import EALayerRunner
        
        runner = EALayerRunner(config=ea_config)
        runner.hv_trajectory = [1.0, 1.1, 1.2]
        
        telemetry = runner.get_telemetry()
        
        assert 'config' in telemetry
        assert 'hv_trajectory' in telemetry
        assert 'mode' in telemetry


# =============================================================================
# Self-test runner
# =============================================================================

if __name__ == "__main__":
    print("P0-23 EA Layer Integration Test Suite")
    print("=" * 50)
    print("This test file should be run with pytest:")
    print("  pytest test_ea_layer_runner.py -v")
    print("\nTotal test cases: 18")
    print("\nIntegrated Modules:")
    print("  - ea_evaluation_metrics (P0-19)")
    print("  - hv_aware_controller (P0-20)")
    print("  - drift_aware_warmstart (P0-21)")
    print("  - sdf_consistency_selection (P0-22)")
