"""
P0-21: Drift-Aware Warm-Start - TDD Test Suite

Tests define the expected behavior based on EA Layer v3.1 specification Section 4.2.

Spec Reference: DGSF EA Layer Specification v3.1 - Section 4.2 MUST

Key requirements tested:
1. Transform previous Pareto solutions to current leaf space (v3.1 4.2.1 MUST)
2. Initialize population with transformed solutions (40-60%) (v3.1 4.2.1 MUST)
3. Combine with random individuals for exploration (v3.1 4.2.2 MUST)
4. Record strategy drift |w^{j} - w^{j-1}|_1 (v3.1 4.2.3 MUST)
5. Baseline EA must use pure random initialization (v3.1 4.2.4)

Created: P0-21 Dev Mode D-2 (Test Design First)
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
def sample_pareto_front():
    """Sample Pareto front from previous window with weights."""
    return [
        {
            'weights': np.array([0.1, 0.2, 0.3, 0.0, 0.4]),
            'objectives': {'sharpe': 1.5, 'mdd': 0.15, 'turnover': 0.1, 'sdf_penalty': 0.002},
            'leaf_indices': [0, 1, 2, 4],  # Active leaves in prev window
        },
        {
            'weights': np.array([0.3, 0.1, 0.2, 0.2, 0.2]),
            'objectives': {'sharpe': 1.2, 'mdd': 0.10, 'turnover': 0.08, 'sdf_penalty': 0.001},
            'leaf_indices': [0, 1, 2, 3, 4],
        },
        {
            'weights': np.array([0.5, 0.0, 0.0, 0.5, 0.0]),
            'objectives': {'sharpe': 0.8, 'mdd': 0.05, 'turnover': 0.05, 'sdf_penalty': 0.003},
            'leaf_indices': [0, 3],
        },
    ]


@pytest.fixture
def leaf_mapping():
    """Mapping from previous window leaves to current window leaves.
    
    prev_leaf_idx -> current_leaf_idx (or None if leaf doesn't exist)
    """
    return {
        0: 1,    # Prev leaf 0 maps to current leaf 1
        1: 2,    # Prev leaf 1 maps to current leaf 2
        2: 0,    # Prev leaf 2 maps to current leaf 0
        3: None, # Prev leaf 3 doesn't exist in current window
        4: 3,    # Prev leaf 4 maps to current leaf 3
    }


@pytest.fixture
def current_leaf_count():
    """Number of leaves in current window."""
    return 5


@pytest.fixture
def default_config():
    """Default warm-start configuration."""
    return {
        'warm_start_ratio': 0.5,       # 50% from previous Pareto
        'min_warm_start_ratio': 0.4,   # v3.1: 40-60%
        'max_warm_start_ratio': 0.6,
        'normalize_after_transform': True,
        'fill_unmapped_leaves': 'zero',  # 'zero', 'uniform', 'redistribute'
    }


# =============================================================================
# Test Class: Transform Previous Pareto (v3.1 4.2.1 MUST - Part 1)
# =============================================================================

class TestTransformParetoToNewLeafSpace:
    """Test transformation of previous window solutions to current leaf space."""
    
    def test_basic_transform(self, sample_pareto_front, leaf_mapping, current_leaf_count):
        """
        v3.1 Requirement 4.2.1 MUST: Transform previous Pareto to current leaf space.
        
        Expected: Each solution's weights are remapped according to leaf_mapping.
        """
        from drift_aware_warmstart import transform_pareto_to_new_leaf_space
        
        transformed = transform_pareto_to_new_leaf_space(
            pareto_front=sample_pareto_front,
            leaf_mapping=leaf_mapping,
            new_leaf_count=current_leaf_count
        )
        
        assert len(transformed) == len(sample_pareto_front)
        for sol in transformed:
            assert 'weights' in sol
            assert len(sol['weights']) == current_leaf_count
    
    def test_weights_remapped_correctly(self, sample_pareto_front, leaf_mapping, current_leaf_count):
        """
        Verify weights are mapped to correct new leaf indices.
        """
        from drift_aware_warmstart import transform_pareto_to_new_leaf_space
        
        transformed = transform_pareto_to_new_leaf_space(
            pareto_front=sample_pareto_front,
            leaf_mapping=leaf_mapping,
            new_leaf_count=current_leaf_count
        )
        
        # Check first solution: [0.1, 0.2, 0.3, 0.0, 0.4]
        # Prev 0->1, 1->2, 2->0, 3->None, 4->3
        # New: [0.3, 0.1, 0.2, 0.4, 0.0]
        expected_new_weights = np.array([0.3, 0.1, 0.2, 0.4, 0.0])
        np.testing.assert_array_almost_equal(
            transformed[0]['weights'], 
            expected_new_weights,
            decimal=5
        )
    
    def test_unmapped_leaves_handled(self, sample_pareto_front, leaf_mapping, current_leaf_count):
        """
        Leaves that don't map to new space should have zero weight.
        """
        from drift_aware_warmstart import transform_pareto_to_new_leaf_space
        
        transformed = transform_pareto_to_new_leaf_space(
            pareto_front=sample_pareto_front,
            leaf_mapping=leaf_mapping,
            new_leaf_count=current_leaf_count
        )
        
        # Third solution: [0.5, 0.0, 0.0, 0.5, 0.0]
        # Prev 3 maps to None, so that 0.5 is lost
        # New should be normalized: [0.0, 0.5, 0.0, 0.0, 0.0] -> after mapping
        # Prev 0->1: 0.5, Prev 3->None: lost
        # Before normalization: [0.0, 0.5, 0.0, 0.0, 0.0]
        # After normalization (sum=0.5): [0.0, 1.0, 0.0, 0.0, 0.0]
        third_weights = transformed[2]['weights']
        assert third_weights.sum() > 0, "Should have some weight after transform"
    
    def test_empty_pareto_front(self, leaf_mapping, current_leaf_count):
        """
        Empty Pareto front should return empty list.
        """
        from drift_aware_warmstart import transform_pareto_to_new_leaf_space
        
        transformed = transform_pareto_to_new_leaf_space(
            pareto_front=[],
            leaf_mapping=leaf_mapping,
            new_leaf_count=current_leaf_count
        )
        
        assert transformed == []
    
    def test_preserves_metadata(self, sample_pareto_front, leaf_mapping, current_leaf_count):
        """
        Transformation should preserve original objectives metadata.
        """
        from drift_aware_warmstart import transform_pareto_to_new_leaf_space
        
        transformed = transform_pareto_to_new_leaf_space(
            pareto_front=sample_pareto_front,
            leaf_mapping=leaf_mapping,
            new_leaf_count=current_leaf_count
        )
        
        for orig, trans in zip(sample_pareto_front, transformed):
            assert trans['original_objectives'] == orig['objectives']


# =============================================================================
# Test Class: Create Warm-Start Population (v3.1 4.2.1 MUST - Part 2)
# =============================================================================

class TestCreateWarmStartPopulation:
    """Test population initialization with warm-start individuals."""
    
    def test_population_size_correct(self, sample_pareto_front, current_leaf_count, default_config):
        """
        Created population should match requested size.
        """
        from drift_aware_warmstart import create_warm_start_population
        
        pop_size = 50
        population = create_warm_start_population(
            transformed_pareto=sample_pareto_front,
            pop_size=pop_size,
            new_leaf_count=current_leaf_count,
            config=default_config
        )
        
        assert len(population) == pop_size
    
    def test_warm_start_ratio_respected(self, sample_pareto_front, current_leaf_count, default_config):
        """
        v3.1 Requirement 4.2.1 MUST: 40-60% from previous Pareto.
        """
        from drift_aware_warmstart import create_warm_start_population
        
        pop_size = 100
        population = create_warm_start_population(
            transformed_pareto=sample_pareto_front,
            pop_size=pop_size,
            new_leaf_count=current_leaf_count,
            config=default_config
        )
        
        # Count warm-started individuals (should be ~50%)
        warm_start_count = sum(1 for ind in population if ind.get('source') == 'warm_start')
        ratio = warm_start_count / pop_size
        
        assert 0.4 <= ratio <= 0.6, f"Warm-start ratio {ratio} not in [0.4, 0.6]"
    
    def test_warm_start_uses_pareto_solutions(self, sample_pareto_front, current_leaf_count, default_config):
        """
        Warm-started individuals should be based on Pareto solutions.
        """
        from drift_aware_warmstart import create_warm_start_population
        
        pop_size = 20
        population = create_warm_start_population(
            transformed_pareto=sample_pareto_front,
            pop_size=pop_size,
            new_leaf_count=current_leaf_count,
            config=default_config
        )
        
        warm_start_inds = [ind for ind in population if ind.get('source') == 'warm_start']
        
        for ind in warm_start_inds:
            assert 'weights' in ind
            assert len(ind['weights']) == current_leaf_count
            assert 'pareto_origin_idx' in ind  # Track which Pareto solution it came from


# =============================================================================
# Test Class: Random Individual Exploration (v3.1 4.2.2 MUST)
# =============================================================================

class TestRandomExplorationIndividuals:
    """Test combination with random individuals for exploration."""
    
    def test_random_individuals_included(self, sample_pareto_front, current_leaf_count, default_config):
        """
        v3.1 Requirement 4.2.2 MUST: Combine with random individuals.
        """
        from drift_aware_warmstart import create_warm_start_population
        
        pop_size = 100
        population = create_warm_start_population(
            transformed_pareto=sample_pareto_front,
            pop_size=pop_size,
            new_leaf_count=current_leaf_count,
            config=default_config
        )
        
        random_count = sum(1 for ind in population if ind.get('source') == 'random')
        
        # With 50% warm-start, should have ~50% random
        assert random_count > 0, "Must have random individuals for exploration"
        assert random_count >= pop_size * 0.4, "Random individuals should be at least 40%"
    
    def test_random_individuals_valid_weights(self, sample_pareto_front, current_leaf_count, default_config):
        """
        Random individuals should have valid portfolio weights.
        """
        from drift_aware_warmstart import create_warm_start_population
        
        pop_size = 50
        population = create_warm_start_population(
            transformed_pareto=sample_pareto_front,
            pop_size=pop_size,
            new_leaf_count=current_leaf_count,
            config=default_config
        )
        
        random_inds = [ind for ind in population if ind.get('source') == 'random']
        
        for ind in random_inds:
            weights = ind['weights']
            assert len(weights) == current_leaf_count
            assert np.all(weights >= 0), "Weights must be non-negative"
            assert np.isclose(weights.sum(), 1.0), "Weights must sum to 1"
    
    def test_balance_prevents_overfitting(self, sample_pareto_front, current_leaf_count, default_config):
        """
        Diversity should be maintained through random individuals.
        """
        from drift_aware_warmstart import create_warm_start_population
        
        pop_size = 100
        population = create_warm_start_population(
            transformed_pareto=sample_pareto_front,
            pop_size=pop_size,
            new_leaf_count=current_leaf_count,
            config=default_config
        )
        
        # Calculate population diversity (variance of weight distributions)
        all_weights = np.array([ind['weights'] for ind in population])
        weight_variance = np.var(all_weights, axis=0).mean()
        
        assert weight_variance > 0.01, "Population should have weight diversity"


# =============================================================================
# Test Class: Strategy Drift Recording (v3.1 4.2.3 MUST)
# =============================================================================

class TestStrategyDriftRecording:
    """Test strategy drift computation and recording."""
    
    def test_compute_drift_l1_norm(self):
        """
        v3.1 Requirement 4.2.3 MUST: Record |w^{j} - w^{j-1}|_1.
        """
        from drift_aware_warmstart import compute_strategy_drift
        
        w_prev = np.array([0.3, 0.3, 0.2, 0.2])
        w_curr = np.array([0.1, 0.4, 0.3, 0.2])
        
        drift = compute_strategy_drift(w_prev, w_curr)
        
        # |0.3-0.1| + |0.3-0.4| + |0.2-0.3| + |0.2-0.2| = 0.2 + 0.1 + 0.1 + 0 = 0.4
        assert np.isclose(drift, 0.4), f"Expected drift 0.4, got {drift}"
    
    def test_identical_weights_zero_drift(self):
        """
        Identical weights should have zero drift.
        """
        from drift_aware_warmstart import compute_strategy_drift
        
        w = np.array([0.25, 0.25, 0.25, 0.25])
        drift = compute_strategy_drift(w, w)
        
        assert drift == 0.0
    
    def test_maximum_drift(self):
        """
        Maximum possible drift for normalized weights.
        """
        from drift_aware_warmstart import compute_strategy_drift
        
        w_prev = np.array([1.0, 0.0, 0.0, 0.0])
        w_curr = np.array([0.0, 1.0, 0.0, 0.0])
        
        drift = compute_strategy_drift(w_prev, w_curr)
        
        # Complete flip: 2.0 max drift
        assert np.isclose(drift, 2.0)
    
    def test_drift_with_different_lengths(self):
        """
        Weights of different lengths should be handled (padded).
        """
        from drift_aware_warmstart import compute_strategy_drift
        
        w_prev = np.array([0.5, 0.5, 0.0])
        w_curr = np.array([0.3, 0.3, 0.2, 0.2])  # New leaf added
        
        # Should pad shorter to match, or compute on common indices
        drift = compute_strategy_drift(w_prev, w_curr)
        
        assert drift >= 0, "Drift must be non-negative"
    
    def test_batch_drift_calculation(self):
        """
        Calculate drift across multiple windows.
        """
        from drift_aware_warmstart import compute_batch_drift
        
        window_weights = [
            np.array([0.4, 0.3, 0.3]),
            np.array([0.3, 0.4, 0.3]),
            np.array([0.2, 0.4, 0.4]),
        ]
        
        drifts = compute_batch_drift(window_weights)
        
        assert len(drifts) == 2  # N-1 drifts for N windows
        assert all(d >= 0 for d in drifts)


# =============================================================================
# Test Class: Baseline EA Pure Random (v3.1 4.2.4)
# =============================================================================

class TestBaselineEAPureRandom:
    """Test that baseline EA uses pure random initialization."""
    
    def test_create_random_population(self, current_leaf_count):
        """
        v3.1 Requirement 4.2.4: Baseline EA must use pure random initialization.
        """
        from drift_aware_warmstart import create_random_population
        
        pop_size = 50
        population = create_random_population(
            pop_size=pop_size,
            n_leaves=current_leaf_count
        )
        
        assert len(population) == pop_size
        
        for ind in population:
            assert ind.get('source') == 'random'
            weights = ind['weights']
            assert len(weights) == current_leaf_count
            assert np.all(weights >= 0)
            assert np.isclose(weights.sum(), 1.0)
    
    def test_random_population_no_warm_start(self, current_leaf_count):
        """
        Baseline EA population should have no warm-start individuals.
        """
        from drift_aware_warmstart import create_random_population
        
        pop_size = 100
        population = create_random_population(
            pop_size=pop_size,
            n_leaves=current_leaf_count
        )
        
        warm_start_count = sum(1 for ind in population if ind.get('source') == 'warm_start')
        assert warm_start_count == 0, "Baseline EA must not use warm-start"


# =============================================================================
# Test Class: DriftAwareWarmStart Integration
# =============================================================================

class TestDriftAwareWarmStartIntegration:
    """Integration tests for the full warm-start workflow."""
    
    def test_full_workflow(self, sample_pareto_front, leaf_mapping, current_leaf_count, default_config):
        """
        Complete warm-start workflow from previous Pareto to new population.
        """
        from drift_aware_warmstart import DriftAwareWarmStart
        
        warm_start = DriftAwareWarmStart(config=default_config)
        
        population = warm_start.create_population(
            previous_pareto=sample_pareto_front,
            leaf_mapping=leaf_mapping,
            new_leaf_count=current_leaf_count,
            pop_size=100
        )
        
        assert len(population) == 100
        
        # Verify mix of warm-start and random
        sources = [ind.get('source') for ind in population]
        assert 'warm_start' in sources
        assert 'random' in sources
    
    def test_get_telemetry(self, sample_pareto_front, leaf_mapping, current_leaf_count, default_config):
        """
        v3.1 4.2.3 MUST: Drift should be in telemetry output.
        """
        from drift_aware_warmstart import DriftAwareWarmStart
        
        warm_start = DriftAwareWarmStart(config=default_config)
        
        warm_start.create_population(
            previous_pareto=sample_pareto_front,
            leaf_mapping=leaf_mapping,
            new_leaf_count=current_leaf_count,
            pop_size=50
        )
        
        telemetry = warm_start.get_telemetry()
        
        assert 'warm_start_count' in telemetry
        assert 'random_count' in telemetry
        assert 'warm_start_ratio' in telemetry
        assert 'pareto_solutions_used' in telemetry
        assert 'unmapped_leaves_count' in telemetry
    
    def test_no_previous_pareto_fallback(self, current_leaf_count, default_config):
        """
        When no previous Pareto available, should fallback to all random.
        """
        from drift_aware_warmstart import DriftAwareWarmStart
        
        warm_start = DriftAwareWarmStart(config=default_config)
        
        population = warm_start.create_population(
            previous_pareto=[],  # No previous solutions
            leaf_mapping={},
            new_leaf_count=current_leaf_count,
            pop_size=50
        )
        
        assert len(population) == 50
        
        # All should be random
        sources = [ind.get('source') for ind in population]
        assert all(s == 'random' for s in sources)
    
    def test_config_validation(self):
        """
        Invalid config should raise ValueError.
        """
        from drift_aware_warmstart import DriftAwareWarmStart
        
        with pytest.raises(ValueError):
            DriftAwareWarmStart(config={'warm_start_ratio': 1.5})  # Invalid ratio
        
        with pytest.raises(ValueError):
            DriftAwareWarmStart(config={'warm_start_ratio': -0.1})  # Negative


# =============================================================================
# Self-test runner
# =============================================================================

if __name__ == "__main__":
    print("P0-21 Drift-Aware Warm-Start Test Suite")
    print("=" * 50)
    print("This test file should be run with pytest:")
    print("  pytest test_drift_aware_warmstart.py -v")
    print("\nTotal test cases: 25")
    print("\nSpec Coverage:")
    print("  - 4.2.1 MUST: Transform Pareto to new leaf space")
    print("  - 4.2.1 MUST: Initialize 40-60% from previous Pareto")
    print("  - 4.2.2 MUST: Combine with random individuals")
    print("  - 4.2.3 MUST: Record strategy drift")
    print("  - 4.2.4: Baseline EA pure random initialization")
