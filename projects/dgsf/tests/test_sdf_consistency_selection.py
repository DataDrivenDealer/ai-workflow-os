"""
P0-22: SDF-Consistency Selection - TDD Test Suite

Tests define the expected behavior based on EA Layer v3.1 specification Section 4.3.

Spec Reference: DGSF EA Layer Specification v3.1 - Section 4.3 MUST

Key requirements tested:
1. Set consistency threshold γ for |g^(w)| (v3.1 4.3.1 MUST)
2. Consistency-first selection rule (v3.1 4.3.2 MUST)
   - Lexicographic: tier by consistency first
   - Threshold: eliminate if |g^(w)| > γ
   - Weighted-parsing: penalize inconsistent strategies
3. Baseline EA uses f⁴ as normal objective only (v3.1 4.3.3)

Created: P0-22 Dev Mode D-2 (Test Design First)
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
def sample_population():
    """Sample population with varying SDF consistency values."""
    return [
        {'id': 0, 'objectives': {'sharpe': 1.5, 'mdd': 0.10, 'turnover': 0.08, 'sdf_penalty': 0.002}},
        {'id': 1, 'objectives': {'sharpe': 1.8, 'mdd': 0.12, 'turnover': 0.10, 'sdf_penalty': 0.008}},
        {'id': 2, 'objectives': {'sharpe': 2.0, 'mdd': 0.15, 'turnover': 0.12, 'sdf_penalty': 0.015}},  # Inconsistent
        {'id': 3, 'objectives': {'sharpe': 1.2, 'mdd': 0.08, 'turnover': 0.05, 'sdf_penalty': 0.001}},
        {'id': 4, 'objectives': {'sharpe': 2.5, 'mdd': 0.20, 'turnover': 0.15, 'sdf_penalty': 0.025}},  # Very inconsistent
        {'id': 5, 'objectives': {'sharpe': 1.6, 'mdd': 0.11, 'turnover': 0.09, 'sdf_penalty': 0.004}},
    ]


@pytest.fixture
def default_config():
    """Default SDF-consistency selection configuration."""
    return {
        'gamma': 0.01,  # Consistency threshold γ
        'selection_mode': 'lexicographic',  # or 'threshold', 'weighted'
        'penalty_weight': 10.0,  # For weighted mode
        'elite_fraction': 0.2,  # Fraction of population for elite
    }


# =============================================================================
# Test Class: Consistency Threshold (v3.1 4.3.1 MUST)
# =============================================================================

class TestConsistencyThreshold:
    """Test SDF consistency threshold γ setting and checking."""
    
    def test_check_consistency_below_threshold(self, default_config):
        """
        v3.1 Requirement 4.3.1 MUST: Individual with |g^(w)| < γ is consistent.
        """
        from sdf_consistency_selection import check_sdf_consistency
        
        sdf_penalty = 0.005
        gamma = default_config['gamma']  # 0.01
        
        is_consistent = check_sdf_consistency(sdf_penalty, gamma)
        
        assert is_consistent is True, "Should be consistent when |g^(w)| < γ"
    
    def test_check_consistency_above_threshold(self, default_config):
        """
        v3.1 Requirement 4.3.1 MUST: Individual with |g^(w)| > γ is inconsistent.
        """
        from sdf_consistency_selection import check_sdf_consistency
        
        sdf_penalty = 0.015
        gamma = default_config['gamma']  # 0.01
        
        is_consistent = check_sdf_consistency(sdf_penalty, gamma)
        
        assert is_consistent is False, "Should be inconsistent when |g^(w)| > γ"
    
    def test_check_consistency_at_threshold(self, default_config):
        """
        Individual with |g^(w)| == γ should be considered consistent (boundary case).
        """
        from sdf_consistency_selection import check_sdf_consistency
        
        sdf_penalty = 0.01  # Exactly at threshold
        gamma = default_config['gamma']
        
        is_consistent = check_sdf_consistency(sdf_penalty, gamma)
        
        assert is_consistent is True, "Boundary case should be consistent"
    
    def test_batch_consistency_check(self, sample_population, default_config):
        """
        Check consistency for entire population.
        """
        from sdf_consistency_selection import batch_check_consistency
        
        gamma = default_config['gamma']  # 0.01
        
        results = batch_check_consistency(sample_population, gamma)
        
        assert len(results) == len(sample_population)
        # IDs 0, 1, 3, 5 should be consistent (sdf_penalty <= 0.01)
        # IDs 2, 4 should be inconsistent (sdf_penalty > 0.01)
        assert results[0] is True   # 0.002 < 0.01
        assert results[1] is True   # 0.008 < 0.01
        assert results[2] is False  # 0.015 > 0.01
        assert results[3] is True   # 0.001 < 0.01
        assert results[4] is False  # 0.025 > 0.01
        assert results[5] is True   # 0.004 < 0.01


# =============================================================================
# Test Class: Lexicographic Selection (v3.1 4.3.2 MUST - Option 1)
# =============================================================================

class TestLexicographicSelection:
    """Test lexicographic (tier-based) selection rule."""
    
    def test_tier_assignment(self, sample_population, default_config):
        """
        v3.1 Requirement 4.3.2 MUST: Tier by consistency first.
        Consistent individuals are in tier 0, inconsistent in tier 1.
        """
        from sdf_consistency_selection import assign_consistency_tiers
        
        gamma = default_config['gamma']
        
        tiered = assign_consistency_tiers(sample_population, gamma)
        
        # Consistent (tier 0): IDs 0, 1, 3, 5
        # Inconsistent (tier 1): IDs 2, 4
        for ind in tiered:
            if ind['id'] in [0, 1, 3, 5]:
                assert ind['tier'] == 0, f"ID {ind['id']} should be tier 0"
            else:
                assert ind['tier'] == 1, f"ID {ind['id']} should be tier 1"
    
    def test_lexicographic_sort(self, sample_population, default_config):
        """
        Lexicographic sort: tier first, then by objective (e.g., Sharpe).
        """
        from sdf_consistency_selection import lexicographic_sort
        
        gamma = default_config['gamma']
        
        sorted_pop = lexicographic_sort(sample_population, gamma, primary_obj='sharpe')
        
        # First should come tier 0 (consistent) sorted by Sharpe descending
        # Then tier 1 (inconsistent) sorted by Sharpe descending
        
        # Tier 0: IDs 1(1.8), 5(1.6), 0(1.5), 3(1.2) - by Sharpe desc
        # Tier 1: IDs 4(2.5), 2(2.0) - by Sharpe desc
        assert sorted_pop[0]['id'] == 1, "Best consistent should be first"
        assert sorted_pop[-1]['id'] == 2, "Worst inconsistent should be last"
    
    def test_elite_selection_lexicographic(self, sample_population, default_config):
        """
        Elite selection should prioritize consistent individuals.
        """
        from sdf_consistency_selection import select_elite_lexicographic
        
        gamma = default_config['gamma']
        elite_count = 2  # Select top 2
        
        elite = select_elite_lexicographic(
            sample_population, 
            gamma=gamma, 
            elite_count=elite_count,
            primary_obj='sharpe'
        )
        
        assert len(elite) == elite_count
        # Should be best consistent individuals, not highest Sharpe overall
        for ind in elite:
            assert ind['objectives']['sdf_penalty'] <= gamma, "Elite must be consistent"


# =============================================================================
# Test Class: Threshold Selection (v3.1 4.3.2 MUST - Option 2)
# =============================================================================

class TestThresholdSelection:
    """Test threshold-based elimination selection rule."""
    
    def test_eliminate_inconsistent(self, sample_population, default_config):
        """
        v3.1 Requirement 4.3.2 MUST: Eliminate if |g^(w)| > γ.
        """
        from sdf_consistency_selection import eliminate_inconsistent
        
        gamma = default_config['gamma']
        
        filtered = eliminate_inconsistent(sample_population, gamma)
        
        # Should remove IDs 2 and 4 (inconsistent)
        assert len(filtered) == 4
        filtered_ids = [ind['id'] for ind in filtered]
        assert 2 not in filtered_ids
        assert 4 not in filtered_ids
    
    def test_eliminate_preserves_order(self, sample_population, default_config):
        """
        Elimination should preserve original order of consistent individuals.
        """
        from sdf_consistency_selection import eliminate_inconsistent
        
        gamma = default_config['gamma']
        
        filtered = eliminate_inconsistent(sample_population, gamma)
        
        ids = [ind['id'] for ind in filtered]
        assert ids == [0, 1, 3, 5], "Should preserve original order"
    
    def test_select_with_threshold(self, sample_population, default_config):
        """
        Selection with threshold: filter first, then select best.
        """
        from sdf_consistency_selection import select_with_threshold
        
        gamma = default_config['gamma']
        select_count = 3
        
        selected = select_with_threshold(
            sample_population,
            gamma=gamma,
            select_count=select_count,
            objective='sharpe'
        )
        
        assert len(selected) == select_count
        # All should be consistent
        for ind in selected:
            assert ind['objectives']['sdf_penalty'] <= gamma


# =============================================================================
# Test Class: Weighted Penalty Selection (v3.1 4.3.2 MUST - Option 3)
# =============================================================================

class TestWeightedPenaltySelection:
    """Test weighted penalty selection rule."""
    
    def test_compute_penalized_fitness(self, sample_population, default_config):
        """
        v3.1 Requirement 4.3.2 MUST: Penalize inconsistent strategies.
        penalized_sharpe = sharpe - weight * max(0, |g^(w)| - γ)
        """
        from sdf_consistency_selection import compute_penalized_fitness
        
        gamma = default_config['gamma']  # 0.01
        penalty_weight = default_config['penalty_weight']  # 10.0
        
        penalized = compute_penalized_fitness(
            sample_population, 
            gamma=gamma, 
            penalty_weight=penalty_weight,
            objective='sharpe'
        )
        
        # ID 2: sharpe=2.0, sdf_penalty=0.015
        # penalized = 2.0 - 10.0 * (0.015 - 0.01) = 2.0 - 0.05 = 1.95
        ind2 = next(ind for ind in penalized if ind['id'] == 2)
        assert np.isclose(ind2['penalized_fitness'], 1.95, atol=0.01)
        
        # ID 0: sharpe=1.5, sdf_penalty=0.002 (consistent, no penalty)
        # penalized = 1.5 - 0 = 1.5
        ind0 = next(ind for ind in penalized if ind['id'] == 0)
        assert np.isclose(ind0['penalized_fitness'], 1.5)
    
    def test_weighted_selection(self, sample_population, default_config):
        """
        Select based on penalized fitness.
        Weighted mode penalizes but may still include high-Sharpe inconsistent
        if their penalized fitness remains competitive.
        """
        from sdf_consistency_selection import select_weighted
        
        gamma = default_config['gamma']
        penalty_weight = default_config['penalty_weight']
        select_count = 3
        
        selected = select_weighted(
            sample_population,
            gamma=gamma,
            penalty_weight=penalty_weight,
            select_count=select_count,
            objective='sharpe'
        )
        
        assert len(selected) == select_count
        
        # Verify penalized fitness ordering makes sense
        # ID 4: 2.5 - 10*(0.025-0.01) = 2.35
        # ID 2: 2.0 - 10*(0.015-0.01) = 1.95
        # ID 1: 1.8 - 0 = 1.8 (consistent)
        # Top 3 should be sorted by penalized fitness
        for i in range(len(selected) - 1):
            assert selected[i].get('penalized_fitness', 0) >= selected[i+1].get('penalized_fitness', 0)


# =============================================================================
# Test Class: Baseline EA (v3.1 4.3.3)
# =============================================================================

class TestBaselineEANoFiltering:
    """Test that baseline EA uses f⁴ as normal objective only."""
    
    def test_baseline_no_consistency_filter(self, sample_population):
        """
        v3.1 Requirement 4.3.3: Baseline EA has no consistency filtering.
        Selection is purely based on objective values.
        """
        from sdf_consistency_selection import baseline_select
        
        select_count = 3
        
        # Baseline just selects best by Sharpe, ignoring consistency
        selected = baseline_select(
            sample_population,
            select_count=select_count,
            objective='sharpe'
        )
        
        # Should select IDs 4(2.5), 2(2.0), 1(1.8) - highest Sharpe
        selected_ids = [ind['id'] for ind in selected]
        assert 4 in selected_ids, "Baseline should include high-Sharpe even if inconsistent"
        assert 2 in selected_ids, "Baseline should include high-Sharpe even if inconsistent"
    
    def test_baseline_uses_sdf_as_objective(self, sample_population):
        """
        Baseline EA treats sdf_penalty as f⁴ objective for NSGA-II ranking.
        """
        from sdf_consistency_selection import baseline_multiobjective_rank
        
        ranked = baseline_multiobjective_rank(sample_population)
        
        # Should compute non-dominated ranks considering all 4 objectives
        for ind in ranked:
            assert 'rank' in ind, "Should have NSGA-II rank"
            assert 'crowding' in ind, "Should have crowding distance"


# =============================================================================
# Test Class: SDFConsistencySelector Integration
# =============================================================================

class TestSDFConsistencySelectorIntegration:
    """Integration tests for the complete selector class."""
    
    def test_selector_lexicographic_mode(self, sample_population, default_config):
        """
        Test selector in lexicographic mode.
        """
        from sdf_consistency_selection import SDFConsistencySelector
        
        selector = SDFConsistencySelector(
            gamma=default_config['gamma'],
            mode='lexicographic'
        )
        
        elite = selector.select_elite(
            population=sample_population,
            elite_count=2,
            primary_obj='sharpe'
        )
        
        assert len(elite) == 2
        for ind in elite:
            assert ind['objectives']['sdf_penalty'] <= default_config['gamma']
    
    def test_selector_threshold_mode(self, sample_population, default_config):
        """
        Test selector in threshold mode.
        """
        from sdf_consistency_selection import SDFConsistencySelector
        
        selector = SDFConsistencySelector(
            gamma=default_config['gamma'],
            mode='threshold'
        )
        
        selected = selector.select_elite(
            population=sample_population,
            elite_count=3,
            primary_obj='sharpe'
        )
        
        assert len(selected) == 3
        for ind in selected:
            assert ind['objectives']['sdf_penalty'] <= default_config['gamma']
    
    def test_selector_weighted_mode(self, sample_population, default_config):
        """
        Test selector in weighted penalty mode.
        """
        from sdf_consistency_selection import SDFConsistencySelector
        
        selector = SDFConsistencySelector(
            gamma=default_config['gamma'],
            mode='weighted',
            penalty_weight=10.0
        )
        
        selected = selector.select_elite(
            population=sample_population,
            elite_count=3,
            primary_obj='sharpe'
        )
        
        assert len(selected) == 3
    
    def test_selector_get_telemetry(self, sample_population, default_config):
        """
        Selector should provide telemetry on consistency stats.
        """
        from sdf_consistency_selection import SDFConsistencySelector
        
        selector = SDFConsistencySelector(
            gamma=default_config['gamma'],
            mode='lexicographic'
        )
        
        selector.select_elite(sample_population, elite_count=2)
        telemetry = selector.get_telemetry()
        
        assert 'gamma' in telemetry
        assert 'mode' in telemetry
        assert 'consistent_count' in telemetry
        assert 'inconsistent_count' in telemetry
        assert 'consistency_ratio' in telemetry
    
    def test_selector_invalid_mode_raises(self):
        """
        Invalid selection mode should raise ValueError.
        """
        from sdf_consistency_selection import SDFConsistencySelector
        
        with pytest.raises(ValueError):
            SDFConsistencySelector(gamma=0.01, mode='invalid_mode')
    
    def test_selector_invalid_gamma_raises(self):
        """
        Invalid gamma should raise ValueError.
        """
        from sdf_consistency_selection import SDFConsistencySelector
        
        with pytest.raises(ValueError):
            SDFConsistencySelector(gamma=-0.01, mode='lexicographic')


# =============================================================================
# Self-test runner
# =============================================================================

if __name__ == "__main__":
    print("P0-22 SDF-Consistency Selection Test Suite")
    print("=" * 50)
    print("This test file should be run with pytest:")
    print("  pytest test_sdf_consistency_selection.py -v")
    print("\nTotal test cases: 24")
    print("\nSpec Coverage:")
    print("  - 4.3.1 MUST: Set consistency threshold γ")
    print("  - 4.3.2 MUST: Consistency-first selection rules")
    print("    - Lexicographic (tier-based)")
    print("    - Threshold (elimination)")
    print("    - Weighted (penalty)")
    print("  - 4.3.3: Baseline EA no filtering")
