"""
Test Suite for EA Evaluation Metrics Module (P0-19).

This test file is written BEFORE implementation following TDD principles.
Tests define the expected behavior based on EA Layer v3.1 specification.

Tested Module: projects/dgsf/scripts/ea_evaluation_metrics.py
Spec Reference: DGSF EA Layer Specification v3.1

Test Categories:
1. Hypervolume (HV) computation
2. HV trajectory tracking
3. HV plateau detection (v3.1 MUST: HV-aware behaviour)
4. Strategy drift calculation (v3.1 MUST: drift-aware warm-start)
5. SDF consistency filtering (v3.1 MUST: consistency threshold γ)

Author: DGSF Pipeline
Date: 2026-02-04
Stage: 5 (EA Optimizer Development)
Task: EA_DEV_001 P0-19
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure scripts directory is importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_pareto_front_2d():
    """2D Pareto front for simple HV testing."""
    # 4 non-dominated solutions in 2-objective space
    # Objectives to MINIMIZE: (f1, f2)
    return np.array([
        [1.0, 4.0],  # Solution A
        [2.0, 3.0],  # Solution B
        [3.0, 2.0],  # Solution C
        [4.0, 1.0],  # Solution D
    ])


@pytest.fixture
def reference_point_2d():
    """Reference point for 2D HV (must dominate all solutions)."""
    return np.array([5.0, 5.0])


@pytest.fixture
def sample_pareto_front_4obj():
    """4-objective Pareto front (EA v3.1: Sharpe, MDD, Turnover, PE)."""
    # 10 non-dominated solutions
    # f1: -Sharpe (minimize, so lower = better Sharpe)
    # f2: MDD (minimize)
    # f3: Turnover (minimize)
    # f4: PE (minimize)
    np.random.seed(42)
    n_solutions = 10
    front = np.zeros((n_solutions, 4))
    for i in range(n_solutions):
        front[i, 0] = -2.0 + i * 0.2  # f1: -Sharpe from -2.0 to 0.0
        front[i, 1] = 0.01 + i * 0.01  # f2: MDD from 0.01 to 0.10
        front[i, 2] = 0.1 + i * 0.05  # f3: Turnover from 0.1 to 0.55
        front[i, 3] = 0.001 + i * 0.001  # f4: PE from 0.001 to 0.010
    return front


@pytest.fixture
def reference_point_4obj():
    """Reference point for 4D HV (must dominate all solutions)."""
    # All objectives minimized, so ref point must be larger than max
    return np.array([1.0, 1.0, 1.0, 0.1])


@pytest.fixture
def hv_trajectory_plateau():
    """HV trajectory with plateau (for plateau detection test)."""
    # Generations 0-10: increasing HV
    # Generations 11-15: plateau (< 1% change)
    return [
        0.10, 0.15, 0.22, 0.30, 0.38,  # gen 0-4: growing
        0.45, 0.51, 0.56, 0.60, 0.63,  # gen 5-9: growing
        0.65,  # gen 10: peak
        0.651, 0.652, 0.652, 0.653, 0.653,  # gen 11-15: plateau
    ]


@pytest.fixture
def hv_trajectory_no_plateau():
    """HV trajectory without plateau (steady growth)."""
    return [0.1 + i * 0.05 for i in range(20)]


@pytest.fixture
def weights_current():
    """Current window weights for drift calculation."""
    return np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02, 0.0])


@pytest.fixture
def weights_previous():
    """Previous window weights for drift calculation."""
    return np.array([0.25, 0.25, 0.1, 0.1, 0.1, 0.08, 0.05, 0.04, 0.02, 0.01])


@pytest.fixture
def pareto_front_with_pe():
    """Pareto front with PE values for consistency filtering."""
    # 10 solutions with varying PE (f4)
    # gamma threshold = 0.005
    return np.array([
        [-2.0, 0.02, 0.1, 0.001],   # PE < gamma ✓
        [-1.8, 0.03, 0.15, 0.003],  # PE < gamma ✓
        [-1.6, 0.04, 0.2, 0.004],   # PE < gamma ✓
        [-1.4, 0.05, 0.25, 0.005],  # PE = gamma (boundary) ✓
        [-1.2, 0.06, 0.3, 0.006],   # PE > gamma ✗
        [-1.0, 0.07, 0.35, 0.008],  # PE > gamma ✗
        [-0.8, 0.08, 0.4, 0.010],   # PE > gamma ✗
        [-0.6, 0.09, 0.45, 0.002],  # PE < gamma ✓
        [-0.4, 0.10, 0.5, 0.015],   # PE > gamma ✗
        [-0.2, 0.11, 0.55, 0.001],  # PE < gamma ✓
    ])


# =============================================================================
# Test Class: Hypervolume Computation
# =============================================================================

class TestHypervolumeComputation:
    """Test HV computation correctness."""

    def test_hv_2d_simple_front(self, sample_pareto_front_2d, reference_point_2d):
        """
        Test HV computation on simple 2D Pareto front.
        
        Expected HV can be calculated analytically:
        HV = sum of rectangle areas from each point to reference
        Using WFG algorithm or analytical: HV ≈ 7.0
        """
        from ea_evaluation_metrics import compute_hypervolume
        
        hv = compute_hypervolume(sample_pareto_front_2d, reference_point_2d)
        
        # Analytical HV for this front ≈ 7.0
        # (5-1)*(5-4) + (5-2)*(4-3) + (5-3)*(3-2) + (5-4)*(2-1) = 4 + 3 + 2 + 1 = 10
        # But with dominated region subtraction: 4 + 3 - 1 + 2 - 1 + 1 = 7
        assert hv > 0, "HV must be positive for valid front"
        assert 6.0 < hv < 12.0, f"HV should be ~7-10, got {hv}"

    def test_hv_4d_front(self, sample_pareto_front_4obj, reference_point_4obj):
        """
        Test HV computation on 4-objective Pareto front (EA v3.1 scenario).
        """
        from ea_evaluation_metrics import compute_hypervolume
        
        hv = compute_hypervolume(sample_pareto_front_4obj, reference_point_4obj)
        
        assert hv > 0, "HV must be positive"
        assert isinstance(hv, float), "HV must be a float"

    def test_hv_single_point(self, reference_point_2d):
        """Test HV with single solution."""
        from ea_evaluation_metrics import compute_hypervolume
        
        single_point = np.array([[2.0, 2.0]])
        hv = compute_hypervolume(single_point, reference_point_2d)
        
        # HV of single point = (5-2) * (5-2) = 9
        assert hv > 0, "HV of single point must be positive"
        assert abs(hv - 9.0) < 0.1, f"Single point HV should be 9.0, got {hv}"

    def test_hv_empty_front(self, reference_point_2d):
        """Test HV with empty front returns 0."""
        from ea_evaluation_metrics import compute_hypervolume
        
        empty_front = np.array([]).reshape(0, 2)
        hv = compute_hypervolume(empty_front, reference_point_2d)
        
        assert hv == 0.0, "Empty front should have HV = 0"

    def test_hv_invalid_reference_point_raises(self, sample_pareto_front_2d):
        """Test that invalid reference point raises error."""
        from ea_evaluation_metrics import compute_hypervolume
        
        # Reference point that does NOT dominate all solutions
        invalid_ref = np.array([2.0, 2.0])  # Solution (4, 1) not dominated
        
        with pytest.raises(ValueError, match="[Rr]eference.*domin"):
            compute_hypervolume(sample_pareto_front_2d, invalid_ref)


# =============================================================================
# Test Class: HV Trajectory Tracking
# =============================================================================

class TestHVTrajectory:
    """Test HV trajectory tracking across generations."""

    def test_compute_hv_trajectory(self, sample_pareto_front_4obj, reference_point_4obj):
        """Test HV trajectory computation from generation history."""
        from ea_evaluation_metrics import compute_hv_trajectory
        
        # Simulate 5 generations with expanding fronts
        generation_fronts = []
        for gen in range(5):
            # Each generation has more solutions
            front_size = gen + 3
            front = sample_pareto_front_4obj[:front_size].copy()
            generation_fronts.append(front)
        
        trajectory = compute_hv_trajectory(generation_fronts, reference_point_4obj)
        
        assert len(trajectory) == 5, "Trajectory should have 5 points"
        assert all(hv >= 0 for hv in trajectory), "All HV values must be non-negative"
        # Generally HV should be non-decreasing (more solutions = equal or higher HV)
        for i in range(1, len(trajectory)):
            assert trajectory[i] >= trajectory[i-1] - 1e-9, \
                f"HV should be non-decreasing: gen {i-1}={trajectory[i-1]}, gen {i}={trajectory[i]}"

    def test_hv_trajectory_returns_list(self, sample_pareto_front_4obj, reference_point_4obj):
        """Test that trajectory returns a list of floats."""
        from ea_evaluation_metrics import compute_hv_trajectory
        
        generation_fronts = [sample_pareto_front_4obj for _ in range(3)]
        trajectory = compute_hv_trajectory(generation_fronts, reference_point_4obj)
        
        assert isinstance(trajectory, list), "Trajectory must be a list"
        assert all(isinstance(hv, float) for hv in trajectory), "All values must be floats"


# =============================================================================
# Test Class: HV Plateau Detection (v3.1 MUST)
# =============================================================================

class TestHVPlateauDetection:
    """Test HV plateau detection for v3.1 HV-aware behaviour."""

    def test_detect_plateau_when_present(self, hv_trajectory_plateau):
        """
        Test plateau detection when HV stagnates.
        
        v3.1 Requirement 4.1.2 MUST: Detect HV plateau after G_plateau generations.
        """
        from ea_evaluation_metrics import detect_hv_plateau
        
        # Check at generation 15 (after plateau starts at gen 11)
        is_plateau, plateau_start = detect_hv_plateau(
            hv_trajectory_plateau,
            g_plateau=5,  # Require 5 generations of stagnation
            epsilon=0.01  # 1% threshold
        )
        
        assert is_plateau is True, "Should detect plateau"
        assert plateau_start == 11, f"Plateau should start at gen 11, got {plateau_start}"

    def test_no_plateau_when_growing(self, hv_trajectory_no_plateau):
        """Test no plateau detected when HV is steadily growing."""
        from ea_evaluation_metrics import detect_hv_plateau
        
        is_plateau, plateau_start = detect_hv_plateau(
            hv_trajectory_no_plateau,
            g_plateau=5,
            epsilon=0.01
        )
        
        assert is_plateau is False, "Should not detect plateau in growing trajectory"
        assert plateau_start is None, "Plateau start should be None"

    def test_plateau_detection_short_trajectory(self):
        """Test plateau detection with trajectory shorter than g_plateau."""
        from ea_evaluation_metrics import detect_hv_plateau
        
        short_trajectory = [0.1, 0.2, 0.3]
        is_plateau, plateau_start = detect_hv_plateau(
            short_trajectory,
            g_plateau=5,
            epsilon=0.01
        )
        
        assert is_plateau is False, "Cannot detect plateau with short trajectory"

    def test_plateau_detection_returns_correct_types(self, hv_trajectory_plateau):
        """Test return types of plateau detection."""
        from ea_evaluation_metrics import detect_hv_plateau
        
        is_plateau, plateau_start = detect_hv_plateau(
            hv_trajectory_plateau,
            g_plateau=5,
            epsilon=0.01
        )
        
        assert isinstance(is_plateau, bool), "is_plateau must be bool"
        assert plateau_start is None or isinstance(plateau_start, int), \
            "plateau_start must be int or None"


# =============================================================================
# Test Class: Strategy Drift Calculation (v3.1 MUST)
# =============================================================================

class TestStrategyDrift:
    """Test strategy drift calculation for v3.1 drift-aware warm-start."""

    def test_compute_strategy_drift_l1(self, weights_current, weights_previous):
        """
        Test L1 drift calculation: |w^{j} - w^{j-1}|_1
        
        v3.1 Requirement 4.2.3 MUST: Record strategy drift.
        """
        from ea_evaluation_metrics import compute_strategy_drift
        
        drift = compute_strategy_drift(weights_current, weights_previous)
        
        # Expected: sum of absolute differences
        expected_drift = np.sum(np.abs(weights_current - weights_previous))
        
        assert isinstance(drift, float), "Drift must be a float"
        assert abs(drift - expected_drift) < 1e-9, \
            f"Drift mismatch: expected {expected_drift}, got {drift}"

    def test_zero_drift_identical_weights(self, weights_current):
        """Test zero drift when weights are identical."""
        from ea_evaluation_metrics import compute_strategy_drift
        
        drift = compute_strategy_drift(weights_current, weights_current.copy())
        
        assert drift == 0.0, "Identical weights should have zero drift"

    def test_max_drift_opposite_weights(self):
        """Test maximum drift when weights completely flip."""
        from ea_evaluation_metrics import compute_strategy_drift
        
        w1 = np.array([1.0, 0.0, 0.0])
        w2 = np.array([0.0, 0.0, 1.0])
        
        drift = compute_strategy_drift(w1, w2)
        
        assert drift == 2.0, "Opposite weights should have drift = 2.0"

    def test_drift_shape_mismatch_raises(self):
        """Test that mismatched weight shapes raise error."""
        from ea_evaluation_metrics import compute_strategy_drift
        
        w1 = np.array([0.5, 0.3, 0.2])
        w2 = np.array([0.5, 0.5])
        
        with pytest.raises(ValueError, match="shape"):
            compute_strategy_drift(w1, w2)


# =============================================================================
# Test Class: SDF Consistency Filtering (v3.1 MUST)
# =============================================================================

class TestSDFConsistencyFiltering:
    """Test SDF consistency filtering for v3.1 requirement."""

    def test_filter_by_consistency_threshold(self, pareto_front_with_pe):
        """
        Test filtering solutions by |g^(w)| ≤ γ.
        
        v3.1 Requirement 4.3.1 MUST: Set consistency threshold γ.
        """
        from ea_evaluation_metrics import filter_sdf_consistent
        
        gamma = 0.005
        consistent_mask, consistent_count = filter_sdf_consistent(
            pareto_front_with_pe,
            gamma=gamma,
            pe_column_idx=3  # f4 is column 3
        )
        
        # Expected: 6 solutions with PE ≤ 0.005
        assert consistent_count == 6, f"Expected 6 consistent solutions, got {consistent_count}"
        assert isinstance(consistent_mask, np.ndarray), "Mask must be numpy array"
        assert consistent_mask.dtype == bool, "Mask must be boolean"
        assert len(consistent_mask) == 10, "Mask length must match front size"

    def test_consistency_ratio(self, pareto_front_with_pe):
        """Test consistency ratio calculation."""
        from ea_evaluation_metrics import compute_consistency_ratio
        
        gamma = 0.005
        ratio = compute_consistency_ratio(
            pareto_front_with_pe,
            gamma=gamma,
            pe_column_idx=3
        )
        
        assert isinstance(ratio, float), "Ratio must be a float"
        assert 0.0 <= ratio <= 1.0, "Ratio must be between 0 and 1"
        assert abs(ratio - 0.6) < 0.01, f"Expected ratio ~0.6, got {ratio}"

    def test_all_consistent_when_gamma_high(self, pareto_front_with_pe):
        """Test all solutions consistent when gamma is very high."""
        from ea_evaluation_metrics import filter_sdf_consistent
        
        gamma = 1.0  # Very high threshold
        _, consistent_count = filter_sdf_consistent(
            pareto_front_with_pe,
            gamma=gamma,
            pe_column_idx=3
        )
        
        assert consistent_count == 10, "All solutions should be consistent with high gamma"

    def test_none_consistent_when_gamma_zero(self, pareto_front_with_pe):
        """Test no solutions consistent when gamma is zero."""
        from ea_evaluation_metrics import filter_sdf_consistent
        
        gamma = 0.0  # Zero threshold (only exact zero PE allowed)
        _, consistent_count = filter_sdf_consistent(
            pareto_front_with_pe,
            gamma=gamma,
            pe_column_idx=3
        )
        
        assert consistent_count == 0, "No solutions should be consistent with gamma=0"


# =============================================================================
# Test Class: Full Evaluation Summary
# =============================================================================

class TestEAEvaluationSummary:
    """Test the complete EA evaluation summary function."""

    def test_evaluate_pareto_front_full(
        self,
        sample_pareto_front_4obj,
        reference_point_4obj,
        weights_current,
        weights_previous,
    ):
        """Test full evaluation summary for a Pareto front."""
        from ea_evaluation_metrics import evaluate_pareto_front
        
        summary = evaluate_pareto_front(
            pareto_front=sample_pareto_front_4obj,
            reference_point=reference_point_4obj,
            current_weights=weights_current,
            previous_weights=weights_previous,
            gamma=0.005,
            pe_column_idx=3,
        )
        
        # Check all expected keys are present
        expected_keys = [
            "hypervolume",
            "solution_count",
            "consistent_count",
            "consistency_ratio",
            "strategy_drift",
            "best_sharpe",
            "mean_mdd",
            "mean_turnover",
            "mean_pe",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"
        
        # Validate types
        assert isinstance(summary["hypervolume"], float)
        assert isinstance(summary["solution_count"], int)
        assert isinstance(summary["consistent_count"], int)
        assert isinstance(summary["consistency_ratio"], float)
        assert isinstance(summary["strategy_drift"], float)
        assert isinstance(summary["best_sharpe"], float)

    def test_evaluate_returns_reasonable_values(
        self,
        sample_pareto_front_4obj,
        reference_point_4obj,
        weights_current,
        weights_previous,
    ):
        """Test that evaluation returns reasonable values."""
        from ea_evaluation_metrics import evaluate_pareto_front
        
        summary = evaluate_pareto_front(
            pareto_front=sample_pareto_front_4obj,
            reference_point=reference_point_4obj,
            current_weights=weights_current,
            previous_weights=weights_previous,
            gamma=0.005,
            pe_column_idx=3,
        )
        
        assert summary["hypervolume"] >= 0, "HV must be non-negative"
        assert summary["solution_count"] == 10, "Should have 10 solutions"
        assert 0 <= summary["consistency_ratio"] <= 1, "Ratio must be [0, 1]"
        assert summary["strategy_drift"] >= 0, "Drift must be non-negative"


# =============================================================================
# Test Class: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_hv_with_nan_raises(self, reference_point_2d):
        """Test that NaN in front raises error."""
        from ea_evaluation_metrics import compute_hypervolume
        
        front_with_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
        
        with pytest.raises(ValueError, match="NaN|nan"):
            compute_hypervolume(front_with_nan, reference_point_2d)

    def test_hv_with_inf_raises(self, reference_point_2d):
        """Test that Inf in front raises error."""
        from ea_evaluation_metrics import compute_hypervolume
        
        front_with_inf = np.array([[1.0, np.inf], [2.0, 3.0]])
        
        with pytest.raises(ValueError, match="Inf|inf|finite"):
            compute_hypervolume(front_with_inf, reference_point_2d)

    def test_drift_with_non_normalized_weights(self):
        """Test drift calculation with non-normalized weights still works."""
        from ea_evaluation_metrics import compute_strategy_drift
        
        w1 = np.array([2.0, 3.0, 5.0])  # Sum = 10
        w2 = np.array([1.0, 4.0, 5.0])  # Sum = 10
        
        drift = compute_strategy_drift(w1, w2)
        
        # |2-1| + |3-4| + |5-5| = 1 + 1 + 0 = 2
        assert abs(drift - 2.0) < 1e-9, f"Drift should be 2.0, got {drift}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
