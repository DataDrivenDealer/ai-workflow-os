"""
EA Evaluation Metrics Module for DGSF Stage 5 (P0-19).

This module provides evaluation metrics for EA Layer v3.1, including:
1. Hypervolume (HV) computation
2. HV trajectory tracking
3. HV plateau detection (v3.1 MUST: HV-aware behaviour)
4. Strategy drift calculation (v3.1 MUST: drift-aware warm-start)
5. SDF consistency filtering (v3.1 MUST: consistency threshold γ)

Spec Reference: DGSF EA Layer Specification v3.1

Author: DGSF Pipeline
Date: 2026-02-04
Stage: 5 (EA Optimizer Development)
Task: EA_DEV_001 P0-19
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


# =============================================================================
# Hypervolume Computation
# =============================================================================

def compute_hypervolume(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """
    Compute the hypervolume indicator for a Pareto front.
    
    Uses the WFG algorithm for 2D and a simplified recursive approach for nD.
    All objectives are assumed to be MINIMIZED.
    
    Parameters
    ----------
    pareto_front : np.ndarray
        Pareto front solutions, shape [N, M] where N is number of solutions
        and M is number of objectives.
    reference_point : np.ndarray
        Reference point, shape [M]. Must dominate all solutions (be worse in all objectives).
    
    Returns
    -------
    hv : float
        Hypervolume indicator value.
    
    Raises
    ------
    ValueError
        If reference point does not dominate all solutions, or if front contains NaN/Inf.
    
    Examples
    --------
    >>> front = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    >>> ref = np.array([5.0, 5.0])
    >>> hv = compute_hypervolume(front, ref)
    """
    # Validate inputs
    if pareto_front.size == 0:
        return 0.0
    
    if pareto_front.ndim == 1:
        pareto_front = pareto_front.reshape(1, -1)
    
    # Check for NaN/Inf
    if np.any(np.isnan(pareto_front)):
        raise ValueError("Pareto front contains NaN values")
    if np.any(np.isinf(pareto_front)):
        raise ValueError("Pareto front contains Inf values (must be finite)")
    
    n_solutions, n_objectives = pareto_front.shape
    
    # Check reference point dominates all solutions
    # For minimization: ref[i] >= max(front[:, i]) for all i
    for i in range(n_objectives):
        max_val = pareto_front[:, i].max()
        if reference_point[i] < max_val:
            raise ValueError(
                f"Reference point does not dominate all solutions. "
                f"Objective {i}: ref={reference_point[i]} < max={max_val}"
            )
    
    if n_objectives == 2:
        return _compute_hv_2d(pareto_front, reference_point)
    else:
        return _compute_hv_nd(pareto_front, reference_point)


def _compute_hv_2d(front: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute 2D hypervolume using the sweeping algorithm.
    
    Time complexity: O(N log N) where N is number of solutions.
    """
    # Sort by first objective (ascending)
    sorted_indices = np.argsort(front[:, 0])
    sorted_front = front[sorted_indices]
    
    hv = 0.0
    prev_y = ref[1]  # Start from reference y
    
    for point in sorted_front:
        x, y = point
        # Rectangle width * height
        # Width: ref[0] - x (from point to reference)
        # Height: prev_y - y (from previous y level to current y)
        if y < prev_y:  # Only count if y improves
            width = ref[0] - x
            height = prev_y - y
            hv += width * height
            prev_y = y
    
    return float(hv)


def _compute_hv_nd(front: np.ndarray, ref: np.ndarray) -> float:
    """
    Compute n-dimensional hypervolume using WFG/HSO algorithm.
    
    This is a simplified recursive implementation suitable for small dimensions (up to 5-6).
    For higher dimensions, specialized libraries should be used.
    """
    n_solutions, n_objectives = front.shape
    
    if n_solutions == 0:
        return 0.0
    
    if n_solutions == 1:
        # Volume of single hypercuboid
        return float(np.prod(ref - front[0]))
    
    if n_objectives == 1:
        # 1D case: just the difference
        return float(ref[0] - front[:, 0].min())
    
    # Recursive inclusion-exclusion for small cases
    # For production, use pymoo or deap's HV implementation
    
    # Sort by last objective
    sorted_indices = np.argsort(front[:, -1])
    sorted_front = front[sorted_indices]
    
    hv = 0.0
    prev_value = ref[-1]
    
    for i, point in enumerate(sorted_front):
        # Slice: all objectives except last
        if point[-1] < prev_value:
            # Height in last dimension
            height = prev_value - point[-1]
            
            # Dominated subfront in remaining dimensions
            subfront = sorted_front[:i+1, :-1]
            subref = ref[:-1]
            
            # Recursive call (or 2D base case)
            if n_objectives - 1 == 2:
                sub_hv = _compute_hv_2d(subfront, subref)
            else:
                sub_hv = _compute_hv_nd(subfront, subref)
            
            hv += height * sub_hv
            prev_value = point[-1]
    
    return float(hv)


# =============================================================================
# HV Trajectory Tracking
# =============================================================================

def compute_hv_trajectory(
    generation_fronts: List[np.ndarray],
    reference_point: np.ndarray,
) -> List[float]:
    """
    Compute HV trajectory across generations.
    
    Parameters
    ----------
    generation_fronts : List[np.ndarray]
        List of Pareto fronts, one per generation. Each front has shape [N_g, M].
    reference_point : np.ndarray
        Reference point for HV computation, shape [M].
    
    Returns
    -------
    trajectory : List[float]
        HV value at each generation.
    
    Examples
    --------
    >>> fronts = [front_gen0, front_gen1, front_gen2]
    >>> trajectory = compute_hv_trajectory(fronts, ref_point)
    >>> print(trajectory)  # [0.5, 0.7, 0.85]
    """
    trajectory = []
    for front in generation_fronts:
        if front.size == 0:
            trajectory.append(0.0)
        else:
            hv = compute_hypervolume(front, reference_point)
            trajectory.append(float(hv))
    return trajectory


# =============================================================================
# HV Plateau Detection (v3.1 MUST: Section 4.1)
# =============================================================================

def detect_hv_plateau(
    hv_trajectory: List[float],
    g_plateau: int = 5,
    epsilon: float = 0.01,
) -> Tuple[bool, Optional[int]]:
    """
    Detect HV plateau in trajectory.
    
    v3.1 Requirement 4.1.2 MUST: Detect HV plateau after G_plateau generations
    of insignificant improvement.
    
    Parameters
    ----------
    hv_trajectory : List[float]
        HV values at each generation.
    g_plateau : int
        Number of consecutive generations with < epsilon improvement to detect plateau.
    epsilon : float
        Relative improvement threshold. Plateau if ΔHV/HV < epsilon.
    
    Returns
    -------
    is_plateau : bool
        True if plateau detected at the end of trajectory.
    plateau_start : Optional[int]
        Generation index where plateau started, or None if no plateau.
    
    Examples
    --------
    >>> trajectory = [0.1, 0.2, 0.3, 0.31, 0.31, 0.31, 0.31, 0.31]
    >>> is_plateau, start = detect_hv_plateau(trajectory, g_plateau=5, epsilon=0.01)
    >>> print(is_plateau, start)  # True, 3
    """
    if len(hv_trajectory) < g_plateau + 1:
        return False, None
    
    # Find consecutive generations with < epsilon relative improvement
    plateau_start_candidate = None
    consecutive_count = 0
    
    for i in range(1, len(hv_trajectory)):
        prev_hv = hv_trajectory[i - 1]
        curr_hv = hv_trajectory[i]
        
        # Compute relative improvement
        if prev_hv == 0:
            rel_improvement = float('inf') if curr_hv > 0 else 0
        else:
            rel_improvement = (curr_hv - prev_hv) / abs(prev_hv)
        
        if rel_improvement < epsilon:
            if plateau_start_candidate is None:
                plateau_start_candidate = i
            consecutive_count += 1
        else:
            # Reset
            plateau_start_candidate = None
            consecutive_count = 0
        
        # Check if we have enough consecutive plateau generations
        if consecutive_count >= g_plateau:
            return True, plateau_start_candidate
    
    return False, None


# =============================================================================
# Strategy Drift Calculation (v3.1 MUST: Section 4.2)
# =============================================================================

def compute_strategy_drift(
    current_weights: np.ndarray,
    previous_weights: np.ndarray,
) -> float:
    """
    Compute strategy drift as L1 norm of weight difference.
    
    v3.1 Requirement 4.2.3 MUST: Record strategy drift |w^{j} - w^{j-1}|_1.
    
    Parameters
    ----------
    current_weights : np.ndarray
        Current window weights, shape [K].
    previous_weights : np.ndarray
        Previous window weights, shape [K].
    
    Returns
    -------
    drift : float
        L1 norm of weight difference.
    
    Raises
    ------
    ValueError
        If weight shapes don't match.
    
    Examples
    --------
    >>> w_curr = np.array([0.3, 0.3, 0.4])
    >>> w_prev = np.array([0.2, 0.4, 0.4])
    >>> drift = compute_strategy_drift(w_curr, w_prev)
    >>> print(drift)  # 0.2
    """
    current_weights = np.asarray(current_weights).flatten()
    previous_weights = np.asarray(previous_weights).flatten()
    
    if current_weights.shape != previous_weights.shape:
        raise ValueError(
            f"Weight shape mismatch: current={current_weights.shape}, "
            f"previous={previous_weights.shape}"
        )
    
    drift = np.sum(np.abs(current_weights - previous_weights))
    return float(drift)


# =============================================================================
# SDF Consistency Filtering (v3.1 MUST: Section 4.3)
# =============================================================================

def filter_sdf_consistent(
    pareto_front: np.ndarray,
    gamma: float,
    pe_column_idx: int = 3,
) -> Tuple[np.ndarray, int]:
    """
    Filter Pareto solutions by SDF consistency threshold.
    
    v3.1 Requirement 4.3.1 MUST: Set consistency threshold γ for |g^(w)|.
    Solutions with PE (pricing error) ≤ γ are considered SDF-consistent.
    
    Parameters
    ----------
    pareto_front : np.ndarray
        Pareto front, shape [N, M]. Column pe_column_idx contains PE values.
    gamma : float
        SDF consistency threshold. Solutions with PE ≤ gamma are consistent.
    pe_column_idx : int
        Column index for pricing error (f4). Default 3 for 4-objective front.
    
    Returns
    -------
    consistent_mask : np.ndarray
        Boolean mask, shape [N]. True for consistent solutions.
    consistent_count : int
        Number of consistent solutions.
    
    Examples
    --------
    >>> front = np.array([[-1.5, 0.02, 0.1, 0.003], [-1.0, 0.03, 0.2, 0.008]])
    >>> mask, count = filter_sdf_consistent(front, gamma=0.005)
    >>> print(count)  # 1 (first solution has PE=0.003 < 0.005)
    """
    if pareto_front.size == 0:
        return np.array([], dtype=bool), 0
    
    pe_values = pareto_front[:, pe_column_idx]
    consistent_mask = pe_values <= gamma
    consistent_count = int(consistent_mask.sum())
    
    return consistent_mask, consistent_count


def compute_consistency_ratio(
    pareto_front: np.ndarray,
    gamma: float,
    pe_column_idx: int = 3,
) -> float:
    """
    Compute the ratio of SDF-consistent solutions in the Pareto front.
    
    Parameters
    ----------
    pareto_front : np.ndarray
        Pareto front, shape [N, M].
    gamma : float
        SDF consistency threshold.
    pe_column_idx : int
        Column index for pricing error.
    
    Returns
    -------
    ratio : float
        Fraction of solutions with PE ≤ gamma. Range [0, 1].
    """
    if pareto_front.size == 0:
        return 0.0
    
    n_solutions = pareto_front.shape[0]
    _, consistent_count = filter_sdf_consistent(pareto_front, gamma, pe_column_idx)
    
    return float(consistent_count / n_solutions)


# =============================================================================
# Full Evaluation Summary
# =============================================================================

def evaluate_pareto_front(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    current_weights: Optional[np.ndarray] = None,
    previous_weights: Optional[np.ndarray] = None,
    gamma: float = 0.01,
    pe_column_idx: int = 3,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation summary for a Pareto front.
    
    This function combines all EA v3.1 evaluation metrics:
    - Hypervolume
    - SDF consistency filtering
    - Strategy drift (if weights provided)
    - Objective statistics
    
    Parameters
    ----------
    pareto_front : np.ndarray
        Pareto front, shape [N, 4] for 4-objective optimization.
        Columns: [f1=-Sharpe, f2=MDD, f3=Turnover, f4=PE]
    reference_point : np.ndarray
        Reference point for HV computation, shape [4].
    current_weights : np.ndarray, optional
        Best weights from current window for drift calculation.
    previous_weights : np.ndarray, optional
        Best weights from previous window for drift calculation.
    gamma : float
        SDF consistency threshold.
    pe_column_idx : int
        Column index for pricing error.
    
    Returns
    -------
    summary : Dict[str, Any]
        Dictionary with evaluation metrics:
        - hypervolume: float
        - solution_count: int
        - consistent_count: int
        - consistency_ratio: float
        - strategy_drift: float (0.0 if weights not provided)
        - best_sharpe: float (from -f1)
        - mean_mdd: float
        - mean_turnover: float
        - mean_pe: float
    """
    n_solutions = pareto_front.shape[0] if pareto_front.size > 0 else 0
    
    # Compute hypervolume
    hv = compute_hypervolume(pareto_front, reference_point) if n_solutions > 0 else 0.0
    
    # Compute consistency filtering
    if n_solutions > 0:
        _, consistent_count = filter_sdf_consistent(pareto_front, gamma, pe_column_idx)
        consistency_ratio = compute_consistency_ratio(pareto_front, gamma, pe_column_idx)
    else:
        consistent_count = 0
        consistency_ratio = 0.0
    
    # Compute strategy drift
    if current_weights is not None and previous_weights is not None:
        strategy_drift = compute_strategy_drift(current_weights, previous_weights)
    else:
        strategy_drift = 0.0
    
    # Objective statistics
    if n_solutions > 0:
        # f1 = -Sharpe, so best Sharpe = -min(f1)
        best_sharpe = float(-pareto_front[:, 0].min())
        mean_mdd = float(pareto_front[:, 1].mean())
        mean_turnover = float(pareto_front[:, 2].mean())
        mean_pe = float(pareto_front[:, pe_column_idx].mean())
    else:
        best_sharpe = 0.0
        mean_mdd = 0.0
        mean_turnover = 0.0
        mean_pe = 0.0
    
    return {
        "hypervolume": hv,
        "solution_count": n_solutions,
        "consistent_count": consistent_count,
        "consistency_ratio": consistency_ratio,
        "strategy_drift": strategy_drift,
        "best_sharpe": best_sharpe,
        "mean_mdd": mean_mdd,
        "mean_turnover": mean_turnover,
        "mean_pe": mean_pe,
    }


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EA Evaluation Metrics Module - Self Test")
    print("=" * 60)
    
    # Test 1: 2D Hypervolume
    print("\n[Test 1] 2D Hypervolume...")
    front_2d = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    ref_2d = np.array([5.0, 5.0])
    hv_2d = compute_hypervolume(front_2d, ref_2d)
    print(f"  Front: {front_2d.tolist()}")
    print(f"  Reference: {ref_2d.tolist()}")
    print(f"  Hypervolume: {hv_2d:.4f}")
    assert hv_2d > 0, "HV should be positive"
    print("  ✅ PASS")
    
    # Test 2: 4D Hypervolume
    print("\n[Test 2] 4D Hypervolume (EA v3.1 scenario)...")
    np.random.seed(42)
    front_4d = np.zeros((5, 4))
    for i in range(5):
        front_4d[i] = [-2.0 + i*0.4, 0.01 + i*0.02, 0.1 + i*0.1, 0.001 + i*0.002]
    ref_4d = np.array([1.0, 0.5, 1.0, 0.05])
    hv_4d = compute_hypervolume(front_4d, ref_4d)
    print(f"  Front shape: {front_4d.shape}")
    print(f"  Hypervolume: {hv_4d:.4f}")
    assert hv_4d > 0, "HV should be positive"
    print("  ✅ PASS")
    
    # Test 3: HV Plateau Detection
    print("\n[Test 3] HV Plateau Detection...")
    trajectory = [0.1, 0.2, 0.3, 0.4, 0.5, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51]
    is_plateau, start = detect_hv_plateau(trajectory, g_plateau=5, epsilon=0.01)
    print(f"  Trajectory: {trajectory}")
    print(f"  Is Plateau: {is_plateau}, Start: {start}")
    assert is_plateau is True, "Should detect plateau"
    print("  ✅ PASS")
    
    # Test 4: Strategy Drift
    print("\n[Test 4] Strategy Drift...")
    w1 = np.array([0.3, 0.3, 0.4])
    w2 = np.array([0.2, 0.4, 0.4])
    drift = compute_strategy_drift(w1, w2)
    print(f"  w_curr: {w1}")
    print(f"  w_prev: {w2}")
    print(f"  Drift (L1): {drift:.4f}")
    assert abs(drift - 0.2) < 1e-9, f"Expected 0.2, got {drift}"
    print("  ✅ PASS")
    
    # Test 5: SDF Consistency Filtering
    print("\n[Test 5] SDF Consistency Filtering...")
    front_with_pe = np.array([
        [-2.0, 0.02, 0.1, 0.001],
        [-1.5, 0.03, 0.2, 0.003],
        [-1.0, 0.04, 0.3, 0.010],  # PE > gamma
    ])
    gamma = 0.005
    mask, count = filter_sdf_consistent(front_with_pe, gamma, pe_column_idx=3)
    print(f"  Gamma: {gamma}")
    print(f"  PE values: {front_with_pe[:, 3]}")
    print(f"  Consistent count: {count}")
    assert count == 2, f"Expected 2 consistent, got {count}"
    print("  ✅ PASS")
    
    # Test 6: Full Evaluation Summary
    print("\n[Test 6] Full Evaluation Summary...")
    summary = evaluate_pareto_front(
        pareto_front=front_4d,
        reference_point=ref_4d,
        current_weights=np.array([0.2] * 5),
        previous_weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        gamma=0.005,
        pe_column_idx=3,
    )
    print(f"  Summary: {summary}")
    assert "hypervolume" in summary
    assert "consistency_ratio" in summary
    assert "best_sharpe" in summary
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("✅ ALL SELF-TESTS PASSED")
    print("=" * 60)
