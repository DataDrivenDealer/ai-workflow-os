"""
SDF-Consistency Selection for DGSF EA Layer v3.1 (P0-22).

This module implements SDF-consistency selection rules per
EA Layer v3.1 Section 4.3 MUST.

Key Features:
1. Set consistency threshold γ for |g^(w)| (4.3.1 MUST)
2. Consistency-first selection rules (4.3.2 MUST):
   - Lexicographic: tier by consistency first
   - Threshold: eliminate if |g^(w)| > γ
   - Weighted: penalize inconsistent strategies
3. Baseline EA uses f⁴ as normal objective only (4.3.3)

Spec Reference: DGSF EA Layer Specification v3.1 - Section 4.3
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_gamma(gamma: float) -> None:
    """
    Validate consistency threshold γ.
    
    Args:
        gamma: Consistency threshold
        
    Raises:
        ValueError: If gamma is invalid
    """
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")


def validate_mode(mode: str) -> None:
    """
    Validate selection mode.
    
    Args:
        mode: Selection mode
        
    Raises:
        ValueError: If mode is invalid
    """
    valid_modes = {'lexicographic', 'threshold', 'weighted'}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode}")


# =============================================================================
# Consistency Checking (v3.1 4.3.1 MUST)
# =============================================================================

def check_sdf_consistency(sdf_penalty: float, gamma: float) -> bool:
    """
    Check if an individual is SDF-consistent.
    
    v3.1 Requirement 4.3.1 MUST: Set consistency threshold γ.
    
    Args:
        sdf_penalty: The |g^(w)| value (SDF pricing error)
        gamma: Consistency threshold
        
    Returns:
        True if consistent (|g^(w)| <= γ), False otherwise
        
    Example:
        >>> check_sdf_consistency(0.005, gamma=0.01)
        True
        >>> check_sdf_consistency(0.015, gamma=0.01)
        False
    """
    return sdf_penalty <= gamma


def batch_check_consistency(
    population: List[Dict[str, Any]],
    gamma: float,
    sdf_key: str = 'sdf_penalty'
) -> List[bool]:
    """
    Check consistency for entire population.
    
    Args:
        population: List of individuals with objectives
        gamma: Consistency threshold
        sdf_key: Key for SDF penalty in objectives dict
        
    Returns:
        List of boolean consistency flags
    """
    results = []
    for ind in population:
        sdf_penalty = ind.get('objectives', {}).get(sdf_key, 0.0)
        results.append(check_sdf_consistency(sdf_penalty, gamma))
    return results


# =============================================================================
# Lexicographic Selection (v3.1 4.3.2 MUST - Option 1)
# =============================================================================

def assign_consistency_tiers(
    population: List[Dict[str, Any]],
    gamma: float,
    sdf_key: str = 'sdf_penalty'
) -> List[Dict[str, Any]]:
    """
    Assign tier based on consistency.
    
    v3.1 Requirement 4.3.2 MUST: Tier by consistency first.
    - Tier 0: Consistent (|g^(w)| <= γ)
    - Tier 1: Inconsistent (|g^(w)| > γ)
    
    Args:
        population: List of individuals
        gamma: Consistency threshold
        sdf_key: Key for SDF penalty
        
    Returns:
        Population with 'tier' field added
    """
    tiered = []
    for ind in population:
        ind_copy = ind.copy()
        sdf_penalty = ind.get('objectives', {}).get(sdf_key, 0.0)
        ind_copy['tier'] = 0 if check_sdf_consistency(sdf_penalty, gamma) else 1
        tiered.append(ind_copy)
    return tiered


def lexicographic_sort(
    population: List[Dict[str, Any]],
    gamma: float,
    primary_obj: str = 'sharpe',
    maximize: bool = True,
    sdf_key: str = 'sdf_penalty'
) -> List[Dict[str, Any]]:
    """
    Sort population lexicographically: tier first, then by objective.
    
    Args:
        population: List of individuals
        gamma: Consistency threshold
        primary_obj: Primary objective to sort by within tier
        maximize: Whether to maximize the objective
        sdf_key: Key for SDF penalty
        
    Returns:
        Sorted population (tier 0 first, then tier 1)
    """
    tiered = assign_consistency_tiers(population, gamma, sdf_key)
    
    # Sort by (tier, -objective) for maximization, (tier, objective) for minimization
    def sort_key(ind):
        obj_value = ind.get('objectives', {}).get(primary_obj, 0.0)
        return (ind['tier'], -obj_value if maximize else obj_value)
    
    return sorted(tiered, key=sort_key)


def select_elite_lexicographic(
    population: List[Dict[str, Any]],
    gamma: float,
    elite_count: int,
    primary_obj: str = 'sharpe',
    maximize: bool = True,
    sdf_key: str = 'sdf_penalty'
) -> List[Dict[str, Any]]:
    """
    Select elite using lexicographic rule.
    
    Args:
        population: List of individuals
        gamma: Consistency threshold
        elite_count: Number of elite to select
        primary_obj: Primary objective for sorting
        maximize: Whether to maximize objective
        sdf_key: Key for SDF penalty
        
    Returns:
        List of elite individuals
    """
    sorted_pop = lexicographic_sort(population, gamma, primary_obj, maximize, sdf_key)
    return sorted_pop[:elite_count]


# =============================================================================
# Threshold Selection (v3.1 4.3.2 MUST - Option 2)
# =============================================================================

def eliminate_inconsistent(
    population: List[Dict[str, Any]],
    gamma: float,
    sdf_key: str = 'sdf_penalty'
) -> List[Dict[str, Any]]:
    """
    Eliminate individuals with |g^(w)| > γ.
    
    v3.1 Requirement 4.3.2 MUST: Threshold elimination.
    
    Args:
        population: List of individuals
        gamma: Consistency threshold
        sdf_key: Key for SDF penalty
        
    Returns:
        Filtered population (only consistent individuals)
    """
    filtered = []
    for ind in population:
        sdf_penalty = ind.get('objectives', {}).get(sdf_key, 0.0)
        if check_sdf_consistency(sdf_penalty, gamma):
            filtered.append(ind)
    return filtered


def select_with_threshold(
    population: List[Dict[str, Any]],
    gamma: float,
    select_count: int,
    objective: str = 'sharpe',
    maximize: bool = True,
    sdf_key: str = 'sdf_penalty'
) -> List[Dict[str, Any]]:
    """
    Select best individuals after threshold filtering.
    
    Args:
        population: List of individuals
        gamma: Consistency threshold
        select_count: Number to select
        objective: Objective to sort by
        maximize: Whether to maximize objective
        sdf_key: Key for SDF penalty
        
    Returns:
        Selected individuals
    """
    # First filter out inconsistent
    filtered = eliminate_inconsistent(population, gamma, sdf_key)
    
    # Sort by objective
    def sort_key(ind):
        obj_value = ind.get('objectives', {}).get(objective, 0.0)
        return -obj_value if maximize else obj_value
    
    sorted_filtered = sorted(filtered, key=sort_key)
    
    # Return top N
    return sorted_filtered[:select_count]


# =============================================================================
# Weighted Penalty Selection (v3.1 4.3.2 MUST - Option 3)
# =============================================================================

def compute_penalized_fitness(
    population: List[Dict[str, Any]],
    gamma: float,
    penalty_weight: float,
    objective: str = 'sharpe',
    sdf_key: str = 'sdf_penalty'
) -> List[Dict[str, Any]]:
    """
    Compute penalized fitness for each individual.
    
    v3.1 Requirement 4.3.2 MUST: Penalize inconsistent strategies.
    
    Formula:
        penalized_fitness = objective - weight * max(0, |g^(w)| - γ)
    
    Args:
        population: List of individuals
        gamma: Consistency threshold
        penalty_weight: Weight for penalty term
        objective: Objective to penalize
        sdf_key: Key for SDF penalty
        
    Returns:
        Population with 'penalized_fitness' field added
    """
    penalized = []
    for ind in population:
        ind_copy = ind.copy()
        obj_value = ind.get('objectives', {}).get(objective, 0.0)
        sdf_penalty = ind.get('objectives', {}).get(sdf_key, 0.0)
        
        # Penalty only applies when above threshold
        excess = max(0.0, sdf_penalty - gamma)
        penalty = penalty_weight * excess
        
        ind_copy['penalized_fitness'] = obj_value - penalty
        penalized.append(ind_copy)
    
    return penalized


def select_weighted(
    population: List[Dict[str, Any]],
    gamma: float,
    penalty_weight: float,
    select_count: int,
    objective: str = 'sharpe',
    sdf_key: str = 'sdf_penalty'
) -> List[Dict[str, Any]]:
    """
    Select based on penalized fitness.
    
    Args:
        population: List of individuals
        gamma: Consistency threshold
        penalty_weight: Weight for penalty
        select_count: Number to select
        objective: Objective for fitness
        sdf_key: Key for SDF penalty
        
    Returns:
        Selected individuals
    """
    penalized = compute_penalized_fitness(
        population, gamma, penalty_weight, objective, sdf_key
    )
    
    # Sort by penalized fitness descending
    sorted_pop = sorted(penalized, key=lambda x: -x.get('penalized_fitness', 0.0))
    
    return sorted_pop[:select_count]


# =============================================================================
# Baseline EA (v3.1 4.3.3) - No Consistency Filtering
# =============================================================================

def baseline_select(
    population: List[Dict[str, Any]],
    select_count: int,
    objective: str = 'sharpe',
    maximize: bool = True
) -> List[Dict[str, Any]]:
    """
    Baseline EA selection: purely objective-based, no consistency filtering.
    
    v3.1 Requirement 4.3.3: Baseline EA uses f⁴ as normal objective only.
    
    Args:
        population: List of individuals
        select_count: Number to select
        objective: Objective to sort by
        maximize: Whether to maximize
        
    Returns:
        Selected individuals (best by objective, ignoring consistency)
    """
    def sort_key(ind):
        obj_value = ind.get('objectives', {}).get(objective, 0.0)
        return -obj_value if maximize else obj_value
    
    sorted_pop = sorted(population, key=sort_key)
    return sorted_pop[:select_count]


def baseline_multiobjective_rank(
    population: List[Dict[str, Any]],
    objectives: Optional[List[str]] = None,
    maximize: Optional[List[bool]] = None
) -> List[Dict[str, Any]]:
    """
    Compute NSGA-II style ranking for baseline EA.
    
    Baseline treats sdf_penalty (f⁴) as a normal objective to minimize.
    
    Args:
        population: List of individuals
        objectives: List of objective keys (default: sharpe, mdd, turnover, sdf_penalty)
        maximize: Whether to maximize each objective
        
    Returns:
        Population with 'rank' and 'crowding' fields
    """
    if objectives is None:
        objectives = ['sharpe', 'mdd', 'turnover', 'sdf_penalty']
    if maximize is None:
        maximize = [True, False, False, False]  # Maximize Sharpe, minimize others
    
    n = len(population)
    if n == 0:
        return []
    
    # Extract objective values
    obj_values = np.zeros((n, len(objectives)))
    for i, ind in enumerate(population):
        for j, obj_key in enumerate(objectives):
            val = ind.get('objectives', {}).get(obj_key, 0.0)
            # Negate if maximizing (for dominance comparison)
            obj_values[i, j] = -val if maximize[j] else val
    
    # Compute domination (simplified non-dominated sorting)
    ranks = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]
    dominates_count = np.zeros(n, dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i dominates j or vice versa
            i_dom_j = np.all(obj_values[i] <= obj_values[j]) and np.any(obj_values[i] < obj_values[j])
            j_dom_i = np.all(obj_values[j] <= obj_values[i]) and np.any(obj_values[j] < obj_values[i])
            
            if i_dom_j:
                dominated_by[j].append(i)
                dominates_count[i] += 1
            elif j_dom_i:
                dominated_by[i].append(j)
                dominates_count[j] += 1
    
    # Assign ranks based on domination count (simplified)
    for i in range(n):
        ranks[i] = len(dominated_by[i])
    
    # Compute crowding distance (simplified: use objective variance)
    crowding = np.zeros(n)
    for j in range(len(objectives)):
        sorted_idx = np.argsort(obj_values[:, j])
        crowding[sorted_idx[0]] = np.inf
        crowding[sorted_idx[-1]] = np.inf
        obj_range = obj_values[sorted_idx[-1], j] - obj_values[sorted_idx[0], j]
        if obj_range > 0:
            for k in range(1, n - 1):
                crowding[sorted_idx[k]] += (
                    obj_values[sorted_idx[k + 1], j] - obj_values[sorted_idx[k - 1], j]
                ) / obj_range
    
    # Add rank and crowding to population
    ranked_pop = []
    for i, ind in enumerate(population):
        ind_copy = ind.copy()
        ind_copy['rank'] = int(ranks[i])
        ind_copy['crowding'] = float(crowding[i])
        ranked_pop.append(ind_copy)
    
    return ranked_pop


# =============================================================================
# SDFConsistencySelector Class (Integration)
# =============================================================================

class SDFConsistencySelector:
    """
    SDF-Consistency Selector for EA Layer v3.1.
    
    Implements consistency-first selection rules per Section 4.3.
    
    Attributes:
        gamma: Consistency threshold
        mode: Selection mode ('lexicographic', 'threshold', 'weighted')
        penalty_weight: Weight for penalty in weighted mode
        telemetry: Recorded statistics
        
    Example:
        >>> selector = SDFConsistencySelector(gamma=0.01, mode='lexicographic')
        >>> elite = selector.select_elite(population, elite_count=10)
        >>> print(selector.get_telemetry())
    """
    
    def __init__(
        self,
        gamma: float,
        mode: str = 'lexicographic',
        penalty_weight: float = 10.0
    ):
        """
        Initialize selector.
        
        Args:
            gamma: Consistency threshold γ
            mode: Selection mode
            penalty_weight: Weight for weighted mode
            
        Raises:
            ValueError: If parameters are invalid
        """
        validate_gamma(gamma)
        validate_mode(mode)
        
        self.gamma = gamma
        self.mode = mode
        self.penalty_weight = penalty_weight
        self.telemetry: Dict[str, Any] = {}
    
    def select_elite(
        self,
        population: List[Dict[str, Any]],
        elite_count: int,
        primary_obj: str = 'sharpe',
        maximize: bool = True,
        sdf_key: str = 'sdf_penalty'
    ) -> List[Dict[str, Any]]:
        """
        Select elite individuals using configured selection rule.
        
        Args:
            population: List of individuals
            elite_count: Number to select
            primary_obj: Primary objective
            maximize: Whether to maximize
            sdf_key: Key for SDF penalty
            
        Returns:
            List of elite individuals
        """
        # Record consistency stats
        consistency_flags = batch_check_consistency(population, self.gamma, sdf_key)
        consistent_count = sum(consistency_flags)
        inconsistent_count = len(population) - consistent_count
        
        # Select based on mode
        if self.mode == 'lexicographic':
            elite = select_elite_lexicographic(
                population, self.gamma, elite_count, primary_obj, maximize, sdf_key
            )
        elif self.mode == 'threshold':
            elite = select_with_threshold(
                population, self.gamma, elite_count, primary_obj, maximize, sdf_key
            )
        elif self.mode == 'weighted':
            elite = select_weighted(
                population, self.gamma, self.penalty_weight, elite_count, primary_obj, sdf_key
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Update telemetry
        self.telemetry = {
            'gamma': self.gamma,
            'mode': self.mode,
            'consistent_count': consistent_count,
            'inconsistent_count': inconsistent_count,
            'consistency_ratio': consistent_count / len(population) if population else 0,
            'elite_count': len(elite),
            'population_size': len(population),
        }
        
        return elite
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get telemetry from last selection.
        
        Returns:
            Dictionary with selection statistics
        """
        return self.telemetry.copy()


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("P0-22 SDF-Consistency Selection - Self Test")
    print("=" * 60)
    
    # Test data
    population = [
        {'id': 0, 'objectives': {'sharpe': 1.5, 'mdd': 0.10, 'turnover': 0.08, 'sdf_penalty': 0.002}},
        {'id': 1, 'objectives': {'sharpe': 1.8, 'mdd': 0.12, 'turnover': 0.10, 'sdf_penalty': 0.008}},
        {'id': 2, 'objectives': {'sharpe': 2.0, 'mdd': 0.15, 'turnover': 0.12, 'sdf_penalty': 0.015}},
        {'id': 3, 'objectives': {'sharpe': 1.2, 'mdd': 0.08, 'turnover': 0.05, 'sdf_penalty': 0.001}},
        {'id': 4, 'objectives': {'sharpe': 2.5, 'mdd': 0.20, 'turnover': 0.15, 'sdf_penalty': 0.025}},
        {'id': 5, 'objectives': {'sharpe': 1.6, 'mdd': 0.11, 'turnover': 0.09, 'sdf_penalty': 0.004}},
    ]
    gamma = 0.01
    
    # Test 1: Consistency check
    print("\n[Test 1] Consistency Check...")
    flags = batch_check_consistency(population, gamma)
    consistent_ids = [pop['id'] for pop, flag in zip(population, flags) if flag]
    print(f"  Gamma: {gamma}")
    print(f"  Consistent IDs: {consistent_ids}")
    assert consistent_ids == [0, 1, 3, 5], "Wrong consistent IDs"
    print("  ✅ Consistency check correct")
    
    # Test 2: Lexicographic selection
    print("\n[Test 2] Lexicographic Selection...")
    elite_lex = select_elite_lexicographic(population, gamma, elite_count=2)
    elite_ids = [ind['id'] for ind in elite_lex]
    print(f"  Elite IDs: {elite_ids}")
    assert all(ind['objectives']['sdf_penalty'] <= gamma for ind in elite_lex)
    print("  ✅ Lexicographic selection correct")
    
    # Test 3: Threshold selection
    print("\n[Test 3] Threshold Selection...")
    filtered = eliminate_inconsistent(population, gamma)
    print(f"  Filtered count: {len(filtered)}")
    assert len(filtered) == 4
    print("  ✅ Threshold elimination correct")
    
    # Test 4: Weighted selection
    print("\n[Test 4] Weighted Selection...")
    penalized = compute_penalized_fitness(population, gamma, penalty_weight=10.0)
    ind2 = next(ind for ind in penalized if ind['id'] == 2)
    print(f"  ID 2 penalized fitness: {ind2['penalized_fitness']:.3f}")
    assert np.isclose(ind2['penalized_fitness'], 1.95, atol=0.01)
    print("  ✅ Weighted penalty correct")
    
    # Test 5: Baseline selection
    print("\n[Test 5] Baseline Selection (no filter)...")
    baseline = baseline_select(population, select_count=3)
    baseline_ids = [ind['id'] for ind in baseline]
    print(f"  Baseline top 3 IDs: {baseline_ids}")
    assert 4 in baseline_ids, "Baseline should include high-Sharpe inconsistent"
    print("  ✅ Baseline selection correct")
    
    # Test 6: Selector class integration
    print("\n[Test 6] SDFConsistencySelector Integration...")
    selector = SDFConsistencySelector(gamma=gamma, mode='lexicographic')
    elite = selector.select_elite(population, elite_count=2)
    telemetry = selector.get_telemetry()
    print(f"  Mode: {telemetry['mode']}")
    print(f"  Consistent: {telemetry['consistent_count']}, Inconsistent: {telemetry['inconsistent_count']}")
    assert telemetry['consistent_count'] == 4
    assert telemetry['inconsistent_count'] == 2
    print("  ✅ Selector integration correct")
    
    # Test 7: Invalid config
    print("\n[Test 7] Config Validation...")
    try:
        SDFConsistencySelector(gamma=-0.01, mode='lexicographic')
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError: {e}")
    
    # Test 8: Multiobjective ranking
    print("\n[Test 8] Baseline Multiobjective Rank...")
    ranked = baseline_multiobjective_rank(population)
    print(f"  Ranked individuals: {len(ranked)}")
    assert all('rank' in ind for ind in ranked)
    assert all('crowding' in ind for ind in ranked)
    print("  ✅ Multiobjective ranking correct")
    
    print("\n" + "=" * 60)
    print("✅ ALL SELF-TESTS PASSED")
    print("=" * 60)
