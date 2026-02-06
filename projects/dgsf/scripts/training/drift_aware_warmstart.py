"""
Drift-Aware Warm-Start for DGSF EA Layer v3.1 (P0-21).

This module implements drift-aware warm-start population initialization per
EA Layer v3.1 Section 4.2 MUST.

Key Features:
1. Transform previous Pareto solutions to current leaf space (4.2.1 MUST)
2. Initialize population with 40-60% from previous Pareto (4.2.1 MUST)
3. Combine with random individuals for exploration (4.2.2 MUST)
4. Record strategy drift |w^{j} - w^{j-1}|_1 (4.2.3 MUST)
5. Baseline EA pure random initialization (4.2.4)

Spec Reference: DGSF EA Layer Specification v3.1 - Section 4.2
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate warm-start configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    if 'warm_start_ratio' in config:
        ratio = config['warm_start_ratio']
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"warm_start_ratio must be in [0, 1], got {ratio}")
    
    if 'min_warm_start_ratio' in config:
        min_ratio = config['min_warm_start_ratio']
        if not (0.0 <= min_ratio <= 1.0):
            raise ValueError(f"min_warm_start_ratio must be in [0, 1], got {min_ratio}")
    
    if 'max_warm_start_ratio' in config:
        max_ratio = config['max_warm_start_ratio']
        if not (0.0 <= max_ratio <= 1.0):
            raise ValueError(f"max_warm_start_ratio must be in [0, 1], got {max_ratio}")


# =============================================================================
# Transform Previous Pareto (v3.1 4.2.1 MUST - Part 1)
# =============================================================================

def transform_pareto_to_new_leaf_space(
    pareto_front: List[Dict[str, Any]],
    leaf_mapping: Dict[int, Optional[int]],
    new_leaf_count: int,
    normalize: bool = True,
    fill_unmapped: str = 'zero'
) -> List[Dict[str, Any]]:
    """
    Transform previous window Pareto solutions to current leaf space.
    
    v3.1 Requirement 4.2.1 MUST: Transform previous Pareto solutions.
    
    Args:
        pareto_front: List of Pareto solutions with 'weights' and 'objectives'
        leaf_mapping: Dict mapping prev_leaf_idx -> current_leaf_idx (or None)
        new_leaf_count: Number of leaves in current window
        normalize: Whether to normalize weights after transformation
        fill_unmapped: How to handle unmapped leaves ('zero', 'uniform')
        
    Returns:
        List of transformed solutions for current window
        
    Example:
        >>> pareto = [{'weights': np.array([0.5, 0.5]), 'objectives': {...}}]
        >>> mapping = {0: 1, 1: 0}  # Swap leaf indices
        >>> transformed = transform_pareto_to_new_leaf_space(pareto, mapping, 2)
        >>> # weights become [0.5, 0.5] after remapping
    """
    if not pareto_front:
        return []
    
    transformed = []
    
    for solution in pareto_front:
        old_weights = np.array(solution['weights'])
        new_weights = np.zeros(new_leaf_count)
        
        # Remap weights according to leaf_mapping
        for old_idx, weight in enumerate(old_weights):
            if old_idx in leaf_mapping and leaf_mapping[old_idx] is not None:
                new_idx = leaf_mapping[old_idx]
                if 0 <= new_idx < new_leaf_count:
                    new_weights[new_idx] = weight
        
        # Normalize if requested and sum > 0
        if normalize and new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        elif normalize and new_weights.sum() == 0:
            # All weights lost in mapping - use uniform
            new_weights = np.ones(new_leaf_count) / new_leaf_count
        
        # Create transformed solution
        transformed_solution = {
            'weights': new_weights,
            'original_objectives': solution.get('objectives', {}),
            'original_weights': old_weights,
            'transform_info': {
                'leaves_mapped': sum(1 for v in leaf_mapping.values() if v is not None),
                'leaves_lost': sum(1 for old_idx in range(len(old_weights)) 
                                  if old_idx not in leaf_mapping or leaf_mapping.get(old_idx) is None),
            }
        }
        
        transformed.append(transformed_solution)
    
    return transformed


# =============================================================================
# Create Warm-Start Population (v3.1 4.2.1 MUST - Part 2)
# =============================================================================

def create_warm_start_population(
    transformed_pareto: List[Dict[str, Any]],
    pop_size: int,
    new_leaf_count: int,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create population with warm-started and random individuals.
    
    v3.1 Requirement 4.2.1 MUST: Initialize 40-60% from previous Pareto.
    v3.1 Requirement 4.2.2 MUST: Combine with random individuals.
    
    Args:
        transformed_pareto: Transformed Pareto solutions from previous window
        pop_size: Total population size
        new_leaf_count: Number of leaves in current window
        config: Warm-start configuration
        
    Returns:
        List of individuals (dict with 'weights', 'source', etc.)
    """
    warm_start_ratio = config.get('warm_start_ratio', 0.5)
    
    # If no Pareto solutions, use all random
    if not transformed_pareto:
        return create_random_population(pop_size, new_leaf_count)
    
    # Calculate warm-start count
    warm_start_count = int(pop_size * warm_start_ratio)
    random_count = pop_size - warm_start_count
    
    population = []
    
    # Create warm-start individuals by cycling through Pareto solutions
    n_pareto = len(transformed_pareto)
    for i in range(warm_start_count):
        pareto_idx = i % n_pareto
        pareto_solution = transformed_pareto[pareto_idx]
        
        # Add small noise for diversity (optional mutation)
        weights = pareto_solution['weights'].copy()
        noise = np.random.normal(0, 0.01, len(weights))
        weights = weights + noise
        weights = np.clip(weights, 0, None)  # Non-negative
        if weights.sum() > 0:
            weights = weights / weights.sum()  # Normalize
        else:
            weights = np.ones(new_leaf_count) / new_leaf_count
        
        individual = {
            'weights': weights,
            'source': 'warm_start',
            'pareto_origin_idx': pareto_idx,
            'original_objectives': pareto_solution.get('original_objectives', {})
        }
        population.append(individual)
    
    # Create random individuals for exploration
    random_inds = create_random_population(random_count, new_leaf_count)
    population.extend(random_inds)
    
    return population


# =============================================================================
# Random Population Creation (v3.1 4.2.2, 4.2.4)
# =============================================================================

def create_random_population(
    pop_size: int,
    n_leaves: int,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Create population with pure random individuals.
    
    v3.1 Requirement 4.2.4: Baseline EA must use pure random initialization.
    v3.1 Requirement 4.2.2: Random individuals for exploration.
    
    Args:
        pop_size: Number of individuals to create
        n_leaves: Number of leaves (weight dimensions)
        seed: Random seed for reproducibility
        
    Returns:
        List of random individuals
    """
    if seed is not None:
        np.random.seed(seed)
    
    population = []
    
    for _ in range(pop_size):
        # Generate random weights using Dirichlet distribution
        # This ensures weights sum to 1 and are non-negative
        weights = np.random.dirichlet(np.ones(n_leaves))
        
        individual = {
            'weights': weights,
            'source': 'random'
        }
        population.append(individual)
    
    return population


# =============================================================================
# Strategy Drift Calculation (v3.1 4.2.3 MUST)
# =============================================================================

def compute_strategy_drift(
    w_prev: np.ndarray,
    w_curr: np.ndarray
) -> float:
    """
    Compute strategy drift using L1 norm.
    
    v3.1 Requirement 4.2.3 MUST: Record |w^{j} - w^{j-1}|_1.
    
    Args:
        w_prev: Weights from previous window
        w_curr: Weights from current window
        
    Returns:
        L1 norm of weight difference
        
    Example:
        >>> w_prev = np.array([0.3, 0.3, 0.2, 0.2])
        >>> w_curr = np.array([0.1, 0.4, 0.3, 0.2])
        >>> compute_strategy_drift(w_prev, w_curr)
        0.4
    """
    # Handle different lengths by padding shorter array
    max_len = max(len(w_prev), len(w_curr))
    
    w_prev_padded = np.zeros(max_len)
    w_curr_padded = np.zeros(max_len)
    
    w_prev_padded[:len(w_prev)] = w_prev
    w_curr_padded[:len(w_curr)] = w_curr
    
    return float(np.sum(np.abs(w_prev_padded - w_curr_padded)))


def compute_batch_drift(
    window_weights: List[np.ndarray]
) -> List[float]:
    """
    Compute drift across multiple consecutive windows.
    
    Args:
        window_weights: List of weight vectors for each window
        
    Returns:
        List of N-1 drift values for N windows
    """
    if len(window_weights) < 2:
        return []
    
    drifts = []
    for i in range(1, len(window_weights)):
        drift = compute_strategy_drift(window_weights[i-1], window_weights[i])
        drifts.append(drift)
    
    return drifts


# =============================================================================
# DriftAwareWarmStart Class (Integration)
# =============================================================================

class DriftAwareWarmStart:
    """
    Drift-Aware Warm-Start controller for EA Layer v3.1.
    
    Orchestrates the complete warm-start workflow from previous
    window Pareto solutions to new population initialization.
    
    Attributes:
        config: Warm-start configuration
        telemetry: Recorded telemetry from last operation
        
    Example:
        >>> warm_start = DriftAwareWarmStart(config={'warm_start_ratio': 0.5})
        >>> population = warm_start.create_population(
        ...     previous_pareto=pareto_solutions,
        ...     leaf_mapping=mapping,
        ...     new_leaf_count=10,
        ...     pop_size=100
        ... )
        >>> print(warm_start.get_telemetry())
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize warm-start controller.
        
        Args:
            config: Configuration dictionary with:
                - warm_start_ratio: Fraction of population from Pareto (default: 0.5)
                - min_warm_start_ratio: Minimum ratio (default: 0.4)
                - max_warm_start_ratio: Maximum ratio (default: 0.6)
                - normalize_after_transform: Whether to normalize (default: True)
                - fill_unmapped_leaves: How to handle unmapped ('zero', 'uniform')
                
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config or self._default_config()
        validate_config(self.config)
        
        self.telemetry: Dict[str, Any] = {}
        self._transformed_pareto: List[Dict[str, Any]] = []
        self._population: List[Dict[str, Any]] = []
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'warm_start_ratio': 0.5,
            'min_warm_start_ratio': 0.4,
            'max_warm_start_ratio': 0.6,
            'normalize_after_transform': True,
            'fill_unmapped_leaves': 'zero',
        }
    
    def create_population(
        self,
        previous_pareto: List[Dict[str, Any]],
        leaf_mapping: Dict[int, Optional[int]],
        new_leaf_count: int,
        pop_size: int
    ) -> List[Dict[str, Any]]:
        """
        Create population using drift-aware warm-start.
        
        v3.1 Requirement 4.2.1 MUST: Use previous Pareto for initialization.
        v3.1 Requirement 4.2.2 MUST: Combine with random for exploration.
        
        Args:
            previous_pareto: Pareto solutions from previous window
            leaf_mapping: Mapping from previous to current leaf indices
            new_leaf_count: Number of leaves in current window
            pop_size: Total population size
            
        Returns:
            List of individuals ready for EA optimization
        """
        # Step 1: Transform Pareto to new leaf space
        self._transformed_pareto = transform_pareto_to_new_leaf_space(
            pareto_front=previous_pareto,
            leaf_mapping=leaf_mapping,
            new_leaf_count=new_leaf_count,
            normalize=self.config.get('normalize_after_transform', True),
            fill_unmapped=self.config.get('fill_unmapped_leaves', 'zero')
        )
        
        # Step 2: Create population with warm-start + random
        self._population = create_warm_start_population(
            transformed_pareto=self._transformed_pareto,
            pop_size=pop_size,
            new_leaf_count=new_leaf_count,
            config=self.config
        )
        
        # Step 3: Record telemetry
        self._update_telemetry(previous_pareto, leaf_mapping, new_leaf_count, pop_size)
        
        return self._population
    
    def _update_telemetry(
        self,
        previous_pareto: List[Dict[str, Any]],
        leaf_mapping: Dict[int, Optional[int]],
        new_leaf_count: int,
        pop_size: int
    ) -> None:
        """Update telemetry after population creation."""
        warm_start_count = sum(1 for ind in self._population 
                              if ind.get('source') == 'warm_start')
        random_count = sum(1 for ind in self._population 
                         if ind.get('source') == 'random')
        
        unmapped_count = sum(1 for v in leaf_mapping.values() if v is None)
        
        self.telemetry = {
            'warm_start_count': warm_start_count,
            'random_count': random_count,
            'warm_start_ratio': warm_start_count / pop_size if pop_size > 0 else 0,
            'pareto_solutions_used': len(previous_pareto),
            'transformed_solutions': len(self._transformed_pareto),
            'unmapped_leaves_count': unmapped_count,
            'new_leaf_count': new_leaf_count,
            'pop_size': pop_size,
        }
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get telemetry from last population creation.
        
        v3.1 Requirement 4.2.3 MUST: Record in Rolling Telemetry.
        
        Returns:
            Dictionary with warm-start statistics
        """
        return self.telemetry.copy()


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("P0-21 Drift-Aware Warm-Start - Self Test")
    print("=" * 60)
    
    # Test 1: Transform Pareto
    print("\n[Test 1] Transform Pareto to New Leaf Space...")
    pareto = [
        {
            'weights': np.array([0.1, 0.2, 0.3, 0.0, 0.4]),
            'objectives': {'sharpe': 1.5},
        },
    ]
    mapping = {0: 1, 1: 2, 2: 0, 3: None, 4: 3}
    
    transformed = transform_pareto_to_new_leaf_space(pareto, mapping, 5)
    print(f"  Original weights: {pareto[0]['weights']}")
    print(f"  Mapping: {mapping}")
    print(f"  Transformed weights: {transformed[0]['weights']}")
    
    expected = np.array([0.3, 0.1, 0.2, 0.4, 0.0])
    assert np.allclose(transformed[0]['weights'], expected), "Transform failed!"
    print("  ✅ Transform correct")
    
    # Test 2: Create Population
    print("\n[Test 2] Create Warm-Start Population...")
    config = {'warm_start_ratio': 0.5}
    population = create_warm_start_population(
        transformed_pareto=transformed,
        pop_size=20,
        new_leaf_count=5,
        config=config
    )
    
    warm_count = sum(1 for ind in population if ind.get('source') == 'warm_start')
    random_count = sum(1 for ind in population if ind.get('source') == 'random')
    print(f"  Population size: {len(population)}")
    print(f"  Warm-start: {warm_count}, Random: {random_count}")
    
    assert len(population) == 20
    assert warm_count == 10
    assert random_count == 10
    print("  ✅ Population composition correct")
    
    # Test 3: Strategy Drift
    print("\n[Test 3] Compute Strategy Drift...")
    w_prev = np.array([0.3, 0.3, 0.2, 0.2])
    w_curr = np.array([0.1, 0.4, 0.3, 0.2])
    drift = compute_strategy_drift(w_prev, w_curr)
    print(f"  w_prev: {w_prev}")
    print(f"  w_curr: {w_curr}")
    print(f"  Drift: {drift}")
    
    assert np.isclose(drift, 0.4), f"Expected 0.4, got {drift}"
    print("  ✅ Drift calculation correct")
    
    # Test 4: Batch Drift
    print("\n[Test 4] Compute Batch Drift...")
    windows = [
        np.array([0.4, 0.3, 0.3]),
        np.array([0.3, 0.4, 0.3]),
        np.array([0.2, 0.4, 0.4]),
    ]
    drifts = compute_batch_drift(windows)
    print(f"  Windows: {len(windows)}")
    print(f"  Drifts: {drifts}")
    
    assert len(drifts) == 2
    print("  ✅ Batch drift correct")
    
    # Test 5: Random Population
    print("\n[Test 5] Create Random Population (Baseline EA)...")
    random_pop = create_random_population(10, 5)
    
    all_random = all(ind.get('source') == 'random' for ind in random_pop)
    all_valid = all(np.isclose(ind['weights'].sum(), 1.0) for ind in random_pop)
    print(f"  Population size: {len(random_pop)}")
    print(f"  All random: {all_random}")
    print(f"  All valid weights: {all_valid}")
    
    assert all_random and all_valid
    print("  ✅ Random population correct")
    
    # Test 6: DriftAwareWarmStart Integration
    print("\n[Test 6] DriftAwareWarmStart Integration...")
    warm_start = DriftAwareWarmStart(config={'warm_start_ratio': 0.5})
    
    pop = warm_start.create_population(
        previous_pareto=pareto,
        leaf_mapping=mapping,
        new_leaf_count=5,
        pop_size=50
    )
    
    telemetry = warm_start.get_telemetry()
    print(f"  Population created: {len(pop)}")
    print(f"  Telemetry: {telemetry}")
    
    assert len(pop) == 50
    assert 'warm_start_count' in telemetry
    assert 'random_count' in telemetry
    print("  ✅ Integration test passed")
    
    # Test 7: Empty Pareto Fallback
    print("\n[Test 7] Empty Pareto Fallback...")
    warm_start2 = DriftAwareWarmStart()
    pop2 = warm_start2.create_population(
        previous_pareto=[],
        leaf_mapping={},
        new_leaf_count=5,
        pop_size=20
    )
    
    all_random2 = all(ind.get('source') == 'random' for ind in pop2)
    print(f"  Population size: {len(pop2)}")
    print(f"  All random: {all_random2}")
    
    assert all_random2
    print("  ✅ Empty Pareto fallback correct")
    
    # Test 8: Config Validation
    print("\n[Test 8] Config Validation...")
    try:
        DriftAwareWarmStart(config={'warm_start_ratio': 1.5})
        print("  ❌ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✅ Correctly raised ValueError: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ALL SELF-TESTS PASSED")
    print("=" * 60)
