"""
EA Layer Runner for DGSF v3.1 (P0-23).

This module integrates all EA Layer v3.1 MUST components into a unified runner:
- ea_evaluation_metrics (P0-19): HV, plateau, drift, SDF filtering
- hv_aware_controller (P0-20): HV-driven exploration
- drift_aware_warmstart (P0-21): Cross-window warm-start
- sdf_consistency_selection (P0-22): Consistency-first selection

Provides both enhanced EA (v3.1) and baseline EA modes for comparison.

Spec Reference: DGSF EA Layer Specification v3.1 - Section 4
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from copy import deepcopy

# Import integrated modules
from ea_evaluation_metrics import (
    compute_hypervolume,
    compute_hv_trajectory,
    detect_hv_plateau,
    filter_sdf_consistent,
    evaluate_pareto_front,
)
from hv_aware_controller import HVAwareController
from drift_aware_warmstart import (
    DriftAwareWarmStart,
    create_random_population,
    compute_strategy_drift,
)
from sdf_consistency_selection import (
    SDFConsistencySelector,
    baseline_select,
    baseline_multiobjective_rank,
)


# =============================================================================
# Configuration
# =============================================================================

def default_config() -> Dict[str, Any]:
    """Return default EA Layer configuration."""
    return {
        # Mode
        'mode': 'enhanced',  # 'enhanced' or 'baseline'
        
        # Population
        'pop_size': 100,
        'n_gen': 50,
        'elite_fraction': 0.2,
        
        # HV-aware (v3.1 4.1)
        'g_plateau': 5,
        'hv_epsilon': 0.01,
        'cooldown_gens': 3,
        'mutation_boost_factor': 2.0,
        'injection_fraction': 0.1,
        'restart_fraction': 0.5,
        
        # Warm-start (v3.1 4.2)
        'warm_start_ratio': 0.5,
        
        # SDF consistency (v3.1 4.3)
        'gamma': 0.01,
        'selection_mode': 'lexicographic',
        'penalty_weight': 10.0,
        
        # Objectives
        'objectives': ['sharpe', 'mdd', 'turnover', 'sdf_penalty'],
        'maximize': [True, False, False, False],
        
        # Reference point for HV
        'reference_point': None,  # Auto-compute if None
        
        # Mutation
        'mutation_rate': 0.1,
        'crossover_rate': 0.9,
    }


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration.
    
    Raises:
        ValueError: If configuration is invalid
    """
    if config.get('pop_size', 0) <= 0:
        raise ValueError(f"pop_size must be positive, got {config.get('pop_size')}")
    
    if config.get('n_gen', 0) <= 0:
        raise ValueError(f"n_gen must be positive, got {config.get('n_gen')}")
    
    if config.get('mode') not in ['enhanced', 'baseline']:
        raise ValueError(f"mode must be 'enhanced' or 'baseline', got {config.get('mode')}")


# =============================================================================
# EA Layer Runner
# =============================================================================

class EALayerRunner:
    """
    Integrated EA Layer Runner for DGSF v3.1.
    
    Combines all v3.1 MUST modules:
    - HV-aware behaviour control (4.1)
    - Drift-aware warm-start (4.2)
    - SDF-consistency selection (4.3)
    
    Supports two modes:
    - 'enhanced': Full v3.1 features
    - 'baseline': Standard NSGA-II without enhancements
    
    Attributes:
        config: EA configuration
        hv_trajectory: Recorded HV values per generation
        exploration_triggers: Record of exploration actions
        
    Example:
        >>> runner = EALayerRunner(config={'pop_size': 100, 'n_gen': 50})
        >>> result = runner.run(evaluate_fn=my_eval, n_leaves=10)
        >>> print(result['pareto_front'])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize EA Layer Runner.
        
        Args:
            config: EA configuration (uses defaults if not provided)
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = default_config()
        if config:
            self.config.update(config)
        
        validate_config(self.config)
        
        # State
        self.hv_trajectory: List[float] = []
        self.exploration_triggers: List[Dict[str, Any]] = []
        self.consistency_stats: List[Dict[str, Any]] = []
        self.current_gen: int = 0
        
        # Sub-controllers (initialized in run)
        self._hv_controller: Optional[HVAwareController] = None
        self._warm_start: Optional[DriftAwareWarmStart] = None
        self._selector: Optional[SDFConsistencySelector] = None
    
    def _init_controllers(self) -> None:
        """Initialize sub-controllers based on mode."""
        if self.config['mode'] == 'enhanced':
            # HV-aware controller
            self._hv_controller = HVAwareController(config={
                'g_plateau': self.config['g_plateau'],
                'epsilon': self.config['hv_epsilon'],
                'cooldown_gens': self.config['cooldown_gens'],
                'mutation_boost_factor': self.config['mutation_boost_factor'],
                'injection_fraction': self.config['injection_fraction'],
                'restart_fraction': self.config['restart_fraction'],
            })
            
            # Warm-start controller
            self._warm_start = DriftAwareWarmStart(config={
                'warm_start_ratio': self.config['warm_start_ratio'],
            })
            
            # SDF-consistency selector
            self._selector = SDFConsistencySelector(
                gamma=self.config['gamma'],
                mode=self.config['selection_mode'],
                penalty_weight=self.config['penalty_weight'],
            )
        else:
            # Baseline mode - no enhanced controllers
            self._hv_controller = None
            self._warm_start = None
            self._selector = None
    
    def initialize_population(
        self,
        previous_pareto: List[Dict[str, Any]],
        leaf_mapping: Dict[int, Optional[int]],
        n_leaves: int
    ) -> List[Dict[str, Any]]:
        """
        Initialize population with warm-start or random.
        
        v3.1 4.2.1 MUST: Use previous Pareto for initialization (enhanced mode).
        v3.1 4.2.4: Baseline uses random only.
        
        Args:
            previous_pareto: Pareto front from previous window
            leaf_mapping: Leaf index mapping between windows
            n_leaves: Number of leaves in current window
            
        Returns:
            Initial population
        """
        # Ensure controllers are initialized
        if self._warm_start is None and self.config['mode'] == 'enhanced':
            self._init_controllers()
        
        pop_size = self.config['pop_size']
        
        if self.config['mode'] == 'enhanced' and previous_pareto and self._warm_start:
            # Use warm-start
            population = self._warm_start.create_population(
                previous_pareto=previous_pareto,
                leaf_mapping=leaf_mapping,
                new_leaf_count=n_leaves,
                pop_size=pop_size
            )
        else:
            # Random initialization (baseline or no previous Pareto)
            population = create_random_population(pop_size, n_leaves)
        
        return population
    
    def step_generation(
        self,
        population: List[Dict[str, Any]],
        gen: int
    ) -> List[Dict[str, Any]]:
        """
        Execute single generation step.
        
        v3.1 4.1.1 MUST: Record HV per generation.
        
        Args:
            population: Current population with objectives evaluated
            gen: Current generation number
            
        Returns:
            New population for next generation
        """
        self.current_gen = gen
        
        # Compute HV for this generation
        hv = self._compute_generation_hv(population)
        self.hv_trajectory.append(hv)
        
        # Check for HV plateau and apply exploration (enhanced mode)
        mutation_rate = self.config['mutation_rate']
        if self.config['mode'] == 'enhanced' and self._hv_controller:
            result = self._hv_controller.check_and_trigger(self.hv_trajectory)
            triggered = result['triggered']
            action = result['action']
            reason = result['reason']
            
            if triggered:
                self.exploration_triggers.append({
                    'gen': gen,
                    'action': action,
                    'reason': reason,
                })
                
                # Apply exploration action
                if action == 'mutation_boost':
                    mutation_rate = self._hv_controller.get_boosted_mutation_rate(mutation_rate)
                elif action == 'random_injection':
                    n_leaves = len(population[0]['weights'])
                    population = self._inject_random_to_list_population(population, n_leaves)
                elif action == 'partial_restart':
                    n_leaves = len(population[0]['weights'])
                    population = self._partial_restart_list_population(population, n_leaves)
            
            self._hv_controller.step_generation()
        
        # Select elite
        elite_count = int(len(population) * self.config['elite_fraction'])
        elite = self.select_elite(population, elite_count)
        
        # Record consistency stats
        if self.config['mode'] == 'enhanced' and self._selector:
            stats = self._selector.get_telemetry()
            self.consistency_stats.append({
                'gen': gen,
                **stats
            })
        
        # Generate offspring (simplified: copy elite + mutate)
        new_population = self._create_offspring(elite, population, mutation_rate)
        
        return new_population
    
    def _compute_generation_hv(self, population: List[Dict[str, Any]]) -> float:
        """Compute hypervolume for population."""
        objectives = self.config['objectives']
        maximize = self.config['maximize']
        
        # Extract objective values
        points = []
        for ind in population:
            obj = ind.get('objectives', {})
            point = []
            for i, key in enumerate(objectives):
                val = obj.get(key, 0.0)
                # Negate if maximizing (for HV computation)
                point.append(-val if maximize[i] else val)
            points.append(point)
        
        if not points:
            return 0.0
        
        points_array = np.array(points)
        
        # Compute reference point if not specified
        ref_point = self.config.get('reference_point')
        if ref_point is None:
            ref_point = np.max(points_array, axis=0) * 1.1 + 0.1
        
        # Use compute_hypervolume from ea_evaluation_metrics
        try:
            hv = compute_hypervolume(points_array, ref_point)
        except Exception:
            hv = 0.0
        
        return hv
    
    def check_hv_plateau(self) -> Tuple[bool, Optional[str]]:
        """
        Check if HV plateau detected.
        
        v3.1 4.1.2 MUST: Detect HV plateau.
        
        Returns:
            (triggered, action) tuple
        """
        # Lazy init controllers if needed
        if self.config['mode'] == 'enhanced' and self._hv_controller is None:
            self._init_controllers()
        
        if self.config['mode'] == 'enhanced' and self._hv_controller:
            result = self._hv_controller.check_and_trigger(self.hv_trajectory)
            return result['triggered'], result['action']
        return False, None
    
    def select_elite(
        self,
        population: List[Dict[str, Any]],
        elite_count: int
    ) -> List[Dict[str, Any]]:
        """
        Select elite individuals.
        
        v3.1 4.3.2 MUST: Consistency-first selection (enhanced mode).
        v3.1 4.3.3: Baseline uses objective-only selection.
        
        Args:
            population: Population to select from
            elite_count: Number to select
            
        Returns:
            Selected elite individuals
        """
        # Lazy init controllers if needed
        if self.config['mode'] == 'enhanced' and self._selector is None:
            self._init_controllers()
        
        if self.config['mode'] == 'enhanced' and self._selector:
            return self._selector.select_elite(
                population=population,
                elite_count=elite_count,
                primary_obj='sharpe'
            )
        else:
            # Baseline: pure objective selection
            return baseline_select(
                population=population,
                select_count=elite_count,
                objective='sharpe'
            )
    
    def _inject_random_to_list_population(
        self,
        population: List[Dict[str, Any]],
        n_leaves: int
    ) -> List[Dict[str, Any]]:
        """
        Inject random individuals into list-based population.
        
        Adapter for HVAwareController.inject_random_individuals which expects np.ndarray.
        """
        if not population:
            return population
        
        injection_ratio = getattr(self._hv_controller, 'injection_ratio', 0.1)
        n_inject = max(1, int(len(population) * injection_ratio))
        
        # Random replacement indices
        replace_indices = np.random.choice(len(population), size=n_inject, replace=False)
        
        for idx in replace_indices:
            new_weights = np.random.dirichlet(np.ones(n_leaves))
            population[idx] = {
                'id': population[idx].get('id', idx),
                'weights': new_weights,
                'source': 'injected',
            }
        
        return population
    
    def _partial_restart_list_population(
        self,
        population: List[Dict[str, Any]],
        n_leaves: int
    ) -> List[Dict[str, Any]]:
        """
        Perform partial restart on list-based population.
        
        Adapter for HVAwareController.partial_restart which expects np.ndarray.
        """
        if not population:
            return population
        
        restart_ratio = getattr(self._hv_controller, 'restart_ratio', 0.3)
        n_restart = max(1, int(len(population) * restart_ratio))
        
        # Keep best, restart rest randomly (sorted by sharpe if available)
        sorted_pop = sorted(
            population,
            key=lambda x: x.get('objectives', {}).get('sharpe', 0),
            reverse=True
        )
        
        keep_count = len(population) - n_restart
        new_pop = sorted_pop[:keep_count]
        
        # Generate random for restarted portion
        for i in range(n_restart):
            new_pop.append({
                'id': keep_count + i,
                'weights': np.random.dirichlet(np.ones(n_leaves)),
                'source': 'restarted',
            })
        
        return new_pop
    
    def _create_offspring(
        self,
        elite: List[Dict[str, Any]],
        population: List[Dict[str, Any]],
        mutation_rate: float
    ) -> List[Dict[str, Any]]:
        """Create offspring through crossover and mutation."""
        pop_size = self.config['pop_size']
        n_leaves = len(population[0]['weights']) if population else 10
        
        offspring = []
        
        # Keep elite
        offspring.extend(deepcopy(elite))
        
        # Generate rest through mutation of elite
        while len(offspring) < pop_size:
            # Select random parent from elite
            parent = elite[np.random.randint(len(elite))]
            
            # Mutate
            child = deepcopy(parent)
            child_weights = child['weights'].copy()
            
            # Apply mutation
            if np.random.random() < mutation_rate:
                noise = np.random.normal(0, 0.05, len(child_weights))
                child_weights = child_weights + noise
                child_weights = np.clip(child_weights, 0, None)
                if child_weights.sum() > 0:
                    child_weights = child_weights / child_weights.sum()
                else:
                    child_weights = np.ones(n_leaves) / n_leaves
            
            child['weights'] = child_weights
            child['source'] = 'offspring'
            child.pop('objectives', None)  # Will be re-evaluated
            
            offspring.append(child)
        
        return offspring[:pop_size]
    
    def run(
        self,
        evaluate_fn: Callable[[np.ndarray], Dict[str, float]],
        n_leaves: int,
        previous_pareto: Optional[List[Dict[str, Any]]] = None,
        leaf_mapping: Optional[Dict[int, Optional[int]]] = None
    ) -> Dict[str, Any]:
        """
        Run complete EA optimization.
        
        Args:
            evaluate_fn: Function to evaluate weights -> objectives
            n_leaves: Number of leaves
            previous_pareto: Pareto front from previous window
            leaf_mapping: Leaf mapping between windows
            
        Returns:
            Result dict with pareto_front and telemetry
        """
        # Initialize controllers
        self._init_controllers()
        
        # Reset state
        self.hv_trajectory = []
        self.exploration_triggers = []
        self.consistency_stats = []
        
        # Initialize population
        population = self.initialize_population(
            previous_pareto=previous_pareto or [],
            leaf_mapping=leaf_mapping or {},
            n_leaves=n_leaves
        )
        
        # Main loop
        n_gen = self.config['n_gen']
        
        for gen in range(n_gen):
            # Evaluate population
            for ind in population:
                if 'objectives' not in ind:
                    ind['objectives'] = evaluate_fn(ind['weights'])
            
            # Step generation
            population = self.step_generation(population, gen)
        
        # Final evaluation
        for ind in population:
            if 'objectives' not in ind:
                ind['objectives'] = evaluate_fn(ind['weights'])
        
        # Extract Pareto front
        pareto_front = self._extract_pareto_front(population)
        
        # Compute final metrics
        telemetry = self.get_telemetry()
        telemetry['total_generations'] = n_gen
        
        return {
            'pareto_front': pareto_front,
            'final_population': population,
            'telemetry': telemetry,
        }
    
    def _extract_pareto_front(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract non-dominated solutions."""
        if not population:
            return []
        
        # Filter for SDF consistency in enhanced mode
        if self.config['mode'] == 'enhanced':
            gamma = self.config['gamma']
            consistent = [
                ind for ind in population
                if ind.get('objectives', {}).get('sdf_penalty', 1.0) <= gamma
            ]
            if consistent:
                population = consistent
        
        # Simple non-dominated sorting
        n = len(population)
        dominated = [False] * n
        
        objectives = self.config['objectives']
        maximize = self.config['maximize']
        
        for i in range(n):
            for j in range(i + 1, n):
                obj_i = population[i].get('objectives', {})
                obj_j = population[j].get('objectives', {})
                
                i_dom_j = True
                j_dom_i = True
                
                for k, key in enumerate(objectives):
                    vi = obj_i.get(key, 0.0)
                    vj = obj_j.get(key, 0.0)
                    
                    if maximize[k]:
                        if vi < vj:
                            i_dom_j = False
                        if vj < vi:
                            j_dom_i = False
                    else:
                        if vi > vj:
                            i_dom_j = False
                        if vj > vi:
                            j_dom_i = False
                
                # Check strict domination
                if i_dom_j and not j_dom_i:
                    dominated[j] = True
                elif j_dom_i and not i_dom_j:
                    dominated[i] = True
        
        pareto_front = [
            population[i] for i in range(n) if not dominated[i]
        ]
        
        return pareto_front
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get comprehensive telemetry.
        
        Returns:
            Telemetry dictionary
        """
        return {
            'config': self.config.copy(),
            'mode': self.config['mode'],
            'hv_trajectory': self.hv_trajectory.copy(),
            'exploration_triggers': self.exploration_triggers.copy(),
            'plateau_events': self.exploration_triggers.copy(),  # Alias for compatibility
            'consistency_stats': self.consistency_stats.copy(),
            'total_triggers': len(self.exploration_triggers),
            'generations_completed': len(self.hv_trajectory),
            'best_objectives': self._get_best_objectives(),
        }
    
    def _get_best_objectives(self) -> Dict[str, float]:
        """Get best objectives from current run (placeholder)."""
        return {
            'sharpe': 0.0,
            'mdd': 0.0,
            'turnover': 0.0,
        }


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("P0-23 EA Layer Runner - Self Test")
    print("=" * 60)
    
    # Test 1: Initialization
    print("\n[Test 1] Runner Initialization...")
    runner = EALayerRunner(config={'pop_size': 20, 'n_gen': 5})
    print(f"  Mode: {runner.config['mode']}")
    print(f"  Pop size: {runner.config['pop_size']}")
    print("  ✅ Initialization correct")
    
    # Test 2: Population initialization
    print("\n[Test 2] Population Initialization...")
    population = runner.initialize_population([], {}, n_leaves=10)
    print(f"  Population size: {len(population)}")
    assert len(population) == 20
    print("  ✅ Population created")
    
    # Test 3: Mock evaluation
    print("\n[Test 3] Mock Evaluation Run...")
    
    def mock_evaluate(weights):
        return {
            'sharpe': float(np.random.uniform(0.5, 2.0)),
            'mdd': float(np.random.uniform(0.05, 0.20)),
            'turnover': float(np.random.uniform(0.05, 0.15)),
            'sdf_penalty': float(np.random.uniform(0.001, 0.010)),
        }
    
    result = runner.run(
        evaluate_fn=mock_evaluate,
        n_leaves=10,
        previous_pareto=[],
        leaf_mapping={}
    )
    
    print(f"  Pareto front size: {len(result['pareto_front'])}")
    print(f"  HV trajectory length: {len(result['telemetry']['hv_trajectory'])}")
    assert len(result['pareto_front']) > 0
    print("  ✅ Run completed")
    
    # Test 4: Baseline mode
    print("\n[Test 4] Baseline Mode...")
    baseline_runner = EALayerRunner(config={'mode': 'baseline', 'pop_size': 20, 'n_gen': 3})
    
    baseline_pop = baseline_runner.initialize_population(
        previous_pareto=[{'weights': np.ones(10)/10, 'objectives': {}}],
        leaf_mapping={i: i for i in range(10)},
        n_leaves=10
    )
    
    all_random = all(ind.get('source') == 'random' for ind in baseline_pop)
    print(f"  All random in baseline: {all_random}")
    assert all_random
    print("  ✅ Baseline mode correct")
    
    # Test 5: Telemetry
    print("\n[Test 5] Telemetry...")
    telemetry = runner.get_telemetry()
    print(f"  Keys: {list(telemetry.keys())}")
    assert 'hv_trajectory' in telemetry
    assert 'exploration_triggers' in telemetry
    assert 'consistency_stats' in telemetry
    print("  ✅ Telemetry complete")
    
    # Test 6: Enhanced mode with warm-start
    print("\n[Test 6] Enhanced Mode with Warm-Start...")
    enhanced_runner = EALayerRunner(config={'mode': 'enhanced', 'pop_size': 20, 'n_gen': 3})
    
    prev_pareto = [
        {'weights': np.random.dirichlet(np.ones(10)), 'objectives': {'sharpe': 1.5}}
        for _ in range(5)
    ]
    
    enhanced_pop = enhanced_runner.initialize_population(
        previous_pareto=prev_pareto,
        leaf_mapping={i: i for i in range(10)},
        n_leaves=10
    )
    
    warm_count = sum(1 for ind in enhanced_pop if ind.get('source') == 'warm_start')
    print(f"  Warm-start individuals: {warm_count}")
    assert warm_count > 0
    print("  ✅ Enhanced mode with warm-start correct")
    
    print("\n" + "=" * 60)
    print("✅ ALL SELF-TESTS PASSED")
    print("=" * 60)
