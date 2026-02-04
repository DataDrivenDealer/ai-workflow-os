"""
HV-Aware Behaviour Controller for DGSF EA Layer v3.1 (P0-20).

This module implements the HV-aware behaviour control mechanism required by
EA Layer v3.1 Section 4.1.3 MUST.

Key features:
1. Integration with P0-19 plateau detection
2. Mutation boost mechanism (2x-3x mutation rate)
3. Random individual injection (10-20% of population)
4. Partial population restart (worst 10%)
5. Cooldown mechanism to prevent over-triggering

Spec Reference: DGSF EA Layer Specification v3.1 - Section 4.1

Author: DGSF Pipeline
Date: 2026-02-04
Stage: 5 (EA Optimizer Development)
Task: EA_DEV_001 P0-20
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Import P0-19 plateau detection
import sys
from pathlib import Path
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ea_evaluation_metrics import detect_hv_plateau


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class HVAwareConfig:
    """Configuration for HV-Aware Behaviour Controller."""
    g_plateau: int = 5              # Generations to detect plateau
    epsilon: float = 0.01           # HV improvement threshold (1%)
    mutation_boost_factor: float = 2.0   # Mutation rate multiplier
    injection_ratio: float = 0.2    # Fraction of population to inject
    restart_ratio: float = 0.1      # Fraction of population to restart
    cooldown_generations: int = 3   # Generations to wait after action
    random_seed: int = 42           # Random seed for reproducibility


# =============================================================================
# HV-Aware Controller
# =============================================================================

class HVAwareController:
    """
    HV-Aware Behaviour Controller for EA Layer v3.1.
    
    Implements Section 4.1 MUST requirements:
    - 4.1.1: Record HV per generation (via P0-19)
    - 4.1.2: Detect HV plateau (via P0-19)
    - 4.1.3: HV-driven exploration (THIS MODULE)
    
    The controller monitors HV trajectory and triggers exploration actions
    when plateau is detected to escape local optima.
    
    Parameters
    ----------
    config : dict or HVAwareConfig, optional
        Configuration parameters. If None, uses defaults.
    
    Attributes
    ----------
    exploration_triggered : bool
        Whether exploration is currently active.
    cooldown_counter : int
        Generations remaining in cooldown period.
    last_action : str or None
        Last exploration action taken.
    
    Examples
    --------
    >>> controller = HVAwareController({"g_plateau": 5, "epsilon": 0.01})
    >>> result = controller.check_and_trigger(hv_trajectory)
    >>> if result["triggered"]:
    ...     modified_pop, n = controller.inject_random_individuals(population)
    """
    
    # Available exploration actions
    ACTIONS = ["mutation_boost", "random_injection", "partial_restart"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the HV-Aware controller."""
        if config is None:
            config = {}
        
        # Parse configuration
        self.g_plateau = config.get("g_plateau", 5)
        self.epsilon = config.get("epsilon", 0.01)
        self.mutation_boost_factor = config.get("mutation_boost_factor", 2.0)
        self.injection_ratio = config.get("injection_ratio", 0.2)
        self.restart_ratio = config.get("restart_ratio", 0.1)
        self.cooldown_generations = config.get("cooldown_generations", 3)
        self.random_seed = config.get("random_seed", 42)
        
        # Initialize random generator
        self.rng = np.random.default_rng(self.random_seed)
        
        # State variables
        self.exploration_triggered = False
        self.cooldown_counter = 0
        self.last_action: Optional[str] = None
        self.action_index = 0  # For action rotation
        
        # Telemetry
        self.total_triggers = 0
        self.actions_taken: List[str] = []
    
    def reset(self) -> None:
        """Reset controller state."""
        self.exploration_triggered = False
        self.cooldown_counter = 0
        self.last_action = None
        self.action_index = 0
        self.total_triggers = 0
        self.actions_taken = []
        self.rng = np.random.default_rng(self.random_seed)
    
    def step_generation(self) -> None:
        """
        Advance controller by one generation.
        
        Decrements cooldown counter if active.
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        # Reset exploration flag after cooldown
        if self.cooldown_counter == 0:
            self.exploration_triggered = False
    
    # =========================================================================
    # Plateau Detection & Trigger
    # =========================================================================
    
    def check_and_trigger(
        self,
        hv_trajectory: List[float],
    ) -> Dict[str, Any]:
        """
        Check HV trajectory for plateau and trigger exploration if needed.
        
        Parameters
        ----------
        hv_trajectory : List[float]
            HV values for each generation so far.
        
        Returns
        -------
        result : dict
            Dictionary with:
            - "triggered": bool, whether exploration was triggered
            - "action": str or None, the exploration action to take
            - "reason": str, explanation of decision
        """
        # Handle edge cases
        if len(hv_trajectory) == 0:
            return {
                "triggered": False,
                "action": None,
                "reason": "insufficient_data",
            }
        
        if len(hv_trajectory) < self.g_plateau + 1:
            return {
                "triggered": False,
                "action": None,
                "reason": "trajectory_too_short",
            }
        
        # Check cooldown
        if self.cooldown_counter > 0:
            return {
                "triggered": False,
                "action": None,
                "reason": "cooldown",
            }
        
        # Use P0-19 plateau detection
        is_plateau, plateau_start = detect_hv_plateau(
            hv_trajectory,
            g_plateau=self.g_plateau,
            epsilon=self.epsilon,
        )
        
        if not is_plateau:
            return {
                "triggered": False,
                "action": None,
                "reason": "no_plateau",
            }
        
        # Plateau detected - trigger exploration!
        action = self.select_action()
        
        # Update state
        self.exploration_triggered = True
        self.cooldown_counter = self.cooldown_generations
        self.last_action = action
        self.total_triggers += 1
        self.actions_taken.append(action)
        
        return {
            "triggered": True,
            "action": action,
            "reason": f"plateau_detected_at_gen_{plateau_start}",
            "plateau_start": plateau_start,
        }
    
    def select_action(self) -> str:
        """
        Select the next exploration action.
        
        Uses round-robin rotation to ensure diversity of exploration strategies.
        
        Returns
        -------
        action : str
            One of "mutation_boost", "random_injection", "partial_restart".
        """
        # Rotate through actions to maintain diversity
        action = self.ACTIONS[self.action_index % len(self.ACTIONS)]
        self.action_index += 1
        return action
    
    # =========================================================================
    # Mutation Boost
    # =========================================================================
    
    def get_boosted_mutation_rate(self, base_rate: float) -> float:
        """
        Get boosted mutation rate for exploration.
        
        Parameters
        ----------
        base_rate : float
            Original mutation rate.
        
        Returns
        -------
        boosted_rate : float
            Boosted mutation rate, capped at 1.0.
        """
        boosted = base_rate * self.mutation_boost_factor
        return min(boosted, 1.0)
    
    def apply_action(
        self,
        action: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Apply an exploration action.
        
        Parameters
        ----------
        action : str
            The action to apply.
        **kwargs : dict
            Action-specific parameters.
        
        Returns
        -------
        result : dict
            Action results (e.g., new mutation rate).
        """
        if action == "mutation_boost":
            current_rate = kwargs.get("current_mutation_rate", 0.1)
            new_rate = self.get_boosted_mutation_rate(current_rate)
            return {
                "mutation_rate": new_rate,
                "boost_factor": self.mutation_boost_factor,
            }
        
        elif action == "random_injection":
            population = kwargs.get("population")
            if population is not None:
                modified, n = self.inject_random_individuals(population)
                return {
                    "modified_population": modified,
                    "n_injected": n,
                }
            return {"error": "no_population_provided"}
        
        elif action == "partial_restart":
            population = kwargs.get("population")
            if population is not None:
                modified, n = self.partial_restart(population)
                return {
                    "modified_population": modified,
                    "n_restarted": n,
                }
            return {"error": "no_population_provided"}
        
        return {"error": f"unknown_action_{action}"}
    
    # =========================================================================
    # Random Individual Injection
    # =========================================================================
    
    def generate_random_individuals(
        self,
        n_individuals: int,
        n_dimensions: int,
    ) -> np.ndarray:
        """
        Generate random portfolio weight individuals.
        
        Uses Dirichlet distribution to generate valid portfolio weights
        that sum to 1 and are non-negative.
        
        Parameters
        ----------
        n_individuals : int
            Number of individuals to generate.
        n_dimensions : int
            Dimension of each individual (number of assets).
        
        Returns
        -------
        individuals : np.ndarray
            Array of shape [n_individuals, n_dimensions] with valid weights.
        """
        if n_individuals <= 0 or n_dimensions <= 0:
            return np.array([]).reshape(0, max(n_dimensions, 1))
        
        # Use Dirichlet to generate normalized weights
        individuals = self.rng.dirichlet(np.ones(n_dimensions), size=n_individuals)
        return individuals
    
    def inject_random_individuals(
        self,
        population: np.ndarray,
        fitness: Optional[np.ndarray] = None,
        return_indices: bool = False,
    ) -> Tuple[np.ndarray, Any]:
        """
        Inject random individuals into population.
        
        Replaces the worst individuals (by crowding distance or random if
        no fitness provided).
        
        Parameters
        ----------
        population : np.ndarray
            Current population, shape [N, K].
        fitness : np.ndarray, optional
            Fitness values, shape [N, M]. Used to identify worst individuals.
        return_indices : bool
            If True, return replaced indices instead of count.
        
        Returns
        -------
        modified_population : np.ndarray
            Population with injected individuals.
        n_replaced : int or list
            Number of replaced individuals, or indices if return_indices=True.
        """
        if population.size == 0:
            return population, [] if return_indices else 0
        
        N, K = population.shape
        n_inject = max(1, int(N * self.injection_ratio))
        
        # Generate new individuals
        new_individuals = self.generate_random_individuals(n_inject, K)
        
        # Determine which individuals to replace
        if fitness is not None:
            # Replace individuals with worst sum of objectives (simple heuristic)
            # In production, use crowding distance
            objective_sum = fitness.sum(axis=1)
            replace_indices = np.argsort(objective_sum)[-n_inject:]
        else:
            # Random replacement
            replace_indices = self.rng.choice(N, size=n_inject, replace=False)
        
        # Perform replacement
        modified = population.copy()
        for i, idx in enumerate(replace_indices):
            modified[idx] = new_individuals[i]
        
        if return_indices:
            return modified, list(replace_indices)
        return modified, n_inject
    
    # =========================================================================
    # Partial Population Restart
    # =========================================================================
    
    def partial_restart(
        self,
        population: np.ndarray,
        fitness: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Restart a portion of the population.
        
        Parameters
        ----------
        population : np.ndarray
            Current population, shape [N, K].
        fitness : np.ndarray, optional
            Fitness values for identifying worst individuals.
        
        Returns
        -------
        modified_population : np.ndarray
            Population with restarted individuals.
        n_restarted : int
            Number of restarted individuals.
        """
        if population.size == 0:
            return population, 0
        
        N, K = population.shape
        n_restart = max(1, int(N * self.restart_ratio))
        
        # Generate new random individuals
        new_individuals = self.generate_random_individuals(n_restart, K)
        
        # Determine which individuals to restart
        if fitness is not None:
            objective_sum = fitness.sum(axis=1)
            restart_indices = np.argsort(objective_sum)[-n_restart:]
        else:
            restart_indices = self.rng.choice(N, size=n_restart, replace=False)
        
        # Perform restart
        modified = population.copy()
        for i, idx in enumerate(restart_indices):
            modified[idx] = new_individuals[i]
        
        return modified, n_restart
    
    # =========================================================================
    # Telemetry
    # =========================================================================
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get controller telemetry for logging.
        
        Returns
        -------
        telemetry : dict
            Dictionary with controller state and statistics.
        """
        return {
            "total_triggers": self.total_triggers,
            "actions_taken": self.actions_taken.copy(),
            "last_action": self.last_action,
            "cooldown_counter": self.cooldown_counter,
            "exploration_triggered": self.exploration_triggered,
            "config": {
                "g_plateau": self.g_plateau,
                "epsilon": self.epsilon,
                "mutation_boost_factor": self.mutation_boost_factor,
                "injection_ratio": self.injection_ratio,
                "restart_ratio": self.restart_ratio,
                "cooldown_generations": self.cooldown_generations,
            },
        }


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HV-Aware Behaviour Controller - Self Test")
    print("=" * 60)
    
    # Test 1: Controller Creation
    print("\n[Test 1] Controller Creation...")
    config = {
        "g_plateau": 5,
        "epsilon": 0.01,
        "mutation_boost_factor": 2.0,
        "injection_ratio": 0.2,
    }
    controller = HVAwareController(config)
    assert controller.g_plateau == 5
    assert controller.exploration_triggered is False
    print("  ✅ Controller created successfully")
    
    # Test 2: No trigger on growing trajectory
    print("\n[Test 2] No Trigger on Growing Trajectory...")
    growing_traj = [0.1 + i * 0.05 for i in range(15)]
    result = controller.check_and_trigger(growing_traj)
    assert result["triggered"] is False
    print(f"  Triggered: {result['triggered']}, Reason: {result['reason']}")
    print("  ✅ Correctly did not trigger")
    
    # Test 3: Trigger on plateau
    print("\n[Test 3] Trigger on Plateau...")
    plateau_traj = [0.1 + i * 0.05 for i in range(10)]
    plateau_traj.extend([0.55] * 6)  # Add plateau
    result = controller.check_and_trigger(plateau_traj)
    assert result["triggered"] is True
    print(f"  Triggered: {result['triggered']}, Action: {result['action']}")
    print("  ✅ Correctly triggered exploration")
    
    # Test 4: Cooldown prevents retrigger
    print("\n[Test 4] Cooldown Prevents Retrigger...")
    result2 = controller.check_and_trigger(plateau_traj)
    assert result2["triggered"] is False
    assert result2["reason"] == "cooldown"
    print(f"  Triggered: {result2['triggered']}, Reason: {result2['reason']}")
    print("  ✅ Cooldown working correctly")
    
    # Test 5: Mutation Boost
    print("\n[Test 5] Mutation Boost...")
    base_rate = 0.1
    boosted = controller.get_boosted_mutation_rate(base_rate)
    assert boosted == base_rate * config["mutation_boost_factor"]
    print(f"  Base rate: {base_rate}, Boosted: {boosted}")
    print("  ✅ Mutation boost working")
    
    # Test 6: Random Injection
    print("\n[Test 6] Random Individual Injection...")
    np.random.seed(42)
    population = np.random.dirichlet(np.ones(10), size=20)
    controller.reset()  # Clear cooldown
    modified, n = controller.inject_random_individuals(population)
    assert modified.shape == population.shape
    assert n == 4  # 20 * 0.2 = 4
    assert np.allclose(modified.sum(axis=1), 1.0, atol=1e-6)
    print(f"  Population: {population.shape}, Injected: {n}")
    print("  ✅ Random injection working")
    
    # Test 7: Partial Restart
    print("\n[Test 7] Partial Restart...")
    modified, n = controller.partial_restart(population)
    assert n == 2  # 20 * 0.1 = 2
    assert np.allclose(modified.sum(axis=1), 1.0, atol=1e-6)
    print(f"  Restarted: {n} individuals")
    print("  ✅ Partial restart working")
    
    # Test 8: Telemetry
    print("\n[Test 8] Telemetry...")
    controller.reset()
    controller.check_and_trigger(plateau_traj)
    telemetry = controller.get_telemetry()
    print(f"  Total triggers: {telemetry['total_triggers']}")
    print(f"  Actions taken: {telemetry['actions_taken']}")
    print("  ✅ Telemetry working")
    
    print("\n" + "=" * 60)
    print("✅ ALL SELF-TESTS PASSED")
    print("=" * 60)
