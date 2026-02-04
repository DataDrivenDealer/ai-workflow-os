"""
Test Suite for HV-Aware Behaviour Controller (P0-20).

This test file is written BEFORE implementation following TDD principles.
Tests define the expected behavior based on EA Layer v3.1 specification Section 4.1.3.

Tested Module: projects/dgsf/scripts/hv_aware_controller.py
Spec Reference: DGSF EA Layer Specification v3.1 - Section 4.1.3 MUST

Test Categories:
1. Plateau detection integration
2. Mutation boost mechanism
3. Random individual injection
4. Partial population restart
5. Controller state management
6. Integration with NSGA-II

Author: DGSF Pipeline
Date: 2026-02-04
Stage: 5 (EA Optimizer Development)
Task: EA_DEV_001 P0-20
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
def controller_config():
    """Default HV-aware controller configuration."""
    return {
        "g_plateau": 5,          # Generations to detect plateau
        "epsilon": 0.01,         # HV improvement threshold (1%)
        "mutation_boost_factor": 2.0,   # 2x mutation rate when triggered
        "injection_ratio": 0.2,  # Inject 20% random individuals
        "restart_ratio": 0.1,    # Restart worst 10% of population
        "cooldown_generations": 3,  # Wait 3 gens after action before re-triggering
    }


@pytest.fixture
def growing_hv_trajectory():
    """HV trajectory showing steady growth (no plateau)."""
    return [0.1 + i * 0.05 for i in range(15)]


@pytest.fixture
def plateau_hv_trajectory():
    """HV trajectory with plateau at the end."""
    # Growth phase: gen 0-9
    # Plateau phase: gen 10-15
    trajectory = [0.1 + i * 0.05 for i in range(10)]  # 0.1 to 0.55
    trajectory.extend([0.55, 0.551, 0.551, 0.552, 0.552, 0.552])  # plateau
    return trajectory


@pytest.fixture
def sample_population():
    """Sample population of portfolio weights [N=20, K=10]."""
    np.random.seed(42)
    pop = np.random.dirichlet(np.ones(10), size=20)
    return pop


@pytest.fixture
def sample_fitness():
    """Sample 4-objective fitness for population [N=20, M=4]."""
    np.random.seed(42)
    fitness = np.zeros((20, 4))
    for i in range(20):
        fitness[i, 0] = -2.0 + i * 0.1  # f1: -Sharpe
        fitness[i, 1] = 0.01 + i * 0.01  # f2: MDD
        fitness[i, 2] = 0.05 + i * 0.02  # f3: Turnover
        fitness[i, 3] = 0.001 + np.random.rand() * 0.01  # f4: PE
    return fitness


# =============================================================================
# Test Class: Controller Initialization
# =============================================================================

class TestControllerInitialization:
    """Test HVAwareController initialization."""

    def test_controller_creation(self, controller_config):
        """Test controller can be created with config."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        assert controller is not None
        assert controller.g_plateau == 5
        assert controller.epsilon == 0.01
        assert controller.mutation_boost_factor == 2.0

    def test_controller_default_config(self):
        """Test controller with default config."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController()
        
        assert controller.g_plateau > 0
        assert controller.epsilon > 0
        assert controller.mutation_boost_factor > 1.0

    def test_controller_initial_state(self, controller_config):
        """Test controller starts in non-triggered state."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        assert controller.exploration_triggered is False
        assert controller.cooldown_counter == 0
        assert controller.last_action is None


# =============================================================================
# Test Class: Plateau Detection Integration
# =============================================================================

class TestPlateauDetection:
    """Test integration with P0-19 plateau detection."""

    def test_no_trigger_on_growing_trajectory(
        self, controller_config, growing_hv_trajectory
    ):
        """Test controller does NOT trigger on growing HV."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        result = controller.check_and_trigger(growing_hv_trajectory)
        
        assert result["triggered"] is False
        assert result["action"] is None

    def test_trigger_on_plateau(
        self, controller_config, plateau_hv_trajectory
    ):
        """Test controller DOES trigger when plateau detected."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        result = controller.check_and_trigger(plateau_hv_trajectory)
        
        assert result["triggered"] is True
        assert result["action"] in ["mutation_boost", "random_injection", "partial_restart"]

    def test_trigger_updates_state(
        self, controller_config, plateau_hv_trajectory
    ):
        """Test that triggering updates controller state."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        controller.check_and_trigger(plateau_hv_trajectory)
        
        assert controller.exploration_triggered is True
        assert controller.last_action is not None


# =============================================================================
# Test Class: Mutation Boost Mechanism
# =============================================================================

class TestMutationBoost:
    """Test mutation boost exploration mechanism."""

    def test_get_boosted_mutation_rate(self, controller_config):
        """Test mutation rate boost calculation."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        base_rate = 0.1
        
        boosted = controller.get_boosted_mutation_rate(base_rate)
        
        expected = base_rate * controller_config["mutation_boost_factor"]
        assert abs(boosted - expected) < 1e-9
        assert boosted > base_rate

    def test_mutation_boost_capped_at_1(self, controller_config):
        """Test mutation rate boost is capped at 1.0."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        base_rate = 0.6  # 0.6 * 2.0 = 1.2, should cap at 1.0
        
        boosted = controller.get_boosted_mutation_rate(base_rate)
        
        assert boosted <= 1.0

    def test_apply_mutation_boost_action(
        self, controller_config, plateau_hv_trajectory
    ):
        """Test applying mutation boost action."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        # Force mutation_boost action
        result = controller.apply_action(
            action="mutation_boost",
            current_mutation_rate=0.1,
        )
        
        assert "mutation_rate" in result
        assert result["mutation_rate"] > 0.1


# =============================================================================
# Test Class: Random Individual Injection
# =============================================================================

class TestRandomInjection:
    """Test random individual injection mechanism."""

    def test_generate_random_individuals(
        self, controller_config, sample_population
    ):
        """Test generation of random individuals."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        n_inject = 4  # 20 * 0.2 = 4
        K = sample_population.shape[1]
        
        new_individuals = controller.generate_random_individuals(n_inject, K)
        
        assert new_individuals.shape == (4, K)
        # Check normalization (weights sum to 1)
        assert np.allclose(new_individuals.sum(axis=1), 1.0, atol=1e-6)

    def test_inject_into_population(
        self, controller_config, sample_population
    ):
        """Test injecting random individuals into population."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        # Inject 20% = 4 individuals
        modified_pop, n_replaced = controller.inject_random_individuals(
            sample_population.copy()
        )
        
        assert modified_pop.shape == sample_population.shape
        assert n_replaced == 4
        # Check normalization maintained
        assert np.allclose(modified_pop.sum(axis=1), 1.0, atol=1e-6)

    def test_injection_replaces_worst_individuals(
        self, controller_config, sample_population, sample_fitness
    ):
        """Test that injection replaces individuals with worst crowding distance."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        modified_pop, replaced_indices = controller.inject_random_individuals(
            sample_population.copy(),
            fitness=sample_fitness,
            return_indices=True,
        )
        
        assert len(replaced_indices) == 4
        # Indices should be valid
        assert all(0 <= idx < len(sample_population) for idx in replaced_indices)


# =============================================================================
# Test Class: Partial Population Restart
# =============================================================================

class TestPartialRestart:
    """Test partial population restart mechanism."""

    def test_restart_portion_of_population(
        self, controller_config, sample_population
    ):
        """Test restarting a portion of the population."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        restart_ratio = controller_config["restart_ratio"]  # 0.1 = 2 individuals
        
        modified_pop, n_restarted = controller.partial_restart(
            sample_population.copy()
        )
        
        assert modified_pop.shape == sample_population.shape
        assert n_restarted == int(len(sample_population) * restart_ratio)

    def test_restart_maintains_population_validity(
        self, controller_config, sample_population
    ):
        """Test that restart maintains valid portfolio weights."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        modified_pop, _ = controller.partial_restart(sample_population.copy())
        
        # All weights should be non-negative
        assert np.all(modified_pop >= 0)
        # All weights should sum to 1
        assert np.allclose(modified_pop.sum(axis=1), 1.0, atol=1e-6)


# =============================================================================
# Test Class: Cooldown Mechanism
# =============================================================================

class TestCooldown:
    """Test cooldown mechanism to prevent over-triggering."""

    def test_cooldown_prevents_immediate_retrigger(
        self, controller_config, plateau_hv_trajectory
    ):
        """Test that cooldown prevents immediate re-triggering."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        # First trigger
        result1 = controller.check_and_trigger(plateau_hv_trajectory)
        assert result1["triggered"] is True
        
        # Second check should be blocked by cooldown
        result2 = controller.check_and_trigger(plateau_hv_trajectory)
        assert result2["triggered"] is False
        assert result2.get("reason") == "cooldown"

    def test_cooldown_decrements_each_generation(self, controller_config):
        """Test cooldown counter decrements each generation."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        controller.cooldown_counter = 3
        
        controller.step_generation()
        assert controller.cooldown_counter == 2
        
        controller.step_generation()
        assert controller.cooldown_counter == 1
        
        controller.step_generation()
        assert controller.cooldown_counter == 0

    def test_trigger_allowed_after_cooldown(
        self, controller_config, plateau_hv_trajectory
    ):
        """Test triggering allowed after cooldown expires."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        # First trigger
        controller.check_and_trigger(plateau_hv_trajectory)
        initial_cooldown = controller.cooldown_counter
        assert initial_cooldown == controller_config["cooldown_generations"]
        
        # Simulate cooldown period
        for _ in range(controller_config["cooldown_generations"]):
            controller.step_generation()
        
        # Cooldown should be zero now
        assert controller.cooldown_counter == 0
        
        # Should be able to trigger again (cooldown doesn't block)
        result = controller.check_and_trigger(plateau_hv_trajectory)
        # If plateau still exists, it should trigger
        assert result["triggered"] is True or result["reason"] == "no_plateau"


# =============================================================================
# Test Class: Action Selection
# =============================================================================

class TestActionSelection:
    """Test exploration action selection logic."""

    def test_select_action_returns_valid_action(self, controller_config):
        """Test action selection returns a valid action."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        action = controller.select_action()
        
        assert action in ["mutation_boost", "random_injection", "partial_restart"]

    def test_action_rotation(self, controller_config):
        """Test that actions rotate to maintain diversity."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        actions = []
        for _ in range(6):
            action = controller.select_action()
            actions.append(action)
            controller.last_action = action
        
        # Should have used multiple different actions
        assert len(set(actions)) >= 2


# =============================================================================
# Test Class: Full Integration
# =============================================================================

class TestFullIntegration:
    """Test full controller integration flow."""

    def test_full_exploration_cycle(
        self,
        controller_config,
        plateau_hv_trajectory,
        sample_population,
    ):
        """Test complete exploration cycle from detection to action."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        
        # Step 1: Check and trigger
        result = controller.check_and_trigger(plateau_hv_trajectory)
        assert result["triggered"] is True
        
        # Step 2: Apply action to population
        if result["action"] == "random_injection":
            modified_pop, n = controller.inject_random_individuals(
                sample_population.copy()
            )
            assert n > 0
        elif result["action"] == "partial_restart":
            modified_pop, n = controller.partial_restart(
                sample_population.copy()
            )
            assert n > 0
        elif result["action"] == "mutation_boost":
            boost_result = controller.apply_action(
                action="mutation_boost",
                current_mutation_rate=0.1,
            )
            assert boost_result["mutation_rate"] > 0.1

    def test_telemetry_output(
        self, controller_config, plateau_hv_trajectory
    ):
        """Test controller produces telemetry output."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        controller.check_and_trigger(plateau_hv_trajectory)
        
        telemetry = controller.get_telemetry()
        
        assert "total_triggers" in telemetry
        assert "actions_taken" in telemetry
        assert "cooldown_counter" in telemetry

    def test_controller_reset(self, controller_config, plateau_hv_trajectory):
        """Test controller reset clears state."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        controller.check_and_trigger(plateau_hv_trajectory)
        
        controller.reset()
        
        assert controller.exploration_triggered is False
        assert controller.cooldown_counter == 0
        assert controller.last_action is None


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trajectory(self, controller_config):
        """Test handling of empty HV trajectory."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        result = controller.check_and_trigger([])
        
        assert result["triggered"] is False
        assert result.get("reason") == "insufficient_data"

    def test_short_trajectory(self, controller_config):
        """Test handling of trajectory shorter than g_plateau."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        short_traj = [0.1, 0.2, 0.3]  # Only 3 points, g_plateau=5
        
        result = controller.check_and_trigger(short_traj)
        
        assert result["triggered"] is False

    def test_zero_population(self, controller_config):
        """Test handling of empty population for injection."""
        from hv_aware_controller import HVAwareController
        
        controller = HVAwareController(controller_config)
        empty_pop = np.array([]).reshape(0, 10)
        
        modified, n = controller.inject_random_individuals(empty_pop)
        
        assert modified.shape == empty_pop.shape
        assert n == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
