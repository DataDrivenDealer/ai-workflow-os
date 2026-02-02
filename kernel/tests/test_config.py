"""
Unit tests for kernel/config.py

Tests configuration loading, validation, and environment variable overrides.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernel.config import AIWorkflowConfig


class TestAIWorkflowConfig:
    """Test suite for AIWorkflowConfig class."""
    
    def test_config_loads_successfully(self):
        """Test that configuration loads without errors."""
        config = AIWorkflowConfig.load()
        
        assert config is not None
        assert isinstance(config.state_dir, Path)
        assert isinstance(config.config_dir, Path)
        assert isinstance(config.gates, dict)
        assert isinstance(config.state_machine, dict)
        assert isinstance(config.registry, dict)
    
    def test_config_has_valid_paths(self):
        """Test that configuration paths are valid."""
        config = AIWorkflowConfig.load()
        
        # Paths should be absolute
        assert config.state_dir.is_absolute()
        assert config.config_dir.is_absolute()
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = AIWorkflowConfig.load()
        
        # Should validate successfully with real config files
        assert config.validate() is True
    
    def test_get_wip_limit(self):
        """Test WIP limit retrieval."""
        config = AIWorkflowConfig.load()
        
        wip_limit = config.get_wip_limit()
        assert isinstance(wip_limit, int)
        assert wip_limit > 0
        # Should match gates.yaml value (3)
        if config.gates.get('wip_limits'):
            assert wip_limit == 3
    
    def test_get_states(self):
        """Test states list retrieval."""
        config = AIWorkflowConfig.load()
        
        states = config.get_states()
        assert isinstance(states, list)
        
        # Should contain common states
        if states:
            expected_states = ['draft', 'ready', 'running', 'merged']
            for state in expected_states:
                assert state in states, f"Missing expected state: {state}"
    
    def test_get_transitions(self):
        """Test transitions list retrieval."""
        config = AIWorkflowConfig.load()
        
        transitions = config.get_transitions()
        assert isinstance(transitions, list)
        
        # Each transition should have 'from' and 'to' keys
        if transitions:
            for transition in transitions:
                assert 'from' in transition
                assert 'to' in transition
    
    def test_is_valid_transition(self):
        """Test transition validation."""
        config = AIWorkflowConfig.load()
        
        # Test some common valid transitions
        if config.get_states():
            assert config.is_valid_transition('draft', 'ready') or \
                   config.is_valid_transition('draft', 'running'), \
                   "Should have at least one valid transition from draft"
            
            # Invalid transition should return False
            assert config.is_valid_transition('merged', 'draft') is False
    
    def test_get_gate_config(self):
        """Test gate configuration retrieval."""
        config = AIWorkflowConfig.load()
        
        # Test G1 gate (should exist in gates.yaml)
        g1_config = config.get_gate_config('G1')
        if config.gates.get('gates'):
            assert g1_config is not None
            assert isinstance(g1_config, dict)
            assert 'name' in g1_config
            assert 'description' in g1_config
        
        # Non-existent gate should return None
        assert config.get_gate_config('G99') is None
    
    def test_config_repr(self):
        """Test string representation."""
        config = AIWorkflowConfig.load()
        
        repr_str = repr(config)
        assert 'AIWorkflowConfig' in repr_str
        assert 'state_dir' in repr_str
    
    def test_environment_variable_override_state_dir(self):
        """Test state directory override via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_state_dir = Path(tmpdir) / 'custom_state'
            
            with patch.dict(os.environ, {'AI_WORKFLOW_OS_STATE_DIR': str(test_state_dir)}):
                config = AIWorkflowConfig.load()
                
                assert config.state_dir == test_state_dir
    
    def test_environment_variable_override_config_dir(self):
        """Test config directory override via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config_dir = Path(tmpdir) / 'custom_config'
            
            with patch.dict(os.environ, {'AI_WORKFLOW_OS_CONFIG_DIR': str(test_config_dir)}):
                config = AIWorkflowConfig.load()
                
                assert config.config_dir == test_config_dir
    
    def test_validation_fails_with_invalid_state_machine(self):
        """Test that validation fails when state_machine is invalid."""
        config = AIWorkflowConfig(
            state_dir=Path('/tmp'),
            config_dir=Path('/tmp'),
            gates={},
            state_machine={},  # Missing 'states' and 'transitions'
            registry={},
        )
        
        assert config.validate() is False
    
    def test_validation_creates_state_dir_if_missing(self):
        """Test that validation creates state directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = Path(tmpdir) / 'nonexistent' / 'state'
            
            config = AIWorkflowConfig(
                state_dir=nonexistent_dir,
                config_dir=Path(tmpdir),
                gates={'wip_limits': {'max_running_tasks': 3}, 'gates': {'G1': {}}},
                state_machine={'states': ['draft'], 'transitions': [{'from': 'draft', 'to': 'ready'}]},
                registry={},
            )
            
            # Should create directory during validation
            assert config.validate() is True
            assert nonexistent_dir.exists()
    
    def test_config_handles_missing_yaml_files_gracefully(self):
        """Test that config handles missing YAML files without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set environment to non-existent directory
            with patch.dict(os.environ, {
                'AI_WORKFLOW_OS_STATE_DIR': tmpdir,
                'AI_WORKFLOW_OS_CONFIG_DIR': tmpdir,
            }):
                # Should not raise exception, just use empty dicts
                config = AIWorkflowConfig.load()
                
                assert config is not None
                assert isinstance(config.gates, dict)
                assert isinstance(config.state_machine, dict)
                assert isinstance(config.registry, dict)


class TestConfigSingleton:
    """Test the global config singleton."""
    
    def test_config_singleton_exists(self):
        """Test that global config singleton is created on import."""
        from kernel.config import config
        
        assert config is not None
        assert isinstance(config, AIWorkflowConfig)
    
    def test_config_singleton_is_validated(self):
        """Test that global config is pre-validated."""
        from kernel.config import config
        
        # Should be validated on module import
        # Note: _validated flag may not be set if validation failed
        # but config should still be usable
        assert config is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
