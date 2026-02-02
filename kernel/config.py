"""
Configuration management for AI Workflow OS.

This module provides centralized configuration loading with environment
variable override support. It follows the singleton pattern for global
configuration access.

Usage:
    from kernel.config import config
    
    # Access configuration
    state_dir = config.state_dir
    max_wip = config.get_wip_limit()
    
    # Check if config is valid
    if not config.validate():
        raise RuntimeError("Configuration validation failed")

Environment Variables:
    AI_WORKFLOW_OS_STATE_DIR: Override default state directory
    AI_WORKFLOW_OS_CONFIG_DIR: Override default config directory
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kernel.paths import (
    STATE_DIR,
    CONFIGS_DIR,
    GATES_CONFIG_PATH,
    STATE_MACHINE_PATH,
    REGISTRY_PATH,
)


# =============================================================================
# Configuration Data Class
# =============================================================================

@dataclass
class AIWorkflowConfig:
    """
    Global configuration for AI Workflow OS.
    
    This class loads and validates all configuration files and provides
    typed access to configuration values.
    """
    
    # Directory paths (can be overridden by environment variables)
    state_dir: Path
    config_dir: Path
    
    # Loaded configuration data
    gates: Dict[str, Any] = field(default_factory=dict)
    state_machine: Dict[str, Any] = field(default_factory=dict)
    registry: Dict[str, Any] = field(default_factory=dict)
    
    # Validation status
    _validated: bool = False
    
    @classmethod
    def load(cls) -> 'AIWorkflowConfig':
        """
        Load configuration from files and environment variables.
        
        Returns:
            AIWorkflowConfig: Loaded configuration instance
            
        Raises:
            FileNotFoundError: If required configuration files are missing
            yaml.YAMLError: If configuration files are malformed
        """
        # Load paths from environment or defaults
        state_dir = Path(os.getenv('AI_WORKFLOW_OS_STATE_DIR', STATE_DIR))
        config_dir = Path(os.getenv('AI_WORKFLOW_OS_CONFIG_DIR', CONFIGS_DIR))
        
        # Determine config file paths
        gates_path = config_dir / 'gates.yaml'
        if not gates_path.exists():
            gates_path = GATES_CONFIG_PATH
            
        state_machine_path = STATE_MACHINE_PATH
        registry_path = REGISTRY_PATH
        
        # Load YAML files
        try:
            with open(gates_path, 'r', encoding='utf-8') as f:
                gates = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"⚠️ Warning: gates.yaml not found at {gates_path}", file=sys.stderr)
            gates = {}
            
        try:
            with open(state_machine_path, 'r', encoding='utf-8') as f:
                state_machine = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"⚠️ Warning: state_machine.yaml not found at {state_machine_path}", file=sys.stderr)
            state_machine = {}
            
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"⚠️ Warning: spec_registry.yaml not found at {registry_path}", file=sys.stderr)
            registry = {}
        
        return cls(
            state_dir=state_dir,
            config_dir=config_dir,
            gates=gates,
            state_machine=state_machine,
            registry=registry,
        )
    
    def validate(self) -> bool:
        """
        Validate loaded configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        errors = []
        
        # Validate state machine has required fields
        if not self.state_machine.get('states'):
            errors.append("state_machine.yaml missing 'states' field")
        if not self.state_machine.get('transitions'):
            errors.append("state_machine.yaml missing 'transitions' field")
            
        # Validate gates configuration
        if self.gates and not self.gates.get('gates'):
            errors.append("gates.yaml missing 'gates' field")
            
        # Check state directory exists or can be created
        if not self.state_dir.exists():
            try:
                self.state_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create state directory: {e}")
        
        if errors:
            for error in errors:
                print(f"❌ Configuration error: {error}", file=sys.stderr)
            return False
            
        self._validated = True
        return True
    
    def get_wip_limit(self) -> int:
        """
        Get WIP (Work-In-Progress) limit from gates configuration.
        
        Returns:
            int: Maximum number of concurrent running tasks (default: 3)
        """
        return self.gates.get('wip_limits', {}).get('max_running_tasks', 3)
    
    def get_states(self) -> List[str]:
        """
        Get list of valid task states.
        
        Returns:
            List[str]: List of state names
        """
        return self.state_machine.get('states', [])
    
    def get_transitions(self) -> List[Dict[str, str]]:
        """
        Get list of valid state transitions.
        
        Returns:
            List[Dict]: List of transition dictionaries with 'from' and 'to' keys
        """
        return self.state_machine.get('transitions', [])
    
    def is_valid_transition(self, from_state: str, to_state: str) -> bool:
        """
        Check if a state transition is valid.
        
        Args:
            from_state: Starting state
            to_state: Target state
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        transitions = self.get_transitions()
        return any(
            t.get('from') == from_state and t.get('to') == to_state
            for t in transitions
        )
    
    def get_gate_config(self, gate_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific gate.
        
        Args:
            gate_id: Gate identifier (e.g., 'G1', 'G2')
            
        Returns:
            Optional[Dict]: Gate configuration or None if not found
        """
        gates = self.gates.get('gates', {})
        return gates.get(gate_id)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"AIWorkflowConfig("
            f"state_dir={self.state_dir}, "
            f"gates={len(self.gates.get('gates', {}))}, "
            f"states={len(self.get_states())}, "
            f"transitions={len(self.get_transitions())})"
        )


# =============================================================================
# Global Configuration Singleton
# =============================================================================

# Load configuration on module import
config = AIWorkflowConfig.load()

# Validate configuration
if not config.validate():
    print("⚠️ Warning: Configuration validation failed. Some features may not work correctly.", file=sys.stderr)
