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


# =============================================================================
# Project Adapter Loading (AEP-1)
# =============================================================================

# Adapter cache with thread-safe loading
_adapter_cache: Dict[str, Dict[str, Any]] = {}
_adapter_versions: Dict[str, int] = {}
_adapter_load_lock = None  # Lazy initialization to avoid import-time issues


def _get_adapter_lock():
    """Get or create the adapter loading lock (thread-safe)."""
    global _adapter_load_lock
    if _adapter_load_lock is None:
        import threading
        _adapter_load_lock = threading.RLock()
    return _adapter_load_lock


def load_project_adapter(
    project_id: str = "dgsf",
    projects_root: Optional[Path] = None,
    force_reload: bool = False,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load a project's adapter configuration with atomic loading and caching.
    
    The adapter binds a project to the Kernel interface, providing:
    - identity: Project metadata
    - thresholds: Success criteria
    - paths: Directory mappings
    - skills: Skill prefix and prompt file mappings
    - rules: Rule inheritance and overrides
    - behavior: Behavioral configuration
    
    Thread-safe with atomic loading to support multi-agent concurrency.
    
    Args:
        project_id: Project identifier (e.g., "dgsf")
        projects_root: Root directory for projects (default: auto-detect)
        force_reload: If True, bypass cache and reload from disk
        validate: If True, validate adapter against interface contract
        
    Returns:
        Dict[str, Any]: Loaded adapter configuration (immutable copy)
        
    Raises:
        FileNotFoundError: If adapter.yaml does not exist
        yaml.YAMLError: If adapter.yaml is malformed
        ValueError: If adapter fails validation
        
    Example:
        >>> adapter = load_project_adapter("dgsf")
        >>> print(adapter["identity"]["project_id"])
        'dgsf'
        >>> print(adapter["skills"]["prefix"])
        'dgsf'
    """
    import copy
    
    if projects_root is None:
        # Auto-detect from current file location
        projects_root = Path(__file__).parent.parent / "projects"
    
    adapter_path = projects_root / project_id / "adapter.yaml"
    cache_key = str(adapter_path)
    
    # Check cache first (fast path without lock)
    if not force_reload and cache_key in _adapter_cache:
        # Verify file hasn't changed (check mtime)
        try:
            current_mtime = int(adapter_path.stat().st_mtime * 1000)
            if _adapter_versions.get(cache_key) == current_mtime:
                return copy.deepcopy(_adapter_cache[cache_key])
        except Exception:
            pass  # Fall through to reload
    
    # Acquire lock for loading (atomic load)
    lock = _get_adapter_lock()
    with lock:
        # Double-check after acquiring lock
        if not force_reload and cache_key in _adapter_cache:
            try:
                current_mtime = int(adapter_path.stat().st_mtime * 1000)
                if _adapter_versions.get(cache_key) == current_mtime:
                    return copy.deepcopy(_adapter_cache[cache_key])
            except Exception:
                pass
        
        # Load from disk
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Project adapter not found: {adapter_path}\n"
                f"Ensure projects/{project_id}/adapter.yaml exists and implements "
                f"configs/project_interface.yaml"
            )
        
        with open(adapter_path, 'r', encoding='utf-8') as f:
            adapter = yaml.safe_load(f) or {}
        
        # Validate minimal required fields
        if not adapter.get("identity", {}).get("project_id"):
            raise ValueError(f"Adapter {adapter_path} missing identity.project_id")
        
        if validate:
            _validate_adapter(adapter, project_id)
        
        # Cache with version
        _adapter_cache[cache_key] = adapter
        try:
            _adapter_versions[cache_key] = int(adapter_path.stat().st_mtime * 1000)
        except Exception:
            _adapter_versions[cache_key] = 0
        
        return copy.deepcopy(adapter)


def _validate_adapter(adapter: Dict[str, Any], project_id: str) -> None:
    """
    Validate adapter against project interface contract.
    
    Args:
        adapter: Loaded adapter configuration
        project_id: Project identifier
        
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    # Required top-level sections
    required_sections = ["identity", "thresholds", "paths", "skills", "behavior"]
    for section in required_sections:
        if section not in adapter:
            errors.append(f"Missing required section: {section}")
    
    # Required identity fields
    identity = adapter.get("identity", {})
    if not identity.get("project_id"):
        errors.append("Missing identity.project_id")
    if not identity.get("project_name"):
        errors.append("Missing identity.project_name")
    if not identity.get("kernel_version"):
        errors.append("Missing identity.kernel_version")
    
    # Required path fields
    paths = adapter.get("paths", {})
    required_paths = ["source", "tests", "experiments", "data_safe", "data_protected"]
    for path_key in required_paths:
        if path_key not in paths:
            errors.append(f"Missing required path: paths.{path_key}")
    
    # Required skill fields
    skills = adapter.get("skills", {})
    if not skills.get("prefix"):
        errors.append("Missing skills.prefix")
    
    # Required behavior fields
    behavior = adapter.get("behavior", {})
    if not behavior.get("scope_pattern"):
        errors.append("Missing behavior.scope_pattern")
    
    if errors:
        raise ValueError(
            f"Adapter validation failed for {project_id}:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


def invalidate_adapter_cache(project_id: Optional[str] = None) -> None:
    """
    Invalidate adapter cache to force reload on next access.
    
    Args:
        project_id: If provided, only invalidate this project. Otherwise, clear all.
    """
    lock = _get_adapter_lock()
    with lock:
        if project_id:
            projects_root = Path(__file__).parent.parent / "projects"
            cache_key = str(projects_root / project_id / "adapter.yaml")
            _adapter_cache.pop(cache_key, None)
            _adapter_versions.pop(cache_key, None)
        else:
            _adapter_cache.clear()
            _adapter_versions.clear()


def get_active_project_id() -> str:
    """
    Get the currently active project ID.
    
    Currently returns the default project. In future, this could be
    determined from environment variables or workspace context.
    
    Returns:
        str: Active project identifier
    """
    return os.getenv("AI_WORKFLOW_OS_PROJECT", "dgsf")


def get_project_path(project_id: str, path_key: str) -> Path:
    """
    Resolve a path from a project's adapter.
    
    Args:
        project_id: Project identifier
        path_key: Key in adapter.paths (e.g., "source", "tests")
        
    Returns:
        Path: Resolved absolute path
        
    Example:
        >>> source_path = get_project_path("dgsf", "source")
        >>> print(source_path)
        PosixPath('projects/dgsf/repo/src/dgsf')
    """
    adapter = load_project_adapter(project_id)
    paths = adapter.get("paths", {})
    
    if path_key not in paths:
        raise KeyError(f"Path key '{path_key}' not found in adapter.paths for {project_id}")
    
    relative_path = paths[path_key]
    projects_root = Path(__file__).parent.parent / "projects"
    
    return projects_root / project_id / relative_path

