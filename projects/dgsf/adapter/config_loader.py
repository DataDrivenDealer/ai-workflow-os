"""
DGSF Config Loader - Configuration loading utilities for DGSF integration.

Provides unified access to DGSF configuration files with validation
and AI Workflow OS integration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


class DGSFConfigLoader:
    """
    Configuration loader for DGSF framework.
    
    Loads and validates DGSF configuration files, providing a unified
    interface for accessing configuration across all DGSF modules.
    
    Configuration Locations:
    - Main configs: {legacy_root}/configs/
    - Module configs: {legacy_root}/src/dgsf/{module}/configs/
    - Run configs: {legacy_root}/runs/{run_id}/config.yaml
    
    Attributes
    ----------
    legacy_root : Path
        Root path to Legacy DGSF
    configs_dir : Path
        Path to main configs directory
    cache : dict
        Loaded configuration cache
    """
    
    # Known configuration files
    KNOWN_CONFIGS = {
        "master": "master.yaml",
        "dataeng": "dataeng.yaml",
        "paneltree": "paneltree.yaml",
        "sdf": "sdf.yaml",
        "ea": "ea.yaml",
        "rolling": "rolling.yaml",
        "backtest": "backtest.yaml",
        "logging": "logging.yaml",
    }
    
    # Required fields for validation
    REQUIRED_FIELDS = {
        "master": ["version", "project_name"],
        "dataeng": ["data_source", "date_range"],
        "paneltree": ["max_depth", "min_samples"],
        "sdf": ["factors", "estimation_method"],
        "ea": ["population_size", "generations"],
        "rolling": ["window_size", "step_size"],
    }
    
    def __init__(self, legacy_root: Path, strict: bool = False):
        """
        Initialize config loader.
        
        Parameters
        ----------
        legacy_root : Path
            Root path to Legacy DGSF
        strict : bool
            If True, raise error when configs_dir not found. Default False.
        """
        self.legacy_root = Path(legacy_root)
        self.configs_dir = self.legacy_root / "configs"
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.strict = strict
        self._configs_available = self.configs_dir.exists()
        
        if yaml is None:
            logger.warning("PyYAML not installed, YAML loading disabled")
        
        if not self._configs_available and strict:
            raise FileNotFoundError(f"Configs directory not found: {self.configs_dir}")
    
    @property
    def is_available(self) -> bool:
        """Check if configs directory is available."""
        return self._configs_available
    
    def load(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Parameters
        ----------
        config_name : str
            Configuration name (without .yaml extension)
        use_cache : bool
            Whether to use cached config if available
        
        Returns
        -------
        dict
            Parsed configuration
        """
        if yaml is None:
            raise ImportError("PyYAML is required for loading configurations")
        
        # Check cache
        if use_cache and config_name in self.cache:
            return self.cache[config_name]
        
        # Resolve config path
        config_path = self._resolve_config_path(config_name)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        
        # Load YAML
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        # Validate if known config
        if config_name in self.REQUIRED_FIELDS:
            self._validate_config(config_name, config)
        
        # Cache and return
        self.cache[config_name] = config
        logger.debug(f"Loaded config: {config_name} from {config_path}")
        
        return config
    
    def _resolve_config_path(self, config_name: str) -> Path:
        """
        Resolve configuration file path.
        
        Parameters
        ----------
        config_name : str
            Configuration name
        
        Returns
        -------
        Path
            Path to config file
        """
        # Check known configs
        if config_name in self.KNOWN_CONFIGS:
            return self.configs_dir / self.KNOWN_CONFIGS[config_name]
        
        # Check if it's a full filename
        if config_name.endswith(".yaml") or config_name.endswith(".yml"):
            return self.configs_dir / config_name
        
        # Try with .yaml extension
        yaml_path = self.configs_dir / f"{config_name}.yaml"
        if yaml_path.exists():
            return yaml_path
        
        # Try with .yml extension
        yml_path = self.configs_dir / f"{config_name}.yml"
        if yml_path.exists():
            return yml_path
        
        # Default to .yaml
        return yaml_path
    
    def _validate_config(self, config_name: str, config: Dict[str, Any]):
        """
        Validate configuration against required fields.
        
        Parameters
        ----------
        config_name : str
            Configuration name
        config : dict
            Configuration to validate
        
        Raises
        ------
        ValueError
            If required fields are missing
        """
        required = self.REQUIRED_FIELDS.get(config_name, [])
        missing = [f for f in required if f not in config]
        
        if missing:
            raise ValueError(f"Config '{config_name}' missing required fields: {missing}")
    
    def list_configs(self) -> List[str]:
        """
        List all available configuration files.
        
        Returns
        -------
        list
            List of configuration names
        """
        configs = []
        
        if self.configs_dir.exists():
            for path in self.configs_dir.glob("*.yaml"):
                configs.append(path.stem)
            for path in self.configs_dir.glob("*.yml"):
                if path.stem not in configs:
                    configs.append(path.stem)
        
        return sorted(configs)
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific module.
        
        Parameters
        ----------
        module_name : str
            Module name (e.g., "paneltree", "sdf")
        
        Returns
        -------
        dict
            Module configuration
        """
        # Try module-specific config first
        try:
            return self.load(module_name)
        except FileNotFoundError:
            pass
        
        # Fall back to master config section
        try:
            master = self.load("master")
            if module_name in master:
                return master[module_name]
        except FileNotFoundError:
            pass
        
        # Return empty config
        logger.warning(f"No configuration found for module: {module_name}")
        return {}
    
    def get_run_config(self, run_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific run.
        
        Parameters
        ----------
        run_id : str
            Run identifier
        
        Returns
        -------
        dict
            Run configuration
        """
        run_path = self.legacy_root / "runs" / run_id / "config.yaml"
        
        if not run_path.exists():
            raise FileNotFoundError(f"Run config not found: {run_path}")
        
        with open(run_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def list_runs(self) -> List[str]:
        """
        List all available runs.
        
        Returns
        -------
        list
            List of run IDs
        """
        runs_dir = self.legacy_root / "runs"
        if not runs_dir.exists():
            return []
        
        runs = []
        for path in runs_dir.iterdir():
            if path.is_dir() and (path / "config.yaml").exists():
                runs.append(path.name)
        
        return sorted(runs)
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configurations.
        
        Parameters
        ----------
        *config_names : str
            Configuration names to merge (later configs override earlier)
        
        Returns
        -------
        dict
            Merged configuration
        """
        merged = {}
        for name in config_names:
            try:
                config = self.load(name)
                merged = self._deep_merge(merged, config)
            except FileNotFoundError:
                logger.warning(f"Config not found for merge: {name}")
        
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Parameters
        ----------
        base : dict
            Base dictionary
        override : dict
            Override dictionary
        
        Returns
        -------
        dict
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def clear_cache(self):
        """Clear configuration cache."""
        self.cache.clear()
        logger.debug("Configuration cache cleared")
    
    def save_config(self, config_name: str, config: Dict[str, Any]):
        """
        Save configuration to file.
        
        Parameters
        ----------
        config_name : str
            Configuration name
        config : dict
            Configuration to save
        """
        if yaml is None:
            raise ImportError("PyYAML is required for saving configurations")
        
        config_path = self._resolve_config_path(config_name)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # Update cache
        self.cache[config_name] = config
        logger.info(f"Saved config: {config_name} to {config_path}")
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary of available configurations.
        
        Returns
        -------
        dict
            Summary including counts and cached configs
        """
        configs = self.list_configs()
        runs = self.list_runs()
        
        return {
            "configs_dir": str(self.configs_dir),
            "total_configs": len(configs),
            "configs": configs,
            "total_runs": len(runs),
            "runs": runs,
            "cached": list(self.cache.keys()),
        }
