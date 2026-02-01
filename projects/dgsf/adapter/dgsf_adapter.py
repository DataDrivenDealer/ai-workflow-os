"""
DGSF Adapter - Main adapter class for DGSF ↔ AI Workflow OS integration.

This module provides the primary interface for accessing DGSF functionality
within the AI Workflow OS governance framework.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .spec_mapper import SpecMapper
from .config_loader import DGSFConfigLoader
from .audit_bridge import DGSFAuditBridge


class DGSFAdapter:
    """
    Main adapter class for DGSF framework integration.
    
    Provides a unified interface for:
    - Accessing DGSF specifications
    - Loading DGSF modules
    - Running DGSF pipelines
    - Bridging audit events
    
    Attributes
    ----------
    legacy_root : Path
        Root path to Legacy DGSF installation
    spec_mapper : SpecMapper
        Specification mapping utility
    config_loader : DGSFConfigLoader
        Configuration loading utility
    audit_bridge : DGSFAuditBridge
        Audit event bridging utility
    """
    
    def __init__(self, legacy_root: Optional[Path] = None):
        """
        Initialize DGSF Adapter.
        
        Parameters
        ----------
        legacy_root : Path, optional
            Root path to Legacy DGSF. If not provided, uses default location.
        """
        if legacy_root is None:
            # Default to relative path from AI Workflow OS root
            self.legacy_root = Path(__file__).parent.parent / "legacy" / "DGSF"
        else:
            self.legacy_root = Path(legacy_root)
        
        # Validate legacy root exists
        if not self.legacy_root.exists():
            raise FileNotFoundError(f"Legacy DGSF not found at: {self.legacy_root}")
        
        # Initialize components
        self.spec_mapper = SpecMapper(self.legacy_root)
        self.config_loader = DGSFConfigLoader(self.legacy_root)
        self.audit_bridge = DGSFAuditBridge()
        
        # Add legacy src to Python path if not already present
        legacy_src = self.legacy_root / "src"
        if str(legacy_src) not in sys.path:
            sys.path.insert(0, str(legacy_src))
    
    def get_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Get DGSF specification by ID.
        
        Parameters
        ----------
        spec_id : str
            Specification identifier (e.g., "DGSF_ARCH_V3", "DGSF_PANELTREE_V3")
        
        Returns
        -------
        dict
            Specification metadata and content path
        """
        return self.spec_mapper.get_spec(spec_id)
    
    def list_specs(self) -> List[str]:
        """
        List all available DGSF specifications.
        
        Returns
        -------
        list
            List of specification IDs
        """
        return self.spec_mapper.list_specs()
    
    def get_module(self, module_name: str) -> Any:
        """
        Get DGSF module by name.
        
        Parameters
        ----------
        module_name : str
            Module name (e.g., "paneltree", "sdf", "ea", "rolling")
        
        Returns
        -------
        module
            Imported Python module
        """
        module_map = {
            "paneltree": "dgsf.paneltree",
            "sdf": "dgsf.sdf",
            "ea": "dgsf.ea",
            "rolling": "dgsf.rolling",
            "backtest": "dgsf.backtest",
            "dataeng": "dgsf.dataeng",
            "config": "dgsf.config",
            "utils": "dgsf.utils",
        }
        
        if module_name not in module_map:
            raise ValueError(f"Unknown module: {module_name}. Available: {list(module_map.keys())}")
        
        import importlib
        return importlib.import_module(module_map[module_name])
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load DGSF configuration file.
        
        Parameters
        ----------
        config_name : str
            Configuration file name (without .yaml extension)
        
        Returns
        -------
        dict
            Parsed configuration
        """
        return self.config_loader.load(config_name)
    
    def list_configs(self) -> List[str]:
        """
        List all available DGSF configurations.
        
        Returns
        -------
        list
            List of configuration names
        """
        return self.config_loader.list_configs()
    
    def get_data_path(self, dataset: str = "full") -> Path:
        """
        Get path to DGSF data directory.
        
        Parameters
        ----------
        dataset : str
            Dataset name ("a0", "full", "final", etc.)
        
        Returns
        -------
        Path
            Path to data directory
        """
        data_path = self.legacy_root / "data" / dataset
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        return data_path
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log audit event through bridge.
        
        Parameters
        ----------
        event_type : str
            Type of event (e.g., "pipeline_start", "gate_check", "drift_detected")
        data : dict
            Event data
        """
        self.audit_bridge.log_event(event_type, data)
    
    def get_version(self) -> str:
        """
        Get DGSF framework version.
        
        Returns
        -------
        str
            Version string
        """
        try:
            import dgsf
            return dgsf.__version__
        except (ImportError, AttributeError):
            return "0.1.0"  # Default version
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on DGSF integration.
        
        Returns
        -------
        dict
            Health check results
        """
        results = {
            "legacy_root_exists": self.legacy_root.exists(),
            "src_importable": False,
            "configs_accessible": False,
            "data_accessible": False,
            "specs_accessible": False,
        }
        
        # Check src importable
        try:
            import dgsf
            results["src_importable"] = True
        except ImportError:
            pass
        
        # Check configs
        try:
            configs = self.list_configs()
            results["configs_accessible"] = len(configs) > 0
        except Exception:
            pass
        
        # Check data
        try:
            self.get_data_path("a0")
            results["data_accessible"] = True
        except Exception:
            pass
        
        # Check specs
        try:
            specs = self.list_specs()
            results["specs_accessible"] = len(specs) > 0
        except Exception:
            pass
        
        return results


# Singleton instance for convenience
_adapter_instance: Optional[DGSFAdapter] = None


def get_adapter() -> DGSFAdapter:
    """
    Get singleton DGSF adapter instance.
    
    Returns
    -------
    DGSFAdapter
        Adapter instance
    """
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = DGSFAdapter()
    return _adapter_instance
