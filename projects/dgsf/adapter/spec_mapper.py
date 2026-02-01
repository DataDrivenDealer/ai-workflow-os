"""
DGSF Spec Mapper - Specification path resolution and mapping utilities.

Maps between AI Workflow OS specification hierarchy and Legacy DGSF specs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import re


class SpecMapper:
    """
    Maps DGSF specifications to AI Workflow OS hierarchy.
    
    Specification ID Format: DGSF_{LAYER}_{VERSION}
    Examples:
    - DGSF_ARCH_V3 -> specs_v3/ARCHITECTURE.md
    - DGSF_PANELTREE_V3 -> specs_v3/PANELTREE_SPEC.md
    
    Attributes
    ----------
    legacy_root : Path
        Root path to Legacy DGSF
    specs_dir : Path
        Path to specs_v3 directory
    """
    
    # Mapping from spec ID components to file paths
    SPEC_FILE_MAP = {
        "ARCH": "ARCHITECTURE.md",
        "PANELTREE": "PANELTREE_SPEC.md",
        "SDF": "SDF_SPEC.md",
        "EA": "EA_SPEC.md",
        "ROLLING": "ROLLING_SPEC.md",
        "DATAENG": "DATAENGINEERING.md",
        "REPORT": "REPORTING_TELEMETRY.md",
        "TELEMETRY": "REPORTING_TELEMETRY.md",
    }
    
    # Layer mapping for AI Workflow OS integration
    LAYER_MAP = {
        "L0": ["DATAENG"],
        "L1": ["DATAENG"],
        "L2": ["PANELTREE"],
        "L3": ["SDF"],
        "L4": ["EA"],
        "L5": ["ROLLING"],
        "L6": ["REPORT"],
        "L7": ["TELEMETRY"],
    }
    
    # Concept alignment map
    CONCEPT_ALIGN = {
        "baseline": "baseline_ecosystem",  # A-H baselines
        "gate": "validation_gate",
        "drift": "temporal_drift",
        "factor": "sdf_factor",
        "tree": "panel_tree",
    }
    
    def __init__(self, legacy_root: Path):
        """
        Initialize spec mapper.
        
        Parameters
        ----------
        legacy_root : Path
            Root path to Legacy DGSF
        """
        self.legacy_root = Path(legacy_root)
        self.specs_dir = self.legacy_root / "specs_v3"
        
        if not self.specs_dir.exists():
            raise FileNotFoundError(f"Specs directory not found: {self.specs_dir}")
    
    def get_spec(self, spec_id: str) -> Dict[str, Any]:
        """
        Get specification by ID.
        
        Parameters
        ----------
        spec_id : str
            Specification ID (e.g., "DGSF_ARCH_V3")
        
        Returns
        -------
        dict
            Specification metadata including:
            - id: Specification ID
            - path: Path to spec file
            - layer: AI Workflow OS layer mapping
            - exists: Whether file exists
            - lines: Line count (if exists)
        """
        # Parse spec ID
        match = re.match(r"DGSF_(\w+)_V(\d+)", spec_id.upper())
        if not match:
            raise ValueError(f"Invalid spec ID format: {spec_id}. Expected: DGSF_{{COMPONENT}}_V{{VERSION}}")
        
        component = match.group(1)
        version = match.group(2)
        
        if component not in self.SPEC_FILE_MAP:
            raise ValueError(f"Unknown spec component: {component}. Available: {list(self.SPEC_FILE_MAP.keys())}")
        
        file_name = self.SPEC_FILE_MAP[component]
        spec_path = self.specs_dir / file_name
        
        # Determine layer mapping
        layer = None
        for l, components in self.LAYER_MAP.items():
            if component in components:
                layer = l
                break
        
        result = {
            "id": spec_id.upper(),
            "component": component,
            "version": f"v{version}",
            "file_name": file_name,
            "path": spec_path,
            "layer": layer,
            "exists": spec_path.exists(),
            "lines": 0,
        }
        
        if result["exists"]:
            with open(spec_path, "r", encoding="utf-8") as f:
                result["lines"] = len(f.readlines())
        
        return result
    
    def list_specs(self) -> List[str]:
        """
        List all available specification IDs.
        
        Returns
        -------
        list
            List of spec IDs
        """
        specs = []
        for component in self.SPEC_FILE_MAP:
            spec_path = self.specs_dir / self.SPEC_FILE_MAP[component]
            if spec_path.exists():
                specs.append(f"DGSF_{component}_V3")
        return specs
    
    def get_spec_content(self, spec_id: str) -> str:
        """
        Get full content of a specification.
        
        Parameters
        ----------
        spec_id : str
            Specification ID
        
        Returns
        -------
        str
            Specification content
        """
        spec = self.get_spec(spec_id)
        if not spec["exists"]:
            raise FileNotFoundError(f"Spec file not found: {spec['path']}")
        
        with open(spec["path"], "r", encoding="utf-8") as f:
            return f.read()
    
    def resolve_path(self, relative_path: str) -> Path:
        """
        Resolve relative path within Legacy DGSF.
        
        Parameters
        ----------
        relative_path : str
            Path relative to legacy root
        
        Returns
        -------
        Path
            Absolute path
        """
        return self.legacy_root / relative_path
    
    def get_layer_specs(self, layer: str) -> List[Dict[str, Any]]:
        """
        Get all specs for a given layer.
        
        Parameters
        ----------
        layer : str
            Layer identifier (L0-L7)
        
        Returns
        -------
        list
            List of spec metadata dicts
        """
        if layer not in self.LAYER_MAP:
            raise ValueError(f"Unknown layer: {layer}. Available: {list(self.LAYER_MAP.keys())}")
        
        specs = []
        for component in self.LAYER_MAP[layer]:
            try:
                spec = self.get_spec(f"DGSF_{component}_V3")
                specs.append(spec)
            except (ValueError, FileNotFoundError):
                pass
        return specs
    
    def align_concept(self, ai_os_concept: str) -> Optional[str]:
        """
        Align AI Workflow OS concept to DGSF concept.
        
        Parameters
        ----------
        ai_os_concept : str
            AI Workflow OS concept name
        
        Returns
        -------
        str or None
            DGSF concept name, or None if no mapping
        """
        return self.CONCEPT_ALIGN.get(ai_os_concept.lower())
    
    def get_module_path(self, module_name: str) -> Path:
        """
        Get path to DGSF Python module.
        
        Parameters
        ----------
        module_name : str
            Module name (e.g., "paneltree", "sdf")
        
        Returns
        -------
        Path
            Path to module directory
        """
        module_path = self.legacy_root / "src" / "dgsf" / module_name
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        return module_path
    
    def get_test_path(self, module_name: str) -> Path:
        """
        Get path to module tests.
        
        Parameters
        ----------
        module_name : str
            Module name
        
        Returns
        -------
        Path
            Path to test directory
        """
        test_path = self.legacy_root / "tests" / module_name
        if not test_path.exists():
            raise FileNotFoundError(f"Tests not found: {test_path}")
        return test_path
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary of available specifications.
        
        Returns
        -------
        dict
            Summary including counts and layer distribution
        """
        specs = self.list_specs()
        total_lines = 0
        layer_counts = {f"L{i}": 0 for i in range(8)}
        
        for spec_id in specs:
            spec = self.get_spec(spec_id)
            total_lines += spec.get("lines", 0)
            if spec["layer"]:
                layer_counts[spec["layer"]] += 1
        
        return {
            "total_specs": len(specs),
            "total_lines": total_lines,
            "layer_distribution": layer_counts,
            "specs": specs,
        }
