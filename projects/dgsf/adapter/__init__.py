"""
DGSF Adapter Package

Provides integration between Legacy DGSF framework and AI Workflow OS.

Modules:
- dgsf_adapter: Main adapter class
- spec_mapper: Specification path resolution
- task_hooks: Task lifecycle hooks
- audit_bridge: Audit event bridging
- config_loader: Configuration utilities
- data_loader: Data loading and validation
"""

from .dgsf_adapter import DGSFAdapter, get_adapter
from .spec_mapper import SpecMapper
from .task_hooks import DGSFTaskHooks, get_hooks
from .audit_bridge import DGSFAuditBridge
from .config_loader import DGSFConfigLoader
from .data_loader import DGSFDataLoader, get_data_loader

__version__ = "1.1.0"
__all__ = [
    "DGSFAdapter",
    "get_adapter",
    "SpecMapper",
    "DGSFTaskHooks",
    "get_hooks",
    "DGSFAuditBridge",
    "DGSFConfigLoader",
    "DGSFDataLoader",
    "get_data_loader",
]
