"""
DGSF Task Hooks - Task lifecycle hooks for DGSF integration.

Provides hooks into AI Workflow OS task lifecycle for DGSF-specific operations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DGSFTaskHooks:
    """
    Task lifecycle hooks for DGSF integration.
    
    Provides pre/post hooks for DGSF task operations to integrate
    with AI Workflow OS task state machine.
    
    Hook Points:
    - pre_task_start: Before task begins execution
    - post_task_complete: After task completes successfully
    - pre_gate_check: Before governance gate check
    - post_gate_pass: After gate passes
    - on_drift_detected: When temporal drift is detected
    
    Attributes
    ----------
    hooks : dict
        Registered hook callbacks by event type
    """
    
    # Task type to DGSF module mapping
    TASK_MODULE_MAP = {
        "data_fetch": "dataeng",
        "data_engineer": "dataeng",
        "panel_construct": "paneltree",
        "tree_build": "paneltree",
        "sdf_train": "sdf",
        "sdf_estimate": "sdf",
        "ea_optimize": "ea",
        "evolution": "ea",
        "rolling_eval": "rolling",
        "backtest": "rolling",
    }
    
    def __init__(self):
        """Initialize task hooks."""
        self.hooks: Dict[str, List[Callable]] = {
            "pre_task_start": [],
            "post_task_complete": [],
            "pre_gate_check": [],
            "post_gate_pass": [],
            "on_drift_detected": [],
            "on_baseline_update": [],
            "on_spec_access": [],
        }
        self._context: Dict[str, Any] = {}
    
    def register(self, event: str, callback: Callable):
        """
        Register a hook callback.
        
        Parameters
        ----------
        event : str
            Event type to hook
        callback : callable
            Function to call on event
        """
        if event not in self.hooks:
            raise ValueError(f"Unknown event: {event}. Available: {list(self.hooks.keys())}")
        self.hooks[event].append(callback)
        logger.debug(f"Registered hook for {event}: {callback.__name__}")
    
    def unregister(self, event: str, callback: Callable):
        """
        Unregister a hook callback.
        
        Parameters
        ----------
        event : str
            Event type
        callback : callable
            Function to unregister
        """
        if event in self.hooks and callback in self.hooks[event]:
            self.hooks[event].remove(callback)
            logger.debug(f"Unregistered hook for {event}: {callback.__name__}")
    
    def trigger(self, event: str, data: Dict[str, Any] = None) -> List[Any]:
        """
        Trigger hooks for an event.
        
        Parameters
        ----------
        event : str
            Event type
        data : dict, optional
            Event data to pass to hooks
        
        Returns
        -------
        list
            Results from each hook callback
        """
        if event not in self.hooks:
            logger.warning(f"Unknown event triggered: {event}")
            return []
        
        data = data or {}
        data["timestamp"] = datetime.now().isoformat()
        data["event"] = event
        
        results = []
        for callback in self.hooks[event]:
            try:
                result = callback(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook error in {callback.__name__}: {e}")
                results.append(None)
        
        return results
    
    def pre_task_start(self, task_id: str, task_type: str, config: Dict[str, Any]):
        """
        Hook called before task starts.
        
        Parameters
        ----------
        task_id : str
            Task identifier
        task_type : str
            Type of task
        config : dict
            Task configuration
        """
        # Store context for later hooks
        self._context[task_id] = {
            "task_type": task_type,
            "config": config,
            "start_time": datetime.now(),
            "module": self.TASK_MODULE_MAP.get(task_type),
        }
        
        data = {
            "task_id": task_id,
            "task_type": task_type,
            "dgsf_module": self._context[task_id]["module"],
            "config": config,
        }
        
        self.trigger("pre_task_start", data)
        logger.info(f"Task {task_id} starting (type={task_type}, module={data['dgsf_module']})")
    
    def post_task_complete(self, task_id: str, result: Dict[str, Any]):
        """
        Hook called after task completes.
        
        Parameters
        ----------
        task_id : str
            Task identifier
        result : dict
            Task result data
        """
        ctx = self._context.pop(task_id, {})
        duration = None
        if "start_time" in ctx:
            duration = (datetime.now() - ctx["start_time"]).total_seconds()
        
        data = {
            "task_id": task_id,
            "task_type": ctx.get("task_type"),
            "dgsf_module": ctx.get("module"),
            "result": result,
            "duration_seconds": duration,
        }
        
        self.trigger("post_task_complete", data)
        logger.info(f"Task {task_id} completed (duration={duration:.2f}s)")
    
    def pre_gate_check(self, gate_name: str, task_id: str, metrics: Dict[str, Any]):
        """
        Hook called before governance gate check.
        
        Parameters
        ----------
        gate_name : str
            Name of gate being checked
        task_id : str
            Associated task ID
        metrics : dict
            Metrics to validate
        """
        data = {
            "gate_name": gate_name,
            "task_id": task_id,
            "metrics": metrics,
        }
        
        self.trigger("pre_gate_check", data)
        logger.info(f"Gate check {gate_name} for task {task_id}")
    
    def post_gate_pass(self, gate_name: str, task_id: str, result: Dict[str, Any]):
        """
        Hook called after gate passes.
        
        Parameters
        ----------
        gate_name : str
            Name of gate that passed
        task_id : str
            Associated task ID
        result : dict
            Gate check result
        """
        data = {
            "gate_name": gate_name,
            "task_id": task_id,
            "result": result,
            "passed": True,
        }
        
        self.trigger("post_gate_pass", data)
        logger.info(f"Gate {gate_name} passed for task {task_id}")
    
    def on_drift_detected(self, drift_type: str, magnitude: float, context: Dict[str, Any]):
        """
        Hook called when temporal drift is detected.
        
        Parameters
        ----------
        drift_type : str
            Type of drift ("concept", "data", "model")
        magnitude : float
            Drift magnitude (0-1 scale)
        context : dict
            Additional context
        """
        data = {
            "drift_type": drift_type,
            "magnitude": magnitude,
            "context": context,
            "threshold_exceeded": magnitude > 0.1,  # Default threshold
        }
        
        self.trigger("on_drift_detected", data)
        logger.warning(f"Drift detected: {drift_type} (magnitude={magnitude:.3f})")
    
    def on_baseline_update(self, baseline_id: str, old_version: str, new_version: str):
        """
        Hook called when a baseline is updated.
        
        Parameters
        ----------
        baseline_id : str
            Baseline identifier (A-H)
        old_version : str
            Previous version
        new_version : str
            New version
        """
        data = {
            "baseline_id": baseline_id,
            "old_version": old_version,
            "new_version": new_version,
        }
        
        self.trigger("on_baseline_update", data)
        logger.info(f"Baseline {baseline_id} updated: {old_version} -> {new_version}")
    
    def on_spec_access(self, spec_id: str, accessor: str, purpose: str):
        """
        Hook called when a specification is accessed.
        
        Parameters
        ----------
        spec_id : str
            Specification ID
        accessor : str
            Who/what is accessing
        purpose : str
            Purpose of access
        """
        data = {
            "spec_id": spec_id,
            "accessor": accessor,
            "purpose": purpose,
        }
        
        self.trigger("on_spec_access", data)
        logger.debug(f"Spec {spec_id} accessed by {accessor} for {purpose}")
    
    def get_dgsf_module(self, task_type: str) -> Optional[str]:
        """
        Get DGSF module for a task type.
        
        Parameters
        ----------
        task_type : str
            Task type identifier
        
        Returns
        -------
        str or None
            DGSF module name, or None if no mapping
        """
        return self.TASK_MODULE_MAP.get(task_type)


# Global hooks instance
_hooks_instance: Optional[DGSFTaskHooks] = None


def get_hooks() -> DGSFTaskHooks:
    """
    Get global task hooks instance.
    
    Returns
    -------
    DGSFTaskHooks
        Hooks instance
    """
    global _hooks_instance
    if _hooks_instance is None:
        _hooks_instance = DGSFTaskHooks()
    return _hooks_instance
