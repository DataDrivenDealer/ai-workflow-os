"""
DGSF Audit Bridge - Bridges DGSF events to AI Workflow OS audit system.

Provides integration between DGSF operations and the AI Workflow OS
audit logging infrastructure.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class DGSFAuditBridge:
    """
    Bridges DGSF audit events to AI Workflow OS audit system.
    
    Maps DGSF event types to AI Workflow OS audit categories and
    formats events according to the governance requirements.
    
    Event Categories:
    - pipeline: Pipeline execution events
    - gate: Governance gate check events
    - drift: Temporal drift detection events
    - baseline: Baseline ecosystem events
    - spec: Specification access events
    - data: Data operation events
    
    Attributes
    ----------
    audit_dir : Path
        Path to audit log directory
    session_id : str
        Current session identifier
    """
    
    # Event type to category mapping
    EVENT_CATEGORY_MAP = {
        "pipeline_start": "pipeline",
        "pipeline_complete": "pipeline",
        "pipeline_error": "pipeline",
        "gate_check": "gate",
        "gate_pass": "gate",
        "gate_fail": "gate",
        "drift_detected": "drift",
        "drift_resolved": "drift",
        "baseline_load": "baseline",
        "baseline_update": "baseline",
        "baseline_compare": "baseline",
        "spec_read": "spec",
        "spec_validate": "spec",
        "data_fetch": "data",
        "data_transform": "data",
        "data_validate": "data",
    }
    
    # Severity levels
    SEVERITY_MAP = {
        "pipeline_error": "ERROR",
        "gate_fail": "WARNING",
        "drift_detected": "WARNING",
        "baseline_update": "INFO",
        "default": "INFO",
    }
    
    def __init__(self, audit_dir: Optional[Path] = None):
        """
        Initialize audit bridge.
        
        Parameters
        ----------
        audit_dir : Path, optional
            Path to audit log directory. If not provided, uses default.
        """
        if audit_dir is None:
            # Default to AI Workflow OS audit directory
            self.audit_dir = Path(__file__).parent.parent.parent.parent / "ops" / "audit"
        else:
            self.audit_dir = Path(audit_dir)
        
        self.session_id = str(uuid.uuid4())[:8]
        self._event_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100  # Flush after this many events
        
        logger.info(f"Audit bridge initialized (session={self.session_id})")
    
    def log_event(self, event_type: str, data: Dict[str, Any], severity: str = None):
        """
        Log an audit event.
        
        Parameters
        ----------
        event_type : str
            Type of event
        data : dict
            Event data
        severity : str, optional
            Override severity level
        """
        category = self.EVENT_CATEGORY_MAP.get(event_type, "general")
        if severity is None:
            severity = self.SEVERITY_MAP.get(event_type, self.SEVERITY_MAP["default"])
        
        event = {
            "event_id": str(uuid.uuid4())[:12],
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            "category": category,
            "severity": severity,
            "source": "dgsf_adapter",
            "data": data,
        }
        
        self._event_buffer.append(event)
        
        # Log based on severity
        log_msg = f"[{category}] {event_type}: {data.get('message', str(data)[:100])}"
        if severity == "ERROR":
            logger.error(log_msg)
        elif severity == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # Flush if buffer is full
        if len(self._event_buffer) >= self._buffer_size:
            self.flush()
    
    def log_pipeline_event(
        self,
        pipeline_id: str,
        stage: str,
        status: str,
        metrics: Dict[str, Any] = None,
    ):
        """
        Log a pipeline execution event.
        
        Parameters
        ----------
        pipeline_id : str
            Pipeline identifier
        stage : str
            Pipeline stage (L0-L7)
        status : str
            Status ("start", "complete", "error")
        metrics : dict, optional
            Pipeline metrics
        """
        event_type = f"pipeline_{status}"
        data = {
            "pipeline_id": pipeline_id,
            "stage": stage,
            "status": status,
            "metrics": metrics or {},
            "message": f"Pipeline {pipeline_id} {stage} {status}",
        }
        self.log_event(event_type, data)
    
    def log_gate_event(
        self,
        gate_name: str,
        task_id: str,
        passed: bool,
        checks: Dict[str, bool],
        reason: str = None,
    ):
        """
        Log a governance gate event.
        
        Parameters
        ----------
        gate_name : str
            Gate name
        task_id : str
            Associated task ID
        passed : bool
            Whether gate passed
        checks : dict
            Individual check results
        reason : str, optional
            Failure reason if applicable
        """
        event_type = "gate_pass" if passed else "gate_fail"
        data = {
            "gate_name": gate_name,
            "task_id": task_id,
            "passed": passed,
            "checks": checks,
            "reason": reason,
            "message": f"Gate {gate_name} {'PASSED' if passed else 'FAILED'}",
        }
        self.log_event(event_type, data)
    
    def log_drift_event(
        self,
        drift_type: str,
        magnitude: float,
        threshold: float,
        affected_components: List[str],
    ):
        """
        Log a drift detection event.
        
        Parameters
        ----------
        drift_type : str
            Type of drift
        magnitude : float
            Drift magnitude
        threshold : float
            Drift threshold
        affected_components : list
            Components affected by drift
        """
        data = {
            "drift_type": drift_type,
            "magnitude": magnitude,
            "threshold": threshold,
            "threshold_exceeded": magnitude > threshold,
            "affected_components": affected_components,
            "message": f"Drift detected: {drift_type} ({magnitude:.3f} > {threshold:.3f})",
        }
        self.log_event("drift_detected", data)
    
    def log_baseline_event(
        self,
        baseline_id: str,
        action: str,
        version: str,
        metrics: Dict[str, Any] = None,
    ):
        """
        Log a baseline ecosystem event.
        
        Parameters
        ----------
        baseline_id : str
            Baseline identifier (A-H)
        action : str
            Action performed ("load", "update", "compare")
        version : str
            Baseline version
        metrics : dict, optional
            Baseline metrics
        """
        event_type = f"baseline_{action}"
        data = {
            "baseline_id": baseline_id,
            "action": action,
            "version": version,
            "metrics": metrics or {},
            "message": f"Baseline {baseline_id} {action} (v{version})",
        }
        self.log_event(event_type, data)
    
    def flush(self):
        """
        Flush event buffer to audit log file.
        """
        if not self._event_buffer:
            return
        
        # Ensure audit directory exists
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate audit file name
        date_str = datetime.now().strftime("%Y%m%d")
        audit_file = self.audit_dir / f"DGSF_{date_str}_{self.session_id}.json"
        
        # Load existing events if file exists
        existing = []
        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted audit file, starting fresh: {audit_file}")
        
        # Append new events
        existing.extend(self._event_buffer)
        
        # Write back
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Flushed {len(self._event_buffer)} events to {audit_file}")
        self._event_buffer.clear()
    
    def get_session_events(self) -> List[Dict[str, Any]]:
        """
        Get all events for current session (including buffered).
        
        Returns
        -------
        list
            List of event records
        """
        # Read from file if exists
        date_str = datetime.now().strftime("%Y%m%d")
        audit_file = self.audit_dir / f"DGSF_{date_str}_{self.session_id}.json"
        
        events = []
        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    events = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Add buffered events
        events.extend(self._event_buffer)
        
        return events
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary of session audit events.
        
        Returns
        -------
        dict
            Summary including counts by category and severity
        """
        events = self.get_session_events()
        
        by_category = {}
        by_severity = {}
        
        for event in events:
            cat = event.get("category", "unknown")
            sev = event.get("severity", "INFO")
            
            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {
            "session_id": self.session_id,
            "total_events": len(events),
            "by_category": by_category,
            "by_severity": by_severity,
            "buffered": len(self._event_buffer),
        }
    
    def __del__(self):
        """Flush remaining events on destruction."""
        try:
            self.flush()
        except Exception:
            pass
