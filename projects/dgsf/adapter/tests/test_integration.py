"""
Integration tests for DGSF Adapter.

Test Strategy (Grady Booch):
    "Integration tests validate the collaboration contracts between components,
    not implementation details. These tests define the expected interface
    behavior and serve as executable specifications."

Test Coverage:
    1. Adapter initialization and health check
    2. Module access and import paths
    3. Audit logging bridge functionality
    4. Configuration and spec loading
    5. [FUTURE] run_experiment() end-to-end workflow (currently unimplemented)

Dependencies:
    - DGSFAdapter must be importable
    - DGSF repo submodule must be present (though may not be fully functional)
    - pytest >= 7.0
"""

import pytest
from pathlib import Path

# conftest.py handles path setup
from adapter.dgsf_adapter import DGSFAdapter, get_adapter


class TestDGSFAdapterInitialization:
    """Test adapter initialization and basic setup."""
    
    def test_adapter_can_be_instantiated(self):
        """Adapter should initialize without raising exceptions."""
        adapter = DGSFAdapter()
        assert adapter is not None
        assert isinstance(adapter, DGSFAdapter)
    
    def test_adapter_has_legacy_root_attribute(self):
        """Adapter should track legacy root path."""
        adapter = DGSFAdapter()
        assert hasattr(adapter, 'legacy_root')
        assert isinstance(adapter.legacy_root, Path)
    
    def test_adapter_checks_availability(self):
        """Adapter should report whether legacy DGSF is available."""
        adapter = DGSFAdapter()
        assert hasattr(adapter, 'is_available')
        # Result depends on environment, just check it's a boolean
        assert isinstance(adapter.is_available, bool)
    
    def test_singleton_adapter_returns_same_instance(self):
        """get_adapter() should return singleton instance."""
        adapter1 = get_adapter()
        adapter2 = get_adapter()
        assert adapter1 is adapter2


class TestDGSFAdapterHealthCheck:
    """Test adapter health check functionality."""
    
    def test_health_check_returns_dict(self):
        """health_check() should return dict with status keys."""
        adapter = DGSFAdapter()
        health = adapter.health_check()
        
        assert isinstance(health, dict)
        # Check expected keys exist
        assert 'legacy_root_exists' in health
        assert 'src_importable' in health
        assert 'configs_accessible' in health
        assert 'data_accessible' in health
        assert 'specs_accessible' in health
    
    def test_health_check_values_are_boolean(self):
        """All health check values should be boolean."""
        adapter = DGSFAdapter()
        health = adapter.health_check()
        
        for key, value in health.items():
            assert isinstance(value, bool), f"{key} should be boolean, got {type(value)}"
    
    def test_health_check_legacy_root_consistent(self):
        """legacy_root_exists should match is_available property."""
        adapter = DGSFAdapter()
        health = adapter.health_check()
        
        assert health['legacy_root_exists'] == adapter.is_available


class TestDGSFAdapterModuleAccess:
    """Test adapter's ability to access DGSF modules."""
    
    def test_get_module_raises_on_unknown_module(self):
        """get_module() should raise ValueError for unknown modules."""
        adapter = DGSFAdapter()
        
        with pytest.raises(ValueError, match="Unknown module"):
            adapter.get_module("nonexistent_module")
    
    def test_get_version_returns_string(self):
        """get_version() should always return a version string."""
        adapter = DGSFAdapter()
        version = adapter.get_version()
        
        assert isinstance(version, str)
        assert len(version) > 0
        # Should match semantic versioning pattern loosely
        assert '.' in version


class TestDGSFAdapterAuditBridge:
    """Test audit logging bridge functionality."""
    
    def test_log_event_does_not_raise(self):
        """log_event() should accept events without raising exceptions."""
        adapter = DGSFAdapter()
        
        # Should not raise
        adapter.log_event("test_event", {"key": "value"})
        adapter.log_event("pipeline_start", {"pipeline_id": "test_123"})
    
    def test_audit_bridge_attribute_exists(self):
        """Adapter should have audit_bridge component."""
        adapter = DGSFAdapter()
        assert hasattr(adapter, 'audit_bridge')
        assert adapter.audit_bridge is not None


class TestDGSFAdapterConfigAccess:
    """Test configuration and spec loading (may fail if DGSF not present)."""
    
    def test_list_specs_returns_list(self):
        """list_specs() should return a list (empty if DGSF unavailable)."""
        adapter = DGSFAdapter()
        specs = adapter.list_specs()
        
        assert isinstance(specs, list)
        # May be empty if legacy not available
    
    def test_list_configs_returns_list(self):
        """list_configs() should return a list (empty if DGSF unavailable)."""
        adapter = DGSFAdapter()
        configs = adapter.list_configs()
        
        assert isinstance(configs, list)
        # May be empty if legacy not available


@pytest.mark.skip(reason="run_experiment() not yet implemented - interface contract TBD")
class TestDGSFAdapterRunExperiment:
    """
    End-to-end test for run_experiment() workflow.
    
    CURRENT STATUS: BLOCKED - METHOD NOT IMPLEMENTED
    
    Expected Interface Contract (to be defined):
        adapter.run_experiment(
            experiment_config: dict,
            output_dir: Path,
            enable_logging: bool = True
        ) -> ExperimentResult
    
    Expected Workflow:
        1. Load experiment configuration (SDF model, hyperparameters, data path)
        2. Initialize DGSF modules (sdf, dataeng, etc.)
        3. Set up logging and audit trail
        4. Run training/backtesting pipeline
        5. Capture metrics and artifacts
        6. Return structured result with status, metrics, artifact paths
    
    Verification Points:
        - Experiment config validated before execution
        - Audit events logged at key checkpoints
        - Output directory contains expected artifacts
        - Result includes timing, metrics, and status
        - Errors are captured and reported gracefully
    
    TODO (when implementing run_experiment):
        1. Define ExperimentResult dataclass
        2. Implement run_experiment() in DGSFAdapter
        3. Remove @pytest.mark.skip decorator
        4. Update this test with actual workflow
    """
    
    def test_adapter_run_experiment_e2e(self):
        """
        Full end-to-end test of experiment workflow.
        
        Placeholder for future implementation.
        """
        adapter = DGSFAdapter()
        
        # Example interface (to be implemented):
        # result = adapter.run_experiment(
        #     experiment_config={
        #         'model_type': 'GenerativeSDF',
        #         'hyperparameters': {...},
        #         'data_path': 'data/a0'
        #     },
        #     output_dir=Path('experiments/test_run'),
        #     enable_logging=True
        # )
        # 
        # assert result.status == 'completed'
        # assert result.metrics is not None
        # assert result.duration > 0
        # assert (output_dir / 'checkpoints').exists()
        
        pytest.fail("run_experiment() interface not yet defined")


# Test Execution Summary
"""
Expected Test Results (as of 2026-02-02):
    
    PASSING (if DGSF repo present):
        ✅ TestDGSFAdapterInitialization (4 tests)
        ✅ TestDGSFAdapterHealthCheck (3 tests)
        ✅ TestDGSFAdapterModuleAccess (2 tests)
        ✅ TestDGSFAdapterAuditBridge (2 tests)
        ✅ TestDGSFAdapterConfigAccess (2 tests)
    
    SKIPPED:
        ⏸️ TestDGSFAdapterRunExperiment (1 test) - awaiting implementation
    
    Total: 13 tests (12 executable + 1 skipped)

Verification Command:
    pytest projects/dgsf/adapter/tests/test_integration.py -v

DoD Criteria:
    - All 12 tests pass OR fail with clear diagnostic messages
    - 1 test properly skipped with implementation note
    - Test file is well-documented with Booch's rationale
    - run_experiment() contract is specified in skip message
"""
