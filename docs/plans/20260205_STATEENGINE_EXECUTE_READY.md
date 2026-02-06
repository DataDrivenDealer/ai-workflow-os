# Plan — StateEngine Contract Satisfaction (Option A)

**Date**: 2026-02-05  
**Mode**: PLAN (no code execution)  
**Decision**: Implement `dgsf.sdf.state_engine` to satisfy the frozen contract.

## Why this plan exists
- `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml` declares `state_engine.module: dgsf.sdf.state_engine`.
- Current repo state is missing that module (imports are commented out), and the E2E script imports it.

## Execute Mode work items (source of truth)
The authoritative, Execute-ready items are in `state/execution_queue.yaml` (queue item IDs 4–7).

### P0 chain
1. Implement `projects/dgsf/repo/src/dgsf/sdf/state_engine.py`
   - Must export: `StateEngine`, `StateEngineConfig`, `create_baseline_engine`, `create_extended_engine`.
2. Implement `projects/dgsf/repo/src/dgsf/sdf/data_adapter.py`
   - Must export: `StateEngineDataAdapter`, `StateEngineData`, `validate_state_engine_output`, `HAS_PANDAS`.
3. Add focused tests under `projects/dgsf/repo/tests/sdf/`
   - Cover synthetic single-asset + panel cases.

### P1
4. Make `projects/dgsf/scripts/test_state_engine_e2e.py` runnable end-to-end and regenerate `projects/dgsf/reports/state_engine_e2e_report.json`.

## Verification commands (to be run only in EXECUTE MODE)
- `python -c "import dgsf.sdf.state_engine as m; print('ok')"`
- `python -c "from dgsf.sdf.data_adapter import StateEngineDataAdapter, StateEngineData, validate_state_engine_output, HAS_PANDAS; print('ok', HAS_PANDAS)"`
- `pytest tests/sdf/test_state_engine* -v --tb=short`
- `python projects/dgsf/scripts/test_state_engine_e2e.py --output projects/dgsf/reports/state_engine_e2e_report.json`

## Governance notes
- The interface contract is now registered in `spec_registry.yaml` as `SDF_INTERFACE_CONTRACT`.
- Taskcard drift was reconciled:
  - Feature engineering task now splits T3.2 (PIT) vs T3.2b (DOC).
  - Integration taskcard deliverables no longer claim ✅ when repo paths are missing.
