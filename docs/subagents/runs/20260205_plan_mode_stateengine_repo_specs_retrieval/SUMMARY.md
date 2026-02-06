# repo_specs_retrieval — StateEngine + SDF Feature Eng (2026-02-05)

## Scope
Read-only retrieval of authoritative specs/taskcards and the current handoff contract impacting StateEngine implementation and SDF feature engineering.

## Authoritative Artifacts (paths)
- Frozen interface contract: `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml`
- SDF dev taskcard (includes StateEngine spec stub): `tasks/SDF_DEV_001.md`
- StateEngine integration taskcard: `tasks/STATE_ENGINE_INTEGRATION_001.md`
- Feature engineering taskcard (T3): `tasks/active/SDF_FEATURE_ENG_001.md`
- Plan→Execute handoff: `state/execution_queue.yaml`
- Escalations (context): `state/escalation_queue.yaml`

## Key Contract Fields — StateEngine
From `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml` (Section 1):
- `state_engine.module`: `dgsf.sdf.state_engine`
- Inputs: `returns: np.ndarray [T,K]` (T≥252, K≥10), `turnover: np.ndarray [T,K]` (non-negative)
- Outputs: `phi: np.ndarray [T,J]` (J=4 baseline, J=5 if crowd), `V_t: np.ndarray [T]`, `L_t: np.ndarray [T]`
- Params: `vol_lookback=12` (frozen), `std_window=36` (frozen), `lambda_ewma=0.8` (frozen), `include_crowd` (toggle)

## Required Deliverables (must exist to claim compliance)
- `projects/dgsf/repo/src/dgsf/sdf/state_engine.py`
- `projects/dgsf/repo/src/dgsf/sdf/data_adapter.py`
- At least one test module under `projects/dgsf/repo/tests/sdf/` covering StateEngine + adapter integration
- E2E script: `projects/dgsf/scripts/test_state_engine_e2e.py` must be runnable after the above exist

## Verification Commands (as governed by the queue/taskcards)
- Import StateEngine:
  - `python -c "import dgsf.sdf.state_engine as m; print('ok')"`
- Import adapter symbols:
  - `python -c "from dgsf.sdf.data_adapter import StateEngineDataAdapter, StateEngineData, validate_state_engine_output, HAS_PANDAS; print('ok', HAS_PANDAS)"`
- Run focused tests:
  - `pytest tests/sdf/test_state_engine* -v --tb=short`
- Run E2E (generate report):
  - `python projects/dgsf/scripts/test_state_engine_e2e.py --output projects/dgsf/reports/state_engine_e2e_report.json`
