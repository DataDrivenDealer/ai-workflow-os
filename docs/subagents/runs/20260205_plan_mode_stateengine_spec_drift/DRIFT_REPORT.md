# spec_drift — StateEngine + Feature Engineering (2026-02-05)

## Drift Items

### 1) Contract points to missing module (CRITICAL)
- Evidence:
  - `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml` declares `state_engine.module: dgsf.sdf.state_engine`
  - `projects/dgsf/repo/src/dgsf/sdf/__init__.py` comments out state_engine import with “module not found”
- Impact:
  - Frozen contract is currently unsatisfiable; E2E script imports fail.
- Recommended resolution (chosen by owner):
  - Implement `projects/dgsf/repo/src/dgsf/sdf/state_engine.py` to satisfy the frozen contract.

### 2) Integration TaskCard claims DONE deliverables that are not present (HIGH)
- Evidence:
  - `tasks/STATE_ENGINE_INTEGRATION_001.md` listed adapter/tests as ✅ DONE but repo paths are missing.
  - `projects/dgsf/scripts/test_state_engine_e2e.py` imports `dgsf.sdf.state_engine` and `dgsf.sdf.data_adapter`.
- Impact:
  - Governance indicates completion that cannot be reproduced.
- Recommended resolution:
  - Rebaseline deliverables to PENDING/BLOCKED/STale and make completion depend on in-repo artifacts.

### 3) T3.2 dual meaning between taskcard and execution queue (HIGH)
- Evidence:
  - `tasks/active/SDF_FEATURE_ENG_001.md`: T3.2 = documentation output.
  - `state/execution_queue.yaml`: T3.2 = point-in-time compliance verification.
  - `state/escalation_queue.yaml` ESC-2026-02-05-001 records the mismatch.
- Impact:
  - Broken traceability; ambiguous completion signals.
- Recommended resolution:
  - Split into T3.2 (PIT) and T3.2b (DOC) in the taskcard.

### 4) Misleading “compute.py is computation logic” (MEDIUM)
- Evidence:
  - `projects/dgsf/repo/src/dgsf/factors/compute.py` is placeholder.
  - DE7 builder contains real factor computation.
- Recommended resolution:
  - Mark factors/compute.py as placeholder and designate DE7 as canonical until refactor.

## Execution Queue Impact
- Insert/ensure P0 chain for option (A):
  1. Implement `dgsf.sdf.state_engine`
  2. Implement `dgsf.sdf.data_adapter`
  3. Add tests under `tests/sdf/`
  4. Fix/verify E2E script and report generation
