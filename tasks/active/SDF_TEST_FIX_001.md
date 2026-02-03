---
task_id: "SDF_TEST_FIX_001"
type: dev
queue: dev
branch: "feature/router-v0"
priority: P0
parent_task: "SDF_DEV_001_T2"
spec_ids:
  - PROJECT_DGSF
  - SDF_INTERFACE_CONTRACT
related_reports:
  - "projects/dgsf/reports/SDF_TEST_FAILURES.md"
  - "projects/dgsf/reports/SDF_MODEL_INVENTORY.json"
verification:
  - "pytest tests/sdf/ --collect-only shows 167 tests collected"
  - "pytest tests/sdf/ -v shows >95% pass rate"
  - "No ModuleNotFoundError for dgsf.sdf.state_engine"
---

# Task SDF_TEST_FIX_001: Fix SDF Test Suite Blocking Import Error

## Summary

**Context**: All 11 SDF test files (167 tests) were blocked by a single import error in `src/dgsf/sdf/__init__.py:53` attempting to import non-existent module `dgsf.sdf.state_engine`. This task resolves the immediate blocker to enable SDF test execution and provides a path forward for remaining test failures.

**Primary Goal**: Unblock SDF test collection and execution by resolving the `state_engine` import error.

**Link to Parent**: This is the first actionable subtask of `SDF_DEV_001_T2` ("Fix SDF Test Failures"), which has 2-week estimated effort. This specific fix is expected to unblock test execution within 5 minutes.

## Background

### Discovery (P0-2 Analysis)
- **Date**: 2026-02-02T18:40:00Z
- **Analyst**: Gene Kim (Execution Flow Expert)
- **Finding**: 100% test collection failure due to single import error
- **Impact**: 0 out of 167 tests executable

### Root Cause
```python
# src/dgsf/sdf/__init__.py:53
from .state_engine import (
    # ... expected imports
)
```

**Problem**: `state_engine.py` module does not exist in `src/dgsf/sdf/` directory.

**Possible Reasons**:
1. Planned but not yet implemented
2. Accidentally deleted or not committed
3. Outdated import statement from refactoring

### Current Status (After P0-3)
- ✅ Immediate fix applied: Commented out `state_engine` imports with FIXME markers
- ✅ Test collection unblocked: 167 tests now collectable
- ⏸️ Test execution status: Unknown (tests not yet run, may have additional failures)

## Implementation Notes

### Phase 1: Immediate Unblocking (✅ COMPLETED in P0-3)
**Actions Taken**:
1. Commented out lines 53-57 in `src/dgsf/sdf/__init__.py` (state_engine import)
2. Commented out lines 66-69 in `__all__` (state_engine exports)
3. Added FIXME comments linking to SDF_TEST_FAILURES.md
4. Committed to DGSF submodule: `8031647`
5. Verified: `pytest tests/sdf/ --collect-only` → 167 tests collected ✅

### Phase 2: Execute and Categorize Remaining Failures (NEXT)
**Objectives**:
1. Run full SDF test suite: `pytest tests/sdf/ -v --tb=short`
2. Categorize failures by type:
   - Import/dependency errors
   - Schema/data validation failures
   - Logic/assertion failures
   - Fixture/setup issues
3. Generate updated failure report: `SDF_TEST_FAILURES_DETAILED.md`
4. Estimate effort for each failure category

**Expected Outcome**:
- Baseline pass rate (likely 20-60% given typical test suite health)
- Clear taxonomy of remaining issues
- Prioritized fix backlog

### Phase 3: Incremental Fixing (FUTURE)
**Strategy**:
1. Fix highest-impact categories first (e.g., fixture setup issues affecting multiple tests)
2. Batch similar failures (e.g., all schema validation errors)
3. Commit after each batch passes
4. Track progress: aim for >95% pass rate (as per SDF_DEV_001_T2 deliverables)

### Phase 4: Decide on state_engine (DEFERRED)
**Decision Point**: After achieving stable test suite
**Options**:
1. **Implement state_engine**: If functionality is needed for SDF pipeline
2. **Remove permanently**: If imports were added prematurely or from abandoned design
3. **Keep commented**: If implementation is planned but not P0

## Verification

### DoD Criteria

**Phase 1 Checklist** (✅ COMPLETED):
- [x] Test collection succeeds without ModuleNotFoundError
- [x] 167 tests detected by pytest
- [x] FIXME comments document temporary nature of fix
- [x] Changes committed to DGSF submodule

**Phase 2 Checklist** (NEXT):
- [ ] Full test suite executed: `pytest tests/sdf/ -v --tb=short > SDF_TEST_RESULTS.txt 2>&1`
- [ ] Failure report generated with categories and counts
- [ ] Pass rate calculated: `(passed / total) * 100`
- [ ] Effort estimated for each failure category

**Phase 3 Checklist** (FUTURE):
- [ ] Pass rate >95% achieved
- [ ] Coverage report generated: `pytest --cov=src/dgsf/sdf tests/sdf/`
- [ ] Coverage >80% verified (SDF_DEV_001_T2 requirement)
- [ ] All commits pass pre-commit checks

**Phase 4 Checklist** (DEFERRED):
- [ ] state_engine decision documented in ARCHITECTURE_DECISION_RECORD.md
- [ ] If removed: __all__ exports cleaned up
- [ ] If implemented: module added with tests

### Verification Commands

```powershell
# Phase 1 Verification (COMPLETED)
cd projects/dgsf/repo
pytest tests/sdf/ --collect-only
# Expected: "collected 167 items"

# Phase 2 Verification (NEXT)
cd projects/dgsf/repo
pytest tests/sdf/ -v --tb=short --maxfail=20 > ../reports/SDF_TEST_RESULTS.txt 2>&1
type ..\reports\SDF_TEST_RESULTS.txt | Select-String -Pattern "passed|failed|error"

# Phase 3 Verification (FUTURE)
pytest tests/sdf/ -v
pytest --cov=src/dgsf/sdf --cov-report=term-missing tests/sdf/

# Phase 4 Verification (DEFERRED)
grep -r "state_engine" src/dgsf/sdf/
# Should show only implementation files or no results if removed
```

## Success Metrics

- **Phase 1**: 167 tests collectable (✅ ACHIEVED)
- **Phase 2**: Failure taxonomy documented, <20 failure categories
- **Phase 3**: >95% pass rate, >80% coverage
- **Phase 4**: Zero FIXME comments related to state_engine

## Dependencies

- **Upstream**: SDF_DEV_001_T1 (✅ COMPLETED - Model Architecture Review)
- **Downstream**: 
  - SDF_DEV_001_T3 (SDF Feature Engineering - blocked until tests pass)
  - SDF_DEV_001_T4 (SDF Training Pipeline Optimization - blocked until tests pass)

## Timeline

- **Phase 1**: ✅ COMPLETED 2026-02-02T18:50Z (5 min actual)
- **Phase 2**: Estimated 30 min (test execution + analysis)
- **Phase 3**: Estimated 1-2 weeks (depends on failure complexity)
- **Phase 4**: Estimated 1-3 days (depends on implementation scope)

**Total Estimated Effort**: Aligns with parent task `SDF_DEV_001_T2` (2 weeks)

## Notes

- **Risk**: Commenting out state_engine may hide functionality issues; carefully validate SDF pipeline behavior after fix
- **Opportunity**: Clean test suite enables confident refactoring and feature development
- **Constraint**: Must maintain >95% pass rate per SDF_DEV_001_T2 requirement before marking parent task complete
- **Next Action**: Execute Phase 2 (run tests and categorize failures) as next P1 task
