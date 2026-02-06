# ⚠️ ARCHIVED - DO NOT MODIFY

**Status**: Historical Archive Only  
**Last Updated**: 2026-02-02  
**Purpose**: Reference and audit trail preservation

---

## 🔒 This Directory is Read-Only

This directory contains **historical DGSF assets** migrated during the legacy integration phase (Stage 0-1, completed 2026-02-01).

**⚠️ ALL ACTIVE DEVELOPMENT MUST OCCUR IN `projects/dgsf/repo/` (Git Submodule)**

---

## 📋 Contents

This archive includes:
- **Legacy source code**: `DGSF/src/dgsf/` (archived implementation)
- **Legacy tests**: `DGSF/tests/` (causing pytest collection errors if not excluded)
- **Legacy configs**: `DGSF/configs/` (historical configurations)
- **Legacy specs**: `DGSF/docs/specs_v3/` (superseded by PROJECT_DGSF.yaml)

---

## 🚫 Why This Directory Exists

These files are preserved for:
1. **Historical Reference**: Understanding past decisions and implementation choices
2. **Migration Validation**: Verifying that Stage 2 data migration was complete
3. **Audit Trails**: Compliance with governance invariants (INV-5: Audit Completeness)
4. **Reproducibility**: Baseline reproduction verification (Stage 3)

---

## ✅ Where to Work

| Activity | Correct Location | Wrong Location |
|----------|-----------------|----------------|
| Add new features | `projects/dgsf/repo/src/` | ❌ `projects/dgsf/legacy/DGSF/src/` |
| Run tests | `cd projects/dgsf/repo && pytest` | ❌ `pytest projects/dgsf/legacy/` |
| Update specs | `projects/dgsf/specs/PROJECT_DGSF.yaml` | ❌ `projects/dgsf/legacy/DGSF/docs/` |
| Modify configs | `projects/dgsf/repo/configs/` | ❌ `projects/dgsf/legacy/DGSF/configs/` |

---

## 🔗 Related Documentation

- [DGSF Execution Plan](../../docs/plans/EXECUTION_PLAN_DGSF_V1.md) - Current research roadmap
- [PROJECT_DGSF.yaml](../specs/PROJECT_DGSF.yaml) - Active project specification
- [DGSF README](../README.md) - Main DGSF project documentation
- [SPEC_MAPPING_PLAN.md](../docs/SPEC_MAPPING_PLAN.md) - Legacy → OS spec mapping

---

## 🛠️ Developer Workflow

If you need to reference legacy code:

```powershell
# View legacy file (read-only)
Get-Content projects/dgsf/legacy/DGSF/src/dgsf/sdf/model.py

# DO NOT copy-paste directly - use as reference only
# Implement in projects/dgsf/repo/ with proper integration
```

---

## 📊 Known Issues

**pytest Collection Errors**: This directory causes 165 import errors when pytest scans it. Solution:
- ✅ AI Workflow OS root `pytest.ini` excludes this directory
- ✅ Use `pytest kernel/tests/` (not `pytest .`)

**Hard-coded Paths**: Legacy code may reference `e:\DGSF\`, which no longer exists. Use `projects/dgsf/repo/` instead.

---

## 🔐 Governance

**Authority**: Archived by decision of Project Owner (2026-02-01)  
**Spec Reference**: PROJECT_DGSF.yaml Section 5 (Legacy Asset Registry)  
**Change Policy**: This directory is frozen. No modifications allowed without formal approval.

---

**Last Verified**: 2026-02-02  
**Verification**: P0-3 execution (DGSF repo submodule confirmed as primary development location)
