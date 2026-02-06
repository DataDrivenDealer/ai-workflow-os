# Proposal: Fix Legacy Spec Path Drift (DGSF)

**Date**: 2026-02-05  
**Scope**: Spec registry path corrections for DGSF legacy specs.

## Problem
`spec_registry.yaml` currently points some DGSF legacy spec IDs to paths under `projects/dgsf/legacy/...` that do not exist in this workspace.

This breaks:
- traceability checks
- spec tooling that expects `location.path` to be resolvable
- automation that uses spec pointers to find authoritative docs

## Evidence (current authoritative files on disk)
- SDF Layer Spec v3.1 exists at:
  - `projects/dgsf/repo/docs/specs_v3/DGSF SDF Layer Specification v3.1.md`
- A standalone `State Engine Spec v1.0.txt` file is **not present** in the workspace.
  - The closest authoritative contract for StateEngine is:
    - `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml` (Section 1: State Engine Interface)

## Proposed Changes
1. Update `DGSF_SDF_V3.1.location.path` to the real on-disk SDF v3.1 spec:
   - `projects/dgsf/repo/docs/specs_v3/DGSF SDF Layer Specification v3.1.md`

2. Update `STATE_ENGINE_V1.0.location` to point to the operationally authoritative contract source until the standalone legacy text spec is recovered:
   - `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml` (format: yaml)

3. Add this proposal doc under each affected spec’s `links.proposals` for auditability.

## Governance / Risk Notes
- This proposal does **not** change the mathematical or interface requirements; it only fixes the pointer to where those requirements live in the current workspace.
- Once a standalone StateEngine legacy doc is restored, `STATE_ENGINE_V1.0.location` can be updated back to the text spec via a follow-up proposal.

## Acceptance
- Spec tooling can resolve the spec files by path.
- References in taskcards/spec pointers remain consistent.
