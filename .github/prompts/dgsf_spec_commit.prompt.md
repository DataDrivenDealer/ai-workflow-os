---
description: Commit an approved spec change with governance verification
mode: agent
inherits_rules: [R1, R3, R4]
tools: [spec_commit, governance_check]
requires: [human_approval]
---

# DGSF Spec Commit Prompt

Commit an approved specification change with full audit trail.

## CORE RULES (from Kernel)

- **R1**: Verify before asserting — confirm approval exists before commit
- **R3**: Stop on failure — abort if any verification fails
- **R4**: Protect raw data — this applies to Canon specs (L0) as read-only

## PREREQUISITES

- Spec change proposal MUST be approved by authorized human
- Approval record MUST exist in `decisions/` or GitHub PR
- For L3 (experiment config): auto-approval allowed if thresholds pass

## INPUTS

| Required | Description |
|----------|-------------|
| Proposal ID | SCP-{YYYY-MM-DD}-{NNN} from `/dgsf_spec_propose` |
| Approval Reference | Decision log path or PR number |

| Optional | Default |
|----------|---------|
| Commit Message | Auto-generated from proposal |
| Run Post-Hooks | `true` |

## PERMISSION MATRIX

| Spec Layer | Approval Required From | Auto-Commit Allowed |
|------------|------------------------|---------------------|
| L0 Canon | Project Owner (freeze required) | ❌ Never |
| L1 Framework | Platform Engineer | ❌ No |
| L2 Project | Project Lead | ❌ No |
| L3 Experiment | Threshold verification | ✓ Yes |

## COMMIT PROTOCOL

```
PHASE 1 — Verify Approval
  □ Check decisions/{proposal_id}.yaml exists
    OR GitHub PR #{pr_number} is merged
  □ Verify approver has required authority
  □ Verify proposal has not expired (< 7 days old)

PHASE 2 — Pre-Commit Hooks
  □ Trigger: hooks/pre-spec-change
  □ Verify: No Canon violations
  □ Verify: Diff matches approved proposal
  □ Verify: All required fields present

PHASE 3 — Apply Change
  □ MCP: spec_commit(proposal_id, approval_ref)
  □ Write change to spec file
  □ Update lineage tracking

PHASE 4 — Post-Commit Hooks
  □ Trigger: hooks/post-spec-change
  □ Update spec_registry.yaml if needed
  □ Trigger regression tests
  □ Notify affected experiment owners

PHASE 5 — Git Operations
  □ Invoke /dgsf_git_ops for commit and push
  □ Tag with spec change ID if L1/L2
```

## OUTPUT FORMAT

```markdown
## Spec Commit: SCP-{YYYY-MM-DD}-{NNN}

**Status**: ✓ Committed
**Spec**: `{spec_path}`
**Commit**: `{git_commit_hash}`
**Timestamp**: {ISO-8601}

### Approval Chain
- Proposed By: AI Agent (session: {session_id})
- Approved By: {human_name} ({role})
- Approval Ref: decisions/{proposal_id}.yaml

### Changes Applied
```diff
{actual_diff_applied}
```

### Post-Commit Actions
- [x] Lineage updated: `projects/dgsf/lineage/spec_changes.yaml`
- [x] Tests triggered: `pytest tests/test_sdf_validation.py`
- [ ] Pending: Rerun affected experiments (t01_baseline, t02_dropout)

### Audit Record
```yaml
# Automatically written to ops/audit/spec_commits.yaml
- id: SCP-2026-02-04-001
  spec: projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml
  change_type: modify
  approved_by: Project Lead
  committed_by: AI Agent
  timestamp: 2026-02-04T10:45:00Z
  git_commit: abc123def
```
```

## EXAMPLE: Successful Commit

```
User: "Commit spec proposal SCP-2026-02-04-001, approved in PR #42"

Copilot actions:
1. Verify approval:
   → GitHub PR #42 status: merged
   → Approver: @project-lead (role: Project Lead)
   → Authority: Valid for L2 specs

2. Run pre-spec-change hook:
   → Canon check: PASS (not touching L0)
   → Diff validation: PASS (matches proposal)

3. Apply change:
   → Updated: projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml
   → Line 46: min_sharpe_threshold: 1.0 → 1.5

4. Run post-spec-change hook:
   → Lineage updated
   → Regression tests queued

5. Git operations:
   → Commit: "spec: update SDF threshold to 1.5 [SCP-2026-02-04-001]"
   → Push: origin/main

Output: "Spec change SCP-2026-02-04-001 committed successfully."
```

## EXAMPLE: Blocked Commit (No Approval)

```
User: "Commit spec proposal SCP-2026-02-04-002"

Copilot actions:
1. Verify approval:
   → decisions/SCP-2026-02-04-002.yaml: NOT FOUND
   → GitHub PR search: No matching PR

2. STOP. Output:
   "Cannot commit SCP-2026-02-04-002: No approval record found.
    
    Required: L2 spec changes need Project Lead approval.
    
    Options:
    1. Create decision record: decisions/SCP-2026-02-04-002.yaml
    2. Create GitHub PR and get approval
    3. If already approved, provide the approval reference"
```

## ROLLBACK PROTOCOL

If post-commit tests fail:

```
1. Identify failure:
   → pytest output shows test_sdf_validation.py FAILED

2. Automatic rollback (if enabled):
   → git revert {commit_hash}
   → Restore previous spec version

3. Create incident record:
   → ops/incidents/INC-{timestamp}.yaml
   → Link to failed commit and test output

4. Notify:
   → "Spec change {SCP-ID} rolled back due to test failure.
      See ops/incidents/INC-{timestamp}.yaml for details."
```

## INTEGRATION WITH VS CODE

When using VS Code + Copilot:

1. **Trigger**: Type `/dgsf_spec_commit` in Copilot Chat
2. **Approval Check**: Copilot will search for decision records
3. **Diff Confirmation**: Shows exact changes before applying
4. **Git Integration**: Uses VS Code Source Control for commit
5. **Test Runner**: Triggers VS Code Test Explorer for regression tests
