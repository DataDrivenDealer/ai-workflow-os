---
description: Propose a change to DGSF specification based on research findings
mode: agent
inherits_rules: [R1, R2, R5]
tools: [spec_read, spec_propose, governance_check]
---

# DGSF Spec Propose Prompt

Generate a formal specification change proposal based on research or problem diagnosis.

## CORE RULES (from Kernel)

- **R1**: Verify before asserting — read current spec before proposing changes
- **R2**: One task at a time — propose ONE spec change per invocation
- **R5**: No assumptions — cite evidence for each proposed change

## PREREQUISITES

- Must have completed `/dgsf_research` or `/dgsf_diagnose` first
- Must have a clear rationale linked to research findings

## INPUTS

| Required | Description |
|----------|-------------|
| Spec Path | Path to spec file (e.g., `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml`) |
| Change Type | `add` \| `modify` \| `deprecate` |
| Rationale | Why this change is needed (link to research) |

| Optional | Default |
|----------|---------|
| Impact Assessment | Auto-generated |
| Related Experiments | None |

## SPEC HIERARCHY (Permissions)

| Layer | Path Pattern | AI Can Propose | AI Can Commit |
|-------|-------------|----------------|---------------|
| L0 Canon | `specs/canon/*` | ❌ No | ❌ No |
| L1 Framework | `specs/framework/*` | ✓ Yes | ❌ No |
| L2 Project | `projects/dgsf/specs/*` | ✓ Yes | ❌ No |
| L3 Experiment | `projects/dgsf/experiments/*/config.yaml` | ✓ Yes | ✓ Yes* |

*L3 auto-commit requires passing threshold verification.

## PROPOSAL PROTOCOL

```
PHASE 1 — Verify Current State
  □ MCP: spec_read(spec_path) → get current content
  □ Identify exact section/field to change
  □ Document current value and location

PHASE 2 — Generate Proposal
  □ Write proposed changes in YAML/Markdown diff format
  □ Link to evidence (research findings, experiment results)
  □ Assess backward compatibility

PHASE 3 — Impact Analysis
  □ Which experiments depend on this spec?
  □ Which code modules reference this spec?
  □ Will existing tests pass after change?

PHASE 4 — Submit for Review
  □ MCP: spec_propose(spec_path, change_type, rationale, diff)
  □ Wait for human approval (L0-L2) or auto-verify (L3)
```

## OUTPUT FORMAT

```markdown
## Spec Change Proposal: SCP-{YYYY-MM-DD}-{NNN}

**Spec**: `{spec_path}`
**Type**: {add | modify | deprecate}
**Status**: Proposed → Pending Review
**Proposed By**: AI Agent (session: {session_id})
**Date**: {timestamp}

### Rationale
{Why this change is needed — 2-3 sentences}

**Evidence**:
- Research: `/dgsf_research` on {date} found {finding}
- Experiment: `experiments/{exp_id}/results.json` showed {metric}

### Current State
```yaml
# {spec_path}:{line_range}
{current_content}
```

### Proposed Change
```diff
- {old_value}
+ {new_value}
```

### Impact Assessment

| Dimension | Assessment |
|-----------|------------|
| Backward Compatible | {Yes/No} |
| Experiments Affected | {list or "None"} |
| Code Modules Affected | {list or "None"} |
| Tests to Rerun | {list or "All"} |

### Verification Plan
1. {What test to run after approval}
2. {What metric to check}

### Approval Required From
- [ ] {Role: Project Lead / Platform Engineer / Project Owner}
```

## EXAMPLE: Good Proposal

```markdown
## Spec Change Proposal: SCP-2026-02-04-001

**Spec**: `projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml`
**Type**: modify
**Status**: Proposed → Pending Review
**Proposed By**: AI Agent (session: abc123)
**Date**: 2026-02-04T10:30:00Z

### Rationale
Current `min_sharpe_threshold` of 1.0 is too lenient, allowing underperforming 
models to pass validation. Research on 2026-02-03 found that industry standard 
is 1.5 for production-grade factor models.

**Evidence**:
- Research: Analyzed 15 academic papers on SDF validation criteria
- Experiment: `experiments/t03_threshold_study/results.json` showed OOS Sharpe 
  below 1.2 correlates with regime instability (p < 0.05)

### Current State
```yaml
# projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml:45-47
validation:
  min_sharpe_threshold: 1.0
  max_drawdown: 0.25
```

### Proposed Change
```diff
validation:
-  min_sharpe_threshold: 1.0
+  min_sharpe_threshold: 1.5
   max_drawdown: 0.25
```

### Impact Assessment

| Dimension | Assessment |
|-----------|------------|
| Backward Compatible | No — stricter threshold |
| Experiments Affected | t01_baseline, t02_dropout |
| Code Modules Affected | src/dgsf/sdf/validator.py |
| Tests to Rerun | test_sdf_validation.py |

### Verification Plan
1. Run `pytest tests/test_sdf_validation.py -v`
2. Verify new threshold correctly rejects models with Sharpe < 1.5

### Approval Required From
- [ ] Project Lead (L2 spec change)
```

## ERROR HANDLING

```
IF spec_path in specs/canon/*:
  → STOP. Output: "Cannot propose changes to Canon specs (L0).
    Contact Project Owner for manual modification."

IF rationale is empty:
  → STOP. Output: "Rationale required. Run /dgsf_research first."

IF impact_assessment shows >10 experiments affected:
  → WARN: "High impact change. Consider phased rollout."
```

## INTEGRATION WITH VS CODE

When using VS Code + Copilot:

1. **Trigger**: Type `/dgsf_spec_propose` in Copilot Chat
2. **Context**: Copilot will read currently open spec file
3. **Diff Preview**: Proposal will be shown in VS Code diff view
4. **Approval**: Create GitHub PR or use `decisions/` folder for record
