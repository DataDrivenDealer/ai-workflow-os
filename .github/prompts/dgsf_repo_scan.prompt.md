---
description: Scan and report on DGSF repository structure and state
mode: agent
---

# DGSF Repo Scan Prompt

Understand project state by scanning actual files. No assumptions.

## WHEN TO INVOKE

- Before `/dgsf_plan` if unsure of current structure
- After `/dgsf_research` to verify internal state
- When user asks "what do we have?" or "what's the status?"

## CORE RULES (from Kernel)

- **R1**: Verify before asserting — scan IS verification
- **R5**: No assumptions — only report what exists

## INPUTS

| Required | Description |
|----------|-------------|
| Question | What to find/understand |

| Optional | Default |
|----------|---------|
| Scope | "full" (or: module, experiments, data) |

## SCAN COMMANDS

```powershell
# Module structure
Get-ChildItem "projects/dgsf/repo/src/dgsf" -Directory | Select-Object Name

# Experiments with dates
Get-ChildItem "projects/dgsf/experiments" -Directory | 
    Select-Object Name, LastWriteTime | Sort-Object LastWriteTime -Descending

# Test count
pytest projects/dgsf/repo/tests --collect-only -q 2>$null | Select-Object -Last 1

# Data files
Get-ChildItem "projects/dgsf/data/processed" -File -Recurse | 
    Select-Object Name, Length, LastWriteTime
```

## OUTPUT FORMAT

```markdown
## Scan: {question answered}

**Scope**: {what was scanned}
**Date**: {timestamp}

### Findings
{direct answer to question}

### Structure
| Path | Status | Last Modified |
|------|--------|---------------|
| {path} | ✅ exists / ❌ missing | {date} |

### Issues (if any)
- {issue}
```

## EXAMPLE

```
User: "What experiments have been run?"

Scan output:
## Scan: Experiment inventory

**Scope**: projects/dgsf/experiments/
**Date**: 2026-02-04 15:00

### Findings
12 experiments found, latest is t5_oos_validation (2026-02-03)

### Structure
| Experiment | Has results.json | Last Modified |
|------------|------------------|---------------|
| t5_oos_validation | ✅ | 2026-02-03 |
| t4_final | ✅ | 2026-02-01 |
| t4_regularization | ✅ | 2026-01-30 |
...

### Issues
- t4_baseline missing run.log
```

## BOUNDARIES

- ONLY report what `ls`/`Get-ChildItem` returns
- NO scanning `docs/`, `specs/`, `kernel/` unless explicitly asked
- MUST include timestamps for experiments
