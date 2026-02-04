---
description: Execute a planned DGSF task with verification
mode: agent
---

# DGSF Execute Prompt

Execute a task with pre-flight verification and post-execution validation.

## CORE RULES (from Kernel)

- **R1**: Verify before asserting
- **R2**: One task at a time
- **R3**: Stop on failure
- **R4**: Protect raw data — never write to `data/raw/`
- **R5**: No assumptions
- **R6**: Long-run handoff — if runtime > 3 min, hand off to human

## INPUTS

| Required | Description |
|----------|-------------|
| Task | What to do (single sentence) |
| Target | File path(s) to modify |

| Optional | Default |
|----------|---------|
| Expected Outcome | "Tests pass" |

## EXECUTION PROTOCOL

```
PRE-FLIGHT (always do first):
  □ ls {target_directory}     → confirm paths exist
  □ pytest --collect-only -q  → confirm test discovery
  □ Check Python env active

EXECUTE (one file at a time):
  □ Read current state
  □ Make minimal change
  □ Validate syntax

POST-FLIGHT (always do after):
  □ pytest -v -x              → stop on first failure
  □ Report results with paths
  □ **→ IF working tree dirty: invoke `/dgsf_git_ops`**

ON FAILURE:
  □ STOP immediately
  □ Invoke /dgsf_diagnose with error message
  □ DO NOT attempt fix without diagnosis
```

## EXAMPLE: Good Execution

```
User: "Add dropout to SDF model"

Copilot actions:
1. ls projects/dgsf/repo/src/dgsf/sdf/
   → Found: model.py, __init__.py, loss.py
   
2. Read model.py lines 1-50
   → Found SDFModel class at line 23
   
3. Edit: Add nn.Dropout(0.3) after line 45
   
4. pytest projects/dgsf/repo/tests/test_sdf.py -v
   → PASSED (3 tests)
   
5. Summary: Added dropout(0.3) to SDFModel forward().
   File: projects/dgsf/repo/src/dgsf/sdf/model.py#L45
```

## EXAMPLE: Correct Failure Handling

```
User: "Fix the bug in data loader"

Copilot actions:
1. ls projects/dgsf/repo/src/dgsf/dataeng/
   → ERROR: Directory not found
   
2. STOP. Report:
   "Cannot proceed. Path 'src/dgsf/dataeng/' not found.
    Did you mean 'projects/dgsf/scripts/data_loaders.py'?"
```

## LONG-RUNNING TASK PROTOCOL

For training > 60 seconds:

```powershell
# 1. Create experiment dir
New-Item -ItemType Directory -Path "projects/dgsf/experiments/t5_new"

# 2. Run in background
Start-Process -NoNewWindow -FilePath "python" -ArgumentList `
    "scripts/train.py --config configs/exp.yaml" `
    -RedirectStandardOutput "experiments/t5_new/run.log" `
    -RedirectStandardError "experiments/t5_new/error.log"

# 3. Return immediately with:
#    - Log path
#    - How to check: Get-Content experiments/t5_new/run.log -Tail 20
```

## BOUNDARIES

- ONE task per invocation (no bundling)
- NO modifications to `data/raw/` or `docs/`, `specs/`, `kernel/`
- NO guessing — if path unclear, ASK
