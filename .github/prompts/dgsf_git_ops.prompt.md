---
description: Git 运维子流程 - 状态检查、提交与同步
mode: agent
inherits_rules: [R1, R3, R4]
---

# DGSF Git Ops Prompt

Git 运维作为 Copilot Runtime OS 的内建子流程，在关键工件生成、结构性修改或阶段性任务完成后自动触发。

## 触发条件

在以下 skill 完成后自动调用：

| 触发来源 | 变更类型 | ConfirmLevel | 自动 Tag |
|----------|----------|--------------|----------|
| `/dgsf_execute` 完成 | kernel/* | CONFIRM | 否 |
| `/dgsf_execute` 完成 | experiments/* | NOTIFY | 否 |
| `/dgsf_state_update` "Complete" | experiments/* | NOTIFY | 是 (`exp/*`) |
| `/dgsf_state_update` "Milestone" | 任意 | CONFIRM | 是 (`milestone/*`) |
| `/dgsf_research_summary` | docs/* | AUTO | 否 |
| 手动调用 `/dgsf_git_ops` | 任意 | BLOCK | 用户指定 |

## CORE RULES (from Kernel)

- **R1**: Verify before asserting — 先 `git status` 再生成 plan
- **R3**: Stop on failure — commit 失败立即停止并报告
- **R6**: Long-run handoff — push 操作需人工确认
- **R7**: Branch naming — 分支命名必须符合 `configs/git_branch_policy.yaml` 规范
- **R8**: Hooks check — 每次 git 操作前检测 hooks 安装状态

## 执行协议

### Phase 0: 预检 (PRE-FLIGHT)

**0a. Hooks 安装检查** (R8)
```python
from kernel.git_setup_check import check_git_hooks, prompt_and_install_hooks
status = check_git_hooks()
if not status.hooks_installed:
    prompt_and_install_hooks(status)
```

如果 hooks 未安装，显示：
```markdown
## ⚠️ Git Hooks 未安装

缺失: pre-commit, pre-push, ...

是否立即安装? [Y/n]
```

**0b. 分支命名验证** (R7)
```python
from kernel.git_branch_validator import validate_branch_name
result = validate_branch_name(current_branch)
if not result.valid:
    # BLOCK: 拒绝操作，显示正确格式
```

分支命名规范（GitHub Flow）：
| 类型 | 格式 | 示例 |
|------|------|------|
| 功能 | `feature/{TASK_ID}-{description}` | `feature/GIT_001-branch-policy` |
| 实验 | `experiment/t{NN}_{name}` | `experiment/t05_sharpe_validation` |
| 修复 | `hotfix/{TASK_ID}-{description}` | `hotfix/URGENT_001-fix-crash` |
| 发布 | `release/v{semver}` | `release/v1.0.0` |

### Phase 1: 状态检查 (STATUS CHECK)

```bash
# 必须先验证 Git 仓库状态
git status --porcelain
git diff --stat
git branch --show-current
git describe --tags --always
```

输出示例：
```markdown
## 🔍 Git Status Report

**Branch**: `feature/t05-oos-validation`
**Latest Tag**: `v3.3.0`
**Remote**: 2 commits ahead of `origin/main`

**Changes**:
  - [M] kernel/git_ops.py (unstaged)
  - [A] kernel/tests/test_git_ops.py (staged)
  - [?] experiments/t05/config.yaml (untracked)
```

### Phase 2: 生成提交方案 (PLAN GENERATION)

根据变更分类自动生成 Conventional Commits 格式的提交消息：

| 类型 | Commit Type | 示例 |
|------|-------------|------|
| kernel/* | `feat` | `feat(kernel): add git_ops module` |
| prompts/* | `feat` | `feat(prompts): add dgsf_git_ops skill` |
| experiments/* | `experiment` | `experiment(t05): complete OOS validation` |
| docs/* | `docs` | `docs: update ARCHITECTURE.md` |
| tests/* | `test` | `test: add git_ops unit tests` |
| configs/* | `chore` | `chore(config): update gates.yaml` |

### Phase 3: 确认与执行 (CONFIRM & EXECUTE)

**确认级别说明**：

| Level | 行为 | 示例场景 |
|-------|------|----------|
| `AUTO` | 直接执行，仅输出日志 | docs/ 变更 |
| `NOTIFY` | 执行并通知用户结果 | experiments/ 变更 |
| `CONFIRM` | 输出完整方案，等待 `[Y/n]` | kernel/, prompts/ 变更 |
| `BLOCK` | 输出命令，人工执行 | data/ 变更, remote push |

## 输出格式

### 提交方案 (Commit Plan)

```markdown
## 📦 Git Commit Plan

**Branch**: `feature/t05-oos-validation`
**Confirm Level**: CONFIRM

### Changes
  - [K] `kernel/git_ops.py`
  - [K] `kernel/tests/test_git_ops.py`

### Commit Message
```
feat(kernel): add git_ops module

Integrated Git operations as internal subprocess.

Changes:
  - [kernel] kernel/git_ops.py
  - [tests] kernel/tests/test_git_ops.py

Task: GIT-OPS-001
```

### Tag
`exp/t05_oos_validation/v1`
```
OOS Sharpe: 1.67
OOS/IS Ratio: 0.94
Config Hash: a1b2c3d4
```

**Proceed? [Y/n]**
```

### 执行结果 (Execution Result)

```markdown
## ✅ Git Ops Complete

**Commit**: `abc1234` feat(kernel): add git_ops module
**Tag**: `exp/t05_oos_validation/v1`
**Actions**:
  - Staged 2 files
  - Committed with message
  - Created annotated tag

**Next**: Push to remote when ready:
```bash
git push origin feature/t05-oos-validation
git push origin exp/t05_oos_validation/v1
```
```

### 阻塞输出 (BLOCK Level)

```markdown
## ⚠️ Git Ops - Manual Execution Required

**Confirm Level**: BLOCK (data/ changes detected)

**Commands to execute**:
```bash
git add data/processed/features.parquet
git commit -m "data(processed): update features"
git push origin main
```

**Reason**: Data files require explicit human confirmation.
```

## EXAMPLE: After /dgsf_execute

```markdown
User: Execute complete - added dropout to SDF model

Copilot:
1. Check git status
   → Found: kernel/sdf/model.py modified

2. Generate commit plan
   → Type: feat(kernel)
   → Level: CONFIRM

3. Output plan for review:

## 📦 Git Commit Plan

**Branch**: `feature/sdf-dropout`
**Changes**:
  - [K] `kernel/sdf/model.py`

**Commit Message**:
feat(kernel): add dropout to SDF model

Added nn.Dropout(0.3) to SDFModel forward pass.

**Proceed? [Y/n]**
```

## EXAMPLE: After /dgsf_state_update "Complete"

```markdown
User: ✅ Complete: t05_oos_validation

Copilot:
1. Check git status
   → Found: experiments/t05_oos_validation/results.json added

2. Generate commit plan with auto-tag
   → Type: experiment(experiments)
   → Level: NOTIFY
   → Tag: exp/t05_oos_validation/20260204-143022

3. Execute and notify:

## ✅ Git Ops Complete

**Commit**: `def5678` experiment(experiments): t05_oos_validation complete
**Tag**: `exp/t05_oos_validation/20260204-143022`
  - OOS Sharpe: 1.67
  - OOS/IS Ratio: 0.94

Automatically committed (NOTIFY level).
```

## ERROR HANDLING

### Git 仓库不存在

```markdown
## ❌ Git Ops Failed

**Error**: Not inside a git repository
**Location**: {current_path}
**Resolution**: Initialize git repo or navigate to correct directory

```bash
git init
# or
cd /path/to/repo
```
```

### Commit 失败

```markdown
## ❌ Git Ops Failed

**Error**: nothing to commit, working tree clean
**Diagnosis**: All changes already committed or discarded
**Next**: Verify working tree state with `git status`
```

### Merge 冲突

```markdown
## ❌ Git Ops Blocked

**Error**: Merge conflict detected
**Files**:
  - kernel/os.py (both modified)

**Resolution**:
1. Resolve conflicts manually
2. `git add kernel/os.py`
3. Re-run `/dgsf_git_ops`
```

## INTEGRATION HOOKS

此 prompt 被以下 skills 自动调用：

```
/dgsf_execute (POST-FLIGHT 后)
    ↓
    IF working tree dirty → /dgsf_git_ops
    
/dgsf_state_update (记录 Complete/Milestone 后)
    ↓
    IF type == "complete" → /dgsf_git_ops with auto_tag=True
    IF type == "milestone" → /dgsf_git_ops with tag_prefix="milestone"
```

## KERNEL MODULE

此 prompt 由 `kernel/git_ops.py` 提供底层实现：

```python
from kernel.git_ops import (
    get_git_status,
    generate_commit_plan,
    execute_plan,
    run_git_ops_workflow,
    ConfirmLevel,
)

# 完整工作流
plan, result, output = run_git_ops_workflow(
    trigger_context="dgsf_execute complete",
    task_id="TASK-001",
    auto_tag=True,
    tag_prefix="exp",
    experiment_metrics={"oos_sharpe": 1.67, "oos_is_ratio": 0.94},
    dry_run=False,
)
```
