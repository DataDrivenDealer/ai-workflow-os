---
description: Execute Pair Programming Review loop (Coder ↔ Reviewer)
mode: agent
triggers:
  - "代码审核"
  - "pair review"
  - "code review"
  - "/dgsf_pair_review"
inherits_rules: [R1, R2, R3]
---

# DGSF Pair Programming Review

> **目的**: 在测试/回测之前强制执行代码审核循环
> **Gate**: Gate-E4.5 "NO REVIEW, NO RUN"
> **输出**: `docs/reviews/{task_id}/REVIEW_1.md`, `PATCH_1.md`, `REVIEW_2.md`

---

## ⚠️ 核心原则

```
CHEAP CHECKS FIRST. EXPENSIVE RUNS LAST.

代码审核 (5 min) ≪ 单元测试 (2 min) ≪ 回测 (30+ min)
```

**强制顺序**:
```
Code Change → REVIEW → PATCH → REVIEW → [APPROVED] → Test/Backtest
                 └─────────┴─────────┘
                      循环直到通过
```

---

## 📥 INPUTS

```yaml
task_id: string           # 当前任务 ID (required)
changed_files: list       # 变更的文件路径列表 (required)
spec_pointers: list       # 相关 Spec 文件 (optional)
coder_id: string          # Coder 代理标识 (auto-detected)
```

---

## 🔄 REVIEW LOOP PROTOCOL

### Phase 1: CODER 提交审核

**Coder** 完成代码更改后，准备审核请求：

```markdown
## 📝 Review Request

**Task**: {task_id}
**Changed Files**:
- [path/to/file1.py](path/to/file1.py) - 新增 XX 功能
- [path/to/file2.py](path/to/file2.py) - 修复 YY 问题

**Change Summary**:
简要描述本次更改的目的和主要内容。

**Self-Check**:
- [ ] 代码通过 lint (ruff/flake8)
- [ ] 无明显类型错误 (pyright)
- [ ] 遵循项目编码规范
- [ ] 无 hardcoded 路径或凭证

**Focus Areas** (请 Reviewer 重点关注):
- Area 1: ...
- Area 2: ...
```

---

### Phase 2: REVIEWER 审核

**Reviewer** 执行以下检查并生成 `REVIEW_1.md`:

#### 审核维度

| 维度 | 检查项 | Severity |
|------|--------|----------|
| **Correctness** | 逻辑正确性、边界条件 | CRITICAL |
| **Spec Compliance** | 是否符合 Spec 定义 | HIGH |
| **Quant Risk** | lookahead, leakage, survivorship | CRITICAL |
| **Code Quality** | 可读性、DRY、命名 | MEDIUM |
| **Performance** | 明显性能问题 | MEDIUM |
| **Security** | 注入、凭证暴露 | CRITICAL |

#### 输出模板: REVIEW_1.md

```markdown
---
task_id: "{task_id}"
reviewer_id: "{reviewer_agent_id}"
review_round: 1
timestamp: "{ISO8601}"
verdict: "NEEDS_CHANGES" | "APPROVED" | "BLOCKED"
---

## Summary

简要描述审核结论和整体评价。

## Issues Found

| # | Severity | File | Line | Description | Suggestion |
|---|----------|------|------|-------------|------------|
| 1 | CRITICAL | path/file.py | 45 | Look-ahead 问题 | 使用 shift(1) |
| 2 | HIGH | path/file.py | 78 | 未处理空值 | 添加 .fillna() |
| 3 | MEDIUM | path/file.py | 12 | 变量命名不清 | 改为 feature_count |

## Evidence

- [path/file.py#L45](path/file.py#L45) - 发现 `df['future_price']` 使用
- [Spec: SDF_SPEC.md#L120](specs/SDF_SPEC.md#L120) - 规范要求

## Verdict

**{NEEDS_CHANGES / APPROVED / BLOCKED}**

{If NEEDS_CHANGES}: 请 Coder 修复 Issue #1, #2 后提交 PATCH_1.md
{If BLOCKED}: 存在架构级问题，需返回 PLAN MODE 重新设计
```

**保存到**: `docs/reviews/{task_id}/REVIEW_1.md`

---

### Phase 3: CODER 响应

**Coder** 根据 REVIEW 反馈修改代码并生成 `PATCH_1.md`:

```markdown
---
task_id: "{task_id}"
coder_id: "{coder_agent_id}"
patch_round: 1
timestamp: "{ISO8601}"
addresses_review: 1
---

## Changes Made

| # | Issue Addressed | File | Change Summary |
|---|-----------------|------|----------------|
| 1 | REVIEW_1 #1 | path/file.py | 添加 .shift(1) 避免 look-ahead |
| 2 | REVIEW_1 #2 | path/file.py | 添加 .fillna(0) 处理空值 |
| 3 | REVIEW_1 #3 | path/file.py | 重命名为 feature_count |

## Files Modified

- [path/file.py](path/file.py#L45-L80) - 修复 look-ahead 和空值处理

## Notes for Reviewer

所有 CRITICAL/HIGH issues 已修复。
Issue #3 (MEDIUM) 已顺带处理。
```

**保存到**: `docs/reviews/{task_id}/PATCH_1.md`

---

### Phase 4: REVIEWER 复核

**Reviewer** 验证修复并生成 `REVIEW_2.md`:

```markdown
---
task_id: "{task_id}"
reviewer_id: "{reviewer_agent_id}"
review_round: 2
timestamp: "{ISO8601}"
verdict: "APPROVED"
---

## Summary

所有 CRITICAL/HIGH issues 已正确修复。代码质量符合标准。

## Issues Verification

| # | Original Issue | Status | Verification |
|---|---------------|--------|--------------|
| 1 | Look-ahead | ✅ FIXED | 确认使用 .shift(1) |
| 2 | 空值处理 | ✅ FIXED | 确认添加 .fillna() |
| 3 | 命名不清 | ✅ FIXED | 已改为 feature_count |

## New Issues

None.

## Verdict

**APPROVED**

Gate-E4.5 通过，可继续执行测试/回测。
```

**保存到**: `docs/reviews/{task_id}/REVIEW_2.md`

---

## ⛔ BLOCKED 处理

如果任何 REVIEW 的 verdict 为 `BLOCKED`:

```
INVOKE /dgsf_escalate WITH:
    type: "review_blocked"
    severity: "high"
    evidence: "docs/reviews/{task_id}/REVIEW_N.md"
    description: "代码审核发现架构级问题，需重新设计"
```

**不可继续执行**，必须返回 PLAN MODE。

---

## 🔗 与 Gate-E4.5 集成

EXECUTE MODE 中的 Gate-E4.5 检查逻辑：

```python
# Pseudocode for Gate-E4.5 check
def check_gate_e4_5(task_id: str, changed_files: list) -> bool:
    if not changed_files:
        return True  # No code changes, skip gate
    
    review_path = f"docs/reviews/{task_id}/REVIEW_2.md"
    
    if not file_exists(review_path):
        # Check for higher round reviews
        for n in range(10, 1, -1):
            alt_path = f"docs/reviews/{task_id}/REVIEW_{n}.md"
            if file_exists(alt_path):
                review_path = alt_path
                break
        else:
            return False  # No approved review found
    
    review = parse_frontmatter(review_path)
    return review.get("verdict") == "APPROVED"
```

---

## 🤖 自动化辅助

### 使用 Subagent 辅助审核

可选择调用 `quant_risk_review` subagent 作为 Reviewer 的辅助工具：

```bash
python kernel/subagent_runner.py quant_risk_review \
    --files "{changed_files}" \
    --review-type focused \
    --focus-areas lookahead leakage protocol
```

Subagent 输出可作为 REVIEW 的 Evidence 部分引用。

### Pre-Commit Hook 集成

`hooks/pre-commit` 会检查：
- 是否存在 `docs/reviews/{task_id}/REVIEW_2.md`
- `verdict` 是否为 `APPROVED`

如未通过，commit 将被阻止（可使用 `--no-verify` bypass，但会被审计）。

---

## 📋 EXAMPLE: 完整循环

```
User: "完成 T3.1 特征盘点的代码"

[Coder 完成代码更改]

Copilot:
> 代码更改已完成。根据 Gate-E4.5，需要进行 Pair Programming Review。
> 
> ## 📝 Review Request
> 
> **Task**: SDF_FEATURE_ENG_001_T3.1
> **Changed Files**:
> - [projects/dgsf/repo/src/dgsf/feature_inventory.py](...)
> 
> [生成 REVIEW_1.md 请求...]

[Reviewer 角色审核]

Copilot (as Reviewer):
> ## REVIEW_1.md
> 
> **Verdict**: NEEDS_CHANGES
> 
> | # | Severity | Issue |
> |---|----------|-------|
> | 1 | HIGH | 缺少类型注解 |
> 
> [保存到 docs/reviews/SDF_FEATURE_ENG_001_T3.1/REVIEW_1.md]

[Coder 修复]

Copilot (as Coder):
> 已修复 Issue #1。
> 
> [保存 PATCH_1.md]

[Reviewer 复核]

Copilot (as Reviewer):
> ## REVIEW_2.md
> 
> **Verdict**: APPROVED ✅
> 
> Gate-E4.5 通过，继续执行测试...
```

---

## 📊 METRICS

每次 Review 循环完成后，更新 `docs/state/COMPLIANCE_METRICS.md`:

```markdown
| Date | Task ID | Review Rounds | Final Verdict | Duration |
|------|---------|---------------|---------------|----------|
| 2026-02-05 | T3.1 | 2 | APPROVED | 15 min |
```

---

*Gate-E4.5 Pair Programming Review — Cheap Checks First*
