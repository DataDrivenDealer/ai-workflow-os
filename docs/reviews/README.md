# Pair Programming Review Artifacts

> 代码审核工作流产出物存储目录

---

## 目录结构

```
docs/reviews/
  {task_id}/
    REVIEW_1.md      # Reviewer 第一轮反馈
    PATCH_1.md       # Coder 响应与补丁摘要
    REVIEW_2.md      # Reviewer 最终审批（或阻塞）
    [REVIEW_N.md]    # 可选：多轮审核
```

## Artifact Schema

所有 review artifact 必须遵循 `configs/review_artifact_schema.yaml` 定义的格式。

### REVIEW_*.md 格式

```yaml
---
task_id: "TASK-001"
reviewer_id: "copilot-agent-xxxxx"
review_round: 1
timestamp: "2026-02-05T14:30:00Z"
verdict: "NEEDS_CHANGES" | "APPROVED" | "BLOCKED"
---

## Summary

简要描述审核结论。

## Issues Found

| # | Severity | File | Line | Description | Suggestion |
|---|----------|------|------|-------------|------------|
| 1 | CRITICAL | path/to/file.py | 45 | 描述问题 | 建议修复 |

## Evidence

- [path/to/file.py#L45](path/to/file.py#L45) - 问题证据

## Blockers (if verdict = BLOCKED)

- [ ] Blocker 1 description
- [ ] Blocker 2 description
```

### PATCH_*.md 格式

```yaml
---
task_id: "TASK-001"
coder_id: "copilot-agent-yyyyy"
patch_round: 1
timestamp: "2026-02-05T15:00:00Z"
addresses_review: 1
---

## Changes Made

| # | Issue Addressed | File | Change Summary |
|---|-----------------|------|----------------|
| 1 | Issue #1 from REVIEW_1 | path/to/file.py | 修复描述 |

## Files Modified

- [path/to/file.py](path/to/file.py) - 修改了第 45-50 行

## Notes for Reviewer

任何需要 Reviewer 特别注意的事项。
```

## Gate-E4.5 集成

在 EXECUTE MODE 中，Gate-E4.5 会检查：

1. 是否存在 `docs/reviews/{task_id}/REVIEW_2.md`
2. 该文件的 `verdict` 字段是否为 `APPROVED`

如果检查失败，执行将被阻塞，直到完成 Pair Programming Review 循环。

## Bypass 机制

如需紧急绕过（仅限 hotfix）：

```bash
# 使用 --no-verify 绕过 pre-commit hook
git commit --no-verify -m "hotfix: critical fix"
```

**注意**: Bypass 将被记录到 `docs/audits/bypasses.log`，需后续补齐 review artifact。

---

*Created: 2026-02-05 | Gate-E4.5 Pair Programming Review*
