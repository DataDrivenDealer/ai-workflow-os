# Audit Logs

> 治理审计日志存储目录

---

## 文件说明

| 文件 | 用途 | 自动生成 |
|------|------|----------|
| `bypasses.log` | 记录 `--no-verify` bypass 事件 | ✅ 由 pre-commit hook |
| `review_audits/` | 审核过程的完整记录 | 可选 |

## bypasses.log 格式

```
{ISO8601 timestamp} | BYPASS | {hook_name} | {commit_sha} | {user} | {reason}
```

**示例**:
```
2026-02-05T14:30:00+08:00 | BYPASS | pre-commit | abc1234 | developer | hotfix: urgent production issue
```

## Compliance 指标

Bypass 事件会被汇总到 `docs/state/COMPLIANCE_METRICS.md` 的周报中。

**目标**: Bypass 率 < 5%

## 补救流程

对于 bypass 的提交，必须在 48 小时内：

1. 补齐 `docs/reviews/{task_id}/REVIEW_2.md`
2. 在 `bypasses.log` 中标记 `REMEDIATED`
3. 或者提交 revert commit

---

*Created: 2026-02-05 | Governance Audit System*
