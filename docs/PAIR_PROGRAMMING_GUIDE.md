# Pair Programming 使用指南

> AI Workflow OS 代码审核工作流指南

---

## 概述

Pair Programming 功能在代码生成后自动触发代码审核，模拟两位开发专家的协作：

- **Builder Agent** - 编写代码的 AI 代理
- **Reviewer Agent** - 审核代码的 AI 代理（必须是不同的代理，禁止自审）

## 快速开始

### 1. 创建支持 Review 的 TaskCard

使用 `templates/TASKCARD_WITH_REVIEW.md` 模板：

```powershell
# 复制模板
cp templates/TASKCARD_WITH_REVIEW.md tasks/MY_TASK_001.md

# 编辑任务卡
# 填写 task_id, requirements, acceptance criteria 等
```

### 2. 启动任务

```powershell
python kernel/os.py task start MY_TASK_001
```

### 3. Builder 提交代码审核

当代码编写完成后，Builder Agent 通过 MCP 工具提交审核：

```json
{
  "tool": "review_submit",
  "arguments": {
    "session_token": "<builder_session_token>",
    "task_id": "MY_TASK_001",
    "artifact_paths": ["kernel/new_module.py", "kernel/tests/test_new_module.py"],
    "notes": "实现了核心功能，请重点审核安全性"
  }
}
```

任务状态变为 `code_review`。

### 4. Reviewer 开始审核

Reviewer Agent 创建审核会话：

```json
{
  "tool": "review_create_session",
  "arguments": {
    "session_token": "<reviewer_session_token>",
    "task_id": "MY_TASK_001",
    "personas": ["security_expert", "architecture_expert"]
  }
}
```

### 5. 获取审核提示

Reviewer 可以获取结构化的审核提示：

```json
{
  "tool": "review_get_prompts",
  "arguments": {
    "session_token": "<reviewer_session_token>",
    "task_id": "MY_TASK_001",
    "dimension": "all"
  }
}
```

返回四个维度的审核提示：Quality、Requirements、Completeness、Optimization。

### 6. 提交审核结果

Reviewer 完成审核后提交结果：

```json
{
  "tool": "review_conduct",
  "arguments": {
    "session_token": "<reviewer_session_token>",
    "review_session_id": "RSESS-MY_TASK_001-xxxxxxxx",
    "quality_issues": [
      {
        "check_id": "Q-006",
        "severity": "CRITICAL",
        "description": "SQL 注入漏洞",
        "file_path": "kernel/new_module.py",
        "line_start": 45,
        "suggested_fix": "使用参数化查询"
      }
    ],
    "requirements_issues": [],
    "completeness_issues": [],
    "optimization_suggestions": [
      {
        "check_id": "O-001",
        "description": "可以使用列表推导简化",
        "file_path": "kernel/new_module.py",
        "current_code": "result = []\\nfor x in items:\\n    result.append(x)",
        "suggested_code": "result = [x for x in items]",
        "rationale": "更 Pythonic，性能更好",
        "impact": "minor"
      }
    ],
    "requirements_coverage_pct": 100,
    "completeness_pct": 95
  }
}
```

### 7. 处理审核结果

**如果 Verdict = NEEDS_REVISION**：

Builder 需要修复问题并重新提交：

```json
{
  "tool": "review_respond",
  "arguments": {
    "session_token": "<builder_session_token>",
    "task_id": "MY_TASK_001",
    "action": "revision_submitted",
    "notes": "已修复 SQL 注入漏洞"
  }
}
```

然后重新走 review_submit → review_create_session → review_conduct 流程。

**如果 Verdict = APPROVED**：

Reviewer 批准代码：

```json
{
  "tool": "review_approve",
  "arguments": {
    "session_token": "<reviewer_session_token>",
    "task_id": "MY_TASK_001",
    "review_session_id": "RSESS-MY_TASK_001-xxxxxxxx",
    "final_notes": "代码质量优秀，批准合并"
  }
}
```

任务状态变为 `reviewing`（等待人工最终审批）。

---

## 审核维度详解

### Quality Check (Q-Check)

| Check ID | 检查项 | 默认严重度 |
|----------|--------|-----------|
| Q-001 | 语法正确性 | CRITICAL |
| Q-002 | 类型安全 | MAJOR |
| Q-003 | 错误处理 | MAJOR |
| Q-004 | 空值处理 | MAJOR |
| Q-005 | 资源清理 | MAJOR |
| Q-006 | 安全漏洞 | CRITICAL |
| Q-007 | 性能反模式 | MINOR |
| Q-008 | 代码重复 | MINOR |

### Requirements Check (R-Check)

| Check ID | 检查项 | 默认严重度 |
|----------|--------|-----------|
| R-001 | 功能需求满足 | CRITICAL |
| R-002 | 输入验证 | MAJOR |
| R-003 | 输出格式 | MAJOR |
| R-004 | 边界情况处理 | MAJOR |
| R-005 | 集成点 | MAJOR |
| R-006 | API 契约 | CRITICAL |

### Completeness Check (C-Check)

| Check ID | 检查项 | 默认严重度 |
|----------|--------|-----------|
| C-001 | 所有需求已覆盖 | CRITICAL |
| C-002 | 验收标准已满足 | MAJOR |
| C-003 | 引用规范已实现 | MAJOR |
| C-004 | 测试已包含 | MINOR |
| C-005 | 文档已更新 | MINOR |

### Optimization Check (O-Check)

| Check ID | 检查项 | 默认严重度 |
|----------|--------|-----------|
| O-001 | 代码可简化 | SUGGESTION |
| O-002 | 算法效率 | SUGGESTION |
| O-003 | 抽象改进 | SUGGESTION |
| O-004 | 命名清晰度 | SUGGESTION |
| O-005 | 语言习惯用法 | SUGGESTION |
| O-006 | 复杂度降低 | SUGGESTION |

---

## 专家角色 (Personas)

可选择的专家视角：

| Persona | 专长领域 | 关注 Check IDs |
|---------|---------|----------------|
| `security_expert` | OWASP, 注入, 认证, 数据保护 | Q-006, R-002 |
| `performance_expert` | 算法, 缓存, I/O 优化 | Q-007, O-002, O-006 |
| `architecture_expert` | SOLID, 设计模式, 耦合 | O-003, Q-008 |
| `domain_expert` | 业务逻辑, 需求对齐 | R-001, R-004, C-001 |
| `testing_expert` | 测试覆盖, 边界情况 | C-004, R-004 |

---

## 严重度级别

| 级别 | 含义 | 是否阻止合并 |
|------|------|------------|
| CRITICAL | 安全漏洞、数据丢失风险、崩溃 | ✅ 是 |
| MAJOR | 功能 Bug、缺失需求 | ✅ 是 |
| MINOR | 代码异味、小 Bug | ❌ 否 |
| SUGGESTION | 优化建议 | ❌ 否 |

---

## 判定逻辑

```python
if has_critical_issues or has_major_issues:
    verdict = "NEEDS_REVISION"
else:
    verdict = "APPROVED"
```

---

## 状态流转图

```
running
   │
   ├──[review_submit]──> code_review
   │                         │
   │                    [review_conduct]
   │                         │
   │                    ┌────┴────┐
   │                    │         │
   │             NEEDS_REVISION  APPROVED
   │                    │         │
   │                    v         v
   │            revision_needed   │
   │                    │         │
   │         [revision_submitted] │
   │                    │         │
   │                    v         │
   └─────────────> code_review    │
                                  │
                         [review_approve]
                                  │
                                  v
                              reviewing
                                  │
                          [human approval]
                                  │
                                  v
                               merged
```

---

## 配置选项

在 `kernel/state_machine.yaml` 的 `pair_programming` 部分：

```yaml
pair_programming:
  enabled: true
  self_review_prohibited: true
  max_revision_cycles: 3
  
  review_required_for:
    - "*.py"
    - "*.js"
    - "*.ts"
    - "*.yaml"
  
  review_optional_for:
    - "*.md"
    - "*.txt"
```

---

## CLI 命令

```powershell
# 查看任务审核状态
python kernel/os.py task status MY_TASK_001

# 查看审核报告（在 state/tasks.yaml 中）
cat state/tasks.yaml | grep -A 50 "reviews:"
```

---

## 最佳实践

1. **小批量提交** - 每次提交审核的代码量不宜过大
2. **明确需求** - TaskCard 中的需求要清晰具体
3. **选择合适的 Personas** - 根据代码类型选择专家视角
4. **及时响应** - 收到 NEEDS_REVISION 后尽快修复
5. **最多3轮** - 超过3轮修订需要人工介入

---

## 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| "Self-review prohibited" | 同一 Agent 试图审核自己的代码 | 使用不同的 Agent 进行审核 |
| "No submission pending" | 任务未提交审核 | 先调用 review_submit |
| "Cannot approve" | 存在未解决的 CRITICAL/MAJOR 问题 | 修复问题后重新审核 |
| "Session not found" | 审核会话已过期或不存在 | 重新创建审核会话 |

---

*AI Workflow OS - Pair Programming v0.1.0*
