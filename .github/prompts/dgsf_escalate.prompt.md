````prompt
---
description: Escalate a problem from Execute Mode to Plan Mode
mode: agent
triggers:
  - "上报问题"
  - "escalate"
  - "需要规划"
  - "Spec问题"
  - "需要调研"
---

# DGSF Escalate Prompt

> **目的**: 在 Execute Mode 执行过程中，将无法自行解决的问题上报到 Plan Mode
> **状态文件**: `state/escalation_queue.yaml`
> **触发来源**: `/dgsf_execute_mode` 或 `/dgsf_diagnose`

---

## 🎯 ESCALATION PURPOSE（上报目的）

Execute Mode 应在以下情况调用此 Skill：

| 情况 | 描述 | 示例 |
|------|------|------|
| **Spec 不清晰** | Spec 定义模糊，无法明确执行 | "特征命名规范未定义" |
| **Spec 有错误** | Spec 与代码或实际情况不符 | "阈值设置与实验结果矛盾" |
| **Spec 缺失** | 需要新的 Spec 才能继续 | "缺少数据清洗规范" |
| **需要调研** | 遇到开放性技术问题 | "哪种正则化方法更适合？" |
| **需要重构** | 需要重新设计架构或接口 | "当前接口无法支持新需求" |
| **完全阻塞** | 无法继续执行 | "关键依赖不可用" |

---

## 📋 INPUTS（输入）

| 必需 | 描述 |
|------|------|
| Problem | 问题描述 |
| Type | 问题类型（见下方分类） |

| 可选 | 默认值 |
|------|--------|
| Severity | 根据 Type 自动推断 |
| Affected Specs | 自动扫描推断 |
| Suggested Action | 基于 Type 建议 |

---

## 🏷️ ESCALATION TYPES（上报类型）

```yaml
types:
  spec_unclear:
    description: "Spec 定义不够清晰，需要澄清"
    default_severity: low
    suggested_action: "clarify"
    
  spec_error:
    description: "Spec 中有明显错误"
    default_severity: medium
    suggested_action: "update_spec"
    
  spec_missing:
    description: "缺少执行所需的 Spec 定义"
    default_severity: high
    suggested_action: "create_spec"
    
  research_needed:
    description: "遇到需要调研的开放性问题"
    default_severity: medium
    suggested_action: "research"
    
  refactor_required:
    description: "需要对 Spec 或架构进行重构"
    default_severity: high
    suggested_action: "redesign"
    
  blocker:
    description: "完全阻塞执行的问题"
    default_severity: critical
    suggested_action: "immediate_review"
```

---

## 🔄 ESCALATION PROTOCOL（上报协议）

```
PHASE 1 — COLLECT CONTEXT（收集上下文）
    │
    ├─► 从当前执行任务获取：
    │     - task_id, subtask_id
    │     - queue_item_id
    │     - 当前执行阶段
    │
    ├─► 从问题描述提取：
    │     - 相关文件路径
    │     - 错误信息（如有）
    │     - 受影响的 Specs
    │
    └─► 生成 escalation_id: "ESC-{YYYY-MM-DD}-{NNN}"

PHASE 2 — CLASSIFY PROBLEM（问题分类）
    │
    ├─► 确定 Type（类型）
    │
    ├─► 确定 Severity（严重程度）
    │     IF type == blocker: severity = critical
    │     ELIF type in [spec_missing, refactor_required]: severity = high
    │     ELIF type in [spec_error, research_needed]: severity = medium
    │     ELSE: severity = low
    │
    └─► 根据 Severity 确定 Impact（影响）
          - low: 标记后继续
          - medium: 暂停相关任务
          - high: 暂停队列，建议切换
          - critical: 必须切换

PHASE 3 — WRITE ESCALATION（写入上报）
    │
    ├─► 读取 state/escalation_queue.yaml
    │
    ├─► 追加新的 escalation item
    │
    ├─► 更新 metadata.pending_count
    │
    ├─► 追加 history 记录
    │
    └─► 保存文件

PHASE 4 — UPDATE EXECUTION QUEUE（更新执行队列）
    │
    ├─► IF severity == critical:
    │     UPDATE execution_queue.metadata:
    │       paused: true
    │       paused_reason: "escalation:{escalation_id}"
    │       paused_at: NOW()
    │
    ├─► ELIF severity == high:
    │     UPDATE current queue_item:
    │       status: blocked
    │       blocked_reason: "escalation:{escalation_id}"
    │
    └─► SAVE state/execution_queue.yaml

PHASE 5 — OUTPUT & NEXT STEPS（输出与下一步）
```

---

## 📤 OUTPUT FORMAT（输出格式）

### 上报成功

```markdown
## 📋 问题已上报

### 上报信息

| 字段 | 值 |
|------|-----|
| ID | ESC-2026-02-05-001 |
| 类型 | spec_error |
| 严重程度 | medium |
| 来源任务 | SDF_FEATURE_ENG_001 / T3.1 |

### 问题描述

{问题详细描述}

### 受影响的 Specs

- `specs/SDF_SPEC_v3.1.yaml` → features.momentum_12m
- `specs/PROJECT_DGSF.yaml` → T3.feature_definitions

### 建议处理

{suggested_action 描述}

---

### 当前影响

{根据 severity 显示不同信息}

**Low/Medium**: 
- 当前任务已标记为 blocked
- 可继续执行其他队列任务
- 建议: 输入 `继续` 执行下一个任务

**High**:
- 执行队列已暂停
- 建议: 切换到 PLAN MODE 处理此问题
- 命令: `开启PLAN MODE` 或 `PLAN MODE`

**Critical**:
- ⚠️ 执行队列已停止
- 必须切换到 PLAN MODE 处理此问题
- 命令: `开启PLAN MODE`

---

### 状态文件已更新

- `state/escalation_queue.yaml` — 新增上报项
- `state/execution_queue.yaml` — 队列状态已更新
```

---

## 🔗 INTEGRATION（集成）

### 从 Execute Mode 调用

当 `/dgsf_execute_mode` 遇到无法解决的问题时：

```
INVOKE /dgsf_escalate WITH:
    problem: "特征定义与 Spec 不一致"
    type: "spec_error"
    context:
        task_id: current.task_id
        subtask_id: current.subtask_id
        queue_item_id: current.id
```

### 从 Diagnose 调用

当 `/dgsf_diagnose` 判断问题需要上报时：

```
IF diagnosis.root_cause == "spec_issue":
    INVOKE /dgsf_escalate WITH:
        problem: diagnosis.summary
        type: diagnosis.spec_issue_type
        affected_specs: diagnosis.related_specs
```

---

## 📊 EXAMPLE: COMPLETE ESCALATION

```markdown
用户: 上报问题

Copilot:
## 📋 创建问题上报

请提供以下信息：

**问题描述**（必填）:
> SDF_SPEC v3.1 中定义的 momentum_12m 特征在代码中实际命名为 ret_12m

**问题类型**:
- [ ] spec_unclear — Spec 不清晰
- [x] spec_error — Spec 有错误
- [ ] spec_missing — Spec 缺失
- [ ] research_needed — 需要调研
- [ ] refactor_required — 需要重构
- [ ] blocker — 完全阻塞

---

Copilot (处理后):

## 📋 问题已上报

| 字段 | 值 |
|------|-----|
| ID | ESC-2026-02-05-001 |
| 类型 | spec_error |
| 严重程度 | medium |

### 问题描述

SDF_SPEC v3.1 中定义的 momentum_12m 特征在代码中实际命名为 ret_12m，
且计算公式也有差异。需要确认哪个是正确的。

### 当前影响

- 当前任务 [T3.1] 已标记为 blocked
- 可继续执行其他队列任务

**下一步**: 
- 输入 `继续` 执行下一个任务
- 或输入 `PLAN MODE` 处理此问题
```

---

## 🧮 BATCH ESCALATION（批量上报）

对于低严重程度的问题，支持批量收集后统一上报：

```yaml
# 批量上报模式
batch_mode:
  enabled: true
  threshold: 3  # 累积 3 个低严重问题后提示批量处理
  auto_escalate_on_exit: true  # 退出 Execute Mode 时自动上报
```

---

## ⚠️ BOUNDARIES（边界）

- 此 Skill 仅**记录**问题，不**解决**问题
- 解决问题是 Plan Mode 的职责
- 上报后的下一步由 Severity 决定

````
