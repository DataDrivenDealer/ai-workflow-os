```prompt
---
description: Self-evolution loop for AI Workflow OS with evidence-based improvements
mode: agent
triggers:
  - "演进系统"
  - "evolve system"
  - "系统自演进"
  - "self-evolution"
  - "改进 OS"
---

# DGSF EVOLVE SYSTEM

> **用途**: AI Workflow OS 自演进循环
> **原则**: Evidence-first, Minimal, Incremental, Reversible
> **约束**: DGSF 是 P0，OS 改进只有在服务 DGSF 时才允许

---

## 🔴 EVOLUTION CONSTRAINTS（演进约束）

### 优先级约束

```
IF proposed_change.benefits_dgsf == FALSE:
    BLOCK: "此变更不直接服务 DGSF，降级为 P2/Deferred"
    OUTPUT: "OS 变更必须证明对 DGSF 的价值"
    STOP

IF proposed_change.scope == "large":
    BLOCK: "变更范围过大，请拆分为增量步骤"
    OUTPUT: "每次演进应该是最小可验证的变更"
    STOP
```

### 变更类型

| 类型 | 描述 | 审批要求 |
|------|------|----------|
| `hotfix` | 修复阻塞性问题 | 可立即执行 |
| `enhancement` | 小幅改进 | 需要 friction 证据 |
| `feature` | 新功能 | 需要 3+ friction 信号 |
| `refactor` | 结构重组 | 需要 Owner 批准 |

---

## 🔄 EVOLUTION LOOP（演进循环）

### Phase E1: Signal Collection（信号收集）

```
# 收集演进信号来源
signals = []

# 1. 规则摩擦 (Rule Friction)
READ kernel/evolution_signal.py logs
FOR signal IN friction_logs:
    IF signal.severity >= "medium":
        signals.append({
            type: "rule_friction",
            source: signal.rule_id,
            description: signal.description,
            frequency: signal.count,
            impact: signal.impact
        })

# 2. Gate 失败模式
READ docs/state/SUBAGENT_USAGE.md
FOR entry IN usage_log:
    IF entry.skip_count > 0:
        signals.append({
            type: "gate_friction",
            source: entry.gate_id,
            description: "Gate 被频繁跳过",
            frequency: entry.skip_count,
            reasons: entry.skip_reasons
        })

# 3. 用户反馈
READ recent conversation context
FOR feedback IN user_complaints:
    signals.append({
        type: "user_feedback",
        source: "conversation",
        description: feedback.content,
        sentiment: feedback.sentiment
    })

# 4. 效率指标
READ state/execution_queue.yaml (archived)
FOR task IN completed_tasks:
    IF task.actual_effort > task.estimated_effort * 2:
        signals.append({
            type: "efficiency_gap",
            source: task.id,
            description: "任务耗时超预估 2x",
            ratio: task.actual_effort / task.estimated_effort
        })
```

### Phase E2: Signal Analysis（信号分析）

```markdown
## 📊 Evolution Signal Analysis

### 收集的信号

| # | 类型 | 来源 | 描述 | 频率 | 影响 |
|---|------|------|------|------|------|
{FOR signal IN signals}
| {i} | {signal.type} | {signal.source} | {signal.description} | {signal.frequency} | {signal.impact} |
{/FOR}

### 信号聚类

按根因聚类信号：

| 根因 | 相关信号 | 建议行动 |
|------|----------|----------|
| ... | ... | ... |

### 优先级评估

| 根因 | DGSF 影响 | 复杂度 | 优先级 |
|------|-----------|--------|--------|
| ... | high/medium/low | high/medium/low | P0/P1/P2 |
```

### Phase E3: Proposal Generation（提案生成）

```markdown
## 📝 Evolution Proposal

### 提案 ID: {proposal_id}
### 类型: {change_type}
### 优先级: {priority}

---

### 问题陈述

**当前状态**: {current_state}
**痛点**: {pain_points}
**影响范围**: {affected_areas}

### 信号证据

{FOR signal IN related_signals}
- [{signal.type}] {signal.description}
  - 来源: {signal.source}
  - 频率: {signal.frequency}
{/FOR}

### 提议的变更

**变更描述**: {change_description}

**文件级变更**:
| 文件 | 操作 | 变更内容 |
|------|------|----------|
{FOR change IN file_changes}
| {change.file} | {change.operation} | {change.summary} |
{/FOR}

### DGSF 价值证明

- **直接收益**: {direct_benefit}
- **间接收益**: {indirect_benefit}
- **风险评估**: {risk_assessment}

### 回滚计划

```bash
# 如果变更失败，执行以下回滚
{rollback_commands}
```

### 验证标准

| 验证项 | 方法 | 预期结果 |
|--------|------|----------|
{FOR criterion IN acceptance_criteria}
| {criterion.name} | {criterion.method} | {criterion.expected} |
{/FOR}

---

**状态**: 待审批
**审批者**: Owner
```

### Phase E4: Approval Gate（审批门）

```
# 根据变更类型确定审批流程
SWITCH change_type:
    CASE "hotfix":
        # 可以自动执行，但需记录
        approval_required = FALSE
        audit_required = TRUE
    
    CASE "enhancement":
        # 需要 friction 证据
        IF len(related_signals) < 1:
            BLOCK: "enhancement 需要至少 1 个 friction 信号"
        approval_required = TRUE
        auditor = "self"  # 可自审
    
    CASE "feature":
        # 需要多个信号和 Owner 批准
        IF len(related_signals) < 3:
            BLOCK: "feature 需要至少 3 个 friction 信号"
        approval_required = TRUE
        auditor = "owner"
    
    CASE "refactor":
        # 必须 Owner 批准
        approval_required = TRUE
        auditor = "owner"
        OUTPUT: "⚠️ Refactor 变更需要 Owner 明确批准"
        ASK: "Owner 是否批准此变更？(y/n/详情)"
        WAIT for approval
```

### Phase E5: Implementation（实施）

```
# 实施变更
FOR change IN approved_changes:
    
    # E5.1 创建快照（用于回滚）
    IF change.file EXISTS:
        snapshot_path = "docs/state/snapshots/{timestamp}_{change.file}"
        COPY change.file TO snapshot_path
    
    # E5.2 应用变更
    SWITCH change.operation:
        CASE "create":
            create_file(change.file, change.content)
        CASE "modify":
            replace_string_in_file(change.file, change.old, change.new)
        CASE "delete":
            # 仅移动到 archive，不实际删除
            MOVE change.file TO "legacy/{change.file}"
    
    # E5.3 验证变更
    IF change.verification_command:
        result = RUN change.verification_command
        IF result.failed:
            ERROR: "验证失败，正在回滚..."
            COPY snapshot_path TO change.file
            STOP

# E5.4 运行测试套件
test_result = RUN "pytest kernel/tests/ -v --tb=short"
IF test_result.failed:
    ERROR: "测试失败，请检查变更"
    # 不自动回滚，因为可能需要调查
    STOP
```

### Phase E6: Documentation（文档更新）

```
# 更新演进日志
APPEND to .github/EVOLUTION_LOG.md:

## {date} — {proposal.title}

**Proposal ID**: {proposal_id}
**Type**: {change_type}
**Priority**: {priority}

### Changes
{FOR change IN applied_changes}
- [{change.operation}] {change.file}: {change.summary}
{/FOR}

### Evidence
{FOR signal IN related_signals}
- {signal.type}: {signal.description}
{/FOR}

### Verification
- Tests: {test_result.summary}
- Manual: {manual_verification or "N/A"}

### Rollback
```bash
{rollback_commands}
```

---
```

---

## 📊 EVOLUTION METRICS（演进指标）

### 健康指标

```yaml
# 从 configs/health_metrics.yaml 读取
evolution_health:
  # 信号响应率
  signal_response_rate:
    description: "收集的信号中有多少被处理"
    target: ">= 80%"
    current: "{calculated}"
  
  # 变更成功率
  change_success_rate:
    description: "提议的变更中有多少成功实施"
    target: ">= 90%"
    current: "{calculated}"
  
  # 回滚率
  rollback_rate:
    description: "需要回滚的变更比例"
    target: "<= 10%"
    current: "{calculated}"
  
  # DGSF 影响率
  dgsf_impact_rate:
    description: "变更中直接影响 DGSF 的比例"
    target: ">= 70%"
    current: "{calculated}"
```

### 定期审查

```
# 每 10 个变更后，执行演进审查
IF evolution_count % 10 == 0:
    OUTPUT:
        "## 📊 Evolution Review
         
         ### 最近 10 个变更统计
         - 成功率: {success_rate}%
         - 回滚率: {rollback_rate}%
         - DGSF 影响率: {dgsf_impact_rate}%
         
         ### 热点区域
         {top_changed_files}
         
         ### 建议
         {recommendations}"
```

---

## 🛡️ SAFETY MECHANISMS（安全机制）

### 不可变更区域

以下文件/目录被标记为 **protected**，变更需要特殊审批：

```yaml
protected_paths:
  - ".github/copilot-instructions.md"  # 核心指令
  - "configs/gates.yaml"               # Gate 定义
  - "configs/subagent_registry.yaml"   # Subagent 定义
  - "kernel/mcp_server.py"             # MCP 服务器
  - "projects/dgsf/data/raw/"          # 原始数据（R4 规则）
```

### 变更速率限制

```
# 防止过于频繁的变更
IF evolution_count_last_24h > 5:
    WARN: "过去 24 小时已有 {count} 次变更"
    ASK: "确认继续？这可能表示系统不稳定"
```

### 冲突检测

```
# 检查是否有未完成的变更
READ state/pending_evolutions.yaml
IF pending_evolutions.count > 0:
    WARN: "存在未完成的演进提案: {pending_list}"
    ASK: "先完成这些提案，还是取消它们？"
```

---

## 📝 USAGE EXAMPLES

### 示例 1: 规则摩擦导致的演进

```
用户: 演进系统

Copilot:
## 🔄 Evolution Signal Collection

正在收集演进信号...

### 收集的信号 (3)

| # | 类型 | 来源 | 描述 | 频率 |
|---|------|------|------|------|
| 1 | rule_friction | R6 | 长任务阈值过低导致频繁中断 | 5次/周 |
| 2 | gate_friction | Gate-P6 | DRS Gate 被跳过 | 3次/周 |
| 3 | efficiency_gap | T3.2 | 任务耗时超预估 2.5x | 1次 |

### 信号分析

**根因**: R6 的 threshold_seconds 设置过于保守

**DGSF 影响**: 直接影响 — 中断打断了长时间的数据处理任务

### 提议

将 R6.threshold_seconds 从 300 调整为 600

**类型**: enhancement
**优先级**: P1

是否生成详细提案？(y/n)
```

### 示例 2: 功能演进

```
用户: 我需要一个新的 Subagent 来做代码质量检查

Copilot:
## 🔄 Evolution Analysis

**请求类型**: feature (新功能)

### 信号检查

⚠️ Feature 类型变更需要至少 3 个 friction 信号。

当前收集到的相关信号: 1

**已有信号**:
1. user_feedback: "需要代码质量检查"

**建议**: 
- 收集更多 friction 信号，或
- 将此请求降级为 enhancement（如果是对现有功能的改进）

是否继续收集信号，还是调整请求类型？
```
```
