````prompt
---
description: Enter EXECUTE MODE - resume execution from planned queue with full traceability
mode: agent
triggers:
  - "执行模式"
  - "EXECUTE MODE"
  - "开始执行"
  - "继续执行"
  - "Switch to EXECUTE MODE"
inherits_rules: [R1, R2, R3, R4, R5, R6]
---

# DGSF EXECUTE MODE

> **模式**: 任务执行 / 验证 / 提交
> **状态文件**: `state/execution_queue.yaml`
> **目的**: 从 Plan Mode 规划的执行队列中恢复，按优先级顺序执行任务

---

## ⚙️ MODE CONFIGURATION（模式配置）

### Expert-Pattern Mode (Review-Only)

```yaml
EXPERT_PATTERN_MODE: REVIEW_ONLY  # 固定值，不可更改
```

**EXECUTE MODE 的 Expert-Pattern 限制**:
- ✅ **仅在 Review Gate 内启用** — 用于验证代码质量和风险检查
- ❌ **不可用于规划** — 不能基于 expert-pattern 分析结果重新规划
- ❌ **不可用于重新排序** — 不能改变执行队列的优先级
- ❌ **不可用于 Spec 修改** — 必须通过 /dgsf_escalate 返回 PLAN MODE

**Review Gate 内的 Expert-Pattern 使用场景**:
| 场景 | 允许 | 行动 |
|------|------|------|
| 检查代码是否符合 Spec | ✅ | 对比实现与规范 |
| 评估量化风险 | ✅ | lookahead/leakage 检查 |
| 发现设计缺陷 | ✅ | 记录并 escalate |
| 建议重新设计 | ❌ | 必须 escalate 到 PLAN MODE |

### Subagent Policy

```yaml
SUBAGENT_ALLOWED: YES  # YES | NO (仅限 Review Gate)
ALLOWED_SUBAGENTS:
  - repo_specs_retrieval  # 用于验证实现与 Spec 一致性
  - quant_risk_review     # 用于风险检查
PROHIBITED_SUBAGENTS:
  - external_research     # 禁止！会导致规划偏离
```

**Subagent 调用限制**:
| 检查项 | 通过条件 |
|--------|----------|
| 调用位置 | 仅在 PHASE 3: VERIFICATION 内 |
| 调用目的 | 证据收集或风险检查 |
| 结果使用 | 仅用于验证，不用于规划 |

**Subagent 结果的处理**:
```
IF subagent.result.verdict == "fail":
    # 不能自行决定修复方案
    INVOKE /dgsf_escalate WITH:
        type: "risk_review_failed"
        evidence: subagent.result
    # 等待 PLAN MODE 决策
    
IF subagent.result.verdict == "warn":
    # 记录警告，继续执行
    LOG warning to execution_queue.result.warnings
    CONTINUE
```

---

## 🔴 EXECUTE MODE HARD CONSTRAINTS（硬性约束）

| 约束 | 描述 | 违反后果 |
|------|------|----------|
| **NO PLANNING** | 不得规划新任务 | 违反 → HALT |
| **NO REPRIORITIZATION** | 不得重新排序执行队列 | 违反 → HALT |
| **NO SPEC CHANGES** | 不得修改 Spec 文件 | 违反 → HALT + Escalate |
| **NO TASK ADDITION** | 不得向队列添加任务 | 违反 → HALT |
| **EVIDENCE REQUIRED** | 任何断言必须有证据 | 违反 → Invalid Claim |

**如果发现需要违反这些约束**:
```
INVOKE /dgsf_escalate WITH:
    type: "constraint_conflict"
    description: "执行过程中发现需要重新规划..."
    suggested_action: "返回 PLAN MODE"
```

---

## 🔴 CORE RULES (from Kernel)

| Rule | Description | Enforcement |
|------|-------------|-------------|
| **R1** | Verify before asserting | 执行前必须确认前置条件 |
| **R2** | One task at a time | 不并行执行队列项 |
| **R3** | Stop on failure | 失败立即停止，调用诊断 |
| **R4** | Protect raw data | 永不写入 `data/raw/` |
| **R5** | No assumptions | 路径不确定时询问 |
| **R6** | Long-run handoff | 超时任务交由人工执行 |

---

## 🚀 ENTRY PROTOCOL（入口协议）

进入 EXECUTE MODE 时，**必须**按以下顺序执行：

### Step 1: 加载执行队列

```
READ state/execution_queue.yaml
IF file not exists OR queue is empty:
    OUTPUT:
        "📭 没有待执行任务。
         
         建议: 运行 PLAN MODE 规划下一批任务。
         命令: `开启PLAN MODE` 或 `PLAN MODE`"
    STOP
```

### Step 2: 显示队列摘要

```markdown
## 🎯 EXECUTE MODE 已激活

### 📋 执行队列状态

| # | Task ID | Subtask | 标题 | 优先级 | 状态 |
|---|---------|---------|------|--------|------|
| 1 | SDF_FEATURE_ENG_001 | T3.1 | 现有特征盘点 | P0 | ⏸️ pending |
| 2 | SDF_FEATURE_ENG_001 | T3.2 | 特征计算验证 | P0 | ⏸️ pending |
| 3 | SDF_TEST_FIX_001 | Phase2 | 测试修复 | P1 | ⏸️ pending |

---

**来源**: Plan Session @ 2026-02-05T11:15:00Z  
**规划摘要**: 诊断 T5.2 OOS 失败，规划 T3 特征工程

### 统计
- 总计: 3 | ⏸️ 待执行: 3 | ✅ 已完成: 0 | 🚫 阻塞: 0
```

### Step 3: 确认继续

```markdown
### 下一个任务

**[1] T3.1 现有特征盘点** (P0)

验收标准:
1. JSON 包含至少 10 个特征定义
2. 每个特征有 name, type, source, formula 字段
3. 标记 SDF_SPEC v3.1 覆盖状态

预估: 4 hours | Spec: tasks/active/SDF_FEATURE_ENG_001.md#T3.1

---

选择操作:
- `继续` / `y` — 开始执行此任务
- `跳过` — 跳过此任务，执行下一个
- `详情` — 查看完整任务定义
- `重新规划` — 返回 PLAN MODE
```

---

## 📋 EXECUTION LOOP（执行循环）

```
WHILE queue has pending items:
    current = get_next_pending_item(queue)
    
    # ========================================
    # GATE-E0: PRE-EXECUTION SUBAGENT CHECK（执行前 Subagent 检查）⭐
    # ========================================
    
    # E0.1 检查 RequiredSubagents
    IF current.required_subagents NOT empty:
        OUTPUT: "🔍 Gate-E0: 检查 RequiredSubagents..."
        
        FOR subagent_id IN current.required_subagents:
            # 检查是否已有输出
            IF current.subagent_artifacts contains subagent_id:
                artifact = get_artifact(subagent_id)
                IF file_exists(artifact.summary_path):
                    OUTPUT: "  ✅ {subagent_id}: 已完成 ({artifact.output_path})"
                    CONTINUE
            
            # 没有输出，必须调用
            OUTPUT: "  ⏳ {subagent_id}: 需要调用..."
            
            # 调用 Subagent
            RUN: python kernel/subagent_runner.py {subagent_id} \
                --question "验证任务 {current.title}" \
                --scope "{inferred from spec_pointers}"
            
            READ: docs/subagents/runs/latest/SUMMARY.md
            
            # 更新 artifacts
            APPEND to current.subagent_artifacts:
                - subagent_id: "{subagent_id}"
                  output_path: "docs/subagents/runs/latest/"
                  invoked_at: NOW()
            
            SAVE state/execution_queue.yaml
        
        OUTPUT: "✅ Gate-E0: 所有 RequiredSubagents 已满足"
    
    # E0.2 检查 SkipJustification
    IF current.required_subagents NOT empty AND current.subagent_artifacts IS empty:
        IF current.skip_justification IS null:
            OUTPUT: "⛔ Gate-E0 失败: RequiredSubagents 未完成且无 skip_justification"
            INVOKE /dgsf_escalate WITH:
                type: "gate_violation"
                gate: "Gate-E0"
                task: current
            BREAK
    
    # ========================================
    # PHASE 1: PRE-FLIGHT CHECK（预检）
    # ========================================
    
    # 1.1 检查依赖
    IF current.dependencies not empty:
        FOR dep_id IN current.dependencies:
            IF queue[dep_id].status != 'completed':
                OUTPUT: "⚠️ 前置任务 #{dep_id} 未完成，跳过此任务"
                CONTINUE to next item
    
    # 1.2 标记开始
    UPDATE queue[current.id]:
        status: in_progress
        started_at: NOW()
    SAVE state/execution_queue.yaml
    
    # 1.3 公告开始
    OUTPUT:
        "## 🚀 开始执行: {current.title}
         
         Task: {current.task_id} / {current.subtask_id}
         Priority: {current.priority}
         
         ### 验收标准
         {formatted acceptance_criteria}"
    
    # ========================================
    # PHASE 2: EXECUTION（执行）
    # ========================================
    
    # 2.1 调用 /dgsf_execute
    result = INVOKE /dgsf_execute WITH:
        task: current.title
        target: inferred from spec_pointers
        expected_outcome: current.acceptance_criteria
    
    # ========================================
    # GATE-E4.5: PAIR PROGRAMMING REVIEW GATE（代码审核门控）⭐ NEW
    # ========================================
    # 
    # 原则: CHEAP CHECKS FIRST, EXPENSIVE RUNS LAST
    # 代码审核 (~5 min) ≪ 单元测试 (~2 min) ≪ 回测 (~30+ min)
    #
    
    # E4.5.1 检测是否有代码变更
    changed_code_files = get_changed_files(extensions=[".py", ".ts", ".js"])
    
    IF changed_code_files NOT empty:
        OUTPUT: "🔍 Gate-E4.5: 检测到代码变更，检查 Pair Programming Review..."
        
        # E4.5.2 检查 REVIEW_2.md 是否存在
        review_path = "docs/reviews/{current.task_id}/REVIEW_2.md"
        
        # 也检查更高轮次的 review
        latest_review = find_latest_review(current.task_id)
        
        IF latest_review IS null:
            OUTPUT:
                "## ⏳ Gate-E4.5: 需要进行代码审核
                 
                 **Changed Files**:
                 {list changed_code_files}
                 
                 → 启动 Pair Programming Review 循环..."
            
            # 调用 pair review prompt
            INVOKE /dgsf_pair_review WITH:
                task_id: current.task_id
                changed_files: changed_code_files
                spec_pointers: current.spec_pointers
            
            # 重新检查
            latest_review = find_latest_review(current.task_id)
        
        # E4.5.3 验证 verdict
        IF latest_review IS null:
            OUTPUT: "⛔ Gate-E4.5 失败: 审核未完成"
            STOP
        
        READ latest_review
        
        IF review.verdict == "BLOCKED":
            OUTPUT:
                "## ⛔ Gate-E4.5 失败: 审核结果为 BLOCKED
                 
                 **Review Path**: {latest_review}
                 **Blockers**: {review.blockers}
                 
                 → 必须返回 PLAN MODE 重新设计"
            
            INVOKE /dgsf_escalate WITH:
                type: "review_blocked"
                severity: "high"
                evidence: latest_review
            BREAK
        
        IF review.verdict == "NEEDS_CHANGES":
            OUTPUT:
                "## ⚠️ Gate-E4.5: 审核需要修改
                 
                 **Review Path**: {latest_review}
                 **Issues**: {review.issues_count}
                 
                 → 请 Coder 修复后重新提交"
            STOP
        
        IF review.verdict == "APPROVED":
            OUTPUT: "✅ Gate-E4.5: 代码审核通过"
            
            # 记录到 subagent_artifacts
            APPEND to current.subagent_artifacts:
                - artifact_type: "pair_review"
                  output_path: latest_review
                  verdict: "APPROVED"
                  reviewed_at: NOW()
    
    ELSE:
        OUTPUT: "ℹ️ Gate-E4.5: 无代码变更，跳过审核"
    
    # ========================================
    # PHASE 3: VERIFICATION（验证）
    # ========================================
    
    # 3.0 [可选] 调用 Subagent 进行风险审查
    # ========================================
    # GATE-E5: RISK REVIEW GATE（风险审查门控）⭐
    # ========================================
    
    # E5.1 检查任务是否触发 Gate-E5
    task_touches_risk_areas = check_if_task_touches(current, ["backtest", "data", "metrics", "evaluation", "strategy", "factor"])
    
    IF task_touches_risk_areas:
        OUTPUT: "🔍 Gate-E5: 检测到任务涉及风险区域，启动 quant_risk_review..."
        
        # 调用 quant_risk_review (强制)
        RUN: python kernel/subagent_runner.py quant_risk_review \
            --files "{changed_files}" \
            --review-type "focused" \
            --focus-areas lookahead leakage protocol reproducibility
        
        READ: docs/subagents/runs/latest/SUMMARY.md
        
        # 更新 subagent_artifacts
        APPEND to current.subagent_artifacts:
            - subagent_id: "quant_risk_review"
              output_path: "docs/subagents/runs/latest/"
              invoked_at: NOW()
              verdict: subagent.verdict
        
        IF subagent.verdict == "fail":
            # 严重风险 → 必须 escalate，不可跳过
            OUTPUT:
                "## ⛔ Gate-E5 失败: 风险审查未通过
                 
                 **Verdict**: FAIL
                 **Critical Issues**: {subagent.critical_issues}
                 **Evidence**: {subagent.output_dir}
                 
                 → 必须返回 PLAN MODE 处理"
            
            INVOKE /dgsf_escalate WITH:
                type: "quant_risk_review_failed"
                severity: "high"
                evidence: subagent.output_dir
                affected_task: current.id
            BREAK
        
        IF subagent.verdict == "warn":
            # 警告 → 记录并询问
            OUTPUT:
                "## ⚠️ Gate-E5 警告: 风险审查发现问题
                 
                 **Verdict**: WARN
                 **Warnings**: {subagent.warnings}
                 **Risk Score**: {subagent.risk_score}/10"
            ASK: "继续执行？(y/查看详情/escalate)"
            IF user chooses escalate:
                INVOKE /dgsf_escalate WITH:
                    type: "quant_risk_review_warnings"
                    severity: "medium"
                BREAK
        
        IF subagent.verdict == "pass":
            OUTPUT: "✅ Gate-E5: 风险审查通过"
    
    # 3.1 [可选] 调用 Subagent 验证 Spec 一致性
    IF SUBAGENT_ALLOWED == YES AND current.spec_pointers:
        
        # 调用 repo_specs_retrieval 验证实现
        RUN: python kernel/subagent_runner.py repo_specs_retrieval \
            --question "验证 {current.title} 的实现是否符合 Spec" \
            --scope "{current.spec_pointers[0].path}"
        
        READ: docs/subagents/runs/latest/SUMMARY.md
        # 记录到结果中
        LOG: subagent.summary to current.result.spec_verification
    
    # 3.2 逐条验证验收标准
    all_passed = TRUE
    FOR ac IN current.acceptance_criteria:
        IF ac.verification_command:
            RUN ac.verification_command
            IF failed:
                all_passed = FALSE
                failure_reason = capture error
                BREAK
        ELSE:
            # 需要人工确认
            ASK: "验收标准 [{ac.id}] {ac.description} 是否满足？(y/n)"
    
    # ========================================
    # PHASE 4: STATE UPDATE + AUDIT（状态更新 + 审计）
    # ========================================
    
    IF all_passed:
        UPDATE queue[current.id]:
            status: completed
            completed_at: NOW()
            actual_effort: calculated from started_at
            result:
                output_files: [collected files]
                notes: "验证通过"
        
        UPDATE queue.stats:
            pending -= 1
            completed += 1
        
        SAVE state/execution_queue.yaml
        
        # 4.0 更新 Subagent 使用日志（E4 审计要求）
        APPEND to docs/state/SUBAGENT_USAGE.md:
            ### Entry: {NOW()}
            **Mode**: EXECUTE
            **Task**: {current.id}
            **Subagents Invoked**: {list subagent_artifacts}
            **Skip Justifications**: {current.skip_justification or "none"}
        
        OUTPUT:
            "## ✅ 任务完成: {current.title}
             
             验收标准: 全部通过
             用时: {actual_effort}
             
             → 正在执行 Git 操作..."
        
        # 4.1 Git 提交
        INVOKE /dgsf_git_ops WITH:
            action: checkpoint
            message: "feat({current.task_id}): {current.title}"
    
    ELSE:  # Verification failed
        UPDATE queue[current.id]:
            status: blocked
            result:
                notes: failure_reason
        
        UPDATE queue.stats:
            pending -= 1
            blocked += 1
        
        SAVE state/execution_queue.yaml
        
        OUTPUT:
            "## 🚫 任务阻塞: {current.title}
             
             失败原因: {failure_reason}
             
             → 正在调用诊断..."
        
        # 4.2 诊断
        INVOKE /dgsf_diagnose WITH:
            error: failure_reason
            context: current
        
        # R3: Stop on failure
        BREAK
    
    # ========================================
    # PHASE 5: NEXT TASK PROMPT（下一任务提示）
    # ========================================
    
    next = get_next_pending_item(queue)
    IF next exists:
        OUTPUT:
            "---
             
             ### 下一个任务
             **[{next.id}] {next.title}** ({next.priority})
             
             继续执行？(y/跳过/暂停)"
        
        WAIT for user input
```

---

## 🔄 QUEUE UPDATE PROTOCOL（队列更新协议）

### 状态转换

```
pending ──────► in_progress ──────► completed
    │               │
    │               └────────► blocked
    │
    └──────────────────────► skipped
```

### 每次状态变更后

1. **立即更新** `state/execution_queue.yaml`
2. **重新计算** `queue.stats`
3. **追加历史** `queue.history`

```yaml
# 示例: 任务完成后的更新
queue:
  - id: 1
    status: completed
    started_at: "2026-02-05T12:00:00Z"
    completed_at: "2026-02-05T14:30:00Z"
    actual_effort: "2.5 hours"
    result:
      output_files:
        - "projects/dgsf/reports/SDF_FEATURE_INVENTORY.json"
      metrics:
        features_count: 12
      notes: "发现 12 个特征定义，其中 3 个缺少文档"

stats:
  total: 3
  pending: 2
  completed: 1

history:
  - timestamp: "2026-02-05T12:00:00Z"
    action: "task_started"
    actor: "execute_mode"
    details:
      task_id: 1
  - timestamp: "2026-02-05T14:30:00Z"
    action: "task_completed"
    actor: "execute_mode"
    details:
      task_id: 1
      actual_effort: "2.5 hours"
```

---

## 🛑 EXIT CONDITIONS（退出条件）

| 条件 | 处理 |
|------|------|
| **队列清空** | 显示完成报告 → 建议 PLAN MODE |
| **遇到 Blocker** | 调用诊断 → 建议 PLAN MODE 重新评估 |
| **用户暂停** | 保存进度 → 可在新对话中恢复 |
| **长时间任务** | 提供命令 → 等待人工执行后报告 |

### 队列清空时的输出

```markdown
## 🎉 执行队列清空

### 完成报告

| Task | Subtask | 用时 | 结果 |
|------|---------|------|------|
| SDF_FEATURE_ENG_001 | T3.1 | 2.5h | ✅ |
| SDF_FEATURE_ENG_001 | T3.2 | 4h | ✅ |
| SDF_TEST_FIX_001 | Phase2 | 1.5h | ✅ |

**总用时**: 8 hours  
**成功率**: 100%

### 下一步

建议运行 PLAN MODE 规划下一阶段任务:
- 命令: `开启PLAN MODE`
- 或: `PLAN MODE`

---

执行队列已归档至 `state/archive/execution_queue_2026-02-05.yaml`
```

---

## 🔗 INTEGRATION WITH OTHER SKILLS（与其他 Skill 的集成）

| 场景 | 调用的 Skill |
|------|--------------|
| 执行具体任务 | `/dgsf_execute` |
| 验证结果 | `/dgsf_verify` |
| 遇到失败 | `/dgsf_diagnose` |
| 完成检查点 | `/dgsf_git_ops` |
| 更新状态 | `/dgsf_state_update` |
| **上报问题到 Plan Mode** | `/dgsf_escalate` |
| 需要重新规划 | `/dgsf_plan_mode` |

---

## 🔺 ESCALATION INTEGRATION（上报集成）

### 何时触发上报

在执行过程中遇到以下情况时，调用 `/dgsf_escalate`：

| 情况 | 类型 | 示例 |
|------|------|------|
| Spec 定义模糊 | `spec_unclear` | "特征命名规范未定义" |
| Spec 与代码不符 | `spec_error` | "阈值与实验结果矛盾" |
| 缺少必要 Spec | `spec_missing` | "缺少数据清洗规范" |
| 遇到技术问题 | `research_needed` | "正则化方法选择" |
| 需要重构 | `refactor_required` | "接口不支持新需求" |
| 完全阻塞 | `blocker` | "关键依赖不可用" |

### 上报后的队列处理

```
WHEN escalation.severity == "low":
    # 记录问题，继续执行
    ADD to escalation_queue
    CONTINUE to next task
    OUTPUT: "📝 问题已记录，继续执行下一任务"

WHEN escalation.severity == "medium":
    # 标记当前任务为 blocked，继续其他任务
    UPDATE current_task.status = "blocked"
    UPDATE current_task.blocked_reason = escalation.id
    ADD to escalation_queue
    CONTINUE to next task
    OUTPUT: "⚠️ 当前任务已阻塞，继续执行下一任务"

WHEN escalation.severity == "high":
    # 暂停队列，建议切换模式
    UPDATE execution_queue.metadata.paused = true
    UPDATE execution_queue.metadata.paused_reason = escalation.id
    ADD to escalation_queue
    OUTPUT: "🔶 队列已暂停，建议切换到 PLAN MODE"
    ASK: "切换到 PLAN MODE？(y/继续)"

WHEN escalation.severity == "critical":
    # 停止队列，强制切换模式
    UPDATE execution_queue.metadata.halted = true
    ADD to escalation_queue
    OUTPUT: "🔴 队列已停止，必须切换到 PLAN MODE"
    INVOKE: transition_to_plan_mode()
```

### 检测待处理上报

在 Entry Protocol 中增加检查：

```
AFTER loading execution_queue:
    READ state/escalation_queue.yaml
    IF escalation_queue.metadata.pending_count > 0:
        OUTPUT:
            "⚠️ 检测到 {N} 个待处理的上报问题
             
             | ID | 类型 | 严重程度 | 标题 |
             |---|------|----------|------|
             | ESC-xxx | spec_error | medium | ... |
             
             建议: 先运行 PLAN MODE 处理这些问题"
        ASK: "继续执行 / 切换到 PLAN MODE?"
```

---

## 🔄 PAUSE & RESUME PROTOCOL（暂停与恢复协议）

### 队列暂停状态

当队列被暂停时，`execution_queue.yaml` 中会包含：

```yaml
execution_queue:
  metadata:
    paused: true
    paused_at: "2026-02-05T14:30:00Z"
    paused_reason: "escalation:ESC-2026-02-05-001"
    paused_by: "execute_mode"
```

### 恢复执行的条件

```
WHEN entering Execute Mode:
    IF queue.metadata.paused == true:
        escalation_id = extract from paused_reason
        READ escalation_queue
        
        IF escalation.status == "resolved":
            # 上报已解决，可以恢复
            OUTPUT: "✅ 上报问题 [{escalation_id}] 已解决"
            SHOW: escalation.resolution.summary
            ASK: "恢复执行？(y/查看详情)"
            
            IF user confirms:
                UPDATE queue.metadata.paused = false
                CONTINUE execution
        ELSE:
            # 上报未解决
            OUTPUT: "⏸️ 队列暂停中，等待问题解决"
            SHOW: escalation details
            SUGGEST: "请先运行 PLAN MODE 处理此问题"
```

### 从 Plan Mode 返回后恢复

当 Plan Mode 解决了上报问题后：

```
1. Plan Mode 更新 escalation_queue:
   - escalation.status = "resolved"
   - escalation.resolution = { action_taken, summary, ... }

2. Plan Mode 更新 execution_queue（如果需要）:
   - 重新排队某些任务
   - 更新 acceptance_criteria
   - 清除 paused 状态

3. Execute Mode 入口检测:
   - 读取 escalation_queue，确认问题已解决
   - 读取 execution_queue，获取更新后的队列
   - 从中断点继续执行
```

---

## 📊 SESSION PERSISTENCE（会话持久化）

Execute Mode 的进度完全保存在文件系统中：

```
state/execution_queue.yaml
├── queue[].status        # 每个任务的状态
├── queue[].started_at    # 开始时间
├── queue[].result        # 执行结果
├── stats                 # 队列统计
└── history               # 操作历史
```

**新对话恢复流程**:

1. 用户输入: `执行模式` / `EXECUTE MODE`
2. Copilot 读取 `state/execution_queue.yaml`
3. 检测 `stats.in_progress > 0` 表示有中断的任务
4. 显示: "检测到上次执行被中断，任务 [X] 正在进行中"
5. 询问: "继续 / 重置 / 标记完成?"

---

## EXAMPLE: COMPLETE EXECUTE MODE SESSION

```markdown
用户: 执行模式

Copilot:
## 🎯 EXECUTE MODE 已激活

### 📋 执行队列状态

| # | Task ID | Subtask | 标题 | 优先级 | 状态 |
|---|---------|---------|------|--------|------|
| 1 | SDF_FEATURE_ENG_001 | T3.1 | 现有特征盘点 | P0 | ⏸️ pending |
| 2 | SDF_FEATURE_ENG_001 | T3.2 | 特征计算验证 | P0 | ⏸️ pending |

---

**来源**: Plan Session @ 2026-02-05T11:15:00Z

### 下一个任务

**[1] T3.1 现有特征盘点** (P0)

验收标准:
1. JSON 包含至少 10 个特征定义
2. 每个特征有 name, type, source, formula 字段

继续执行？(y/跳过/详情)

---

用户: y

Copilot:
## 🚀 开始执行: T3.1 现有特征盘点

正在扫描 repo/src/dgsf/factors/ ...

(执行 /dgsf_execute)

...执行过程...

## ✅ 任务完成: T3.1 现有特征盘点

验收标准: 2/2 通过
- ✅ JSON 包含 12 个特征定义
- ✅ 字段完整性验证通过

用时: 2.5 hours
输出: projects/dgsf/reports/SDF_FEATURE_INVENTORY.json

→ Git 提交: feat(SDF_FEATURE_ENG_001): T3.1 现有特征盘点

---

### 下一个任务

**[2] T3.2 特征计算验证** (P0)

继续执行？
```

````
