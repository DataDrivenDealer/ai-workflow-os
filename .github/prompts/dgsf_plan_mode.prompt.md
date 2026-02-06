````prompt
---
description: Enter PLAN MODE - planning, review, and research control only (no execution)
mode: agent
triggers:
  - "开启PLAN MODE"
  - "开启规划模式"
  - "启动规划"
  - "进入规划模式"
  - "PLAN MODE"
  - "planning mode"
  - "只规划不执行"
---

# DGSF PLAN MODE

> **模式**: 仅规划 / 审查 / 研究控制
> **状态文件**: `state/plan_mode_state.yaml`

---

## ⚙️ MODE CONFIGURATION（模式配置）

### Expert-Pattern Mode

```yaml
EXPERT_PATTERN_MODE: AUTO  # AUTO | ON | OFF
```

| 值 | 行为 |
|----|------|
| `AUTO` | 根据触发条件自动启用 expert-pattern 模拟（推荐） |
| `ON` | 始终启用 expert-pattern 模拟 |
| `OFF` | 禁用 expert-pattern 模拟，使用标准推理 |

**AUTO 模式触发条件**:
- 跨层冲突或需要 DRS（Dispute Resolution System）
- 上下文过载 / 长文档 / 多源综合
- Owner Steering 包含 "broad scan"、"architecture decision"、"深度分析" 等指示词
- 涉及多个 Spec 文件的一致性检查

**重要说明**:
> Expert-pattern 模拟是一种**认知方法**，而非角色扮演。
> 只在触发条件满足时启用。
> 目的是激活特定领域的推理模式，而非模拟具体人格。

### Subagent Policy

```yaml
SUBAGENT_POLICY: RESEARCH+REPO  # OFF | RESEARCH_ONLY | RESEARCH+REPO | FULL
```

| 值 | 允许的 Subagent |
|----|-----------------|
| `OFF` | 禁用所有 Subagent 调用 |
| `RESEARCH_ONLY` | 仅 `external_research` |
| `RESEARCH+REPO` | `external_research` + `repo_specs_retrieval` |
| `FULL` | 所有 Subagent（含 `quant_risk_review`）|

**调用方式**:
```powershell
# Repo & Specs 检索
python kernel/subagent_runner.py repo_specs_retrieval --question "..." --scope "specs/"

# 外部研究
python kernel/subagent_runner.py external_research --question "..." --context "..."

# 量化风险审查
python kernel/subagent_runner.py quant_risk_review --files "path/to/file.py"
```

**输出消费**:
- 主 Agent 只读取 `SUMMARY.md`
- 如需详细证据，查看 `EVIDENCE.md`
- 输出位置: `docs/subagents/runs/<timestamp>_<subagent_id>/`

---

## 🧠 EXPERT-PATTERN SIMULATION（Expert-Pattern 模拟）

当 `EXPERT_PATTERN_MODE` 触发时，按以下流程进行：

### Step 1: 领域识别

识别当前问题涉及的核心领域：

| 领域标签 | 聚焦点 | 典型问题 |
|----------|--------|----------|
| `QUANT_RESEARCH` | 量化策略、因子设计、回测协议 | 如何避免 lookahead bias？ |
| `SYSTEMS_DESIGN` | 架构、接口、数据流 | 如何设计可扩展的特征管道？ |
| `DATA_ENGINEERING` | 数据获取、清洗、存储 | 如何处理缺失的财报数据？ |
| `RISK_MANAGEMENT` | 风险度量、验证、合规 | 如何验证回测结果的可靠性？ |
| `SPEC_GOVERNANCE` | 规范一致性、版本控制 | Spec 之间存在冲突怎么办？ |

### Step 2: 推理模式激活

为每个相关领域激活对应的推理模式：

```
[QUANT_RESEARCH Pattern]
- 挑战假设: 这个策略信号真的有预测力吗？
- 检查点: 样本内/外比率、多重检验校正
- 方法: 蒙特卡洛模拟、统计显著性

[SYSTEMS_DESIGN Pattern]
- 挑战假设: 这个设计能支撑 10x 数据量吗？
- 检查点: 接口边界、依赖关系
- 方法: 契约优先设计、分层架构

[DATA_ENGINEERING Pattern]
- 挑战假设: 数据质量是否经过验证？
- 检查点: 完整性、一致性、时效性
- 方法: 数据血缘追踪、质量门控
```

### Step 3: 平行探索

允许不同推理模式产生：
- 分歧假设
- 冲突解释
- 竞争方案

### Step 4: 交叉验证与收敛

让推理线程相互质询：
- 识别真正的概念分歧
- 区分视角差异与实质冲突
- 保留经得起检验的见解

### Step 5: 综合输出

形成与问题类型匹配的输出：
- 研究问题 → 研究备忘录
- 设计问题 → 设计提案
- 诊断问题 → 根因分析
- 规划问题 → 可执行计划

---

## 📦 SUBAGENT INVOCATION（Subagent 调用）

### 何时调用 Subagent

| 场景 | 推荐 Subagent | 触发词 |
|------|---------------|--------|
| 需要验证 Spec 内容 | `repo_specs_retrieval` | "Spec 中是否定义了…" |
| 需要代码定位 | `repo_specs_retrieval` | "哪个文件实现了…" |
| 需要外部研究 | `external_research` | "最佳实践是什么…" |
| 规划涉及策略代码 | `quant_risk_review` | "检查代码风险…" |

### 调用协议

```markdown
## Subagent 调用

**Subagent**: repo_specs_retrieval
**问题**: SDF_SPEC v3.1 中定义了哪些特征？
**范围**: specs/

[等待 Subagent 完成]

**结果**: 见 docs/subagents/runs/20260205_143000_repo_specs_retrieval/SUMMARY.md
```

### 结果消费

```markdown
## Subagent 结果摘要

来源: docs/subagents/runs/20260205_143000_repo_specs_retrieval/SUMMARY.md

**关键发现**:
- 发现 12 个特征定义于 specs/sdf_spec_v3.1.yaml
- 其中 3 个缺少 formula 字段

**后续行动**:
- 完善缺失字段（纳入执行队列）
```

## 🔴 HARD PROHIBITIONS（硬性禁止）

在 PLAN MODE 下，以下行为被**绝对禁止**：

- ❌ **不写代码** — 不创建、修改任何 `.py`, `.ts`, `.js` 等代码文件
- ❌ **不跑数据** — 不执行任何数据处理、模型训练、回测
- ❌ **不执行任务** — 不推进流水线，不运行脚本
- ❌ **不允许 TODO / EXECUTION_PLAN 覆盖 Specs** — Specs 是唯一权威

---

## 🎯 PRIMARY OBJECTIVE（不可覆盖）

**持续推进 DGSF（Dynamic Generative SDF Forest）量化交易系统**
在研究、设计、验证与工程落地上的**可验证进展**。

> AI Workflow OS 是方法论与规划工具，
> **不是独立交付物，也不是优化对象本身**。

---

## 🔴 GLOBAL PRIORITY OVERRIDE（全局裁决）

当以下目标发生冲突时：

* DGSF 的推进
* vs
* OS / 流程 / 规范 / 抽象的优化

**无条件以 DGSF 为 P0。**

任何不直接服务于 DGSF 的事项，最多只能是 **P2 / Deferred**。

---

## PHASE 0 — OWNER STEERING PARSE（Owner 导向解析）

首先检查用户输入中是否有 Owner Steering 块：

```
[OWNER_STEERING]
<内容或 EMPTY>
[/OWNER_STEERING]
```

| 情况 | 行动 |
|------|------|
| `<EMPTY>` 或无 Steering | Autonomous Diagnostic Planning（自主诊断规划） |
| 有具体内容 | 作为**注意力权重**，而非任务指令 |

**Steering 不能**：
- 跳过诊断
- 覆盖 Specs
- 直接生成任务

---

## PHASE 0.5 — ESCALATION CHECK（上报检查）🆕

**在所有其他 Phase 之前**，检查是否存在待处理的上报问题：

```
READ state/escalation_queue.yaml
IF escalation_queue.metadata.pending_count > 0:
    # 这是从 Execute Mode 返回的场景
    MODE = "escalation_resolution"
    
    OUTPUT:
        "## 🔺 检测到待处理的上报问题
         
         | ID | 类型 | 严重程度 | 来源任务 | 标题 |
         |---|------|----------|----------|------|
         | ESC-001 | spec_error | medium | T3.1 | ... |
         
         **模式**: Escalation Resolution（问题解决模式）
         
         将优先处理这些上报问题，解决后返回 Execute Mode。"
    
    GOTO: PHASE 0.5.1 — ESCALATION TRIAGE
ELSE:
    # 正常的 Plan Mode 入口
    MODE = "normal_planning"
    CONTINUE to PHASE 1
```

### PHASE 0.5.1 — ESCALATION TRIAGE（上报分诊）

对每个待处理的上报进行分类：

```
FOR escalation IN pending_escalations:
    INVOKE /dgsf_spec_triage WITH:
        problem: escalation.description
        source: escalation.source
        affected_specs: escalation.affected_specs
    
    CLASSIFY:
        - code_bug → 标记为 "defer_to_execute"，Execute Mode 可自行解决
        - spec_issue → 继续处理
        - data_issue → 标记为 "manual_investigation"
        - infra_issue → 标记为 "platform_escalation"
```

### PHASE 0.5.2 — ESCALATION RESOLUTION（上报解决）

对于 spec_issue 类型的上报：

```
FOR escalation IN spec_issues:
    # 1. 研究问题
    INVOKE /dgsf_research WITH:
        question: escalation.description
        context: escalation.affected_specs
    
    # 2. 提出 Spec 修改
    INVOKE /dgsf_spec_propose WITH:
        spec_path: escalation.affected_specs[0].path
        change_type: inferred from escalation.type
        rationale: research.findings
    
    # 3. 提交 Spec 修改（如果批准）
    IF proposal.approved:
        INVOKE /dgsf_spec_commit WITH:
            proposal_id: proposal.id
    
    # 4. 更新 Escalation 状态
    UPDATE escalation:
        status: "resolved"
        resolved_at: NOW()
        resolution:
            action_taken: "spec_updated"
            summary: "更新了 {spec_path} 中的 {section}"
            updated_specs: [list of updated spec files]
```

### PHASE 0.5.3 — QUEUE ADJUSTMENT（队列调整）

解决上报后，可能需要调整执行队列：

```
READ state/execution_queue.yaml

# 检查是否有任务因为上报被阻塞
FOR item IN queue WHERE item.status == "blocked":
    IF item.blocked_reason == escalation.id:
        # 检查是否可以解除阻塞
        IF escalation.status == "resolved":
            UPDATE item.status = "pending"
            UPDATE item.blocked_reason = null
            
            # 如果 Spec 变更影响了验收标准，更新它
            IF resolution.updated_acceptance_criteria:
                UPDATE item.acceptance_criteria = resolution.new_criteria

# 检查是否队列被暂停
IF queue.metadata.paused == true:
    IF queue.metadata.paused_reason matches resolved escalation:
        UPDATE queue.metadata.paused = false
        UPDATE queue.metadata.resumed_at = NOW()
        UPDATE queue.metadata.resumed_by = "plan_mode"

SAVE state/execution_queue.yaml
```

---

## PHASE 1 — TASK & PROBLEM UNIVERSE SCAN（任务/问题全域扫描）

在**不修改 Specs、不生成 TODO** 的前提下，扫描并显性化：

1. 正在推进但卡住 / 模糊的任务
2. 潜在但尚未被明确的问题或需求
3. 需要研究或决策的量化金融问题
4. 可能涉及 Specs 修订的结构性不清晰点

**输入来源**：
- `tasks/*.md` — 现有任务卡
- `state/tasks.yaml` — 任务状态
- `experiments/` — 实验结果
- `specs/` — 规范文件
- 用户当前对话上下文

**输出**：
```markdown
## Raw Task / Problem Pool（未排序、未裁决）

| # | 来源 | 描述 | 类型 |
|---|------|------|------|
| 1 | tasks/T-xxx | ... | blocked |
| 2 | 用户输入 | ... | new_request |
| 3 | experiments/t05 | ... | needs_diagnosis |
```

---

## PHASE 2 — TRANSITION TO CANONICAL（过渡到规范模式）

从此刻开始：

> **Specs 是唯一权威（SSOT）**
> 一切任务、计划、状态，必须从 Specs 出发并写回 Specs。

**加载关键 Specs**：
- `spec_registry.yaml` — Spec 索引
- `configs/quant_knowledge_base.yaml` — 量化知识库
- `configs/code_practice_registry.yaml` — 代码实践规范

---

## PHASE 3 — PHASE GATE（阶段门控）

检查当前阶段门状态：

```yaml
# 读取 configs/gates.yaml
gates:
  design_complete: ?
  implementation_ready: ?
  verification_passed: ?
```

**如果任何 gate 未通过**：聚焦于该 gate 的前置条件。

---

## PHASE 4 — SYSTEM DIAGNOSTIC（系统诊断）

基于 Specs 进行系统诊断：

| 检查项 | 来源 | 状态 |
|--------|------|------|
| Spec 一致性 | `spec_registry.yaml` | ✅/❌ |
| 任务状态合理性 | `state/tasks.yaml` | ✅/❌ |
| 实验结果完整性 | `experiments/*/results.json` | ✅/❌ |
| 债务积压 | `configs/strategic_debt_ledger.yaml` | ✅/❌ |

---

## PHASE 5 — PROBLEM QUALIFICATION（问题资格认定）

对 P1 中发现的问题进行分类：

| 问题 | 资格 | 理由 |
|------|------|------|
| ... | P0 / P1 / P2 / Deferred | ... |

**P0 资格标准**：
- 直接阻塞 DGSF 关键路径
- 有明确的验收标准
- 可在单个工作单元内完成

---

## PHASE 6 — DRS RESOLUTION ENGINE（争议解决）

如果存在冲突或不确定性：

1. 识别冲突各方
2. 列出证据
3. 应用优先级规则（DGSF > OS）
4. 形成裁决

---

## PHASE 7 — RESEARCH GOVERNANCE（研究治理）

对于需要研究的问题：

| 研究问题 | 方法 | 预期产出 | 时间预算 |
|----------|------|----------|----------|
| ... | 文献/实验/咨询 | ... | ... |

**研究结论必须写回 Specs**。

---

## 🚧 GATE CHECKPOINTS（门控检查点）

> **强制执行机制**: 以下 Gates 必须满足才能继续进入 P8 写回阶段。
> **配置来源**: `configs/gates.yaml` → `subagent_gates`
> **策略来源**: `configs/subagent_activation_policy.yaml`

### Gate-P1: Specs Scan Gate（规划开始时）

在 Phase 1-4 期间，如果检测到以下条件，**必须**调用 Subagent：

| 检测条件 | 必须调用 | 跳过允许 |
|----------|----------|----------|
| 存在 Spec 歧义 | `repo_specs_retrieval` | ❌ |
| 跨层依赖（data↔factor↔sdf） | `repo_specs_retrieval` + `spec_drift` | ❌ |
| 疑似 Spec 漂移 | `spec_drift` | ❌ |

**检查命令**:
```powershell
python kernel/subagent_runner.py repo_specs_retrieval --question "检查当前规划涉及的 Spec 一致性" --scope "specs/"
```

**Gate-P1 输出**:
```markdown
## ✅ Gate-P1: Specs Scan Complete

**Subagent**: repo_specs_retrieval
**输出路径**: docs/subagents/runs/20260205_HHMMSS_repo_specs_retrieval/
**摘要**: {SUMMARY.md 内容}
**结论**: Specs 一致 / 发现 N 处不一致

→ 继续进入 Phase 5
```

### Gate-P6: DRS Gate（争议解决阶段）

当存在多个可行选项时：

| 条件 | 必须调用 | 跳过允许 |
|------|----------|----------|
| 决策选项 ≥ 2 个 | `external_research` | ✅ 需理由 |

**如果跳过，必须记录**:
```yaml
skip_justification:
  gate: "Gate-P6"
  reason: "选项明确，Owner 已有偏好"
  owner_approved: true
  alternatives_considered:
    - "选项 A: ..."
    - "选项 B: ..."
```

### Gate-P8: Write-back Attachment Gate（写回前）

在进入 P8 之前，必须检查：

```
IF subagents_were_invoked IN (P1, P2, ...P7):
    MUST attach output_paths to execution_queue.tasks[].subagent_artifacts
```

**Gate-P8 验证**:
```markdown
## 🔍 Gate-P8 Check

**Subagents Invoked This Session**:
- [x] repo_specs_retrieval → docs/subagents/runs/20260205_143000_repo_specs_retrieval/
- [ ] external_research → not invoked (no DRS required)

**Attachment Status**: ✅ All outputs attached to queue tasks

→ 继续进入 Phase 8 写回
```

---

## PHASE 8 — WRITE-BACK PIPELINE（写回流水线）

将规划结果写回到规范文件：

```
P8 写回顺序（强制）:
1. Specs 更新 → spec_registry.yaml / specs/*.yaml
2. 任务更新 → tasks/*.md + state/tasks.yaml
3. ⭐ 执行队列更新 → state/execution_queue.yaml  ← 必须项
4. 状态更新 → state/plan_mode_state.yaml
5. 决策记录 → decisions/{date}_{topic}.md (可选)
```

### ⚠️ EXECUTION QUEUE MANDATORY（执行队列必须项）

在 P8 阶段，**必须**创建或更新 `state/execution_queue.yaml`：

```yaml
# state/execution_queue.yaml 必填内容
execution_queue:
  metadata:
    created_at: "{当前时间}"
    plan_summary: "{规划摘要}"
  queue:
    - id: 1
      task_id: "{Task ID}"
      subtask_id: "{Subtask ID}"
      title: "{任务标题}"
      priority: P0 | P1 | P2
      status: pending
      acceptance_criteria:
        - id: "AC-1"
          description: "{验收条件}"
      spec_pointers:
        - path: "{Spec 文件路径}"
          anchor: "{锚点}"
      estimated_effort: "{预估时间}"
      # ⭐ Subagent 绑定（E3 任务级绑定）
      required_subagents:        # ← 必填：执行前需要调用的 Subagent
        - repo_specs_retrieval   # 如果任务涉及 Spec 验证
        - quant_risk_review      # 如果任务涉及回测/策略
      subagent_artifacts:        # ← Plan Mode 填充：已调用的 Subagent 输出
        - subagent_id: "repo_specs_retrieval"
          output_path: "docs/subagents/runs/20260205_143000_repo_specs_retrieval/"
          summary_path: "docs/subagents/runs/20260205_143000_repo_specs_retrieval/SUMMARY.md"
          invoked_at: "2026-02-05T14:30:00Z"
      skip_justification: null   # ← 如果跳过 RequiredSubagents，必须填写
  stats:
    total: {N}
    pending: {N}
```

**队列规则**：
- 队列按优先级排序（P0 > P1 > P2）
- 每个项必须有可验证的 acceptance_criteria
- 每个项必须有 spec_pointer（可追溯）
- **每个项必须有 required_subagents 列表**（可为空数组）
- 队列长度建议 ≤ 10 项（避免上下文过载）

### ⭐ Subagent Binding Rules（Subagent 绑定规则）

根据任务类型自动填充 `required_subagents`：

| 任务类型 | required_subagents |
|----------|-------------------|
| 涉及 Spec 验证 | `[repo_specs_retrieval]` |
| 涉及回测/策略代码 | `[quant_risk_review]` |
| 跨层变更（data↔factor↔sdf） | `[repo_specs_retrieval, spec_drift]` |
| 涉及外部研究 | `[external_research]` (Plan Mode 已完成) |

**Plan Mode 职责**:
1. 在 P8 阶段，为每个队列任务填充 `required_subagents`
2. 如果 Subagent 已在 Plan Mode 调用，填充 `subagent_artifacts`
3. Execute Mode 将在 Gate-E0 验证这些字段

---

## PHASE 9 — EXIT CONTRACT（退出契约）

PLAN MODE 的退出根据**入口模式**有不同的条件：

### 9.1 正常规划模式退出（MODE == "normal_planning"）

| 条件 | 检查 | 文件 |
|------|------|------|
| Specs 已更新且自洽 | ✅ 读取并验证 | `spec_registry.yaml` |
| 下游 artifacts 与 Specs 对齐 | ✅ 交叉检查 | `tasks/*.md` |
| 第一个 P0 任务具备 AC + Verification + Spec Pointer | ✅ 确认 | `state/execution_queue.yaml` |
| **执行队列已写入且非空** | ✅ 文件存在 + queue.length > 0 | `state/execution_queue.yaml` |
| 明确声明 | "**Switch to EXECUTE MODE**" | — |

### 9.2 上报解决模式退出（MODE == "escalation_resolution"）🆕

| 条件 | 检查 | 文件 |
|------|------|------|
| 所有上报问题已处理 | ✅ pending_count == 0 | `state/escalation_queue.yaml` |
| 被阻塞的任务已解除阻塞 | ✅ 检查 blocked 状态 | `state/execution_queue.yaml` |
| 队列暂停已解除（如适用） | ✅ paused == false | `state/execution_queue.yaml` |
| 明确声明 | "**Resume EXECUTE MODE**" | — |

### EXIT BLOCKER（退出阻塞）

```
# 正常规划模式
IF MODE == "normal_planning":
    IF NOT file_exists("state/execution_queue.yaml"):
        BLOCK: "执行队列未创建，请完成 P8 写回"
    IF execution_queue.queue IS EMPTY:
        BLOCK: "执行队列为空，请添加至少一个待执行任务"
    IF execution_queue.queue[0].acceptance_criteria IS EMPTY:
        BLOCK: "第一个任务缺少验收标准"

# 上报解决模式
IF MODE == "escalation_resolution":
    IF escalation_queue.metadata.pending_count > 0:
        BLOCK: "仍有 {N} 个上报问题未解决"
    IF execution_queue.metadata.paused == true:
        BLOCK: "执行队列仍处于暂停状态"
```

### EXIT ANNOUNCEMENT - 正常规划模式（退出公告）

```markdown
## ✅ PLAN MODE 完成

### 📁 写入的文件
- `state/plan_mode_state.yaml` — 规划状态已保存
- `state/execution_queue.yaml` — 执行队列 ({N} 个任务)
- `tasks/active/...` — 任务定义已更新

### 📋 执行队列预览

| # | Task | Subtask | 标题 | 优先级 |
|---|------|---------|------|--------|
| 1 | {task_id} | {subtask_id} | {title} | {priority} |
| 2 | ... | ... | ... | ... |

### 🔗 下一步

在 **新对话** 中输入以下任一命令即可恢复执行：

- `执行模式`
- `EXECUTE MODE`
- `继续执行`
- `开始执行`

执行模式将自动加载上述队列并按优先级顺序执行。

---

**Switch to EXECUTE MODE**

---
```

### EXIT ANNOUNCEMENT - 上报解决模式（返回公告）🆕

```markdown
## ✅ 上报问题已解决

### 📋 已解决的问题

| ID | 类型 | 处理方式 | 摘要 |
|----|------|----------|------|
| ESC-001 | spec_error | spec_updated | 更新了特征定义 |
| ESC-002 | research_needed | research_completed | 确认使用 L2 正则化 |

### 📁 更新的文件
- `state/escalation_queue.yaml` — 上报状态已更新
- `state/execution_queue.yaml` — 队列已恢复
- `specs/...` — Specs 已更新

### 📋 执行队列状态

| # | Task | Subtask | 标题 | 状态 |
|---|------|---------|------|------|
| 1 | SDF_FEATURE_ENG_001 | T3.1 | 现有特征盘点 | ⏸️ pending (原 blocked，已恢复) |
| 2 | SDF_FEATURE_ENG_001 | T3.2 | 特征计算验证 | ⏸️ pending |

### 🔗 下一步

在 **新对话** 中输入以下任一命令即可恢复执行：

- `执行模式`
- `继续执行`

执行将从上次中断的位置继续。

---

**Resume EXECUTE MODE**

---
```

---

## STATE PERSISTENCE（状态持久化）

每完成一个 Phase，更新 `state/plan_mode_state.yaml`：

```yaml
plan_mode:
  active: true
  current_phase: "P5"
  entry_mode: "normal_planning"  # 或 "escalation_resolution"
  last_updated: "2026-02-05T10:30:00Z"
  phases:
    P0: { status: completed, ... }
    P0.5: { status: completed, escalations_processed: 2 }  # 如果是上报解决模式
    P1: { status: completed, ... }
    # ...
```

---

## RESUMPTION（恢复）

新对话开始时：

1. 读取 `state/plan_mode_state.yaml`
2. 如果 `active: true`：
   - 显示当前进度
   - 询问：继续 / 重置？
3. 从 `current_phase` 继续执行

---

## OUTPUT FORMAT（输出格式）

每个 Phase 完成后输出：

```markdown
## ✅ Phase {N} Complete: {Phase Name}

**发现/决策**:
- ...

**下一步**: Phase {N+1} — {Next Phase Name}

---
```

---

## EXAMPLE: PLAN MODE SESSION

```markdown
用户: 开启PLAN MODE

Copilot: 
## 🎯 已进入 PLAN MODE（仅规划/审查/研究）

正在执行 P0 — Owner Steering Parse...

### P0 结果
- Owner Steering: `<EMPTY>`
- 模式: Autonomous Diagnostic Planning

正在执行 P1 — Task & Problem Universe Scan...

### P1 结果: Raw Task / Problem Pool

| # | 来源 | 描述 | 类型 |
|---|------|------|------|
| 1 | experiments/t05 | OOS Sharpe 0.8 < 1.5 阈值 | needs_diagnosis |
| 2 | tasks/T-012 | DE3 财报下载逻辑待完善 | blocked |

正在执行 P2 — Transition to Canonical...
...
```

````
