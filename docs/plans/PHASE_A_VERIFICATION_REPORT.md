# Phase A Verification Report — Workspace Constitution

> **完成时间**: 2026-02-05  
> **验证状态**: ✅ PASSED  
> **测试结果**: 237 tests passed

---

## 📋 Deliverables Checklist

| 序号 | 文件 | 操作 | 状态 | 证据路径 |
|------|------|------|------|----------|
| A1 | `.github/copilot-instructions.md` | 修改 | ✅ | 添加 SUBAGENT ENFORCEMENT MECHANISMS 章节 |
| A2 | `configs/subagent_activation_policy.yaml` | 新建 | ✅ | 完整的激活策略定义 |
| A3 | `configs/gates.yaml` | 修改 | ✅ | 添加 subagent_gates 配置 |
| A4 | `docs/state/SUBAGENT_USAGE.md` | 新建 | ✅ | 审计日志模板 |
| A5 | `.github/prompts/dgsf_plan_mode.prompt.md` | 修改 | ✅ | 添加 Gate-P1/P6/P8 检查点 |
| A6 | `.github/prompts/dgsf_execute_mode.prompt.md` | 修改 | ✅ | 添加 Gate-E0/E5 检查点 |

---

## 📁 Files Changed

### New Files Created

1. **configs/subagent_activation_policy.yaml** (完整策略定义)
   - AUTO triggers 定义
   - Gate triggers (P1, P6, P8, E0, E5)
   - Task-level binding schema
   - Audit requirements
   - Enforcement actions

2. **docs/state/SUBAGENT_USAGE.md** (审计日志)
   - Usage summary template
   - Breakdown by subagent
   - Recent invocation log
   - Skip reason analysis
   - Audit trail template

### Modified Files

1. **.github/copilot-instructions.md**
   - 新增章节: SUBAGENT ENFORCEMENT MECHANISMS
   - E1: GATED STEPS (Plan Mode: P1, P6, P8 | Execute Mode: E0, E5)
   - E2: ACTIVATION POLICY (自动触发条件)
   - E3: TASK-LEVEL BINDING (任务级绑定 schema)
   - E4: AUDIT + METRICS (审计与指标)

2. **configs/gates.yaml**
   - 新增: subagent_gates 配置块
   - Gate-P1: Specs Scan Gate
   - Gate-P6: DRS Gate
   - Gate-P8: Write-back Attachment Gate
   - Gate-E0: Pre-execution Check Gate
   - Gate-E5: Risk Review Gate

3. **.github/prompts/dgsf_plan_mode.prompt.md**
   - 新增: GATE CHECKPOINTS 章节 (在 P7 和 P8 之间)
   - 更新: EXECUTION QUEUE MANDATORY 章节，增加 required_subagents 和 subagent_artifacts 字段
   - 新增: Subagent Binding Rules 表格

4. **.github/prompts/dgsf_execute_mode.prompt.md**
   - 新增: Gate-E0 PRE-EXECUTION SUBAGENT CHECK 逻辑
   - 更新: Gate-E5 RISK REVIEW GATE 逻辑（更详细的输出和处理）
   - 新增: Subagent 使用日志更新逻辑 (E4 审计)

---

## ✅ Verification Evidence

### Test Results

```
================================================ 237 passed in 42.01s =================================================
```

所有 kernel 测试通过，确认变更未破坏现有功能。

### File Existence Check

| 文件 | 存在 |
|------|------|
| `configs/subagent_activation_policy.yaml` | ✅ |
| `docs/state/SUBAGENT_USAGE.md` | ✅ |
| `.github/copilot-instructions.md` (updated) | ✅ |
| `configs/gates.yaml` (updated) | ✅ |
| `.github/prompts/dgsf_plan_mode.prompt.md` (updated) | ✅ |
| `.github/prompts/dgsf_execute_mode.prompt.md` (updated) | ✅ |

---

## 🔗 Enforcement Mechanism Summary

### E1: Gated Steps

| Gate | Mode | 触发条件 | 必须调用 | 跳过允许 |
|------|------|----------|----------|----------|
| Gate-P1 | PLAN | 歧义/跨层/漂移 | repo_specs_retrieval, spec_drift | ❌ |
| Gate-P6 | PLAN | 决策 ≥2 选项 | external_research | ✅ 需理由 |
| Gate-P8 | PLAN | 写回退出 | 附加输出路径 | ❌ |
| Gate-E0 | EXECUTE | 任务有 RequiredSubagents | 按任务定义 | ❌ |
| Gate-E5 | EXECUTE | 涉及 backtest/data/metrics | quant_risk_review | ❌ |

### E2: Activation Policy

| 条件 | 自动调用 |
|------|----------|
| 跨层变更 | repo_specs_retrieval, spec_drift |
| Spec 歧义/冲突 | repo_specs_retrieval, spec_drift |
| 评估指标/回测变更 | quant_risk_review |
| DRS 需求 | external_research |
| 上下文过载 (>10 files) | repo_specs_retrieval |

### E3: Task-Level Binding

```yaml
task:
  required_subagents: [...]     # Plan Mode 必须填充
  subagent_artifacts: [...]     # 调用后自动填充
  skip_justification: null      # 跳过时必须填写
```

### E4: Audit + Metrics

- 审计日志: `docs/state/SUBAGENT_USAGE.md`
- 滚动窗口: 20 任务
- 警报阈值: 使用率 < 40%

---

## 🚀 VS Code Verification

### How Instructions Are Applied

1. **File Location**: `.github/copilot-instructions.md` 位于 workspace root，VS Code Copilot 会自动加载此文件作为 workspace-level instructions。

2. **Prompt Files**: `.github/prompts/dgsf_*.prompt.md` 文件定义了可触发的 skills (如 `/dgsf_plan_mode`)。

3. **Trigger Words**: 当用户输入 "PLAN MODE"、"开启规划模式" 等触发词时，Copilot 会根据 prompt 文件中的 `triggers` 字段激活相应的 skill。

### Verification Commands

```powershell
# 1. 检查 instructions 文件格式
Get-Content ".github/copilot-instructions.md" | Select-String "SUBAGENT ENFORCEMENT"

# 2. 检查 gates 配置
Get-Content "configs/gates.yaml" | Select-String "subagent_gates" -Context 0,5

# 3. 检查 activation policy
Test-Path "configs/subagent_activation_policy.yaml"

# 4. 检查 audit log 模板
Test-Path "docs/state/SUBAGENT_USAGE.md"
```

---

## 📌 Next Steps (Phase B Preview)

Phase B 将创建可复用的 Workflow prompt 文件：

1. `prompts/RUN_SUBAGENT.md` — 标准化 Subagent 调用包装器
2. `prompts/EVOLVE_SYSTEM.md` — 自演进循环
3. Custom Agents 配置 — Plan Agent + Execute Agent 的明确交接

---

**Phase A 已完成。准备进入 Phase B。**

---

# Phase B Verification Report — Reusable Workflows

> **完成时间**: 2026-02-05  
> **验证状态**: ✅ PASSED  
> **测试结果**: 237 tests passed

---

## 📋 Deliverables Checklist

| 序号 | 文件 | 操作 | 状态 | 描述 |
|------|------|------|------|------|
| B1 | `.github/prompts/dgsf_run_subagent.prompt.md` | 新建 | ✅ | 标准化 Subagent 调用包装器 |
| B2 | `.github/prompts/dgsf_evolve_system.prompt.md` | 新建 | ✅ | 自演进循环 |
| B3 | `configs/agent_modes.yaml` | 新建 | ✅ | Plan/Execute Agent 定义 |
| B4 | `.github/copilot-instructions.md` | 修改 | ✅ | 添加 agent_modes 引用 |

---

## 📁 New Skills Created

### `/dgsf_run_subagent` — 标准化 Subagent 调用

**触发词**: "运行 subagent", "调用 subagent", "run subagent", "invoke subagent"

**功能**:
- 验证调用权限（模式限制）
- 准备输入参数（基于 input_contract）
- 执行 Subagent 逻辑（repo_specs_retrieval, external_research, quant_risk_review, spec_drift）
- 生成标准化输出（SUMMARY.md + EVIDENCE.md）
- 更新审计日志

**输出目录**: `docs/subagents/runs/<timestamp>_<subagent_id>/`

### `/dgsf_evolve_system` — 自演进循环

**触发词**: "演进系统", "evolve system", "系统自演进", "self-evolution"

**功能**:
- E1: 信号收集（规则摩擦、Gate 失败、用户反馈、效率指标）
- E2: 信号分析（聚类、根因识别）
- E3: 提案生成（变更描述、DGSF 价值证明、回滚计划）
- E4: 审批门（基于变更类型：hotfix/enhancement/feature/refactor）
- E5: 实施（快照、应用、验证）
- E6: 文档更新

**约束**: DGSF 是 P0，OS 改进只有在服务 DGSF 时才允许

---

## 📁 Agent Mode Definitions

### Plan Agent

| 属性 | 值 |
|------|-----|
| **能力** | 读取、搜索、写 specs/state/docs、调用 Subagent |
| **禁止** | 写代码、运行终端、运行测试、修改实现 |
| **Subagents** | repo_specs_retrieval, external_research, quant_risk_review, spec_drift |
| **Gates** | Gate-P1 (必须), Gate-P6 (可选), Gate-P8 (必须) |
| **产出** | execution_queue.yaml, plan_mode_state.yaml |

### Execute Agent

| 属性 | 值 |
|------|-----|
| **能力** | 读取、搜索、写代码、运行终端、运行测试、调用 Subagent (限 review) |
| **禁止** | 写 specs、重排队列、添加任务、修改优先级、网络研究 |
| **Subagents** | repo_specs_retrieval, quant_risk_review (限 review gate) |
| **Gates** | Gate-E0 (必须), Gate-E5 (必须) |
| **产出** | 实现代码、测试代码、队列状态更新 |

### Handoff Protocol

```
Plan → Execute:
  触发: "Switch to EXECUTE MODE"
  前置: execution_queue 存在且非空
  交接: execution_queue.yaml + plan_mode_state.yaml

Execute → Plan:
  触发: escalation / blocker / queue complete
  交接: escalation_queue.yaml + execution_queue.yaml
```

---

## ✅ Verification Evidence

### Test Results

```
================================================ 237 passed in 41.93s =================================================
```

### File Existence Check

| 文件 | 存在 | 大小 |
|------|------|------|
| `.github/prompts/dgsf_run_subagent.prompt.md` | ✅ | 12,503 bytes |
| `.github/prompts/dgsf_evolve_system.prompt.md` | ✅ | 10,949 bytes |
| `configs/agent_modes.yaml` | ✅ | 已验证 |

---

## 🔗 Updated Skills Inventory

Phase B 后的完整 Skills 列表：

| Skill | 类别 | Phase |
|-------|------|-------|
| `/dgsf_plan_mode` | Mode Control | A |
| `/dgsf_execute_mode` | Mode Control | A |
| `/dgsf_escalate` | Mode Control | A |
| `/dgsf_run_subagent` | **Subagent Wrapper** | **B** |
| `/dgsf_evolve_system` | **Evolution** | **B** |
| `/dgsf_research` | Cognitive | existing |
| `/dgsf_plan` | Cognitive | existing |
| `/dgsf_execute` | Execution | existing |
| `/dgsf_verify` | Review | existing |
| `/dgsf_diagnose` | Execution | existing |
| `/dgsf_abort` | Control | existing |
| `/dgsf_decision_log` | Audit | existing |
| `/dgsf_state_update` | State | existing |
| `/dgsf_research_summary` | Cognitive | existing |
| `/dgsf_repo_scan` | Discovery | existing |
| `/dgsf_git_ops` | Execution | existing |
| `/dgsf_spec_triage` | Governance | existing |
| `/dgsf_spec_propose` | Governance | existing |
| `/dgsf_spec_commit` | Governance | existing |

---

**Phase B 已完成。准备进入 Phase C。**
