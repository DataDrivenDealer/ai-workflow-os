```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4A90D9', 'secondaryColor': '#7AB648', 'tertiaryColor': '#F5A623'}}}%%

flowchart TD
    subgraph Triggers["🎯 触发层 (Problem Discovery)"]
        T1[🧪 实验失败<br/>OOS Sharpe < 阈值]
        T2[🔴 测试失败<br/>AssertionError]
        T3[👁️ 代码审查<br/>发现设计问题]
        T4[📊 监控告警<br/>指标异常]
    end
    
    subgraph Skills["🧠 Skills 层 (Cognitive Workflow)"]
        S1["/dgsf_spec_triage<br/>问题分类与定性"]
        S2["/dgsf_research<br/>调研分析"]
        S3["/dgsf_plan<br/>规划方案"]
        S4["/dgsf_spec_propose<br/>生成变更提案"]
        S5["/dgsf_spec_commit<br/>提交变更"]
        S6["/dgsf_verify<br/>验证结果"]
    end
    
    subgraph MCP["⚙️ MCP Tools 层 (Atomic Operations)"]
        M1["spec_triage()<br/>分析问题类型"]
        M2["spec_read()<br/>读取当前Spec"]
        M3["spec_list()<br/>列出所有Specs"]
        M4["spec_propose()<br/>创建变更提案"]
        M5["spec_commit()<br/>应用变更"]
    end
    
    subgraph Hooks["🔗 Hooks 层 (Guardrails)"]
        H1["pre-spec-change<br/>• Canon保护<br/>• 权限验证<br/>• 格式检查"]
        H2["post-spec-change<br/>• Lineage更新<br/>• 审计日志<br/>• 触发测试"]
    end
    
    subgraph Governance["🛡️ 治理层 (Human-in-Loop)"]
        G1{人工审批<br/>Project Lead}
        G2[decisions/*.yaml<br/>审批记录]
        G3[ops/audit/*.yaml<br/>审计追踪]
    end
    
    subgraph Outputs["📤 输出层"]
        O1[Spec 已更新]
        O2[实验需重跑]
        O3[Lineage 已记录]
    end
    
    %% 触发流程
    T1 & T2 & T3 & T4 --> S1
    
    %% Skills 调用 MCP
    S1 --> M1
    M1 --> |"spec_issue"| S2
    M1 --> |"code_bug"| Diag["/dgsf_diagnose"]
    
    S2 --> M2 & M3
    S2 --> S3
    S3 --> S4
    S4 --> M4
    
    %% 治理流程
    M4 --> G1
    G1 --> |"Approved"| G2
    G2 --> S5
    
    %% Hooks 拦截
    S5 --> M5
    M5 --> H1
    H1 --> |"Pass"| Apply[应用变更]
    H1 --> |"Block"| Reject[拒绝变更]
    
    Apply --> H2
    H2 --> G3
    H2 --> O1 & O2 & O3
    
    %% 验证闭环
    O1 --> S6
    S6 --> |"Pass"| Done[✅ 完成]
    S6 --> |"Fail"| Rollback[回滚]
    
    %% 样式
    classDef trigger fill:#FFE4B5,stroke:#F5A623,stroke-width:2px
    classDef skill fill:#E8F5E9,stroke:#7AB648,stroke-width:2px
    classDef mcp fill:#E3F2FD,stroke:#4A90D9,stroke-width:2px
    classDef hook fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    classDef gov fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    
    class T1,T2,T3,T4 trigger
    class S1,S2,S3,S4,S5,S6,Diag skill
    class M1,M2,M3,M4,M5 mcp
    class H1,H2 hook
    class G1,G2,G3 gov
```

---

# Spec Evolution Workflow 架构说明

## 1. 层次职责

| 层 | 职责 | 实现位置 | VS Code 集成 |
|---|------|---------|-------------|
| **Triggers** | 问题发现 | 实验结果、测试输出、代码审查 | Problems Panel, Test Explorer |
| **Skills** | 认知流程编排 | `.github/prompts/dgsf_spec_*.prompt.md` | Copilot Chat 命令 |
| **MCP Tools** | 原子操作 | `kernel/mcp_server.py` | Copilot 自动调用 |
| **Hooks** | 强制检查点 | `hooks/pre-spec-change`, `hooks/post-spec-change` | Git hooks, 手动触发 |
| **Governance** | 人工审批 | `decisions/*.yaml` | PR Review, 手动创建 |

## 2. 权限矩阵

| Spec 层级 | 路径模式 | AI 可提议 | AI 可提交 | 审批者 |
|----------|---------|----------|----------|--------|
| L0 Canon | `specs/canon/*` | ❌ | ❌ | Project Owner (freeze) |
| L1 Framework | `specs/framework/*` | ✅ | ❌ | Platform Engineer |
| L2 Project | `projects/*/specs/*` | ✅ | ❌ | Project Lead |
| L3 Experiment | `experiments/*/config.yaml` | ✅ | ✅* | Auto (threshold pass) |

*L3 自动提交需通过阈值验证

## 3. 数据流

```
问题 → Triage → Research → Plan → Propose → [Approval] → Commit → Verify
                                      ↓
                              decisions/SCP-*.yaml
                                      ↓
                              ops/audit/spec_commits.yaml
                                      ↓
                              projects/dgsf/lineage/spec_changes.yaml
```

## 4. VS Code + Copilot 使用指南

### 4.1 触发 Spec Triage

在 Copilot Chat 中输入：
```
/dgsf_spec_triage
问题：实验 t05 的 OOS Sharpe = 0.8，低于阈值
来源：experiment
```

### 4.2 读取 Spec

```
读取 projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml 的 validation 部分
```

### 4.3 提出 Spec 变更

```
/dgsf_spec_propose
修改 SDF_INTERFACE_CONTRACT.yaml
将 min_sharpe_threshold 从 1.0 改为 1.5
理由：行业标准要求生产级 SDF 模型 Sharpe >= 1.5
```

### 4.4 提交变更（需先获得审批）

```
/dgsf_spec_commit
提案 ID: SCP-2026-02-04-001
审批参考: PR#42 或 decisions/APPROVED.yaml
```

## 5. 文件清单

### 新增 Skills (Prompts)
- `.github/prompts/dgsf_spec_triage.prompt.md`
- `.github/prompts/dgsf_spec_propose.prompt.md`
- `.github/prompts/dgsf_spec_commit.prompt.md`

### 新增 MCP Tools
- `spec_read` - 读取 Spec 内容
- `spec_propose` - 创建变更提案
- `spec_commit` - 提交已批准变更
- `spec_triage` - 问题分类

### 新增 Hooks
- `hooks/pre-spec-change` - 变更前验证
- `hooks/post-spec-change` - 变更后操作

### 测试文件
- `projects/dgsf/tests/test_spec_evolution_e2e.py`
- `projects/dgsf/scripts/validate_spec_workflow.py`
