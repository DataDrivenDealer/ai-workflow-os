# AEP-7: 组织扩展支持（Organizational Scaling Support）

**Status**: Draft  
**Created**: 2026-02-04  
**Addresses**: Tension B1 (人类瓶颈), B2 (多项目治理真空)

---

## 1. 动机

当前系统假设：
- 单一 Human-Agent 协作模式
- 单项目（DGSF）作为业务实例
- Agent 信任从零开始，无跨项目传递

这在机构化场景下产生：
- 人类审批成为瓶颈
- 新项目无法复用已验证 Agent
- 跨项目协调无明确机制

## 2. 提案

### 2.1 组织层引入（Organization Layer）

扩展四层架构为五层：

```
┌─────────────────────────────────────────┐
│  Organization Layer (NEW)               │
│  - 组织级策略、跨项目治理、信任传递     │
├─────────────────────────────────────────┤
│  Kernel Layer                           │
│  - 平台级不变量                         │
├─────────────────────────────────────────┤
│  Adapter Layer                          │
│  - 项目到 Kernel 绑定                   │
├─────────────────────────────────────────┤
│  Project Layer                          │
│  - 领域特定配置                         │
├─────────────────────────────────────────┤
│  Experiment Layer                       │
│  - 实验实例                             │
└─────────────────────────────────────────┘
```

### 2.2 组织配置文件

新增 `configs/organization.yaml`：

```yaml
version: "1.0.0"

# 组织身份
identity:
  org_id: "quant_research_division"
  org_name: "Quantitative Research Division"

# 组织级角色
roles:
  org_admin:
    description: "组织管理员，可管理跨项目策略"
    permissions:
      - manage_org_policy
      - grant_project_access
      - transfer_agent_trust
      
  project_owner:
    description: "项目所有者，继承自 org_admin 授权"
    permissions:
      - manage_project
      - approve_merges
      - promote_agents
      
  cross_project_coordinator:
    description: "跨项目协调员"
    permissions:
      - view_all_projects
      - allocate_shared_resources
      - mediate_conflicts

# 信任传递规则
trust_transfer:
  enabled: true
  
  # Agent 信任可在项目间传递的条件
  transfer_conditions:
    - source_level: ">= 1"  # 至少 Level 1
    - target_project_type: "same_domain"  # 同领域项目
    - transfer_discount: 0.5  # 传递后信任度折半
    
  # 传递审批
  approval:
    auto_approve: false
    approver_role: "org_admin"
    
# 资源分配
resource_allocation:
  compute:
    strategy: "fair_share"  # fair_share | priority_queue | dedicated
    quotas:
      default_per_project: "100 GPU-hours/week"
      
  data:
    shared_datasets:
      - id: "market_data_2020_2025"
        access: "read_only"
        projects: ["dgsf", "alpha_model", "risk_model"]

# 跨项目知识迁移
knowledge_transfer:
  # 演化信号是否跨项目共享
  evolution_signals:
    share_mode: "aggregated"  # none | aggregated | detailed
    anonymize: true  # 移除项目特定上下文
    
  # 成功模式传播
  pattern_propagation:
    enabled: true
    trigger: "pattern_success_rate >= 0.8"
    propagate_to: "same_domain_projects"
```

### 2.3 多项目状态管理

新增 `state/organization.yaml`：

```yaml
version: "1.0.0"

# 活跃项目列表
projects:
  - id: "dgsf"
    status: "active"
    lead: "dgsf-team"
    resource_allocation: "standard"
    
  - id: "alpha_model"
    status: "active"
    lead: "alpha-team"
    resource_allocation: "standard"

# 跨项目依赖
cross_project_dependencies:
  - from: "alpha_model"
    to: "dgsf"
    dependency_type: "data"
    artifact: "dgsf/experiments/t10_final/features.parquet"
    checksum: "sha256:abc123..."

# Agent 注册表（组织级）
agent_registry:
  - agent_id: "agent_001"
    trust_levels:
      dgsf: 2
      alpha_model: 1  # 传递后折半
    last_activity: "2026-02-04T10:00:00Z"
    
# 组织级演化信号聚合
org_evolution_signals:
  last_aggregation: "2026-02-01"
  cross_project_patterns:
    - pattern: "threshold_tension"
      affected_projects: ["dgsf", "alpha_model"]
      recommended_action: "组织级阈值校准"
```

### 2.4 委托审批机制

扩展 Authority Level 系统：

```yaml
# 在 state_machine.yaml 中扩展
authority_delegation:
  enabled: true
  
  # 委托链
  delegation_chain:
    - from: "org_admin"
      to: "project_owner"
      scope: "project_level_approvals"
      revocable: true
      
    - from: "project_owner"
      to: "senior_agent"  # Level 2 Agent
      scope: "routine_merges"
      conditions:
        - "change_scope <= 50 lines"
        - "test_coverage >= 0.9"
        - "no_protected_files"
      audit: true  # 事后审计
      
  # 紧急升级
  escalation:
    trigger: "approval_pending > 24h"
    action: "notify_org_admin"
```

## 3. 接口变更

### 3.1 project_interface.yaml 扩展

```yaml
# 新增组织绑定字段
interface:
  organization:
    optional:
      - org_id: "string"  # 所属组织
      - resource_profile: "string"  # 资源配额配置文件
      - cross_project_access: "list[project_id]"  # 可访问的其他项目
```

### 3.2 新增 Skill

```yaml
skills:
  org_coordinate:
    description: "跨项目协调操作"
    role_modes: [cross_project_coordinator]
    operations:
      - request_resource
      - transfer_artifact
      - escalate_conflict
```

## 4. 实施路线

| 阶段 | 内容 | 时间估计 |
|------|------|----------|
| P1 | 定义 organization.yaml schema | 1 周 |
| P2 | 实现信任传递逻辑 | 2 周 |
| P3 | 实现委托审批 | 2 周 |
| P4 | 集成跨项目状态管理 | 2 周 |
| P5 | 文档与迁移指南 | 1 周 |

## 5. 向后兼容

- 组织层为可选（`organization.yaml` 不存在时退化为单项目模式）
- 现有 DGSF 适配器无需修改
- 信任传递默认关闭

## 6. 风险

| 风险 | 缓解 |
|------|------|
| 组织层增加复杂度 | 提供最小化配置模板 |
| 信任传递滥用 | 强制审计 + 折扣机制 |
| 委托链失控 | 深度限制 + 撤销机制 |

---

**审批流程**: Platform Engineer → Org Admin → Apply
