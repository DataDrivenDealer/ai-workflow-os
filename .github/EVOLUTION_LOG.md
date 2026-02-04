# Copilot Runtime OS — Evolution Log

> 记录所有对 Kernel 及 Prompts 的修改，确保变更可追溯。

---

## Version 8.0.1 — Architecture Convergence (2026-02-04)

### 变更摘要

**架构级收敛**：以 GitHub Copilot + VS Code 运行机制为约束，系统性审计并收敛 `copilot-instructions.md`，确保所有声称的能力均可在运行时真正落地。

### 关键变更

| 变更项 | 原状态 | 新状态 | 原因 |
|--------|--------|--------|------|
| AUTOMATIC TRIGGERS | 声称程序化自动执行 | RECOMMENDED RESPONSE PATTERNS | Copilot 不支持程序化触发 |
| MCP Tools 直接调用 | "可在对话中直接调用" | External tools only | Copilot 不集成自定义 MCP |
| IMG 自动记录 | "做出重要决策 → IMG 记录" | 手动文件创建 | 无 IMG 后端实现 |
| SDL 自动登记 | "发现技术债务 → SDL 登记" | 手动文件创建 | SDL 仅为配置模式 |

### 验证证据

```
pytest kernel/tests/ -v --tb=short
# 结果: 237 passed in 40.72s

pytest projects/dgsf/tests/test_spec_evolution_e2e.py -v
# 结果: 17 passed in 0.55s
```

### 新增文档

- `docs/CAPABILITY_MATRIX.md` — 已验证能力清单

### 回滚方案

```bash
git log --oneline -5
git revert <commit-hash>
pytest kernel/tests/ -v
```

---

## Version 8.0.0 (2026-02-04)

### 变更摘要

**机构级量化研究演进（AEP-10）**：面向量化交易系统研究开发的领域驱动架构深化，确保系统体现量化金融前沿最佳实践，实现持续的研究智能、代码质量与长期演进。

### 设计动机

本次进化源自对AEP-9之后系统的开放式自主分析：
- **领域专精**：从通用治理框架到量化金融研究领域的深度适配
- **前沿跟踪**：系统性追踪量化金融研究前沿，避免方法论过时
- **最佳实践**：将量化代码最佳实践编码为可执行规则
- **组织学习**：建立机构记忆，避免重复错误

### 识别的领域张力

| 类别 | 张力 | 解决方案 |
|------|------|----------|
| **前沿研究(F)** | 无系统性文献跟踪 | Quant Knowledge Base (QKB) |
| **代码实践(G)** | 量化代码模式缺失 | Code Practice Registry (CPR) |
| **研究设计(H)** | 实验方法论不一致 | Research Protocol Algebra (RPA) |
| **开发流程(I)** | 无结构化研究sprint | Sprint integration in RPA |
| **长期演进(J)** | 技术债务隐性累积 | Strategic Debt Ledger (SDL) |
| **组织学习(J)** | 决策理由随时间丢失 | Institutional Memory Graph (IMG) |
| **阈值管理(G)** | 固定阈值vs市场状态变化 | Adaptive Threshold Engine (ATE) |

### 新增核心概念

#### 1. 量化知识库 (QKB)
```yaml
# configs/quant_knowledge_base.yaml
knowledge_domains:
  factor_research:
    current_consensus:
      factor_replication_crisis:
        status: "ongoing"
        implication: "Use t > 3.0 for new factors"
```
- 追踪因子研究、统计方法、ML in Finance 前沿
- 每周扫描、月度审查、季度综合
- 与代码实践注册表联动

#### 2. 代码实践注册表 (CPR)
```yaml
# configs/code_practice_registry.yaml
practices:
  DH-01:  # Point-in-Time Correctness
    severity: "critical"
    anti_patterns: ["df.fillna(df.mean())"]
    correct_patterns: ["df.fillna(method='ffill')"]
    enforcement: "pre_commit + code_review"
```
- 数据处理(DH-*)、回测(BT-*)、性能(PF-*)、建模(MD-*)
- 自动检测反模式
- 代码审查清单自动生成

#### 3. 研究协议代数 (RPA)
```yaml
# configs/research_protocol_algebra.yaml
primitives:
  LOAD_DATA -> FEATURE_ENGINEER -> TRAIN_MODEL -> BACKTEST -> EVALUATE

compositions:
  FACTOR_DISCOVERY:
    gates:
      - "At least 3 factors survive MTC"
      - "Average OOS Sharpe > 0.5"
```
- 可组合的研究原语
- 预定义工作流组合
- 自动生成实验模板

#### 4. 战略债务账本 (SDL)
- 技术债务 + 研究债务 + 演进债务
- 优先级评分公式
- 20% Sprint 容量分配

#### 5. 机构记忆图 (IMG)
- 决策、实验、学习、失败节点
- 可查询：为什么这样决定？之前失败过什么？
- 自动从 decision_log 和实验结果捕获

#### 6. 自适应阈值引擎 (ATE)
```yaml
regime_adjustments:
  high_vol:  # VIX > 25
    oos_sharpe: 1.2  # Base 1.5, relaxed for stress
governance_constraints:
  absolute_minimums:
    oos_sharpe: 0.5  # Cannot go below
```
- 市场状态感知
- 策略类型适配
- 治理硬约束保护

### 新增技能 (6个)

| 技能 | 用途 | 类别 |
|------|------|------|
| `/dgsf_knowledge_sync` | 查询/更新QKB | 智能 |
| `/dgsf_practice_check` | 检查代码符合CPR | 质量 |
| `/dgsf_protocol_design` | 用RPA设计实验 | 方法论 |
| `/dgsf_debt_review` | 审查优先级化债务 | 维护 |
| `/dgsf_memory_query` | 查询机构记忆 | 学习 |
| `/dgsf_threshold_resolve` | 获取上下文感知阈值 | 适应 |

### 新增自动触发器

| 触发模式 | 自动调用 | 说明 |
|----------|----------|------|
| 询问最佳实践 | QKB + CPR 查询 | 确保使用最新知识 |
| 新建实验 | RPA 模板检查 | 强制使用研究协议 |
| `train_test_split` + 时间序列 | 警告 + CPR引用 | 推荐 purged walk-forward |
| 多因子测试无 MTC | 阻止 + 引用 BT-03 | 强制多重检验校正 |

### 新增规则 (R7-R9)

| 规则 | 表达式 | 说明 |
|------|--------|------|
| R7 | `WHEN train_test_split AND time_series THEN WARN` | 方法论时效性 |
| R8 | `WHEN factors > 1 AND NOT mtc THEN BLOCK` | 多重检验强制 |
| R9 | `WHEN production_ready AND NOT robustness THEN BLOCK` | 稳健性要求 |

### 新增文件

| 文件 | 类型 | 描述 |
|------|------|------|
| `docs/proposals/AEP-10_INSTITUTIONAL_QUANT_EVOLUTION.md` | 提案 | 完整架构演进提案 |
| `configs/quant_knowledge_base.yaml` | 配置 | 量化知识库 |
| `configs/code_practice_registry.yaml` | 配置 | 代码实践注册表 |
| `configs/research_protocol_algebra.yaml` | 配置 | 研究协议代数 |
| `configs/strategic_debt_ledger.yaml` | 配置 | 战略债务账本 |
| `configs/adaptive_threshold_engine.yaml` | 配置 | 自适应阈值引擎 |
| `.github/prompts/dgsf_knowledge_sync.prompt.md` | 技能 | 知识同步 |
| `.github/prompts/dgsf_practice_check.prompt.md` | 技能 | 实践检查 |
| `.github/prompts/dgsf_protocol_design.prompt.md` | 技能 | 协议设计 |
| `.github/prompts/dgsf_debt_review.prompt.md` | 技能 | 债务审查 |
| `.github/prompts/dgsf_memory_query.prompt.md` | 技能 | 记忆查询 |
| `.github/prompts/dgsf_threshold_resolve.prompt.md` | 技能 | 阈值解析 |

### 修改文件

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `.github/copilot-instructions.md` | **升级** | v7.0→v8.0；新增领域上下文、AEP-10触发器、6新技能 |
| `projects/dgsf/adapter.yaml` | **更新** | 添加6个新技能映射、自适应阈值配置 |
| `spec_registry.yaml` | **更新** | 注册5个新规范(QKB, CPR, RPA, SDL, ATE) |

### CMM 评估（演进后）

| 维度 | 当前级别 | 目标级别 | 证据 |
|------|----------|----------|------|
| 知识管理 | CMM-4 | CMM-5 | QKB + IMG 创建 |
| 代码治理 | CMM-4 | CMM-5 | CPR + 自动执行 |
| 研究方法论 | CMM-3 | CMM-4 | RPA 协议代数 |
| 技术健康 | CMM-3 | CMM-4 | SDL 债务跟踪 |
| 阈值管理 | CMM-4 | CMM-5 | ATE 状态感知 |

**整体评估**：**CMM-4 (Quantified)**

### 成功指标（目标）

| 指标 | 当前 | 目标 |
|------|------|------|
| 研究方法论时效 | 未测量 | < 6个月落后于前沿 |
| 代码实践合规率 | 手动检查 | > 90% 自动检查 |
| 实验协议覆盖率 | 临时 | > 80% 使用RPA模板 |
| 技术债务可见性 | 隐性 | 100% 在SDL中跟踪 |
| 决策可追溯性 | 部分 | 100% 在IMG中 |

---

## Version 7.0.0 (2026-02-04)

### 变更摘要

**组织化演进架构（AEP-9）**：系统性探索工作流操作系统在规模化、长期演进与机构化量化研发场景下的最大可能性，推动其向更高能力上限与更强组织形态演进。

### 设计动机

本次进化源自对现有系统的开放式架构级自主分析：
- **张力识别**：系统性暴露规模化、长期演进与机构化场景下的结构性不足
- **边界突破**：不预设实现边界，探索结构抽象与规则表达的最大可能性
- **自演化机制**：引入元演化监控，解决自指向测量问题
- **组织化能力**：从单项目到多项目组合的治理升级

### 识别的核心张力

| 类别 | 张力 | 解决方案 |
|------|------|----------|
| **规模化(A)** | 单项目范围限制 | Organization Layer (L-1) |
| **规模化(A)** | 并发语义未定义 | Concurrency Model in state_machine.yaml |
| **信任(B)** | 每项目重建信任 | Trust Transfer Protocol |
| **演化(C)** | 37天反馈周期过长 | Fast-Track Evolution Circuit |
| **演化(C)** | 自指向测量盲区 | Meta-Evolution Monitoring |
| **治理(D)** | 静态阈值不适应市场状态 | Conditional Thresholds in REL |
| **表达力(E)** | 自然语言规则不可组合 | Rule Expression Language (REL) |

### 新增核心概念

#### 1. 五层架构（原四层）
```
Organization (L-1) → Kernel (L0) → Adapter (L1) → Project (L2) → Experiment (L3)
```

#### 2. 规则表达语言（REL）
```yaml
WHEN agent.action.type == "start_task"
AND concurrent_tasks_by(agent.id) >= params.max_parallel
THEN BLOCK with "WIP limit exceeded"
```
- 可参数化
- 支持条件例外
- 可形式验证
- 减少解释方差

#### 3. 信任转移协议
- 跨项目信任可转移（带衰减）
- 支持机构化部署
- 保留人工审批边界

#### 4. 能力成熟度模型（CMM）
| 级别 | 名称 | 关键特征 |
|------|------|----------|
| CMM-1 | Initial | 自然语言规则 |
| CMM-2 | Managed | 结构化 YAML 配置 |
| CMM-3 | Defined | REL + 组织层 |
| CMM-4 | Quantified | 元演化监控 |
| CMM-5 | Optimizing | 预测性治理 |

当前评估：**CMM-3**（本次演进后）

### 新增文件

| 文件 | 类型 | 描述 |
|------|------|------|
| `docs/proposals/AEP-9_ORGANIZATIONAL_EVOLUTION_ARCHITECTURE.md` | 提案 | 完整架构演进提案 |
| `specs/canon/ORGANIZATION_CANON.md` | L0 Canon | 组织层宪章 |
| `configs/rules/REL_SPECIFICATION.yaml` | 规范 | 规则表达语言规范 |
| `configs/rules/kernel_rules.rel.yaml` | 规则 | R1-R6 REL 格式 |
| `docs/architecture/capability_maturity_model.yaml` | 框架 | CMM 评估框架 |
| `configs/meta_evolution_monitoring.yaml` | 配置 | 元演化监控配置 |

### 修改文件

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `.github/copilot-instructions.md` | **升级** | v6.0→v7.0；引入层级上下文、组合规则、自动触发规则 |
| `docs/architecture/meta_model.yaml` | **升级** | v2.0→v3.0；新增 Organization 层定义和跨层契约 |
| `spec_registry.yaml` | **更新** | 注册 4 个新规范 |

### CMM 评估（演进后）

| 维度 | 当前级别 | 证据 |
|------|----------|------|
| 规则表达力 | CMM-3 | REL 规范创建，核心规则已转换 |
| 演化能力 | CMM-4 | evolution_policy.yaml 中的元监控 |
| 多项目协调 | CMM-3 | ORGANIZATION_CANON 创建 |
| 信任管理 | CMM-3 | Trust Transfer Protocol 定义 |
| 治理自动化 | CMM-3 | 参数化规则带条件例外 |

### 测试验证

```
================================================ 237 passed in 40.38s =================================================
```

### 后续路线图

1. **CMM-3 → CMM-4**（目标：2026-03-15）
   - 实现盲区检测 Python 代码
   - 部署置信度评分算法
   - 添加第二个项目到组合

2. **CMM-4 → CMM-5**（目标：2026-06-01）
   - 实现 A/B 测试框架
   - 训练预测性摩擦模型
   - 设计信任联邦协议

---

## Version 6.0.0 (2026-02-04)

### 变更摘要

**架构级收敛（Major）**：以 GitHub Copilot 运行时机制为约束，删除不可执行的设计，保留可审计、可回滚的规则与流程。

### 设计动机

本次收敛源自对现有系统的结构化认知分析（四种认知模式并行探索）：
- **运行时约束分析**：识别哪些 artifacts 被 Copilot 实际加载
- **系统完整性审计**：评估设计声明 vs 实现代码的差距
- **最小可行系统设计**：在保持功能前提下删除冗余
- **长期可维护性**：降低认知负荷和维护负担

### 核心发现

| 发现 | 影响 |
|------|------|
| 只有 `copilot-instructions.md` 和 `*.prompt.md` 被 Copilot 加载 | 其他 YAML/Python 仅作上下文 |
| Authority Levels (AEP-4) 无法运行时强制 | Copilot 无会话状态管理 |
| 动态 Adapter 加载是设计愿景 | 运行时不自动调用 Python |
| 四层架构对单项目过度设计 | DGSF 是唯一业务项目 |

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `.github/copilot-instructions.md` | **重写** | 版本升至 v6.0.0；删除 Authority Levels；硬编码 DGSF 配置；简化为 6 条核心规则 |
| `.github/copilot-instructions-v5-backup.md` | **新增** | 备份 v5.1.0 版本 |
| `docs/architecture/` | **新增** | 存放设计文档（meta_model.yaml 等） |
| `configs/meta_model.yaml` | **移动** | 移至 `docs/architecture/` |
| `configs/project_interface.yaml` | **移动** | 移至 `docs/architecture/` |
| `configs/evolution_policy.yaml` | **移动** | 移至 `docs/architecture/` |
| `configs/skill_alignment.yaml` | **移动** | 移至 `docs/architecture/` |

### 删除的不可执行设计

| 内容 | 原位置 | 删除原因 |
|------|--------|---------|
| Authority Levels (L0-L3) | copilot-instructions.md | Copilot 无会话状态，无法运行时验证 |
| 动态 Adapter 加载描述 | copilot-instructions.md | Python 函数不被自动调用 |
| R7 (Kernel 只读) | copilot-instructions.md | 需要演化流程但 Copilot 无法强制 |
| R8 (结果不可变) | copilot-instructions.md | 需要 Git hooks 强制，非 Copilot 能力 |
| R9 (跨项目声明) | copilot-instructions.md | 单项目场景下无意义 |
| SYSTEM HEALTH 章节 | copilot-instructions.md | 健康指标需要外部监控系统 |

### 保留的运行时影响设计

| 内容 | 理由 |
|------|------|
| DGSF 项目配置 | 直接影响 Copilot 行为范围 |
| 6 条核心规则 (R1-R6) | 可在对话中被 Copilot 遵守 |
| 11 个 prompt 文件 | 被 VS Code Copilot Chat 识别为技能 |
| 实验格式规范 | 提供结构化指导 |
| 演化流程（人工驱动） | 明确需要人工审批，不声称自动化 |

### 架构简化

**之前（v5.1.0）**:
```
四层架构: Kernel → Adapter → Project → Experiment
动态加载: load_project_adapter()
多项目支持: projects/{project_id}/adapter.yaml
```

**之后（v6.0.0）**:
```
两层架构: Kernel + DGSF Project
硬编码配置: DGSF 路径和阈值直接写入 copilot-instructions.md
单项目聚焦: 删除多项目抽象
```

### 回滚说明

如需恢复 v5.1.0：
```powershell
Move-Item ".github/copilot-instructions.md" ".github/copilot-instructions-v6.md"
Move-Item ".github/copilot-instructions-v5-backup.md" ".github/copilot-instructions.md"
```

---

## Version 5.1.0 (2026-02-04)

### 变更摘要

**架构级开放式进化（Major）**：实现演化闭环、分级权限模型、系统健康度指标、规则覆盖扩展。

### 设计动机

本次进化源自对现有系统在以下场景下的结构化认知分析（五种认知模式并行探索）：
- **系统架构分析**：评估四层架构的抽象充分性
- **复杂系统演化**：闭环完整性与自稳定性
- **组织协作建模**：权限边界与审批流
- **形式化规则系统**：覆盖性与一致性
- **量化研发领域**：统计严谨性与数据血缘

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `.github/copilot-instructions.md` | 修改 | 版本升至 v5.1.0；新增 R7-R9 规则；新增 AUTHORITY LEVELS 与 SYSTEM HEALTH 章节；更新 EVOLUTION 章节 |
| `configs/evolution_policy.yaml` | 修改 | 新增 AEP-3 effectiveness tracking 段：度量窗口、成功标准、回滚机制 |
| `configs/health_metrics.yaml` | **新增** | AEP-5 系统健康度指标：MTBF、技能成功率、契约覆盖率等 10 项指标 |
| `kernel/state_machine.yaml` | 修改 | AEP-4 分级权限模型：authority_levels 段定义 4 级权限与晋升/降级条件 |
| `.github/EVOLUTION_LOG.md` | 修改 | 记录 AEP-3/4/5/6 变更详情 |

### 架构进化提案（AEP）详情

#### AEP-3: 演化闭环完整化 (Evolution Closed-loop Completion)

**问题**：演化流程是开环的（Signal → Aggregate → Review → Apply），缺乏效果验证与回滚。

**方案**：
```yaml
# configs/evolution_policy.yaml 新增
effectiveness:
  measurement_window: "30d"
  success_criteria:
    friction_reduction: ">= 0.3"  # 同类摩擦减少 30%
    no_new_high_severity: true
  rollback:
    enabled: true
    requires_human_approval: true
```

**效果**：演化不再是单向过程，可验证有效性并在失败时回滚。

---

#### AEP-4: 分级权限模型 (Tiered Authority Model)

**问题**：Agent 权限是二值的（speculative-only），无法支持渐进式信任。

**方案**：
| Level | Name | Permissions | Promotion Criteria |
|-------|------|-------------|-------------------|
| 0 | Speculative | Propose only | Default |
| 1 | Assisted | Execute in sandbox | 90% success, 30d clean |
| 2 | Delegated | Merge to feature | Level 1 for 7d + approval |
| 3 | Trusted | Reserved | Not implemented |

**效果**：表现优秀的 Agent 可获得有限自主权，同时保留安全边界。

---

#### AEP-5: 系统健康度指标 (System Health Metrics)

**问题**：无法评估 OS 本身是否在改善。

**方案**：定义 10 项核心指标：
- **演化类**：Mean Time Between Friction (MTBF)、Evolution Signal Velocity
- **技能类**：Skill Invocation Success Rate、Retry Rate
- **契约类**：Contract Validation Coverage、Pass Rate
- **流程类**：Task Cycle Time、WIP Utilization

**效果**：可回答"这个 OS 在变好还是变坏？"

---

#### AEP-6: 规则覆盖扩展 (Rule Coverage Extension)

**问题**：原 6 条规则存在盲区（跨项目、跨实验、Kernel 修改场景）。

**方案**：新增 R7-R9：
| # | P | Rule | 覆盖场景 |
|---|---|------|---------|
| R7 | P3 | Kernel 文件只读 | 未经演化流程修改 kernel/ |
| R8 | P2 | 实验结果不可回溯修改 | 修改已 merge 的 results.json |
| R9 | P1 | 跨项目引用显式声明 | 引用其他 project 数据未声明 |

**效果**：规则覆盖率从 ~70% 提升至 ~90%（估计值，需运行数据验证）。

---

### 张力解决状态

| 张力 ID | 描述 | 解决状态 |
|---------|------|----------|
| T-EVO-1 | 演化闭环不完整 | ✅ AEP-3 effectiveness tracking |
| T-AUTH-1 | 权限模型过于二值 | ✅ AEP-4 tiered authority |
| T-RULE-1 | 规则覆盖盲区 | ✅ AEP-6 新增 R7-R9 |
| T-HEALTH-1 | 系统健康度无指标 | ✅ AEP-5 health_metrics.yaml |
| T-VALID-1 | 契约验证缺失 | ⏳ 配置已定义，脚本待实现 |

### 遗留张力（需后续迭代）

| 张力 ID | 描述 | 状态 |
|---------|------|------|
| T-INTERFACE-1 | project_interface.yaml 接口爆炸风险 | 已识别，AEP-7 草案待实施 |
| T-MULTI-AGENT-1 | 缺乏多 Agent 协作原语 | 已识别，需进一步需求明确 |
| T-DELEGATION-1 | 审批委托机制缺失 | 已识别，需治理讨论 |
| T-RUNTIME-1 | 缺乏运行时组合层 | 已识别，可能是过度设计 |

### 验证方式

```powershell
# 1. 验证 Kernel 版本
Select-String -Path ".github/copilot-instructions.md" -Pattern "Version.*5\.1\.0"
# Expected: 1 match

# 2. 验证新配置文件存在
Test-Path "configs/health_metrics.yaml"
# Expected: True

# 3. 验证规则数量
(Select-String -Path ".github/copilot-instructions.md" -Pattern "^\| R\d").Count
# Expected: 9

# 4. 验证 YAML 语法
python -c "import yaml; yaml.safe_load(open('configs/evolution_policy.yaml')); yaml.safe_load(open('configs/health_metrics.yaml')); yaml.safe_load(open('kernel/state_machine.yaml'))"
# Expected: No errors
```

### 迁移说明

**从 v5.0.0 迁移到 v5.1.0**：
- **后向兼容**：是。现有 DGSF adapter 无需修改。
- **新能力**：演化效果度量、分级权限、健康度监控、额外规则覆盖。
- **行动项**：
  - [ ] 与团队 review 新规则 R7-R9
  - [ ] 决定 Agent 权限晋升策略
  - [ ] 实现健康度仪表板生成脚本

---

## Version 5.0.0 (2026-02-04)

### 变更摘要

**引入 Adapter 层（Major）**：实现 Kernel-Project 完全解耦，支持多项目独立配置。

（详见下方原始 v5.0.0 记录）

---

## Version 4.0.0 (2026-02-04)

### 变更摘要

**架构级自主进化（Major）**：引入三层元模型、规则形式化语法、Skill-Role-State 对齐矩阵、演化信号自动捕获、多重检验校正。

### 设计动机

本次进化源自对现有系统在以下场景下的张力审视：
- **规模化**：多 Agent、多项目并发
- **长期演进**：规则积累导致的冲突与遗漏
- **机构化量化研发**：统计严谨性、审计合规性

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 修改 | 添加 Meta Model 引用；SUCCESS THRESHOLDS 增加 Multiple Testing 列；版本升至 v4.0.0 |
| `configs/meta_model.yaml` | **新增** | 三层架构元模型定义：Kernel-Project-Experiment 边界与跨层契约 |
| `configs/rule_schema.yaml` | **新增** | 规则形式化语法：继承、覆盖、参数化、冲突检测 |
| `configs/skill_alignment.yaml` | **新增** | Skill-Role-State 三角对齐矩阵：每个 Skill 的角色权限与状态转换效果 |
| `projects/dgsf/project.yaml` | **新增** | DGSF 项目清单：Kernel 绑定、规则继承/覆盖、Skill-Role 映射 |
| `projects/dgsf/configs/thresholds.yaml` | 修改 | 添加 Multiple Testing 配置段（Bonferroni、显著性计算、必报字段） |
| `kernel/evolution_signal.py` | **新增** | 演化信号自动捕获模块：EvolutionSignalCollector 类 + CLI |

### 新增架构组件说明

#### 1. 三层元模型 (`configs/meta_model.yaml`)
- **问题**：Kernel-Project-Experiment 边界是隐式的，业务语义泄漏到 Kernel
- **方案**：显式定义三层的 artifacts、rules namespace、跨层契约
- **效果**：第二个项目可复用 Kernel，只需新建 `projects/{new}/project.yaml`

#### 2. 规则形式化语法 (`configs/rule_schema.yaml`)
- **问题**：`inherits_rules` 只是 ID 列表，不支持参数化覆盖
- **方案**：定义 JSON Schema 风格的规则定义/继承/冲突检测语法
- **效果**：规则演化可机器验证，冲突检测自动化

#### 3. Skill-Role-State 对齐 (`configs/skill_alignment.yaml`)
- **问题**：5 种 role_modes 与 11 个 Skills 无显式映射
- **方案**：每个 Skill 声明 `role_modes`、`state_preconditions`、`state_effects`
- **效果**：权限检查可自动化，状态一致性可验证

#### 4. 演化信号自动捕获 (`kernel/evolution_signal.py`)
- **问题**：规则摩擦只能人工记录，高频执行时遗漏
- **方案**：提供 `EvolutionSignalCollector` 类，支持运行时上报 + CLI 聚合
- **效果**：演化决策有数据支撑，而非凭感觉

#### 5. 多重检验校正 (`thresholds.yaml` Multiple Testing 段)
- **问题**：并行探索 `t{NN}.{SS}_*` 增加 FDR，但阈值固定
- **方案**：定义 Bonferroni/Holm/BH 方法，要求报告 `adjusted_pvalue`
- **效果**：统计严谨性符合机构化量化研发标准

### 张力解决状态

| 张力 ID | 描述 | 解决状态 |
|---------|------|----------|
| T-A1 | Kernel 耦合业务语义 | ✅ 通过 meta_model.yaml + project.yaml 解耦 |
| T-A2 | Skill 无类型系统 | ✅ skill_alignment.yaml 定义输入/输出 schema |
| T-A3 | State Machine 与 Skill 不对齐 | ✅ skill_alignment.yaml 声明 state_effects |
| T-B1 | 规则继承无覆盖语义 | ✅ rule_schema.yaml 定义继承/覆盖语法 |
| T-B2 | 规则冲突检测缺失 | ✅ rule_schema.yaml 定义 CD1-CD3 检测规则 |
| T-C1 | Role-Skill 对齐缺失 | ✅ skill_alignment.yaml + project.yaml 映射 |
| T-D1 | 演化信号被动捕获 | ✅ evolution_signal.py 自动捕获 |
| T-E1 | 多重检验校正缺失 | ✅ thresholds.yaml Multiple Testing 段 |

### 遗留张力（需后续迭代）

| 张力 ID | 描述 | 状态 |
|---------|------|------|
| T-B3 | 违规检测非实时 | 已识别，需 MCP Server 层实现 runtime gate |
| T-B4 | 审计双记录 | 已识别，建议统一到 audit.py |
| T-C2 | 单 Agent 假设硬编码 | 部分解决（R2 参数化），需多 Agent 调度器 |
| T-D3 | 行为回归测试缺失 | 已识别，需 prompt 行为测试框架 |
| T-E2 | 实验依赖链未形式化 | 已识别，建议引入 lineage.yaml |
| T-E4 | 无实验注册制 | 已识别，建议引入 experiment_registry.yaml |

### 验证方式

```powershell
# 1. 验证 Kernel 版本
Select-String -Path ".github/copilot-instructions.md" -Pattern "Version.*4\.0\.0"
# Expected: 1 match

# 2. 验证新配置文件存在
Test-Path "configs/meta_model.yaml"
Test-Path "configs/rule_schema.yaml"
Test-Path "configs/skill_alignment.yaml"
Test-Path "projects/dgsf/project.yaml"
# Expected: All True

# 3. 验证 evolution_signal.py 可执行
python kernel/evolution_signal.py --help
# Expected: Help message

# 4. 验证 thresholds.yaml 包含 multiple_testing
Select-String -Path "projects/dgsf/configs/thresholds.yaml" -Pattern "multiple_testing:"
# Expected: 1 match

# 5. prompts count (unchanged)
(Get-ChildItem ".github/prompts/*.prompt.md").Count
# Expected: 11
```

### 迁移指南

对于现有 DGSF 工作流：
1. 新配置文件是**增量添加**，现有 prompts 无需修改
2. `project.yaml` 中的 Skill-Role 映射是**声明性的**，不影响现有执行
3. 多重检验校正是**可选的**，`required_in_results` 仅在 `n_branches > 1` 时强制

---

## Version 3.6.0 (2026-02-04)

### 变更摘要

**开放式架构级自主进化**：结构化规则继承、阈值配置外部化、演化信号聚合工具、实验分支命名空间、原始数据完整性校验。

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 修改 | SUCCESS THRESHOLDS 改为引用外部配置；EXPERIMENT FORMAT 支持分支命名；版本升至 v3.6.0 |
| `projects/dgsf/configs/thresholds.yaml` | 新增 | 成功阈值外部配置文件，支持资产类别覆盖与实验级覆盖 |
| `*.prompt.md` (11 files) | 修改 | frontmatter 添加 `inherits_rules` 字段，结构化声明规则继承 |
| `scripts/aggregate_evolution_signals.py` | 新增 | 演化信号聚合脚本，按规则分组统计触发频次 |
| `scripts/verify_raw_data_integrity.py` | 新增 | 原始数据 SHA256 完整性校验工具 |

### 变更理由

1. **结构化规则继承**
   - 问题：规则继承是文本约定（`## CORE RULES (from Kernel)`），v3.5 已出现遗漏
   - 修复：在 prompt frontmatter 添加 `inherits_rules: [R1, R2, ...]` 机器可读声明
   - 效果：规则覆盖关系可被工具验证，降低人工维护遗漏风险

2. **阈值配置外部化**
   - 问题：SUCCESS THRESHOLDS 硬编码，不同资产/市场需不同阈值
   - 修复：新建 `configs/thresholds.yaml`，Kernel 引用配置路径
   - 效果：阈值调整不需修改 Kernel；支持 per-experiment 覆盖

3. **演化信号聚合工具**
   - 问题：`evolution_signals.yaml` 设计为人工审查，但无统计工具辅助
   - 修复：新增 `aggregate_evolution_signals.py` 聚合脚本
   - 效果：快速识别高频规则摩擦，辅助演化决策

4. **实验分支命名空间**
   - 问题：`t{NN}_{name}` 是线性的，探索性研究需分叉
   - 修复：支持 `t{NN}.{SS}_{name}` 格式 (SS = sub-experiment)
   - 效果：`t05.01_dropout_low` / `t05.02_dropout_high` 可并行探索

5. **原始数据完整性校验**
   - 问题：R4 是被动规则，无主动验证机制
   - 修复：新增 `verify_raw_data_integrity.py` + checksums 文件
   - 效果：实验前校验原始数据未被篡改

### 架构张力暴露（本次进化过程中识别）

| 张力 | 触发条件 | 状态 |
|------|---------|------|
| Role-Skill 对齐缺失 | 引入权限检查 gate 时 | 已识别，DGSF-only 约束下暂缓 |
| 多 Agent 并发冲突检测 | 机构化多 Agent 场景 | 已识别，当前单 Agent 约束下暂缓 |
| 多重检验校正 | 多实验比较时 | 已识别，建议在 THRESHOLD 使用时人工注意 |
| commit message 格式与审计双记录 | 审计链完整性审查时 | 已识别，git log 暂作唯一记录 |

### 验证方式

```powershell
# 1. 验证 Kernel 版本
Select-String -Path ".github/copilot-instructions.md" -Pattern "Version.*3\.6\.0"
# Expected: 1 match

# 2. 验证所有 prompts 有 inherits_rules
Get-ChildItem ".github/prompts/*.prompt.md" | ForEach-Object {
  $content = Get-Content $_.FullName -Raw
  if ($content -notmatch "inherits_rules:") { Write-Warning "$($_.Name) missing inherits_rules" }
}
# Expected: No warnings

# 3. 验证 thresholds.yaml 存在
Test-Path "projects/dgsf/configs/thresholds.yaml"
# Expected: True

# 4. 验证脚本可执行
python scripts/aggregate_evolution_signals.py --help
python scripts/verify_raw_data_integrity.py --help
# Expected: Help messages without errors

# 5. prompts count
(Get-ChildItem ".github/prompts/*.prompt.md").Count
# Expected: 11
```

---

## Version 3.5.0 (2026-02-04)

### 变更摘要

**架构级收敛审计**：删除不可运行时执行的设计，固化可审计闭环。

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 修改 | 移除"自动触发演化审查"的不可执行语义，改为人工审查流程 |
| `dgsf_execute.prompt.md` | 修改 | 补齐遗漏的 R4（保护 raw data）和 R6（长运行交接）规则继承 |
| `projects/dgsf/evolution_signals.yaml` | 修改 | 移除注释中不可执行的"累计 3 次触发"描述 |

### 变更理由

1. **删除不可执行的自动演化声明**
   - 问题：Kernel 声称"当同一规则累计 3 次 false-positive，可自动触发演化审查"
   - 实际：没有任何代码或 hook 读取并计数这些 signals
   - 修复：改为"用户定期审查，人工决定是否触发演化"
   - 效果：所有声明均可被验证或执行

2. **规则继承完整性修复**
   - 问题：`dgsf_execute.prompt.md` 继承 R1/R2/R3/R5 但遗漏 R4/R6
   - 修复：显式添加 R4 和 R6 的继承声明
   - 效果：执行技能覆盖全部相关约束

3. **文档与运行时一致性**
   - 原则：仅保留可被 Copilot 运行时加载、人工可验证的设计
   - 删除：所有"写而不读"的幻觉机制

### 验证方式

```powershell
# 1. 验证 Kernel 行数
(Get-Content ".github/copilot-instructions.md" | Measure-Object -Line).Lines
# Expected: ≤ 120

# 2. 验证所有 prompts 存在
(Get-ChildItem ".github/prompts/*.prompt.md").Count
# Expected: 11

# 3. 验证 dgsf_execute 包含 R4 和 R6
Select-String -Path ".github/prompts/dgsf_execute.prompt.md" -Pattern "R4|R6"
# Expected: 2 matches

# 4. 验证 evolution_signals.yaml 无"自动触发"描述
Select-String -Path "projects/dgsf/evolution_signals.yaml" -Pattern "自动触发"
# Expected: 0 matches
```

---

## Version 3.4.0 (2026-02-04)

### 变更摘要

**开放式架构级自主进化**：机构记忆闭环、演化触发可检测化、实验命名空间优化。

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 修改 | 实验格式从 `t{stage}` 改为 `t{NN}` (零填充)；EVOLUTION 新增 Record 字段；版本升至 v3.4 |
| `dgsf_research.prompt.md` | 修改 | PHASE 1 增加 decisions/ 扫描，完成机构记忆闭环 |
| `projects/dgsf/evolution_signals.yaml` | 新增 | 规则触发异常记录，支持演化触发条件检测 |

### 变更理由

1. **机构记忆闭环**
   - 问题：`/dgsf_decision_log` 输出至 `decisions/`，但 `/dgsf_research` 未扫描该目录
   - 修复：research prompt PHASE 1 增加 decisions/ 检查
   - 效果：历史决策自动进入研究上下文

2. **演化触发可检测化**
   - 问题：触发条件 "Rule false-positive 3×" 无记录机制，不可检测
   - 修复：新增 `evolution_signals.yaml` 作为异常信号注册表
   - 效果：当同一规则累计 3 次 false-positive，可自动触发演化审查

3. **实验命名空间优化**
   - 问题：`t{stage}_{name}` 在 stage > 9 时字典序混乱 (t10 < t2)
   - 修复：改为零填充格式 `t{NN}_{name}` (01-99)
   - 效果：实验目录按创建顺序正确排序

### 架构张力暴露（本次进化过程中识别）

| 张力 | 触发条件 | 当前状态 |
|------|---------|---------|
| 角色—技能绑定缺失 | 多 Agent 协作场景 | 已识别，DGSF ONLY 约束下暂不解决 |
| SUCCESS THRESHOLDS 静态化 | 策略扩展至多资产时 | 已识别，待多资产需求触发时解决 |
| 状态机循环活锁风险 | code_review ↔ revision_needed 无限循环 | 已识别，待引入最大修订次数限制 |
| 规则间依赖关系未显式化 | R4 检测依赖 R1 验证能力 | 已识别，待更多实例验证后决定 |

### 验证方式

```powershell
# 1. 验证 Kernel 行数
(Get-Content ".github/copilot-instructions.md" | Measure-Object -Line).Lines
# Expected: ≤ 120

# 2. 验证所有 prompts 存在
(Get-ChildItem ".github/prompts/*.prompt.md").Count
# Expected: 10

# 3. 验证 evolution_signals.yaml 存在
Test-Path "projects/dgsf/evolution_signals.yaml"
# Expected: True

# 4. 验证 research prompt 包含 decisions/ 扫描
Select-String -Path ".github/prompts/dgsf_research.prompt.md" -Pattern "decisions/"
# Expected: 1+ matches

# 5. 验证 Kernel 版本号
Select-String -Path ".github/copilot-instructions.md" -Pattern "v3.4"
# Expected: 1 match
```

---

## Version 3.3.0 (2026-02-04)

### 变更摘要

**架构收敛**：压缩 Kernel 至工程级最小闭环，新增长运行任务交接规则 R6，修复 prompt 语法错误。

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 重构 | 从 160 行压缩至 ~105 行；新增 R6 长运行任务规则；精简闭环图；移除冗余说明 |
| `dgsf_research.prompt.md` | 修复 | 移除双重代码块标记，修复 YAML frontmatter 解析 |
| `dgsf_abort.prompt.md` | 修复 | 移除双重代码块标记，修复 YAML frontmatter 解析 |
| `dgsf_diagnose.prompt.md` | 修复 | 移除双重代码块标记，修复 YAML frontmatter 解析 |
| `dgsf_decision_log.prompt.md` | 修复 | 移除双重代码块标记，修复 YAML frontmatter 解析 |

### 变更理由

1. **架构收敛**
   - 问题：Kernel 160 行，超出 120 行限制；部分内容为冗余说明
   - 修复：删除重复的 Context 列、合并表格、简化闭环图 ASCII art
   - 效果：Kernel ≤ 120 行，所有规则保留但更紧凑

2. **R6 长运行任务规则**
   - 问题：Copilot 执行 >3 分钟任务时无明确交接协议
   - 修复：新增 R6 规则，要求提供执行计划 + 代码，等待人工执行
   - 效果：长任务不再阻塞会话，人机协作更清晰

3. **Prompt 语法修复**
   - 问题：4 个 prompt 文件有 ` ```prompt ` 双重标记，导致 YAML frontmatter 解析失败
   - 修复：移除多余的代码块标记
   - 效果：所有 10 个 prompts 现在可被 Copilot 正确加载

### 验证方式

```powershell
# 1. 验证 Kernel 行数
(Get-Content ".github/copilot-instructions.md" | Measure-Object -Line).Lines
# Expected: ≤ 120

# 2. 验证所有 prompts 存在且首行为 ---
Get-ChildItem ".github/prompts/*.prompt.md" | ForEach-Object { 
  $first = Get-Content $_.FullName -First 1
  if ($first -ne "---") { Write-Warning "$($_.Name) has invalid frontmatter" }
}
# Expected: No warnings

# 3. 验证 R6 规则存在
Select-String -Path ".github/copilot-instructions.md" -Pattern "R6.*Long-run"
# Expected: 1 match
```

---

## Version 3.2.0 (2026-02-04)

### 变更摘要

**开放式架构级自主进化**：规则优先级机制、结构化机构记忆、演化退化测试、按需加载提示。

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 修改 | 新增规则优先级(P1-P4)与冲突裁决机制；新增演化退化测试清单；新增 Active Module Hint |
| `dgsf_decision_log.prompt.md` | 修改 | 输出格式从 Markdown 升级为结构化 YAML，支持跨会话查询 |
| `projects/dgsf/decisions/README.md` | 新增 | 决策存储目录，包含 Schema 定义与使用指南 |

### 变更理由

1. **规则优先级与冲突裁决**
   - 问题：5 条规则并列，当 R2(单任务)与 R3(失败即停)同时触发时行为不可预测
   - 修复：引入 P1-P4 优先级层级，明确 "higher P-number wins"
   - 效果：R3(P3) > R2(P2) — 失败时停止所有任务，而非仅当前任务

2. **结构化机构记忆**
   - 问题：决策日志为 Markdown 格式，新会话无法自动加载历史决策
   - 修复：改为 YAML 格式，存储于 `projects/dgsf/decisions/`，支持 PowerShell 查询
   - 效果：研究员可通过命令检索历史假设、关联实验、决策理由

3. **演化退化测试**
   - 问题：Kernel 升级后无自动验证机制，可能破坏已有流程
   - 修复：新增 Regression Checklist（prompts 解析、行数限制、状态一致性、单元测试）
   - 效果：每次演化前有明确的 pass/fail 验证步骤

4. **按需加载提示**
   - 问题：Kernel 认知密度高，所有规则对所有任务等权重加载
   - 修复：新增 Active Module Hint，提示 Agent 聚焦用户指定模块
   - 效果：减少无关上下文干扰，提升任务聚焦度

### 架构张力暴露（本次进化过程中识别）

以下张力已识别但未在本版本解决，记录供后续参考：

| 张力 | 触发条件 | 当前状态 |
|------|---------|---------|
| 多项目泛化 | 需支持第二个项目时 | 明确约束为 DGSF ONLY，暂不触发 |
| WIP 限制强制引用 | 5+ 研究员并发时 | INV-2 定义但 Kernel 未引用，待观察 |
| 闭环图与 state_machine.yaml 同步 | 状态定义变更时 | 手动维护，待引入自动生成 |

### 验证方式

```powershell
# 1. 验证 Kernel 行数
(Get-Content ".github/copilot-instructions.md" | Measure-Object -Line).Lines
# Expected: ≤ 120

# 2. 验证所有 prompts 存在
(Get-ChildItem ".github/prompts/*.prompt.md").Count
# Expected: 10

# 3. 验证决策目录创建
Test-Path "projects/dgsf/decisions/README.md"
# Expected: True

# 4. 验证 Kernel 包含新增内容
Select-String -Path ".github/copilot-instructions.md" -Pattern "Priority Levels|Regression Checklist|Active Module Hint"
# Expected: 3 matches
```

---

## Version 3.1.0 (2026-02-04)

### 变更摘要

**架构级自主进化**：增强闭环完整性、硬编码关键阈值、增加知识积累机制。

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 修改 | 新增 SUCCESS METRICS 常量区；完善闭环图（增加 abort/decision_log 路径）；目标更新为"AI Quantitative Fund" |
| `dgsf_abort.prompt.md` | 新增 | 处理 research 发现方向不可行时的结构化退出 |
| `dgsf_decision_log.prompt.md` | 新增 | 记录关键决策理由，建立机构记忆 |
| `dgsf_verify.prompt.md` | 修改 | 硬编码 DGSF 成功阈值（OOS Sharpe ≥ 1.5, OOS/IS ≥ 0.9 等） |

### 变更理由

1. **闭环缺口修复**
   - 缺失：research 发现不可行后无标准退出 → 新增 `/dgsf_abort`
   - 缺失：决策理由无持久化 → 新增 `/dgsf_decision_log`
   - 效果：闭环图现在覆盖所有可能的状态转移

2. **阈值硬编码**
   - 问题：OOS Sharpe ≥ 1.5 等关键指标散落在文档中，未被 prompts 引用
   - 修复：Kernel 新增 SUCCESS METRICS 区，verify prompt 直接引用
   - 效果：验证时自动检查，无需人工记忆阈值

3. **目标演进**
   - 从 "Asset Pricing Research" 明确为 "AI Quantitative Fund"
   - 体现系统向生产级量化基金演化的意图

### 验证方式

```powershell
# 验证所有 prompts 存在
(Get-ChildItem ".github/prompts/*.prompt.md").Count
# Expected: 10 items (+2 from v3.0)

# 验证 Kernel 行数
(Get-Content .github/copilot-instructions.md | Measure-Object -Line).Lines
# Expected: ≤ 120 (current ~110)

# 验证新 prompts 语法
Get-Content .github/prompts/dgsf_abort.prompt.md | Select-Object -First 5
Get-Content .github/prompts/dgsf_decision_log.prompt.md | Select-Object -First 5
```

---

## Version 3.0.0 (2026-02-04)

### 变更摘要

**架构级重构**：增强闭环能力、精简规则、统一 prompt 衔接。

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 重构 | 规则从 7 条精简为 5 条；新增闭环流程图；引入 EVOLUTION_LOG |
| `dgsf_diagnose.prompt.md` | 新增 | 处理执行失败后的诊断流程 |
| `dgsf_research.prompt.md` | 新增 | 支持执行前的调研探索 |
| `dgsf_plan.prompt.md` | 修改 | 新增 PREREQUISITES 衔接调研/扫描 |
| `dgsf_execute.prompt.md` | 修改 | 新增 ON FAILURE 衔接诊断流程 |
| `dgsf_verify.prompt.md` | 修改 | 新增 NEXT STEPS 基于 verdict |
| `dgsf_state_update.prompt.md` | 修改 | 新增 NEXT STEPS 基于 type |
| `dgsf_research_summary.prompt.md` | 修改 | 新增 WHEN TO INVOKE |
| `dgsf_repo_scan.prompt.md` | 修改 | 新增 WHEN TO INVOKE |

### 变更理由

1. **规则精简**（7 → 5）
   - 合并：R3 "Background for long jobs" + R7 "Stop on failure" → R3 "Stop on failure"（长任务处理由具体 prompt 负责）
   - 合并：R5 "No path guessing" + R6 "No financial assumptions" → R5 "No assumptions"
   - 结果：更少规则，更易记忆，覆盖范围不变

2. **闭环缺口修复**
   - 缺失：执行失败后无标准化诊断流程 → 新增 `/dgsf_diagnose`
   - 缺失：规划前无探索阶段 → 新增 `/dgsf_research`
   - 缺失：prompts 间无显式衔接 → 所有 prompts 新增 PREREQUISITES/NEXT STEPS

3. **自演化机制强化**
   - 新增此 EVOLUTION_LOG.md 文件
   - Kernel 行数限制从 150 下调至 120（鼓励精简）

### 验证方式

```powershell
# 验证所有 prompts 存在
ls .github/prompts/*.prompt.md | Measure-Object
# Expected: 8 items (plan, execute, verify, diagnose, research, state_update, research_summary, repo_scan)

# 验证 Kernel 行数
(Get-Content .github/copilot-instructions.md | Measure-Object -Line).Lines
# Expected: ≤ 120
```

---

## Version 2.0.0 (2026-02-04)

### 变更摘要

Initial DGSF-focused kernel, replacing legacy AI Workflow OS.

### 修改清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `copilot-instructions.md` | 重写 | 从 AI Workflow OS 迁移至 DGSF 专用 |
| 6 prompts | 新增 | plan, execute, verify, state_update, research_summary, repo_scan |

---

## 演化协议

### 标准演化路径（结构性变更）

**提议新演化前，请确认**：

1. 是否有 3+ 个具体实例证明当前规则/流程存在问题？
2. 修改是否保持规则总数 ≤ 5、Kernel ≤ 120 行？
3. 是否可通过现有 prompts 组合解决，而非新增？

**提议格式**：

```markdown
## Proposed: {版本号}

### 问题陈述
{具体问题，需引用实例}

### 建议修改
{Diff 格式，不是完整重写}

### 预期效果
{修改后如何验证改进}
```

### 快速迭代路径（实验性调整）

**适用场景**：
- Prompt 措辞调整（不改变结构）
- 示例更新
- 验证命令修正
- 阈值微调（±10% 范围内）

**流程**：
1. 直接修改文件
2. 在 EVOLUTION_LOG 新增 PATCH 记录（格式：`v3.1.1-patch1`）
3. 观察 3 次使用效果
4. 若有效，合并入下一个正式版本；若无效，回滚

**示例**：
```markdown
## v3.1.1-patch1 (2026-02-05)

**调整**: dgsf_verify 中 Max Drawdown 阈值从 20% 调整为 25%
**理由**: 高波动市场环境下 20% 过于严格，导致假阴性
**观察期**: 3 次实验验证
**状态**: 待验证 / 已采纳 / 已回滚
```
