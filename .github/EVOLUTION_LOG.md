# Copilot Runtime OS — Evolution Log

> 记录所有对 Kernel 及 Prompts 的修改，确保变更可追溯。

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
