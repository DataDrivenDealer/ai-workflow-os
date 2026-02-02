# DGSF Execution Plan V1

**Created**: 2026-02-02  
**Authority**: Derived from "scan → diagnose → plan → execute" cycle  
**Status**: Active  
**Primary Objective**: 持续推进DGSF（Dynamic Generative SDF Forest）项目的开发、验证与研究产出

---

## 🎯 Objectives & Non-goals

### Objectives（目标）
1. **解除DGSF的开发阻塞** - 清除165个pytest错误噪声
2. **明确DGSF的下一步研究任务** - Stage 4不应标记为"completed"
3. **建立DGSF开发工作流文档** - 开发者明确应在repo/工作
4. **保持AI Workflow OS作为支撑基础设施** - 提供治理和审计，不干扰研究

### Non-goals（不做什么）
- ❌ **不优化AI Workflow OS的架构**（除非直接阻塞DGSF）
- ❌ **不清理Legacy资产**（标记为archive即可，清理是P2）
- ❌ **不重构kernel/模块**（已通过186个测试，无需改动）
- ❌ **不推送OS到远程**（DGSF工作优先，OS同步可稍后）

---

## 📊 Current State Summary（聚焦DGSF）

### DGSF项目状态
- **规范**: PROJECT_DGSF.yaml v2.1.0（2026-02-01更新）
- **Pipeline**: Stage 4 "Research Continuation" - 标记为completed（❌ 误导）
- **代码**: 
  - 活跃开发: `projects/dgsf/repo/`（git submodule）
  - 过期资产: `projects/dgsf/legacy/DGSF/`（引发165个测试错误）
- **Adapter**: `projects/dgsf/adapter/`（DGSF ↔ OS桥接）✅

### 关键阻塞点
1. **pytest噪声**: 165个错误来自legacy/（掩盖真实问题）
2. **任务缺失**: Stage 4无具体的下一步研究任务
3. **文档不足**: 开发者不知道应在哪个目录工作

### AI Workflow OS状态
- **分支**: feature/router-v0（领先origin 19个提交）
- **测试**: kernel/ 186个测试通过 ✅
- **未提交**: 2个文件（state logs）

---

## 🛣️ Workstreams（≤3，至少1条DGSF本体）

### Workstream 1: DGSF Environment Preparation（P0）
**Owner**: Copilot Agent  
**Objective**: 清除开发环境障碍，使DGSF研究者能专注repo/

**Tasks**:
1. P0-1: 配置pytest排除Legacy DGSF
2. P0-3: 验证DGSF repo submodule状态
3. P1-3: 提交pending changes

**Completion Criteria**:
- ✅ `pytest --collect-only`显示0个legacy错误
- ✅ DGSF repo submodule与远程同步
- ✅ `git status`显示工作区干净

---

### Workstream 2: DGSF Research Task Definition（P0）
**Owner**: Project Owner（需确认）  
**Objective**: 明确Stage 4的下一步研究任务

**Tasks**:
1. P0-2: 定义DGSF Stage 4的具体任务
2. P1-4: 重构Stage 4状态为in_progress

**Completion Criteria**:
- ✅ PROJECT_DGSF.yaml包含至少3个active research tasks
- ✅ 每个任务有明确的deliverable和验收标准

**Blockers**:
- 需要Project Owner输入：优先级是baseline复现？新实验？还是论文撰写？

---

### Workstream 3: DGSF Developer Experience（P1）
**Owner**: Copilot Agent  
**Objective**: 建立清晰的DGSF开发文档

**Tasks**:
1. P1-1: 标记Legacy DGSF为archive-only
2. P1-2: 文档化DGSF开发工作流

**Completion Criteria**:
- ✅ `projects/dgsf/legacy/README.md`包含"DO NOT MODIFY"警告
- ✅ `projects/dgsf/README.md`包含"How to Develop in repo/"指南

---

## 🗓️ Milestones / Sprint Sequence

### Sprint 0: Environment Cleanup（今天完成）
- **Duration**: 1小时
- **Goal**: 清除pytest噪声 + 验证submodule
- **Deliverables**:
  - pytest.ini或pyproject.toml更新
  - DGSF repo submodule状态报告
  - 提交pending changes

### Sprint 1: Task Definition（等待Project Owner）
- **Duration**: TBD（需要Project Owner输入）
- **Goal**: 明确Stage 4的研究任务
- **Deliverables**:
  - PROJECT_DGSF.yaml更新（包含3+个active tasks）
  - 每个任务的TaskCard或详细描述

### Sprint 2: Documentation（今天完成）
- **Duration**: 30分钟
- **Goal**: 文档化DGSF开发工作流
- **Deliverables**:
  - projects/dgsf/legacy/README.md（archive警告）
  - projects/dgsf/README.md（开发指南）

---

## ✅ Definition of Done（以DGSF可验证产出为核心）

### Sprint 0（Environment Cleanup）
- [ ] pytest收集测试时不显示legacy/错误（验证：`pytest --collect-only | Select-String "ERROR"`为空）
- [ ] DGSF repo submodule无未提交变更（验证：`cd projects/dgsf/repo && git status`）
- [ ] AI Workflow OS工作区干净（验证：`git status`）

### Sprint 1（Task Definition）- 需Project Owner验收
- [ ] PROJECT_DGSF.yaml包含至少3个active tasks
- [ ] 每个任务有deliverable、effort、verification
- [ ] 至少1个任务可立即开始（无依赖）

### Sprint 2（Documentation）
- [ ] Legacy README包含"ARCHIVED - DO NOT MODIFY"标题
- [ ] Main README包含"Development Workflow"章节（≥100字）
- [ ] 文档链接到DGSF repo的实际开发指南（如果存在）

---

## 🔁 Verification Loop

每完成一个Sprint：
1. **验证DoD** - 运行验证命令，确保所有条件满足
2. **更新PROJECT_STATE.md** - 记录完成的任务、验证证据、下一步
3. **检查DGSF关联** - 这个Sprint是否真正推进了DGSF？还是仅仅优化了OS？

---

## 🛑 Stop Doing List（明确当前不该做的OS工作）

以下任务**暂停**，直到DGSF有明确的阻塞需求：

1. ❌ **kernel/模块的导入路径重构** - 当前测试已通过，无需改动
2. ❌ **架构边界验证脚本优化** - 已有脚本运行正常
3. ❌ **docs/重构与合并** - 文档数量多但不影响DGSF工作
4. ❌ **CI管道修复** - 可在DGSF有产出后再推送
5. ❌ **清理projects/dgsf/legacy/目录** - 标记为archive即可，删除是P2
6. ❌ **state/sessions.yaml的过期记录清理** - 不影响DGSF
7. ❌ **WIP Limit enforcement增强** - 当前规则已足够
8. ❌ **度量体系建立** - 等DGSF有稳定产出后再建立

**原则：除非某个OS任务直接解除DGSF的阻塞，否则一律降级为P2**

---

## 📝 Dependencies & Risks

### Dependencies
- **Sprint 1阻塞于Project Owner输入** - 需要明确研究优先级
- **DGSF repo外部依赖** - 如果repo/有自己的依赖，需要在repo/内安装

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Project Owner未及时响应 | Sprint 1延迟 | 先完成Sprint 0+2，提供默认任务建议 |
| DGSF repo submodule过期 | 实验不可复现 | Sprint 0立即验证并更新 |
| pytest配置失效 | 165错误仍显示 | 使用`--ignore`参数，并验证 |

---

## 📂 Artifacts

所有产出物路径：

| Artifact | Path | Status |
|----------|------|--------|
| Execution Plan | docs/plans/EXECUTION_PLAN_DGSF_V1.md | ✅ Created |
| TODO List | docs/plans/TODO_NEXT.md | 🔄 To be updated |
| State Log | docs/state/PROJECT_STATE.md | 🔄 To be updated |
| pytest Config | pytest.ini or pyproject.toml | 🔄 To be updated |
| Legacy README | projects/dgsf/legacy/README.md | 🔄 To be created |
| Main DGSF README | projects/dgsf/README.md | 🔄 To be updated |
| PROJECT_DGSF.yaml | projects/dgsf/specs/PROJECT_DGSF.yaml | 🔄 To be updated (Sprint 1) |

---

## 🔄 Next Review

- **When**: Sprint 0完成后（预计1小时内）
- **What**: 验证pytest噪声是否清除，submodule是否同步
- **Who**: Copilot Agent自我验证，然后提交给Project Owner review

---

**END OF EXECUTION PLAN V1**
