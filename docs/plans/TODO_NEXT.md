# TODO_NEXT - DGSF驱动的执行队列

**Created**: 2026-02-02  
**Updated**: 2026-02-02T18:00Z (Project Orchestrator Refresh)  
**Purpose**: DGSF项目的canonical execution queue  
**Priority Order**: P0（直接推进DGSF）→ P1（解除阻塞）→ P2（延后）  
**Primary Objective**: 推进DGSF（Dynamic Generative SDF Forest）项目的开发、验证与研究产出

---

## 🎯 Global Priority Override Rule

**DGSF Priority Override**: 当DGSF项目推进与AI Workflow OS层面的改进发生冲突时，**无条件以DGSF的开发与验证为最高优先级（P0）**。

所有OS层面的工作必须满足以下至少一条，才允许进入执行队列：
- 直接解除DGSF的开发阻塞
- 显著降低DGSF的实验/回测/迭代成本
- 为DGSF的阶段性成果提供必要的可验证性与可追溯性

否则，一律降级为**Deferred / P2**。

---

## 📊 Current Context（基于证据 · 2026-02-02T18:00Z）

**DGSF项目状态**:
- Pipeline: Stage 4 "SDF Layer Development" - in_progress ✅
- 代码: repo/（活跃，submodule 同步）✅
- 下一步: SDF_DEV_001_T1（SDF Model Architecture Review, P0, 1周估算）
- 测试状态: 26 collection errors in tests/sdf/（待诊断）

**AI Workflow OS状态**:
- 分支: feature/router-v0（领先origin 22个提交）
- 测试: kernel/ 186个通过 ✅
- Working tree: clean ✅
- Legacy: 已隔离，pytest不再扫描 ✅

---

## 🔴 P0任务（直接推进DGSF）

### P0-1: 执行 SDF_DEV_001_T1 - SDF Model Architecture Review ✅ COMPLETED
**DGSF关联**: Stage 4首个子任务，识别所有SDF模型及技术债  
**Effort**: 20分钟  
**Dependencies**: 无  
**Status**: ✅ COMPLETED (2026-02-02T18:25)

**执行结果**:
- ✅ 生成 `projects/dgsf/reports/SDF_MODEL_INVENTORY.json`
- ✅ 识别 4 个模型: GenerativeSDF, DevSDFModel, LinearSDFModel, MLPSDFModel
- ✅ 识别 5 项技术债 (4 Medium + 1 Low)
- ✅ 分析依赖关系和架构模式
- ✅ 提供 immediate/short-term/long-term 推荐行动

**验收标准（DoD）**:
- ✅ JSON 包含所有 `.py` 文件中的模型类
- ✅ 每个模型记录：name, file_path, dependencies, status, notes
- ✅ 识别至少 3 个 technical debt 或 improvement areas（实际识别5个）
- ✅ 验证命令通过: `python -c "import json; data=json.load(open('projects/dgsf/reports/SDF_MODEL_INVENTORY.json')); assert len(data['models']) > 0"`

---

### P0-2: 明确 SDF_DEV_001_T2 的失败详情 ✅ COMPLETED
**DGSF关联**: 为修复测试准备（T2任务前置条件）  
**Effort**: 10分钟  
**Dependencies**: 无  
**Status**: ✅ COMPLETED (2026-02-02T18:45)

**执行结果**:
- ✅ 生成 `projects/dgsf/reports/SDF_TEST_FAILURES.txt`（156行原始输出）
- ✅ 生成 `projects/dgsf/reports/SDF_TEST_FAILURES.md`（分类汇总报告）
- ✅ 识别根本原因: **单一导入错误** `ModuleNotFoundError: No module named 'dgsf.sdf.state_engine'`
- ✅ 11/11 测试文件阻塞（100%）
- ✅ 提供 3 个修复方案（注释/占位符/移除）

**验收标准（DoD）**:
- ✅ 记录所有失败测试用例（11 collection errors）
- ✅ 分类失败原因（1 类: Missing Module）
- ✅ 提供修复建议（3 个 options）
- ✅ 验证命令通过: `Select-String -Path projects/dgsf/reports/SDF_TEST_FAILURES.md -Pattern "Category"`

---

### P0-3: 修复 SDF 导入错误（state_engine 缺失）✅ COMPLETED
**DGSF关联**: 解除 100% 测试阻塞，使测试可执行  
**Effort**: 5分钟  
**Dependencies**: P0-2 ✅ COMPLETED  
**Status**: ✅ COMPLETED (2026-02-02T18:50)

**执行结果**:
- ✅ 注释掉 `src/dgsf/sdf/__init__.py` 中的 `state_engine` 导入
- ✅ 更新 `__all__` 列表（移除 4 个 state_engine 导出）
- ✅ 添加 FIXME 注释（说明原因和后续 TODO）
- ✅ **167 tests collected in 1.55s**（修复前: 0 tests, 11 errors）

**验收标准（DoD）**:
- ✅ pytest 成功收集至少 1 个测试（实际: 167 tests）
- ✅ 无 ModuleNotFoundError 错误
- ✅ 验证命令通过: `python -m pytest tests/sdf/ --collect-only 2>&1 | Select-String "collected"`

---

## 🟡 P1任务（解除对DGSF的阻塞）

### P1-1: 创建 Adapter 层集成测试
**DGSF关联**: 验证 OS ↔ DGSF 接口可用性，防止首次实验时暴雷  
**Effort**: 30分钟  
**Dependencies**: 无  
**Status**: ⏸️ READY

**执行步骤**:
1. 创建 `projects/dgsf/adapter/tests/test_integration.py`
2. 实现测试用例: `test_adapter_run_experiment_e2e()`
3. 验证流程: `DGSFAdapter.run_experiment()` → 检查日志/状态同步
4. 运行测试: `pytest projects/dgsf/adapter/tests/test_integration.py -v`

**验收标准（DoD）**:
- 测试通过（exit code 0）
- 覆盖 `DGSFAdapter.run_experiment()` 主流程
- Mock 外部依赖（避免真实实验）
- 验证命令: `pytest projects/dgsf/adapter/tests/test_integration.py -v`

---

### P1-2: 推送 feature/router-v0 到 origin
**DGSF关联**: 确保工作可共享，降低协作风险  
**Effort**: 2分钟  
**Dependencies**: 无  
**Status**: ⏸️ READY

**执行步骤**:
1. `git push origin feature/router-v0`
2. 验证远程分支存在: `git ls-remote --heads origin feature/router-v0`

**验收标准（DoD）**:
- 远程分支与本地同步（22 commits 可见）
- 验证命令: `git rev-parse origin/feature/router-v0` 返回与 `HEAD` 相同的 commit hash

---

### P1-3: 提交 P0-1 执行结果
**DGSF关联**: 保存 SDF Model Inventory，防止工作丢失  
**Effort**: 3分钟  
**Dependencies**: P0-1 ✅ COMPLETED  
**Status**: ⏸️ READY

**执行步骤**:
1. `git add projects/dgsf/reports/SDF_MODEL_INVENTORY.json`
2. `git add docs/plans/TODO_NEXT.md docs/state/PROJECT_STATE.md`
3. `git commit -m "feat(dgsf): complete SDF Model Architecture Review (SDF_DEV_001_T1)"`

**验收标准（DoD）**:
- 工作区干净（no uncommitted changes）
- 提交包含 SDF_MODEL_INVENTORY.json 和状态更新
- 验证命令: `git log -1 --stat | Select-String "SDF_MODEL_INVENTORY"`

---

### P1-4: 验证 DGSF repo 测试环境
**DGSF关联**: 确保 pytest 可在 repo/ 中运行（P0-2 前置条件）  
**Effort**: 5分钟  
**Dependencies**: 无  
**Status**: ⏸️ READY

**执行步骤**:
1. `cd projects/dgsf/repo/`
2. `python -m pytest --version` （验证 pytest 可用）
3. `python -m pytest tests/ --collect-only` （验证测试收集）
4. 记录环境信息到 `../../reports/DGSF_TEST_ENV.txt`

**验收标准（DoD）**:
- pytest 版本 >= 7.0
- 可成功收集测试（即使有 errors）
- 验证命令: `Select-String -Path projects/dgsf/reports/DGSF_TEST_ENV.txt -Pattern "pytest"`

---

### P1-5: 创建 SDF 测试失败修复 TaskCard
**DGSF关联**: 为 SDF_DEV_001_T2 准备可执行任务  
**Effort**: 10分钟  
**Dependencies**: P0-2 ✅ COMPLETED  
**Status**: ⏸️ BLOCKED (需 P0-2)

**执行步骤**:
1. 基于 P0-2 的失败分类，创建 `tasks/active/SDF_TEST_FIX_001.md`
2. 使用 TaskCard 模板，定义修复目标、验收标准
3. 更新 `state/tasks.yaml` 注册任务
4. 链接到 PROJECT_DGSF.yaml 的 SDF_DEV_001_T2

**验收标准（DoD）**:
- TaskCard 包含失败分类和修复策略
- tasks.yaml 中 status="active"
- 验证命令: `Select-String -Path tasks/active/SDF_TEST_FIX_001.md -Pattern "task_id: SDF_TEST_FIX_001"`

---

## ⚪ P2任务（延后 · 非DGSF直接需求）

### P2-1: 修复 kernel 导入路径（相对 → 绝对）
**原因**: 虽然 EXECUTION_PLAN_V1.md 标记为 P0，但不直接阻塞 DGSF  
**触发条件**: DGSF 实验调用 kernel 模块时出现导入错误  
**Effort**: 1.5小时  
**建议方案**: 批量替换 `from audit import` → `from kernel.audit import`

**执行步骤**:
1. 使用 multi_replace_string_in_file 批量修改 kernel/ 导入
2. 运行 `pyright kernel/` 验证类型检查
3. 运行 `pytest kernel/tests/ -v` 验证测试通过
4. 提交: `git commit -m "fix(kernel): use absolute imports for CI compatibility"`

**验收标准（DoD）**:
- pyright 通过（0 errors）
- pytest 通过（186 tests）
- 验证命令: `pyright kernel/ --outputjson | python -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if data['summary']['errorCount']==0 else 1)"`

---

### P2-2: 精简 PROJECT_STATE.md ⚠️ DEFERRED
**原因**: 4000+ 行难以检索，但不阻塞 DGSF  
**触发条件**: 用户明确要求或日志查询失败超过 3 次  
**建议方案**: 归档历史记录到 `docs/state/archive/PROJECT_STATE_2026Q1.md`

---

### P2-3: 形式化验证 Adapter 层因果性 ⚠️ DEFERRED
**原因**: 无证据表明当前有数据泄漏问题  
**触发条件**: 出现回测异常（未来收益率泄漏到训练集）  
**建议方案**: 使用形式化方法（如 TLA+）验证时间依赖

---

### P2-4: 重构 Adapter 层为通用接口 ⚠️ DEFERRED
**原因**: 仅 1 个项目使用，过早抽象（违反 YAGNI 原则）  
**触发条件**: 第 2 个 L2 项目出现且需要类似接口  
**建议方案**: 提取通用基类 `BaseProjectAdapter`

---

### P2-5: 实现 State Machine 验证器 ⚠️ DEFERRED
**原因**: EXECUTION_PLAN_V1.md P1-1，但不直接阻塞 DGSF  
**触发条件**: 任务状态转换违规（如 draft → completed 跳过 in_progress）  
**Effort**: 2小时  
**建议方案**: 创建 `scripts/verify_state_machine.py`，加载 `kernel/state_machine.yaml`，验证 tasks.yaml 的转换历史

---

## 📋 执行队列汇总（接下来 10 个步骤）

**更新时间**: 2026-02-02T18:45Z  
**当前进度**: 2/12 完成（P0-1 ✅, P0-2 ✅）

| # | Task ID | Priority | Status | Effort | Dependencies |
|---|---------|----------|--------|--------|--------------|
| 1 | P0-1 | P0 | ✅ COMPLETED | 20 min | 无 |
| 2 | P0-2 | P0 | ✅ COMPLETED | 10 min | 无 |
| 3 | P0-3 | P0 | ⏸️ READY | 5 min | P0-2 ✅ |
| 4 | P1-3 | P1 | ⏸️ READY | 3 min | P0-1 ✅ |
| 5 | P1-4 | P1 | ⏸️ READY | 5 min | 无 |
| 6 | P1-1 | P1 | ⏸️ READY | 30 min | 无 |
| 7 | P1-2 | P1 | ⏸️ READY | 2 min | 无 |
| 8 | P1-5 | P1 | ⏸️ BLOCKED | 10 min | P0-2 ✅→P0-3 |
| 9 | P2-1 | P2 | ⚠️ DEFERRED | 1.5 hr | 无（需触发） |
| 10 | P2-2 | P2 | ⚠️ DEFERRED | - | 无（需触发） |
| 11 | P2-3 | P2 | ⚠️ DEFERRED | - | 无（需触发） |
| 12 | P2-4 | P2 | ⚠️ DEFERRED | - | 无（需触发） |

**Next Step**: **P0-3** - 修复 SDF 导入错误（state_engine 缺失）

---

## 📝 Expert Panel Insights（专家观点 · 仅供参考）

### Grady Booch（Architecture）
- **核心风险**: SDF 架构审查缺乏具体执行路径
- **建议**: 先执行 P0-1 生成模型清单，再决定重构策略
- **Stop Doing**: 停止为了"优雅"而优化 Adapter 层

### Gene Kim（Execution Flow）
- **核心风险**: 22 个未推送的 commits 增加协作风险
- **建议**: 立即执行 P1-2 推送到 origin
- **Stop Doing**: 停止为每个执行步骤生成长篇文档（PROJECT_STATE 已 4000+ 行）

### Leslie Lamport（Definition of Done）
- **核心风险**: SDF 子任务缺乏量化验收标准
- **建议**: 为 P0-1 定义 JSON 格式的 artifact（已在 DoD 中明确）
- **Stop Doing**: 停止创建"评估报告"作为交付物（研究人员需要代码和数据）

---

## 🚀 Next Single Step（只能一个）

**选择**: **P0-2 - 明确 SDF_DEV_001_T2 的失败详情**

**理由**:
1. ✅ 零依赖（无需等待其他任务）
2. ✅ 直接推进 DGSF Stage 4（T2 任务的前置条件）
3. ✅ 产出明确（分类的测试失败报告）
4. ✅ 验证简单（检查报告包含失败分类）

**执行计划**:
```powershell
# 1. 切换到 DGSF repo
cd projects/dgsf/repo/

# 2. 运行 SDF 测试
pytest tests/sdf/ -v --tb=short > ../../reports/SDF_TEST_FAILURES.txt 2>&1

# 3. 分析失败原因（import, schema, dtype, assertion）
# 4. 生成分类汇总报告
# 5. 为每类失败提供修复建议
```

**验收标准**:
- 记录所有失败测试用例（test name, error message, file location）
- 分类失败原因（至少3类）
- 提供修复建议（每类至少1条）

---

**End of TODO_NEXT.md**

**原因**: EXECUTION_PLAN_V1.md P1-1，但不直接阻塞 DGSF  
**触发条件**: 任务状态转换违规（如 draft → completed 跳过 in_progress）  
**Effort**: 2小时  
**建议方案**: 创建 `scripts/verify_state_machine.py`，加载 `kernel/state_machine.yaml`，验证 tasks.yaml 的转换历史

---

## ✅ 已完成任务（归档）

### P0-1-OLD: 配置pytest排除Legacy DGSF ✅ COMPLETED
**完成时间**: 2026-02-02T15:00  
**Result**: pytest.ini created with testpaths=["kernel/tests"], 0 legacy errors verified

---

### P0-2-OLD: Define DGSF Stage 4 SDF tasks ✅ COMPLETED
**完成时间**: 2026-02-02T17:10  
**Result**: Added 5 SDF development tasks to PROJECT_DGSF.yaml (SDF_DEV_001_T1 到 T5)

---

### P0-3-OLD: 验证DGSF repo submodule状态 ✅ COMPLETED
**完成时间**: 2026-02-02T15:15  
**Result**: submodule synced with origin/master (commit fb208e4), clean working tree

---

### P1-4-OLD: 更新 Stage 4 状态为 in_progress ✅ COMPLETED
**完成时间**: 2026-02-02T17:10  
**Result**: PROJECT_DGSF.yaml - Stage 4 status="in_progress", started_date="2026-02-02"

---

## 📝 Expert Panel Insights（专家观点 · 仅供参考）

### Grady Booch（Architecture）
- **核心风险**: SDF 架构审查缺乏具体执行路径
- **建议**: 先执行 P0-1 生成模型清单，再决定重构策略
- **Stop Doing**: 停止为了"优雅"而优化 Adapter 层

### Gene Kim（Execution Flow）
- **核心风险**: 22 个未推送的 commits 增加协作风险
- **建议**: 立即执行 P1-2 推送到 origin
- **Stop Doing**: 停止为每个执行步骤生成长篇文档（PROJECT_STATE 已 4000+ 行）

### Leslie Lamport（Definition of Done）
- **核心风险**: SDF 子任务缺乏量化验收标准
- **建议**: 为 P0-1 定义 JSON 格式的 artifact（已在 DoD 中明确）
- **Stop Doing**: 停止创建"评估报告"作为交付物（研究人员需要代码和数据）

---

## 🚀 Next Single Step（只能一个）

**选择**: **P0-1 - 执行 SDF_DEV_001_T1 (SDF Model Architecture Review)**

**理由**:
1. ✅ 零依赖（无需等待其他任务）
2. ✅ 直接推进 DGSF Stage 4
3. ✅ 产出明确（JSON 格式的模型清单）
4. ✅ 验证简单（断言 JSON 包含模型）

**执行计划**:
```powershell
# 1. 扫描 SDF 目录
cd projects/dgsf/repo/src/dgsf/sdf/
Get-ChildItem -Recurse -Filter "*.py" | Select-Object Name, FullName

# 2. 识别模型类（手动或脚本辅助）
# 3. 生成 JSON 清单
# 4. 验证 JSON 格式
python -c "import json; data=json.load(open('projects/dgsf/reports/SDF_MODEL_INVENTORY.json')); print(f'Found {len(data[\"models\"])} models')"
```

**验收标准**:
- JSON 包含至少 1 个模型
- 每个模型有 name, file_path, dependencies, status 字段
- 识别至少 3 个 technical debt

---

**End of TODO_NEXT.md**
