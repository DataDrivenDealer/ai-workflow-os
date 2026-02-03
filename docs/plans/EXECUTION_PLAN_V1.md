# AI Workflow OS - 执行计划 V1.3（DGSF 驱动）

**文档ID**: EXECUTION_PLAN_V1  
**创建日期**: 2026-02-02  
**最后更新**: 2026-02-03T23:55:00Z（T3.3.3 完成，准备 T3.3.4）  
**状态**: ACTIVE  
**主要目标**: **推进 DGSF（Dynamic Generative SDF Forest）项目的开发、验证与研究产出**  
**目标周期**: 2026-02-03 至 2026-02-28 (4周，含 T3 Feature Engineering)  
**WIP限制**: 最多3个并行工作流，**至少1个必须是 DGSF 本体相关**

**关联文档**: 
- [TODO_NEXT](TODO_NEXT.md) - DGSF 驱动的执行队列
- [PROJECT_DGSF.yaml](../../projects/dgsf/specs/PROJECT_DGSF.yaml) - DGSF 项目规范
- [STAGE_4_ACCEPTANCE_CRITERIA.md](../../projects/dgsf/docs/STAGE_4_ACCEPTANCE_CRITERIA.md) - Stage 4 验收标准

---

## 🔴 Global Priority Override Rule（全局优先级裁决规则）

> **DGSF Priority Override：**
> 当 **DGSF 项目推进** 与 **AI Workflow OS 层面的改进** 发生冲突时：
>
> **无条件以 DGSF 的开发与验证为最高优先级（P0）。**
>
> OS 层面的工作必须满足以下至少一条：
> - 直接解除 DGSF 的开发阻塞
> - 显著降低 DGSF 的实验/回测/迭代成本
> - 为 DGSF 的阶段性成果提供必要的可验证性与可追溯性

---

## ⚠️ 重大更新说明（2026-02-03T23:55Z）

**DGSF Stage 4 T3 Feature Engineering 进展**:
1. ✅ **T3.3.1 完成** - Pipeline 基础框架 + CLI 接口 (485 lines)
2. ✅ **T3.3.2 完成** - 数据加载模块 (569 lines, 21 tests)
3. ✅ **T3.3.3 完成** - Firm Characteristics 计算 (516 lines, 19 tests)
4. 🎯 **T3.3.4 就绪** - Cross-Sectional Spreads + Factors（下一步）

**当前测试状态**: **40/40 passed** (scripts + adapter tests)

**当前焦点**: 
1. **Git Checkpoint** - 提交 T3.3.3 成果防止丢失
2. **T3.3.4** - 实现 Cross-Sectional Spreads 和 5 Factors

---

## 0. Objectives & Non-goals（目标与非目标）

### ✅ Objectives（目标 · DGSF 驱动）
1. **推进 DGSF Stage 4** - 完成 T3 Feature Engineering、T4 Training Optimization
2. **支撑 DGSF 验证** - 确保测试通过率 ≥95%，实验可复现
3. **降低迭代摩擦** - 提供快速验证脚本、Daily Workflow 文档化
4. **保持 OS 稳定** - 不因 OS 重构阻塞 DGSF 开发

### ❌ Non-goals（非目标 · Stop Doing List）
- ❌ 不优化 Adapter 接口（run_experiment 未实现且不阻塞）
- ❌ 不追求 OS 100% 测试覆盖率（聚焦 DGSF）
- ❌ 不为每个微小进展创建独立 audit JSON
- ❌ 不重构 kernel 导入路径（除非阻塞 DGSF）
- ❌ 不精简 PROJECT_STATE.md（除非查询失败）

---

## 1. Current State Summary（当前状态摘要）

**证据来源**: Git analysis @ 2026-02-03T21:00:00Z

### 1.1 DGSF 项目状态（主要指标）
| 维度 | 状态 | 证据 | 变化 |
|-----|------|------|------|
| **Stage 4 进度** | T3.3.4 就绪 | T3.3.1-3.3.3 ✅, T3.3.4 READY | ↑ |
| **测试通过率** | 100% (40/40 scripts) | `pytest tests/ -v` | ↑+100% |
| **Firm Characteristics** | ✅ 5/5 实现 | [firm_characteristics.py](../../projects/dgsf/scripts/firm_characteristics.py) | ✅ |
| **技术债** | 3 TODOs in run_feature_engineering.py | T3.3.4 将解决 | 已识别 |
| **未提交文件** | 14 untracked | `git status` | ⚠️ 需 checkpoint |

**DGSF 评分**: T3.3.4 **READY** ✅（可进入 Cross-Sectional Spreads 阶段）

### 1.2 AI Workflow OS 状态（支撑系统）
| 维度 | 状态 | 变化 |
|-----|------|------|
| **分支** | feature/router-v0 | ↔️ |
| **kernel 测试** | 186 passed | ✅ |
| **Working tree** | 6 modified, 14 untracked | 待提交 |

### 1.3 关键风险（DGSF 影响排序）
- 🟡 **P0 Risk**: 14 untracked files 未提交 Git（数据丢失风险）
- 🟢 **无阻塞性风险** - T3.3.4 所有依赖已就绪
- ⚪ **P2 Risk**: state_engine 模块缺失（不阻塞 T3）

### 1.4 未提交变更（待本轮提交）
- [firm_characteristics.py](../../projects/dgsf/scripts/firm_characteristics.py) - T3.3.3 核心模块
- [test_firm_characteristics.py](../../projects/dgsf/tests/test_firm_characteristics.py) - 19 单元测试
- [data_loaders.py](../../projects/dgsf/scripts/data_loaders.py) - T3.3.2 数据加载
- [test_data_loading.py](../../projects/dgsf/tests/test_data_loading.py) - 21 单元测试
- [SDF_FEATURE_DEFINITIONS.md](../../projects/dgsf/docs/SDF_FEATURE_DEFINITIONS.md) - 10 特征定义

---

## 2. Workstreams（工作流 - 最多3个并行，至少1个 DGSF）

### 🔴 Workstream 1: DGSF Stage 4 开发（P0 · 主线）
**Owner**: DGSF Researcher  
**Duration**: Week 1-4  
**Goal**: 完成 T3 Feature Engineering + T4 Training Optimization

**Milestones**:
- **M1.1** (Week 1 Day 1-2): T3 任务拆解，创建 TaskCard → **P0-7** 🎯
- **M1.2** (Week 1 Day 3-5): 特征定义文档化，baseline 特征集确定
- **M1.3** (Week 2): Feature construction pipeline 实现
- **M1.4** (Week 3): Ablation study 实验设计与执行
- **M1.5** (Week 4): T3 验收，启动 T4

### 🟡 Workstream 2: DGSF 开发支撑（P1 · 降低摩擦）
**Owner**: Platform Engineer  
**Duration**: Week 1-2  
**Goal**: 提供快速验证工具，降低 DGSF 迭代成本

**Milestones**:
- **M2.1** (Week 1 Day 1): 创建快速验证脚本 → P1-1
- **M2.2** (Week 1 Day 2): 定义 T3 → T4 Gate → P1-2
- **M2.3** (Week 1 Day 3): Daily Workflow Checklist → P1-3
- **M2.4** (Week 2): 恢复 7 个 data-dependent skipped tests（可选）

### ⚪ Workstream 3: OS 维护（P2 · 延后）
**Owner**: Platform Engineer  
**Duration**: 仅在 DGSF 不阻塞时执行  
**Goal**: 维持 OS 稳定性，不主动优化

**Deferred Tasks（触发条件激活）**:
- P2-1: T4/T5 TaskCard（T3 完成度 >80% 时）
- P2-2: RESEARCH_MILESTONES.md（有论文 deadline 时）
- P2-3: 聚合 audit JSON（audit/ 目录 >50 文件时）
- P2-4: Troubleshooting 章节（同一问题出现 ≥2 次时）
- P2-5: kernel 导入路径修复（DGSF 调用 kernel 出错时）
- P2-6: PROJECT_STATE.md 精简（查询失败 ≥3 次时）

---

## 3. Week-by-Week Sequence（周序列 · DGSF 聚焦）

### Week 1: T3 启动（LAUNCH T3）
**Theme**: 完成 T3 任务拆解，建立验证基础设施

| Day | Task | Priority | Output | Verification |
|-----|------|----------|--------|-------------|
| Mon | **P0-7**: T3 任务拆解 | P0 | `tasks/active/SDF_FEATURE_ENG_001.md` | TaskCard 包含 ≥5 子任务 |
| Mon | P1-1: 快速验证脚本 | P1 | `scripts/dgsf_quick_check.ps1` | 运行 <10s，输出 4 状态项 |
| Tue | P1-2: 定义 T3→T4 Gate | P1 | 更新 `STAGE_4_ACCEPTANCE_CRITERIA.md` | Gate 包含数值阈值 |
| Tue | P1-3: Daily Workflow | P1 | 更新 `projects/dgsf/README.md` | Checklist 5-7 项 |
| Wed-Fri | T3 开发: 特征定义 | P0 | Feature definitions doc | 与 SDF_SPEC v3.1 对齐 |

### Week 2-3: T3 实现（IMPLEMENT T3）
**Theme**: Feature Engineering Pipeline 开发

| Period | Task | Output | Verification |
|--------|------|--------|-------------|
| Week 2 | Feature construction | `scripts/run_feature_engineering.py` | Script 可执行 |
| Week 3 | Ablation study | `experiments/feature_ablation/results.json` | ≥3 features p<0.05 |

### Week 4: T3 验收 → T4 启动（GATE T3→T4）
**Theme**: 验收 T3，规划 T4

| Task | Output | Verification |
|------|--------|-------------|
| T3 验收 | AC-3 ACHIEVED | 满足 T3→T4 Gate 条件 |
| T4 规划 | `tasks/active/SDF_TRAINING_OPT_001.md` | TaskCard 创建 |

---

## 4. Definition of Done（以 DGSF 可验证产出为核心）

### Stage 4 完成标准
| AC | 描述 | 验证命令 | 状态 |
|----|------|----------|------|
| AC-1 | Test pass rate ≥95% | `pytest tests/sdf/ -v` | 🟡 93.4% |
| AC-2 | Model Inventory 完成 | `Test-Path reports/SDF_MODEL_INVENTORY.json` | ✅ |
| AC-3 | Feature Engineering | `Test-Path scripts/run_feature_engineering.py` | ⏸️ |
| AC-4 | Training Optimization | Sharpe ≥1.5 OOS | ⏸️ |
| AC-5 | Evaluation Framework | All metrics in SDF_SPEC v3.1 | ⏸️ |

### Verification Loop
每次执行后验证：
1. `pytest tests/sdf/ -v` - 测试通过率未下降
2. `cd repo; git status` - 无未提交的阻塞性变更
3. `Test-Path` 相关产出文件 - 产出存在

---

## 5. Stop Doing List（当前不该做的 OS 工作）

| 任务 | 原因 | 触发条件 |
|------|------|----------|
| 优化 Adapter 接口 | `run_experiment` 未实现且不阻塞 | 永不 |
| 重构 kernel 导入 | 不阻塞 DGSF | DGSF 调用出错时 |
| 精简 PROJECT_STATE | 不阻塞 DGSF | 查询失败 ≥3 次 |
| 创建独立 audit JSON | 仓库污染 | 仅重大决策 |
| OS 100% 测试覆盖 | 非必要 | 永不 |
| OS Dashboard 建设 | 非 DGSF 需求 | 永不（本周期内）|

### Week 3: 自动化完成+可观测性启动（OBSERVE）
**Theme**: 完成Gate自动化，建立度量基础

| Day | Task | Owner | Output | Verification |
|-----|------|-------|--------|-------------|
| Mon | P1-5: Gate G5脚本 | DevOps | [scripts/run_gate_g5.py](../../scripts/run_gate_g5.py) | G5检查可执行 |
| Tue | P1-5: Gate G6脚本 | DevOps | [scripts/run_gate_g6.py](../../scripts/run_gate_g6.py) | G6检查可执行 |
| Wed | P2-3: 看板可视化 | Data | [scripts/generate_kanban.py](../../scripts/generate_kanban.py) | 输出Markdown看板 |
| Thu | P2-1: YAML工具模块 | DevOps | [kernel/yaml_utils.py](../../kernel/yaml_utils.py) | 重构后测试通过 |
| Fri | P2-5: 架构测试 | Platform | [kernel/tests/test_architecture.py](../../kernel/tests/test_architecture.py) | 验证依赖方向 |

### Week 4: 度量体系建设（MEASURE）
**Theme**: Dashboard和持续改进机制

| Day | Task | Owner | Output | Verification |
|-----|------|-------|--------|-------------|
| Mon | P2-2: Metrics收集脚本(1/2) | Data | 基础度量收集 | Cycle Time计算正确 |
| Tue | P2-2: Metrics收集脚本(2/2) | Data | [scripts/collect_metrics.py](../../scripts/collect_metrics.py) | 完整度量报告 |
| Wed | P2-4: 度量Dashboard(1/2) | Data | Dashboard框架 | HTML生成成功 |
| Thu | P2-4: 度量Dashboard(2/2) | Data | [scripts/generate_metrics_dashboard.py](../../scripts/generate_metrics_dashboard.py) | Dashboard可视化 |
| Fri | P2-6: Tech Debt Registry | Platform | [docs/TECH_DEBT_REGISTRY.md](../../docs/TECH_DEBT_REGISTRY.md) | 所有TODO分类 |

### Week 2: 自动化基础（AUTOMATE）
**Theme**: CI/CD流水线

| Day | Task | Owner | Output | Verification |
|-----|------|-------|--------|-------------|
| Mon | B-7: 配置管理统一 | Platform | `kernel/config.py` | Config加载测试 |
| Tue | B-8: GitHub Actions配置 | DevOps | `.github/workflows/ci.yml` | CI绿灯 |
| Wed | B-6: 状态验证脚本 | Platform | `scripts/verify_state.py` | 检测非法转换 |
| Thu | B-9: WIP限制实现 | Platform | state_store.py更新 | ≤3任务running |
| Fri | 集成测试周 | Team | 完整流程验证 | End-to-end通过 |

### Week 3: 质量提升（IMPROVE）
**Theme**: 测试和度量

| Day | Task | Owner | Output | Verification |
|-----|------|-------|--------|-------------|
| Mon | B-13: DGSF测试套件 | QA | `projects/dgsf/repo/tests/` | Pytest独立运行 |
| Tue | B-11: Coverage报告 | DevOps | CI coverage report | >80% coverage |
| Wed | B-10: Metrics dashboard(1/2) | Data | `reports/metrics_dashboard.md` | Cycle time可见 |
| Thu | B-10: Metrics dashboard(2/2) | Data | 图表生成 | Throughput可见 |
| Fri | B-12: 不变量定义 | Platform | `kernel/invariants.py` | 10+不变量文档 |

### Week 4: 长期优化（OPTIMIZE）
**Theme**: 架构演进

| Day | Task | Owner | Output | Verification |
|-----|------|-------|--------|-------------|
| Mon | B-14: State接口抽象 | Architect | 接口定义 | YAML/SQLite可切换 |
| Tue | B-5: SQLite迁移脚本 | Platform | 迁移工具 | 测试数据迁移成功 |
| Wed | B-15: Blueprint检查器 | Platform | 文档验证工具 | 链接有效性100% |
| Thu | 回归测试周 | QA | 完整测试套件 | All tests green |
| Fri | 发布准备 | Team | Release notes | v0.2.0 ready |

---

## 4. Definition of Done（完成标准）

### 全局DoD（每个任务必须满足）
- [ ] 代码已提交到feature分支
- [ ] 单元测试覆盖新代码（>80%）
- [ ] 所有CI checks通过（pytest + gate_check）
- [ ] 文档更新（README/API文档/Architecture蓝图）
- [ ] Code review完成（至少1位reviewer）
- [ ] 无blocking comments

### 里程碑DoD（每周结束时）
- [ ] 所有planned任务完成或defer决策明确
- [ ] 集成测试通过
- [ ] Demo可运行展示进展
- [ ] Retrospective记录经验教训

### 发布DoD（Week 4结束时）
- [ ] 所有P0和P1任务完成
- [ ] 回归测试套件100%通过
- [ ] 性能基准测试无退化（cycle time ±10%以内）
- [ ] 安全审查完成（无critical漏洞）
- [ ] Release notes发布
- [ ] Deployment runbook更新

---

## 5. Verification Loop（验证循环）

### 每日验证（Automated）
```powershell
# 在pre-commit hook中自动运行
python -m pytest kernel/tests/ -v
python scripts/verify_state.py
python scripts/gate_check.py
```

### 每周验证（Manual + Automated）
```powershell
# Week-end健康检查
python -m pytest --cov=kernel --cov-report=html
python scripts/gate_report.py --since=7days
python scripts/check_blueprint_consistency.py
```

### 发布前验证（Comprehensive）
```powershell
# 完整回归测试
python -m pytest kernel/tests/ projects/dgsf/repo/tests/ -v
python scripts/verify_state.py --strict
python scripts/simulate_agent_workflow.py
git log --since="4.weeks.ago" --pretty=format:"%h %s" > release_notes.txt
```

---

## 6. "Stop Doing" List（反忙碌清单）

### ❌ 停止做（浪费时间的事）
1. **手动复制Git hooks** - 已有install_hooks.ps1，强制使用
2. **在Slack讨论架构决策** - 必须记录到`ops/decision-log/`
3. **直接修改state/ YAML** - 必须通过kernel/os.py CLI操作
4. **没有TaskCard就开始编码** - 强制执行task new → task start流程
5. **跳过Gate检查直接merge** - pre-push hook强制执行
6. **追求完美的架构** - 采用Strangler Fig，允许临时方案
7. **同时进行>3个feature分支** - WIP限制=3
8. **写代码不写测试** - Coverage gate强制>80%

### ✅ 继续做（高价值的事）
1. Blueprint-first设计（先更新架构图再写代码）
2. Event sourcing审计追踪（所有操作记录到events）
3. Template-driven development（TaskCard模板标准化）
4. Small batch commits（每个commit ≤200 lines）
5. Pair programming for critical changes（P0/P1任务）

---

## 7. Risk Mitigation（风险缓解）

| Risk | Probability | Impact | Mitigation | Contingency |
|------|------------|--------|-----------|------------|
| State corruption | High | Critical | B-1并发锁实现 | 定期备份state/ |
| Dependency break | Med | High | B-2版本锁定 | Docker镜像freeze |
| WIP overload | High | Med | B-9强制WIP≤3 | 每周prioritize |
| Test coverage drop | Med | Med | B-11 CI coverage | 每周review report |
| Blueprint drift | Low | Med | B-15自动检查 | 每月manual audit |

---

## 8. Metrics & KPIs（度量指标）

### 过程度量（每周追踪）
- **Cycle Time**: Task从running→merged的天数（目标: <3天）
- **Throughput**: 每周完成的任务数（目标: ≥5个）
- **WIP**: 同时进行的任务数（目标: ≤3个）
- **Gate Pass Rate**: Gate检查通过率（目标: >90%）

### 质量度量（每次CI运行）
- **Test Coverage**: 代码覆盖率（目标: >80%）
- **Failed Tests**: 失败的测试数量（目标: 0）
- **Lint Errors**: Black/isort报告的错误数（目标: 0）

### 架构度量（每月）
- **Blueprint Consistency**: 文档与代码一致性（目标: 100%）
- **Dependency Freshness**: 依赖更新延迟天数（目标: <30天）
- **Tech Debt Items**: 未解决的TODO/FIXME数量（目标: 下降趋势）

---

## 9. Communication Plan（沟通计划）

### Daily Standup（每日站会 - 10分钟）
- 时间: 每天10:00 AM
- 参与者: Platform Engineer, DevOps, QA
- 内容:
  - 昨天完成: 哪些任务merged
  - 今天计划: 哪些任务开始
  - 阻塞点: 需要协助的问题

### Weekly Review（每周回顾 - 1小时）
- 时间: 每周五16:00 PM
- 参与者: 全团队 + Stakeholders
- 内容:
  - Demo本周完成的功能
  - 回顾度量指标（cycle time, coverage等）
  - Retrospective: 做得好的和需要改进的

### Milestone Review（里程碑回顾 - 2小时）
- 时间: Week 2/4结束时
- 参与者: 全团队 + Executive Sponsor
- 内容:
  - 演示系统端到端运行
  - 架构决策记录（ADR）回顾
  - 下一阶段规划调整

---

## 10. Success Criteria（成功标准）

### Week 1 Success（核心稳定性）
- [x] State store并发锁实现并通过测试
- [x] 依赖版本锁定文件生成
- [x] 所有scripts使用统一路径管理

### Week 2 Success（自动化基础）
- [x] GitHub Actions CI自动运行pytest + gate_check
- [x] WIP限制强制执行（≤3任务running）
- [x] 状态验证脚本可检测非法转换

### Week 4 Success（最终验证）
- [x] 所有P0和P1任务完成
- [x] Test coverage >80%
- [x] Metrics dashboard可自动生成
- [x] Blueprint consistency检查100%通过
- [x] 完整端到端流程可复现

---

## 11. Rollout Plan（推广计划）

### Phase 1: Internal Validation（Week 4）
- 团队内部使用新流程1周
- 收集反馈并快速迭代

### Phase 2: Limited Rollout（Week 5-6）
- 选择1-2个pilot项目（如DGSF）
- 提供培训和支持

### Phase 3: Full Rollout（Week 7+）
- 所有新项目强制使用新流程
- Legacy项目逐步迁移

---

## 12. Appendix（附录）

### 12.1 参考文档
- [ARCHITECTURE_PACK_INDEX.md](../ARCHITECTURE_PACK_INDEX.md) - 架构蓝图索引
- [PROJECT_PLAYBOOK.md](../PROJECT_PLAYBOOK.md) - 项目生命周期指南
- [SPEC_GOVERNANCE_MODEL.mmd](../SPEC_GOVERNANCE_MODEL.mmd) - 规范治理模型

### 12.2 工具清单
- pytest: 测试框架
- pytest-cov: 覆盖率报告
- black/isort: 代码格式化
- pyright: 静态类型检查
- GitHub Actions: CI/CD平台

### 12.3 联系人
- **Platform Engineer**: 负责kernel/核心功能
- **DevOps Engineer**: 负责CI/CD和自动化
- **Data Engineer**: 负责metrics和可观测性
- **QA Engineer**: 负责测试策略和质量保证

---

**Last Updated**: 2026-02-02  
**Next Review**: 2026-02-09 (Week 1结束时)  
**Status**: 🟢 ACTIVE - 等待执行
