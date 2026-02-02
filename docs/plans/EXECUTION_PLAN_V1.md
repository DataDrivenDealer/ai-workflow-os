# AI Workflow OS - 执行计划 V1.0

**文档ID**: EXECUTION_PLAN_V1  
**创建日期**: 2026-02-02  
**状态**: ACTIVE  
**基于**: 2026-02-02 六专家委员会诊断报告  
**目标周期**: 2026-02-03 至 2026-03-02 (4周)  
**WIP限制**: 最多3个并行工作流

---

## 0. Objectives & Non-goals（目标与非目标）

### ✅ Objectives（目标）
1. **健壮化State管理** - 消除并发写入风险，支持ACID事务
2. **自动化Gate检查** - CI/CD流水线自动执行治理检查
3. **可观测性提升** - 建立度量体系，可视化cycle time和throughput
4. **代码质量稳定** - 测试覆盖率>80%，所有scripts可复现运行
5. **架构一致性** - 文档与代码保持同步，blueprint可自动验证

### ❌ Non-goals（非目标）
- 不重写整个系统（采用Strangler Fig渐进式迁移）
- 不追求100%测试覆盖率（聚焦核心路径）
- 不立即迁移到Kubernetes（先完成单机稳定性）

---

## 1. Current State Summary（当前状态摘要）

**证据来源**: Git analysis @ 2026-02-02

### 1.1 系统健康度
| 维度 | 评分 | 证据 |
|-----|------|------|
| 架构设计 | 85/100 | ✅ 清晰的三层分离（kernel/projects/specs），MCP协议隔离 |
| 代码质量 | 72/100 | ✅ 128个单元测试通过，⚠️ 缺少projects层测试 |
| 流程自动化 | 65/100 | ✅ Git hooks存在，⚠️ 手动安装，无CI/CD |
| 可观测性 | 45/100 | ⚠️ 有audit日志但无度量仪表板 |
| 文档覆盖 | 80/100 | ✅ 13个架构蓝图，⚠️ 4个标记为"planned" |

**综合评分**: 69/100（架构优秀但运维滞后）

### 1.2 关键风险
- 🔴 **P0 Risk**: state/ YAML文件无并发控制，可能发生race condition
- 🟠 **P1 Risk**: 依赖版本未锁定，生产环境可能不一致
- 🟠 **P1 Risk**: 无WIP限制，多任务并行导致上下文切换成本高

### 1.3 未提交变更
- 8个modified文件（主要是state/和configs/）
- 3个untracked执行计划文档

---

## 2. Workstreams（工作流 - 最多3个并行）

### Workstream 1: 核心稳定性（P0优先级）
**Owner**: Platform Engineer  
**Duration**: Week 1-2  
**Goal**: 消除阻塞性技术风险

**Milestones**:
- **M1.1** (Week 1 Day 3): State store并发锁实现
- **M1.2** (Week 1 Day 5): 依赖版本锁定生成
- **M1.3** (Week 2 Day 2): 路径管理重构完成

### Workstream 2: 自动化增强（P1优先级）
**Owner**: DevOps Engineer  
**Duration**: Week 2-3  
**Goal**: 建立CI/CD流水线

**Milestones**:
- **M2.1** (Week 2 Day 4): GitHub Actions配置完成
- **M2.2** (Week 2 Day 5): 状态验证脚本集成
- **M2.3** (Week 3 Day 3): WIP限制逻辑部署

### Workstream 3: 可观测性（P2优先级）
**Owner**: Data Engineer  
**Duration**: Week 3-4  
**Goal**: 建立度量体系

**Milestones**:
- **M3.1** (Week 3 Day 5): Metrics dashboard原型
- **M3.2** (Week 4 Day 2): Coverage报告自动生成
- **M3.3** (Week 4 Day 5): 历史趋势可视化

---

## 3. Week-by-Week Sequence（周序列）

### Week 1: 突破阻塞（UNBLOCK）
**Theme**: 消除P0风险

| Day | Task | Owner | Output | Verification |
|-----|------|-------|--------|-------------|
| Mon | B-1: State store并发锁 | Platform | `kernel/state_store.py` | 并发测试通过 |
| Tue | B-2: 生成requirements-lock | Platform | `requirements-lock.txt` | `pip-sync`无错误 |
| Wed | B-3: 提交执行计划 | Platform | Git commit | `git status` clean |
| Thu | B-4: 路径管理重构(1/2) | Platform | `kernel/paths.py` | Import测试通过 |
| Fri | B-4: 路径管理重构(2/2) | Platform | 所有scripts迁移 | Smoke test通过 |

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
