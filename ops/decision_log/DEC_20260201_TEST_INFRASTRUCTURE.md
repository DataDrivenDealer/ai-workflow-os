# Decision Record: DEC_20260201_TEST_INFRASTRUCTURE

**ID**: DEC_20260201_001  
**Date**: 2026-02-01  
**Status**: accepted  
**Decider**: Platform Engineering (Alex 部署)

---

## Context

AI Workflow OS kernel模块缺少单元测试覆盖，`requirements.txt`中也未包含测试框架依赖。这导致：
1. CI流水线无法验证代码质量
2. 本地开发无法运行测试
3. 重构代码时缺乏回归保护

按照项目扫描分析，这是P0级别的阻塞性问题。

## Decision

**添加完整的测试基础设施**：

1. 更新`requirements.txt`添加以下依赖：
   - `pytest>=8.0.0`
   - `pytest-cov>=4.1.0`
   - `pytest-asyncio>=0.23.0`
   - 开发工具：`black`, `isort`, `mypy`

2. 创建`kernel/tests/`目录结构：
   - `test_state_store.py` - 状态存储测试
   - `test_task_parser.py` - TaskCard解析测试
   - `test_os.py` - 核心CLI命令测试

3. 更新CI workflow添加`kernel-tests` job

## Rationale

### Alternatives Considered

1. **不添加测试** 
   - Pros: 无需额外工作
   - Cons: 代码质量无法保证，CI形同虚设

2. **仅添加集成测试**
   - Pros: 覆盖端到端流程
   - Cons: 调试困难，反馈周期长

3. **添加单元测试（选择此方案）**
   - Pros: 快速反馈，易于维护，覆盖核心逻辑
   - Cons: 需要初始投入时间

### Why This Choice

单元测试是质量保证的基础层，符合项目渐进式改进原则。投入产出比最高。

## Consequences

### Positive

- ✅ CI可以验证每次提交的代码质量
- ✅ 开发者可以本地快速验证修改
- ✅ 为后续重构提供安全网
- ✅ 32个测试覆盖核心模块

### Negative

- ⚠️ 增加了依赖包数量
- ⚠️ 需要维护测试代码

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 测试维护成本高 | Low | Low | 保持测试简洁，使用fixtures |
| 测试覆盖不足 | Medium | Medium | 持续添加测试 |

## Related

- **Specs**: ARCH_BLUEPRINT_MASTER
- **Tasks**: TASK_INFRA_0001 (pending creation)
- **Previous Decisions**: None

## Acceptance

| Role | Name | Date | Status |
|------|------|------|--------|
| Owner | Platform Engineering | 2026-02-01 | ✅ accepted |
| Reviewer | Dr. 陈守护 | 2026-02-01 | ✅ accepted |

---

**Change Log**:
| Date | Author | Change |
|------|--------|--------|
| 2026-02-01 | Alex 部署 | Initial creation |
| 2026-02-01 | Dr. 陈守护 | Review and acceptance |
