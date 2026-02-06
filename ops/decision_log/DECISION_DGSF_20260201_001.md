# DGSF 方向变更决策记录

**决策 ID**: DECISION_DGSF_20260201_001  
**日期**: 2026-02-01  
**决策者**: Project Owner  
**状态**: ✅ CONFIRMED

---

## 1. 决策内容

### 选择的方向
**选项 A**: 继承 Legacy DGSF (SDF Asset Pricing Framework)

### 摒弃的方向
**原 RESEARCH_1 路线**: 加密货币动态网格策略

---

## 2. 决策依据

### Legacy DGSF 资产价值
| 资产类型 | 数量/规模 | 评估价值 |
|----------|-----------|----------|
| 核心规范 | 8份 (Architecture v3.0 含 3907 行) | ⭐⭐⭐⭐⭐ |
| 源代码模块 | 12个核心模块 | ⭐⭐⭐⭐ |
| 配置文件 | 75+ 个 | ⭐⭐⭐⭐ |
| 数据资产 | 多目录结构化数据 | ⭐⭐⭐⭐ |
| 研究成果 | OOS 证据包 + 稳健性报告 | ⭐⭐⭐⭐⭐ |

### 架构优势
- 完整的 L0-L5 六层架构设计
- 严格的因果性原则 (无 look-ahead 泄漏)
- Baseline A-H 生态系统支持科学对照
- 滚动窗口机制保证稳健性

---

## 3. 已执行操作

### 删除的文件
- [x] `tasks/RESEARCH_1_DGSF_001.md` - 网格策略研究设计
- [x] `tasks/DATA_2_DGSF_001.md` - 网格策略数据工程
- [x] `projects/dgsf/docs/RESEARCH_1_design.md` - 研究设计文档
- [x] `projects/dgsf/docs/RESEARCH_1_repro.md` - 可复现性包
- [x] `projects/dgsf/configs/RESEARCH_1.yaml` - 实验配置
- [x] `ops/audit/RESEARCH_1_DGSF_001.md` - 审计记录

### 更新的文件
- [x] `state/tasks.yaml` - 移除已删除任务，添加新任务
- [x] `projects/dgsf/specs/PROJECT_DGSF.yaml` - 升级到 v2.0.0，反映新方向

### 创建的文件
- [x] `tasks/LEGACY_DGSF_ASSESS_001.md` - Legacy 资产评估任务卡
- [x] 本决策记录

---

## 4. 下一步行动

### Phase 1: Legacy 资产评估 (当前)
1. 架构可复用性评估 → `ARCH_REUSE_ASSESSMENT.md`
2. 规范学术价值评估 → `SPEC_VALUE_ASSESSMENT.md`
3. 数据资产清点 → `DATA_ASSET_INVENTORY.md`
4. 测试覆盖率报告 → `TEST_COVERAGE_REPORT.md`

### Phase 2: 规范集成 (待评估完成后)
- 设计 Legacy specs_v3 → AI Workflow OS 的映射方案
- 建立统一规范层级

### Phase 3: 数据迁移 (待集成方案确定后)
- 链接或迁移 Legacy 数据资产
- 验证数据完整性

### Phase 4: 可复现性验证 (待数据就绪后)
- 运行现有实验
- 验证结果一致性

---

## 5. 专家团队分工

| 角色 | 负责人 | 当前任务 |
|------|--------|----------|
| 首席架构师 | 李架构 | 架构可复用性评估 |
| 首席量化研究员 | 陈研究 | 规范学术价值评估 |
| 首席数据工程师 | 王数据 | 数据资产清点 |
| 平台架构师 | 张平台 | 集成方案设计 |
| QA 工程师 | 林质量 | 测试覆盖率报告 |

---

## 6. 签署

```
决策确认: Project Owner
日期: 2026-02-01
签名: [CONFIRMED]
```
