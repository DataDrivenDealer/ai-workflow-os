---
task_id: "LEGACY_DGSF_ASSESS_001"
type: research
queue: research
branch: "feature/LEGACY_DGSF_ASSESS_001"
priority: P0
spec_ids:
  - ARCH_BLUEPRINT_MASTER
  - PROJECT_DELIVERY_PIPELINE
  - GOVERNANCE_INVARIANTS
  - DGSF_Architecture_v3.0
verification:
  - "Architecture reusability assessment completed"
  - "Specification value assessment documented"
  - "Data asset inventory compiled"
  - "Test coverage report generated"
  - "Integration recommendation approved by Project Owner"
---

# TaskCard: LEGACY_DGSF_ASSESS_001

> **Stage**: 0 · Legacy Asset Assessment  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `LEGACY_DGSF_ASSESS_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `architect` / `analyst` |
| **Authority** | `speculative` |
| **Authorized By** | Project Owner |
| **上游决策** | 继承 Legacy DGSF (SDF Asset Pricing Framework) |

---

## 1. 任务背景

### 1.1 决策记录
Project Owner 于 2026-02-01 确认：
- ✅ 选择 **选项 A**: 继承 Legacy DGSF (SDF Asset Pricing Framework)
- ❌ 摒弃之前的 RESEARCH_1 网格策略研究路线
- 🗑️ 已删除相关开发文档 (RESEARCH_1_DGSF_001, DATA_2_DGSF_001)

### 1.2 Legacy DGSF 概述
**Dynamic Generative SDF Forest (DGSF)** 是一个专业级量化研究框架，包含六大核心层：

| 层级 | 模块 | 规范版本 | 状态 |
|------|------|----------|------|
| L0-L1 | Data Engineering | v4.2 | 规范完成 |
| L2 | PanelTree | v3.0.2 | 规范+代码 |
| L3 | SDF Layer | v3.1 | 规范+代码 |
| L4 | EA Optimizer | v3.1 | 规范+代码 |
| L5 | Rolling & Evaluation | v3.0 | 规范+代码 |
| L6-L7 | Telemetry & Stability | - | 待补充 |

---

## 2. 评估目标

### 2.1 架构可复用性评估 (李架构 负责)
- [ ] 评估 DGSF Architecture v3.0 与 AI Workflow OS 架构的兼容性
- [ ] 识别需要适配的接口层
- [ ] 评估代码模块的独立性和可集成性
- [ ] 生成 `ARCH_REUSE_ASSESSMENT.md`

### 2.2 规范学术价值评估 (陈研究 负责)
- [ ] 审查 specs_v3/ 目录下所有规范的学术完整性
- [ ] 评估 Baseline A-H 生态系统的科学严谨性
- [ ] 验证方法论与主流学术文献的一致性
- [ ] 生成 `SPEC_VALUE_ASSESSMENT.md`

### 2.3 数据资产清点 (王数据 负责)
- [ ] 清点 data/ 目录下的所有数据资产
- [ ] 验证数据完整性和可用性
- [ ] 评估数据流水线配置的可复用性
- [ ] 生成 `DATA_ASSET_INVENTORY.md`

### 2.4 测试覆盖率报告 (林质量 负责)
- [ ] 运行现有测试套件
- [ ] 评估测试覆盖率
- [ ] 识别测试缺口
- [ ] 生成 `TEST_COVERAGE_REPORT.md`

---

## 3. Legacy 资产清单

### 3.1 规范文档 (docs/specs_v3/)
```
├── DGSF Architecture v3.0 _ Final.md (3907 lines, 母规范)
├── DGSF Project Specification Master Roadmap v3.0.md
├── DGSF PanelTree Layer Specification v3.0.2.md
├── DGSF SDF Layer Specification v3.1.md
├── DGSF EA Layer Specification v3.1.md
├── DGSF Rolling & Evaluation Specification v3.0.md
├── DGSF Baseline System Specification v4.3.md
├── DGSF Rolling Baseline Execution Framework v3.1.md
└── DGSF spec_version_index.md
```

### 3.2 源代码 (src/dgsf/)
```
├── backtest/    # 回测引擎
├── config/      # 配置管理
├── data/        # 数据加载
├── dataeng/     # 数据工程
├── ea/          # 演化算法优化器
├── eval/        # 模型评估
├── experiments/ # 实验运行器
├── factors/     # 因子计算
├── paneltree/   # 面板树模型
├── rolling/     # 滚动窗口
├── sdf/         # 随机折现因子
└── utils/       # 工具函数
```

### 3.3 配置资产 (configs/)
- 75+ YAML 配置文件
- 覆盖：数据工程流水线、因子面板、回测参数、滚动窗口等

### 3.4 数据资产 (data/)
```
├── a0/          # A股原始数据
├── cache/       # 缓存数据
├── final/       # 最终处理结果
├── full/        # 完整数据集
├── interim/     # 中间数据
├── paneltree/   # PanelTree 输出
├── processed/   # 处理后数据
└── raw/         # 原始数据
```

### 3.5 研究成果 (results/)
- SDF gamma grid 静态 OOS 证据包
- OOS horizon 稳健性报告
- Expanding minloop 报告

---

## 4. 输出 Artifacts

| Artifact | 路径 | 负责人 | 状态 |
|----------|------|--------|------|
| 架构可复用性评估 | `projects/dgsf/docs/ARCH_REUSE_ASSESSMENT.md` | 李架构 | ✅ `complete` |
| 规范价值评估 | `projects/dgsf/docs/SPEC_VALUE_ASSESSMENT.md` | 陈研究 | ✅ `complete` |
| 数据资产清单 | `projects/dgsf/docs/DATA_ASSET_INVENTORY.md` | 王数据 | ✅ `complete` |
| 测试覆盖报告 | `projects/dgsf/docs/TEST_COVERAGE_REPORT.md` | 林质量 | ✅ `complete` |
| 集成建议书 | `projects/dgsf/docs/INTEGRATION_RECOMMENDATION.md` | 团队 | ✅ `complete` |

---

## 5. Gate & 下游依赖

- **Gate G0**: Legacy Assessment Review
  - 所有评估报告完成
  - Project Owner 批准集成建议书
- **后续 TaskCard**: `SPEC_INTEGRATION_001`
- **依赖**: 无上游任务

---

## 6. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: Project Owner
  scope: legacy_assessment
  decision_required: true
  
# 评估完成后需要 Project Owner accept 集成建议书
# 才能进入 SPEC_INTEGRATION_001
```

---

## 7. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T22:00:00Z | project_owner | `direction_confirmed` | 选择继承 Legacy DGSF |
| 2026-02-01T22:00:00Z | system | `task_created` | 创建 Legacy 评估任务 |
| 2026-02-01T22:00:00Z | system | `deprecated_tasks_removed` | 删除 RESEARCH_1, DATA_2 相关文档 |
