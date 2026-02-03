# TaskCard: SDF Feature Engineering Pipeline (T3)

**Task ID**: SDF_FEATURE_ENG_001  
**Parent**: SDF_DEV_001_T3 (Stage 4 - SDF Layer Development)  
**Created**: 2026-02-03  
**Status**: IN_PROGRESS  
**Priority**: P0  
**Estimated Effort**: 3 weeks  
**Owner**: DGSF Researcher

---

## 📋 Task Overview

**Objective**: 构建增强型特征工程管道，为 SDF（Stochastic Discount Factor）模型提供高质量输入特征，包括因子（factors）、收益率（returns）和公司特征（firm characteristics）。

**Alignment**: 
- [PROJECT_DGSF.yaml - SDF_DEV_001_T3](../../specs/PROJECT_DGSF.yaml)
- [STAGE_4_ACCEPTANCE_CRITERIA.md - AC-3](../STAGE_4_ACCEPTANCE_CRITERIA.md)
- [SDF_SPEC v3.1](../../legacy/DGSF/docs/specs_v3/DGSF%20SDF%20Layer%20Specification%20v3.1.md)

**Dependencies**:
- ✅ SDF_DEV_001_T1 (Model Architecture Review) - COMPLETED
- ✅ SDF_DEV_001_T2 (Fix Test Failures) - COMPLETED (93.4% pass rate)
- ✅ T2 → T3 Readiness Gate: OPEN

---

## 🎯 Acceptance Criteria (from AC-3)

| # | Criterion | Verification | Status |
|---|-----------|--------------|--------|
| 1 | Feature engineering pipeline executable | `Test-Path scripts/run_feature_engineering.py` | ⏸️ |
| 2 | Feature importance analysis complete | `Test-Path experiments/feature_ablation/results.json` | ⏸️ |
| 3 | Feature definitions documented | Aligned with SDF_SPEC v3.1 | ⏸️ |
| 4 | ≥1 experiment shows feature contribution | Ablation study with p < 0.05 | ⏸️ |

---

## 📁 Existing Resources

**现有代码模块**（位于 `repo/src/dgsf/factors/`）:
- `definitions.py` - 特征定义
- `compute.py` - 特征计算逻辑
- `cleaning.py` - 数据清洗
- `leaf_features.py` - 叶子节点特征（PanelTree 相关）

**现有特征类型**（待盘点确认）:
- [ ] Firm characteristics (size, book-to-market, momentum, etc.)
- [ ] Returns (raw, excess, risk-adjusted)
- [ ] Factors (market, SMB, HML, etc.)
- [ ] Tree-derived features (leaf assignments, cluster means)

---

## 📋 Subtask Breakdown（子任务拆解）

### T3.1: 现有特征盘点 (Day 1-2)
**Effort**: 4 hours  
**Output**: `reports/SDF_FEATURE_INVENTORY.json`

**Steps**:
1. 扫描 `repo/src/dgsf/factors/` 所有模块
2. 识别已定义的特征（名称、类型、计算公式）
3. 与 SDF_SPEC v3.1 对比，标记缺失特征
4. 生成 JSON 格式清单

**DoD**:
- [ ] JSON 包含至少 10 个特征定义
- [ ] 每个特征有 name, type, source, formula 字段
- [ ] 标记 SDF_SPEC v3.1 覆盖状态

**Verification**:
```powershell
$inv = Get-Content projects/dgsf/reports/SDF_FEATURE_INVENTORY.json | ConvertFrom-Json
$inv.features.Count -ge 10  # Expected: True
```

---

### T3.2: 特征定义文档化 (Day 3-4)
**Effort**: 6 hours  
**Output**: `docs/SDF_FEATURE_DEFINITIONS.md`

**Steps**:
1. 整合 T3.1 的清单与 SDF_SPEC v3.1 要求
2. 为每个特征撰写：定义、计算公式、数据来源、更新频率
3. 定义特征分类（firm characteristics / returns / factors / tree-derived）
4. 标注必需 vs. 可选特征

**DoD**:
- [ ] 文档覆盖所有 SDF_SPEC v3.1 必需特征
- [ ] 每个特征有完整 5 要素（定义、公式、来源、频率、类别）
- [ ] 包含特征依赖图（哪些特征依赖其他特征）

**Verification**:
```powershell
Test-Path projects/dgsf/docs/SDF_FEATURE_DEFINITIONS.md  # Expected: True
Select-String -Path projects/dgsf/docs/SDF_FEATURE_DEFINITIONS.md -Pattern "## Feature:" | Measure-Object
# Expected: Count >= 10
```

---

### T3.3: Feature Construction Pipeline 骨架 (Day 5-7)
**Effort**: 12 hours  
**Output**: `scripts/run_feature_engineering.py`

**Steps**:
1. 创建主入口脚本，接受配置文件参数
2. 实现特征计算流程：Load Data → Clean → Compute → Validate → Output
3. 支持增量更新（仅计算新日期的特征）
4. 添加日志和错误处理

**接口设计**:
```python
# scripts/run_feature_engineering.py
def main(config_path: str, output_dir: str, start_date: str, end_date: str):
    """
    Run feature engineering pipeline.
    
    Args:
        config_path: Path to feature config YAML
        output_dir: Directory to save computed features
        start_date: Start date for computation (YYYY-MM-DD)
        end_date: End date for computation (YYYY-MM-DD)
    """
    pass
```

**DoD**:
- [ ] 脚本可执行 `python scripts/run_feature_engineering.py --help`
- [ ] 支持 `--config`, `--output-dir`, `--start-date`, `--end-date` 参数
- [ ] 有 `--dry-run` 模式（不实际计算，仅打印计划）

**Verification**:
```powershell
cd projects/dgsf/repo
python scripts/run_feature_engineering.py --help  # Expected: 无错误
```

---

### T3.4: Baseline 特征集验证 (Day 8-10)
**Effort**: 8 hours  
**Output**: `experiments/feature_baseline/results.json`

**Steps**:
1. 定义 baseline 特征集（基于 SDF_SPEC v3.1 的核心特征）
2. 使用 T3.3 的 pipeline 计算特征
3. 验证特征统计量（mean, std, min, max, missing rate）
4. 与历史数据对比，确保一致性

**DoD**:
- [ ] Baseline 特征集包含 ≥5 个核心特征
- [ ] 所有特征 missing rate < 5%
- [ ] 特征统计量与历史数据偏差 < 1%

**Verification**:
```powershell
Test-Path projects/dgsf/repo/experiments/feature_baseline/results.json
$results = Get-Content projects/dgsf/repo/experiments/feature_baseline/results.json | ConvertFrom-Json
$results.missing_rate -lt 0.05  # Expected: True
```

---

### T3.5: Ablation Study 设计 (Day 11-12)
**Effort**: 6 hours  
**Output**: `experiments/feature_ablation/design.yaml`

**Steps**:
1. 定义消融实验设计（每次移除一个特征组）
2. 指定评估指标（Sharpe, pricing error, R²）
3. 定义统计显著性检验方法（t-test, bootstrap）
4. 创建实验配置文件

**消融分组（示例）**:
- Group A: Remove firm size features
- Group B: Remove momentum features
- Group C: Remove book-to-market features
- Group D: Remove tree-derived features

**DoD**:
- [ ] 设计包含 ≥4 个消融组
- [ ] 每组指定移除的特征列表
- [ ] 定义统计显著性阈值（p < 0.05）

**Verification**:
```powershell
Test-Path projects/dgsf/repo/experiments/feature_ablation/design.yaml
```

---

### T3.6: Ablation 实验执行 (Day 13-17)
**Effort**: 16 hours  
**Output**: `experiments/feature_ablation/results.json`

**Steps**:
1. 运行 baseline 模型（所有特征）
2. 依次运行每个消融组
3. 记录每组的评估指标
4. 计算特征贡献（baseline vs. ablation 差异）
5. 执行统计显著性检验

**DoD**:
- [ ] 所有消融组完成训练和评估
- [ ] 结果 JSON 包含每组的 Sharpe, pricing_error, R², p_value
- [ ] ≥3 个特征组显示统计显著贡献（p < 0.05）

**Verification**:
```powershell
$results = Get-Content projects/dgsf/repo/experiments/feature_ablation/results.json | ConvertFrom-Json
($results.groups | Where-Object { $_.p_value -lt 0.05 }).Count -ge 3  # Expected: True
```

---

### T3.7: 特征重要性报告 (Day 18-21)
**Effort**: 8 hours  
**Output**: `reports/SDF_FEATURE_IMPORTANCE_REPORT.md`

**Steps**:
1. 整合 T3.6 的实验结果
2. 排序特征组的贡献度（按 Sharpe 影响）
3. 可视化特征重要性（bar chart）
4. 撰写结论和建议

**DoD**:
- [ ] 报告包含特征重要性排序
- [ ] 包含可视化图表（PNG 或嵌入）
- [ ] 结论明确哪些特征是"must-have"

**Verification**:
```powershell
Test-Path projects/dgsf/reports/SDF_FEATURE_IMPORTANCE_REPORT.md
Select-String -Path projects/dgsf/reports/SDF_FEATURE_IMPORTANCE_REPORT.md -Pattern "Conclusion"
# Expected: Match found
```

---

## 📅 Timeline

| Week | Subtasks | Deliverables |
|------|----------|--------------|
| Week 1 | T3.1, T3.2, T3.3 | Inventory, Definitions, Pipeline script |
| Week 2 | T3.4, T3.5, T3.6 (partial) | Baseline validation, Ablation design |
| Week 3 | T3.6 (complete), T3.7 | Ablation results, Importance report |

---

## 🚦 T3 → T4 Readiness Gate

**Gate Definition**: T3 完成后，满足以下条件可启动 T4 (Training Optimization)

1. ✅ `scripts/run_feature_engineering.py` 可执行
2. ✅ `experiments/feature_ablation/results.json` 存在
3. ✅ Ablation 结果: ≥3 features 的 p-value < 0.05
4. ✅ `SDF_FEATURE_DEFINITIONS.md` 完成

---

## 📝 Notes

- **数据依赖**: T3.4-T3.6 需要真实数据，确保 `projects/dgsf/data/` 可访问
- **GPU 需求**: T3.6 的模型训练可能需要 CUDA（若无 GPU 可使用 CPU 模式，但耗时更长）
- **规范对齐**: 所有特征定义必须与 SDF_SPEC v3.1 一致

---

**Last Updated**: 2026-02-03T21:30Z  
**Created By**: Project Orchestrator
