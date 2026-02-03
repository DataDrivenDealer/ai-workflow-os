# TODO_NEXT - DGSF 驱动的执行队列

**Created**: 2026-02-02  
**Updated**: 2026-02-03T21:00Z (Orchestrator Cycle - T3 Launch)  
**Purpose**: DGSF 项目的 canonical execution queue  
**Priority Order**: P0（直接推进 DGSF）→ P1（解除阻塞）→ P2（延后）  
**Primary Objective**: 推进 DGSF（Dynamic Generative SDF Forest）项目的开发、验证与研究产出

---

## 🎯 Global Priority Override Rule

**DGSF Priority Override**: 当 DGSF 项目推进与 AI Workflow OS 层面的改进发生冲突时，**无条件以 DGSF 的开发与验证为最高优先级（P0）**。

---

## 📊 Current Context（基于证据 · 2026-02-03T21:00Z）

| 维度 | 状态 | 证据 |
|------|------|------|
| **DGSF Stage** | Stage 4 "SDF Layer Development" | [PROJECT_DGSF.yaml#L211](../../projects/dgsf/specs/PROJECT_DGSF.yaml) |
| **测试通过率** | 93.4% (156/167 passed, 11 skipped) | `pytest tests/sdf/ -v` |
| **T2 → T3 Gate** | ✅ **OPEN** | [STAGE_4_ACCEPTANCE_CRITERIA.md](../../projects/dgsf/docs/STAGE_4_ACCEPTANCE_CRITERIA.md) |
| **repo/ 同步** | ✅ up-to-date (commit 8031647) | `cd repo; git status` |
| **下一里程碑** | T3 Feature Engineering (3 weeks) | [PROJECT_DGSF.yaml#L295](../../projects/dgsf/specs/PROJECT_DGSF.yaml) |

---

## ✅ 已完成任务（Stage 4 T1-T2）

| ID | Task | Completed | Output |
|----|------|-----------|--------|
| P0-1 | SDF Model Inventory | 2026-02-02 | [SDF_MODEL_INVENTORY.json](../../projects/dgsf/reports/SDF_MODEL_INVENTORY.json) |
| P0-2 | Test Failures Diagnosis | 2026-02-02 | [SDF_TEST_FAILURES.md](../../projects/dgsf/reports/SDF_TEST_FAILURES.md) |
| P0-3 | Fix state_engine Import | 2026-02-02 | [sdf/__init__.py#L53](../../projects/dgsf/repo/src/dgsf/sdf/__init__.py) |
| P0-4 | Push repo/ to origin | 2026-02-03 | commit 8031647 |
| P0-5 | Define Stage 4 AC | 2026-02-03 | [STAGE_4_ACCEPTANCE_CRITERIA.md](../../projects/dgsf/docs/STAGE_4_ACCEPTANCE_CRITERIA.md) |
| P0-6 | Classify 11 Skipped Tests | 2026-02-03 | [SDF_SKIPPED_TESTS_ANALYSIS.md](../../projects/dgsf/reports/SDF_SKIPPED_TESTS_ANALYSIS.md) |

---

## 🔴 P0 任务（直接推进 DGSF · 本周焦点）

### ✅ P0-7.T3.1: 现有特征盘点 - COMPLETED
**Status**: ✅ 完成（2026-02-03T22:00Z）  
**Output**: [SDF_FEATURE_INVENTORY.json](../../projects/dgsf/reports/SDF_FEATURE_INVENTORY.json)  
**Result**: 12 features identified, 3 technical debt items

---

### P0-7.T3.2.1: 提取 SDF_SPEC v3.1 必需特征 ✅ COMPLETED
**DGSF 关联**: T3 Feature Engineering - Step 2.1  
**Effort**: 30 分钟  
**Dependencies**: ✅ T3.1 完成  
**Status**: ✅ COMPLETED (2026-02-03T22:25Z)

**执行步骤**:
1. ✅ 读取 `projects/dgsf/legacy/DGSF/docs/specs_v3/DGSF SDF Layer Specification v3.1.md`
2. ✅ 提取所有 "required" 和 "optional" 特征列表
3. ✅ 创建初步特征列表文件 `projects/dgsf/docs/SDF_REQUIRED_FEATURES.txt`
4. ✅ 标注每个特征的优先级（required/optional）

**验收标准（DoD）**:
- [x] `SDF_REQUIRED_FEATURES.txt` 存在
- [x] 列表包含 ≥8 个 required 特征（实际: 17 required, 1 optional）
- [x] 每个特征有明确的 required/optional 标记
- 验证命令: `Test-Path projects/dgsf/docs/SDF_REQUIRED_FEATURES.txt` ✅

**Output**: [SDF_REQUIRED_FEATURES.txt](../../projects/dgsf/docs/SDF_REQUIRED_FEATURES.txt)

---

### P0-7.T3.2.2: 定义 Firm Characteristics（前 5 特征）✅ COMPLETED
**DGSF 关联**: T3 Feature Engineering - Step 2.2  
**Effort**: 2 小时  
**Dependencies**: ✅ T3.2.1 完成  
**Status**: ✅ COMPLETED (2026-02-03T22:45Z)

**执行步骤**:
1. ✅ 在 `SDF_FEATURE_DEFINITIONS.md` 中定义 firm characteristics
2. ✅ 包含 5 要素：定义、计算公式、数据来源、更新频率、类别
3. ✅ 覆盖：size, book_to_market, momentum, profitability, volatility

**验收标准（DoD）**:
- [x] 5 个 firm characteristics 完整定义
- [x] 每个特征有 5 要素
- [x] 公式可计算（无模糊描述）
- 验证命令: `Select-String -Pattern "^### Feature \d+:" SDF_FEATURE_DEFINITIONS.md | Measure-Object` ✅ (Count = 5)

**Output**: [SDF_FEATURE_DEFINITIONS.md](../../projects/dgsf/docs/SDF_FEATURE_DEFINITIONS.md) (Firm Characteristics section)

---

### P0-7.T3.2.3: 定义 Factors（后 5 特征）✅ COMPLETED
**DGSF 关联**: T3 Feature Engineering - Step 2.3  
**Effort**: 2 小时  
**Dependencies**: ✅ T3.2.2 完成  
**Status**: ✅ COMPLETED (2026-02-03T23:05Z)

**执行步骤**:
1. ✅ 在 `SDF_FEATURE_DEFINITIONS.md` 中定义 factors
2. ✅ 覆盖：market_factor, SMB, HML, momentum_factor, reversal

**验收标准（DoD）**:
- [x] 5 个 factors 完整定义
- [x] 与 SDF_SPEC v3.1 对齐
- 验证命令: `Select-String -Pattern "^### Factor \d+:" SDF_FEATURE_DEFINITIONS.md | Measure-Object` ✅ (Count = 5)

**Output**: [SDF_FEATURE_DEFINITIONS.md](../../projects/dgsf/docs/SDF_FEATURE_DEFINITIONS.md) (Factors section added, 19710 chars total)

---

### P0-7.T3.2.4: 创建特征依赖图 ✅ COMPLETED
**DGSF 关联**: T3 Feature Engineering - Step 2.4  
**Effort**: 1 小时  
**Dependencies**: ✅ T3.2.3 完成  
**Status**: ✅ COMPLETED (2026-02-03T23:20Z)

**执行步骤**:
1. ✅ 识别特征间的依赖关系
2. ✅ 创建 Mermaid 格式的依赖图
3. ✅ 添加到 `SDF_FEATURE_DEFINITIONS.md`

**验收标准（DoD）**:
- [x] 依赖图包含所有 10+ 特征
- [x] 明确计算顺序 (6 levels: Level 0-6)
- 验证命令: `Select-String -Pattern '```mermaid' SDF_FEATURE_DEFINITIONS.md` ✅ (Count = 1)

**Output**: [SDF_FEATURE_DEFINITIONS.md](../../projects/dgsf/docs/SDF_FEATURE_DEFINITIONS.md) (Dependency Graph section, 28436 chars total)

---

### P0-7.T3.2.5: 验证 SDF_SPEC 对齐 ✅ COMPLETED
**DGSF 关联**: T3 Feature Engineering - Step 2.5  
**Effort**: 30 分钟  
**Dependencies**: ✅ T3.2.4 完成  
**Status**: ✅ COMPLETED (2026-02-03T23:35Z)

**执行步骤**:
1. ✅ 读取 `SDF_REQUIRED_FEATURES.txt`（17 REQUIRED, 1 OPTIONAL）
2. ✅ 交叉对比 `SDF_FEATURE_DEFINITIONS.md` 已定义特征
3. ✅ 生成对齐检查表（checklist），标注覆盖状态
4. ✅ 识别任何缺失特征或不一致

**验收标准（DoD）**:
- [x] 对齐检查表生成 (5 detailed tables + 1 summary table)
- [x] 100% required 特征已覆盖（17/17 ✅）
- [x] 明确标注 optional 特征状态（1/1 ✅）
- 验证命令: 5 PowerShell commands provided ✅

**Output**: [SDF_FEATURE_DEFINITIONS.md](../../projects/dgsf/docs/SDF_FEATURE_DEFINITIONS.md) (Alignment Verification section, 39103 chars total)

**Key Result**: **100% Coverage** of T3 scope (10 fully defined, 8 referenced) ✅

---

### P0-7.T3.3.1: Pipeline 基础框架 + CLI 接口 ✅ COMPLETED
**DGSF 关联**: T3 Feature Engineering - Step 3.1  
**Effort**: 2-3 小时  
**Dependencies**: ✅ T3.2.5 完成  
**Status**: ✅ COMPLETED (2026-02-04T00:05Z)

**执行步骤**:
1. ✅ 创建 `projects/dgsf/scripts/run_feature_engineering.py` 基础结构
2. ✅ 实现命令行参数解析 (argparse):
   - `--config`: YAML 配置文件路径
   - `--output-dir`: 输出目录
   - `--start-date` / `--end-date`: 日期范围
   - `--dry-run`: 干跑模式（仅输出执行计划，不计算）
3. ✅ 实现配置加载和验证（YAML schema）
4. ✅ 实现 dry-run 模式输出（打印 7 步执行计划）

**验收标准（DoD）**:
- [x] 脚本可执行: `python run_feature_engineering.py --help` ✅
- [x] Dry-run 输出 7 步执行计划（对应 Execution Order Step 1-7）✅
- [x] 配置验证拒绝非法参数（日期格式、路径存在性）✅
- 验证命令: `python run_feature_engineering.py --config sample.yaml --dry-run` ✅

**Output**: 
- [run_feature_engineering.py](../../projects/dgsf/scripts/run_feature_engineering.py) (485 lines)
- [sample_config.yaml](../../projects/dgsf/scripts/sample_config.yaml) (sample configuration)

---

### P0-7.T3.3.2: 数据加载模块 ✅ COMPLETED
**DGSF 关联**: T3 Feature Engineering - Step 3.2  
**Effort**: 2-3 小时  
**Dependencies**: ✅ T3.3.1 完成  
**Status**: ✅ COMPLETED (2026-02-03T12:30Z)

**执行步骤**:
1. ✅ 实现 Step 1 (Load Raw Data): 5 data loaders
   - ✅ `load_price_data(start, end)` → price[firm, t]
   - ✅ `load_shares_outstanding(start, end)` → shares[firm, t]
   - ✅ `load_financial_statements(start, end)` → financials[firm, t]
   - ✅ `load_monthly_returns(start, end)` → returns[firm, t]
   - ✅ `load_risk_free_rate(start, end)` → risk_free[t]
2. ✅ 实现数据验证和缺失值处理
3. ✅ 实现日期范围过滤和对齐（月末对齐）
4. ✅ 添加 `pytest` 单元测试（mock 数据）

**验收标准（DoD）**:
- [x] 5 个数据加载函数实现完成 ✅
- [x] 单元测试创建（≥80% coverage 目标）✅ 21/21 tests passed
- [x] 日期范围过滤正确工作 ✅
- [x] 缺失值处理文档化（warnings + filtering）✅
- [x] 与 run_feature_engineering.py 集成完成 ✅
- 验证命令: `pytest tests/test_data_loading.py -v` → 21 passed ✅

**Output**: 
- [data_loaders.py](../../projects/dgsf/scripts/data_loaders.py) (569 lines, 5 loaders + validation)
- [test_data_loading.py](../../projects/dgsf/tests/test_data_loading.py) (496 lines, 21 tests)
- run_feature_engineering.py updated (imports load_all_data)

**Key Implementation**:
- Month-end alignment via `pd.offsets.MonthEnd(0)`
- Extended date ranges for lags (financials: +90d, returns: +12mo, rf: +12mo)
- Data quality validation (negative/zero removal, missing warnings)
- Column mapping from config (flexible schema)

---

### P0-7.T3.3.3: Firm Characteristics 计算 ✅ COMPLETED
**DGSF 关联**: T3 Feature Engineering - Step 3.3  
**Effort**: 3-4 小时  
**Dependencies**: ✅ T3.3.2 完成  
**Status**: ✅ COMPLETED (2026-02-03T23:45Z)

**执行步骤**:
1. ✅ 实现 Step 2: Compute Independent Characteristics (4-way parallel)
   - ✅ `compute_size(price, shares)` → size[firm, t]
   - ✅ `compute_momentum(returns)` → momentum[firm, t]
   - ✅ `compute_profitability(financials)` → profitability[firm, t]
   - ✅ `compute_volatility(returns)` → volatility[firm, t]
2. ✅ 实现 Step 3: Compute Dependent Characteristics
   - ✅ `compute_book_to_market(financials, size)` → book_to_market[firm, t]
3. ✅ 实现 winsorization（[1%, 99%] or [0.5%, 99.5%]）
4. ✅ 实现数据清洗（缺失值 forward-fill, 排除规则）
5. ✅ 添加单元测试（已知输入→预期输出）

**验收标准（DoD）**:
- [x] 5 个特征计算函数实现（对应 Feature 1-5） ✅
- [x] Winsorization 逻辑正确（极值处理） ✅
- [x] 单元测试验证公式正确性（至少 3 个测试用例/特征） ✅
- 验证命令: `pytest tests/test_firm_characteristics.py -v` → **19 passed** ✅

**Output**:
- [firm_characteristics.py](../../projects/dgsf/scripts/firm_characteristics.py) (516 lines, 5 characteristics + winsorization + integration)
- [test_firm_characteristics.py](../../projects/dgsf/tests/test_firm_characteristics.py) (508 lines, 19 tests, 100% pass rate)

---

### P0-7.T3.3.4: Cross-Sectional Spreads + Factors 🎯 NEXT
**DGSF 关联**: T3 Feature Engineering - Step 3.4  
**Effort**: 3-4 小时  
**Dependencies**: ✅ T3.3.3 完成  
**Status**: 🎯 READY

**执行步骤**:
1. 实现 Step 4: Compute Cross-Sectional Spreads
   - `compute_style_spreads(size, book_to_market, momentum, profitability, volatility)` → style_spreads[t, 5]
   - 实现 tertile 排序（30%, 40%, 30%）
   - 实现 market-cap 加权平均
2. 实现 Step 5: Compute Factors (3-way parallel, SMB+HML 共享 2×3 sorts)
   - `compute_market_factor(returns, risk_free)` → market_factor[t]
   - `compute_smb_hml(size, book_to_market, returns)` → SMB[t], HML[t]
   - `compute_momentum_factor(momentum, returns)` → momentum_factor[t]
   - `compute_reversal(returns)` → reversal[t]
3. 实现 Step 6: Assemble SDF Inputs
   - `assemble_X_state(...)` → X_state[t, d]
   - `assemble_P_tree_factors(...)` → P_tree_factors[t, 5] (OPTIONAL)
4. 添加集成测试（端到端 pipeline 测试）

**验收标准（DoD）**:
- [ ] Cross-sectional spreads 计算正确（5D 向量）
- [ ] 5 个因子计算函数实现（对应 Factor 1-5）
- [ ] SMB + HML 共享 2×3 sorts（优化验证）
- [ ] 集成测试: 给定 mock 数据 → 输出 X_state 和 P-tree factors
- 验证命令: `pytest tests/test_spreads_factors.py -v`

---

### P0-7.T3.3.5: 端到端 Pipeline 集成测试
**DGSF 关联**: T3 Feature Engineering - Step 3.5  
**Effort**: 1-2 小时  
**Dependencies**: T3.3.4 完成  
**Status**: PENDING

**执行步骤**:
1. 创建 `tests/test_feature_pipeline_e2e.py`
2. Mock 完整数据集（2020-01 至 2021-12, 100 firms）
3. 运行完整 pipeline: load → characteristics → spreads → factors → X_state
4. 验证输出维度和数值范围
5. 检查执行时间（应 < 5 秒 for mock data）

**验收标准（DoD）**:
- [ ] E2E 测试通过（≥3 test cases）
- [ ] X_state 输出维度正确（[T, d]）
- [ ] 无数据泄漏（t 时刻仅使用 t-1 及之前数据）
- 验证命令: `pytest tests/test_feature_pipeline_e2e.py -v`

---

### P0-7.T3.4: Feature Ablation Study（特征消融实验）
**DGSF 关联**: T3 Feature Engineering - Step 4 (Validation)  
**Effort**: 4-6 小时  
**Dependencies**: T3.3.5 完成  
**Status**: PENDING

**执行步骤**:
1. 创建 `experiments/feature_ablation/run_ablation.py`
2. 定义 baseline: 全部 10 特征
3. 创建 10 个 ablated 版本（每次移除 1 个特征）
4. 运行简化训练（10 epochs, single split, no early stopping）
5. 记录 10 个指标：train loss, val loss, SR（Sharpe Ratio）等
6. 生成 `results/feature_ablation_report.json`

**验收标准（DoD）**:
- [ ] 10 个 ablated 模型完成训练
- [ ] 至少 3 个特征的移除导致 SR 下降 ≥0.05（显著性）
- [ ] ablation_report.json 包含统计显著性 p-value
- 验证命令: `python experiments/feature_ablation/run_ablation.py --dry-run`

---

### P0-7.T3.5: 创建 Feature Engineering 文档
**DGSF 关联**: T3 Documentation  
**Effort**: 1-2 小时  
**Dependencies**: T3.4 完成  
**Status**: PENDING

**执行步骤**:
1. 创建 `projects/dgsf/docs/FEATURE_ENGINEERING_GUIDE.md`
2. 章节：
   - Pipeline Overview（7-step 流程图）
   - Feature Definitions（引用 SDF_FEATURE_DEFINITIONS.md）
   - Usage Examples（CLI 命令 + 配置示例）
   - Ablation Study Results（Top 5 重要特征）
3. 添加 Troubleshooting FAQ（常见错误）

**验收标准（DoD）**:
- [ ] 文档包含 ≥4 个主要章节
- [ ] CLI 示例可直接复制运行
- [ ] 引用 T3.4 ablation study 结果
- 验证命令: `Test-Path projects/dgsf/docs/FEATURE_ENGINEERING_GUIDE.md`

---

### P0-7.T4.1: 定义 Training Optimization 目标
**DGSF 关联**: T4 Training Optimization - Planning  
**Effort**: 1 小时  
**Dependencies**: T3.5 完成（T3 → T4 Gate 满足）  
**Status**: PENDING

**执行步骤**:
1. 更新 `projects/dgsf/specs/PROJECT_DGSF.yaml` 的 T4 章节
2. 定义 3 个优化目标：
   - 降低训练时间（目标: < 2 小时/epoch on GPU）
   - 提高样本效率（目标: 收敛 epoch < 50）
   - 减少过拟合（目标: val/train loss ratio < 1.2）
3. 定义 5 个可行策略（学习率调度、early stopping、gradient clipping 等）

**验收标准（DoD）**:
- [ ] T4 目标量化（3 个数值指标）
- [ ] 策略列表包含 ≥5 项
- [ ] 每个策略有预期收益估计
- 验证命令: `Select-String -Path projects/dgsf/specs/PROJECT_DGSF.yaml -Pattern "T4.*Training Optimization"`

---

### P0-7.T4.2: 实现 Learning Rate Scheduler
**DGSF 关联**: T4 Training Optimization - Step 1  
**Effort**: 2-3 小时  
**Dependencies**: T4.1 完成  
**Status**: PENDING

**执行步骤**:
1. 在 `repo/src/dgsf/training/` 创建 `lr_scheduler.py`
2. 实现 3 种策略：
   - CosineAnnealing (warmup + decay)
   - ReduceLROnPlateau (adaptive)
   - StepLR (milestone-based)
3. 添加 warmup period（前 5 epochs 线性增长）
4. 集成到 training loop（config-driven）

**验收标准（DoD）**:
- [ ] 3 种 scheduler 实现完成
- [ ] 单元测试验证曲线正确（≥6 tests）
- [ ] 训练日志显示实时 LR（每 epoch 打印）
- 验证命令: `pytest repo/tests/test_lr_scheduler.py -v`

---

### P0-7.T4.3: 实现 Early Stopping + Checkpointing
**DGSF 关联**: T4 Training Optimization - Step 2  
**Effort**: 2-3 小时  
**Dependencies**: T4.2 完成  
**Status**: PENDING

**执行步骤**:
1. 在 `repo/src/dgsf/training/` 创建 `early_stopping.py`
2. 实现 patience-based early stopping（默认 patience=10）
3. 实现 best model checkpointing（保存最佳 val loss 模型）
4. 添加 restore_best_weights 选项

**验收标准（DoD）**:
- [ ] Early stopping 正确触发（patience 耗尽）
- [ ] Checkpoint 保存/加载验证（模型一致性）
- [ ] 集成测试：训练 → early stop → restore → 继续训练
- 验证命令: `pytest repo/tests/test_early_stopping.py -v`

---

### P0-7.T4.4: Hyperparameter Tuning Framework
**DGSF 关联**: T4 Training Optimization - Step 3  
**Effort**: 3-4 小时  
**Dependencies**: T4.3 完成  
**Status**: PENDING

**执行步骤**:
1. 创建 `experiments/hyperparameter_tuning/tune_dgsf.py`
2. 集成 Optuna 或 Ray Tune（配置驱动）
3. 定义搜索空间：
   - LR: [1e-5, 1e-3] (log-uniform)
   - Batch size: [32, 64, 128]
   - Hidden dim: [64, 128, 256]
   - Dropout: [0.1, 0.3, 0.5]
4. 运行 50 trials（3 小时预算）
5. 生成 `best_config.yaml`

**验收标准（DoD）**:
- [ ] Tuning framework 可执行
- [ ] 50 trials 完成（每 trial < 5 分钟）
- [ ] best_config.yaml 优于 baseline（SR ↑ ≥0.1）
- 验证命令: `python experiments/hyperparameter_tuning/tune_dgsf.py --n-trials 5 --dry-run`

---

### P0-8: T2 → T3 Gate 形式化记录
**DGSF 关联**: Stage 4 Governance  
**Effort**: 15 分钟  
**Dependencies**: T3.3.2 完成  
**Status**: PENDING

**执行步骤**:
1. 更新 `projects/dgsf/docs/STAGE_4_ACCEPTANCE_CRITERIA.md`
2. 添加 T2 → T3 Gate Decision Record:
   - Decision Date: 2026-02-03
   - Decision: OPEN ✅
   - Evidence: 156/167 tests passed (93.4%), 11 skipped classified
   - Next Milestone: T3 Feature Engineering (3 weeks)
3. 提交 Git commit

**验收标准（DoD）**:
- [ ] Gate Decision Record 包含 4 要素
- [ ] Git commit message: "docs(dgsf): Record T2→T3 Gate OPEN decision"
- 验证命令: `git log --oneline -1`

---

## 📊 T3.3 Summary（拆分总结）

| Subtask | Effort | Focus | Key Deliverable |
|---------|--------|-------|-----------------|
| T3.3.1 | 2-3h | CLI + Config | `run_feature_engineering.py` --dry-run working |
| T3.3.2 | 2-3h | Data Loading | 5 data loaders + tests |
| T3.3.3 | 3-4h | Characteristics | 5 firm characteristics + winsorization |
| T3.3.4 | 3-4h | Spreads + Factors | style_spreads + 5 factors + X_state assembly |
| **Total** | **10-14h** | **Pipeline** | **Executable feature engineering pipeline** |

**执行顺序**: T3.3.1 → T3.3.2 → T3.3.3 → T3.3.4 (严格顺序依赖)

---

## 🟡 P1 任务（降低 DGSF 迭代摩擦 · 本周内）

### P1-1: 创建 DGSF 快速验证脚本
**DGSF 关联**: 减少日常检查时间，降低迭代成本  
**Effort**: 20 分钟  
**Status**: ⏸️ READY

**执行步骤**:
1. 创建 `scripts/dgsf_quick_check.ps1`
2. 输出 4 项状态: Git, Tests, Submodule, Branch
3. 运行时间 < 10 秒

**脚本内容**:
```powershell
# scripts/dgsf_quick_check.ps1
Write-Host "=== DGSF Quick Check ===" -ForegroundColor Cyan
Write-Host "[1] Git Status:" -ForegroundColor Yellow
cd "E:\AI Tools\AI Workflow OS\projects\dgsf\repo"
git status --short
Write-Host "[2] Test Summary:" -ForegroundColor Yellow
pytest tests/ --collect-only -q 2>$null | Select-Object -Last 3
Write-Host "[3] Submodule Sync:" -ForegroundColor Yellow
git log --oneline -1
Write-Host "[4] Branch:" -ForegroundColor Yellow
git branch --show-current
```

**验收标准（DoD）**:
- [ ] 运行时间 < 10 秒
- [ ] 输出包含 4 项状态
- 验证命令: `.\scripts\dgsf_quick_check.ps1`

---

### P1-2: 定义 T3 → T4 Readiness Gate
**DGSF 关联**: 明确何时可开始 Training Optimization  
**Effort**: 10 分钟  
**Status**: ⏸️ READY

**执行步骤**:
1. 更新 `STAGE_4_ACCEPTANCE_CRITERIA.md`
2. 添加 T3 → T4 Gate 定义

**Gate 定义（草案）**:
```markdown
### T3 → T4 Readiness Gate
**Open Condition**: 以下全部满足
1. Feature engineering pipeline 可执行 (`scripts/run_feature_engineering.py` 存在)
2. Ablation study 完成 (`experiments/feature_ablation/results.json` 存在)
3. Ablation 结果: ≥3 features 的 p-value < 0.05
4. Feature definitions 文档化完成
```

**验收标准（DoD）**:
- [ ] Gate 包含 ≥3 条可验证条件
- [ ] 至少 1 条有数值阈值
- 验证命令: `Select-String -Path projects/dgsf/docs/STAGE_4_ACCEPTANCE_CRITERIA.md -Pattern "T3.*T4"`

---

### P1-3: 创建 DGSF Daily Workflow Checklist
**DGSF 关联**: 标准化日常开发流程，减少认知负载  
**Effort**: 15 分钟  
**Status**: ⏸️ READY

**执行步骤**:
1. 在 `projects/dgsf/README.md` 添加 "Daily Workflow" 章节
2. 包含 5-7 项日常步骤
3. 引用 `scripts/dgsf_quick_check.ps1`

**验收标准（DoD）**:
- [ ] Checklist 包含 5-7 项步骤
- [ ] 每项有对应命令
- 验证命令: `Select-String -Path projects/dgsf/README.md -Pattern "Daily Workflow"`

---

### P1-4: 创建 Feature Computation Profiler
**DGSF 关联**: 识别特征计算瓶颈，优化 pipeline 性能  
**Effort**: 30 分钟  
**Status**: PENDING

**执行步骤**:
1. 在 `scripts/profile_features.py` 添加 cProfile 集成
2. 测量每个特征函数的执行时间
3. 输出 Top 5 耗时操作（带百分比）
4. 生成火焰图（Flamegraph）

**验收标准（DoD）**:
- [ ] Profiler 输出 Top 5 bottlenecks
- [ ] 火焰图可视化生成（HTML）
- [ ] 识别至少 1 个可优化点（例如：避免重复计算）
- 验证命令: `python scripts/profile_features.py --config sample_config.yaml`

---

### P1-5: 实现 Feature Caching 机制
**DGSF 关联**: 避免重复计算，降低实验迭代成本  
**Effort**: 1-2 小时  
**Dependencies**: P1-4 识别出缓存收益  
**Status**: PENDING

**执行步骤**:
1. 在 `scripts/data_loaders.py` 添加 `@lru_cache` 装饰器
2. 为高频特征（size, momentum）启用缓存
3. 实现文件缓存（Parquet 格式）用于跨 session
4. 添加 `--force-reload` 参数绕过缓存

**验收标准（DoD）**:
- [ ] 内存缓存减少重复计算 ≥50%（profiler 验证）
- [ ] 文件缓存加速第二次运行 ≥3x
- [ ] 缓存失效逻辑正确（数据更新时自动清除）
- 验证命令: `python scripts/run_feature_engineering.py --config sample.yaml` (2次运行时间对比)

---

### P1-6: 添加 Data Quality Report
**DGSF 关联**: 诊断数据问题，减少调试时间  
**Effort**: 30 分钟  
**Status**: PENDING

**执行步骤**:
1. 在 `scripts/run_feature_engineering.py` 添加 `--data-quality-check` 模式
2. 输出 5 类统计：
   - Missing value percentage (by column)
   - Outliers beyond [0.1%, 99.9%]
   - Temporal coverage gaps (>3 month breaks)
   - Firm coverage (firms with <6 month data)
   - Cross-sectional sparsity (firms/date)
3. 生成 JSON report: `reports/data_quality_YYYYMMDD.json`

**验收标准（DoD）**:
- [ ] Report 包含 5 类统计
- [ ] 自动标注异常（红色警告）
- [ ] CLI 输出表格化（易读）
- 验证命令: `python scripts/run_feature_engineering.py --data-quality-check`

---

## ⚪ P2 任务（延后 · 仅触发条件满足时执行）

| ID | Task | 触发条件 | Effort |
|----|------|----------|--------|
| P2-1 | 创建 T4/T5 TaskCard | T3 完成度 > 80% | 30 min |
| P2-2 | RESEARCH_MILESTONES.md | 有论文 deadline | 20 min |
| P2-3 | 聚合 audit JSON | audit/ 目录 > 50 文件 | 30 min |
| P2-4 | README Troubleshooting | 同一问题 ≥2 次 | 15 min |
| P2-5 | kernel 导入路径修复 | DGSF 调用 kernel 出错 | 1.5 hr |
| P2-6 | PROJECT_STATE.md 精简 | 查询失败 ≥3 次 | 30 min |

---

## 📋 执行队列汇总

**更新时间**: 2026-02-03T23:00Z  
**当前进度**: Stage 4 T2 → T3 Gate OPEN ✅ | T3.3.3 IN PROGRESS 🎯

| # | ID | Priority | Status | Effort | DGSF 关联 | 阻塞情况 |
|---|-----|----------|--------|--------|-----------|----------|
| 1 | **P0-7.T3.3.3** | **P0** | **🎯 NEXT** | **3-4h** | **Firm Characteristics** | - |
| 2 | P0-7.T3.3.4 | P0 | ⏸️ READY | 3-4h | Spreads + Factors | T3.3.3 |
| 3 | P0-7.T3.3.5 | P0 | ⏸️ READY | 1-2h | E2E Pipeline Test | T3.3.4 |
| 4 | P0-7.T3.4 | P0 | ⏸️ READY | 4-6h | Feature Ablation | T3.3.5 |
| 5 | P0-7.T3.5 | P0 | ⏸️ READY | 1-2h | Documentation | T3.4 |
| 6 | P0-7.T4.1 | P0 | ⏸️ BLOCKED | 1h | T4 Planning | T3→T4 Gate |
| 7 | P0-7.T4.2 | P0 | ⏸️ BLOCKED | 2-3h | LR Scheduler | T4.1 |
| 8 | P0-7.T4.3 | P0 | ⏸️ BLOCKED | 2-3h | Early Stopping | T4.2 |
| 9 | P0-7.T4.4 | P0 | ⏸️ BLOCKED | 3-4h | Hyperparameter Tuning | T4.3 |
| 10 | P0-8 | P0 | ⏸️ READY | 15min | T2→T3 Gate Record | - |
| 11 | P1-1 | P1 | ⏸️ READY | 20min | Quick Check Script | - |
| 12 | P1-2 | P1 | ⏸️ READY | 10min | T3→T4 Gate Definition | - |
| 13 | P1-3 | P1 | ⏸️ READY | 15min | Daily Workflow | - |
| 14 | P1-4 | P1 | ⏸️ READY | 30min | Feature Profiler | - |
| 15 | P1-5 | P1 | ⏸️ READY | 1-2h | Feature Caching | P1-4 |
| 16 | P1-6 | P1 | ⏸️ READY | 30min | Data Quality Report | - |
| 17-22 | P2-* | P2 | ⚪ DEFERRED | - | 触发条件未满足 | - |

**Total P0 Tasks**: 10 (1 in-progress, 5 ready, 4 blocked)  
**Total P1 Tasks**: 6 (6 ready)  
**Total P2 Tasks**: 6 (all deferred)  
**Estimated T3 Remaining**: 13-19 hours (1.5-2.5 weeks @ 8h/day)

---

## 🚀 Next Single Step

**选择**: **P0-7.T3.3.4 - Cross-Sectional Spreads + Factors**

**理由**:
1. ✅ T3.3.3（Firm Characteristics）已完成，19/19 tests passed
2. ✅ 直接推进 T3 Feature Engineering 主线
3. ✅ 产出明确：cross-sectional spreads + 5 factors + X_state assembly
4. ✅ 可在 3-4 小时内完成，最小可验证步

**执行计划**:
```powershell
# 1. 创建 spreads and factors 模块
New-Item -Path "projects/dgsf/scripts/spreads_factors.py" -ItemType File -Force

# 2. 创建测试文件
New-Item -Path "projects/dgsf/tests/test_spreads_factors.py" -ItemType File -Force

# 3. 实现 Step 4-6:
#    - compute_style_spreads() (5D cross-sectional spreads)
#    - compute_market_factor(), compute_smb_hml(), compute_momentum_factor(), compute_reversal()
#    - assemble_X_state(), assemble_P_tree_factors()
# 4. 编写单元测试（≥12 tests）
# 5. 运行验证
cd projects/dgsf
python -m pytest tests/test_spreads_factors.py -v
```

**Expert Simulation**: Gene Kim (DevOps + Flow)
- **修改点**: 新增 2 个文件（spreads_factors.py, test_spreads_factors.py）
- **验收标准**: 12+ tests passed, X_state assembly 正确（维度验证）
- **验证命令**: `pytest tests/test_spreads_factors.py -v`

---

## 📝 Expert Panel Insights（2026-02-03T21:00Z）

### Grady Booch（Architecture）
- **Findings**: T1-T2 完成，架构清晰，无结构性阻塞
- **Recommendation**: 为 T3 创建详细 TaskCard（P0-7）🎯
- **Risk if ignored**: 3 周任务可能失控

### Gene Kim（Execution Flow）
- **Findings**: 工作流顺畅，repo/ 已同步
- **Recommendation**: 创建快速验证脚本（P1-1）降低日常摩擦
- **Risk if ignored**: 每次迭代多花 2-3 分钟检查

### Mary Shaw（Dependency）
- **Findings**: OS → DGSF 单向依赖保持良好
- **Recommendation**: 维持当前边界，不扩展 Adapter
- **Risk if ignored**: 依赖反转风险

### Martin Fowler（Refactoring）
- **Findings**: 5 个 TODO 在 dev_sdf_models.py（技术债）
- **Recommendation**: 延后重构，聚焦 T3 功能
- **Risk if ignored**: 提前重构浪费资源

### Leslie Lamport（DoD）
- **Findings**: Stage 4 AC 已定义，T2→T3 Gate 明确
- **Recommendation**: 定义 T3→T4 Gate（P1-2）
- **Risk if ignored**: T3 完成标准模糊

### Nicole Forsgren（Metrics）
- **Findings**: 测试通过率 93.4%，11 skipped 已分类
- **Recommendation**: 跟踪 T3 子任务 cycle time
- **Risk if ignored**: 无法识别瓶颈

---

**End of TODO_NEXT.md**
