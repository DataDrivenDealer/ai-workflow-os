# AI Workflow OS - Q1 2026 执行计划书

**文档ID**: EXECUTION_PLAN_2026_Q1  
**创建日期**: 2026-02-02  
**制定者**: 赵执行 (项目管理大师 - Scrum方法论专家)  
**状态**: ACTIVE - 待执行  
**目标周期**: 2026-02-03 至 2026-03-02 (4周冲刺)  
**关联分析**: 基于2026-02-02专家团队战略分析报告

---

## 📋 执行总览

### 当前项目状态
- **项目健康度**: 🟡 68/100 (架构优秀但执行滞后)
- **关键风险**: M1里程碑(2月17日)和M2里程碑(2月23日)均为高风险
- **核心问题**: 11个已识别问题，其中3个P0级阻塞性问题

### 执行目标
1. **Week 1**: 解除P0阻塞项，让系统可以运行
2. **Week 2**: 完成SDF Layer核心组件
3. **Week 3**: 实现EA Layer，达成M2里程碑
4. **Week 4**: 验证可复现性，完成baseline复现

### 成功标准
- ✅ SDF模型可以训练并收敛
- ✅ EA优化器可以生成组合权重
- ✅ 端到端backtest可以运行
- ✅ Baseline指标可复现(±5%误差内)
- ✅ CI/CD流水线正常运行

---

## 🎯 第一周执行计划 (2026-02-03 至 02-09)

### 主题: 突破关键阻塞 (UNBLOCK CRITICAL PATH)

---

### ☑️ P0-1: SDF Model整合 【阻塞性】

**任务ID**: `SDF_DEV_001.2`  
**负责人**: 李架构 (首席架构师)  
**预计工时**: 12小时  
**依赖项**: ✅ StateEngine已完成  
**优先级**: 🔴 P0 - 最高优先级

#### 执行步骤

- [ ] **Step 1.1**: 审查Legacy `model.py` 代码 (1h)
  - 位置: `projects/dgsf/repo/legacy/model.py` (如果存在)
  - 理解GenerativeSDF架构和参数
  - 识别需要保留和重构的部分

- [ ] **Step 1.2**: 创建生产级SDF模型文件 (3h)
  - 创建文件: `projects/dgsf/repo/src/dgsf/sdf/model.py`
  - 实现`GenerativeSDF`类:
    ```python
    class GenerativeSDF(nn.Module):
        """
        Generative SDF Model v3.1
        Architecture: log m_t = c · tanh(h_θ(Info_t))
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64, 
                     num_layers: int = 2, c: float = 4.0)
        def forward(self, info: torch.Tensor) -> torch.Tensor
        def compute_sdf(self, info: torch.Tensor) -> torch.Tensor
    ```
  - 确保boundedness约束 (c=4.0 frozen)
  - 添加normalization layer

- [ ] **Step 1.3**: 编写单元测试 (2h)
  - 创建文件: `projects/dgsf/repo/tests/sdf/test_sdf_model.py`
  - 测试用例:
    - `test_model_initialization` - 模型参数正确初始化
    - `test_forward_pass` - forward pass维度正确
    - `test_sdf_boundedness` - SDF输出在合理范围内
    - `test_normalization` - E[m_t] ≈ 1
    - `test_gradient_flow` - 梯度可以反向传播
  - 运行: `pytest tests/sdf/test_sdf_model.py -v`

- [ ] **Step 1.4**: 与StateEngine集成测试 (2h)
  - 编写集成测试: `test_state_engine_to_model_integration`
  - 测试流程: StateEngine输出 → SDF Model输入 → SDF输出
  - 验证维度匹配和数据流正确

- [ ] **Step 1.5**: 添加配置支持 (2h)
  - 在`projects/dgsf/repo/configs/model.yaml`中添加模型配置
  - 实现配置加载逻辑
  - 支持不同的hidden_dim/num_layers配置

- [ ] **Step 1.6**: 文档更新 (2h)
  - 更新`tasks/SDF_DEV_001.md` - 标记1.2为完成
  - 创建`projects/dgsf/repo/docs/sdf_model_api.md`
  - 添加使用示例和API文档

#### 验收标准
- [x] `model.py` 文件创建，包含完整的GenerativeSDF类
- [x] 至少5个单元测试，全部通过
- [x] forward pass可以执行并返回正确维度的SDF
- [x] 与StateEngine集成测试通过
- [x] 代码通过pyright类型检查
- [x] 文档已更新

#### 交付物
```
projects/dgsf/repo/src/dgsf/sdf/model.py
projects/dgsf/repo/tests/sdf/test_sdf_model.py
projects/dgsf/repo/configs/model.yaml
projects/dgsf/repo/docs/sdf_model_api.md
```

---

### ☑️ P0-2: Moment Estimation实现 【阻塞性】

**任务ID**: `SDF_DEV_001.3`  
**负责人**: 张数据 (数据科学家)  
**预计工时**: 10小时  
**依赖项**: ⏳ 需要SDF Model完成  
**优先级**: 🔴 P0

#### 执行步骤

- [ ] **Step 2.1**: 研究Moment Matching理论 (2h)
  - 阅读SDF Layer Specification v3.1 §3
  - 理解Pricing Kernel的moment条件
  - 确认需要估计的moment: E[m_t], E[m_t · r_t], etc.

- [ ] **Step 2.2**: 实现Robust Moment Estimator (4h)
  - 创建文件: `projects/dgsf/repo/src/dgsf/sdf/moments.py`
  - 实现类:
    ```python
    class MomentEstimator:
        def __init__(self, window_size: int = 252, 
                     robust_method: str = 'huber')
        def estimate_first_moment(self, sdf: np.ndarray) -> float
        def estimate_cross_moment(self, sdf: np.ndarray, 
                                  returns: np.ndarray) -> np.ndarray
        def compute_pricing_errors(self, sdf, returns, risk_free) -> np.ndarray
    ```
  - 实现Huber损失robust estimation
  - 处理missing data和outliers

- [ ] **Step 2.3**: 编写单元测试 (2h)
  - 创建文件: `projects/dgsf/repo/tests/sdf/test_moments.py`
  - 测试用例:
    - `test_first_moment_estimation` - 均值估计正确
    - `test_cross_moment_estimation` - 协方差估计正确
    - `test_robust_to_outliers` - 对异常值鲁棒
    - `test_pricing_error_computation` - PE计算正确
  - Mock数据生成器

- [ ] **Step 2.4**: 性能优化 (1h)
  - 使用NumPy向量化操作
  - 避免循环，使用矩阵运算
  - Profiling: 确保1000股×252天数据<1秒

- [ ] **Step 2.5**: 文档与示例 (1h)
  - 创建notebook: `projects/dgsf/repo/notebooks/moment_estimation_demo.ipynb`
  - 演示moment estimation的使用
  - 可视化moment随时间的变化

#### 验收标准
- [x] `moments.py` 实现完整
- [x] 单元测试覆盖率>85%
- [x] 性能满足要求 (<1秒处理1000股×252天)
- [x] 可以正确计算Pricing Error
- [x] Demo notebook可以运行

#### 交付物
```
projects/dgsf/repo/src/dgsf/sdf/moments.py
projects/dgsf/repo/tests/sdf/test_moments.py
projects/dgsf/repo/notebooks/moment_estimation_demo.ipynb
```

---

### ☑️ P0-3: 数据管道启动 【阻塞性】

**任务ID**: `DATA_EXPANSION_001.1`  
**负责人**: 王数据 (数据工程师)  
**预计工时**: 16小时  
**依赖项**: 无  
**优先级**: 🔴 P0

#### 执行步骤

- [ ] **Step 3.1**: 定义股票池逻辑 (2h)
  - 创建文件: `projects/dgsf/adapter/universe_selector.py`
  - 实现函数:
    ```python
    def get_universe(date: str, 
                     exclude_st: bool = True,
                     min_listing_days: int = 90) -> List[str]:
        """获取指定日期的有效A股股票池"""
        pass
    ```
  - 规则: 剔除ST、新股(<90天)、停牌(>20天)

- [ ] **Step 3.2**: 先获取100股样本数据 (4h)
  - 选择中证100成分股作为初始测试集
  - 数据源: 本地数据/在线API (需确认)
  - 时间范围: 2023-01-01 至 2024-12-31 (2年)
  - 字段: OHLCV + 市值 + 换手率
  - 保存格式: Parquet (高效压缩)
  - 存储位置: `projects/dgsf/data/raw/csi100_sample/`

- [ ] **Step 3.3**: 数据质量初步验证 (3h)
  - 创建脚本: `projects/dgsf/scripts/validate_data_quality.py`
  - 检查项:
    - Missing data比例 <5%
    - 价格异常值检测 (日涨跌幅>±30%)
    - 成交量为0的天数统计
    - 时间序列连续性
  - 生成质量报告: `data/reports/csi100_quality_report.md`

- [ ] **Step 3.4**: 创建数据加载器 (4h)
  - 更新文件: `projects/dgsf/adapter/data_loader.py`
  - 实现类:
    ```python
    class DGSFDataLoader:
        def __init__(self, data_dir: str, universe: str = 'csi100')
        def load_prices(self, start_date, end_date) -> pd.DataFrame
        def load_returns(self, start_date, end_date) -> pd.DataFrame
        def load_features(self, feature_list: List[str]) -> pd.DataFrame
    ```
  - 支持lazy loading
  - 缓存机制

- [ ] **Step 3.5**: 集成测试 (2h)
  - 编写测试: `projects/dgsf/repo/tests/data/test_data_loader.py`
  - 测试可以成功加载100股×2年数据
  - 验证返回的DataFrame格式正确

- [ ] **Step 3.6**: 文档与下一步规划 (1h)
  - 更新`tasks/DATA_EXPANSION_001.md`
  - 记录数据获取流程
  - 规划800股→5000股扩展路径

#### 验收标准
- [x] 成功获取100股×2年日频数据
- [x] 数据质量验证通过 (missing<5%)
- [x] DataLoader可以正确加载数据
- [x] 单元测试通过
- [x] 质量报告已生成

#### 交付物
```
projects/dgsf/adapter/universe_selector.py
projects/dgsf/data/raw/csi100_sample/*.parquet
projects/dgsf/scripts/validate_data_quality.py
projects/dgsf/adapter/data_loader.py (updated)
projects/dgsf/data/reports/csi100_quality_report.md
projects/dgsf/repo/tests/data/test_data_loader.py
```

---

### ☑️ P0-4: 建立基础CI/CD流水线 【基础设施】

**任务ID**: `INFRA_CI_001`  
**负责人**: 刘运维 (DevOps工程师)  
**预计工时**: 8小时  
**依赖项**: 无  
**优先级**: 🔴 P0

#### 执行步骤

- [ ] **Step 4.1**: 创建GitHub Actions配置 (2h)
  - 创建文件: `.github/workflows/ci.yml`
  - 配置触发条件: push, pull_request
  - 定义job: test, lint, type-check
  - Python版本矩阵: 3.10, 3.11

- [ ] **Step 4.2**: 配置测试流水线 (2h)
  - 安装依赖: `pip install -r requirements.txt`
  - 运行pytest: `pytest kernel/tests/ -v --cov`
  - 上传coverage报告
  - 失败时阻止merge

- [ ] **Step 4.3**: 配置代码质量检查 (2h)
  - 集成black格式检查: `black --check kernel/ projects/`
  - 集成isort检查: `isort --check-only kernel/ projects/`
  - 集成pyright类型检查: `pyright kernel/`
  - 可选: 集成flake8/pylint

- [ ] **Step 4.4**: 配置gate检查 (1h)
  - 运行scripts/gate_check.py
  - 运行scripts/check_lookahead.py (如果适用)
  - 失败时提供清晰的错误信息

- [ ] **Step 4.5**: 本地pre-commit hooks集成 (1h)
  - 确保hooks/pre-commit可以自动运行
  - 测试在commit前自动执行检查
  - 更新README安装说明

#### 验收标准
- [x] CI workflow文件创建并正常运行
- [x] 所有测试在CI中通过
- [x] 代码质量检查正常运行
- [x] PR中显示CI状态badge
- [x] 本地hooks可以正常工作

#### 交付物
```
.github/workflows/ci.yml
.github/workflows/gate-check.yml (optional)
README.md (updated with CI badge)
```

---

### ☑️ P0-5: 完善Python依赖声明 【基础设施】

**任务ID**: `ENV_DEPS_001`  
**负责人**: 王技术 (技术领导)  
**预计工时**: 4小时  
**依赖项**: 无  
**优先级**: 🟡 P1 (但建议本周完成)

#### 执行步骤

- [ ] **Step 5.1**: 审计当前代码依赖 (1h)
  - 扫描所有Python文件的import语句
  - 识别缺失的第三方库
  - 记录到临时清单

- [ ] **Step 5.2**: 更新requirements.txt (2h)
  - 添加核心依赖:
    ```
    # Deep Learning
    torch>=2.1.0
    numpy>=1.24.0
    pandas>=2.0.0
    
    # Scientific Computing
    scipy>=1.11.0
    scikit-learn>=1.3.0
    
    # Data Processing
    pyarrow>=14.0.0  # for Parquet
    
    # Visualization (optional)
    matplotlib>=3.7.0
    seaborn>=0.12.0
    
    # Experiment Tracking (optional)
    tensorboard>=2.15.0
    ```
  - 固定版本号，避免依赖冲突

- [ ] **Step 5.3**: 创建开发依赖文件 (0.5h)
  - 创建`requirements-dev.txt`:
    ```
    -r requirements.txt
    jupyter>=1.0.0
    ipykernel>=6.25.0
    notebook>=7.0.0
    ```

- [ ] **Step 5.4**: 测试依赖安装 (0.5h)
  - 在干净的虚拟环境中测试:
    ```powershell
    python -m venv test_env
    test_env\Scripts\activate
    pip install -r requirements.txt
    pytest kernel/tests/ --collect-only
    ```
  - 确保所有import正常

#### 验收标准
- [x] requirements.txt包含所有必需依赖
- [x] 版本号固定，避免冲突
- [x] 在新环境中可以成功安装
- [x] 所有import语句正常工作

#### 交付物
```
requirements.txt (updated)
requirements-dev.txt (new)
```

---

## 📊 第一周进度跟踪表

| 任务 | 负责人 | 状态 | 进度 | 预计完成 | 实际完成 | 阻塞项 |
|------|--------|------|------|----------|----------|--------|
| P0-1: SDF Model | 李架构 | ⏳ PENDING | 0% | 02-05 | - | - |
| P0-2: Moments | 张数据 | ⏳ PENDING | 0% | 02-06 | - | 依赖P0-1 |
| P0-3: Data Pipeline | 王数据 | ⏳ PENDING | 0% | 02-07 | - | - |
| P0-4: CI/CD | 刘运维 | ⏳ PENDING | 0% | 02-05 | - | - |
| P0-5: Dependencies | 王技术 | ⏳ PENDING | 0% | 02-04 | - | - |

**更新频率**: 每日更新此表格

---

## 🎯 第二周执行计划 (2026-02-10 至 02-16)

### 主题: 完成SDF Layer核心组件

---

### ☑️ P1-1: Trainer实现

**任务ID**: `SDF_DEV_001.4`  
**负责人**: 李架构  
**预计工时**: 14小时  
**依赖项**: ⏳ P0-2 Moment Estimation

#### 执行步骤

- [ ] **Step 6.1**: 设计训练循环架构 (2h)
  - 研究PyTorch Lightning或原生训练循环
  - 定义训练配置schema
  - 设计checkpoint机制

- [ ] **Step 6.2**: 实现基础Trainer (6h)
  - 创建文件: `projects/dgsf/repo/src/dgsf/sdf/trainer.py`
  - 实现类:
    ```python
    class SDFTrainer:
        def __init__(self, model, config, device='cuda')
        def train_epoch(self, dataloader) -> Dict[str, float]
        def validate(self, dataloader) -> Dict[str, float]
        def fit(self, train_loader, val_loader, epochs: int)
        def save_checkpoint(self, path: str)
        def load_checkpoint(self, path: str)
    ```

- [ ] **Step 6.3**: 集成Loss函数 (3h)
  - 实现Pricing Error loss
  - 实现normalization constraint loss
  - 实现temporal smoothness loss (optional)
  - 总损失: weighted sum

- [ ] **Step 6.4**: 添加训练监控 (2h)
  - TensorBoard logging
  - 记录: loss, PE, normalization error
  - Early stopping机制

- [ ] **Step 6.5**: 单元测试与集成测试 (1h)
  - 测试训练循环可以运行
  - 测试checkpoint保存/加载
  - 测试在toy数据上可以收敛

#### 验收标准
- [x] Trainer可以完整运行train-validate循环
- [x] 可以保存和加载checkpoint
- [x] TensorBoard可以显示训练曲线
- [x] 在toy数据上验证收敛

#### 交付物
```
projects/dgsf/repo/src/dgsf/sdf/trainer.py
projects/dgsf/repo/tests/sdf/test_trainer.py
projects/dgsf/repo/configs/training.yaml
```

---

### ☑️ P1-2: Oracle API实现

**任务ID**: `SDF_DEV_001.5`  
**负责人**: 李架构  
**预计工时**: 10小时  
**依赖项**: ⏳ P1-1 Trainer

#### 执行步骤

- [ ] **Step 7.1**: 设计Oracle接口 (2h)
  - 定义API契约
  - 输入: portfolio weights [N]
  - 输出: Pricing Error (scalar)
  - 定义缓存策略

- [ ] **Step 7.2**: 实现PricingErrorOracle (4h)
  - 创建文件: `projects/dgsf/repo/src/dgsf/sdf/oracle.py`
  - 实现类:
    ```python
    class PricingErrorOracle:
        def __init__(self, sdf_model, state_engine, data)
        def compute_pe(self, weights: np.ndarray) -> float
        def batch_compute_pe(self, weights_matrix: np.ndarray) -> np.ndarray
    ```
  - 优化性能: 使用batch inference

- [ ] **Step 7.3**: 单元测试 (2h)
  - 测试PE计算的正确性
  - 测试batch computation
  - 测试边界情况 (零权重、等权重等)

- [ ] **Step 7.4**: 性能优化 (1h)
  - GPU加速 (如果适用)
  - 结果缓存
  - Benchmark: 1000次评估<10秒

- [ ] **Step 7.5**: 文档与示例 (1h)
  - API文档
  - 使用示例notebook

#### 验收标准
- [x] Oracle可以正确计算Pricing Error
- [x] 性能满足要求
- [x] 单元测试通过
- [x] 与EA Layer的接口契约明确

#### 交付物
```
projects/dgsf/repo/src/dgsf/sdf/oracle.py
projects/dgsf/repo/tests/sdf/test_oracle.py
projects/dgsf/repo/notebooks/oracle_demo.ipynb
```

---

### ☑️ P1-3: 数据清洗与质量验证

**任务ID**: `DATA_EXPANSION_001.2`  
**负责人**: 王数据  
**预计工时**: 20小时  
**依赖项**: ✅ P0-3 完成

#### 执行步骤

- [ ] **Step 8.1**: 扩展到中证800 (4h)
  - 获取中证800成分股列表
  - 下载800股×2年数据
  - 存储到`projects/dgsf/data/raw/csi800/`

- [ ] **Step 8.2**: 实现数据清洗管道 (8h)
  - 创建文件: `projects/dgsf/adapter/data_cleaner.py`
  - 功能:
    - 处理missing data (前向填充/线性插值)
    - 剔除异常值 (涨跌幅>±30%)
    - 处理停牌日 (标记为NaN)
    - 公司行动调整 (股票分割、分红)
  - 生成cleaned数据: `data/processed/csi800_cleaned/`

- [ ] **Step 8.3**: 严格质量验证 (4h)
  - 更新`validate_data_quality.py`
  - 检查项扩展:
    - 前视偏差检测 (look-ahead bias)
    - 幸存者偏差检测 (survivorship bias)
    - 时间对齐验证
    - 极端值统计
  - 生成详细报告: `csi800_quality_report_detailed.md`

- [ ] **Step 8.4**: 因果性验证 (3h)
  - 确保所有特征在时刻t只使用t-1及之前的信息
  - 编写测试: `test_causality_constraint.py`
  - 验证数据加载器不会泄露未来信息

- [ ] **Step 8.5**: 文档与数据字典 (1h)
  - 创建数据字典: `data/DATA_DICTIONARY.md`
  - 记录每个字段的含义、单位、更新频率
  - 记录数据处理的所有变换

#### 验收标准
- [x] 800股数据清洗完成
- [x] 质量验证通过 (missing<3%, 无明显异常)
- [x] 因果性约束验证通过
- [x] 数据字典完整

#### 交付物
```
projects/dgsf/data/raw/csi800/*.parquet
projects/dgsf/data/processed/csi800_cleaned/*.parquet
projects/dgsf/adapter/data_cleaner.py
projects/dgsf/data/reports/csi800_quality_report_detailed.md
projects/dgsf/data/DATA_DICTIONARY.md
projects/dgsf/repo/tests/data/test_causality_constraint.py
```

---

### ☑️ P1-4: SDF Layer集成测试

**任务ID**: `SDF_DEV_001.6`  
**负责人**: 周治理  
**预计工时**: 12小时  
**依赖项**: ⏳ 全部SDF子任务

#### 执行步骤

- [ ] **Step 9.1**: 设计E2E测试场景 (2h)
  - 定义测试流程: Data → StateEngine → Model → Training → Oracle
  - 使用小规模数据 (10股×100天)

- [ ] **Step 9.2**: 实现E2E集成测试 (6h)
  - 创建文件: `projects/dgsf/repo/tests/test_sdf_e2e.py`
  - 测试场景:
    - `test_full_training_pipeline` - 完整训练流程
    - `test_oracle_from_trained_model` - 训练后的Oracle可用
    - `test_checkpoint_resume` - checkpoint恢复训练
    - `test_reproducibility` - 固定随机种子的可复现性

- [ ] **Step 9.3**: 性能benchmarking (2h)
  - 记录训练时间 (100股×252天)
  - 记录内存占用
  - 记录Oracle评估速度
  - 生成benchmark报告

- [ ] **Step 9.4**: 覆盖率报告 (1h)
  - 运行: `pytest tests/sdf/ --cov=src/dgsf/sdf --cov-report=html`
  - 确保覆盖率>80%
  - 识别未覆盖的关键路径

- [ ] **Step 9.5**: 更新文档 (1h)
  - 更新`tasks/SDF_DEV_001.md` - 标记所有子任务完成
  - 生成SDF Layer完成报告
  - 提交到ops/reports/

#### 验收标准
- [x] E2E测试全部通过
- [x] 测试覆盖率>80%
- [x] 性能benchmark符合预期
- [x] 可复现性验证通过

#### 交付物
```
projects/dgsf/repo/tests/test_sdf_e2e.py
ops/reports/SDF_LAYER_COMPLETION_REPORT.md
coverage_report.html
```

---

## 📊 第二周进度跟踪表

| 任务 | 负责人 | 状态 | 进度 | 预计完成 | 实际完成 | 阻塞项 |
|------|--------|------|------|----------|----------|--------|
| P1-1: Trainer | 李架构 | ⏳ PENDING | 0% | 02-13 | - | 依赖Week1 |
| P1-2: Oracle | 李架构 | ⏳ PENDING | 0% | 02-15 | - | 依赖P1-1 |
| P1-3: Data Cleaning | 王数据 | ⏳ PENDING | 0% | 02-15 | - | 依赖Week1 |
| P1-4: Integration Test | 周治理 | ⏳ PENDING | 0% | 02-16 | - | 依赖全部 |

---

## 🎯 第三周执行计划 (2026-02-17 至 02-23)

### 主题: EA Layer冲刺 & M2里程碑达成

---

### ☑️ P1-5: EA优化器核心实现

**任务ID**: `EA_DEV_001.1`  
**负责人**: 张数据  
**预计工时**: 16小时  
**依赖项**: ✅ P1-2 Oracle API

#### 执行步骤

- [ ] **Step 10.1**: 研究EA算法 (2h)
  - 阅读EA Layer Specification v3.1
  - 选择算法: CMA-ES / Genetic Algorithm / PSO
  - 确定超参数范围

- [ ] **Step 10.2**: 实现EA Optimizer (8h)
  - 创建文件: `projects/dgsf/repo/src/dgsf/ea/optimizer.py`
  - 实现类:
    ```python
    class EAOptimizer:
        def __init__(self, oracle: PricingErrorOracle, 
                     population_size: int = 50, 
                     max_generations: int = 100)
        def optimize(self, initial_weights: Optional[np.ndarray] = None,
                    constraints: Dict = None) -> OptimizationResult
    ```
  - 约束: 权重和为1, 多空限制, 杠杆限制

- [ ] **Step 10.3**: 单元测试 (3h)
  - 测试收敛性 (在convex问题上)
  - 测试约束满足
  - 测试不同初始化的鲁棒性

- [ ] **Step 10.4**: 与SDF Oracle集成测试 (2h)
  - 使用真实Oracle进行优化
  - 验证可以找到更优组合
  - 记录优化轨迹

- [ ] **Step 10.5**: 文档与可视化 (1h)
  - 创建notebook展示优化过程
  - 可视化: PE随迭代的变化, 权重分布

#### 验收标准
- [x] EA Optimizer可以成功优化权重
- [x] 约束条件满足
- [x] 在toy问题上验证收敛
- [x] 与Oracle集成无问题

#### 交付物
```
projects/dgsf/repo/src/dgsf/ea/optimizer.py
projects/dgsf/repo/tests/ea/test_optimizer.py
projects/dgsf/repo/notebooks/ea_optimization_demo.ipynb
```

---

### ☑️ P1-6: EA Layer完整实现

**任务ID**: `EA_DEV_001.2-001.4`  
**负责人**: 张数据  
**预计工时**: 24小时 (分3个子任务)  
**依赖项**: ⏳ P1-5

#### 执行步骤 (简化版)

- [ ] **Step 11.1**: 实现Portfolio构造器 (8h)
  - `portfolio.py`: 从EA权重构造最终组合
  - 风险控制: 单股限制, 行业中性约束

- [ ] **Step 11.2**: 实现回测引擎 (10h)
  - `backtester.py`: 向量化回测
  - 计算: Sharpe, Max DD, Turnover
  - OOS验证: rolling window

- [ ] **Step 11.3**: EA E2E测试 (6h)
  - 完整流程: SDF训练 → EA优化 → 组合构造 → 回测
  - 验证在历史数据上可以生成正收益
  - 性能benchmark

#### 验收标准
- [x] 完整的EA Layer实现
- [x] 回测引擎正常工作
- [x] E2E测试通过
- [x] 性能指标计算正确

---

### ☑️ P1-7: 数据扩展到5000股

**任务ID**: `DATA_EXPANSION_001.3`  
**负责人**: 王数据  
**预计工时**: 16小时  
**依赖项**: ✅ P1-3

#### 执行步骤 (简化版)

- [ ] **Step 12.1**: 获取全量A股数据 (6h)
  - ~5000股×10年日频数据
  - 分批下载, 避免API限流

- [ ] **Step 12.2**: 应用清洗管道 (6h)
  - 复用data_cleaner.py
  - 并行处理提升速度

- [ ] **Step 12.3**: 质量验证与因果性检查 (4h)
  - 完整验证流程
  - 生成最终质量报告

#### 验收标准
- [x] 5000股数据获取完成
- [x] 质量验证通过
- [x] 数据加载器支持

---

## 📊 第三周进度跟踪表

| 任务 | 负责人 | 状态 | 进度 | 预计完成 | 实际完成 | 阻塞项 |
|------|--------|------|------|----------|----------|--------|
| P1-5: EA Optimizer | 张数据 | ⏳ PENDING | 0% | 02-19 | - | 依赖Week2 |
| P1-6: EA Full Impl | 张数据 | ⏳ PENDING | 0% | 02-22 | - | 依赖P1-5 |
| P1-7: Data 5000股 | 王数据 | ⏳ PENDING | 0% | 02-21 | - | 依赖Week2 |

**关键里程碑**: 🎯 **M2 (EA Layer Complete)** - 预计 2026-02-23

---

## 🎯 第四周执行计划 (2026-02-24 至 03-02)

### 主题: 验证与优化

---

### ☑️ P2-1: Baseline可复现性验证

**任务ID**: `BASELINE_REPRO_001`  
**负责人**: 张数据 + 周治理  
**预计工时**: 24小时  
**依赖项**: ✅ 全部前序任务

#### 执行步骤

- [ ] **Step 13.1**: 设置可复现环境 (2h)
  - 固定所有随机种子 (PyTorch, NumPy, random)
  - 记录硬件环境 (CPU/GPU型号)
  - 记录软件版本 (Python, PyTorch, CUDA)

- [ ] **Step 13.2**: 运行完整训练流程 (8h)
  - 训练SDF模型 (800股×5年)
  - EA优化
  - 生成最终组合
  - 记录所有中间结果

- [ ] **Step 13.3**: OOS验证 (6h)
  - Train: 2015-2020
  - Validation: 2021-2022
  - Test: 2023-2024
  - 计算OOS性能指标

- [ ] **Step 13.4**: 可复现性测试 (4h)
  - 重复运行3次
  - 验证结果在±5%误差内一致
  - 记录任何偏差来源

- [ ] **Step 13.5**: 创建Reproducibility Package (4h)
  - 打包: 代码, 配置, 数据快照, 结果
  - 编写README: 精确复现步骤
  - 保存到`ops/repro_packages/BASELINE_V1/`

#### 验收标准
- [x] Baseline指标可复现 (±5%)
- [x] OOS性能验证通过
- [x] Reproducibility Package完整
- [x] 文档详尽

#### 交付物
```
ops/repro_packages/BASELINE_V1/
├── code_snapshot/
├── configs/
├── data_manifest.yaml
├── results/
└── REPRODUCE_INSTRUCTIONS.md
```

---

### ☑️ P2-2: 监控与可观测性

**任务ID**: `MONITOR_001`  
**负责人**: 刘运维  
**预计工时**: 10小时

#### 执行步骤 (简化版)

- [ ] TensorBoard集成
- [ ] Weights & Biases (可选)
- [ ] 训练进度监控脚本
- [ ] 性能监控仪表板

---

### ☑️ P2-3: 审计日志激活

**任务ID**: `AUDIT_ACTIVATION_001`  
**负责人**: 周治理  
**预计工时**: 6小时

#### 执行步骤

- [ ] 确保所有task transition记录到ops/audit/
- [ ] 生成审计报告
- [ ] 与Gate检查集成

---

### ☑️ P2-4: 文档同步更新

**任务ID**: `DOC_SYNC_001`  
**负责人**: 全员  
**预计工时**: 8小时

#### 执行步骤

- [ ] 更新所有TaskCard状态
- [ ] 更新state/project.yaml
- [ ] 同步架构文档
- [ ] 生成Q1完成报告

---

## 📊 第四周进度跟踪表

| 任务 | 负责人 | 状态 | 进度 | 预计完成 | 实际完成 | 阻塞项 |
|------|--------|------|------|----------|----------|--------|
| P2-1: Baseline Repro | 张数据+周治理 | ⏳ PENDING | 0% | 02-28 | - | 依赖Week3 |
| P2-2: Monitoring | 刘运维 | ⏳ PENDING | 0% | 02-27 | - | - |
| P2-3: Audit | 周治理 | ⏳ PENDING | 0% | 02-26 | - | - |
| P2-4: Doc Sync | 全员 | ⏳ PENDING | 0% | 03-01 | - | - |

---

## 🎯 成功标准与验收

### 阶段性验收 (每周五)

#### Week 1 验收标准
- [ ] SDF Model可以forward pass并返回有效输出
- [ ] Moment Estimator可以计算Pricing Error
- [ ] 100股样本数据已获取并验证
- [ ] CI/CD流水线正常运行
- [ ] 所有新增代码测试覆盖率>80%

#### Week 2 验收标准
- [ ] SDF模型可以训练并收敛
- [ ] Oracle API可以评估portfolio PE
- [ ] 800股数据清洗完成
- [ ] SDF Layer E2E测试通过
- [ ] M1里程碑达成 (虽然延期到Week2)

#### Week 3 验收标准
- [ ] EA Optimizer可以优化权重
- [ ] 完整回测流程可以运行
- [ ] 5000股数据已准备就绪
- [ ] M2里程碑达成

#### Week 4 验收标准
- [ ] Baseline可复现 (±5%)
- [ ] OOS性能验证通过
- [ ] 监控系统上线
- [ ] 所有文档同步完成

---

## 📈 关键指标跟踪

### 代码质量指标
- **测试覆盖率**: 目标 >80%, 当前: TBD
- **Type Check通过率**: 目标 100%, 当前: TBD
- **Linter警告数**: 目标 <50, 当前: TBD

### 性能指标
- **SDF训练时间** (800股×252天): 目标 <2小时
- **EA优化时间** (100次迭代): 目标 <30分钟
- **完整Backtest时间** (5年): 目标 <10分钟

### 模型指标
- **Normalization Error**: 目标 |E[m_t]-1| < 0.05
- **OOS Sharpe Ratio**: 目标 >1.0
- **Max Drawdown**: 目标 <30%

---

## 🚨 风险管理与应急预案

### 高风险项与缓解措施

| 风险 | 概率 | 影响 | 缓解措施 | Plan B |
|------|------|------|----------|--------|
| **SDF模型不收敛** | 40% | 高 | 1. 先用小数据验证<br>2. 调整学习率和loss权重<br>3. 检查数据质量 | 回退到Legacy实现 |
| **数据获取困难** | 30% | 中 | 1. 提前联系数据供应商<br>2. 准备备用数据源 | 先用800股完成pipeline |
| **EA优化太慢** | 50% | 中 | 1. 使用GPU加速<br>2. 减少种群大小<br>3. 并行评估 | 简化为Grid Search |
| **M1/M2延期** | 70% | 高 | 1. 简化初版功能<br>2. 延后优化到下个sprint | 已在本计划中调整 |
| **人员瓶颈** | 40% | 中 | 1. 任务并行化<br>2. 知识分享会<br>3. Pair Programming | 申请外部支援 |
| **技术债累积** | 60% | 中 | 1. 建立TECH_DEBT.md追踪<br>2. 每周Code Review<br>3. 定期重构时间 | 在Week4集中重构 |

### 阻塞情况处理流程

1. **识别阻塞**: 每日站会中立即暴露
2. **评估影响**: 是否影响关键路径
3. **寻找解决方案**: 
   - 技术方案调整
   - 任务重新分配
   - 范围裁剪
4. **升级机制**: 
   - 超过1天阻塞 → 通知项目经理
   - 超过3天阻塞 → 通知Project Owner
   - 影响里程碑 → 召开紧急会议

---

## 📞 沟通与协作机制

### 每日站会 (Daily Standup)
- **时间**: 每天上午9:00
- **时长**: 15分钟
- **形式**: 线上会议/异步更新
- **内容**:
  - 昨天完成了什么
  - 今天计划做什么
  - 有什么阻塞

### 每周同步会 (Weekly Sync)
- **时间**: 每周五下午4:00
- **时长**: 1小时
- **内容**:
  - 回顾本周进度
  - 验收完成任务
  - 调整下周计划
  - 风险识别

### 代码审查 (Code Review)
- **频率**: 每个PR必须Review
- **Reviewer**: 至少1人, 关键模块需2人
- **标准**: 遵循Pair Programming流程
- **工具**: GitHub PR + Code Review Engine

### 文档更新
- **频率**: 每完成一个子任务
- **位置**: 
  - 任务状态 → `state/tasks.yaml`
  - 进度更新 → `ops/EXECUTION_PLAN_2026_Q1.md` (本文档)
  - 技术文档 → `projects/dgsf/repo/docs/`

---

## 📚 参考资料与资源链接

### 内部文档
- [PROJECT_PLAYBOOK.md](../docs/PROJECT_PLAYBOOK.md) - 项目执行手册
- [OS_OPERATING_MODEL.md](../docs/OS_OPERATING_MODEL.md) - 操作模式
- [SDF_DEV_001.md](../tasks/SDF_DEV_001.md) - SDF开发任务卡
- [EA_DEV_001.md](../tasks/EA_DEV_001.md) - EA开发任务卡
- [DATA_EXPANSION_001.md](../tasks/DATA_EXPANSION_001.md) - 数据扩展任务卡

### 技术规范
- SDF Layer Specification v3.1
- EA Layer Specification v3.1
- Data Engineering v4.2
- STATE_ENGINE_V1.0

### 工具与库
- PyTorch Documentation: https://pytorch.org/docs/
- CMA-ES Python: https://github.com/CMA-ES/pycma
- TensorBoard Guide: https://www.tensorflow.org/tensorboard

---

## 🔄 计划更新机制

### 本文档更新频率
- **日常更新**: 每日更新进度跟踪表中的状态
- **每周更新**: 每周五同步会后更新实际完成情况
- **重大调整**: 遇到关键阻塞或范围变更时立即更新

### 版本控制
- **当前版本**: v1.0 (2026-02-02 初始版本)
- **变更记录**: 见文档底部

### 与State同步
- 本计划的任务状态应与`state/tasks.yaml`保持一致
- 每周五执行同步检查

---

## ✅ 检查清单 (Checklist for Next Reviewer)

在下次扫描项目时，请首先检查本计划：

- [ ] 阅读本执行计划书 (5分钟)
- [ ] 检查当前处于哪一周
- [ ] 查看对应周的任务清单
- [ ] 检查进度跟踪表，识别延期任务
- [ ] 识别阻塞项
- [ ] 根据计划执行下一个未完成的任务
- [ ] 完成后更新本文档的进度

---

## 📝 执行记录 (Execution Log)

### 2026-02-02 (Week 0)
- ✅ 执行计划书创建完成
- ⏳ 等待团队kickoff会议
- ⏳ 下一步: P0-5 (Dependencies) - 最容易启动的任务

### Week 1 记录
_待填写..._

### Week 2 记录
_待填写..._

### Week 3 记录
_待填写..._

### Week 4 记录
_待填写..._

---

## 📄 文档变更历史

| 版本 | 日期 | 变更者 | 变更内容 |
|------|------|--------|----------|
| v1.0 | 2026-02-02 | 赵执行 | 初始版本创建 |
| v1.1 | 待定 | - | 待更新 |

---

## 🎯 最后的话

> **"计划不是一成不变的，但没有计划的执行注定失败。"**  
> — 赵执行 (项目管理大师)

本执行计划是一份**活文档**，应随着项目进展不断更新。它的价值不在于预测未来，而在于：
1. **建立共识** - 让团队对接下来4周的工作有清晰认知
2. **可追溯性** - 记录决策和变更的原因
3. **风险可见** - 及早暴露问题而非隐藏
4. **持续改进** - 从执行中学习，优化流程

**关键原则**:
- ✅ **小步快跑** - 每周都有可验证的产出
- ✅ **测试驱动** - 代码和测试一起写
- ✅ **透明沟通** - 问题越早暴露越好
- ✅ **质量优先** - 快速但不草率
- ✅ **可复现性** - 所有结果都应该可以复现

**致全体团队成员**:
这个计划是雄心勃勃的，但也是可实现的。关键在于专注、协作和坚持。让我们一起把AI Workflow OS和DGSF项目带到新的高度！

---

**下一步行动**: 
1. 召开Kickoff会议，讨论本计划
2. 确认资源分配
3. 开始执行P0-5 (最简单的任务作为热身)

**祝执行顺利！** 🚀
