---
task_id: "DATA_EXPANSION_001"
type: data
queue: data
branch: "feature/DATA_EXPANSION_001"
priority: P1
spec_ids:
  - "DGSF_DataEng_V4.2"
  - "DATA_QUALITY_STANDARD"
  - "GOVERNANCE_INVARIANTS"
verification:
  - "Full A-share universe coverage (~5000 stocks)"
  - "Daily frequency data 2015-2025"
  - "Causality validation passed"
  - "Data loader integration complete"
---

# TaskCard: DATA_EXPANSION_001

> **Phase**: 0 · Data Expansion (并行任务)  
> **Pipeline**: DGSF Development Pipeline  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `DATA_EXPANSION_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `data_engineer` |
| **Authority** | `accepted` |
| **Authorized By** | Project Owner |
| **并行任务** | `SDF_DEV_001`, `EA_DEV_001` |
| **下游依赖** | `BASELINE_REPRO_001`, Phase 4 Validation |

---

## 1. 任务背景

### 1.1 问题陈述

**当前状态**:
- ✅ 数据工程规范 v4.2 (FINAL)
- ✅ 因果性数据管道框架
- ⚠️ **仅使用中证800成分股进行回测**

**问题**:
- 中证800 仅覆盖 ~800 只股票，样本代表性不足
- 无法支撑可复现的学术发表
- OOS 验证范围受限

**目标**:
- 扩展到 **全量 A 股 (~5000 只)** 日频数据
- 覆盖 **2015-2025** 完整时间范围
- 确保因果性约束

### 1.2 数据规模对比

| 指标 | 当前 (中证800) | 目标 (全量A股) | 增量 |
|------|----------------|----------------|------|
| 股票数量 | ~800 | ~5000 | **6x** |
| 日期范围 | 2015-2023 | 2015-2025 | +2年 |
| 数据量 | ~1.25 GB | ~8-10 GB | **8x** |
| 特征维度 | 94 | 94 | - |
| 交易日数 | ~2000 | ~2500 | +500 |

---

## 2. 任务范围

### 2.1 股票池定义 (DATA_EXPANSION_001.1)

**工作内容**:

```python
# 股票池定义逻辑
def define_universe(date: str) -> List[str]:
    """
    获取指定日期的有效股票池
    
    规则:
    1. 全部 A 股 (沪深主板 + 创业板 + 科创板)
    2. 剔除 ST/*ST 股票
    3. 剔除上市不满 90 天的新股
    4. 剔除停牌超过 20 天的股票
    5. 剔除退市股票
    
    Returns:
        股票代码列表 (约 4500-5000 只)
    """
    pass
```

**历史成分股处理**:
- 需要维护 **每日有效股票池** (survivorship bias free)
- 退市股票需要保留历史数据
- 新股需要从上市第 91 天开始纳入

**交付物**:
- [ ] `universe_daily.parquet` - 每日股票池
- [ ] 股票池定义文档

**工作量**: 4h

---

### 2.2 数据采集 (DATA_EXPANSION_001.2)

**数据类型**:

| 数据类型 | 频率 | 字段 | 来源 |
|----------|------|------|------|
| 行情数据 | 日频 | OHLCV, 涨跌幅, 换手率 | Wind/聚宽 |
| 财务数据 | 季频 | ROE, EP, BP, etc. | Wind/聚宽 |
| 特征因子 | 日频 | 94 个标准特征 | 自行计算 |

**94 特征因子列表** (与 Legacy 保持一致):

```
技术类 (20):
- 动量: ret_1m, ret_3m, ret_6m, ret_12m
- 波动: vol_20d, vol_60d, vol_252d
- 流动性: turnover_20d, amihud_illiq
- ...

基本面类 (30):
- 估值: ep, bp, sp, cfp
- 盈利: roe, roa, gpm, npm
- 成长: rev_growth, eps_growth
- ...

风险类 (20):
- 市场 beta, 行业暴露
- 规模因子 (log_mcap)
- ...

其他 (24):
- 分析师预期
- 机构持仓
- ...
```

**交付物**:
- [ ] `daily_quotes.parquet` - 日频行情
- [ ] `quarterly_fundamentals.parquet` - 季频财务
- [ ] `daily_features.parquet` - 94 特征因子

**工作量**: 16h

---

### 2.3 数据清洗 (DATA_EXPANSION_001.3)

**清洗规则**:

```python
class DataCleaner:
    """数据清洗器"""
    
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        缺失值处理:
        - 行情数据: 停牌日填充 NaN，不做前向填充
        - 财务数据: 使用最近可得值 (point-in-time)
        - 因子数据: 横截面中位数填充
        """
        pass
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        异常值检测:
        - 涨跌幅 > ±20% 标记 (新股除外)
        - 换手率 > 50% 标记
        - 财务指标 MAD 3σ 截断
        """
        pass
    
    def align_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间戳对齐:
        - 确保所有数据使用交易日
        - 财务数据使用报告发布日 (非报告期末)
        """
        pass
```

**交付物**:
- [ ] 清洗后数据集
- [ ] 数据质量报告 (缺失率、异常值统计)

**工作量**: 8h

---

### 2.4 因果性验证 (DATA_EXPANSION_001.4)

**验证项目**:

| 验证项 | 方法 | 通过标准 |
|--------|------|----------|
| Look-ahead 检测 | 时间戳检查 | 100% 通过 |
| t/t+1 分离 | 收益计算验证 | 100% 正确 |
| 财务数据时点 | Point-in-time 验证 | 100% 正确 |
| 特征计算 | 只用 t 及之前数据 | 100% 正确 |

**验证代码**:

```python
def validate_causality(df: pd.DataFrame) -> Dict[str, bool]:
    """
    因果性验证
    
    检查项:
    1. feature_date <= trade_date (特征只用历史数据)
    2. return_date = trade_date + 1 (收益是下一期)
    3. fundamental_date = report_publish_date (财务用发布日)
    """
    results = {
        "feature_causality": check_feature_dates(df),
        "return_causality": check_return_dates(df),
        "fundamental_causality": check_fundamental_dates(df)
    }
    return results
```

**交付物**:
- [ ] 因果性验证报告
- [ ] 验证脚本

**工作量**: 4h

---

### 2.5 存储与索引 (DATA_EXPANSION_001.5)

**存储结构**:

```
projects/dgsf/data/full_universe/
├── universe/
│   └── daily_universe.parquet      # 每日股票池
├── quotes/
│   ├── 2015.parquet
│   ├── 2016.parquet
│   └── ...                         # 按年分区
├── fundamentals/
│   └── quarterly.parquet           # 季频财务
├── features/
│   ├── 2015.parquet
│   ├── 2016.parquet
│   └── ...                         # 按年分区
└── metadata/
    ├── schema.yaml
    ├── quality_report.md
    └── causality_report.md
```

**数据加载器适配**:

```python
# 更新 DGSFDataLoader 支持全量数据
class DGSFDataLoader:
    def __init__(self, data_path: str, universe: str = "full"):
        """
        Args:
            universe: "csi800" (当前) | "full" (全量A股)
        """
        self.universe = universe
        self.data_path = data_path / universe
```

**交付物**:
- [ ] Parquet 分区存储
- [ ] 更新后的 data_loader.py
- [ ] 性能测试报告

**工作量**: 4h

---

## 3. 交付物汇总

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 每日股票池 | `data/full_universe/universe/daily_universe.parquet` | `pending` |
| 日频行情 | `data/full_universe/quotes/*.parquet` | `pending` |
| 季频财务 | `data/full_universe/fundamentals/quarterly.parquet` | `pending` |
| 日频特征 | `data/full_universe/features/*.parquet` | `pending` |
| 数据质量报告 | `docs/DATA_QUALITY_REPORT.md` | `pending` |
| 因果性验证报告 | `docs/CAUSALITY_VERIFICATION_FULL.md` | `pending` |
| 更新后数据加载器 | `adapter/data_loader.py` | `pending` |

---

## 4. 验收标准

### 4.1 数据完整性
- [ ] 股票池覆盖 ≥4500 只 (剔除后)
- [ ] 时间范围 2015-01-01 ~ 2025-12-31
- [ ] 94 特征全部计算完成
- [ ] 缺失率 <5% (行情数据)

### 4.2 因果性
- [ ] 100% 通过 look-ahead 检测
- [ ] 100% 通过 t/t+1 分离验证
- [ ] 100% 通过财务数据时点验证

### 4.3 性能
- [ ] 单年数据加载 <10s
- [ ] 全量数据加载 <60s
- [ ] Rolling window 迭代可接受

---

## 5. 时间估算

| 子任务 | 工作量 | 负责人 | 预计完成 |
|--------|--------|--------|----------|
| 股票池定义 | 4h | 王数据 | 02/05 |
| 数据采集 | 16h | 王数据 | 02/14 |
| 数据清洗 | 8h | 王数据 | 02/21 |
| 因果性验证 | 4h | 王数据 | 02/25 |
| 存储与索引 | 4h | 王数据 | 02/28 |
| **总计** | **36h (~5 天)** | - | **03/15** |

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 数据源不可用 | 低 | 高 | 准备备用数据源 |
| 数据质量问题 | 中 | 中 | 增量验证流程 |
| 存储空间不足 | 低 | 中 | 压缩 + 云存储 |
| 因果性违规 | 低 | 极高 | 严格验证流程 |

---

## 7. 依赖关系

```
DATA_EXPANSION_001 (并行)
        │
        ├── SDF_DEV_001 (不依赖)
        ├── EA_DEV_001 (不依赖)
        │
        ▼
PIPELINE_INTEGRATION_001 ───► BASELINE_REPRO_001 (需要全量数据)
                                      │
                                      ▼
                              Phase 4: OOS Validation (需要全量数据)
```

---

## 8. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T23:55:00Z | system | `task_created` | 任务创建 - 数据扩展 |

