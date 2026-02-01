# DGSF 数据路径验证报告

> **Task**: DATA_MIGRATION_001  
> **生成日期**: 2026-02-01  
> **执行者**: Copilot Agent (王数据 角色)  
> **状态**: ✅ PASSED

---

## 1. 执行摘要

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 数据根目录 | ✅ PASS | `projects/dgsf/legacy/DGSF/data/` 存在 |
| 核心数据集 | ✅ PASS | a0, full, final 均可访问 |
| 关键文件 | ✅ PASS | Parquet 文件完整 |
| 文件可读性 | ✅ PASS | 所有文件可被 PyArrow 读取 |

**总体评估**: 数据基础设施健康，可继续数据迁移流程。

---

## 2. 数据集清单

### 2.1 目录统计

| 目录 | 文件数 | 大小 (MB) | 描述 |
|------|--------|-----------|------|
| `a0/` | 21 | 125.66 | A股基础数据集 (mini) |
| `full/` | 26 | 398.25 | 完整特征数据集 |
| `cache/` | 3 | 295.08 | 缓存数据 |
| `interim/` | 12 | 28.50 | 中间处理数据 |
| `final/` | 4 | 2.73 | 最终处理数据 |
| `paneltree/` | 5 | 0.51 | PanelTree 输出 |
| `paneltree_v2/` | 5 | 0.31 | PanelTree v2 输出 |
| `processed/` | 3 | 0.07 | 已处理数据 |
| `raw/` | 7 | 0.06 | 原始数据 |
| `deprecated/` | 236 | 400.27 | 已弃用数据 |
| `external/` | 2 | 0.22 | 外部数据 |
| **总计** | **324** | **1,251.66** | - |

### 2.2 核心数据文件 (full/)

| 文件名 | 大小 (MB) | 状态 |
|--------|-----------|------|
| `de1_canonical_daily.parquet` | 195.17 | ✅ |
| `de1_daily_basic.parquet` | 127.71 | ✅ |
| `de7_factor_panel_v2.parquet` | 10.30 | ✅ |
| `monthly_features_v2.parquet` | 9.69 | ✅ |
| `de7_factor_panel.parquet` | 6.15 | ✅ |
| `monthly_features.parquet` | 5.50 | ✅ |
| `de1_canonical_monthly.parquet` | 5.36 | ✅ |
| `de4_valuation_factors.parquet` | 4.52 | ✅ |
| `de1_adj_factor.parquet` | 4.27 | ✅ |
| `de5_microstructure_monthly.parquet` | 3.46 | ✅ |

### 2.3 基础数据文件 (a0/)

| 文件名 | 大小 (MB) | 状态 |
|--------|-----------|------|
| `daily_basic.parquet` | 0.55 | ✅ |
| `daily_prices.parquet` | 0.23 | ✅ |
| `monthly_prices.parquet` | 0.12 | ✅ |
| `adj_factor.parquet` | 0.04 | ✅ |
| `sdf_linear_rolling_oos_metrics.parquet` | 0.01 | ✅ |
| `style_spreads_monthly.parquet` | 0.01 | ✅ |

### 2.4 中间数据文件 (a0/interim/)

| 文件名 | 大小 (MB) | 状态 |
|--------|-----------|------|
| `de1_canonical_full.parquet` | 92.14 | ✅ |
| `de7_factor_panel.parquet` | 7.81 | ✅ |
| `de3_fina_cashflow_eff.parquet` | 5.28 | ✅ |
| `de3_fina_balance_eff.parquet` | 4.79 | ✅ |
| `de1_canonical_monthly.parquet` | 3.91 | ✅ |
| `de3_fina_income_eff.parquet` | 3.80 | ✅ |
| `de5_market_micro_monthly.parquet` | 3.42 | ✅ |
| `de4_fina_indicator_eff.parquet` | 3.39 | ✅ |

---

## 3. 数据管道映射

根据 DGSF Data Engineering Spec v4.2：

```
┌─────────────────────────────────────────────────────────────┐
│                    DGSF Data Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [raw/]  →  [interim/]  →  [processed/]  →  [full/]         │
│     │           │              │              │              │
│     └───────────┼──────────────┼──────────────┘              │
│                 ↓              ↓                             │
│           de1_canonical   de7_factor_panel                   │
│           de3_fina_*      monthly_features                   │
│           de4_valuation                                      │
│           de5_microstructure                                 │
│           de6_universe                                       │
│                                                              │
│  [a0/] = Mini dataset for quick iteration                   │
│  [final/] = Production-ready outputs                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 DE (Data Engineering) 模块对应

| DE 模块 | 描述 | 输出文件 |
|---------|------|----------|
| DE1 | Canonical daily/monthly | `de1_canonical_*.parquet` |
| DE2 | Stock pool universe | (integrated in DE6) |
| DE3 | Financial statements | `de3_fina_*.parquet` |
| DE4 | Valuation factors | `de4_valuation_factors.parquet` |
| DE5 | Market microstructure | `de5_microstructure_monthly.parquet` |
| DE6 | Universe & index | `de6_universe_mask.parquet`, `de6_index_churn.parquet` |
| DE7 | Factor panel | `de7_factor_panel.parquet` |

---

## 4. 访问验证

### 4.1 适配层健康检查

```python
# 验证代码 (via DGSFDataLoader.health_check())
{
    "data_root_exists": True,
    "pandas_available": True,
    "pyarrow_available": True,
    "a0_accessible": True,
    "full_accessible": True,
    "key_files_present": True,
}
```

### 4.2 API 访问测试

| 操作 | 方法 | 结果 |
|------|------|------|
| 列出数据集 | `loader.list_datasets()` | ✅ 11 datasets |
| 列出文件 | `loader.list_files("full")` | ✅ 26 files |
| 加载数据 | `loader.load("a0", "daily_basic")` | ✅ DataFrame |
| 获取元数据 | `loader.get_metadata("full", "de1_canonical_daily")` | ✅ Schema info |

---

## 5. 建议与后续步骤

### 5.1 数据优化建议

1. **deprecated/ 清理**: 400MB deprecated 数据可考虑归档或删除
2. **缓存策略**: cache/ 目录 295MB，建议定期清理
3. **分区优化**: 大文件 (>100MB) 考虑按日期分区

### 5.2 后续任务

- [x] 数据路径验证 ← 本报告
- [ ] 因果性验证 (CAUSALITY_VERIFICATION.md)
- [ ] 数据加载 API 集成测试

---

## 6. 签核

| 角色 | 签核 | 日期 |
|------|------|------|
| 王数据 (Data Analyst) | ✅ | 2026-02-01 |
| 林质量 (QA Lead) | ⏳ Pending | - |

---

*本报告由 AI Workflow OS 自动生成，作为 DATA_MIGRATION_001 任务交付物之一。*
