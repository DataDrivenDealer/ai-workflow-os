---
task_id: "DATA_MIGRATION_001"
type: migration
queue: dev
branch: "feature/DATA_MIGRATION_001"
priority: P1
spec_ids:
  - GOVERNANCE_INVARIANTS
  - PROJECT_DGSF
  - DGSF_DATAENG_V3
verification:
  - "Data paths validated and accessible"
  - "Causality constraints verified (no look-ahead)"
  - "Data access API integrated with adapter"
  - "Gate checks pass"
---

# TaskCard: DATA_MIGRATION_001

> **Stage**: 2 · Data Migration  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `DATA_MIGRATION_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `builder` / `analyst` |
| **Authority** | `accepted` |
| **Authorized By** | Project Owner (via pipeline approval) |
| **上游 Task** | `SPEC_INTEGRATION_001` (✅ COMPLETED) |

---

## 1. 任务背景

### 1.1 前置条件
规范集成已完成，适配层已就绪：

| 完成项 | 状态 |
|--------|------|
| SPEC_MAPPING_PLAN.md | ✅ |
| Adapter Layer (5 modules) | ✅ |
| PROJECT_DGSF.yaml v2.1.0 | ✅ |

### 1.2 数据资产概览
根据 DATA_ASSET_INVENTORY.md 评估报告：

| 目录 | 大小 | 文件数 | 说明 |
|------|------|--------|------|
| `data/a0/` | 61 MB | 5 | A 股基础数据集 |
| `data/full/` | 689 MB | 86 | 完整特征数据集 |
| `data/final/` | 483 MB | 40 | 最终处理数据集 |
| `data/interim/` | - | - | 中间处理数据 |
| `data/processed/` | - | - | 已处理数据 |
| **总计** | **1.25 GB** | **327** | - |

---

## 2. 任务范围

### 2.1 数据路径验证 (王数据 负责)

#### 2.1.1 验证清单
- [x] 验证 `data/a0/` 目录存在且可读
- [x] 验证 `data/full/` 目录存在且可读
- [x] 验证 `data/final/` 目录存在且可读
- [x] 验证关键 Parquet 文件完整性
- [x] 验证 Arrow/Feather 文件可加载

#### 2.1.2 关键数据文件
```
data/full/
├── OHLCV_full.parquet          # 行情数据
├── firm_chars_full.parquet     # 公司特征
├── macro_vars_full.parquet     # 宏观变量
└── factors_full.parquet        # 因子数据

data/a0/
├── base_chars.parquet          # 基础特征
├── returns.parquet             # 收益率
└── market_caps.parquet         # 市值
```

### 2.2 因果性验证 (林质量 负责)

#### 2.2.1 Look-Ahead 检测
根据 DGSF Data Engineering Spec v4.2 的要求：
- [x] 验证数据时间戳对齐规则
- [x] 验证滞后变量正确应用
- [x] 验证训练/测试集分割无泄漏
- [x] 运行 `check_lookahead.py` 脚本

#### 2.2.2 因果约束规则
| 约束 | 描述 | 验证方法 |
|------|------|----------|
| C1 | 特征必须使用 t-1 或更早数据 | 时间戳检查 |
| C2 | 标签使用 t 期数据 | 对齐验证 |
| C3 | 市场变量必须提前 1 期 | 滞后检查 |
| C4 | 无未来信息泄漏 | 脚本验证 |

### 2.3 数据访问 API 集成 (张平台 负责)

#### 2.3.1 适配层扩展
在 `dgsf_adapter.py` 中添加：
```python
def get_data(self, dataset: str, subset: str = None) -> pd.DataFrame:
    """Load dataset from Legacy DGSF data directory."""
    
def list_datasets(self) -> List[str]:
    """List available datasets."""
    
def validate_causality(self, df: pd.DataFrame) -> Dict[str, bool]:
    """Validate causality constraints on dataframe."""
```

#### 2.3.2 数据加载器
创建 `projects/dgsf/adapter/data_loader.py`：
- Parquet/Arrow 文件加载
- 内存映射支持
- 缓存机制
- 因果性验证集成

---

## 3. 交付物

| 交付物 | 路径 | 状态 |
|--------|------|------|
| 数据路径验证报告 | `projects/dgsf/docs/DATA_PATH_VALIDATION.md` | ✅ `completed` |
| 因果性验证报告 | `projects/dgsf/docs/CAUSALITY_VERIFICATION.md` | ✅ `completed` |
| 数据加载器模块 | `projects/dgsf/adapter/data_loader.py` | ✅ `completed` |
| 更新的适配层 | `projects/dgsf/adapter/__init__.py` v1.1.0 | ✅ `completed` |

---

## 4. 验收标准

### 4.1 必须完成
- [x] 所有数据目录可访问
- [x] 关键 Parquet 文件可加载
- [x] 因果性验证通过
- [x] 数据加载 API 可用

### 4.2 Gate 检查
```yaml
gates:
  data_migration:
    trigger: "pre_stage_2"
    checks:
      - name: "data_integrity"
        description: "Verify data integrity after migration"
      - name: "causality_preserved"
        description: "No look-ahead leakage introduced"
```

---

## 5. 时间估算

| 子任务 | 工作量 | 负责人 |
|--------|--------|--------|
| 数据路径验证 | 0.5 天 | 王数据 |
| 因果性验证 | 1 天 | 林质量 |
| 数据加载器开发 | 1 天 | 张平台 |
| 集成测试 | 0.5 天 | 林质量 |
| **总计** | **3 天** | - |

---

## 6. Gate & 下游依赖

- **Gate G2**: Data Migration Review
  - 数据路径验证通过
  - 因果性检查通过
  - 数据加载 API 可用
- **后续 TaskCard**: `REPRO_VERIFY_001`
- **依赖**: `SPEC_INTEGRATION_001` (✅ COMPLETED)

---

## 7. Authority 声明

```yaml
authority:
  type: accepted
  granted_by: Project Owner
  scope: data_migration
  decision_date: 2026-02-01
  
# 通过 pipeline 批准，本任务具有执行权限
```

---

## 8. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T23:00:00Z | system | `task_created` | Stage 2 任务创建 |
| 2026-02-01T23:00:00Z | system | `task_start` | 任务开始执行 |
| 2026-02-01T23:15:00Z | copilot_agent | `deliverable_completed` | DATA_PATH_VALIDATION.md 完成 |
| 2026-02-01T23:15:00Z | copilot_agent | `deliverable_completed` | CAUSALITY_VERIFICATION.md 完成 |
| 2026-02-01T23:15:00Z | copilot_agent | `deliverable_completed` | data_loader.py 完成 |
| 2026-02-01T23:15:00Z | copilot_agent | `gate_passed` | Gate G2 (Data Migration) PASSED |
| 2026-02-01T23:15:00Z | system | `task_finish` | 任务完成 |

---

## 9. 完成摘要

### 9.1 数据基础设施

| 指标 | 值 |
|------|-----|
| 总数据集 | 11 个目录 |
| 总文件数 | 324 个 Parquet 文件 |
| 总数据量 | 1.25 GB |
| 核心数据集 | a0, full, final |
| 数据管道 | DE1-DE7 完整 |

### 9.2 适配层扩展

| 模块 | 类 | 新增功能 |
|------|-----|----------|
| `data_loader.py` | `DGSFDataLoader` | Parquet 加载、缓存、因果性验证 |
| `__init__.py` | - | 导出 `get_data_loader()` |

### 9.3 Gate 状态

- **Gate G2 (Data Migration)**: ✅ PASSED
  - `data_integrity`: ✅
  - `causality_preserved`: ✅

