# TaskCard: [DATA_2_XXX]

> **Stage**: 2 · Data Engineering (Versioned Data)  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `DATA_2_XXX` |
| **创建日期** | YYYY-MM-DD |
| **Role Mode** | `executor` / `builder` |
| **Authority** | `speculative` |
| **Authorized By** | [Project Owner ID] |
| **上游 Task** | `RESEARCH_1_XXX` |

---

## 1. 数据需求规范

### 1.1 来自 Research Design
<!-- 从 RESEARCH_1 继承的数据需求 -->
- **Universe**: 
- **时间范围**: 
- **频率**: 
- **数据类型**: 

### 1.2 数据源清单
| 数据源 | 类型 | 字段 | 获取方式 |
|--------|------|------|----------|
| | Raw Price | OHLCV | API / File |
| | Fundamental | | |
| | Alternative | | |

---

## 2. Raw Ingest

### 2.1 Schema 定义
```yaml
# 原始数据 Schema
raw_price:
  columns:
    - name: date
      type: datetime
      nullable: false
    - name: symbol
      type: string
      nullable: false
    - name: open
      type: float64
      nullable: true
    # ...
```

### 2.2 Schema 验证检查
- [ ] 必填字段完整性
- [ ] 数据类型符合预期
- [ ] 无异常空值
- [ ] 无重复记录

---

## 3. Cleaning & Alignment

### 3.1 Corporate Actions 处理
- [ ] 分红调整方式: _______________
- [ ] 拆股调整方式: _______________
- [ ] 配股处理: _______________

### 3.2 缺失数据策略
| 字段 | 缺失处理 | 阈值 |
|------|----------|------|
| price | forward-fill | max 5 days |
| volume | zero-fill | - |
| | | |

### 3.3 Calendar Alignment
- **交易日历**: 
- **时区处理**: 
- **节假日处理**: 

---

## 4. Feature/Factor Panel

### 4.1 Factor 定义
| Factor ID | 名称 | 公式/描述 | Lag |
|-----------|------|-----------|-----|
| F001 | | | |
| F002 | | | |

### 4.2 Leakage Controls
- [ ] Point-in-time 检查
- [ ] Look-ahead 检查
- [ ] Survivorship 处理
- [ ] Lag 规则验证

### 4.3 Lag Rules
```yaml
lag_rules:
  - factor: financial_data
    lag_days: 90  # 季报发布延迟
  - factor: price_data
    lag_days: 1   # T+1 可用
```

---

## 5. Data Snapshot

### 5.1 版本信息
| 字段 | 值 |
|------|-----|
| **Snapshot ID** | `DS_YYYYMMDD_HHMMSS` |
| **创建时间** | |
| **数据范围** | YYYY-MM-DD to YYYY-MM-DD |
| **记录数** | |
| **文件大小** | |

### 5.2 Checksums
```yaml
checksums:
  raw_price: sha256:...
  factors: sha256:...
  metadata: sha256:...
```

### 5.3 Metadata
```yaml
metadata:
  universe_count: 
  date_range: [start, end]
  factor_count: 
  frequency: daily
  created_by: [Agent ID]
  created_at: 
```

---

## 6. Gate G1: Data Quality

### 6.1 必须通过的检查
| 检查项 | 状态 | 说明 |
|--------|------|------|
| Schema 验证 | `pending` | |
| 缺失率 < 5% | `pending` | |
| 无 look-ahead | `pending` | |
| Snapshot 不可变 | `pending` | |
| Checksum 生成 | `pending` | |

### 6.2 Gate 结果
- [ ] **PASS** → 进入 Stage 3
- [ ] **FAIL** → 修复后重新验证

### 6.3 Gate 证据
<!-- 附上验证脚本输出或报告链接 -->
```
Gate G1 验证报告: [路径]
```

---

## 7. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Data Snapshot | `data/snapshots/DS_XXX/` | `pending` |
| Factor Library | `data/factors/FL_XXX/` | `pending` |
| Data Quality Report | `reports/DATA_2_XXX_quality.md` | `pending` |
| Checksums | `data/snapshots/DS_XXX/checksums.yaml` | `pending` |

---

## 8. 下游依赖

- **后续 TaskCard**: `DEV_3_XXX`
- **Stage 3 需要**: 
  - Immutable Data Snapshot ID
  - Factor Library with definitions
  - Lag rules

---

## 9. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: [Project Owner ID]
  scope: data_engineering
  expires: YYYY-MM-DD
  
# 数据 Snapshot 一旦创建即不可变
# 任何修改需创建新版本
```

---

## 10. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| | | `task_created` | From RESEARCH_1_XXX |
| | | `ingest_started` | |
| | | `gate_g1_passed` | |

---

*Template: TASKCARD_DATA_2 v1.0.0*
