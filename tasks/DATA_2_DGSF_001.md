---
task_id: "DATA_2_DGSF_001"
type: data
queue: data
branch: "feature/DATA_2_DGSF_001"
priority: P2
spec_ids:
  - ARCH_BLUEPRINT_MASTER
  - TASK_STATE_MACHINE
  - GOVERNANCE_INVARIANTS
verification:
  - "Raw data schema validation passed"
  - "Missing data rate < 5%"
  - "No look-ahead leakage detected"
  - "Immutable snapshot created with checksums"
---

# TaskCard: DATA_2_DGSF_001

> **Stage**: 2 · Data Engineering (Versioned Data)  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `DATA_2_DGSF_001` |
| **创建日期** | 2026-02-01 |
| **Role Mode** | `executor` / `builder` |
| **Authority** | `speculative` |
| **Authorized By** | Project Owner |
| **上游 Task** | `RESEARCH_1_DGSF_001` |

---

## 1. 数据需求规范

### 1.1 来自 Research Design
<!-- 从 RESEARCH_1_DGSF_001 继承的数据需求 -->
- **Universe**: BTC/USDT, ETH/USDT
- **时间范围**: 2020-01-01 ~ 2025-12-31
- **频率**: 1h (hourly)
- **数据类型**: OHLCV (Open, High, Low, Close, Volume)

### 1.2 数据源清单
| 数据源 | 类型 | 字段 | 获取方式 |
|--------|------|------|----------|
| Binance API | Raw Price | OHLCV | REST API |
| Historical Files | Backup | OHLCV | CSV Files |
| CoinGecko | Reference | Market Cap | API |

---

## 2. Raw Ingest

### 2.1 Schema 定义
```yaml
# 原始数据 Schema
raw_ohlcv:
  columns:
    - name: timestamp
      type: datetime
      nullable: false
      description: "开盘时间 (UTC)"
    - name: symbol
      type: string
      nullable: false
      description: "交易对符号"
    - name: open
      type: float64
      nullable: false
    - name: high
      type: float64
      nullable: false
    - name: low
      type: float64
      nullable: false
    - name: close
      type: float64
      nullable: false
    - name: volume
      type: float64
      nullable: true
      description: "成交量 (base currency)"
    - name: quote_volume
      type: float64
      nullable: true
      description: "成交额 (quote currency)"
```

### 2.2 Schema 验证检查
- [ ] 必填字段完整性 (timestamp, symbol, OHLC)
- [ ] 数据类型符合预期 (float64 for prices)
- [ ] 无异常空值 (< 0.1% missing)
- [ ] 无重复记录 (unique timestamp+symbol)

---

## 3. Cleaning & Alignment

### 3.1 异常值处理
- [ ] 价格跳跃检测: > 10% 单小时变动标记
- [ ] 成交量异常: > 3σ 标记审查
- [ ] Zero price handling: 标记为 missing

### 3.2 缺失数据策略
| 字段 | 缺失处理 | 阈值 |
|------|----------|------|
| OHLC | forward-fill | max 6 hours |
| volume | zero-fill | - |
| quote_volume | interpolate | max 12 hours |

### 3.3 Calendar Alignment
- **交易日历**: 7x24 (加密货币)
- **时区处理**: All timestamps in UTC
- **维护时段**: Exchange maintenance windows excluded

---

## 4. Feature/Factor Panel

### 4.1 Factor 定义
| Factor ID | 名称 | 公式/描述 | Lag |
|-----------|------|-----------|-----|
| F001 | ATR_14 | 14周期ATR | 1h |
| F002 | BB_Width | 布林带宽度 | 1h |
| F003 | ADX_14 | 趋势强度指标 | 1h |
| F004 | RSI_14 | 相对强弱指数 | 1h |
| F005 | Grid_Level | 当前网格位置 | 0h |
| F006 | Position_Ratio | 当前仓位比例 | 0h |

### 4.2 Leakage Controls
- [ ] Point-in-time 检查: 使用 close 价格计算信号
- [ ] Look-ahead 检查: 无未来数据引用
- [ ] 无 survivorship bias (加密货币不适用)
- [ ] Lag 规则验证: 所有指标至少1小时延迟

### 4.3 Lag Rules
```yaml
lag_rules:
  - factor: technical_indicators
    lag_hours: 1     # 当前bar close后可用
  - factor: position_state
    lag_hours: 0     # 实时可用
  - factor: volume_metrics
    lag_hours: 1     # 当前bar结束后可用
```

---

## 5. Data Snapshot

### 5.1 版本信息
| 字段 | 值 |
|------|-----|
| **Snapshot ID** | `DS_DGSF_20260201_V1` |
| **创建时间** | pending |
| **数据范围** | 2020-01-01 to 2025-12-31 |
| **记录数** | ~52,560 per symbol (6 years * 8760 hours) |
| **预估文件大小** | ~50MB |

### 5.2 Checksums
```yaml
checksums:
  raw_btc: sha256:pending
  raw_eth: sha256:pending
  factors: sha256:pending
  metadata: sha256:pending
```

### 5.3 Metadata
```yaml
metadata:
  universe_count: 2
  date_range: ["2020-01-01", "2025-12-31"]
  factor_count: 6
  frequency: 1h
  created_by: data_engineer
  created_at: pending
  project: DGSF
  version: "1.0.0"
```

---

## 6. Gate G1: Data Quality

### 6.1 必须通过的检查
| 检查项 | 状态 | 说明 |
|--------|------|------|
| Schema 验证 | `pending` | OHLCV 字段完整 |
| 缺失率 < 5% | `pending` | 允许短暂交易所维护 |
| 无 look-ahead | `pending` | 脚本自动验证 |
| Snapshot 不可变 | `pending` | 只读文件权限 |
| Checksum 生成 | `pending` | SHA256 |

### 6.2 Gate 结果
- [ ] **PASS** → 进入 Stage 3 (DEV_3_DGSF_001)
- [ ] **FAIL** → 修复后重新验证

### 6.3 Gate 证据
```
Gate G1 验证报告: reports/gates/DATA_2_DGSF_001_G1.md
验证脚本: scripts/validate_data_quality.py
```

---

## 7. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Data Snapshot | `projects/dgsf/data/snapshots/DS_DGSF_20260201_V1/` | `pending` |
| Factor Library | `projects/dgsf/data/factors/FL_DGSF_V1/` | `pending` |
| Data Quality Report | `reports/gates/DATA_2_DGSF_001_G1.md` | `pending` |
| Checksums | `projects/dgsf/data/snapshots/DS_DGSF_20260201_V1/checksums.yaml` | `pending` |

---

## 8. 下游依赖

- **后续 TaskCard**: `DEV_3_DGSF_001`
- **Stage 3 需要**: 
  - Immutable Data Snapshot ID: `DS_DGSF_20260201_V1`
  - Factor Library: 6 factors defined
  - Lag rules: 1h for indicators, 0h for position state

---

## 9. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: Project Owner
  scope: data_engineering
  expires: 2026-03-31
  
# 数据 Snapshot 一旦创建即不可变
# 任何修改需创建新版本
```

---

## 10. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| 2026-02-01T18:30:00Z | liu_pm | `task_created` | From RESEARCH_1_DGSF_001 |

---

*Template: TASKCARD_DATA_2 v1.0.0*  
*Created by: 刘PM (Project Manager)*
