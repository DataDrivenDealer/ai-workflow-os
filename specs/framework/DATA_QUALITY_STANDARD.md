# DATA_QUALITY_STANDARD

**Spec ID**: DATA_QUALITY_STANDARD  
**Scope**: L1 Framework  
**Version**: 1.0.0  
**Status**: Active  
**Owner**: System Architect

---

## 1. Purpose

This specification defines data quality standards for all data engineering tasks within AI Workflow OS. It provides requirements for G1 (Data Quality Gate) compliance.

---

## 2. Data Quality Dimensions

### 2.1 Completeness

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Missing Rate (overall) | ≤ 5% | Error |
| Missing Rate (critical fields) | ≤ 0.1% | Error |
| Coverage (date range) | 100% of specified range | Warning |

### 2.2 Accuracy

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Schema Compliance | 100% | Error |
| Type Compliance | 100% | Error |
| Range Validation | 100% within bounds | Error |

### 2.3 Consistency

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Duplicate Records | 0 | Error |
| Referential Integrity | 100% | Error |
| Cross-source Consistency | ≥ 99.9% | Warning |

### 2.4 Timeliness

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Data Freshness | Within SLA | Warning |
| Update Latency | < configured limit | Warning |

---

## 3. No Look-Ahead Policy

### 3.1 Definition
Look-ahead bias occurs when future information is used to make decisions about past events. This MUST be prevented in all data processing.

### 3.2 Checks

```python
def check_no_lookahead(df: pd.DataFrame, target_col: str, feature_cols: list) -> bool:
    """
    Verify no feature uses future target values.
    
    For each timestamp t:
    - Features at t can only use data from timestamps ≤ t
    - Target at t can use data from timestamps > t
    """
    # Implementation required for G1 gate
    pass
```

### 3.3 Common Violations

- Using close price to predict same-period returns
- Including future MA values in features
- Survivorship bias in universe selection

---

## 4. Data Versioning

### 4.1 Snapshot Requirements

Every data snapshot MUST include:

```yaml
snapshot:
  id: "DS_{PROJECT}_{DATE}_V{N}"
  created_at: "ISO8601 timestamp"
  created_by: "Agent/User ID"
  source_version: "upstream data version"
  schema_version: "schema definition version"
  
  contents:
    - file: "relative/path/to/file.parquet"
      sha256: "checksum"
      rows: 12345
      size_bytes: 1234567
```

### 4.2 Immutability

Once created, snapshots are IMMUTABLE:
- No in-place modifications
- Create new version for updates
- Archive old versions, don't delete

---

## 5. Schema Definition

### 5.1 Required Schema Elements

```yaml
schema:
  name: "schema_name"
  version: "1.0.0"
  columns:
    - name: column_name
      type: data_type  # int64, float64, string, datetime, bool
      nullable: true | false
      description: "Human readable description"
      constraints:  # Optional
        min: value
        max: value
        enum: [val1, val2]
```

### 5.2 OHLCV Standard Schema

```yaml
ohlcv_standard:
  columns:
    - name: timestamp
      type: datetime64[ns, UTC]
      nullable: false
      description: "Candle open time in UTC"
    - name: symbol
      type: string
      nullable: false
      description: "Trading pair symbol"
    - name: open
      type: float64
      nullable: false
      constraints:
        min: 0
    - name: high
      type: float64
      nullable: false
      constraints:
        min: 0
    - name: low
      type: float64
      nullable: false
      constraints:
        min: 0
    - name: close
      type: float64
      nullable: false
      constraints:
        min: 0
    - name: volume
      type: float64
      nullable: true
      constraints:
        min: 0
```

---

## 6. G1 Gate Checklist

```yaml
G1_gate_checklist:
  schema_valid:
    - All columns present
    - All types match
    - No unexpected columns
    
  completeness:
    - Missing rate < 5%
    - Critical fields < 0.1% missing
    - Date range complete
    
  no_lookahead:
    - Feature-target alignment verified
    - No future data in features
    
  immutability:
    - Snapshot checksum verified
    - No modifications since creation
```

---

*Spec maintained by: System Architect*  
*Last updated: 2026-02-01*
