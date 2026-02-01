# DGSF Data Directory

**Created by**: Data Engineer (王数据)  
**Date**: 2026-02-01  
**Task**: DATA_2_DGSF_001

---

## Directory Structure

```
data/
├── raw/              # 原始数据 (不可变)
│   ├── ohlcv/        # OHLCV价格数据
│   └── metadata/     # 数据元信息
├── processed/        # 处理后数据
│   ├── features/     # 特征数据
│   └── signals/      # 信号数据
├── snapshots/        # 版本化快照
│   └── DS_DGSF_{date}_V{n}/
└── checksums.yaml    # 数据完整性校验
```

## Data Sources

| Source | Type | Fields | Update Frequency |
|--------|------|--------|------------------|
| Binance API | OHLCV | timestamp, open, high, low, close, volume | 1H |
| CoinGecko | Reference | market_cap, circulating_supply | Daily |

## Usage

```python
from pathlib import Path

DATA_ROOT = Path("projects/dgsf/data")
SNAPSHOT_DIR = DATA_ROOT / "snapshots"

# Load latest snapshot
def load_snapshot(snapshot_id: str):
    path = SNAPSHOT_DIR / snapshot_id
    # Implementation
```

## Data Quality Gates (G1)

- [ ] Schema validation passed
- [ ] Missing rate < 5%
- [ ] No look-ahead leakage
- [ ] Checksum verified
