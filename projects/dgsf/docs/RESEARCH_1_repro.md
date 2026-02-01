# RESEARCH_1_DGSF_001 — 可复现性包

**文档 ID**: RESEARCH_1_repro  
**Task**: RESEARCH_1_DGSF_001  
**作者**: 陈研究 (Quant Researcher)  
**创建日期**: 2026-02-01  
**状态**: Active

---

## 0. 概述

本文档定义了 DGSF 动态网格策略研究的可复现性要求，确保任何研究员可以在相同条件下复现实验结果。

---

## 1. 环境要求

### 1.1 硬件要求
```yaml
minimum_requirements:
  cpu: "4 cores"
  ram: "16 GB"
  disk: "50 GB SSD"
  
recommended:
  cpu: "8 cores"
  ram: "32 GB"
  disk: "100 GB SSD"
```

### 1.2 软件环境
```yaml
python_version: "3.11.x"
os_support:
  - "Windows 10/11"
  - "Ubuntu 22.04+"
  - "macOS 13+"

core_dependencies:
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scipy>=1.10.0
  - scikit-learn>=1.3.0
  
trading_dependencies:
  - ccxt>=4.0.0     # 交易所API
  - ta-lib>=0.4.0   # 技术指标（可选）
  
visualization:
  - matplotlib>=3.7.0
  - plotly>=5.15.0
```

### 1.3 环境设置
```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境 (Windows)
.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r projects/dgsf/requirements.txt
```

---

## 2. 数据依赖

### 2.1 数据源
| 数据 | 来源 | 格式 | 大小 |
|------|------|------|------|
| BTC/USDT 1H | Binance API | Parquet | ~20MB |
| ETH/USDT 1H | Binance API | Parquet | ~20MB |

### 2.2 数据快照
```yaml
snapshot:
  id: "DS_DGSF_20260201_V1"
  location: "projects/dgsf/data/snapshots/"
  checksum_file: "checksums.yaml"
  
verification:
  - "加载数据前验证 checksum"
  - "确保使用正确的 snapshot version"
```

### 2.3 数据获取脚本
```python
# projects/dgsf/scripts/fetch_data.py
from pathlib import Path
import ccxt
import pandas as pd

def fetch_ohlcv(symbol: str, timeframe: str = "1h"):
    """获取历史OHLCV数据"""
    exchange = ccxt.binance()
    # ... 实现
```

---

## 3. 随机性控制

### 3.1 种子设置
```python
# 实验开始时执行
import random
import numpy as np
import os

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
```

### 3.2 确定性操作
- ✅ 使用固定种子初始化所有随机过程
- ✅ 避免使用 `dict.keys()` 或 `set` 遍历（Python < 3.7 顺序不确定）
- ✅ 时间戳使用 UTC 标准化
- ✅ 浮点数运算使用 `np.float64`

---

## 4. 实验运行

### 4.1 单次实验
```bash
cd projects/dgsf
python scripts/run_experiment.py \
  --config configs/RESEARCH_1.yaml \
  --seed 42 \
  --output results/baseline/
```

### 4.2 消融实验
```bash
# 运行所有消融变体
python scripts/run_ablation.py \
  --config configs/RESEARCH_1.yaml \
  --ablation A1_grid_spacing \
  --output results/ablation_A1/
```

### 4.3 预期输出
```
results/
├── baseline/
│   ├── metrics.yaml
│   ├── backtest_report.html
│   └── equity_curve.png
├── ablation_A1/
│   ├── variant_0.005/
│   ├── variant_0.010/
│   ├── variant_0.015/
│   └── variant_0.020/
```

---

## 5. 验证清单

### 5.1 复现前检查
- [ ] Python 版本匹配
- [ ] 所有依赖已安装（版本匹配）
- [ ] 数据快照 checksum 验证通过
- [ ] 配置文件 `RESEARCH_1.yaml` 存在

### 5.2 复现后验证
- [ ] Sharpe Ratio 与基线差异 < 1%
- [ ] Max Drawdown 与基线差异 < 0.5%
- [ ] 交易次数完全一致
- [ ] 随机种子验证通过

---

## 6. 已知问题

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| TA-Lib 安装困难 | 部分技术指标不可用 | 使用 pandas-ta 替代 |
| Windows路径长度 | 超过260字符失败 | 使用短路径或启用长路径支持 |

---

## 7. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2026-02-01 | 初始版本 |

---

*本文档遵循 AI Workflow OS 的 PROJECT_DELIVERY_PIPELINE Stage 1 规范*
