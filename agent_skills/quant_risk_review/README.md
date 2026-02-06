# Skill: quant_risk_review

> **ID**: `quant_risk_review`
> **Version**: 1.0.0
> **Purpose**: 检查量化策略代码和实验的常见风险

---

## 📋 Contract

### Input

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `target_files` | list | ✅ | 要审查的文件列表 |
| `review_type` | string | ❌ | "full" / "incremental" / "focused" (默认: "full") |
| `focus_areas` | list | ❌ | 聚焦区域 (默认: 全部) |
| `baseline_commit` | string | ❌ | 增量审查的基线 commit |
| `experiment_id` | string | ❌ | 关联的实验 ID |

### Focus Areas

| 区域 | 检查内容 |
|------|----------|
| `lookahead` | 前瞻偏差（使用未来数据） |
| `leakage` | 数据泄露（训练/测试数据混用） |
| `protocol` | 评估协议错误 |
| `reproducibility` | 可复现性问题 |

### Output

| 文件 | 描述 |
|------|------|
| `SUMMARY.md` | Verdict (pass/warn/fail), Risk Score, 关键问题 |
| `EVIDENCE.md` | 详细问题列表，包含代码片段和建议修复 |
| `CHECKLIST.md` | 按类别的检查清单 |
| `metadata.yaml` | 运行元数据 |

### Allowed Modes

- ✅ PLAN
- ✅ EXECUTE (限 Gate-E5 review)

---

## 🚀 Usage

### CLI

```bash
# Full review
python kernel/subagent_runner.py quant_risk_review \
    --files "projects/dgsf/repo/src/dgsf/backtest/engine.py" \
    --review-type full

# Focused review
python kernel/subagent_runner.py quant_risk_review \
    --files "projects/dgsf/repo/src/dgsf/sdf/model.py" \
    --review-type focused \
    --focus-areas lookahead leakage
```

---

## 📊 Risk Score Calculation

| 类别 | 权重 | 描述 |
|------|------|------|
| `lookahead_bias` | 3x | 前瞻偏差（最严重） |
| `data_leakage` | 3x | 数据泄露（最严重） |
| `evaluation_protocol` | 2x | 评估协议问题 |
| `reproducibility` | 1x | 可复现性问题 |

**Verdict 阈值**:
- **pass**: risk_score < 3
- **warn**: 3 ≤ risk_score < 6
- **fail**: risk_score ≥ 6

---

## 🔍 Detection Patterns

### Lookahead Bias

```python
# 检测模式
r"\.shift\(-"           # 负向 shift（使用未来数据）
r"future"               # 命名中包含 'future'
r"\.iloc\[-1\]"         # 访问最后一行（无时间上下文）
```

### Data Leakage

```python
# 检测模式
r"train_test_split.*shuffle.*True"  # 时间序列不应 shuffle
r"fit_transform.*test"               # 在测试数据上 fit
r"\.fit\(.*X\)"                       # Fitting 未使用 purging
```

### Evaluation Protocol

```python
# 检测模式
r"accuracy"                          # 使用 accuracy 而非风险调整指标
r"cross_val_score.*cv=\d"            # 标准 CV 而非 walk-forward
```

### Reproducibility

```python
# 检测模式
r"random_state\s*=\s*None"           # 未设置随机种子
r"np\.random\."                      # 直接使用 numpy random
```

---

## 📝 Example Output

### SUMMARY.md

```markdown
# Subagent Summary: Quant Risk Review

**Verdict**: ⚠️ **WARN**

**Risk Score**: 4/10

## Overview

| Metric | Value |
|--------|-------|
| Files Reviewed | 2 |
| Critical Issues | 1 |
| Warnings | 3 |

## Critical Issues

- **data_leakage**: fit_transform on test data (`engine.py:45`)

## Warnings

- **reproducibility**: No random seed set (`model.py:23`)
- **evaluation_protocol**: Using accuracy instead of risk-adjusted metrics (`eval.py:67`)
```

### CHECKLIST.md

```markdown
# Quant Risk Review Checklist

## 🔴 Data Leakage

**Status**: fail

**Issues**:
- [ ] Fix `engine.py:45`: fit_transform on test data

## ⚠️ Reproducibility

**Status**: warn

**Issues**:
- [ ] Fix `model.py:23`: No random seed set
```

---

## 🔗 Integration

### Gate-E5 (Risk Review)

当 EXECUTE MODE 检测到任务涉及以下区域时，**必须**调用 quant_risk_review：

- backtest
- data processing
- metrics calculation
- evaluation

**如果 verdict == "fail"**:
```
⛔ Gate-E5 失败 → STOP → ESCALATE → 返回 PLAN MODE
```

---

## ⚠️ Limitations

1. **静态分析**: 仅检测代码模式，不执行代码
2. **误报**: 某些模式可能在特定上下文中是正确的
3. **覆盖范围**: 不检测所有可能的风险

---

*quant_risk_review v1.0.0 — AI Workflow OS*
