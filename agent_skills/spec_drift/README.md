# Skill: spec_drift

> **ID**: `spec_drift`
> **Version**: 1.0.0
> **Purpose**: 检测 Spec 与实现之间的漂移，分类问题并提供建议

---

## 📋 Contract

### Input

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `scope` | string | ❌ | Spec 文件范围 (默认: "specs/") |
| `compare_to` | string | ❌ | 实现代码范围 (默认: "projects/dgsf/repo/src/") |
| `spec_files` | list | ❌ | 指定的 Spec 文件列表 |
| `check_cross_refs` | bool | ❌ | 是否检查 Spec 之间的交叉引用 (默认: true) |

### Output

| 文件 | 描述 |
|------|------|
| `SUMMARY.md` | 漂移分类统计、建议 |
| `EVIDENCE.md` | 详细漂移列表，包含 Spec 摘录和实现对比 |
| `metadata.yaml` | 运行元数据 |

### Drift Categories

| 类别 | 描述 | 建议行动 |
|------|------|----------|
| `SPEC_LAG` | Spec 落后于实现 | 更新 Spec |
| `CODE_DRIFT` | 实现偏离 Spec | 修复代码或更新 Spec |
| `MUTUAL_INCONSISTENCY` | Specs 之间存在冲突 | 调和 Specs |

### Allowed Modes

- ✅ PLAN only

---

## 🚀 Usage

### CLI

```bash
# 检测所有 Spec 漂移
python kernel/subagent_runner.py spec_drift \
    --scope "specs/" \
    --compare-to "projects/dgsf/repo/src/"

# 检测特定 Spec
python kernel/subagent_runner.py spec_drift \
    --spec-files "specs/sdf_spec_v3.1.yaml" "specs/feature_registry.yaml"
```

---

## 📊 Detection Logic

### SPEC_LAG Detection

```python
# Spec 中定义了接口，但实现中有额外功能
# 或实现有新的参数/方法未在 Spec 中记录

for contract in spec.contracts:
    impl = find_implementation(contract)
    if impl.has_undocumented_features():
        drift = SPEC_LAG
```

### CODE_DRIFT Detection

```python
# 实现与 Spec 定义不一致
# 参数类型、返回值、行为不匹配

for contract in spec.contracts:
    impl = find_implementation(contract)
    if not impl.matches(contract):
        drift = CODE_DRIFT
```

### MUTUAL_INCONSISTENCY Detection

```python
# Specs 之间交叉引用冲突
# 或定义的接口/数据结构冲突

for spec_a, spec_b in spec_pairs:
    if has_conflict(spec_a, spec_b):
        drift = MUTUAL_INCONSISTENCY
```

---

## 📝 Example Output

### SUMMARY.md

```markdown
# Subagent Summary: Spec Drift Analysis

**Total Drift Items**: 5

## By Category

| Category | Count |
|----------|-------|
| SPEC_LAG | 2 |
| CODE_DRIFT | 2 |
| MUTUAL_INCONSISTENCY | 1 |

## Recommendations

1. **[SPEC_LAG]** specs/sdf_spec_v3.1.yaml: 
   实现中新增了 `momentum_60d` 特征，Spec 未更新
   → 建议: 更新 Spec 添加新特征定义

2. **[CODE_DRIFT]** specs/data_pipeline.yaml vs src/data/loader.py:
   Spec 要求 `date_column` 参数，实现使用 `timestamp_column`
   → 建议: 统一命名

3. **[MUTUAL_INCONSISTENCY]** specs/sdf_spec_v3.1.yaml vs specs/evaluation.yaml:
   SDF_SPEC 定义 Sharpe 阈值为 1.5，Evaluation Spec 定义为 1.2
   → 建议: 调和两个 Spec 的阈值定义
```

### EVIDENCE.md

```markdown
# Evidence: Spec Drift Analysis

## SPEC_LAG

### specs/sdf_spec_v3.1.yaml

**Contract**: feature_definitions
**Expected in Spec**: 12 features
**Found in Implementation**: 15 features

#### Spec Content
```yaml
feature_definitions:
  - name: momentum_20d
  - name: momentum_5d
  ...
```

#### Implementation Content (src/dgsf/features.py)
```python
FEATURES = [
    "momentum_20d",
    "momentum_5d",
    "momentum_60d",  # Not in Spec!
    ...
]
```

#### Discrepancy
实现中存在 3 个未在 Spec 中定义的特征

---

## CODE_DRIFT

### specs/data_pipeline.yaml vs src/data/loader.py

**Contract**: DataLoader.load()
**Spec Signature**: `load(date_column: str, ...)`
**Impl Signature**: `load(timestamp_column: str, ...)`

...
```

---

## 🔗 Integration

### Gate-P1 (Specs Scan)

当检测到以下条件时，PLAN MODE 调用 spec_drift：

- 跨层依赖（data↔factor↔sdf↔evaluation）
- 疑似版本不匹配
- 用户明确请求漂移检查

### 与 /dgsf_spec_triage 配合

spec_drift 输出可以作为 /dgsf_spec_triage 的输入：

```
spec_drift → 检测漂移 → 输出分类
     ↓
/dgsf_spec_triage → 分诊每个漂移项 → 确定处理方式
     ↓
/dgsf_spec_propose → 提出修改建议
     ↓
/dgsf_spec_commit → 应用批准的修改
```

---

## ⚠️ Limitations

1. **静态分析**: 基于模式匹配，可能有误报
2. **复杂契约**: 无法理解复杂的业务逻辑
3. **动态行为**: 不检测运行时行为差异

---

*spec_drift v1.0.0 — AI Workflow OS*
