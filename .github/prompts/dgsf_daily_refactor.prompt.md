---
description: Trigger daily code refactoring and cleanup
mode: agent
triggers:
  - "每日重构"
  - "daily refactor"
  - "代码整理"
  - "/dgsf_daily_refactor"
---

# DGSF Daily Refactor

> **目的**: 自动化代码清理和轻量级重构
> **工具**: `tools/daily_refactor/run.py`
> **输出**: `docs/refactor/YYYY-MM-DD/REPORT.md`

---

## 🎯 设计原则

1. **仅重构，不改行为** — 所有变换必须保持语义等价
2. **Dry-run 默认** — 需显式 `--apply` 才实际修改
3. **增量处理** — 仅处理自上次重构后变更的文件
4. **可审计** — 所有操作生成报告

---

## 📥 INPUTS

```yaml
since: string        # 对比基准 (default: HEAD~1)
safe_only: bool      # 仅安全变换 (default: true)
apply: bool          # 实际应用 (default: false)
commit: bool         # 自动提交 (default: false)
```

---

## 🔧 变换分类

### 安全变换 (Safe) — 自动应用

| 变换 | 工具 | 描述 |
|------|------|------|
| 代码格式化 | Black | 统一代码风格 |
| Import 排序 | isort | 按规范排列 imports |
| 移除尾部空白 | Ruff | 清理 trailing whitespace |
| 修复行尾 | Ruff | 统一 line endings |

### 中等变换 (Moderate) — 需确认

| 变换 | 工具 | 描述 |
|------|------|------|
| 移除未使用 import | Ruff F401 | 删除未引用的 import |
| 简化布尔返回 | Ruff | `if x: return True else: return False` → `return x` |
| 使用 f-string | Ruff | `"%s" % x` → `f"{x}"` |

### 高风险变换 (Risky) — 需 `--include-risky`

| 变换 | 工具 | 描述 |
|------|------|------|
| 移除未使用变量 | Ruff F841 | 可能误删有意的占位符 |
| 内联单次使用变量 | 手动 | 可能降低可读性 |
| 简化条件 | 手动 | 可能改变边界行为 |

---

## 📋 EXECUTION PROTOCOL

### Step 1: 检测变更

```bash
# 自动检测自上次 commit 的变更
python tools/daily_refactor/run.py

# 或指定基准
python tools/daily_refactor/run.py --since origin/main
```

**输出**:
```
📂 Detecting changed files...
Found 5 file(s) to process:
  - kernel/config.py
  - kernel/state_store.py
  ...
```

### Step 2: Dry-Run 预览

```bash
python tools/daily_refactor/run.py --verbose
```

**输出**:
```
🔧 Running transformations...
  [1/4] Black (formatter)...
  [2/4] isort (import sorter)...
  [3/4] Ruff (linter + fix)...
  [4/4] Pyright (type check, report only)...

📄 Reports generated in: docs/refactor/2026-02-05/

💡 This was a DRY-RUN. To apply changes, run with --apply
```

### Step 3: 审查报告

```
READ docs/refactor/YYYY-MM-DD/REPORT.md
READ docs/refactor/YYYY-MM-DD/RISKS.md
```

检查：
- [ ] 变更文件列表合理
- [ ] 无高风险警告
- [ ] 错误数为 0

### Step 4: 应用变更

```bash
# 仅安全变换
python tools/daily_refactor/run.py --apply --safe-only

# 包含中等变换
python tools/daily_refactor/run.py --apply

# 自动提交
python tools/daily_refactor/run.py --apply --commit
```

### Step 5: 验证

```bash
# 运行测试确保无回归
pytest kernel/tests -x -q
```

---

## 🖥️ VS Code Task 触发

已配置 VS Code Task，可通过以下方式触发：

1. `Ctrl+Shift+P` → "Tasks: Run Task"
2. 选择 "Daily Refactor"
3. 或使用快捷键（如已配置）

---

## ⏰ 自动化调度

### GitHub Actions Nightly

每天 UTC 02:00 自动运行并创建 PR：

```yaml
# .github/workflows/nightly_refactor.yaml
on:
  schedule:
    - cron: '0 2 * * *'
```

### 本地 Cron (可选)

```bash
# 添加到 crontab
0 9 * * * cd /path/to/workspace && python tools/daily_refactor/run.py --apply --commit
```

---

## 📊 COMPLIANCE 集成

每次运行后更新 `docs/state/COMPLIANCE_METRICS.md`:

```markdown
## Daily Refactor Cadence

| Date | Files Changed | Safe Transforms | Status |
|------|---------------|-----------------|--------|
| 2026-02-05 | 5 | 23 | ✅ Clean |
```

---

## ⚠️ 故障处理

### 工具未安装

```
Command not found: black
```

**解决**:
```bash
pip install black isort ruff
```

### 变换冲突

如果 Black 和 isort 产生冲突：

```bash
# 使用 black profile
isort --profile=black .
```

### 测试失败

如果重构后测试失败：

```bash
# 回滚所有变更
git checkout -- .

# 或查看具体哪个变换导致问题
git diff
```

---

## 📁 输出 Artifacts

```
docs/refactor/
  2026-02-05/
    REPORT.md      # 详细变更报告
    DIFFSTAT.txt   # Git diffstat
    RISKS.md       # 风险评估
```

---

## 🔗 与 Pair Programming 集成

Daily Refactor 产生的代码变更**不需要**完整的 Pair Programming Review，因为：

1. 所有变换都是语义保持的
2. 工具自动验证正确性
3. 测试确保无回归

但如果 Daily Refactor 触发了 **Risky** 变换，建议运行 Review：

```
IF refactor.includes_risky:
    INVOKE /dgsf_pair_review WITH:
        task_id: "DAILY_REFACTOR_{date}"
        changed_files: refactor.changed_files
```

---

*Daily Refactor — Keep the Codebase Clean*
