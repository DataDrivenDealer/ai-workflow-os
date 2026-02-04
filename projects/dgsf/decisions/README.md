# DGSF Decision Registry

> 结构化决策存储，支持跨会话机构记忆。

---

## 目录结构

```
decisions/
├── README.md                              # 本文件
├── YYYY-MM-DD_DEC-{seq}_{slug}.yaml       # 单个决策记录
└── ...
```

## 决策 YAML Schema

```yaml
# 必填字段
id: DEC-{sequence}          # 全局唯一标识符，如 DEC-001
date: YYYY-MM-DD            # 决策日期
title: string               # 5-10 词简述
context: string             # 触发决策的背景

# 可选字段（有默认值）
urgency: low | normal | high                          # 默认 normal
reversibility: easily_reversible | with_effort | irreversible  # 默认 easily_reversible

# 必填：选项分析
options:
  - name: string
    chosen: boolean         # 仅一个为 true
    pros: [string]
    cons: [string]
    effort: string          # 如 "2h", "1d"
  - ...

# 必填：决策理由
rationale: string           # 2-3 句话解释为什么选择该选项

# 必填：关键假设
assumptions:
  - string                  # 若该假设失效，应重新审视决策

# 必填：成功标准
success_criteria:
  - string                  # 可验证的成功判据

# 可选：复查触发
review_trigger: string      # 什么时候应该重新审视此决策

# 可选：关联
related_experiments: [string]   # 如 ["t4_early_stopping"]
related_decisions: [string]     # 如 ["DEC-002"]
```

## 使用示例

### 创建新决策

```powershell
# 获取下一个序号
$count = (Get-ChildItem "projects/dgsf/decisions/*.yaml" -ErrorAction SilentlyContinue).Count
$seq = $count + 1
$id = "DEC-{0:D3}" -f $seq
$date = Get-Date -Format "yyyy-MM-dd"
$slug = "your-decision-slug"
$file = "projects/dgsf/decisions/${date}_${id}_${slug}.yaml"

# 创建文件（由 /dgsf_decision_log prompt 自动执行）
```

### 查询历史决策

```powershell
# 列出所有决策
Get-ChildItem "projects/dgsf/decisions/*.yaml" | Sort-Object Name

# 搜索特定主题
Get-ChildItem "projects/dgsf/decisions/*.yaml" | ForEach-Object {
    $content = Get-Content $_ -Raw
    if ($content -match "dropout|regularization") { $_.Name }
}

# 查看最近 5 个决策
Get-ChildItem "projects/dgsf/decisions/*.yaml" | Sort-Object Name -Descending | Select-Object -First 5
```

### 检查假设是否仍然有效

```powershell
# 提取所有假设
Get-ChildItem "projects/dgsf/decisions/*.yaml" | ForEach-Object {
    $yaml = Get-Content $_ -Raw | ConvertFrom-Yaml  # 需要 powershell-yaml 模块
    Write-Host "=== $($_.Name) ==="
    $yaml.assumptions | ForEach-Object { Write-Host "  - $_" }
}
```

## 与其他组件的关系

| 组件 | 关系 |
|------|------|
| `/dgsf_decision_log` | 该 prompt 负责创建决策文件 |
| `/dgsf_abort` | 当方向不可行时，abort 会触发新决策记录 |
| `/dgsf_research` | research 输出常作为决策的 context |
| `experiments/` | 决策通常关联一个或多个实验 |

## 维护指南

1. **序号冲突**：若多人同时创建决策，可能产生序号冲突。解决：使用 timestamp 作为后备唯一性保证
2. **决策更新**：不修改已有决策文件。若需修订，创建新决策并在 `related_decisions` 中引用旧决策
3. **归档**：超过 1 年的决策可移至 `decisions/archive/` 子目录

---

*Created as part of Kernel v3.2.0 — Institutional Memory Enhancement*
