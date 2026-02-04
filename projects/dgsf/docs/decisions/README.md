# DGSF 决策日志目录

此目录存储所有重要决策记录，按日期命名。

## 命名规范

`YYYY-MM-DD_{slug}.md`

示例：
- `2026-02-04_early-stopping-patience.md`
- `2026-02-05_dropout-rate-selection.md`

## 检索命令

```powershell
# 列出所有决策
Get-ChildItem "projects/dgsf/docs/decisions/" -Filter "*.md" | Sort-Object Name

# 搜索特定主题
Select-String -Path "projects/dgsf/docs/decisions/*.md" -Pattern "dropout"
```
