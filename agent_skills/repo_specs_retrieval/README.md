# Skill: repo_specs_retrieval

> **ID**: `repo_specs_retrieval`
> **Version**: 1.0.0
> **Purpose**: 在本地工作区搜索代码、配置和规范文件，提供精确的文件路径和行号引用

---

## 📋 Contract

### Input

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `question` | string | ✅ | 要回答的问题 |
| `scope` | string | ❌ | 搜索范围，如 "specs/", "kernel/", "." (默认: ".") |
| `file_patterns` | list | ❌ | 文件模式，如 ["*.yaml", "*.py"] |
| `keywords` | list | ❌ | 关键词列表（自动从问题提取） |

### Output

| 文件 | 描述 |
|------|------|
| `SUMMARY.md` | 简短摘要，包含答案、置信度、关键发现 |
| `EVIDENCE.md` | 详细证据，包含文件路径、行号、代码片段 |
| `metadata.yaml` | 运行元数据 |

### Allowed Modes

- ✅ PLAN
- ✅ EXECUTE (限 review gate)

### Allowed Tools

- ripgrep (rg)
- file_tree (ls, find)
- read_file
- grep_search

### Prohibited Tools

- web_fetch
- external_api
- code_execution
- file_write

---

## 🚀 Usage

### CLI

```bash
python kernel/subagent_runner.py repo_specs_retrieval \
    --question "SDF_SPEC v3.1 中定义了哪些特征？" \
    --scope "specs/"
```

### Programmatic

```python
from kernel.subagent_runner import run_subagent
import argparse

args = argparse.Namespace(
    question="SDF_SPEC v3.1 中定义了哪些特征？",
    scope="specs/",
    keywords=None,
    files=None,
    review_type=None,
    context=None,
    focus_areas=None
)
result = run_subagent("repo_specs_retrieval", args)
print(result["output_dir"])
```

---

## 📝 Example Output

### SUMMARY.md

```markdown
# Subagent Summary: Repo & Specs Retrieval

**Question**: SDF_SPEC v3.1 中定义了哪些特征？

**Confidence**: high

## Key Findings

- Found **12** matches across **3** files.
- `specs/sdf_spec_v3.1.yaml`: 8 matches
- `specs/feature_registry.yaml`: 3 matches
- `docs/specs/SDF_SPEC.md`: 1 match

## Answer

Based on the search results, relevant content was found in the files listed above.
```

### EVIDENCE.md

```markdown
# Evidence: Repo & Specs Retrieval

## File References

### `specs/sdf_spec_v3.1.yaml`

**Line 12** (keyword: `feature`)
```
feature_definitions:
```

**Line 15** (keyword: `feature`)
```
  - name: momentum_20d
```

...
```

---

## 🔗 Integration

### Gate-P1 (Specs Scan)

当 PLAN MODE 检测到以下条件时，自动调用：

- 存在 Spec 歧义
- 跨层依赖（data↔factor↔sdf）
- 疑似 Spec 漂移

### /dgsf_run_subagent

```markdown
用户: 运行 subagent repo_specs_retrieval 检查 SDF_SPEC 中的特征定义

Copilot: 
## ⏳ 正在调用 repo_specs_retrieval...

**输出目录**: docs/subagents/runs/20260205_143000_repo_specs_retrieval/

[执行中...]

## ✅ 完成

**Confidence**: high
**Findings**: 12 matches across 3 files
```

---

## ⚠️ Limitations

1. **搜索深度**: 每个关键词最多返回 50 个结果
2. **文件大小**: 跳过超过 1MB 的文件
3. **二进制文件**: 自动跳过
4. **ripgrep 依赖**: 如果 ripgrep 不可用，使用 Python 实现（较慢）

---

*repo_specs_retrieval v1.0.0 — AI Workflow OS*
