# Skill: external_research

> **ID**: `external_research`
> **Version**: 1.0.0
> **Purpose**: 执行网络搜索和文献研究，返回带引用链接的决策导向摘要

---

## 📋 Contract

### Input

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `research_question` | string | ✅ | 研究问题 |
| `context` | string | ❌ | 背景上下文 |
| `source_types` | list | ❌ | 来源类型，如 ["papers", "docs", "blogs"] |
| `time_range` | string | ❌ | 时间范围，如 "last_2_years" |
| `domain_focus` | string | ❌ | 领域聚焦，如 "quant_finance", "ml_ops" |

### Output

| 文件 | 描述 |
|------|------|
| `SUMMARY.md` | 决策导向的回答、建议、局限性 |
| `EVIDENCE.md` | 引用列表（标题、URL、类型、关键引用） |
| `metadata.yaml` | 运行元数据 |

### Allowed Modes

- ✅ PLAN only

### Allowed Tools

- web_search
- fetch_webpage
- academic_search

### Prohibited Tools

- code_execution
- file_write
- repo_modification

---

## 🚀 Usage

### CLI

```bash
python kernel/subagent_runner.py external_research \
    --question "purged walk-forward CV 的最佳实践是什么？" \
    --context "量化策略回测"
```

### Programmatic

```python
from kernel.subagent_runner import run_subagent
import argparse

args = argparse.Namespace(
    question="purged walk-forward CV 的最佳实践是什么？",
    context="量化策略回测",
    scope=None,
    keywords=None,
    files=None,
    review_type=None,
    focus_areas=None
)
result = run_subagent("external_research", args)
```

---

## ⚠️ Current Status: Placeholder

当前版本是占位实现。要启用实际的网络研究：

1. 配置 Web Search API（如 Bing, Google, 或学术 API）
2. 设置 `WEB_SEARCH_API_KEY` 环境变量
3. 更新 `kernel/subagent_runner.py` 中的 `ExternalResearchAgent` 实现

---

## 📝 Expected Output (Full Implementation)

### SUMMARY.md

```markdown
# Subagent Summary: External Research

**Research Question**: purged walk-forward CV 的最佳实践是什么？

**Context**: 量化策略回测

**Confidence**: high

## Recommendations

1. 使用 purged k-fold 避免数据泄露（López de Prado, 2018）
2. embargo 期应至少覆盖最大特征计算窗口
3. 在 combinatorial CV 中使用 purging + embargo

## Limitations

- 大部分研究基于股票市场，其他资产类别可能需要调整
- embargo 期的选择依赖于具体策略
```

### EVIDENCE.md

```markdown
# Evidence: External Research

## Citations

### Advances in Financial Machine Learning
**URL**: https://www.wiley.com/...
**Type**: book
**Key Quote**: "The embargo period should be at least as long as the maximum period used to compute features."
**Relevance**: 直接讨论 purged CV 的实现细节

### Cross-Validation in Finance (arXiv)
**URL**: https://arxiv.org/abs/...
**Type**: paper
**Key Quote**: "Standard CV leads to overfitting in time series..."
**Relevance**: 提供了理论基础
```

---

## 🔗 Integration

### Gate-P6 (DRS)

当 PLAN MODE 检测到决策存在多个可行选项时，调用 external_research 进行研究。

如果跳过，必须记录 `skip_justification`：

```yaml
skip_justification:
  gate: "Gate-P6"
  reason: "选项明确，Owner 已有偏好"
  owner_approved: true
```

---

## 🛡️ Restrictions

| 限制 | 值 |
|------|-----|
| 仅限 PLAN MODE | ✅ |
| 禁止在 EXECUTE MODE | ✅ |
| 原因 | 防止执行过程中规划偏离 |

---

*external_research v1.0.0 — AI Workflow OS*
