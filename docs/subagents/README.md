# Subagent Framework — AI Workflow OS

> **版本**: 1.0.0  
> **创建日期**: 2026-02-05  
> **目的**: 提供可委托的认知辅助能力，支持 PLAN MODE 和 EXECUTE MODE

---

## 概述

Subagent 是一组可独立调用的认知辅助模块，用于：
- **证据收集**：从本地仓库或外部资源获取精确引用
- **风险审查**：检查量化策略代码的常见问题
- **研究综合**：整理外部研究成决策导向的摘要

### 设计原则

1. **只读操作**：Subagent 不修改文件，只产生报告
2. **结构化输出**：每次运行产生 `SUMMARY.md` + `EVIDENCE.md`
3. **可追溯性**：输出包含元数据和时间戳
4. **模式感知**：不同操作模式有不同的 subagent 权限

---

## 可用 Subagent

| ID | 用途 | PLAN MODE | EXECUTE MODE |
|----|------|-----------|--------------|
| `repo_specs_retrieval` | 本地仓库与规范检索 | ✅ | ✅ (review only) |
| `external_research` | 外部网络研究 | ✅ | ❌ |
| `quant_risk_review` | 量化风险审查 | ✅ | ✅ (review only) |

---

## 快速使用

### 1. Repo & Specs Retrieval

在本地工作区搜索代码、配置和规范文件。

```powershell
# 搜索特征定义
python kernel/subagent_runner.py repo_specs_retrieval \
    --question "SDF_SPEC v3.1 中定义了哪些特征？" \
    --scope "specs/"

# 搜索特定关键词
python kernel/subagent_runner.py repo_specs_retrieval \
    --question "Where is the backtest engine defined?" \
    --keywords "BacktestEngine" "run_backtest"
```

**输出**:
- `SUMMARY.md`: 简短回答 + 关键发现
- `EVIDENCE.md`: 文件路径 + 行号 + 代码片段

### 2. External Research

执行网络搜索和文献研究。

```powershell
# 研究最佳实践
python kernel/subagent_runner.py external_research \
    --question "purged walk-forward cross-validation 的最佳实践是什么？" \
    --context "量化策略回测，需要避免数据泄露"
```

**输出**:
- `SUMMARY.md`: 决策导向的回答 + 引用
- `EVIDENCE.md`: 链接 + 关键引用

> ⚠️ **注意**: 当前为占位实现，需要配置 Web Search API。

### 3. Quant Risk Review

检查量化策略代码的常见风险。

```powershell
# 完整审查
python kernel/subagent_runner.py quant_risk_review \
    --files "projects/dgsf/repo/src/dgsf/backtest/engine.py" \
    --review-type full

# 聚焦审查
python kernel/subagent_runner.py quant_risk_review \
    --files "projects/dgsf/repo/src/dgsf/factors/alpha.py" \
    --focus-areas lookahead leakage
```

**输出**:
- `SUMMARY.md`: Verdict + 风险分数 + 关键问题
- `EVIDENCE.md`: 详细证据
- `CHECKLIST.md`: 可操作的检查清单

---

## 输出目录结构

所有 subagent 输出保存在 `docs/subagents/runs/`:

```
docs/subagents/runs/
├── 20260205_143000_repo_specs_retrieval/
│   ├── SUMMARY.md
│   ├── EVIDENCE.md
│   └── metadata.yaml
├── 20260205_144500_quant_risk_review/
│   ├── SUMMARY.md
│   ├── EVIDENCE.md
│   ├── CHECKLIST.md
│   └── metadata.yaml
└── ...
```

---

## 配置

### Subagent Registry

Subagent 定义在 `configs/subagent_registry.yaml`:

```yaml
subagents:
  repo_specs_retrieval:
    id: "repo_specs_retrieval"
    allowed_modes: [PLAN, EXECUTE]
    allowed_tools: [ripgrep, read_file]
    input_contract:
      required:
        - question: string
        - scope: string
    output_contract:
      summary_file: "SUMMARY.md"
      evidence_file: "EVIDENCE.md"
```

### 策略配置

在 PLAN MODE 和 EXECUTE MODE prompt 中通过以下参数控制：

```yaml
# PLAN MODE
SUBAGENT_POLICY: OFF | RESEARCH_ONLY | RESEARCH+REPO | FULL

# EXECUTE MODE
SUBAGENT_ALLOWED: YES  # 仅限 review gate
```

---

## 与 PLAN/EXECUTE MODE 的集成

### PLAN MODE 调用 Subagent

```markdown
## 触发条件

当检测到以下情况时，PLAN MODE 可自动调用 subagent：

| 触发条件 | 推荐 Subagent |
|----------|---------------|
| 需要验证 Spec 内容 | repo_specs_retrieval |
| 需要外部研究支持 | external_research |
| 规划涉及回测/策略代码 | quant_risk_review |

## 调用方式

1. 主 Agent 准备 subagent 任务规格
2. 调用 `python kernel/subagent_runner.py <subagent_id> ...`
3. 读取 `SUMMARY.md` 作为主要输入
4. 根据需要查看 `EVIDENCE.md`
```

### EXECUTE MODE 调用 Subagent

```markdown
## 限制

EXECUTE MODE 只能在 **Review Gate** 内调用 subagent：
- ✅ repo_specs_retrieval — 验证实现是否符合 Spec
- ✅ quant_risk_review — 检查代码风险
- ❌ external_research — 禁止（会导致规划偏离）

## 调用流程

1. 进入 Review Gate（验证阶段）
2. 调用 quant_risk_review 检查代码
3. 如果发现问题，触发 /dgsf_escalate
4. 不得基于 subagent 结果重新规划或添加任务
```

---

## 禁用 Subagent

### 完全禁用

在 PLAN MODE prompt 中设置：

```yaml
SUBAGENT_POLICY: OFF
```

### 仅禁用特定 Subagent

在 EXECUTE MODE prompt 中设置：

```yaml
SUBAGENT_ALLOWED: NO  # 禁止所有 subagent
```

或在 subagent_registry.yaml 中移除特定 subagent 的 mode 权限。

---

## 开发新 Subagent

### 1. 添加 Registry 条目

在 `configs/subagent_registry.yaml` 添加：

```yaml
subagents:
  my_new_subagent:
    id: "my_new_subagent"
    purpose: "描述用途"
    allowed_modes: [PLAN]
    allowed_tools: [...]
    input_contract:
      required: [...]
    output_contract:
      summary_file: "SUMMARY.md"
      evidence_file: "EVIDENCE.md"
```

### 2. 实现 Agent 类

在 `kernel/subagent_runner.py` 添加：

```python
class MyNewSubagent(SubagentBase):
    def run(self, **kwargs) -> dict:
        # 实现逻辑
        self.write_summary(summary)
        self.write_evidence(evidence)
        return {"status": "success"}

# 注册
SUBAGENT_CLASSES["my_new_subagent"] = MyNewSubagent
```

### 3. 测试

```powershell
python kernel/subagent_runner.py my_new_subagent --help
```

---

## FAQ

**Q: Subagent 会修改我的代码吗？**

A: 不会。所有 subagent 都是只读操作，只产生报告文件。

**Q: 如何清理旧的 subagent 输出？**

A: 输出默认保留 30 天，可通过 `policies.archive_after_days` 配置。手动清理：
```powershell
Remove-Item -Recurse docs/subagents/runs/*
```

**Q: external_research 需要什么 API？**

A: 当前为占位实现。完整实现需要：
- Bing Search API 或 Google Custom Search
- 可选：arXiv API、SSRN API

---

*文档最后更新: 2026-02-05*
