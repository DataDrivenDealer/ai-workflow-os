# Agent Skills

> **用途**: 可复用的 Agent Skills，每个 Skill 包含契约定义、实现和示例
> **Runner**: `kernel/subagent_runner.py`
> **Registry**: `configs/subagent_registry.yaml`

---

## 📁 Directory Structure

```
agent_skills/
├── README.md                    # 本文件
├── repo_specs_retrieval/        # 本地仓库与规范检索
│   ├── README.md               # 契约定义
│   └── examples/               # 使用示例
├── external_research/           # 外部网络研究
│   ├── README.md               # 契约定义
│   └── examples/               # 使用示例
├── quant_risk_review/          # 量化风险审查
│   ├── README.md               # 契约定义
│   └── examples/               # 使用示例
└── spec_drift/                 # Spec 漂移检测
    ├── README.md               # 契约定义
    └── examples/               # 使用示例
```

---

## 🚀 Quick Start

### 运行 Subagent

```bash
# 列出可用的 Subagents
python kernel/subagent_runner.py --list

# Repo & Specs Retrieval
python kernel/subagent_runner.py repo_specs_retrieval \
    --question "SDF_SPEC v3.1 中定义了哪些特征？" \
    --scope "specs/"

# External Research
python kernel/subagent_runner.py external_research \
    --question "purged walk-forward CV 的最佳实践" \
    --context "量化策略回测"

# Quant Risk Review
python kernel/subagent_runner.py quant_risk_review \
    --files "projects/dgsf/repo/src/dgsf/backtest/engine.py" \
    --review-type full

# Spec Drift Detection
python kernel/subagent_runner.py spec_drift \
    --scope "specs/" \
    --compare-to "projects/dgsf/repo/src/"
```

### 输出位置

所有 Subagent 输出到：
```
docs/subagents/runs/<timestamp>_<subagent_id>/
├── SUMMARY.md       # 主 Agent 消费的简短摘要
├── EVIDENCE.md      # 详细证据（路径、行号、引用）
├── CHECKLIST.md     # 仅 quant_risk_review
└── metadata.yaml    # 运行元数据
```

---

## 📋 Available Skills

| Skill ID | 用途 | 允许模式 |
|----------|------|----------|
| `repo_specs_retrieval` | 本地仓库与规范检索 | PLAN, EXECUTE |
| `external_research` | 外部网络研究 | PLAN only |
| `quant_risk_review` | 量化风险审查 | PLAN, EXECUTE |
| `spec_drift` | Spec 漂移检测 | PLAN only |

---

## 🔗 Integration with Gates

| Gate | 触发条件 | 调用的 Skill |
|------|----------|-------------|
| Gate-P1 | 存在歧义、跨层依赖 | `repo_specs_retrieval`, `spec_drift` |
| Gate-P6 | 决策 ≥2 选项 | `external_research` |
| Gate-E0 | 任务有 RequiredSubagents | 按任务定义 |
| Gate-E5 | 涉及 backtest/data/metrics | `quant_risk_review` |

---

## 📝 Adding a New Skill

1. 在 `configs/subagent_registry.yaml` 添加配置
2. 在 `kernel/subagent_runner.py` 添加实现类
3. 创建 `agent_skills/<skill_id>/README.md` 契约文档
4. 添加示例到 `agent_skills/<skill_id>/examples/`
5. 更新 `SUBAGENT_CLASSES` 字典

---

*Agent Skills v1.0 — AI Workflow OS*
