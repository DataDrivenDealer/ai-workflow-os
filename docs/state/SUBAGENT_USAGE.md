# Subagent Usage Log & Metrics

> **文件用途**: 记录 Subagent 调用历史和使用统计
> **更新频率**: 每次 PLAN/EXECUTE 循环后
> **配置来源**: `configs/subagent_activation_policy.yaml`

---

## 📊 Usage Summary (Rolling 20 Tasks)

| Metric | Value | Status |
|--------|-------|--------|
| Total Tasks | 1 | — |
| Tasks with Subagent | 1 | — |
| Usage Rate | 100% | ✅ Above threshold |
| Alert Threshold | 40% | — |

---

## 📈 Breakdown by Subagent

| Subagent | Invocations | Last Used | Status |
|----------|-------------|-----------|--------|
| `repo_specs_retrieval` | 1 | 2026-02-05 | ✅ Active |
| `external_research` | 1 | 2026-02-05 | ✅ Active |
| `quant_risk_review` | 1 | 2026-02-05 | ✅ Active |
| `spec_drift` | 1 | 2026-02-05 | ✅ Active |

---

## 📋 Recent Invocation Log

<!-- 最新 20 条记录 -->

| Timestamp | Mode | Task ID | Subagents Invoked | Skip Reason | Evidence Paths |
|-----------|------|---------|-------------------|-------------|----------------|
| 2026-02-05T10:12:19 | VERIFY | phase-d-verify | repo_specs_retrieval | — | docs/subagents/runs/20260205_101219_repo_specs_retrieval/ |
| 2026-02-05T10:12:23 | VERIFY | phase-d-verify | external_research | — | docs/subagents/runs/20260205_101223_external_research/ |
| 2026-02-05T10:12:27 | VERIFY | phase-d-verify | quant_risk_review | — | docs/subagents/runs/20260205_101227_quant_risk_review/ |
| 2026-02-05T10:11:25 | VERIFY | phase-d-verify | spec_drift | — | docs/subagents/runs/20260205_101125_spec_drift/ |

---

## ⚠️ Skip Reason Analysis

| Reason | Count | Percentage |
|--------|-------|------------|
| — | — | — |

**Top 3 Skip Reasons**:
1. (无数据)
2. (无数据)
3. (无数据)

---

## 🔧 System Health

```yaml
last_updated: "2026-02-05T10:12:30Z"
health_status: "healthy"
issues: []
recommendations:
  - "external_research is placeholder - configure WEB_SEARCH_API_KEY for full functionality"
```

---

## 📝 Audit Trail Template

<!-- 每次 Plan/Execute 循环后追加以下格式 -->

```markdown
### Entry: {timestamp}

**Mode**: PLAN / EXECUTE  
**Session ID**: {session_id}  
**Task(s)**: {task_ids}  

**Subagents Invoked**:
- [ ] repo_specs_retrieval → {output_path or "not invoked"}
- [ ] external_research → {output_path or "not invoked"}
- [ ] quant_risk_review → {output_path or "not invoked"}
- [ ] spec_drift → {output_path or "not invoked"}

**Skip Justifications** (if any):
- {subagent_id}: {reason}

**Evidence Paths**:
- {path_1}
- {path_2}

---
```

---

### Entry: 2026-02-05

**Mode**: PLAN  
**Session ID**: (ad-hoc)  
**Task(s)**: SDF_DEV_001.1, STATE_ENGINE_INTEGRATION_001, SDF_FEATURE_ENG_001  

**Subagents Invoked**:
- [x] repo_specs_retrieval → docs/subagents/runs/20260205_plan_mode_stateengine_repo_specs_retrieval/
- [ ] external_research → not invoked (no multi-option DRS)
- [ ] quant_risk_review → not invoked (PLAN mode, no backtest/evaluation changes)
- [x] spec_drift → docs/subagents/runs/20260205_plan_mode_stateengine_spec_drift/

**Skip Justifications** (if any):
- external_research: not required (no competing solution options)
- quant_risk_review: not required (PLAN mode; execution not started)

**Evidence Paths**:
- docs/subagents/runs/20260205_plan_mode_stateengine_repo_specs_retrieval/SUMMARY.md
- docs/subagents/runs/20260205_plan_mode_stateengine_spec_drift/DRIFT_REPORT.md
- state/execution_queue.yaml
- tasks/active/SDF_FEATURE_ENG_001.md
- tasks/STATE_ENGINE_INTEGRATION_001.md

---

### Entry: 2026-02-05 (DE7 Plan)

**Mode**: PLAN  
**Session ID**: (ad-hoc)  
**Task(s)**: DE7 completion / audit / upgrade research plan

**Subagents Invoked**:
- [x] repo_specs_retrieval → docs/subagents/runs/20260205_plan_mode_de7_repo_specs_retrieval/
- [x] external_research → docs/subagents/runs/20260205_plan_mode_de7_external_research/
- [ ] quant_risk_review → not invoked (PLAN mode; no backtest/evaluation execution)
- [x] spec_drift → docs/subagents/runs/20260205_plan_mode_de7_spec_drift/

**Skip Justifications** (if any):
- quant_risk_review: not required for planning-only scope

**Evidence Paths**:
- docs/subagents/runs/20260205_plan_mode_de7_repo_specs_retrieval/SUMMARY.md
- docs/subagents/runs/20260205_plan_mode_de7_spec_drift/DRIFT_REPORT.md
- docs/subagents/runs/20260205_plan_mode_de7_external_research/RESEARCH_OPTIONS.md
- docs/plans/DE7_COMPLETION_AUDIT_UPGRADE_PLAN_20260205.md
- state/execution_queue.yaml

