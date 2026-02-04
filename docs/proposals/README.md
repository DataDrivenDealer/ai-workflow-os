# Evolution Proposals Index

**Last Updated**: 2026-02-04

本目录包含 Copilot Workflow OS 的架构演化提案（Architecture Evolution Proposals, AEPs）。

---

## Active Proposals

| AEP | Title | Status | Priority | Target Phase |
|-----|-------|--------|----------|--------------|
| [AEP-6](AEP-6_meta_evolution_monitoring.md) | 元演化监控 | Draft | P2 | Phase 2 |
| [AEP-7](AEP-7_org_scaling.md) | 组织扩展支持 | Draft | P3 | Phase 4 |
| [AEP-8](AEP-8_rule_expressiveness.md) | 规则表达力增强 | Draft | P2 | Phase 3 |

## Supporting Documents

| Document | Purpose |
|----------|---------|
| [TENSION_ANALYSIS.md](TENSION_ANALYSIS.md) | 张力分析完整报告（2026-02-04 认知探索输出） |
| [EVOLUTION_ROADMAP.md](EVOLUTION_ROADMAP.md) | 四阶段实施路线图 |

## Historical AEPs

| AEP | Title | Status | Outcome |
|-----|-------|--------|---------|
| AEP-1 | Kernel-Project Decoupling | Implemented | v4.0.0 引入 Adapter Layer |
| AEP-2 | Evolution Closed-loop Automation | Implemented | evolution_policy.yaml |
| AEP-3 | Evolution Effectiveness Tracking | Implemented | 30d measurement window |
| AEP-4 | Tiered Authority Levels | Implemented | Level 0-3 体系 |
| AEP-5 | System Health Monitoring | Implemented | health_metrics.yaml |

---

## Proposal Lifecycle

```
Draft → Review → Approved → Implementing → Implemented → Verified
                    ↓
               Rejected/Deferred
```

## How to Contribute

1. 识别张力或改进机会
2. 创建 `AEP-{N}_{short_name}.md`
3. 填写标准模板（动机、提案、接口变更、实施路线、风险）
4. 提交 PR 并请求 Platform Engineer 审阅

---

*See [configs/evolution_policy.yaml](../../configs/evolution_policy.yaml) for formal evolution process.*
