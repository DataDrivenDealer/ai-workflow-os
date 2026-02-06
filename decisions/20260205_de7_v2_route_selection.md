# Decision Record: DE7 v2 Route Selection

**Date**: 2026-02-05
**Topic**: DE7 v2 路线选型与迁移策略

## Decision
- **短期执行**：Option 1（Surgical v2：`merge_asof` + schema normalization + QA gates）
- **中长期演进**：Option 2（上游 monthly PIT canonical panels）

## Rationale
- 已有 v2 配置 `configs/de7_fullwindow_v2.yaml`，指向 A0 interim，避免覆盖 frozen 产物并修复 100% NaN fundamentals。
- 中长期从源头统一 effective_date 与字段命名，减少适配分支并提升覆盖。

## Risks
- A0 interim 覆盖有限，ROE/ROA/NET_MARGIN 仍约 54% 缺失。
- 上游重建周期较长，短期仍需依赖 v2 适配路径。

## Rollback Plan
- 保留 `data/full/de7_factor_panel.parquet` 作为 frozen v1。
- 下游若出现异常，切回 v1 路径并暂停 v2 替换。

## Migration Strategy
- 新增输出 `data/full/de7_factor_panel_v2.parquet`，不覆盖 frozen。
- Shadow compare：行数/重复键/missingness/分布对比通过后切换下游消费路径。

## Acceptance Thresholds
- **Fail-fast**：关键因子组 100% NaN；`(ts_code, date)` 关键键重复。
- **Missingness 目标**（v2 预期）：EP/BM/CFP ~6.9%，DY ~26%，ROE/ROA/NET_MARGIN ~54%。
- **Warn**：fundamentals missingness > 80%；月度覆盖断档。

## References
- [docs/plans/DE7_COMPLETION_AUDIT_UPGRADE_PLAN_20260205.md](docs/plans/DE7_COMPLETION_AUDIT_UPGRADE_PLAN_20260205.md)
- [projects/dgsf/repo/configs/de7_fullwindow_v2.yaml](projects/dgsf/repo/configs/de7_fullwindow_v2.yaml)
- [projects/dgsf/repo/docs/qa/DE7_QUALITY_HARDENING_REPORT.md](projects/dgsf/repo/docs/qa/DE7_QUALITY_HARDENING_REPORT.md)
