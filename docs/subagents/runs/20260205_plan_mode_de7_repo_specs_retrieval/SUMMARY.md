# repo_specs_retrieval — DE7 Specs Summary (2026-02-05)

## Scope
DE7 factor panel (Fullwindow + A0 namespace) and downstream consumers (DE8, PanelTree A0, style spreads).

## Key Repo Artifacts (authoritative pointers)
- Fullwindow DE7 raw builder (61-factor, internal `date` = YYYYMM):
  - `projects/dgsf/repo/src/dgsf/dataeng/de7_factor_panel.py`
- A0 DE7 builder (canonical `month_end` datetime, drops extra date-like columns):
  - `projects/dgsf/repo/src/dgsf/dataeng/de7_factor_panel_a0_builder.py`
- Schema / factor universe constant:
  - `projects/dgsf/repo/src/dgsf/config/schema.py` (contains `FACTOR_RAW_PANEL_SCHEMA_V42`)
- Tests:
  - `projects/dgsf/repo/tests/dataeng/test_de7_factor_panel.py`
  - `projects/dgsf/repo/tests/dataeng/test_de7_style_spreads.py`
- A0 PanelTree consumer (expects `data/a0/interim/de7_factor_panel.parquet`):
  - `projects/dgsf/repo/src/dgsf/paneltree/a0_runner.py`
- QA workflow doc for DE7:
  - `projects/dgsf/repo/docs/qa/de7_qa_loop.md`
- Fullwindow configs (v1 + v2 fixed):
  - `projects/dgsf/repo/configs/de7_fullwindow.yaml`
  - `projects/dgsf/repo/configs/de7_fullwindow_v2.yaml`
  - `projects/dgsf/repo/configs/de7_fullwindow_v2_dev_test.yaml`
- A0 style spreads config:
  - `projects/dgsf/repo/configs/de7_style_spreads.yaml`

## Contract Table (disambiguated)
| Contract Name | Canonical Path | Time Key | Factor Columns | Notes |
|---|---|---|---|---|
| **DE7_RAW_61 (Fullwindow)** | `data/full/de7_factor_panel.parquet` (or `*_v2.parquet`) | `date` (int YYYYMM) | 61-factor set per `FACTOR_RAW_PANEL_SCHEMA_V42` (mostly lowercase) | Produced by `DE7FactorPanelBuilder` |
| **DE7_A0_21 (PanelTree A0)** | `data/a0/interim/de7_factor_panel.parquet` | `month_end` (datetime64[ns]) | 21 A0 factors (mixed-case names in `A0_FACTOR_LIST`) | Consumed by `paneltree/a0_runner.py` |

## Known Ambiguities / Open Questions
- A0 factor naming (e.g., `EP`, `BM`, `ROE`) vs raw schema naming (lowercase in v4.2): needs a documented mapping layer.
- QA doc references a QA runner script that is not present (see drift report).
- Rolling/other downstream modules sometimes refer to `data/full/de7_factor_panel.parquet` but may actually expect DE8 outputs; needs explicit contract pointers.
