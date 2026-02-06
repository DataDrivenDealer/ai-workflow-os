# spec_drift — DE7 Drift Report (2026-02-05)

## HIGH: QA runner referenced but missing
- Doc: `projects/dgsf/repo/docs/qa/de7_qa_loop.md` instructs `python tools/qa_de7_runner.py`
- Repo: no `projects/dgsf/repo/tools/qa_de7_runner.py` found
- Fix options:
  1) Implement the missing runner script
  2) Update the doc to point at existing commands / runners

## HIGH: “DE7 factor panel” refers to multiple incompatible artifacts
- A0 PanelTree expects: `data/a0/interim/de7_factor_panel.parquet` with `month_end` datetime and mixed-case factor names (`EP`, `BM`, ...)
  - Consumer: `projects/dgsf/repo/src/dgsf/paneltree/a0_runner.py`
- Fullwindow raw DE7 is: `data/full/de7_factor_panel.parquet` with internal `date` YYYYMM and schema-driven columns
  - Producer: `projects/dgsf/repo/src/dgsf/dataeng/de7_factor_panel.py`
- Recommendation: explicitly name two contracts in docs/configs:
  - `DE7_RAW_61` vs `DE7_A0_21`, plus a mapping table

## HIGH: Factor naming convention drift (doc vs code vs downstream)
- QA doc lists lowercase factor names but points at A0 parquet path.
- PanelTree A0 requires mixed-case A0 names (see `A0_FACTOR_LIST`).
- Recommendation: decide a stable naming policy and provide a rename/alias adapter.

## HIGH: Canonical time key drift
- A0: `month_end` datetime (canonical)
- Raw/fullwindow: `date` int YYYYMM (internal)
- DE8 likely uses `month_end` in YYYYMMDD int in its schema class.
- Recommendation: document canonical time key per artifact and provide explicit conversion utilities.

## HIGH: A0 builder doc/implementation mismatch on input paths
- `projects/dgsf/repo/src/dgsf/dataeng/de7_factor_panel_a0_builder.py` header comments vs loader path construction appear inconsistent (risk of `.../interim/interim/...`).
- Recommendation: align loader implementation and docstring; add a test covering path resolution.

## MED: QA doc references obsolete agent/tool names
- `projects/dgsf/repo/docs/qa/de7_qa_loop.md` mentions manual agents/tools that don’t match current OS subagent model.
- Recommendation: update to reference OS subagents (`repo_specs_retrieval`, `spec_drift`, etc.) or mark as “manual process not implemented”.
