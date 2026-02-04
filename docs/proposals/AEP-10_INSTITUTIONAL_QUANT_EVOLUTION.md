# AEP-10: Institutional Quantitative Research Evolution

**Proposal ID**: AEP-10  
**Title**: Institutional Quantitative Research Evolution — Domain-Driven Architecture Deepening  
**Status**: Active  
**Created**: 2026-02-04  
**Author**: Copilot (Architecture Evolution Agent)  
**Type**: Major Architecture Evolution  
**Affects**: All Layers + New Domain-Specific Capabilities  
**Supersedes**: Extends AEP-9  

---

## 0. Executive Summary

AEP-10 represents a **domain-driven deepening** of AI Workflow OS architecture, specifically targeting the unique requirements of **institutional-scale quantitative trading research**. While AEP-9 addressed organizational scalability, AEP-10 focuses on **domain excellence** — ensuring the system embodies cutting-edge practices in quantitative finance research, code quality, and long-term evolution.

**Core Thesis**: A workflow OS for quantitative trading must evolve beyond generic governance to become a **domain-aware research accelerator** that actively tracks, enforces, and propagates best practices from the frontiers of quantitative finance.

**Key Innovations:**

| Innovation | Problem Addressed | Impact |
|------------|-------------------|--------|
| **Quant Knowledge Base (QKB)** | No systematic frontier tracking | Continuous research intelligence |
| **Code Practice Registry (CPR)** | Best practices drift over time | Living standards enforcement |
| **Research Protocol Algebra (RPA)** | Experiment methodology gaps | Composable research workflows |
| **Strategic Debt Ledger (SDL)** | Technical debt accumulates silently | Visible, prioritized remediation |
| **Institutional Memory Graph (IMG)** | Decision rationale lost over time | Queryable organizational memory |
| **Adaptive Threshold Engine (ATE)** | Fixed thresholds vs regime changes | Context-aware success criteria |

---

## 1. Identified Tensions (Quantitative Finance Domain)

### 1.1 Frontier Research Tracking (Category F)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| F1 | No systematic literature tracking | Ad-hoc research | Miss important advances (e.g., new factor zoo findings) |
| F2 | No arxiv/SSRN integration | Manual discovery | Delayed awareness of relevant papers |
| F3 | No methodology evolution tracking | Static practices | Use outdated statistical tests |
| F4 | No competitive landscape awareness | Isolated development | Duplicate solved problems |

### 1.2 Code Best Practices (Category G)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| G1 | No quantitative code patterns | Generic Python patterns | Miss domain-specific idioms |
| G2 | No backtesting anti-patterns registry | Learn by failure | Repeat known mistakes (lookahead, survivorship) |
| G3 | No performance benchmark standards | Informal optimization | Inconsistent computational efficiency |
| G4 | No data pipeline standards | Project-specific | Reinvent data engineering per project |

### 1.3 Research Design Standards (Category H)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| H1 | No experiment design templates | Ad-hoc experiments | Missing controls, invalid comparisons |
| H2 | No statistical power guidelines | Underpowered tests | False discoveries |
| H3 | No multiple testing protocol enforcement | Manual correction | Overstated significance |
| H4 | No robustness check registry | Inconsistent validation | Cherry-picked results |

### 1.4 Development Process (Category I)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| I1 | No research sprint structure | Continuous flow | No natural checkpoints |
| I2 | No paper-to-code pipeline | Manual translation | Research-implementation gaps |
| I3 | No replication protocol | Ad-hoc reproduction | Unreliable baselines |
| I4 | No production handoff standard | Manual deployment | Research-production gap |

### 1.5 Long-term Evolution (Category J)

| ID | Tension | Current State | Impact |
|----|---------|---------------|--------|
| J1 | No strategic roadmap integration | Tactical fixes | Lack of coherent direction |
| J2 | No deprecation lifecycle | Eternal legacy | Accumulated cruft |
| J3 | No capability gap analysis | Reactive development | Miss systematic improvements |
| J4 | No institutional learning extraction | Implicit knowledge | Repeated mistakes |

---

## 2. Proposed Solutions

### 2.1 Quant Knowledge Base (QKB)

**Purpose**: Systematic tracking and integration of quantitative finance research frontiers.

```yaml
# configs/quant_knowledge_base.yaml
version: "1.0.0"

knowledge_domains:
  factor_research:
    description: "Factor zoo, risk premia, anomalies"
    key_sources:
      - type: "academic"
        name: "Journal of Finance"
        tracking: "rss"
      - type: "preprint"
        name: "SSRN Finance"
        tracking: "keyword_monitor"
        keywords: ["factor", "anomaly", "risk premium", "SDF"]
      - type: "practitioner"
        name: "AQR Research"
        tracking: "manual_quarterly"
    
    current_consensus:
      factor_replication_crisis:
        status: "ongoing"
        implication: "Use stringent thresholds (t > 3.0)"
        last_updated: "2026-01"
        sources: ["Harvey et al 2016", "Hou et al 2020"]
      
      machine_learning_in_finance:
        status: "maturing"
        implication: "Ensemble methods preferred, beware overfit"
        last_updated: "2026-01"
        sources: ["Gu et al 2020", "Avramov et al 2023"]
    
    methodology_standards:
      portfolio_sorts:
        current_best_practice: "Double-sorted with controls"
        avoid: "Single-sorted without controls"
        reference: "Fama-MacBeth with Newey-West"
      
      out_of_sample_testing:
        current_best_practice: "Purged walk-forward"
        avoid: "Simple train-test split"
        reference: "de Prado 2018 AFML"

  statistical_methods:
    description: "Statistical testing, inference, multiple testing"
    current_consensus:
      multiple_testing:
        standard: "Benjamini-Hochberg or Bonferroni"
        advanced: "Romano-Wolf stepdown"
        implication: "All factor discoveries must survive correction"
      
      bootstrap_methods:
        preferred: "Stationary bootstrap for time series"
        avoid: "IID bootstrap on correlated data"

  ml_in_finance:
    description: "Machine learning for alpha, risk, execution"
    current_consensus:
      tree_ensembles:
        status: "proven"
        caveat: "Prone to overfit without proper CV"
      
      deep_learning:
        status: "emerging"
        caveat: "Sample size requirements, interpretability"
      
      transformers:
        status: "experimental"
        applications: ["NLP for sentiment", "sequence modeling"]

# Knowledge refresh protocol
refresh_protocol:
  weekly_scan:
    - source: "arxiv.org/list/q-fin/new"
      filter: "keywords match knowledge_domains"
      action: "Add to review queue"
    
  monthly_review:
    - action: "Review accumulated papers"
    - action: "Update current_consensus if warranted"
    - action: "Propose methodology updates via spec_propose"
    
  quarterly_synthesis:
    - action: "Produce research_summary for each domain"
    - action: "Update project thresholds if consensus changed"
```

### 2.2 Code Practice Registry (CPR)

**Purpose**: Living registry of quantitative code best practices with enforcement.

```yaml
# configs/code_practice_registry.yaml
version: "1.0.0"

practices:
  # ---------------------------------------------------------------------------
  # DATA HANDLING
  # ---------------------------------------------------------------------------
  data_handling:
    DH-01:
      name: "Point-in-Time Correctness"
      severity: "critical"
      description: "All features must use only information available at prediction time"
      anti_patterns:
        - pattern: "df['future_return'] = df['price'].shift(-N)"
          violation: "Using future data in feature"
        - pattern: "df.fillna(df.mean())"
          violation: "Using full-sample statistics (lookahead)"
      correct_patterns:
        - pattern: "df['return'] = df['price'].pct_change().shift(1)"
        - pattern: "df.fillna(method='ffill')"
      enforcement: "lint_rule + code_review"
      reference: "Lopez de Prado, AFML Ch. 7"
    
    DH-02:
      name: "Survivorship Bias Prevention"
      severity: "critical"
      description: "Universe must include delisted securities"
      anti_patterns:
        - pattern: "universe = get_current_sp500()"
          violation: "Using current constituents for historical backtest"
      correct_patterns:
        - pattern: "universe = get_historical_constituents(date)"
      enforcement: "data_lineage_check"
      reference: "Elton et al 1996"
    
    DH-03:
      name: "Data Leakage Prevention"
      severity: "critical"
      description: "Prevent information leakage in cross-validation"
      anti_patterns:
        - pattern: "StandardScaler().fit_transform(X)"
          context: "Before train-test split"
          violation: "Fitting on full data including test set"
      correct_patterns:
        - pattern: "pipeline = Pipeline([('scaler', StandardScaler()), ...])"
        - pattern: "scaler.fit(X_train); X_test = scaler.transform(X_test)"
      enforcement: "lint_rule"
      reference: "Kaufman & Rosset 2012"

  # ---------------------------------------------------------------------------
  # BACKTESTING
  # ---------------------------------------------------------------------------
  backtesting:
    BT-01:
      name: "Transaction Cost Realism"
      severity: "high"
      description: "Include realistic transaction costs"
      required_parameters:
        - name: "slippage_bps"
          minimum: 5
          typical: 10
        - name: "commission_bps"
          minimum: 1
        - name: "market_impact"
          model: "square_root or linear"
      enforcement: "config_validation"
    
    BT-02:
      name: "Walk-Forward Validation"
      severity: "high"
      description: "Use expanding or rolling walk-forward for OOS testing"
      anti_patterns:
        - pattern: "train_test_split(X, y, test_size=0.2, shuffle=True)"
          violation: "Shuffling time series destroys temporal structure"
      correct_patterns:
        - pattern: "TimeSeriesSplit(n_splits=5)"
        - pattern: "PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(days=2))"
      enforcement: "experiment_template"
      reference: "de Prado 2018"
    
    BT-03:
      name: "Multiple Testing Correction"
      severity: "high"
      description: "Adjust significance for number of tests"
      formula: |
        adjusted_threshold = base_threshold / num_tests  # Bonferroni
        # Or use Benjamini-Hochberg for FDR control
      enforcement: "verify_prompt_check"

  # ---------------------------------------------------------------------------
  # PERFORMANCE PATTERNS
  # ---------------------------------------------------------------------------
  performance:
    PF-01:
      name: "Vectorized Operations"
      severity: "medium"
      description: "Prefer vectorized operations over loops"
      anti_patterns:
        - pattern: "for i in range(len(df)):"
          violation: "Row-wise iteration in pandas"
      correct_patterns:
        - pattern: "df.apply(func, axis=1)"
        - pattern: "np.vectorize(func)(array)"
        - pattern: "numba.jit decorated function"
      enforcement: "code_review + profiling"
    
    PF-02:
      name: "Memory-Efficient Data Types"
      severity: "medium"
      description: "Use appropriate dtypes for large datasets"
      recommendations:
        - "Use float32 instead of float64 when precision allows"
        - "Use category dtype for repeated strings"
        - "Use datetime instead of string dates"
      enforcement: "data_loading_template"

  # ---------------------------------------------------------------------------
  # MODEL DEVELOPMENT
  # ---------------------------------------------------------------------------
  modeling:
    MD-01:
      name: "Reproducibility Requirements"
      severity: "critical"
      description: "All experiments must be reproducible"
      requirements:
        - "Set random seed in config"
        - "Log all hyperparameters"
        - "Version control data and code"
        - "Record environment (requirements.txt hash)"
      enforcement: "experiment_template"
    
    MD-02:
      name: "Interpretability Documentation"
      severity: "medium"
      description: "Document model decisions and feature importance"
      requirements:
        - "SHAP or permutation importance for tree models"
        - "Coefficient significance for linear models"
        - "Attention visualization for transformers"
      enforcement: "experiment_checklist"

# Enforcement integration
enforcement:
  pre_commit:
    - practice_ids: ["DH-01", "DH-02", "DH-03", "BT-02"]
      hook: "hooks/check_quant_practices.py"
  
  code_review:
    - practice_ids: "all"
      checklist_template: "templates/code_review_checklist.md"
  
  experiment_gate:
    - practice_ids: ["BT-01", "BT-02", "BT-03", "MD-01"]
      gate: "configs/gates.yaml#experiment_completion"
```

### 2.3 Research Protocol Algebra (RPA)

**Purpose**: Composable, verifiable research workflow patterns.

```yaml
# configs/research_protocol_algebra.yaml
version: "1.0.0"

description: |
  Research Protocol Algebra defines composable research patterns
  that can be combined to create validated experiment workflows.

primitives:
  # Atomic research operations
  LOAD_DATA:
    inputs: ["data_source", "date_range", "universe"]
    outputs: ["dataset"]
    validates: ["point_in_time", "survivorship_free"]
    
  FEATURE_ENGINEER:
    inputs: ["dataset", "feature_config"]
    outputs: ["feature_matrix"]
    validates: ["no_lookahead", "finite_values"]
    
  TRAIN_MODEL:
    inputs: ["feature_matrix", "model_config", "cv_config"]
    outputs: ["model", "cv_metrics"]
    validates: ["purged_cv", "reproducible"]
    
  PREDICT:
    inputs: ["model", "feature_matrix"]
    outputs: ["predictions"]
    validates: ["chronological_order"]
    
  BACKTEST:
    inputs: ["predictions", "universe", "cost_config"]
    outputs: ["performance_metrics", "trade_log"]
    validates: ["realistic_costs", "capacity_check"]
    
  EVALUATE:
    inputs: ["performance_metrics"]
    outputs: ["evaluation_report"]
    validates: ["multiple_testing_corrected"]

compositions:
  # Compound research patterns
  FACTOR_DISCOVERY:
    description: "Standard factor discovery workflow"
    steps:
      - LOAD_DATA -> [universe_data, price_data]
      - FEATURE_ENGINEER(universe_data) -> [factors]
      - FOR each factor:
          - BACKTEST(factor) -> [factor_perf]
          - EVALUATE(factor_perf) -> [factor_report]
      - AGGREGATE(factor_reports) -> [discovery_summary]
      - APPLY_MTC(discovery_summary) -> [significant_factors]
    gates:
      - "At least 3 factors survive MTC"
      - "Average OOS Sharpe > 0.5"
    
  MODEL_ITERATION:
    description: "ML model development iteration"
    steps:
      - LOAD_DATA -> [dataset]
      - FEATURE_ENGINEER(dataset) -> [features]
      - HYPERPARAMETER_SEARCH(features, model_space) -> [best_params]
      - TRAIN_MODEL(features, best_params) -> [model, cv_metrics]
      - PREDICT(model, holdout_features) -> [predictions]
      - BACKTEST(predictions) -> [performance]
      - EVALUATE(performance) -> [report]
    gates:
      - "cv_metrics.sharpe > 1.0"
      - "performance.oos_sharpe / cv_metrics.sharpe > 0.8"
    
  ROBUSTNESS_CHECK:
    description: "Standard robustness battery"
    parallel_streams:
      - SUBPERIOD_ANALYSIS:
          split: ["2010-2015", "2015-2020", "2020-2025"]
          require: "Sharpe > 0.5 in all subperiods"
      - UNIVERSE_PERTURBATION:
          variations: ["drop_10_pct", "sector_exclusion", "size_quintile"]
          require: "Direction consistent across variations"
      - COST_SENSITIVITY:
          variations: ["1x_cost", "2x_cost", "3x_cost"]
          require: "Profitable at 2x cost"
      - PARAMETER_SENSITIVITY:
          variations: ["±20% on all params"]
          require: "Sharpe > 1.0 for 80% of variations"
    aggregate_gate: "Pass 3 of 4 streams"

templates:
  experiment_from_protocol:
    input: "protocol_name"
    output: "experiments/{exp_id}/config.yaml"
    auto_generates:
      - "Validation gates from protocol.gates"
      - "Lineage template from protocol.steps"
      - "Checkpoint structure from protocol.steps"
```

### 2.4 Strategic Debt Ledger (SDL)

**Purpose**: Track technical and research debt systematically.

```yaml
# configs/strategic_debt_ledger.yaml
version: "1.0.0"

description: |
  Strategic Debt Ledger tracks accumulated technical and research debt
  with explicit cost/benefit analysis and prioritized remediation paths.

debt_categories:
  technical_debt:
    code_quality:
      description: "Deferred refactoring, duplication, complexity"
      impact_metrics: ["maintenance_cost", "bug_rate", "onboarding_time"]
    
    test_coverage:
      description: "Missing tests, brittle tests, slow tests"
      impact_metrics: ["confidence", "deployment_risk", "iteration_speed"]
    
    infrastructure:
      description: "Manual processes, missing automation"
      impact_metrics: ["operational_cost", "error_rate", "scalability"]
  
  research_debt:
    methodology_debt:
      description: "Using outdated statistical methods"
      impact_metrics: ["false_discovery_risk", "replication_probability"]
      examples:
        - debt_id: "RD-001"
          description: "Some experiments use simple train-test split instead of purged CV"
          affected: ["t01_baseline", "t02_simple"]
          remediation: "Re-run with PurgedKFold"
          effort: "medium"
          priority: "high"
    
    replication_debt:
      description: "Results not yet replicated with alternative methods"
      impact_metrics: ["confidence", "robustness"]
    
    documentation_debt:
      description: "Missing decision rationale, methodology notes"
      impact_metrics: ["knowledge_transfer", "auditability"]
  
  evolutionary_debt:
    spec_drift:
      description: "Implementation diverged from specification"
      impact_metrics: ["governance_integrity", "predictability"]
    
    knowledge_decay:
      description: "Institutional knowledge not captured"
      impact_metrics: ["onboarding_cost", "repeated_mistakes"]

ledger_entries:
  # Example entries
  - debt_id: "TD-001"
    category: "technical_debt.code_quality"
    description: "Backtest module has high cyclomatic complexity"
    location: "projects/dgsf/repo/src/dgsf/backtest/engine.py"
    created: "2026-01-15"
    interest_rate: "high"  # Getting worse over time
    remediation:
      description: "Extract strategy execution into separate classes"
      effort_days: 3
      risk: "low"
    priority_score: 0.85
    status: "open"
    
  - debt_id: "RD-002"
    category: "research_debt.methodology_debt"
    description: "Factor significance not adjusted for multiple testing in early experiments"
    affected_experiments: ["t01", "t02", "t03"]
    created: "2026-01-20"
    interest_rate: "medium"
    remediation:
      description: "Retroactively apply Bonferroni correction and update conclusions"
      effort_days: 1
      risk: "medium"  # May invalidate some findings
    priority_score: 0.90
    status: "open"

prioritization:
  formula: |
    priority_score = (
      impact_weight * impact_score +
      interest_rate_weight * interest_rate_score +
      remediation_ease_weight * (1 - effort_normalized)
    )
  weights:
    impact: 0.4
    interest_rate: 0.3
    remediation_ease: 0.3
  
  review_cadence: "weekly"
  review_action: "Select top 3 items for sprint"

integration:
  sprint_planning:
    allocation: "20% of capacity reserved for debt remediation"
    selection: "Top priority_score items"
  
  evolution_signals:
    on_new_debt: "Log to evolution_signals with category=debt_accumulated"
    on_remediation: "Log to evolution_signals with category=debt_remediated"
  
  reporting:
    dashboard: "reports/debt_dashboard.md"
    metrics:
      - "Total debt items by category"
      - "Debt age distribution"
      - "Remediation velocity (items/week)"
      - "Interest accumulation trend"
```

### 2.5 Institutional Memory Graph (IMG)

**Purpose**: Queryable graph of decisions, rationale, and learnings.

```yaml
# configs/institutional_memory_graph.yaml
version: "1.0.0"

description: |
  Institutional Memory Graph captures and connects decisions, experiments,
  learnings, and failures to enable organizational learning and prevent
  repeated mistakes.

node_types:
  decision:
    required_fields:
      - decision_id
      - timestamp
      - description
      - rationale
      - alternatives_considered
      - decision_maker
    optional_fields:
      - outcome_assessment
      - lessons_learned
      - superseded_by
    
  experiment:
    required_fields:
      - experiment_id
      - hypothesis
      - methodology
      - outcome
    optional_fields:
      - unexpected_findings
      - follow_up_experiments
    
  learning:
    required_fields:
      - learning_id
      - source  # What generated this learning
      - insight
      - applicability
    optional_fields:
      - caveats
      - expiry  # When learning may no longer apply
    
  failure:
    required_fields:
      - failure_id
      - what_failed
      - root_cause
      - prevention_measure
    optional_fields:
      - detection_delay
      - impact_severity

edge_types:
  LED_TO:
    description: "A led to B (causal)"
  INVALIDATED:
    description: "A invalidated B"
  SUPPORTS:
    description: "A provides evidence for B"
  CONTRADICTS:
    description: "A contradicts B"
  SUPERSEDES:
    description: "A replaces B as current understanding"
  APPLIES_TO:
    description: "Learning A applies to context B"

queries:
  # Standard queries the system should support
  - name: "why_was_this_decided"
    description: "Trace decision rationale"
    pattern: "MATCH (d:decision {id: $id})-[:LED_TO|SUPPORTS*]-(context) RETURN context"
    
  - name: "what_did_we_learn_about"
    description: "Find learnings about a topic"
    pattern: "MATCH (l:learning)-[:APPLIES_TO]->(topic) WHERE topic.name CONTAINS $topic RETURN l"
    
  - name: "what_failed_before"
    description: "Find similar past failures"
    pattern: "MATCH (f:failure) WHERE f.what_failed CONTAINS $pattern RETURN f"
    
  - name: "is_this_still_valid"
    description: "Check if a learning/decision is still valid"
    pattern: "MATCH (n {id: $id})<-[:INVALIDATED|SUPERSEDES]-(newer) RETURN newer IS NULL"

auto_capture:
  from_decision_log:
    trigger: "New entry in decisions/*.yaml"
    action: "Create decision node, link to context"
    
  from_experiment:
    trigger: "Experiment marked complete"
    action: "Create experiment node, extract learnings"
    
  from_failure:
    trigger: "Experiment fails verification"
    action: "Create failure node after diagnosis"
    
  from_spec_change:
    trigger: "Spec modification committed"
    action: "Create decision node, link to affected nodes"

storage:
  format: "yaml_graph"  # Simple graph in YAML for now
  path: "state/memory_graph/"
  files:
    - "nodes/decisions/*.yaml"
    - "nodes/experiments/*.yaml"
    - "nodes/learnings/*.yaml"
    - "nodes/failures/*.yaml"
    - "edges.yaml"
  
  future_upgrade: "Neo4j or similar graph DB when scale requires"
```

### 2.6 Adaptive Threshold Engine (ATE)

**Purpose**: Context-aware success thresholds that adapt to market regimes.

```yaml
# configs/adaptive_threshold_engine.yaml
version: "1.0.0"

description: |
  Adaptive Threshold Engine allows success criteria to vary based on
  market regime, strategy type, and sample characteristics while
  maintaining overall governance integrity.

base_thresholds:
  # Default thresholds (from adapter.yaml)
  oos_sharpe: 1.5
  oos_is_ratio: 0.9
  max_drawdown: 0.20
  turnover: 2.0

regime_adjustments:
  market_volatility:
    description: "Adjust for VIX regime"
    regimes:
      low_vol:  # VIX < 15
        oos_sharpe: 1.8  # Stricter in calm markets
        rationale: "Alpha should be more apparent without noise"
      
      normal_vol:  # VIX 15-25
        oos_sharpe: 1.5  # Base case
      
      high_vol:  # VIX > 25
        oos_sharpe: 1.2  # More lenient in stressed markets
        max_drawdown: 0.25  # Accept higher drawdowns
        rationale: "Excess returns naturally lower, drawdowns higher"
    
    detection:
      indicator: "vix_regime_avg"
      lookback: "63d"  # ~3 months
  
  strategy_type:
    description: "Adjust by strategy characteristics"
    types:
      momentum:
        oos_sharpe: 1.3  # Lower bar due to known decay
        turnover: 4.0  # Accept higher turnover
      
      value:
        oos_sharpe: 1.2  # Lower bar due to longer horizon
        turnover: 0.5  # Require low turnover
      
      statistical_arbitrage:
        oos_sharpe: 2.0  # Higher bar for capacity-constrained
        turnover: 10.0  # Accept very high turnover
        
  sample_characteristics:
    description: "Adjust for sample size and period"
    adjustments:
      short_sample:  # < 3 years OOS
        oos_sharpe_multiplier: 1.2  # Require higher to compensate for variance
        confidence_penalty: 0.2
        
      long_sample:  # > 10 years OOS
        oos_sharpe_multiplier: 0.9  # Can accept slightly lower
        bonus: "survivability_evidence"

governance_constraints:
  # Hard floors that adaptive engine cannot breach
  absolute_minimums:
    oos_sharpe: 0.5  # Below this, strategy is noise
    oos_is_ratio: 0.5  # Below this, overfit is certain
    max_drawdown: 0.40  # Above this, too risky for any regime
  
  adjustment_limits:
    max_relaxation: 0.3  # Can relax threshold by at most 30%
    max_tightening: 0.5  # Can tighten by at most 50%
  
  human_override:
    required_for: "Adjustments beyond limits"
    approval_path: "/dgsf_spec_propose -> human review"

threshold_resolution:
  protocol: |
    1. Start with base_thresholds
    2. Detect current regime for each adjustment category
    3. Apply regime-specific multipliers
    4. Clamp to governance_constraints.absolute_minimums
    5. Clamp adjustments to adjustment_limits
    6. Return resolved thresholds with regime_context

  output_format:
    resolved_thresholds:
      oos_sharpe: 1.35  # After adjustments
      regime_context:
        volatility: "high_vol"
        strategy: "momentum"
        sample: "normal"
      adjustments_applied:
        - "high_vol: -0.3 to oos_sharpe"
        - "momentum: +0.0 to oos_sharpe (already adjusted)"

integration:
  verify_prompt:
    action: "Resolve thresholds before verification"
    display: "Show regime context in verification output"
    
  experiment_config:
    action: "Allow experiments to declare strategy_type"
    validation: "Must match one of defined types"
```

---

## 3. Kernel Instruction Enhancements

### 3.1 New Automatic Triggers

| Trigger Pattern | Auto-Invoke | Rationale |
|-----------------|-------------|-----------|
| "Best practice for X" question | QKB lookup | Ensure current knowledge |
| New experiment created | RPA template check | Enforce protocol compliance |
| Code review requested | CPR checklist | Apply practice registry |
| Threshold comparison fails | ATE regime check | Consider adaptive thresholds |
| Decision being made | IMG capture | Record for institutional memory |
| Debt observed | SDL logging | Track for prioritized remediation |

### 3.2 New Skills

| Skill | Purpose | Dependencies |
|-------|---------|--------------|
| `/dgsf_knowledge_sync` | Update QKB from recent sources | QKB |
| `/dgsf_practice_check` | Verify code against CPR | CPR |
| `/dgsf_protocol_design` | Design experiment using RPA | RPA |
| `/dgsf_debt_review` | Prioritize and plan debt remediation | SDL |
| `/dgsf_memory_query` | Query institutional memory | IMG |
| `/dgsf_threshold_resolve` | Get context-aware thresholds | ATE |

### 3.3 Enhanced Rule Expressions

```yaml
# New rules for quantitative research governance
rules:
  R7:
    id: "R7"
    name: "methodology_currency"
    expression: |
      WHEN experiment.methodology IN ["train_test_split", "k_fold"]
      AND experiment.data_type == "time_series"
      THEN WARN "Consider purged walk-forward validation"
      WITH reference = "configs/code_practice_registry.yaml#BT-02"
    
  R8:
    id: "R8"
    name: "multiple_testing_enforcement"
    expression: |
      WHEN experiment.factors_tested > 1
      AND NOT experiment.mtc_applied
      THEN BLOCK with "Multiple testing correction required"
      WITH reference = "configs/code_practice_registry.yaml#BT-03"
    
  R9:
    id: "R9"
    name: "robustness_requirement"
    expression: |
      WHEN experiment.claims.production_ready == true
      AND NOT experiment.robustness_battery_passed
      THEN BLOCK with "Robustness checks required before production claim"
      WITH reference = "configs/research_protocol_algebra.yaml#ROBUSTNESS_CHECK"
```

---

## 4. CMM Level 5 Characteristics (Target State)

| Dimension | CMM-4 (Current Target) | CMM-5 (AEP-10 Vision) |
|-----------|------------------------|------------------------|
| Knowledge Management | Evolution signals | Frontier tracking (QKB) + Memory graph (IMG) |
| Code Governance | REL rules | Living practice registry (CPR) + auto-enforcement |
| Research Methodology | Manual protocols | Composable protocol algebra (RPA) |
| Technical Health | Ad-hoc fixes | Strategic debt ledger (SDL) |
| Threshold Management | Static or parameterized | Adaptive engine (ATE) with regime awareness |
| Organizational Learning | Implicit | Explicit memory graph with queries |

---

## 5. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [x] Create AEP-10 proposal document
- [ ] Create initial QKB with factor research domain
- [ ] Create initial CPR with critical practices
- [ ] Update copilot-instructions.md with new triggers

### Phase 2: Protocol & Debt (Weeks 3-4)
- [ ] Implement RPA with 3 core protocols
- [ ] Create SDL with initial debt inventory
- [ ] Add debt review to sprint planning

### Phase 3: Memory & Adaptation (Weeks 5-6)
- [ ] Implement IMG node types and storage
- [ ] Create ATE with regime detection
- [ ] Integrate adaptive thresholds into verify prompt

### Phase 4: Skill Integration (Weeks 7-8)
- [ ] Create 6 new skill prompts
- [ ] Update adapter.yaml with new skills
- [ ] Add auto-triggers to copilot-instructions.md

### Phase 5: Testing & Refinement (Weeks 9-10)
- [ ] End-to-end testing of new capabilities
- [ ] Refine based on usage patterns
- [ ] Document institutional deployment guide

---

## 6. Success Metrics

| Metric | Current | Target (Post-AEP-10) |
|--------|---------|----------------------|
| Research methodology currency | Unmeasured | < 6 months behind frontier |
| Code practice compliance | Manual check | > 90% auto-checked |
| Experiment protocol coverage | Ad-hoc | > 80% use RPA templates |
| Technical debt visibility | Hidden | 100% tracked in SDL |
| Decision traceability | Partial | 100% in IMG |
| Threshold regime awareness | None | All verifications context-aware |

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| QKB staleness | Medium | Medium | Automated refresh protocols + manual quarterly review |
| CPR over-enforcement | Medium | Low | Start with critical-only, expand gradually |
| RPA rigidity | Low | Medium | Keep protocols composable, allow bypass with justification |
| SDL overhead | Medium | Low | Integrate into existing sprint flow |
| IMG scalability | Low | Low | Start with YAML, migrate to graph DB when needed |
| ATE complexity | Medium | Medium | Simple regime detection first, enhance iteratively |

---

*AEP-10 — Toward Institutional Quantitative Research Excellence*
