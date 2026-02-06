```prompt
---
description: Standard wrapper for invoking subagents with proper contracts and output handling
mode: agent
triggers:
  - "运行 subagent"
  - "调用 subagent"
  - "run subagent"
  - "invoke subagent"
---

# DGSF RUN SUBAGENT

> **用途**: 标准化 Subagent 调用包装器
> **配置来源**: `configs/subagent_registry.yaml`
> **输出目录**: `docs/subagents/runs/<timestamp>_<subagent_id>/`

---

## 📋 AVAILABLE SUBAGENTS（可用 Subagent）

| Subagent ID | 用途 | 允许模式 |
|-------------|------|----------|
| `repo_specs_retrieval` | 本地仓库与规范检索 | PLAN, EXECUTE |
| `external_research` | 外部网络研究 | PLAN only |
| `quant_risk_review` | 量化风险审查 | PLAN, EXECUTE |
| `spec_drift` | Spec 漂移检测 | PLAN only |

---

## 🚀 INVOCATION PROTOCOL（调用协议）

### Step 1: 验证调用权限

```
READ current_mode FROM context (PLAN or EXECUTE)
READ subagent_id FROM user request

LOAD configs/subagent_registry.yaml
subagent = registry.subagents[subagent_id]

IF current_mode NOT IN subagent.allowed_modes:
    OUTPUT: "⛔ 当前模式 ({current_mode}) 不允许调用 {subagent_id}"
    OUTPUT: "允许的模式: {subagent.allowed_modes}"
    STOP
```

### Step 2: 准备输入参数

```markdown
## 🔧 Subagent 调用准备

**Subagent**: {subagent_id}
**版本**: {subagent.version}
**用途**: {subagent.purpose}

### 输入参数

根据 {subagent_id} 的 input_contract，需要以下参数：

**必填**:
{FOR field IN subagent.input_contract.required}
- `{field.name}`: {field.type} — {field.description}
{/FOR}

**可选**:
{FOR field IN subagent.input_contract.optional}
- `{field.name}`: {field.type} — {field.description}
{/FOR}

---

请提供参数，或我将根据上下文推断。
```

### Step 3: 执行调用

```
# 生成时间戳
timestamp = format(NOW(), "%Y%m%d_%H%M%S")
output_dir = "docs/subagents/runs/{timestamp}_{subagent_id}/"

# 创建输出目录
MKDIR output_dir

# 记录调用开始
OUTPUT:
    "## ⏳ 正在调用 {subagent_id}...
     
     **输出目录**: {output_dir}
     **超时**: {subagent.timeout_seconds} 秒"
```

### Step 4: 执行 Subagent 逻辑

根据 subagent_id 执行对应的逻辑：

#### repo_specs_retrieval

```
# 1. 解析问题和范围
question = input.question
scope = input.scope  # e.g., "specs/", "kernel/", "full_repo"

# 2. 使用 grep_search 和 read_file 工具收集证据
evidence_items = []

# 搜索相关文件
IF input.keywords:
    FOR keyword IN input.keywords:
        results = grep_search(keyword, scope)
        evidence_items.extend(results)

# 读取相关内容
FOR file IN matched_files:
    content = read_file(file, relevant_lines)
    evidence_items.append({
        file_path: file,
        line_range: relevant_lines,
        quote: content,
        relevance: "..."
    })

# 3. 生成输出
WRITE to {output_dir}/SUMMARY.md:
    ## Summary
    **Question**: {question}
    **Answer**: {synthesized_answer}
    **Confidence**: high|medium|low
    **Key Findings**:
    - ...

WRITE to {output_dir}/EVIDENCE.md:
    ## Evidence Items
    {FOR item IN evidence_items}
    ### {item.file_path}
    **Lines**: {item.line_range}
    ```
    {item.quote}
    ```
    **Relevance**: {item.relevance}
    {/FOR}
```

#### external_research

```
# 1. 解析研究问题
research_question = input.research_question
context = input.context

# 2. 使用 fetch_webpage 工具进行研究
# 注意：仅在 PLAN MODE 允许

# 3. 生成输出
WRITE to {output_dir}/SUMMARY.md:
    ## Research Summary
    **Question**: {research_question}
    **Answer**: {synthesized_answer}
    **Confidence**: high|medium|low
    **Recommendations**:
    - ...
    **Limitations**:
    - ...

WRITE to {output_dir}/EVIDENCE.md:
    ## Citations
    {FOR citation IN citations}
    ### {citation.title}
    **URL**: {citation.url}
    **Type**: {citation.type}
    **Key Quote**: "{citation.key_quote}"
    **Relevance**: {citation.relevance}
    {/FOR}
```

#### quant_risk_review

```
# 1. 解析目标文件
target_files = input.target_files
review_type = input.review_type  # "full" | "incremental" | "focused"
focus_areas = input.focus_areas or ["lookahead", "leakage", "protocol", "reproducibility"]

# 2. 对每个文件进行静态分析
issues = []
warnings = []

FOR file IN target_files:
    content = read_file(file)
    
    # Lookahead bias 检查
    IF "lookahead" IN focus_areas:
        # 检查是否使用未来数据
        lookahead_patterns = [
            r"shift\(-\d+\)",           # 负向 shift
            r"\.iloc\[.*:\]",           # 可能的未来切片
            r"future|forward|next",     # 可疑命名
        ]
        FOR pattern IN lookahead_patterns:
            matches = regex_search(content, pattern)
            IF matches:
                issues.append({
                    type: "lookahead_bias",
                    file: file,
                    line: match.line,
                    snippet: match.context,
                    severity: "high"
                })
    
    # Data leakage 检查
    IF "leakage" IN focus_areas:
        # 检查是否在训练中使用测试数据
        leakage_patterns = [
            r"train_test_split.*shuffle=True",  # 时间序列不应 shuffle
            r"fit\(.*test",                      # 在测试数据上 fit
        ]
        # ... 类似检查

# 3. 计算风险评分
risk_score = calculate_risk_score(issues, warnings)
verdict = "pass" IF risk_score < 3 ELSE "warn" IF risk_score < 7 ELSE "fail"

# 4. 生成输出
WRITE to {output_dir}/SUMMARY.md:
    ## Risk Review Summary
    **Verdict**: {verdict}
    **Risk Score**: {risk_score}/10
    **Files Reviewed**: {len(target_files)}
    
    ### Critical Issues ({len(critical_issues)})
    {FOR issue IN critical_issues}
    - [{issue.type}] {issue.file}:{issue.line} — {issue.description}
    {/FOR}
    
    ### Warnings ({len(warnings)})
    {FOR warning IN warnings}
    - [{warning.type}] {warning.file}:{warning.line} — {warning.description}
    {/FOR}

WRITE to {output_dir}/EVIDENCE.md:
    ## Detailed Evidence
    {FOR issue IN all_issues}
    ### Issue: {issue.id}
    **Type**: {issue.type}
    **File**: {issue.file}
    **Line**: {issue.line}
    **Severity**: {issue.severity}
    
    #### Code Snippet
    ```python
    {issue.snippet}
    ```
    
    #### Problem
    {issue.problem_description}
    
    #### Suggested Fix
    {issue.suggested_fix}
    {/FOR}

WRITE to {output_dir}/CHECKLIST.md:
    ## Risk Checklist
    
    | Category | Status | Issues |
    |----------|--------|--------|
    | Lookahead Bias | {status} | {count} |
    | Data Leakage | {status} | {count} |
    | Evaluation Protocol | {status} | {count} |
    | Reproducibility | {status} | {count} |
```

#### spec_drift

```
# 1. 比较 Spec 与实现
spec_files = find_files("specs/*.yaml")
impl_files = find_files("projects/dgsf/repo/src/**/*.py")

drift_items = []

FOR spec IN spec_files:
    spec_content = read_file(spec)
    
    # 提取 Spec 中定义的接口/契约
    contracts = extract_contracts(spec_content)
    
    FOR contract IN contracts:
        # 查找实现
        impl = find_implementation(contract, impl_files)
        
        IF impl IS NULL:
            drift_items.append({
                type: "SPEC_LAG",
                spec: spec,
                contract: contract,
                description: "Spec 定义了契约，但未找到实现"
            })
        ELSE:
            # 检查实现是否符合 Spec
            IF NOT matches_spec(impl, contract):
                drift_items.append({
                    type: "CODE_DRIFT",
                    spec: spec,
                    impl: impl.file,
                    description: "实现与 Spec 不一致"
                })

# 2. 检查交叉引用一致性
FOR spec_a, spec_b IN spec_pairs:
    conflicts = check_cross_reference(spec_a, spec_b)
    IF conflicts:
        drift_items.append({
            type: "MUTUAL_INCONSISTENCY",
            specs: [spec_a, spec_b],
            description: "Specs 之间存在冲突"
        })

# 3. 生成输出
WRITE to {output_dir}/SUMMARY.md:
    ## Spec Drift Analysis
    
    **Total Drift Items**: {len(drift_items)}
    
    ### By Category
    | Category | Count |
    |----------|-------|
    | SPEC_LAG | {count_spec_lag} |
    | CODE_DRIFT | {count_code_drift} |
    | MUTUAL_INCONSISTENCY | {count_mutual} |
    
    ### Recommendations
    {FOR item IN drift_items}
    - [{item.type}] {item.description}
      → Recommended action: {item.recommendation}
    {/FOR}

WRITE to {output_dir}/EVIDENCE.md:
    ## Drift Evidence
    {FOR item IN drift_items}
    ### {item.type}: {item.id}
    **Spec**: {item.spec}
    **Implementation**: {item.impl or "N/A"}
    
    #### Spec Content
    ```yaml
    {item.spec_excerpt}
    ```
    
    #### Implementation Content
    ```python
    {item.impl_excerpt or "Not found"}
    ```
    
    #### Discrepancy
    {item.description}
    {/FOR}
```

---

## 📤 OUTPUT HANDLING（输出处理）

### Step 5: 验证输出

```
# 检查输出文件是否创建
required_files = [
    "{output_dir}/SUMMARY.md",
    "{output_dir}/EVIDENCE.md"
]

FOR file IN required_files:
    IF NOT file_exists(file):
        ERROR: "Subagent 未生成必需的输出文件: {file}"
        RETRY or FAIL

# 验证 SUMMARY.md 格式
summary = read_file("{output_dir}/SUMMARY.md")
IF len(summary) > subagent.output_contract.max_summary_tokens:
    WARN: "SUMMARY 超过 token 限制，建议精简"
```

### Step 6: 返回结果

```markdown
## ✅ Subagent 调用完成

**Subagent**: {subagent_id}
**输出目录**: {output_dir}
**用时**: {elapsed_time} 秒

### 摘要

{INCLUDE: {output_dir}/SUMMARY.md}

---

**完整证据**: [{output_dir}/EVIDENCE.md]({output_dir}/EVIDENCE.md)

### 后续操作

- 将此结果附加到当前任务的 `subagent_artifacts`
- 如需详细信息，查看 EVIDENCE.md
```

---

## 🔗 INTEGRATION（集成）

### 与 Gate 系统集成

当 Gate 要求调用 Subagent 时：

```
# Gate-P1 触发
IF Gate-P1.triggered:
    INVOKE /dgsf_run_subagent WITH:
        subagent_id: "repo_specs_retrieval"
        question: "{derived from context}"
        scope: "specs/"

# Gate-E5 触发
IF Gate-E5.triggered:
    INVOKE /dgsf_run_subagent WITH:
        subagent_id: "quant_risk_review"
        target_files: "{changed_files}"
        review_type: "focused"
```

### 更新 Subagent 使用日志

调用完成后，自动追加到审计日志：

```
APPEND to docs/state/SUBAGENT_USAGE.md:

### Entry: {timestamp}

**Mode**: {current_mode}
**Subagent**: {subagent_id}
**Task**: {current_task_id or "standalone"}

**Input**:
- question/target: {input_summary}

**Output**:
- verdict: {verdict or "N/A"}
- output_path: {output_dir}

---
```

---

## ⚠️ ERROR HANDLING（错误处理）

| 错误类型 | 处理 |
|----------|------|
| Subagent 不存在 | 输出可用列表，建议选择 |
| 模式不允许 | 输出错误，建议切换模式 |
| 超时 | 重试一次，仍失败则记录并继续 |
| 输出验证失败 | 标记为 partial，记录问题 |

---

## 📝 USAGE EXAMPLES

### 示例 1: 规范检索

```
用户: 运行 subagent repo_specs_retrieval，检查 SDF_SPEC 中的特征定义

Copilot:
## 🔧 Subagent 调用准备

**Subagent**: repo_specs_retrieval
**版本**: 1.0.0

### 推断的参数
- question: "SDF_SPEC 中定义了哪些特征？"
- scope: "specs/"
- keywords: ["feature", "SDF_SPEC"]

确认执行？(y/修改参数)
```

### 示例 2: 风险审查

```
用户: 调用 quant_risk_review 检查 backtest/engine.py

Copilot:
## 🔧 Subagent 调用准备

**Subagent**: quant_risk_review
**版本**: 1.0.0

### 参数
- target_files: ["projects/dgsf/repo/src/dgsf/backtest/engine.py"]
- review_type: "full"
- focus_areas: ["lookahead", "leakage", "protocol", "reproducibility"]

确认执行？(y/修改参数)
```
```
