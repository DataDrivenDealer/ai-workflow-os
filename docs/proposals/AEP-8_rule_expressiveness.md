# AEP-8: 规则表达力增强（Rule Expressiveness Enhancement）

**Status**: Draft  
**Created**: 2026-02-04  
**Addresses**: Tension E1 (规则非组合), E3 (条件逻辑缺失), C2 (优先级静态性)

---

## 1. 动机

当前规则系统存在表达力不足：

| 限制 | 示例 |
|------|------|
| 无参数化 | `R2: One task at a time` 无法配置为 `parallelism_limit=2` |
| 无条件逻辑 | 无法表达 `if risk_level == high then sharpe >= 2.0` |
| 无组合 | 无法表达 `R2 AND NOT hotfix_mode` |
| 静态优先级 | 无运行时优先级调整机制 |

这导致：
- 规则豁免靠自然语言描述，难以自动验证
- 复杂治理场景需大量特例规则
- 规则演化缺乏形式化基础

## 2. 提案

### 2.1 规则表达语言（Rule Expression Language, REL）

引入轻量级 DSL 用于规则定义：

```yaml
# 在 configs/rule_schema.yaml 中定义 REL 语法

rel_schema:
  version: "1.0.0"
  
  # 基础类型
  types:
    - boolean
    - number
    - string
    - list[T]
    - context  # 运行时上下文对象
    
  # 运算符
  operators:
    logical: [AND, OR, NOT, IMPLIES]
    comparison: [">=", "<=", "==", "!=", "IN"]
    quantifiers: [ALL, ANY, NONE]
    
  # 上下文变量（运行时可用）
  context_variables:
    task:
      type: "task_status"
      priority: "P0-P4"
      queue: "string"
      is_hotfix: "boolean"
      
    agent:
      authority_level: "0-3"
      session_duration: "duration"
      consecutive_failures: "number"
      
    experiment:
      risk_level: "low|medium|high"
      parallel_count: "number"
      
    environment:
      market_regime: "bull|bear|neutral"
      time_of_day: "datetime"
```

### 2.2 规则定义格式

扩展 `copilot-instructions.md` 中的规则表达：

```yaml
# 从自然语言
| R2 | One task at a time | Editing multiple modules simultaneously |

# 到结构化表达
rules:
  R2:
    id: "R2"
    name: "Task Parallelism Control"
    priority: "P2"
    
    # 核心条件（REL 表达式）
    condition: |
      context.agent.running_tasks <= params.parallelism_limit
      
    # 参数化（可被 Adapter/Project 覆盖）
    parameters:
      parallelism_limit:
        type: number
        default: 1
        min: 1
        max: 5
        override_level: "adapter"  # kernel|adapter|project|experiment
        
    # 例外条件（REL 表达式）
    exceptions:
      - name: "hotfix_override"
        condition: "context.task.is_hotfix AND context.agent.authority_level >= 2"
        effect: "parallelism_limit = 2"
        audit: true
        
      - name: "ab_testing"
        condition: "context.experiment.type == 'ab_test'"
        effect: "parallelism_limit = 2"
        requires_approval: true
        
    # 违规处理
    violation:
      action: "block"
      message: "Running tasks ({running}) exceeds limit ({limit})"
      evolution_signal: true  # 自动记录摩擦
      
    # 验证方法
    verification:
      auto_check: true
      script: "scripts/check_wip_limit.py"
```

### 2.3 条件阈值表达

扩展 `thresholds` 支持条件逻辑：

```yaml
thresholds:
  primary_metrics:
    oos_sharpe:
      # 简单形式（向后兼容）
      operator: ">="
      value: 1.5
      
      # 条件形式（新增）
      conditional:
        - when: "context.experiment.risk_level == 'high'"
          value: 2.0
        - when: "context.environment.market_regime == 'bear'"
          value: 1.2
        - otherwise:
          value: 1.5
          
      # 动态计算形式（高级）
      computed:
        expression: "base_sharpe * risk_multiplier"
        variables:
          base_sharpe: 1.5
          risk_multiplier: "lookup('risk_table', context.experiment.risk_level)"
```

### 2.4 规则组合语法

```yaml
# 复合规则定义
compound_rules:
  CR1:
    name: "Production Deployment Gate"
    expression: "R1 AND R3 AND R4 AND (R2 OR exception.hotfix)"
    
    # 组合后的优先级
    priority: "MAX(R1.priority, R3.priority, R4.priority)"
    
    # 组合验证
    verification:
      mode: "all_must_pass"  # all_must_pass | any_must_pass | weighted
      
  CR2:
    name: "Research Mode"
    expression: "R1 AND R5 AND NOT(R2)"  # 研究模式允许并行
    applies_when: "context.task.queue == 'research'"
```

### 2.5 运行时优先级调整

```yaml
# 优先级借用机制
priority_modifiers:
  # 临时提升
  elevation:
    trigger: "context.task.priority == 'P0' AND context.task.is_production_incident"
    effect:
      rule: "R2"
      new_priority: "P1"  # 从 P2 降到 P1
      duration: "4h"
      audit: true
      requires_human_approval: true
      
  # 临时降低
  demotion:
    trigger: "context.agent.consecutive_failures >= 3"
    effect:
      all_rules:
        new_priority: "P4"  # 所有规则变为最高优先级（最严格）
      duration: "until_success"
```

## 3. 实现架构

### 3.1 REL 解释器

```python
# kernel/rule_engine.py

class RuleEngine:
    """规则表达式解释器"""
    
    def __init__(self, rules_config: dict, context: RuntimeContext):
        self.rules = self._parse_rules(rules_config)
        self.context = context
        
    def evaluate_rule(self, rule_id: str) -> RuleResult:
        """评估单条规则"""
        rule = self.rules[rule_id]
        
        # 检查例外条件
        for exception in rule.exceptions:
            if self._eval_expression(exception.condition):
                return RuleResult(
                    passed=True,
                    via_exception=exception.name,
                    audit_required=exception.audit
                )
        
        # 评估核心条件
        params = self._resolve_parameters(rule.parameters)
        passed = self._eval_expression(rule.condition, params)
        
        return RuleResult(
            passed=passed,
            violation_message=rule.violation.message if not passed else None
        )
        
    def evaluate_compound(self, compound_id: str) -> RuleResult:
        """评估复合规则"""
        compound = self.compound_rules[compound_id]
        return self._eval_expression(compound.expression)
```

### 3.2 类型检查

```python
# kernel/rule_type_checker.py

class RuleTypeChecker:
    """规则表达式静态类型检查"""
    
    def check_rule(self, rule: Rule) -> list[TypeError]:
        errors = []
        
        # 检查条件表达式类型
        condition_type = self._infer_type(rule.condition)
        if condition_type != "boolean":
            errors.append(TypeError(
                f"Rule condition must be boolean, got {condition_type}"
            ))
            
        # 检查参数引用
        for param_ref in self._extract_param_refs(rule.condition):
            if param_ref not in rule.parameters:
                errors.append(TypeError(
                    f"Undefined parameter: {param_ref}"
                ))
                
        return errors
```

## 4. 迁移路径

### 4.1 现有规则迁移

```yaml
# 迁移映射示例

migrations:
  R1:
    from:
      markdown: "| R1 | Verify before asserting | 'File at X' without Test-Path |"
    to:
      rel:
        condition: "ALL(claims, claim => claim.has_verification)"
        parameters: {}
        exceptions: []
        
  R2:
    from:
      markdown: "| R2 | One task at a time | Editing multiple modules simultaneously |"
    to:
      rel:
        condition: "context.agent.running_tasks <= params.parallelism_limit"
        parameters:
          parallelism_limit: {default: 1, override_level: "adapter"}
        exceptions:
          - {condition: "context.task.is_hotfix", effect: "parallelism_limit = 2"}
```

### 4.2 向后兼容

- Markdown 规则表仍然作为"人类可读文档"保留
- REL 规则与 Markdown 表双向同步
- 缺少 REL 定义时降级为字符串匹配

## 5. 验证方法

```yaml
validation:
  static:
    - type_checking: "所有 REL 表达式通过类型检查"
    - completeness: "所有 Markdown 规则有对应 REL 定义"
    - consistency: "REL 语义与 Markdown 描述一致"
    
  runtime:
    - unit_tests: "kernel/tests/test_rule_engine.py"
    - integration: "模拟场景验证规则行为"
    - fuzzing: "随机上下文测试边界条件"
```

## 6. 风险

| 风险 | 缓解 |
|------|------|
| 学习曲线 | 提供 REL 可视化编辑器 |
| 过度复杂化 | 限制嵌套深度（max 3 层）|
| 性能 | 表达式编译缓存 |
| 安全 | 禁止副作用表达式 |

---

**审批流程**: Platform Engineer → Architecture Review → Apply
