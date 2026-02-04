# AEP-6: Meta-Evolution Monitoring（元演化监控）

**Status**: Draft  
**Created**: 2026-02-04  
**Addresses**: Tension C1 (自指度量问题), B3 (演化决策归属)

---

## 1. 动机

当前演化闭环存在"度量盲区"：
- 演化有效性依赖摩擦信号收集
- 但演化本身可能改变摩擦信号的收集率
- 无法区分"真正改善"和"盲点扩大"

## 2. 提案

### 2.1 引入二阶度量（Meta-Metrics）

在 `configs/evolution_policy.yaml` 中新增：

```yaml
meta_monitoring:
  enabled: true
  
  # 演化系统自身的健康指标
  health_metrics:
    signal_coverage:
      description: "代码路径中有多少比例被摩擦信号检测覆盖"
      target: ">= 0.8"
      measurement: "scripts/measure_signal_coverage.py"
      
    evolution_velocity:
      description: "从信号到实施的平均周期"
      target: "<= 14d"
      measurement: "time(actioned) - time(first_signal)"
      
    regression_rate:
      description: "演化后 30 天内被回滚的比例"
      target: "<= 0.1"
      measurement: "count(rollback) / count(applied)"
      
    blind_spot_proxy:
      description: "新代码路径在 7 天内未产生任何信号的比例"
      target: "<= 0.2"
      measurement: "新路径中无信号覆盖的比例"
      alert: "high"  # 高值可能表示检测盲区扩大
      
  # 自监控触发
  self_monitoring:
    frequency: "weekly"
    report_path: "reports/meta_evolution_health_{date}.md"
    alert_on:
      - metric: "blind_spot_proxy"
        condition: "> 0.3"
        action: "flag_for_coverage_review"
```

### 2.2 新增 Blind Spot Detection 机制

```yaml
blind_spot_detection:
  # 主动探测：对新增代码路径进行"摩擦诱导测试"
  proactive:
    enabled: true
    trigger: "new_code_path_added"
    action: "inject_edge_case_scenarios"
    
  # 被动监控：追踪哪些代码区域从未触发摩擦
  passive:
    enabled: true
    cold_zone_threshold: "30d"  # 30 天无摩擦信号
    action: "flag_for_manual_review"
```

### 2.3 演化效果的 A/B 验证

```yaml
evolution_validation:
  # 在正式应用前进行 A/B 测试
  ab_testing:
    enabled: true
    control_group: "10% of experiments"
    treatment_group: "10% of experiments"
    duration: "14d"
    success_criterion: "friction_delta <= -0.2"  # 摩擦降低 ≥20%
    
  # 回滚触发条件
  rollback_triggers:
    - condition: "new_friction_type_count > 3"
      description: "演化引入了新类型的摩擦"
    - condition: "regression_rate > 0.15"
      description: "回归率过高"
```

## 3. 接口变更

### 3.1 新增 MCP 工具

```json
{
  "tool": "evolution_health_check",
  "description": "检查演化系统自身的健康状态",
  "input": {
    "scope": "string",  // "global" | "project:{id}"
    "window": "string"  // "7d" | "30d" | "90d"
  },
  "output": {
    "metrics": "object",
    "alerts": "array",
    "recommendations": "array"
  }
}
```

### 3.2 新增状态

在 `kernel/state_machine.yaml` 的 evolution lifecycle 中新增：

```yaml
evolution_states:
  - proposed          # 信号聚合后提出
  - ab_testing        # A/B 测试中（新增）
  - validated         # A/B 通过
  - applied           # 已应用
  - monitoring        # 30d 观察期（新增）
  - confirmed         # 确认有效
  - rolled_back       # 已回滚（新增）
```

## 4. 实施路线

| 阶段 | 内容 | 依赖 |
|------|------|------|
| P1 | 添加 meta_monitoring 配置 | 无 |
| P2 | 实现 blind_spot_detection 脚本 | P1 |
| P3 | 集成 A/B 测试框架 | P1, P2 |
| P4 | 更新 state_machine.yaml | P3 |

## 5. 验证方法

- **单元测试**: `kernel/tests/test_meta_evolution.py`
- **集成测试**: 模拟演化周期，验证度量收集
- **回归测试**: 确保现有演化流程不受影响

## 6. 风险

| 风险 | 缓解 |
|------|------|
| A/B 测试增加延迟 | 可配置为可选 |
| 过度监控增加开销 | 采样策略 |
| 回滚机制复杂 | 保持幂等性 |

---

**审批流程**: Platform Engineer → Project Owner → Apply
