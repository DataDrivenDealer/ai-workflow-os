# Evidence: Repo & Specs Retrieval

## File References

### `configs\agent_modes.yaml`

**Line 68** (keyword: `gate`)
```
# 必须满足的 Gates
```

**Line 69** (keyword: `gate`)
```
required_gates:
```

**Line 70** (keyword: `gate`)
```
- Gate-P1                      # Specs Scan
```

**Line 71** (keyword: `gate`)
```
- Gate-P8                      # Write-back Attachment
```

**Line 73** (keyword: `gate`)
```
optional_gates:
```

**Line 74** (keyword: `gate`)
```
- Gate-P6                      # DRS (可跳过但需理由)
```

**Line 138** (keyword: `gate`)
```
- invoke_subagent              # 调用 Subagent (限 review gate)
```

**Line 147** (keyword: `gate`)
```
# 可调用的 Subagents (限 Review Gate)
```

**Line 155** (keyword: `gate`)
```
# 必须满足的 Gates
```

**Line 156** (keyword: `gate`)
```
required_gates:
```

### `configs\code_practice_registry.yaml`

**Line 274** (keyword: `gate`)
```
gates:
```

**Line 275** (keyword: `gate`)
```
- "configs/gates.yaml#backtest_completion"
```

**Line 392** (keyword: `gate`)
```
experiment_gate:
```

**Line 611** (keyword: `gate`)
```
experiment_gate:
```

**Line 612** (keyword: `gate`)
```
gate_config: "configs/gates.yaml#experiment_completion"
```

**Line 654** (keyword: `gate`)
```
3. Aggregate for monthly review
```

### `configs\gates.yaml`

**Line 1** (keyword: `gate`)
```
# Gate Configuration
```

**Line 2** (keyword: `gate`)
```
# Defines thresholds and rules for PROJECT_DELIVERY_PIPELINE gates
```

**Line 30** (keyword: `gate`)
```
gates:
```

**Line 32** (keyword: `gate`)
```
# G1: Data Quality Gate (Stage 2 Exit)
```

**Line 81** (keyword: `gate`)
```
# G2: Sanity Checks Gate (Stage 3 Exit)
```

**Line 132** (keyword: `gate`)
```
# G3: Performance & Robustness Gate (Stage 4 Exit)
```

**Line 192** (keyword: `gate`)
```
# G4: Approval Gate (Stage 5 Exit)
```

**Line 201** (keyword: `gate`)
```
description: "G3 gate has been passed"
```

**Line 250** (keyword: `gate`)
```
# G5: Live Safety Gate (Stage 6 Continuous)
```

**Line 308** (keyword: `gate`)
```
# Subagent Invocation Gates (Plan/Execute Mode Enforcement)
```

