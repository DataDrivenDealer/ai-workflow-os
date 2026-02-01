# TaskCard: [OPS_6_XXX]

> **Stage**: 6 · Deploy & Operate  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `OPS_6_XXX` |
| **创建日期** | YYYY-MM-DD |
| **Role Mode** | `executor` |
| **Authority** | `accepted` (inherited from Release) |
| **Authorized By** | [Project Owner ID] |
| **上游 Task** | `RELEASE_5_XXX` |
| **Release Candidate** | `RC_XXX` |

---

## 1. Deployment

### 1.1 部署状态
| 阶段 | 开始日期 | 状态 | 规模 | 备注 |
|------|----------|------|------|------|
| Paper Trade | | `pending` | 100% | |
| Shadow Mode | | `pending` | 100% | |
| Pilot | | `pending` | 10% | |
| Full Production | | `pending` | 100% | |

### 1.2 环境配置
```yaml
deployment:
  environment: production
  
  infrastructure:
    compute: 
    memory: 
    storage: 
    
  connections:
    market_data: 
    execution: 
    database: 
```

### 1.3 版本追踪
| 组件 | 生产版本 | 最后部署时间 |
|------|----------|--------------|
| Strategy Code | `SP_XXX` @ commit | |
| Config | | |
| Data Pipeline | | |

---

## 2. Monitoring

### 2.1 PnL Attribution
| 维度 | 今日 | 本周 | 本月 | YTD |
|------|------|------|------|-----|
| 总收益 | | | | |
| Alpha | | | | |
| Beta | | | | |
| Transaction Cost | | | | |

### 2.2 Risk Metrics Dashboard
| 指标 | 当前值 | 警戒线 | 限制线 | 状态 |
|------|--------|--------|--------|------|
| Current DD | | 10% | 15% | |
| Daily VaR | | | | |
| Leverage | | | | |
| Concentration | | | | |

### 2.3 Drift Detection
| 信号 | 当前值 | 基准范围 | 偏离? |
|------|--------|----------|--------|
| Signal Distribution | | | |
| Turnover | | | |
| Factor Exposure | | | |
| Performance vs Backtest | | | |

### 2.4 监控告警配置
```yaml
alerts:
  - name: drawdown_warning
    condition: drawdown > 10%
    severity: warning
    notify: [ops_team]
    
  - name: drawdown_critical
    condition: drawdown > 15%
    severity: critical
    notify: [ops_team, risk_team, owner]
    action: auto_reduce_50%
    
  - name: daily_loss
    condition: daily_pnl < -3%
    severity: critical
    notify: [all]
    action: pause_trading
    
  - name: model_drift
    condition: signal_drift > 2_std
    severity: warning
    notify: [quant_team]
```

---

## 3. Incident Response

### 3.1 告警分级处理
| 级别 | 响应时间 | 处理流程 | 升级路径 |
|------|----------|----------|----------|
| Info | 1 day | Log & Review | - |
| Warning | 4 hours | Investigate | → Critical if persists |
| Critical | 15 min | Immediate action | → Owner |
| Emergency | Immediate | Kill switch | → All stakeholders |

### 3.2 Rollback 流程
```yaml
rollback_procedure:
  steps:
    1. Halt new order generation
    2. Cancel pending orders
    3. Flatten positions (if required)
    4. Switch to benchmark/cash
    5. Notify stakeholders
    6. Create incident report
    
  authorized_by: [ops_lead, owner]
  max_time_to_flat: 30min
```

### 3.3 Incident Log
| 时间戳 | 级别 | 描述 | 响应 | 解决时间 |
|--------|------|------|------|----------|
| | | | | |

---

## 4. Gate G5: Live Safety

### 4.1 每日安全检查
| 检查项 | 状态 | 说明 |
|--------|------|------|
| Kill-switch 可用 | `pending` | |
| 监控正常 | `pending` | |
| 数据 feed 正常 | `pending` | |
| 执行通道正常 | `pending` | |
| 风险限制生效 | `pending` | |

### 4.2 定期审查
| 审查项 | 频率 | 最后审查 | 下次审查 |
|--------|------|----------|----------|
| Performance Review | Weekly | | |
| Risk Review | Weekly | | |
| Model Drift Check | Daily | | |
| Compliance Check | Monthly | | |

### 4.3 Gate G5 状态
- [ ] **HEALTHY** → 正常运行
- [ ] **WARNING** → 观察中
- [ ] **DEGRADED** → 部分功能受限
- [ ] **CRITICAL** → 需立即处理

---

## 5. Runbook

### 5.1 日常操作
| 操作 | 时间 | 负责人 | 检查项 |
|------|------|--------|--------|
| Pre-market check | 09:00 | | Data feed, positions |
| Post-market review | 16:30 | | PnL, fills, anomalies |
| Weekly report | Friday | | Summary to stakeholders |

### 5.2 应急联系人
| 角色 | 姓名 | 联系方式 | 备注 |
|------|------|----------|------|
| Primary Ops | | | |
| Backup Ops | | | |
| Quant Lead | | | |
| Risk Officer | | | |
| Project Owner | | | |

### 5.3 关键命令
```bash
# 暂停交易
./ops.sh pause --strategy=XXX

# 恢复交易
./ops.sh resume --strategy=XXX

# 紧急平仓
./ops.sh flatten --strategy=XXX --confirm

# 状态检查
./ops.sh status --strategy=XXX
```

---

## 6. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Runbook | `runbooks/OPS_6_XXX.md` | `pending` |
| Monitoring Config | `configs/monitoring/XXX.yaml` | `pending` |
| Alert Rules | `configs/alerts/XXX.yaml` | `pending` |
| Incident Log | `logs/incidents/XXX/` | `active` |

---

## 7. 下游依赖

- **后续 TaskCard**: `GOV_7_XXX` (Post-mortem)
- **触发条件**: 
  - 策略下线
  - 重大 incident
  - 定期审查 (quarterly)

---

## 8. Authority 声明

```yaml
authority:
  type: accepted  # inherited from Release
  granted_by: [Project Owner ID]
  scope: operations
  
# Ops 变更需要 deviation 记录
# Kill-switch 触发需要 incident report
```

---

## 9. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| | | `task_created` | From RELEASE_5_XXX |
| | | `paper_trade_started` | |
| | | `pilot_started` | |
| | | `full_production` | |

---

*Template: TASKCARD_OPS_6 v1.0.0*
