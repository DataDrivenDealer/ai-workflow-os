# TaskCard: [RELEASE_5_XXX]

> **Stage**: 5 · Release Review (Decision Gate)  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `RELEASE_5_XXX` |
| **创建日期** | YYYY-MM-DD |
| **Role Mode** | `architect` / `planner` |
| **Authority** | `speculative` → `accepted` (if approved) |
| **Authorized By** | [Project Owner ID] |
| **上游 Task** | `EVAL_4_XXX` |

---

## 1. Release Candidate Bundle

### 1.1 版本锁定清单
| 组件 | ID | Git Commit | Checksum | 锁定时间 |
|------|----|------------|----------|----------|
| Strategy Package | `SP_XXX` | | | |
| Data Snapshot | `DS_XXX` | | | |
| Config | | | | |
| Evaluation Report | | | | |

### 1.2 Release Candidate ID
```yaml
release_candidate:
  id: RC_XXX_YYYYMMDD
  created_at: 
  created_by: 
  
  pinned:
    code_commit: 
    data_snapshot: DS_XXX
    config_hash: 
    eval_report_hash: 
```

### 1.3 Artifact 完整性验证
- [ ] 所有 checksum 验证通过
- [ ] Git commit 可复现
- [ ] 依赖版本锁定
- [ ] 环境可复现

---

## 2. Decision Memo

### 2.1 策略概述
<!-- 简要描述策略的核心逻辑 -->

### 2.2 为什么有效 (Why It Works)
<!-- 描述 alpha 来源的假设和证据 -->
1. 
2. 
3. 

### 2.3 已知限制 (Known Limitations)
| 限制 | 影响 | 缓解措施 |
|------|------|----------|
| | | |

### 2.4 失效模式 (Failure Modes)
| 失效模式 | 触发条件 | 预期损失 | 检测方法 |
|----------|----------|----------|----------|
| Regime shift | | | |
| Crowding | | | |
| Data quality | | | |

### 2.5 Kill-Switch 条件
```yaml
kill_switch:
  triggers:
    - condition: drawdown > 15%
      action: reduce_position_50%
    - condition: drawdown > 25%
      action: full_exit
    - condition: daily_loss > 5%
      action: pause_24h
    - condition: model_drift > threshold
      action: alert_and_review
```

---

## 3. Risk Spec 合规检查

### 3.1 对照 Risk & Compliance Spec (L2)
| 规则 | Spec 要求 | 实际值 | 合规? |
|------|-----------|--------|-------|
| Max Drawdown | | | |
| Max Leverage | | | |
| Max Exposure | | | |
| Turnover Limit | | | |
| Position Concentration | | | |

### 3.2 合规声明
- [ ] 所有 L2 Risk Spec 要求已满足
- [ ] 偏离已记录为 Deviation

---

## 4. Gate G4: Approval

### 4.1 审批清单
| 审批项 | 审批人 | 状态 | 日期 |
|--------|--------|------|------|
| Technical Review | | `pending` | |
| Risk Review | | `pending` | |
| Compliance Check | | `pending` | |
| Final Approval | Project Owner | `pending` | |

### 4.2 审批条件
- [ ] Gate G3 已通过
- [ ] Decision Memo 完整
- [ ] Kill-Switch 定义
- [ ] Risk Spec 合规
- [ ] Runbook 草案就绪

### 4.3 审批决策
- [ ] **APPROVED** → 进入 Stage 6 (Deploy)
- [ ] **CONDITIONAL** → 附条件批准，列出条件
- [ ] **REJECTED** → 记录原因，返回修改
- [ ] **DEFERRED** → 延后决策，记录原因

### 4.4 审批记录
```yaml
approval:
  decision: [APPROVED/CONDITIONAL/REJECTED/DEFERRED]
  approved_by: 
  approved_at: 
  conditions: []
  comments: 
```

---

## 5. Release Plan (如批准)

### 5.1 部署阶段
| 阶段 | 开始日期 | 持续时间 | 规模 | 退出条件 |
|------|----------|----------|------|----------|
| Paper Trade | | 2 weeks | 100% | 无异常 |
| Shadow Mode | | 2 weeks | 100% | 与回测一致 |
| Pilot | | 4 weeks | 10% | DD < 10% |
| Full | | - | 100% | - |

### 5.2 回滚计划
```yaml
rollback:
  trigger_conditions:
    - pilot_dd > 10%
    - daily_loss > 3%
  procedure:
    1. Halt new orders
    2. Flatten positions
    3. Switch to benchmark
  responsible: 
```

---

## 6. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Release Candidate Bundle | `releases/RC_XXX/` | `pending` |
| Decision Memo | `releases/RC_XXX/decision_memo.md` | `pending` |
| Approval Record | `releases/RC_XXX/approval.yaml` | `pending` |
| Release Plan | `releases/RC_XXX/release_plan.md` | `pending` |

---

## 7. 下游依赖

- **后续 TaskCard**: `OPS_6_XXX`
- **Stage 6 需要**: 
  - Approved Release Candidate
  - Deployment Plan
  - Runbook draft

---

## 8. Authority 声明

```yaml
authority:
  type: speculative  # → accepted upon approval
  granted_by: [Project Owner ID]
  scope: release_decision
  
# Gate G4 通过后，Authority 状态变为 accepted
# Release Candidate 进入 frozen 状态
```

### 8.1 Authority 状态转换
```
speculative → [Gate G4 PASS] → accepted → [Deploy Complete] → frozen
```

---

## 9. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| | | `task_created` | From EVAL_4_XXX |
| | | `rc_bundle_created` | |
| | | `decision_memo_drafted` | |
| | | `gate_g4_submitted` | |
| | | `gate_g4_approved` | |

---

*Template: TASKCARD_RELEASE_5 v1.0.0*
