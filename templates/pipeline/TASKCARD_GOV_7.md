# TaskCard: [GOV_7_XXX]

> **Stage**: 7 · Post-mortem & Spec Promotion  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `GOV_7_XXX` |
| **创建日期** | YYYY-MM-DD |
| **Role Mode** | `architect` |
| **Authority** | `speculative` |
| **Authorized By** | [Project Owner ID] |
| **上游 Task** | `OPS_6_XXX` |
| **触发原因** | [下线/Incident/定期审查] |

---

## 1. Post-mortem 触发

### 1.1 触发类型
- [ ] **策略下线** - 策略终止运行
- [ ] **重大 Incident** - 发生严重问题
- [ ] **定期审查** - Quarterly/Annual review
- [ ] **主动优化** - 基于运行经验改进

### 1.2 审查范围
| 项目 | 审查对象 |
|------|----------|
| 策略 | |
| 时间范围 | |
| 相关 TaskCards | |

---

## 2. What Failed

### 2.1 失效分析
| 失效点 | 根因 | 影响 | 发现方式 |
|--------|------|------|----------|
| | | | |

### 2.2 假设验证
| 原假设 | 实际情况 | 偏差原因 |
|--------|----------|----------|
| | | |

### 2.3 未预见的 Failure Modes
| Failure Mode | 描述 | 应纳入哪个 Gate |
|--------------|------|-----------------|
| | | |

---

## 3. What Was Expensive

### 3.1 成本分析
| 成本项 | 预估 | 实际 | 偏差 | 原因 |
|--------|------|------|------|------|
| 开发时间 | | | | |
| 数据成本 | | | | |
| 计算成本 | | | | |
| 机会成本 | | | | |
| 运维成本 | | | | |

### 3.2 ROI 评估
```yaml
roi_analysis:
  total_investment: 
  total_return: 
  roi: 
  break_even_time: 
  assessment: [positive/negative/neutral]
```

---

## 4. What Was Fragile

### 4.1 脆弱点识别
| 组件 | 脆弱表现 | 根因 | 改进建议 |
|------|----------|------|----------|
| | | | |

### 4.2 依赖风险
| 依赖 | 风险等级 | 发生过问题? | 缓解措施 |
|------|----------|-------------|----------|
| 数据源 | | | |
| 执行通道 | | | |
| 基础设施 | | | |
| 人员 | | | |

---

## 5. Spec Updates (L2)

### 5.1 Pipeline Spec 修改建议
| 修改项 | 当前 | 建议 | 理由 |
|--------|------|------|------|
| | | | |

### 5.2 Gate 阈值调整
| Gate | 当前阈值 | 建议阈值 | 理由 |
|------|----------|----------|------|
| G1 | | | |
| G2 | | | |
| G3 | | | |
| G4 | | | |
| G5 | | | |

### 5.3 新增检查项
| Gate | 新检查项 | 理由 |
|------|----------|------|
| | | |

### 5.4 L2 Spec 变更提案
```yaml
spec_change:
  type: L2_update
  target_spec: 
  changes:
    - section: 
      before: 
      after: 
      rationale: 
```

---

## 6. Promotion Proposal (L1/L0)

### 6.1 可推广经验
| 经验 | 适用范围 | 推广级别 |
|------|----------|----------|
| | | L1 / L0 |

### 6.2 L1 Framework 提案
<!-- 如果有跨项目通用价值 -->
```yaml
promotion_proposal:
  target_level: L1
  title: 
  abstract: 
  
  applicability:
    - project_type: 
    - conditions: 
    
  evidence:
    - source: [this project]
      result: 
      
  recommendation: [adopt / pilot / reject]
```

### 6.3 L0 Canon 提案
<!-- 如果有全局不变式价值 -->
```yaml
canon_proposal:
  target: L0
  type: [invariant / principle / constraint]
  statement: 
  rationale: 
  evidence: 
  
  recommendation: [adopt / pilot / reject]
```

---

## 7. Action Items

### 7.1 立即行动
| 行动 | 负责人 | 截止日期 | 状态 |
|------|--------|----------|------|
| | | | |

### 7.2 长期改进
| 改进项 | 优先级 | 预计工作量 | 状态 |
|--------|--------|-----------|------|
| | | | |

### 7.3 知识沉淀
| 文档 | 路径 | 状态 |
|------|------|------|
| Post-mortem Report | | |
| Lessons Learned | | |
| Updated Runbook | | |

---

## 8. Governance Flow

### 8.1 变更审批路径
```
Post-mortem Report
    ↓
L2 Spec 变更提案 → Project Owner 审批 → L2 Spec 更新
    ↓
L1 Promotion 提案 → Framework Council 审批 → L1 Framework 更新
    ↓
L0 Canon 提案 → Canon Council 审批 → L0 Canon 更新
```

### 8.2 审批状态
| 提案 | 提交日期 | 审批人 | 状态 |
|------|----------|--------|------|
| L2 Spec Update | | | `pending` |
| L1 Promotion | | | `pending` |
| L0 Canon | | | `pending` |

---

## 9. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Post-mortem Report | `reports/postmortem/GOV_7_XXX.md` | `pending` |
| L2 Change Proposal | `ops/proposals/L2_XXX.md` | `pending` |
| L1 Promotion Proposal | `ops/proposals/L1_XXX.md` | `pending` |
| Lessons Learned | `docs/lessons/GOV_7_XXX.md` | `pending` |

---

## 10. 闭环验证

### 10.1 变更实施追踪
| 变更 | 实施日期 | 验证方式 | 验证结果 |
|------|----------|----------|----------|
| | | | |

### 10.2 后续监控
- [ ] 变更已在下一个策略周期中应用
- [ ] 新增检查项在 CI 中生效
- [ ] 更新的阈值已验证合理

---

## 11. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: [Project Owner ID]
  scope: governance
  
# Post-mortem 结论需要相应层级审批
# L2 变更 → Project Owner
# L1 变更 → Framework Council
# L0 变更 → Canon Council
```

---

## 12. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| | | `task_created` | From OPS_6_XXX |
| | | `postmortem_started` | |
| | | `l2_proposal_submitted` | |
| | | `l2_approved` | |

---

*Template: TASKCARD_GOV_7 v1.0.0*
