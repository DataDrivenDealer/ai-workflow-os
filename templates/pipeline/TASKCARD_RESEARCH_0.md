# TaskCard: [RESEARCH_0_XXX]

> **Stage**: 0 · Idea Intake & Triage  
> **Pipeline**: PROJECT_DELIVERY_PIPELINE  
> **Template Version**: 1.0.0

---

## 元信息

| 字段 | 值 |
|------|-----|
| **Task ID** | `RESEARCH_0_XXX` |
| **创建日期** | YYYY-MM-DD |
| **Role Mode** | `architect` / `planner` |
| **Authority** | `speculative` |
| **Authorized By** | [Project Owner ID] |

---

## 1. 背景与来源

### 1.1 想法来源
<!-- 标记想法来源类型 -->
- [ ] 学术论文
- [ ] 行业研报
- [ ] 内部 alpha idea
- [ ] Post-mortem 教训
- [ ] 其他: _____________

### 1.2 来源引用
<!-- 列出具体参考资料 -->
- **论文/报告**: 
- **链接/DOI**: 
- **关键发现摘要**: 

---

## 2. Triage 评估

### 2.1 新颖性 vs 成本矩阵
| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | | |
| 实现成本 | | |
| 数据可得性 | | |
| 潜在 alpha 强度 | | |

### 2.2 数据可得性检查
- [ ] 所需数据类型: _______________
- [ ] 数据源: _______________
- [ ] 数据时间跨度: _______________
- [ ] 数据质量预估: _______________

### 2.3 Edge 假设
<!-- 用一句话描述预期的超额收益来源 -->
> 

---

## 3. 决策

### 3.1 Triage 结论
- [ ] **PROCEED** → 进入 Stage 1 (Research Design)
- [ ] **HOLD** → 等待更多数据/资源
- [ ] **REJECT** → 记录原因后归档

### 3.2 拒绝/暂缓理由（如适用）
<!-- 如果 HOLD 或 REJECT，说明原因 -->

---

## 4. 输出 Artifacts

| Artifact | 路径 | 状态 |
|----------|------|------|
| Triage Report | `ops/decision-log/RESEARCH_0_XXX_triage.md` | `pending` |

---

## 5. 下游依赖

- **后续 TaskCard**: `RESEARCH_1_XXX` (如 PROCEED)
- **Gate 要求**: 无 (Stage 0 无 Gate)

---

## 6. Authority 声明

```yaml
authority:
  type: speculative
  granted_by: [Project Owner ID]
  scope: research_triage
  expires: YYYY-MM-DD
  
# 警告：此任务的所有输出均为 speculative 状态
# 需要 Project Owner 显式 accept 后方可作为后续决策依据
```

---

## 7. Audit Trail

| 时间戳 | Agent ID | 操作 | 说明 |
|--------|----------|------|------|
| | | `task_created` | |

---

*Template: TASKCARD_RESEARCH_0 v1.0.0*
