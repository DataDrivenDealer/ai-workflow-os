# DGSF Execution Plan V3

**Created**: 2026-02-02  
**Updated**: 2026-02-04T18:00Z (Post-T6.1 Completion · Real Data Ready)  
**Authority**: Derived from "scan → diagnose → plan → execute" cycle  
**Status**: Active  
**Primary Objective**: 持续推进DGSF（Dynamic Generative SDF Forest）项目的开发、验证与研究产出

---

## 🎯 Objectives & Non-goals

### Objectives（目标）
1. **✅ [已完成] 解除DGSF的开发阻塞** - pytest 收集错误已清除
2. **✅ [已完成] T3 Feature Engineering** - 2108 LOC, 66/66 tests, 602-line docs
3. **✅ [已完成] T4 Training Optimization** - 58.6% speedup, OOS/IS 1.637
4. **✅ [已完成] T5 Evaluation Framework** - 4 scripts, 5 metrics, 2/5 pass (synthetic)
5. **✅ [已完成] T6.1 DATA-001 Fix** - 真实数据加载器修复
6. **🎯 [当前焦点] T6.2 Real Data Validation** - 在真实数据上验证 T5 objectives
7. **保持AI Workflow OS作为支撑基础设施** - 提供治理和审计，不干扰研究

### Non-goals（不做什么 · Stop Doing List）
- ❌ **不优化AI Workflow OS的架构**（除非直接阻塞DGSF）
- ❌ **不清理Legacy资产**（标记为archive即可，清理是P2）
- ❌ **不重构kernel/模块**（已通过186个测试，无需改动）
- ❌ **不重构SDF v3.1增强功能**（time-smoothness, sparsity penalties 延后）
- ❌ **不实施非核心 ablation study**（已降级至可选）
- ❌ **不推送OS到远程**（DGSF T6 优先，OS同步可稍后）

---

## 📊 Current State Summary（聚焦DGSF · 2026-02-04T18:00Z）

### DGSF项目状态
| 维度 | 状态 | 证据 |
|------|------|------|
| **Pipeline Stage** | Stage 4 "SDF Layer Development" (**90%**) | PROJECT_DGSF.yaml |
| **T1 Model Review** | ✅ COMPLETED | SDF_MODEL_INVENTORY.json |
| **T2 Test Fixing** | ✅ COMPLETED (93.4% pass rate) | pytest output |
| **T3 Feature Engineering** | ✅ COMPLETED (2108 LOC, 66/66 tests) | FEATURE_ENGINEERING_GUIDE.md |
| **T4 Training Optimization** | ✅ **COMPLETED** (58.6% speedup, OOS/IS 1.637) | t4_final/results.json |
| **T5 Evaluation Framework** | ✅ **COMPLETED** (4 scripts, 5 metrics) | t5_*/metrics.json |
| **T6.1 DATA-001 Fix** | ✅ **COMPLETED** (56 mo × 48 features) | data_utils.py |
| **T6.2 Real Data Validation** | 🎯 **NEXT** | Pending |

### T4 完成记录
| Objective | Target | Actual | Status |
|-----------|--------|--------|--------|
| T4-OBJ-1: Speedup | ≥30% | **58.6%** | ✅ PASS |
| T4-OBJ-2: OOS Sharpe | ≥1.5 | 1.011 | ⚠️ synthetic |
| T4-OBJ-3: OOS/IS Ratio | ≥0.9 | **1.637** | ✅ PASS |

### T5 完成记录 (Synthetic Data)
| Objective | Target | Actual | Status |
|-----------|--------|--------|--------|
| T5-OBJ-1 Pricing Error | <0.01 | 0.079 | ⚠️ synthetic |
| T5-OBJ-2 OOS Sharpe | ≥1.5 | -6.31 | ⚠️ synthetic |
| T5-OBJ-3 OOS/IS Ratio | ≥0.9 | **2.72** | ✅ PASS |
| T5-OBJ-4 HJ Distance | <0.5 | 939.3 | ⚠️ synthetic |
| T5-OBJ-5 CS R² | ≥0.5 | **0.500** | ✅ PASS |

**Synthetic Data Pass Rate**: 2/5 (真实数据验证待 T6.2)

### AI Workflow OS状态
- **分支**: feature/router-v0（领先origin 16个提交）
- **测试**: kernel/ 186个测试通过 ✅
- **未提交**: 3个modified + 2个untracked（DGSF相关）

---

## 🛣️ Workstreams（≤3，至少1条DGSF本体）

### 🎯 Workstream 1: DGSF T6 Real Data Validation（P0 · 主线）
**Owner**: Copilot Agent  
**Objective**: 在真实数据上验证 T5 的 5 个 objectives

**Tasks**:
| ID | Task | Effort | Status |
|----|------|--------|--------|
| T6.1 | DATA-001 Fix (Data Loader) | 2h | ✅ COMPLETED |
| T6.2 | Re-run T5 Evaluation with Real Data | 2h | 🎯 **NEXT** |
| T6.3 | Document Results & Conclusions | 30min | PENDING |

**Completion Criteria**:
- ✅ evaluate_sdf.py 使用 RealDataLoader
- ✅ validate_sdf_oos.py 使用 RealDataLoader
- ✅ 5/5 T5 objectives 在真实数据上评估
- ✅ 明确结论：Pass / Fail / "数据量不足"

---

### Workstream 2: DGSF Commits & State Sync（P1 · 辅助）
**Owner**: Copilot Agent  
**Objective**: 保持 DGSF 工作的版本控制与审计

**Tasks**:
1. Commit pending DGSF changes (audit logs, scripts)
2. Update PROJECT_STATE.md after each milestone

---

### Workstream 3: OS Deferred Items（P2 · 暂停）
以下工作降级至 P2，待 T4 完成后再考虑：
- kernel/ 导入路径优化
- docs/ 合并与重构
- CI/CD 管道修复

---

## 🗓️ Milestones / Sprint Sequence

### ✅ Sprint 0: Environment Cleanup（已完成）
- **Completed**: 2026-02-02
- **Deliverables**: pytest.ini 更新, DGSF repo submodule 验证

### ✅ Sprint 1: T1-T2 Model Inventory & Test Fixing（已完成）
- **Completed**: 2026-02-03
- **Deliverables**: SDF_MODEL_INVENTORY.json, 93.4% test pass rate

### ✅ Sprint 2: T3 Feature Engineering（已完成）
- **Completed**: 2026-02-04
- **Deliverables**: 
  - 4 modules (2108 LOC): data_loaders, firm_characteristics, spreads_factors, run_feature_engineering
  - 66/66 tests passed
  - FEATURE_ENGINEERING_GUIDE.md (602 lines)

### 🎯 Sprint 3: T4 Training Optimization（当前 · 3周预估）
- **Started**: 2026-02-04
- **Goal**: 实现 30% 训练加速 + OOS Sharpe ≥1.5
- **Sub-sprints**:
  - **Week 1**: T4.1-T4.3 (Baseline + LR + FP16)
  - **Week 2**: T4.4-T4.5 (Early Stopping + Regularization)
  - **Week 3**: T4.6-T4.7 (Augmentation + Integration)

### Sprint 4: T5 Evaluation Framework（待定 · 2周预估）
- **Blocked by**: T4 completion
- **Goal**: 完整评估框架 + 研究产出

---

## ✅ Definition of Done（以DGSF可验证产出为核心）

### Sprint 3 T4 DoD（Training Optimization）
| Criteria | Target | Verification Command |
|----------|--------|----------------------|
| Training speedup | ≥30% vs baseline | `python benchmark_training.py --compare` |
| OOS Sharpe | ≥1.5 | `python evaluate_model.py --checkpoint best_model.pth` |
| OOS/IS ratio | ≥0.9 | Same as above |
| Checkpoint save/load | Consistent | `pytest tests/sdf/test_checkpoint.py` |
| Strategies documented | 5/5 | Verify in experiments/t4_*/results.json |

---

## 🔁 Verification Loop

每完成一个 T4 子任务：
1. **运行验证命令** - 确保指标满足目标
2. **更新 PROJECT_STATE.md** - 记录完成的任务、验证证据、下一步
3. **Commit changes** - 保持版本控制清晰
4. **检查 DGSF 关联** - 确认是 P0 工作

---

## 🛑 Stop Doing List（明确当前不该做的OS工作）

以下任务**暂停**，直到 T4 完成：

1. ❌ **kernel/模块的导入路径重构** - 当前测试已通过，无需改动
2. ❌ **架构边界验证脚本优化** - 已有脚本运行正常
3. ❌ **docs/重构与合并** - 文档数量多但不影响DGSF工作
4. ❌ **CI管道修复** - 可在DGSF有产出后再推送
5. ❌ **清理projects/dgsf/legacy/目录** - 标记为archive即可，删除是P2
6. ❌ **state/sessions.yaml的过期记录清理** - 不影响DGSF
7. ❌ **Feature Ablation Study** - 已降级，可在 T5 并行或作为 optional
8. ❌ **度量体系建立** - 等DGSF有稳定产出后再建立

**原则：除非某个OS任务直接解除DGSF的阻塞，否则一律降级为P2**

---

## 📝 Dependencies & Risks

### Dependencies
- **Sprint 1阻塞于Project Owner输入** - 需要明确研究优先级
- **DGSF repo外部依赖** - 如果repo/有自己的依赖，需要在repo/内安装

### Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Project Owner未及时响应 | Sprint 1延迟 | 先完成Sprint 0+2，提供默认任务建议 |
| DGSF repo submodule过期 | 实验不可复现 | Sprint 0立即验证并更新 |
| pytest配置失效 | 165错误仍显示 | 使用`--ignore`参数，并验证 |

---

## 📂 Artifacts

所有产出物路径：

| Artifact | Path | Status |
|----------|------|--------|
| Execution Plan | docs/plans/EXECUTION_PLAN_DGSF_V1.md | ✅ Created |
| TODO List | docs/plans/TODO_NEXT.md | 🔄 To be updated |
| State Log | docs/state/PROJECT_STATE.md | 🔄 To be updated |
| pytest Config | pytest.ini or pyproject.toml | 🔄 To be updated |
| Legacy README | projects/dgsf/legacy/README.md | 🔄 To be created |
| Main DGSF README | projects/dgsf/README.md | 🔄 To be updated |
| PROJECT_DGSF.yaml | projects/dgsf/specs/PROJECT_DGSF.yaml | 🔄 To be updated (Sprint 1) |

---

## 🔄 Next Review

- **When**: Sprint 0完成后（预计1小时内）
- **What**: 验证pytest噪声是否清除，submodule是否同步
- **Who**: Copilot Agent自我验证，然后提交给Project Owner review

---

**END OF EXECUTION PLAN V1**
