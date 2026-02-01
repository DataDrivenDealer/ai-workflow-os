# 📐 DGSF 规范映射计划

**文档 ID**: SPEC_MAPPING_PLAN  
**编写人**: 张平台 (平台架构师)  
**日期**: 2026-02-01  
**状态**: ✅ COMPLETED

---

## 0. 执行摘要

本文档定义了 Legacy DGSF specs_v3 规范体系到 AI Workflow OS 治理框架的映射方案。

### 映射原则
1. **保持原位**: Legacy 规范保持原有位置，通过引用方式集成
2. **层级对齐**: DGSF 层级映射到 AI Workflow OS 的 L2 项目规范
3. **治理增强**: 在 DGSF 规范基础上增加 AI Workflow OS 治理约束
4. **无破坏性**: 不修改任何 Legacy 代码和规范

---

## 1. 规范层级映射

### 1.1 AI Workflow OS 规范层级

```
L0 Canon (冻结)
├── GOVERNANCE_INVARIANTS
├── AUTHORITY_CANON
└── ROLE_MODE_CANON

L1 Framework
├── ARCH_BLUEPRINT_MASTER
└── PROJECT_DELIVERY_PIPELINE

L2 Project (DGSF)
├── PROJECT_DGSF.yaml (主规范)
├── Legacy Specs (引用)
│   ├── DGSF Architecture v3.0
│   ├── DGSF PanelTree v3.0.2
│   ├── DGSF SDF v3.1
│   ├── DGSF EA v3.1
│   ├── DGSF Rolling v3.0
│   └── DGSF Baseline v4.3
└── Adapter Layer (新增)
```

### 1.2 DGSF 层级到 AI Workflow OS 映射

| DGSF 层级 | DGSF 规范 | AI Workflow OS 位置 | 映射类型 |
|-----------|-----------|---------------------|----------|
| 母规范 | Architecture v3.0 | L2.project.architecture | 引用 |
| L2 | PanelTree v3.0.2 | L2.modules.paneltree | 引用 |
| L3 | SDF v3.1 | L2.modules.sdf | 引用 |
| L4 | EA v3.1 | L2.modules.ea | 引用 |
| L5 | Rolling v3.0 | L2.modules.rolling | 引用 |
| Baseline | Baseline v4.3 | L2.baselines | 引用 |
| DataEng | DataEng v4.2 | L2.data_engineering | 引用 |

---

## 2. 治理概念映射

### 2.1 核心概念对齐

| DGSF 概念 | AI Workflow OS 概念 | 对齐方式 |
|-----------|---------------------|----------|
| Rolling Windows | Pipeline Stages | 每个 Rolling 窗口 = 一个 Stage 周期 |
| Train/Val/OOS | Gate Checkpoints | Train→Val = G1, Val→OOS = G2 |
| Baseline A-H | Governance Invariants | 扩展为基线约束规则 |
| Drift Detection | Audit Events | 漂移检测结果写入审计日志 |
| Telemetry | Audit Trail | 系统遥测集成到审计系统 |

### 2.2 Authority 映射

| DGSF Authority | AI Workflow OS Authority | 说明 |
|----------------|--------------------------|------|
| Architecture v3.0 | L0 级约束 | 架构变更需 Owner 批准 |
| Layer Specs | L2 级约束 | 模块变更需 Reviewer 批准 |
| Config Changes | Speculative | 配置变更可自主执行 |
| Data Changes | Gate Required | 数据变更需通过 Gate |

### 2.3 Gate 映射

| DGSF Checkpoint | AI Workflow OS Gate | 触发条件 |
|-----------------|---------------------|----------|
| PanelTree Fit Complete | G_PANELTREE | 结构学习完成 |
| SDF Training Complete | G_SDF | SDF 模型训练完成 |
| EA Optimization Complete | G_EA | Pareto 前沿生成 |
| Rolling Window Complete | G_ROLLING | 单窗口 OOS 完成 |
| Baseline Comparison | G_BASELINE | 所有基线对比完成 |

---

## 3. 文件路径映射

### 3.1 规范文件映射

```yaml
spec_paths:
  architecture:
    source: "projects/dgsf/legacy/DGSF/docs/specs_v3/DGSF Architecture v3.0 _ Final.md"
    alias: "DGSF_ARCH_V3"
    
  paneltree:
    source: "projects/dgsf/legacy/DGSF/docs/specs_v3/DGSF PanelTree Layer Specification v3.0.2.md"
    alias: "DGSF_PANELTREE_V3"
    
  sdf:
    source: "projects/dgsf/legacy/DGSF/docs/specs_v3/DGSF SDF Layer Specification v3.1.md"
    alias: "DGSF_SDF_V3"
    
  ea:
    source: "projects/dgsf/legacy/DGSF/docs/specs_v3/DGSF EA Layer Specification v3.1.md"
    alias: "DGSF_EA_V3"
    
  rolling:
    source: "projects/dgsf/legacy/DGSF/docs/specs_v3/DGSF Rolling & Evaluation Specification v3.0.md"
    alias: "DGSF_ROLLING_V3"
    
  baseline:
    source: "projects/dgsf/legacy/DGSF/docs/specs_v3/DGSF Baseline System Specification v4.3.md"
    alias: "DGSF_BASELINE_V4"
```

### 3.2 代码路径映射

```yaml
code_paths:
  legacy_root: "projects/dgsf/legacy/DGSF"
  
  modules:
    paneltree: "src/dgsf/paneltree"
    sdf: "src/dgsf/sdf"
    ea: "src/dgsf/ea"
    rolling: "src/dgsf/rolling"
    backtest: "src/dgsf/backtest"
    dataeng: "src/dgsf/dataeng"
    
  configs: "configs"
  data: "data"
  tests: "tests"
```

### 3.3 适配层路径

```yaml
adapter_paths:
  root: "projects/dgsf/adapter"
  
  files:
    - "__init__.py"
    - "dgsf_adapter.py"      # 主适配器
    - "spec_mapper.py"       # 规范映射
    - "task_hooks.py"        # 任务钩子
    - "audit_bridge.py"      # 审计桥接
    - "config_loader.py"     # 配置加载
```

---

## 4. 接口定义

### 4.1 适配器接口

```python
class DGSFAdapter:
    """DGSF ↔ AI Workflow OS 适配器"""
    
    def get_spec(self, spec_id: str) -> dict:
        """获取 DGSF 规范"""
        
    def get_module(self, module_name: str) -> ModuleInterface:
        """获取 DGSF 模块"""
        
    def run_pipeline(self, config: dict) -> PipelineResult:
        """运行 DGSF 流水线"""
        
    def get_audit_events(self) -> List[AuditEvent]:
        """获取审计事件"""
```

### 4.2 任务钩子接口

```python
class DGSFTaskHooks:
    """DGSF 任务生命周期钩子"""
    
    def on_task_start(self, task_id: str):
        """任务开始时调用"""
        
    def on_task_finish(self, task_id: str, result: dict):
        """任务完成时调用"""
        
    def on_gate_check(self, gate_id: str) -> GateResult:
        """Gate 检查时调用"""
```

### 4.3 审计桥接接口

```python
class DGSFAuditBridge:
    """DGSF 审计日志桥接"""
    
    def log_event(self, event_type: str, data: dict):
        """记录审计事件"""
        
    def log_drift(self, drift_type: str, metrics: dict):
        """记录漂移检测结果"""
        
    def log_telemetry(self, telemetry_data: dict):
        """记录系统遥测"""
```

---

## 5. 配置映射

### 5.1 PROJECT_DGSF.yaml 扩展

```yaml
# v2.1.0 新增配置
spec_mappings:
  architecture: "DGSF_ARCH_V3"
  modules:
    paneltree: "DGSF_PANELTREE_V3"
    sdf: "DGSF_SDF_V3"
    ea: "DGSF_EA_V3"
    rolling: "DGSF_ROLLING_V3"
  baselines: "DGSF_BASELINE_V4"

gates:
  G_PANELTREE:
    type: "quality"
    checks:
      - "leaf_count >= 5"
      - "leaf_count <= 20"
      - "min_leaf_size >= 30"
  G_SDF:
    type: "quality"
    checks:
      - "pricing_error < 0.1"
      - "m_positivity == true"
  G_ROLLING:
    type: "quality"
    checks:
      - "oos_sharpe >= baseline_sharpe"
      - "drift_score < threshold"

adapter:
  enabled: true
  module: "projects.dgsf.adapter"
  hooks:
    - "on_task_start"
    - "on_task_finish"
    - "on_gate_check"
```

---

## 6. 实施检查清单

### 6.1 规范映射
- [x] 层级映射定义完成
- [x] 概念对齐定义完成
- [x] 路径映射定义完成
- [x] 接口定义完成

### 6.2 待实施
- [ ] 创建适配层代码
- [ ] 更新 PROJECT_DGSF.yaml v2.1.0
- [ ] 集成测试验证

---

**签署**: 张平台 (平台架构师)  
**日期**: 2026-02-01
