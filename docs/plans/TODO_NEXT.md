# AI Workflow OS - TODO Next Steps（项目编排版）

**文档ID**: TODO_NEXT_ORCHESTRATED  
**创建日期**: 2026-02-02  
**最后更新**: 2026-02-02T12:00:00Z  
**状态**: ACTIVE  
**编排依据**: 专家微型小组分析（Grady Booch + Gene Kim + Leslie Lamport）  
**证据来源**: git status, pytest结果, docs/audits/DRIFT_REPORT_20260202.md  
**WIP限制**: 最多3个任务同时进行

---

## ⚠️ 编排说明

本文档由**项目编排者（Project Orchestrator）**基于证据驱动分析生成，替代原有TODO_NEXT.md。所有任务优先级经过三位虚拟专家共识评审。

**关键发现**（基于2026-02-02T12:00:00Z状态扫描）:
1. ✅ **186个测试全部通过** - 核心功能稳定
2. 🔴 **6,572行未提交变更** - 存在工作丢失风险
3. 🟡 **CI导入路径错误** - 阻塞远程pipeline
4. 🟡 **4个不变量验证缺失** - INV-1/4/5/8未自动化

**优先级原则**:
- 🔴 **P0**: 阻塞性问题，立即执行（2小时内）
- 🟠 **P1**: 高价值任务，本周完成（3天内）
- 🟡 **P2**: 质量改进，可延后（2周内）

---

## 🎯 前10个优先任务（P0 → P1 → P2）

### 🔴 P0 任务（阻塞性 - 立即执行）

#### P0-1: 提交当前所有变更 ⏳ **NEXT STEP**
**预计工时**: 10分钟  
**依赖**: 无  
**专家共识**: Booch + Kim + Lamport 全部推荐（3/3）

**问题描述**:
根据 `git status` 输出，当前工作区有23个已修改文件和14个未追踪文件（总计6,572行新增），存在工作丢失风险。

**受影响文件**:
- **Modified**: [.github/workflows/ci.yml](../../.github/workflows/ci.yml), [README.md](../../README.md), [kernel/os.py](../../kernel/os.py), [kernel/mcp_server.py](../../kernel/mcp_server.py), [docs/state/PROJECT_STATE.md](../state/PROJECT_STATE.md) 等23个
- **Untracked**: [docs/SYSTEM_INVARIANTS.md](../SYSTEM_INVARIANTS.md), [kernel/governance_action.py](../../kernel/governance_action.py), scripts/check_*.py 等14个

**操作步骤**:
```powershell
# 1. 审查变更（可选但推荐）
git diff --stat

# 2. 添加所有文件
git add -A

# 3. 提交（使用详细的多模块commit message）
git commit -m "chore(multi): consolidate drift fixes and governance enhancements

📦 New Modules:
- kernel/governance_action.py: Freeze/Acceptance operations (359 LOC)
- kernel/yaml_utils.py: YAML utilities with atomic writes
- kernel/tests/test_governance_action.py: 12 governance tests

🔧 Core Enhancements:
- Artifact locking in AgentSession (lock/unlock/get_holder)
- MCP Server: 22 tools (added lock_artifact, unlock_artifact)
- State Store: Enhanced concurrency tests (20 new tests)

📜 Governance & Scripts:
- scripts/check_wip_limit.py: INV-2 WIP limit verification
- scripts/check_mcp_interface.py: INV-9 MCP consistency check
- scripts/run_gate_g{2-6}.py: 5 gate execution scripts

📚 Documentation:
- docs/SYSTEM_INVARIANTS.md: 10 formal invariants
- docs/audits/DRIFT_REPORT_20260202.md: 23-item drift audit
- docs/plans/MINIMAL_PATCHLIST.md: 9-patch remediation plan

✅ Test Status: 186 tests passing (7.93s)
✅ Coverage: 71% (kernel/)

Co-authored-by: AI Claude <ai@anthropic.com>"
```

**验收标准**:
- [x] `git status` 显示 "nothing to commit, working tree clean"
- [x] `git log -1 --stat` 显示37个文件变更
- [x] Commit SHA生成成功

**验证方法**:
```powershell
git status                  # 预期: nothing to commit
git log -1 --oneline       # 预期: 显示新提交SHA
git log -1 --stat | wc -l  # 预期: >50行（大提交）
```

**为什么是P0-1？**
- ✅ 零依赖（无需等待其他任务）
- ✅ 解除阻塞（清空工作区才能安全操作）
- ✅ 风险最低（纯状态保存，无功能变更）
- ✅ 审计追溯（满足INV-5审计完整性）

---

#### P0-2: 修复kernel模块导入路径
**预计工时**: 1.5小时  
**依赖**: P0-1（需干净工作区）  
**专家共识**: Booch (架构) + Kim (CI/CD)

**问题描述**:
根据 [TODO_NEXT.md](TODO_NEXT.md#L65-L85) 和 CI失败日志，kernel/内部模块使用相对导入（如 `from audit import`），导致CI环境下导入失败（governance-check job exit code 1）。

**受影响文件**:
- [kernel/os.py](../../kernel/os.py#L12-L18): 7个相对导入
- [kernel/mcp_server.py](../../kernel/mcp_server.py#L31-L32): 3个相对导入
- [kernel/mcp_stdio.py](../../kernel/mcp_stdio.py): 2个相对导入
- [kernel/config.py](../../kernel/config.py): 4个相对导入

**操作步骤**:
1. 批量替换导入语句（使用multi_replace_string_in_file）
2. 运行pyright类型检查: `pyright kernel/`
3. 运行测试套件: `pytest kernel/tests/ -v`
4. 提交修复: `git commit -m "fix(kernel): use absolute imports for CI compatibility"`

**验收标准**:
- [x] 所有 `from xxx import` → `from kernel.xxx import`（kernel/内部）
- [x] pyright 无错误输出
- [x] pytest 186个测试全部通过
- [x] 创建 kernel/tests/test_imports.py 验证导入路径规范

**验证方法**:
```powershell
# 1. 类型检查
pyright kernel/ --project pyrightconfig.json

# 2. 测试套件
pytest kernel/tests/ -v --tb=short

# 3. 导入路径检查
python kernel/tests/test_imports.py
```

**详细修改计划**（见 [MINIMAL_PATCHLIST.md](MINIMAL_PATCHLIST.md#L54-L120)）

---

#### P0-3: 本地运行G3-G6门禁验证
**预计工时**: 30分钟  
**依赖**: P0-2（导入路径修复后才能运行）  
**专家共识**: Kim (流程前移)

**问题描述**:
[.github/workflows/ci.yml](../../.github/workflows/ci.yml#L200-L280) 已集成G3-G6门禁任务，但本地未验证，可能存在运行时错误导致推送后CI失败。

**操作步骤**:
```powershell
# 依次执行4个门禁脚本
python scripts/run_gate_g3.py --output text  # 架构一致性
python scripts/run_gate_g4.py --output text  # 文档完整性
python scripts/run_gate_g5.py --output text  # 变更审查
python scripts/run_gate_g6.py --output text  # 发布就绪检查
```

**验收标准**:
- [x] 所有脚本退出码为0（ERROR级别为0）
- [x] WARNING级别可接受（≤3个）
- [x] 输出包含明确的PASS/FAIL判断

**验证方法**:
```powershell
# 批量执行并检查退出码
foreach ($gate in 3..6) {
    Write-Host "Running Gate G$gate..."
    python scripts/run_gate_g$gate.py --output text
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Gate G$gate FAILED"
        exit 1
    }
}
Write-Host "✅ All gates passed"
```

---

### 🟠 P1 任务（高价值 - 本周完成）

#### P1-1: 实现INV-1验证脚本（状态转换合法性）
**预计工时**: 3小时  
**依赖**: 无  
**专家共识**: Lamport (形式化验证)

**问题描述**:
[docs/SYSTEM_INVARIANTS.md](../SYSTEM_INVARIANTS.md#L11) 定义了INV-1（任务状态机），但验证脚本 scripts/verify_state_transitions.py 仅在TODO中提及，实际未实现。

**实现规格**:
```python
# scripts/verify_state_transitions.py
import yaml
from pathlib import Path
from kernel.paths import get_state_dir, get_kernel_dir

def load_state_machine():
    """加载state_machine.yaml转换规则"""
    path = get_kernel_dir() / "state_machine.yaml"
    return yaml.safe_load(path.read_text())

def load_tasks():
    """加载所有任务及其事件历史"""
    tasks_path = get_state_dir() / "tasks.yaml"
    if not tasks_path.exists():
        return {}
    return yaml.safe_load(tasks_path.read_text()) or {}

def verify_transition(from_state, to_state, transitions):
    """验证状态转换是否合法"""
    allowed = transitions.get(from_state, [])
    return to_state in allowed

def main():
    sm = load_state_machine()
    tasks = load_tasks()
    violations = []
    
    for task_id, task_data in tasks.items():
        events = task_data.get("events", [])
        for i in range(len(events) - 1):
            from_state = events[i].get("status")
            to_state = events[i+1].get("status")
            if not verify_transition(from_state, to_state, sm["transitions"]):
                violations.append({
                    "task_id": task_id,
                    "from": from_state,
                    "to": to_state,
                    "timestamp": events[i+1].get("timestamp"),
                    "event_index": i+1
                })
    
    if violations:
        print(f"❌ Found {len(violations)} state machine violations:")
        for v in violations:
            print(f"  [{v['task_id']}] {v['from']} → {v['to']} @ {v['timestamp']} (event #{v['event_index']})")
        return 1
    else:
        print("✅ All task state transitions are valid")
        return 0

if __name__ == "__main__":
    exit(main())
```

**验收标准**:
- [x] 脚本创建完成（~100行）
- [x] 读取 [kernel/state_machine.yaml](../../kernel/state_machine.yaml)
- [x] 验证 [state/tasks.yaml](../../state/tasks.yaml) 所有任务事件
- [x] 输出格式清晰（任务ID + 违规转换 + 时间戳）
- [x] 集成到 [hooks/pre-push](../../hooks/pre-push)

**验证方法**:
```powershell
python scripts/verify_state_transitions.py
# 预期输出示例（如无违规）:
# ✅ All task state transitions are valid
```

---

#### P1-2: 实现INV-4验证脚本（时间戳单调性）
**预计工时**: 2小时  
**依赖**: 无  
**专家共识**: Lamport (因果一致性)

**问题描述**:
[docs/SYSTEM_INVARIANTS.md](../SYSTEM_INVARIANTS.md#L28) 定义了INV-4（事件时间戳单调性），但无自动化验证。

**实现规格**:
```python
# scripts/check_timestamp_monotonicity.py
import yaml
from pathlib import Path
from datetime import datetime
from kernel.paths import get_state_dir

def load_tasks():
    tasks_path = get_state_dir() / "tasks.yaml"
    if not tasks_path.exists():
        return {}
    return yaml.safe_load(tasks_path.read_text()) or {}

def parse_timestamp(ts_str):
    """解析ISO 8601时间戳"""
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

def main():
    tasks = load_tasks()
    violations = []
    
    for task_id, task_data in tasks.items():
        events = task_data.get("events", [])
        for i in range(len(events) - 1):
            ts1 = parse_timestamp(events[i].get("timestamp"))
            ts2 = parse_timestamp(events[i+1].get("timestamp"))
            if ts2 < ts1:
                violations.append({
                    "task_id": task_id,
                    "event1_index": i,
                    "event2_index": i+1,
                    "ts1": events[i].get("timestamp"),
                    "ts2": events[i+1].get("timestamp"),
                    "delta": (ts1 - ts2).total_seconds()
                })
    
    if violations:
        print(f"❌ Found {len(violations)} timestamp violations:")
        for v in violations:
            print(f"  [{v['task_id']}] Event {v['event1_index']} ({v['ts1']}) > Event {v['event2_index']} ({v['ts2']})")
            print(f"    Δ = {v['delta']:.2f} seconds backward")
        return 1
    else:
        print("✅ All event timestamps are monotonic")
        return 0

if __name__ == "__main__":
    exit(main())
```

**验收标准**:
- [x] 脚本创建完成（~80行）
- [x] 支持ISO 8601时间戳解析
- [x] 报告时间戳倒序及偏移量
- [x] 集成到pre-push hook

**验证方法**:
```powershell
python scripts/check_timestamp_monotonicity.py
```

---

#### P1-3: 清理过期session记录
**预计工时**: 1小时  
**依赖**: P0-1（提交后操作安全）  
**专家共识**: Lamport (状态一致性)

**问题描述**:
[state/sessions.yaml](../../state/sessions.yaml) 包含 expires_at < 当前时间且 state=active 的会话（如 sess-f6d22ba9, expires_at: 2026-02-02T04:41），违反生命周期不变量。

**操作步骤**:
```python
# 一次性清理脚本（可选择合并到os.py或独立运行）
import yaml
from pathlib import Path
from datetime import datetime, timezone

sessions_path = Path("state/sessions.yaml")
data = yaml.safe_load(sessions_path.read_text())

now = datetime.now(timezone.utc)
cleaned = 0

for session_id, session in data["sessions"].items():
    if session["state"] == "active":
        expires_at = datetime.fromisoformat(session["expires_at"].replace('Z', '+00:00'))
        if expires_at < now:
            session["state"] = "terminated"
            session["events"].append({
                "timestamp": now.isoformat(),
                "action": "session_terminated",
                "details": {"reason": "expired", "auto_cleanup": True}
            })
            cleaned += 1

sessions_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))
print(f"✅ Cleaned {cleaned} expired sessions")
```

**验收标准**:
- [x] 所有 active 且 expires_at < now 的会话改为 terminated
- [x] 添加 auto_cleanup 事件到事件历史
- [x] YAML格式保持一致

**验证方法**:
```powershell
# 检查无active过期会话
python -c "import yaml; from datetime import datetime, timezone; data = yaml.safe_load(open('state/sessions.yaml')); expired = [s for s in data['sessions'].values() if s['state'] == 'active' and datetime.fromisoformat(s['expires_at'].replace('Z', '+00:00')) < datetime.now(timezone.utc)]; print(f'Expired active sessions: {len(expired)}'); exit(len(expired))"
```

---

#### P1-4: 创建架构边界审计脚本
**预计工时**: 2.5小时  
**依赖**: 无  
**专家共识**: Booch (架构完整性)

**问题描述**:
根据 [PROJECT_STATE.md](../state/PROJECT_STATE.md#L2587) 提到的"单向依赖边界模糊"风险，需验证 kernel/ 不依赖 projects/。

**实现规格**:
```python
# scripts/check_dependency_direction.py
import ast
from pathlib import Path

def extract_imports(file_path):
    """提取Python文件中的所有import语句"""
    try:
        tree = ast.parse(file_path.read_text(encoding='utf-8'))
    except SyntaxError:
        return []
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports

def main():
    kernel_dir = Path("kernel")
    violations = []
    
    for py_file in kernel_dir.rglob("*.py"):
        imports = extract_imports(py_file)
        for imp in imports:
            if imp.startswith("projects."):
                violations.append({
                    "file": str(py_file),
                    "import": imp
                })
    
    if violations:
        print(f"❌ Found {len(violations)} dependency violations:")
        print("   (kernel/ must NOT import from projects/)")
        for v in violations:
            print(f"  {v['file']}: import {v['import']}")
        return 1
    else:
        print("✅ No reverse dependencies detected (kernel/ → projects/)")
        return 0

if __name__ == "__main__":
    exit(main())
```

**验收标准**:
- [x] 脚本创建完成（~70行）
- [x] 使用AST解析（而非正则）
- [x] 检测所有 `import projects.*` 或 `from projects. import`
- [x] 退出码非零表示违规

**验证方法**:
```powershell
python scripts/check_dependency_direction.py
# 预期: ✅ No reverse dependencies detected
```

---

### 🟡 P2 任务（质量改进 - 可延后）

#### P2-1: 补充README架构快速链接
**预计工时**: 30分钟  
**依赖**: 无  
**专家共识**: Booch (文档导航)

**问题描述**:
[README.md](../../README.md) 缺少到核心架构图的快速链接，增加新人onboarding成本。

**修改内容**:
在 README.md 添加"架构文档"章节（第120行附近）:
```markdown
## 📐 Architecture Documentation

- [Master Blueprint](docs/ARCH_BLUEPRINT_MASTER.mmd) - 系统架构总览
- [Kernel Runtime Flow](docs/KERNEL_V0_RUNTIME_FLOW.mmd) - 内核执行流程
- [Interface Layer Map](docs/INTERFACE_LAYER_MAP.mmd) - 接口层架构
- [Task State Machine](docs/TASK_STATE_MACHINE.mmd) - 任务状态转换
- [Spec Governance Model](docs/SPEC_GOVERNANCE_MODEL.mmd) - 规范治理模型
- [Security Trust Boundary](docs/SECURITY_TRUST_BOUNDARY.mmd) - 安全边界定义
```

**验收标准**:
- [x] 新增章节包含6个.mmd文件链接
- [x] 所有链接可访问（文件存在）
- [x] 格式与现有章节一致

**验证方法**:
手动review + 点击所有链接

---

#### P2-2: 创建度量收集脚本
**预计工时**: 3小时  
**依赖**: P1-1, P1-2（状态数据清洁后更准确）  
**专家共识**: Kim (可观测性)

**问题描述**:
缺少自动化度量收集，无法量化cycle time、lead time等关键指标。

**实现规格**:
```python
# scripts/collect_metrics.py
import yaml
import json
from pathlib import Path
from datetime import datetime
from kernel.paths import get_state_dir

def calculate_cycle_time(events):
    """计算从running到done的时长（小时）"""
    running_ts = None
    done_ts = None
    for event in events:
        if event.get("status") == "running" and not running_ts:
            running_ts = datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
        if event.get("status") in ["done", "delivered"] and not done_ts:
            done_ts = datetime.fromisoformat(event["timestamp"].replace('Z', '+00:00'))
    
    if running_ts and done_ts:
        return (done_ts - running_ts).total_seconds() / 3600
    return None

def main():
    tasks = yaml.safe_load((get_state_dir() / "tasks.yaml").read_text()) or {}
    
    metrics = {
        "total_tasks": len(tasks),
        "by_status": {},
        "cycle_times": [],
        "timestamp": datetime.now().isoformat()
    }
    
    for task_id, task_data in tasks.items():
        status = task_data.get("status", "unknown")
        metrics["by_status"][status] = metrics["by_status"].get(status, 0) + 1
        
        ct = calculate_cycle_time(task_data.get("events", []))
        if ct:
            metrics["cycle_times"].append({
                "task_id": task_id,
                "cycle_time_hours": round(ct, 2)
            })
    
    # 计算平均cycle time
    if metrics["cycle_times"]:
        avg = sum(t["cycle_time_hours"] for t in metrics["cycle_times"]) / len(metrics["cycle_times"])
        metrics["avg_cycle_time_hours"] = round(avg, 2)
    
    print(json.dumps(metrics, indent=2))
    return 0

if __name__ == "__main__":
    exit(main())
```

**验收标准**:
- [x] 输出JSON格式度量数据
- [x] 包含: total_tasks, by_status, cycle_times, avg_cycle_time
- [x] cycle_time计算准确（running → done时长）

**验证方法**:
```powershell
python scripts/collect_metrics.py | jq .
# 预期: JSON对象包含所有指标
```

---

#### P2-3: 推送到远程并验证CI
**预计工时**: 15分钟（等待时间）  
**依赖**: P0-1 + P0-2 + P0-3（确保本地验证通过）  
**专家共识**: Kim (持续集成)

**操作步骤**:
```powershell
# 1. 推送到远程
git push origin feature/router-v0

# 2. 监控CI状态
# 访问: https://github.com/<org>/AI-Workflow-OS/actions
# 或使用gh CLI:
gh run watch
```

**验收标准**:
- [x] 推送成功（无冲突）
- [x] GitHub Actions 所有任务显示绿色✅
- [x] 特别关注: governance-check, gate-g2-sanity, gate-g3至gate-g6

**验证方法**:
```powershell
# 检查最新workflow run状态
gh run list --branch feature/router-v0 --limit 1
# 预期: STATUS = completed, CONCLUSION = success
```

---

## 📊 任务依赖图（Dependency Graph）

```
P0-1 (提交变更) → P0-2 (修复导入) → P0-3 (门禁验证) → P2-3 (推送验证)
                    ↓
                    P1-3 (清理session)

P1-1 (INV-1验证) ──┐
P1-2 (INV-4验证) ──┼→ P2-2 (度量收集)
P1-4 (边界审计) ──┘

P2-1 (README链接) - 独立任务
```

---

## ✅ 执行检查清单（Execution Checklist）

完成每个任务后，更新此检查清单：

- [ ] **P0-1**: git commit完成，工作区干净
- [ ] **P0-2**: 导入路径修复，186测试通过
- [ ] **P0-3**: G3-G6本地验证通过
- [ ] **P1-1**: verify_state_transitions.py创建并运行
- [ ] **P1-2**: check_timestamp_monotonicity.py创建并运行
- [ ] **P1-3**: 过期session清理完成
- [ ] **P1-4**: check_dependency_direction.py创建并运行
- [ ] **P2-1**: README架构链接添加
- [ ] **P2-2**: collect_metrics.py创建并运行
- [ ] **P2-3**: 远程CI全部通过

---

## 🚫 停止做（Stop Doing）清单

基于专家反模式识别：

1. ⚠️ **停止在单分支累积多个unrelated功能** - 应使用topic branches（如 feature/INV-1-validator）
2. ⚠️ **停止跳过本地CI模拟** - 推送前必须运行 `scripts/run_gate_g*.py`
3. ⚠️ **停止在未定义验收标准时标记VERIFIED** - 强制填写DONE_CRITERIA
4. ⚠️ **停止硬编码路径** - 必须使用 kernel/paths.py
5. ⚠️ **停止手动修改YAML** - 使用 StateStore API 确保原子性

---

## 📅 预估时间线

| 时间段 | 任务 | 累计工时 |
|-------|------|---------|
| 0-2h | P0-1, P0-2 | 2h |
| 2-3h | P0-3 | 0.5h |
| 3-6h | P1-1 | 3h |
| 6-8h | P1-2 | 2h |
| 8-9h | P1-3 | 1h |
| 9-11.5h | P1-4 | 2.5h |
| 11.5-12h | P2-1 | 0.5h |
| 12-15h | P2-2 | 3h |
| 15-15.25h | P2-3 | 0.25h |

**总计**: ~15小时（约2个工作日，假设8h/天）

---

## 📝 元数据

**决策框架**: 证据驱动 + 专家共识  
**证据来源**:
- `git status` (2026-02-02T12:00:00Z)
- `pytest kernel/tests/` 输出
- [docs/audits/DRIFT_REPORT_20260202.md](../audits/DRIFT_REPORT_20260202.md)
- [docs/state/PROJECT_STATE.md](../state/PROJECT_STATE.md)

**专家小组成员**:
- Grady Booch: 架构完整性专家
- Gene Kim: DevOps与流程专家
- Leslie Lamport: 形式化验证专家

**下次review时间**: 2026-02-03T00:00:00Z（P0任务完成后）

---

**Status**: ⏳ PENDING EXECUTION  
**Next Action**: 执行 P0-1（提交当前所有变更）
