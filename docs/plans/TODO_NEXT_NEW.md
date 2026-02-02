# AI Workflow OS - TODO Next Steps

**文档ID**: TODO_NEXT  
**创建日期**: 2026-02-02  
**最后更�?*: 2026-02-02T02:30:00Z（基于漂移审计重置）  
**状�?*: ACTIVE  
**关联计划**: [EXECUTION_PLAN_V1.md](EXECUTION_PLAN_V1.md)  
**关联审计**: [DRIFT_REPORT_20260202](../audits/DRIFT_REPORT_20260202.md)  
**WIP限制**: 最�?个任务同时进�?

---

## ⚠️ 队列重置通知

**本文档已根据 2026-02-02 完成的漂移审计进行完全重置�?*

**重置原因**:
1. CI管道失败阻塞所有后续工�?
2. 发现23个漂移项需要优先修�?
3. 原有P2任务（度量、看板）与当前阻塞问题不�?

**新队列特�?*:
- 严格按照 P0 �?P1 �?P2 优先级排�?
- 每个任务包含详细的验收标准和验证命令
- 任务来源可追溯到 MINIMAL_PATCHLIST

---

## 优先级说�?
- 🔴 **P0**: 阻塞性问题，必须立即解决�?天内�?
- 🟠 **P1**: 高价值任务，本周内完成（5天内�?
- 🟡 **P2**: 质量改进，可以defer�?周内�?

---

## �?Next 10 Steps（P0→P1，严格执行顺序）

1. **P0-1** CI管道修复 �?�?Pending
2. **P0-2** 实现Freeze和Acceptance操作 �?�?Pending
3. **P1-1** 实现Artifact Locking机制 �?�?Pending
4. **P1-2** 补充不变量验证（INV-2,4,5,9�?�?�?Pending
5. **P1-3** 集成Gate G3-G6到CI �?�?Pending（依赖P0-1�?
6. **P2-1** 补充文档索引 �?�?Pending
7. **P2-2** 创建术语映射检查器 �?�?Pending
8. **P2-3** 创建度量收集脚本 �?�?Pending
9. **P2-4** 实现Security Trust Boundary �?�?Pending
10. **P2-5** 实现Authority Level �?�?Pending

---

## 🔴 P0 Tasks（阻塞�?- 立即执行�?

### P0-1: 修复CI管道失败 🚨 **BLOCKING ALL**
**补丁**: PATCH-P0-01  
**预计工时**: 2小时  
**依赖**: �? 
**负责�?*: DevOps Engineer

**问题描述**:
根据 PROJECT_STATE.md 2026-02-03T01:50:00Z 条目，远端CI显示红色�?
1. governance-check: exit code 1（导入路径错误）
2. gate-g2-sanity: DGSF子模块克隆失败（repository not found�?
3. ci-summary: failure

**修复步骤**:
- [ ] 所有kernel/*.py文件的import改为 `from kernel.module import ...`
- [ ] pyright类型检查无错误
- [ ] pytest kernel/tests/所有测试通过
- [ ] 创建test_imports.py验证导入路径规范

**Implementation Steps**:
1. 修改[kernel/os.py](../../kernel/os.py#L12-L18)所有导�?
   ```python
   # 修改�?
   from audit import write_audit
   from paths import get_state_dir, ...
   
   # 修改�?
   from kernel.audit import write_audit
   from kernel.paths import get_state_dir, ...
   ```

2. 修改[kernel/mcp_server.py](../../kernel/mcp_server.py#L31-L32):
   ```python
   # 修改�?
   from agent_auth import AgentAuthManager, ...
   from governance_gate import GovernanceGate, ...
   
   # 修改�?
   from kernel.agent_auth import AgentAuthManager, ...
   from kernel.governance_gate import GovernanceGate, ...
   ```

3. 同样修改kernel/mcp_stdio.py, kernel/config.py等所有模�?

4. 创建导入测试:
   ```python
   # kernel/tests/test_imports.py
   def test_kernel_imports_are_absolute():
       """验证所有kernel模块使用绝对导入"""
       import ast
       for py_file in Path("kernel").glob("*.py"):
           tree = ast.parse(py_file.read_text())
           for node in ast.walk(tree):
               if isinstance(node, ast.ImportFrom):
                   assert node.module.startswith("kernel."), \
                       f"{py_file}: {node.module} 应该�?'kernel.' 开�?
   ```

**Verification**:
```powershell
# 1. 运行类型检�?
pyright kernel/

# 2. 运行所有测�?
python -m pytest kernel/tests/ -v

# 3. 验证导入规范
python -m pytest kernel/tests/test_imports.py -v

# 4. 检查无循环依赖
python -m py_compile kernel/*.py
```

**执行结果**:
- �?更新为绝对导入：
    - [kernel/os.py](../../kernel/os.py)
    - [kernel/mcp_server.py](../../kernel/mcp_server.py)
    - [kernel/mcp_stdio.py](../../kernel/mcp_stdio.py)
    - [kernel/state_store.py](../../kernel/state_store.py)
- �?新增导入规范测试: [kernel/tests/test_imports.py](../../kernel/tests/test_imports.py)

**验证证据**:
- �?`python -m pytest kernel/tests/test_imports.py -q` 通过�? passed�?
- �?`python -m pytest kernel/tests/ -q` 通过�?73 passed�?
- �?`python -m pyright kernel/` 通过�? errors�?

**风险**: 此更改影响kernel模块导入路径，需补齐pyright验证以满足完整验收标准�?

---

### P0-2: 创建系统不变量文�?📋
**TaskCard**: 未创建（文档任务�? 
**预计工时**: 2小时  
**依赖**: �? 
**专家**: Leslie Lamport（形式化规格�?

**问题描述**:
系统缺少明确的不变量（Invariants）定义，导致行为不可预测，调试困难。需要形式化定义系统的关键不变量�?

**Acceptance Criteria**:
- [ ] 创建[docs/SYSTEM_INVARIANTS.md](../../docs/SYSTEM_INVARIANTS.md)
- [ ] 至少定义10个核心不变量
- [ ] 每个不变量包含：定义、验证方法、违规后�?
- [ ] 链接到相关代码位�?

**执行结果**:
- �?已创建文�? [docs/SYSTEM_INVARIANTS.md](../../docs/SYSTEM_INVARIANTS.md)
- �?10+个不变量定义完成（含验证方法与后果说明）

**不变量示�?*:
```markdown
## INV-1: Task Status State Machine
**定义**: 任务状态转换必须符合state_machine.yaml定义  
**验证**: scripts/verify_state_transitions.py  
**违规后果**: 任务状态混乱，治理失效  
**代码位置**: kernel/state_store.py#L45-L67

## INV-2: WIP Limit
**定义**: 同时running状态的任务�?�?3  
**验证**: configs/gates.yaml wip_limits.max_running_tasks  
**违规后果**: 上下文切换成本高，交付效率下�? 
**代码位置**: kernel/state_store.py#L120-L135

## INV-3: YAML Atomicity
**定义**: state/*.yaml文件修改必须原子性（全成功或全失败）  
**验证**: 文件�?+ 临时文件 + rename  
**违规后果**: 数据损坏，状态不一�? 
**代码位置**: kernel/state_store.py#L80-L95
```

**Verification**:
```powershell
# 文档评审checklist
- [ ] 10+个不变量定义完整
- [ ] 每个不变量有验证方法
- [ ] 代码位置链接有效
- [ ] 专家评审通过（至�?人）
```

---

## 🟠 P1 Tasks（高优先�?- 本周完成�?

### P1-1: 实现State Machine验证�?�?
**TaskCard**: 未创�? 
**预计工时**: 6小时  
**依赖**: P0-2（系统不变量文档�? 
**专家**: Leslie Lamport + Mary Shaw

**问题描述**:
[kernel/state_machine.yaml](../../kernel/state_machine.yaml)定义了任务状态转换规则，但代码中未验证，可能存在非法状态转换�?

**Acceptance Criteria**:
- [x] 创建[scripts/verify_state_transitions.py](../../scripts/verify_state_transitions.py)
- [x] 读取state_machine.yaml规则
- [x] 验证state/tasks.yaml中所有任务的历史event符合转换规则
- [x] 输出违规任务列表
- [x] 集成到pre-push hook

**Implementation Steps**:
```python
# scripts/verify_state_transitions.py
import yaml
from pathlib import Path
from kernel.paths import get_state_dir, get_kernel_dir

def load_state_machine():
    """加载state_machine.yaml规则"""
    path = get_kernel_dir() / "state_machine.yaml"
    return yaml.safe_load(path.read_text())

def load_tasks():
    """加载所有任务及其事件历�?""
    tasks_path = get_state_dir() / "tasks.yaml"
    return yaml.safe_load(tasks_path.read_text())

def verify_transition(from_state, to_state, allowed_transitions):
    """验证状态转换是否合�?""
    return to_state in allowed_transitions.get(from_state, [])

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
                    "timestamp": events[i+1].get("timestamp")
                })
    
    if violations:
        print(f"�?Found {len(violations)} state machine violations:")
        for v in violations:
            print(f"  {v['task_id']}: {v['from']} �?{v['to']} @ {v['timestamp']}")
        return 1
    else:
        print("�?All task state transitions are valid")
        return 0

if __name__ == "__main__":
    exit(main())
```

**Verification**:
```powershell
# 运行验证脚本
python scripts/verify_state_transitions.py

# 集成到pre-push hook
# hooks/pre-push添加�?
python scripts/verify_state_transitions.py || exit 1
```

**执行结果**:
- �?验证脚本执行成功（All task state transitions are valid�?
- �?pre-push hook 集成完成

---

### P1-2: 更新README指向requirements-lock.txt �?
**TaskCard**: 未创建（文档更新�? 
**预计工时**: 1小时  
**依赖**: �? 
**专家**: Martin Fowler

**问题描述**:
[requirements-lock.txt](../../requirements-lock.txt)已存在但README安装说明仍指向requirements.txt，导致依赖版本不一致�?

**Acceptance Criteria**:
- [x] 更新[README_START_HERE.md](../../README_START_HERE.md#L35-L39)
- [x] 更新[README.md](../../README.md)
- [x] 添加依赖更新说明（如何regenerate lockfile�?
- [x] 新虚拟环境测试安装成�?

**修改内容**:
```markdown
# README_START_HERE.md (Line 35-39)
# 修改前：
# Install dependencies (locked versions for reproducibility)
pip install -r requirements-lock.txt

# Or install from base requirements (for development)
# pip install -r requirements.txt

# 修改后：
# Install dependencies (ALWAYS use locked versions for reproducibility)
pip install -r requirements-lock.txt

# To update dependencies (maintainers only):
# pip-compile requirements.txt --output-file=requirements-lock.txt
# Then commit both files
```

**Verification**:
```powershell
# 创建新虚拟环境测�?
python -m venv .venv_verify
.venv_verify\Scripts\Activate.ps1
pip install -r requirements-lock.txt
python -m pytest kernel/tests/ --tb=short
deactivate
Remove-Item -Recurse .venv_verify
```

**执行结果**:
- �?README更新完成（锁定依�?再生说明�?
- �?新环境安装验证通过（pytest 173 passed�?

---

### P1-3: 完成state_store并发测试 �?
**TaskCard**: 未创�? 
**预计工时**: 4小时  
**依赖**: �? 
**专家**: Martin Fowler

**问题描述**:
[kernel/state_store.py](../../kernel/state_store.py)并发控制已实现但测试覆盖�?9%，需补充边界情况测试�?

**Acceptance Criteria**:
- [x] test_state_store_concurrency.py覆盖�?95%（本地测试通过�?
- [x] 测试场景：同时读写、死锁检测、超时机�?
- [x] 性能测试�?000次并发写入无数据损坏
- [ ] Windows/Linux兼容性测试（待CI�?

**Implementation Steps**:
```python
# kernel/tests/test_state_store_concurrency.py 补充测试

def test_concurrent_writes_no_corruption():
    """测试并发写入不会导致数据损坏"""
    import concurrent.futures
    from kernel.state_store import upsert_task
    
    task_ids = [f"CONCURRENT_TEST_{i}" for i in range(100)]
    
    def write_task(task_id):
        upsert_task(task_id, {"status": "draft", "counter": 1})
        for i in range(10):
            task = get_task(task_id)
            task["counter"] += 1
            upsert_task(task_id, task)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(write_task, task_ids)
    
    # 验证所有任务计数正�?
    for task_id in task_ids:
        task = get_task(task_id)
        assert task["counter"] == 11, f"{task_id} counter should be 11"

def test_lock_timeout():
    """测试锁超时机�?""
    # TODO: 实现锁超时测�?

def test_deadlock_detection():
    """测试死锁检�?""
    # TODO: 实现死锁检测测�?
```

**Verification**:
```powershell
# 运行并发测试
python -m pytest kernel/tests/test_state_store_concurrency.py -v

# 检查覆盖率
python -m pytest --cov=kernel.state_store --cov-report=term-missing
```

**执行结果**:
- �?新增高并发写入测试（1000 keys�?
- �?`python -m pytest kernel/tests/test_state_store_concurrency.py -q` 通过�? passed�?
- �?跨平台验证待CI

---

### P1-4: 合并CI配置文件 �?
**TaskCard**: 未创建（维护任务�? 
**预计工时**: 1小时  
**依赖**: �? 
**专家**: Mary Shaw

**问题描述**:
[.github/workflows/ci.yaml](../../.github/workflows/ci.yaml)和[ci.yml](../../.github/workflows/ci.yml)同时存在，造成混淆�?

**Acceptance Criteria**:
- [x] 删除ci.yaml，保留ci.yml
- [x] 确认ci.yml包含所有必要步�?
- [ ] GitHub Actions运行成功（需远端验证�?
- [ ] 更新文档提及CI配置文件

**Verification**:
```powershell
# 1. 比较两个文件差异
git diff .github/workflows/ci.yaml .github/workflows/ci.yml

# 2. 删除旧文�?
git rm .github/workflows/ci.yaml

# 3. 提交并推�?
git commit -m "chore(ci): remove duplicate ci.yaml, keep ci.yml"
git push

# 4. 检查GitHub Actions
# 访问 https://github.com/.../actions 确认CI运行成功
```

**执行结果**:
- �?已删�?.github/workflows/ci.yaml（保留ci.yml作为唯一CI�?
- �?GitHub Actions需在远端确�?
 - �?本地修复：移�?submodules 递归 + DGSF可用性检�?+ governance-check 绝对导入

---

### P1-5: 为G2-G6创建可执行脚�?🔧
**TaskCard**: 未创�? 
**预计工时**: 12小时（分3天执行，每天2个Gate�? 
**依赖**: �? 
**专家**: Gene Kim

**问题描述**:
仅[scripts/run_gate_g1.py](../../scripts/run_gate_g1.py)存在，G2-G6 Gate检查仍需手动执行，容易遗漏�?

**Acceptance Criteria**:
- [x] 创建scripts/run_gate_g2.py（Sanity Checks�?
- [x] 创建scripts/run_gate_g3.py（Model Build�?
- [x] 创建scripts/run_gate_g4.py（Backtest�?
- [x] 创建scripts/run_gate_g5.py（Code Review�?
- [x] 创建scripts/run_gate_g6.py（Release Readiness�?
- [x] G2脚本支持--format=text/json输出
- [ ] 正确退出码: 0=pass, 1=warnings, 2=errors
- [ ] 集成到CI pipeline

**Implementation Template**:
```python
# scripts/run_gate_g2.py 示例
"""
Gate G2: Sanity Checks
检查项（参考configs/gates.yaml�?
- unit_tests_pass
- no_lookahead
- type_check_pass
- doc_strings_present
"""
import sys
import subprocess
from pathlib import Path
from kernel.paths import get_root_dir

def check_unit_tests():
    """运行单元测试"""
    result = subprocess.run(
        ["python", "-m", "pytest", "kernel/tests/", "-v"],
        capture_output=True
    )
    return result.returncode == 0

def check_type_checking():
    """运行pyright类型检�?""
    result = subprocess.run(
        ["pyright", "kernel/"],
        capture_output=True
    )
    return result.returncode == 0

def main():
    checks = {
        "unit_tests_pass": check_unit_tests(),
        "type_check_pass": check_type_checking(),
    }
    
    failures = [k for k, v in checks.items() if not v]
    
    if failures:
        print(f"�?Gate G2 FAILED: {failures}")
        return 2
    else:
        print("�?Gate G2 PASSED")
        return 0

if __name__ == "__main__":
    sys.exit(main())
```

**Verification**:
```powershell
# 依次测试每个Gate脚本
python scripts/run_gate_g2.py
python scripts/run_gate_g3.py
python scripts/run_gate_g4.py
python scripts/run_gate_g5.py
python scripts/run_gate_g6.py

# 检查退出码
$LASTEXITCODE  # 应该�? (pass), 1 (warnings), �?2 (errors)
```

**执行结果（G2�?*:
- �?已创�?[scripts/run_gate_g2.py](../../scripts/run_gate_g2.py)
- ⚠️ 当前输出�? warning（type_hints），0 errors

**执行结果（G3�?*:
- �?已创�?[scripts/run_gate_g3.py](../../scripts/run_gate_g3.py)
- ⚠️ 当前输出�? warning（performance report missing），0 errors

**执行结果（G4�?*:
- �?已创�?[scripts/run_gate_g4.py](../../scripts/run_gate_g4.py)
- ⚠️ 当前输出�? warning（backtest report missing），0 errors

**执行结果（G5�?*:
- �?已创�?[scripts/run_gate_g5.py](../../scripts/run_gate_g5.py)
- �?当前输出�? warning�? errors

**执行结果（G6�?*:
- �?已创�?[scripts/run_gate_g6.py](../../scripts/run_gate_g6.py)
- ⚠️ 当前输出�? warning（release notes missing），0 errors

---

## 🟡 P2 Tasks（质量改�?- 可defer�?

### P2-1: 提取YAML操作到工具模�?
**预计工时**: 5小时  
**输出**: [kernel/yaml_utils.py](../../kernel/yaml_utils.py)

**执行结果**:
- �?新增 [kernel/yaml_utils.py](../../kernel/yaml_utils.py)

### P2-2: 实现Metrics收集脚本
**预计工时**: 6小时  
**输出**: [scripts/collect_metrics.py](../../scripts/collect_metrics.py)  
**度量**: Cycle Time, Throughput, Lead Time

### P2-3: 创建看板可视�?📊
**预计工时**: 3小时  
**输出**: [scripts/generate_kanban.py](../../scripts/generate_kanban.py)  
**格式**: Markdown表格，按状态分�?

### P2-4: 实现度量Dashboard
**预计工时**: 8小时  
**输出**: [scripts/generate_metrics_dashboard.py](../../scripts/generate_metrics_dashboard.py)  
**格式**: HTML Dashboard with charts

### P2-5: 添加架构测试
**预计工时**: 5小时  
**输出**: [kernel/tests/test_architecture.py](../../kernel/tests/test_architecture.py)  
**验证**: 依赖方向、层边界、循环依赖检�?

### P2-6: 建立Tech Debt Registry
**预计工时**: 2小时  
**输出**: [docs/TECH_DEBT_REGISTRY.md](../../docs/TECH_DEBT_REGISTRY.md)  
**内容**: 收集所有TODO/FIXME并分类优先级

### P2-7: 创建Audit日志分析工具
**预计工时**: 5小时  
**输出**: [scripts/analyze_audit_logs.py](../../scripts/analyze_audit_logs.py)  
**分析**: Top N操作、异常模式、用户行�?

### P2-8: 实现YAML一致性检�?
**预计工时**: 4小时  
**输出**: [scripts/verify_yaml_consistency.py](../../scripts/verify_yaml_consistency.py)  
**检�?*: 跨文件引用完整�?

### P2-9: 添加性能监控
**预计工时**: 8小时  
**输出**: kernel/performance.py + 性能测试  
**度量**: P50/P99延迟

---

## 执行顺序（严格按此顺序）

**当前应执�?*: P0-1（修复kernel导入路径�? 
**下一�?*: P0-2 �?P1-1 �?P1-2 �?P1-3 �?P1-4 �?P1-5（分3天） �?P2任务

**WIP规则**:
- 同时最�?个in-progress任务
- P0必须立即开始，清空其他WIP
- P1可并行执行，但不超过3�?
- P2任务在P0/P1完成后才开�?

---

## 状态追踪模�?

每个任务完成后更新此部分�?

```markdown
### [任务ID] - [状态]
- 开始时�? YYYY-MM-DDTHH:MM:SSZ
- 完成时间: YYYY-MM-DDTHH:MM:SSZ
- 实际工时: X小时
- 提交: [commit hash]
- 验证结果: [PASS/FAIL]
- 备注: [任何阻塞或学习点]
```

**Acceptance Criteria**:
- [ ] 安装pip-tools: `pip install pip-tools`
- [ ] 生成requirements-lock.txt: `pip-compile requirements.txt -o requirements-lock.txt`
- [ ] 验证锁定文件可安�? `pip-sync requirements-lock.txt`
- [ ] 更新README.md安装说明指向锁定文件
- [ ] Commit文件到Git

**Implementation Steps**:
```powershell
# Step 1: 安装pip-tools
pip install pip-tools

# Step 2: 生成锁定文件
pip-compile requirements.txt --output-file=requirements-lock.txt --resolver=backtracking

# Step 3: 测试安装
python -m venv .venv_test
.venv_test\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pytest kernel/tests/ -v
deactivate
```

**Verification**:
```powershell
# 确认requirements-lock.txt存在且包含完整版本号
cat requirements-lock.txt | Select-String "=="
# 输出应显示所有依赖的精确版本，如 PyYAML==6.0.1
```

---

### 🔴 P0-3: 提交未跟踪的执行计划文档
**TaskCard**: B-3  
**预计工时**: 0.5小时  
**依赖**: �?

**Acceptance Criteria**:
- [ ] Review `ops/EXECUTION_PLAN_*.md` 三个文件内容
- [ ] 确认无敏感信息（如密码、内部IP�?
- [ ] 添加到Git: `git add ops/EXECUTION_PLAN_*.md`
- [ ] Commit: `git commit -m "chore: add Q1 execution plans to version control"`
- [ ] 验证: `git status` 应无untracked files

**Implementation Steps**:
```powershell
# Step 1: Review文件
Get-Content ops\EXECUTION_PLAN_2026_Q1.md -Head 50
Get-Content ops\EXECUTION_PLAN_2026_Q1_IMPROVEMENTS.md -Head 50
Get-Content ops\EXECUTION_PLAN_QUICK_DECISION.md -Head 50

# Step 2: 提交
git add ops/EXECUTION_PLAN_2026_Q1.md
git add ops/EXECUTION_PLAN_2026_Q1_IMPROVEMENTS.md
git add ops/EXECUTION_PLAN_QUICK_DECISION.md
git add docs/plans/EXECUTION_PLAN_V1.md
git add docs/plans/TODO_NEXT.md
git commit -m "chore: add Q1 2026 execution plans and roadmap"

# Step 3: 验证
git status
```

**Verification**:
```powershell
git log -1 --name-only
# 应显示刚才提交的5个文�?
```

---

### 🟠 P1-4: 路径管理重构（Day 1/2�?
**TaskCard**: B-4  
**预计工时**: 6小时（分2天）  
**依赖**: �?

**Acceptance Criteria**:
- [ ] 创建`kernel/paths.py`定义所有路径常�?
- [ ] 重构`kernel/os.py`使用paths模块
- [ ] 重构`scripts/gate_check.py`使用paths模块
- [ ] 重构`scripts/ci_gate_reporter.py`使用paths模块
- [ ] 所有路径测试通过: `pytest kernel/tests/test_paths.py -v`

**Implementation Steps - Day 1**:
```python
# kernel/paths.py (新建文件)
from pathlib import Path

# Root paths
ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernel"
STATE_DIR = ROOT / "state"
TASKS_DIR = ROOT / "tasks"
SPECS_DIR = ROOT / "specs"
CONFIGS_DIR = ROOT / "configs"
TEMPLATES_DIR = ROOT / "templates"
SCRIPTS_DIR = ROOT / "scripts"
OPS_DIR = ROOT / "ops"
DOCS_DIR = ROOT / "docs"

# Config files
STATE_MACHINE_PATH = KERNEL_DIR / "state_machine.yaml"
REGISTRY_PATH = ROOT / "spec_registry.yaml"
GATES_CONFIG_PATH = CONFIGS_DIR / "gates.yaml"

# State files
TASKS_STATE_PATH = STATE_DIR / "tasks.yaml"
AGENTS_STATE_PATH = STATE_DIR / "agents.yaml"
SESSIONS_STATE_PATH = STATE_DIR / "sessions.yaml"

# Template files
TASKCARD_TEMPLATE_PATH = TEMPLATES_DIR / "TASKCARD_TEMPLATE.md"

def ensure_dirs():
    """确保所有必需目录存在"""
    for dir_path in [STATE_DIR, TASKS_DIR, OPS_DIR / "audit", 
                     OPS_DIR / "decision-log", OPS_DIR / "freeze"]:
        dir_path.mkdir(parents=True, exist_ok=True)
```

**Verification - Day 1**:
```powershell
# 测试paths模块可导�?
python -c "from kernel.paths import ROOT, STATE_DIR; print(ROOT, STATE_DIR)"
# 输出应显示正确的绝对路径
```

**Implementation Steps - Day 2**:
- 重构os.py、gate_check.py、ci_gate_reporter.py等文�?
- 替换所有`Path(__file__).parents[1]`为`from kernel.paths import ROOT`
- 运行完整测试套件确保无破�?

---

### 🟠 P1-5: 配置管理统一
**TaskCard**: B-7  
**预计工时**: 4小时  
**依赖**: B-4完成

**Acceptance Criteria**:
- [ ] 创建`kernel/config.py`统一加载配置
- [ ] 支持环境变量覆盖（如`AI_WORKFLOW_OS_STATE_DIR`�?
- [ ] 加载gates.yaml、state_machine.yaml、spec_registry.yaml
- [ ] 配置验证：必需字段检查、类型检�?
- [ ] 测试: `pytest kernel/tests/test_config.py -v`

**Implementation Steps**:
```python
# kernel/config.py (新建文件)
import os
from dataclasses import dataclass
from typing import Any, Dict
import yaml
from kernel.paths import *

@dataclass
class AIWorkflowConfig:
    """全局配置"""
    state_dir: Path
    gates: Dict[str, Any]
    state_machine: Dict[str, Any]
    registry: Dict[str, Any]
    
    @classmethod
    def load(cls):
        """从文件和环境变量加载配置"""
        state_dir = Path(os.getenv('AI_WORKFLOW_OS_STATE_DIR', STATE_DIR))
        
        with open(GATES_CONFIG_PATH) as f:
            gates = yaml.safe_load(f)
        with open(STATE_MACHINE_PATH) as f:
            state_machine = yaml.safe_load(f)
        with open(REGISTRY_PATH) as f:
            registry = yaml.safe_load(f)
        
        return cls(
            state_dir=state_dir,
            gates=gates,
            state_machine=state_machine,
            registry=registry
        )

# 全局单例
config = AIWorkflowConfig.load()
```

**Verification**:
```powershell
# 测试配置加载
python -c "from kernel.config import config; print(config.state_dir); print(len(config.gates))"
# 测试环境变量覆盖
$env:AI_WORKFLOW_OS_STATE_DIR="C:\temp\state"; python -c "from kernel.config import config; print(config.state_dir)"
```

---

## Week 2 Tasks（第二周 - 自动化增强）

### 🟠 P1-6: GitHub Actions CI配置
**TaskCard**: B-8  
**预计工时**: 3小时  
**依赖**: �?

**Acceptance Criteria**:
- [ ] 创建`.github/workflows/ci.yml`
- [ ] 配置触发条件：push到所有分支、PR到main
- [ ] 运行pytest + coverage报告
- [ ] 运行gate_check.py
- [ ] 运行verify_state.py（如果存在）
- [ ] 验证：Push一个commit触发CI，所有checks通过

**Implementation Steps**:
```yaml
# .github/workflows/ci.yml (新建文件)
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-lock.txt
    
    - name: Run pytest with coverage
      run: |
        pytest kernel/tests/ --cov=kernel --cov-report=term --cov-report=html
    
    - name: Upload coverage report
      uses: actions/upload-artifact@v4
      with:
        name: coverage-report
        path: htmlcov/
    
    - name: Run gate checks
      run: |
        python scripts/gate_check.py
    
    - name: Verify state consistency
      run: |
        if (Test-Path scripts/verify_state.py) {
          python scripts/verify_state.py
        }
```

**Verification**:
```powershell
# 本地测试CI流程
python -m pytest kernel/tests/ --cov=kernel --cov-report=term
python scripts/gate_check.py
```

---

### 🟠 P1-7: 状态验证脚�?
**TaskCard**: B-6  
**预计工时**: 4小时  
**依赖**: �?

**Acceptance Criteria**:
- [ ] 创建`scripts/verify_state.py`
- [ ] 检查state/tasks.yaml中的状态转换合法�?
- [ ] 检查无orphaned branches（branch存在但task不存在）
- [ ] 检查task events时间戳递增
- [ ] 返回错误码：0=正常�?=警告�?=错误
- [ ] 测试：故意制造非法状态，脚本应检测到

**Implementation Steps**:
```python
# scripts/verify_state.py (新建文件)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from kernel.state_store import read_yaml
from kernel.paths import TASKS_STATE_PATH, STATE_MACHINE_PATH
from datetime import datetime

def verify_state_transitions():
    """验证状态转换合法�?""
    tasks = read_yaml(TASKS_STATE_PATH).get('tasks', {})
    state_machine = read_yaml(STATE_MACHINE_PATH)
    transitions = {(t['from'], t['to']) for t in state_machine['transitions']}
    
    errors = []
    for task_id, task_data in tasks.items():
        events = task_data.get('events', [])
        for i in range(len(events) - 1):
            from_state = events[i].get('to')
            to_state = events[i+1].get('to')
            if (from_state, to_state) not in transitions:
                errors.append(f"�?{task_id}: 非法转换 {from_state} �?{to_state}")
    
    return errors

def verify_event_timestamps():
    """验证事件时间戳递增"""
    tasks = read_yaml(TASKS_STATE_PATH).get('tasks', {})
    errors = []
    
    for task_id, task_data in tasks.items():
        events = task_data.get('events', [])
        for i in range(len(events) - 1):
            t1 = events[i].get('timestamp')
            t2 = events[i+1].get('timestamp')
            if t1 and t2:
                try:
                    if datetime.fromisoformat(t1.replace('Z', '+00:00')) > \
                       datetime.fromisoformat(t2.replace('Z', '+00:00')):
                        errors.append(f"�?{task_id}: 时间戳逆序 {t1} > {t2}")
                except ValueError:
                    errors.append(f"⚠️ {task_id}: 时间戳格式错�?{t1}")
    
    return errors

if __name__ == '__main__':
    print("🔍 验证State一致�?..\n")
    
    errors = []
    errors.extend(verify_state_transitions())
    errors.extend(verify_event_timestamps())
    
    if not errors:
        print("�?State验证通过�?)
        sys.exit(0)
    else:
        for err in errors:
            print(err)
        print(f"\n�?发现 {len(errors)} 个问�?)
        sys.exit(2 if any('�? in e for e in errors) else 1)
```

**Verification**:
```powershell
# 正常情况应通过
python scripts/verify_state.py
# 输出: �?State验证通过�?
```

---

### 🟠 P1-8: WIP限制实现
**TaskCard**: B-9  
**预计工时**: 3小时  
**依赖**: �?

**Acceptance Criteria**:
- [ ] 在`kernel/state_store.py`添加`check_wip_limit()`函数
- [ ] 修改`kernel/os.py` task start命令，检查WIP�?
- [ ] 在`state/tasks.yaml` schema添加注释说明WIP限制
- [ ] 测试：尝试start�?个任务，应被拒绝
- [ ] 测试命令：`pytest kernel/tests/test_wip_limit.py -v`

**Implementation Steps**:
```python
# kernel/state_store.py 新增
def get_running_tasks_count() -> int:
    """获取当前running状态的任务�?""
    tasks = read_yaml(TASKS_STATE_PATH).get('tasks', {})
    return sum(1 for t in tasks.values() if t.get('status') == 'running')

def check_wip_limit(limit: int = 3) -> None:
    """检查WIP限制，超过限制抛出异�?""
    count = get_running_tasks_count()
    if count >= limit:
        raise RuntimeError(
            f"WIP限制超出：当�?{count} 个running任务，最多允�?{limit} 个�?
            f"请先完成部分任务再开始新任务�?
        )

# kernel/os.py 修改 task_start 函数
def task_start(task_id: str):
    """开始任�?""
    ensure_git_repo()
    
    # 检查WIP限制
    check_wip_limit(limit=3)
    
    task = get_task(task_id)
    # ... 后续逻辑
```

**Verification**:
```powershell
# 测试WIP限制
python kernel/os.py task start TASK_1
python kernel/os.py task start TASK_2
python kernel/os.py task start TASK_3
# �?个应失败
python kernel/os.py task start TASK_4
# 预期输出: RuntimeError: WIP限制超出
```

---

## Week 3 Tasks（第三周 - 质量提升�?

### 🟡 P2-9: DGSF项目测试套件
**TaskCard**: B-13  
**预计工时**: 6小时  
**依赖**: �?

**Acceptance Criteria**:
- [ ] 创建`projects/dgsf/repo/tests/`目录
- [ ] 添加至少3个测试文件：test_sdf_model.py, test_dataloader.py, test_integration.py
- [ ] 每个文件至少5个测试用�?
- [ ] 测试可独立运行：`pytest projects/dgsf/repo/tests/ -v`
- [ ] Coverage >70%: `pytest projects/dgsf/repo/tests/ --cov=projects/dgsf/repo/src`

**Implementation Steps**:
```python
# projects/dgsf/repo/tests/test_sdf_model.py (示例)
import pytest
import torch
from dgsf.sdf.model import GenerativeSDF  # 假设存在

def test_model_initialization():
    """测试模型初始�?""
    model = GenerativeSDF(input_dim=10, hidden_dim=64)
    assert model is not None
    assert model.input_dim == 10

def test_forward_pass():
    """测试forward pass"""
    model = GenerativeSDF(input_dim=10)
    x = torch.randn(32, 10)
    output = model(x)
    assert output.shape == (32, 1)

def test_sdf_boundedness():
    """测试SDF boundedness约束"""
    model = GenerativeSDF(input_dim=10, c=4.0)
    x = torch.randn(1000, 10)
    sdf = model.compute_sdf(x)
    assert sdf.min() >= -4.0
    assert sdf.max() <= 4.0

# ... 更多测试
```

**Verification**:
```powershell
# 运行DGSF测试
pytest projects/dgsf/repo/tests/ -v --cov=projects/dgsf/repo/src --cov-report=term
```

---

### 🟡 P2-10: Metrics Dashboard原型
**TaskCard**: B-10  
**预计工时**: 8小时（分2天）  
**依赖**: �?

**Acceptance Criteria**:
- [ ] 创建`scripts/generate_metrics.py`
- [ ] 从`state/tasks.yaml`计算cycle time、throughput
- [ ] 生成`reports/metrics_dashboard.md`包含表格和图表（ASCII art或mermaid�?
- [ ] 支持时间范围参数：`--since=7days`
- [ ] 自动化：每周五自动生成并commit

**Implementation Steps**:
```python
# scripts/generate_metrics.py (新建文件)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from kernel.state_store import read_yaml
from kernel.paths import TASKS_STATE_PATH
from datetime import datetime, timedelta
from collections import defaultdict

def calculate_cycle_time(task_data):
    """计算任务的cycle time（running �?merged�?""
    events = task_data.get('events', [])
    start_time = None
    end_time = None
    
    for event in events:
        if event.get('to') == 'running' and not start_time:
            start_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
        if event.get('to') == 'merged':
            end_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
    
    if start_time and end_time:
        return (end_time - start_time).total_seconds() / 3600  # 小时
    return None

def generate_dashboard(since_days=7):
    """生成metrics dashboard"""
    tasks = read_yaml(TASKS_STATE_PATH).get('tasks', {})
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    
    cycle_times = []
    throughput_by_week = defaultdict(int)
    
    for task_id, task_data in tasks.items():
        last_updated = datetime.fromisoformat(task_data['last_updated'].replace('Z', '+00:00'))
        if last_updated < cutoff:
            continue
        
        cycle_time = calculate_cycle_time(task_data)
        if cycle_time:
            cycle_times.append(cycle_time)
        
        if task_data.get('status') == 'merged':
            week = last_updated.strftime('%Y-W%U')
            throughput_by_week[week] += 1
    
    # 生成Markdown报告
    report = f"""# Metrics Dashboard

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**时间范围**: 最近{since_days}�?

## 📊 关键指标

| 指标 | �?| 目标 | 状�?|
|-----|----|----|-----|
| 平均Cycle Time | {sum(cycle_times)/len(cycle_times):.1f}h | <72h | {'�? if sum(cycle_times)/len(cycle_times) < 72 else '�?} |
| 周Throughput | {sum(throughput_by_week.values())/len(throughput_by_week):.1f} | �? | {'�? if sum(throughput_by_week.values())/len(throughput_by_week) >= 5 else '⚠️'} |
| 当前WIP | {sum(1 for t in tasks.values() if t.get('status') == 'running')} | �? | {'�? if sum(1 for t in tasks.values() if t.get('status') == 'running') <= 3 else '�?} |

## 📈 Cycle Time分布

```
{' '.join(['�? if ct < 24 else '�? if ct < 72 else '�? for ct in cycle_times])}
```

## 🚀 每周Throughput

| �?| 完成任务�?|
|---|----------|
{chr(10).join([f"| {week} | {count} |" for week, count in sorted(throughput_by_week.items())])}
"""
    
    # 写入文件
    output_path = Path(__file__).parents[1] / 'reports' / 'metrics_dashboard.md'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"�?Metrics dashboard生成完成: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--since', default='7days', help='时间范围，如7days, 30days')
    args = parser.parse_args()
    
    since_days = int(args.since.replace('days', ''))
    generate_dashboard(since_days)
```

**Verification**:
```powershell
# 生成dashboard
python scripts/generate_metrics.py --since=7days
# 查看报告
Get-Content reports\metrics_dashboard.md
```

---

## Week 4 Tasks（第四周 - 长期优化�?

### 🟡 P2-11: State接口抽象（Strangler Fig第一步）
**TaskCard**: B-14  
**预计工时**: 6小时  
**依赖**: �?

**Acceptance Criteria**:
- [ ] 创建`kernel/state_interface.py`定义抽象接口
- [ ] 实现YAMLStateStore和SQLiteStateStore（空实现�?
- [ ] 修改state_store.py使用接口
- [ ] 测试可以切换backend: `pytest kernel/tests/test_state_backend.py`

**Implementation Steps**:
```python
# kernel/state_interface.py (新建文件)
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path

class StateStore(ABC):
    """状态存储抽象接�?""
    
    @abstractmethod
    def read_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """读取任务"""
        pass
    
    @abstractmethod
    def write_task(self, task_id: str, data: Dict[str, Any]) -> None:
        """写入任务"""
        pass
    
    @abstractmethod
    def list_tasks(self) -> Dict[str, Dict[str, Any]]:
        """列出所有任�?""
        pass
    
    @abstractmethod
    def append_event(self, task_id: str, event: Dict[str, Any]) -> None:
        """追加事件"""
        pass

class YAMLStateStore(StateStore):
    """YAML文件存储（当前实现）"""
    
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.tasks_path = state_dir / 'tasks.yaml'
    
    def read_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        # 实现YAML读取逻辑
        pass
    
    # ... 其他方法

class SQLiteStateStore(StateStore):
    """SQLite数据库存储（未来实现�?""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # TODO: 初始化SQLite连接
    
    def read_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        # TODO: 实现SQLite查询
        pass
```

**Verification**:
```powershell
# 测试接口可以切换backend
python -c "from kernel.state_interface import YAMLStateStore; store = YAMLStateStore(Path('state')); print(store)"
```

---

### 🟡 P2-12: Blueprint一致性检查器
**TaskCard**: B-15  
**预计工时**: 5小时  
**依赖**: �?

**Acceptance Criteria**:
- [ ] 创建`scripts/check_blueprint_consistency.py`
- [ ] 检查docs/中的Markdown链接有效�?
- [ ] 检查架构图引用的文件是否存�?
- [ ] 检查ARCHITECTURE_PACK_INDEX中的blueprint状态与实际文件一�?
- [ ] 生成报告: `reports/blueprint_consistency.md`

**Implementation Steps**:
```python
# scripts/check_blueprint_consistency.py (新建文件)
import re
from pathlib import Path

def check_markdown_links(docs_dir: Path):
    """检查Markdown文件中的链接有效�?""
    errors = []
    
    for md_file in docs_dir.rglob('*.md'):
        content = md_file.read_text(encoding='utf-8')
        # 查找链接 [text](path)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        for text, link in links:
            if link.startswith('http'):
                continue  # 跳过外部链接
            
            target = (md_file.parent / link).resolve()
            if not target.exists():
                errors.append(f"�?{md_file.name}: 断开的链�?{link}")
    
    return errors

def check_blueprint_status():
    """检查blueprint状态与实际文件一致�?""
    index_path = Path('docs/ARCHITECTURE_PACK_INDEX.md')
    content = index_path.read_text(encoding='utf-8')
    
    # 解析状态表格（简化版�?
    errors = []
    # TODO: 实现完整的表格解析和验证
    
    return errors

if __name__ == '__main__':
    print("🔍 检查Blueprint一致�?..\n")
    
    errors = []
    errors.extend(check_markdown_links(Path('docs')))
    errors.extend(check_blueprint_status())
    
    if not errors:
        print("�?Blueprint一致性检查通过�?)
    else:
        for err in errors:
            print(err)
        print(f"\n�?发现 {len(errors)} 个问�?)
```

**Verification**:
```powershell
python scripts/check_blueprint_consistency.py
```

---

## 📌 立即执行的第一步（NEXT ACTION�?

**选择**: 🔴 P0-1 State Store并发锁实�?

**原因**:
1. 阻塞性最�?- 并发写入可能导致数据损坏
2. 无依�?- 可以立即开�?
3. 影响范围�?- 仅修改state_store.py
4. 风险可控 - 有明确的测试方案

**详细执行步骤见上方P0-1章节**

---

**Last Updated**: 2026-02-02  
**Next Review**: 每日standup时更新进�? 
**Status**: 🟢 ACTIVE

