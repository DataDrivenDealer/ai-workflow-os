# AI Workflow OS - TODO Next Steps

**文档ID**: TODO_NEXT  
**创建日期**: 2026-02-02  
**状态**: ACTIVE  
**关联计划**: [EXECUTION_PLAN_V1.md](EXECUTION_PLAN_V1.md)  
**WIP限制**: 最多3个任务同时进行

---

## 优先级说明
- 🔴 **P0**: 阻塞性问题，必须立即解决
- 🟠 **P1**: 高价值任务，本周内完成
- 🟡 **P2**: 质量改进，可以defer

---

## Week 1 Tasks（第一周 - 核心稳定性）

### 🔴 P0-1: State Store并发锁实现
**TaskCard**: B-1  
**预计工时**: 4小时  
**依赖**: 无

**Acceptance Criteria**（验收标准）:
- [ ] `kernel/state_store.py`添加文件锁机制（Windows使用msvcrt，Linux使用fcntl）
- [ ] 实现`with lock_state_file(path)` context manager
- [ ] 所有write_yaml调用包裹在锁内
- [ ] 编写并发测试：2个进程同时写入tasks.yaml，验证无数据损坏
- [ ] 测试命令：`python kernel/tests/test_state_store_concurrency.py`

**Implementation Steps**:
```python
# kernel/state_store.py 新增
import contextlib
import msvcrt  # Windows
import fcntl   # Unix/Linux

@contextlib.contextmanager
def lock_state_file(file_path: Path):
    """文件锁context manager"""
    with open(file_path, 'r+') as f:
        try:
            if sys.platform == 'win32':
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield f
        finally:
            if sys.platform == 'win32':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

**Verification**:
```powershell
# 并发测试
python -c "import concurrent.futures; from kernel.state_store import upsert_task; with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: executor.map(lambda i: upsert_task(f'TEST_{i}', {'status': 'draft'}), range(10))"
# 验证state/tasks.yaml内容完整无corruption
```

---

### 🔴 P0-2: 生成依赖版本锁定文件
**TaskCard**: B-2  
**预计工时**: 1小时  
**依赖**: 无

**Acceptance Criteria**:
- [ ] 安装pip-tools: `pip install pip-tools`
- [ ] 生成requirements-lock.txt: `pip-compile requirements.txt -o requirements-lock.txt`
- [ ] 验证锁定文件可安装: `pip-sync requirements-lock.txt`
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
**依赖**: 无

**Acceptance Criteria**:
- [ ] Review `ops/EXECUTION_PLAN_*.md` 三个文件内容
- [ ] 确认无敏感信息（如密码、内部IP）
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
# 应显示刚才提交的5个文件
```

---

### 🟠 P1-4: 路径管理重构（Day 1/2）
**TaskCard**: B-4  
**预计工时**: 6小时（分2天）  
**依赖**: 无

**Acceptance Criteria**:
- [ ] 创建`kernel/paths.py`定义所有路径常量
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
# 测试paths模块可导入
python -c "from kernel.paths import ROOT, STATE_DIR; print(ROOT, STATE_DIR)"
# 输出应显示正确的绝对路径
```

**Implementation Steps - Day 2**:
- 重构os.py、gate_check.py、ci_gate_reporter.py等文件
- 替换所有`Path(__file__).parents[1]`为`from kernel.paths import ROOT`
- 运行完整测试套件确保无破坏

---

### 🟠 P1-5: 配置管理统一
**TaskCard**: B-7  
**预计工时**: 4小时  
**依赖**: B-4完成

**Acceptance Criteria**:
- [ ] 创建`kernel/config.py`统一加载配置
- [ ] 支持环境变量覆盖（如`AI_WORKFLOW_OS_STATE_DIR`）
- [ ] 加载gates.yaml、state_machine.yaml、spec_registry.yaml
- [ ] 配置验证：必需字段检查、类型检查
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
**依赖**: 无

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

### 🟠 P1-7: 状态验证脚本
**TaskCard**: B-6  
**预计工时**: 4小时  
**依赖**: 无

**Acceptance Criteria**:
- [ ] 创建`scripts/verify_state.py`
- [ ] 检查state/tasks.yaml中的状态转换合法性
- [ ] 检查无orphaned branches（branch存在但task不存在）
- [ ] 检查task events时间戳递增
- [ ] 返回错误码：0=正常，1=警告，2=错误
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
    """验证状态转换合法性"""
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
                errors.append(f"❌ {task_id}: 非法转换 {from_state} → {to_state}")
    
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
                        errors.append(f"❌ {task_id}: 时间戳逆序 {t1} > {t2}")
                except ValueError:
                    errors.append(f"⚠️ {task_id}: 时间戳格式错误 {t1}")
    
    return errors

if __name__ == '__main__':
    print("🔍 验证State一致性...\n")
    
    errors = []
    errors.extend(verify_state_transitions())
    errors.extend(verify_event_timestamps())
    
    if not errors:
        print("✅ State验证通过！")
        sys.exit(0)
    else:
        for err in errors:
            print(err)
        print(f"\n❌ 发现 {len(errors)} 个问题")
        sys.exit(2 if any('❌' in e for e in errors) else 1)
```

**Verification**:
```powershell
# 正常情况应通过
python scripts/verify_state.py
# 输出: ✅ State验证通过！
```

---

### 🟠 P1-8: WIP限制实现
**TaskCard**: B-9  
**预计工时**: 3小时  
**依赖**: 无

**Acceptance Criteria**:
- [ ] 在`kernel/state_store.py`添加`check_wip_limit()`函数
- [ ] 修改`kernel/os.py` task start命令，检查WIP≤3
- [ ] 在`state/tasks.yaml` schema添加注释说明WIP限制
- [ ] 测试：尝试start第4个任务，应被拒绝
- [ ] 测试命令：`pytest kernel/tests/test_wip_limit.py -v`

**Implementation Steps**:
```python
# kernel/state_store.py 新增
def get_running_tasks_count() -> int:
    """获取当前running状态的任务数"""
    tasks = read_yaml(TASKS_STATE_PATH).get('tasks', {})
    return sum(1 for t in tasks.values() if t.get('status') == 'running')

def check_wip_limit(limit: int = 3) -> None:
    """检查WIP限制，超过限制抛出异常"""
    count = get_running_tasks_count()
    if count >= limit:
        raise RuntimeError(
            f"WIP限制超出：当前 {count} 个running任务，最多允许 {limit} 个。"
            f"请先完成部分任务再开始新任务。"
        )

# kernel/os.py 修改 task_start 函数
def task_start(task_id: str):
    """开始任务"""
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
# 第4个应失败
python kernel/os.py task start TASK_4
# 预期输出: RuntimeError: WIP限制超出
```

---

## Week 3 Tasks（第三周 - 质量提升）

### 🟡 P2-9: DGSF项目测试套件
**TaskCard**: B-13  
**预计工时**: 6小时  
**依赖**: 无

**Acceptance Criteria**:
- [ ] 创建`projects/dgsf/repo/tests/`目录
- [ ] 添加至少3个测试文件：test_sdf_model.py, test_dataloader.py, test_integration.py
- [ ] 每个文件至少5个测试用例
- [ ] 测试可独立运行：`pytest projects/dgsf/repo/tests/ -v`
- [ ] Coverage >70%: `pytest projects/dgsf/repo/tests/ --cov=projects/dgsf/repo/src`

**Implementation Steps**:
```python
# projects/dgsf/repo/tests/test_sdf_model.py (示例)
import pytest
import torch
from dgsf.sdf.model import GenerativeSDF  # 假设存在

def test_model_initialization():
    """测试模型初始化"""
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
**依赖**: 无

**Acceptance Criteria**:
- [ ] 创建`scripts/generate_metrics.py`
- [ ] 从`state/tasks.yaml`计算cycle time、throughput
- [ ] 生成`reports/metrics_dashboard.md`包含表格和图表（ASCII art或mermaid）
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
    """计算任务的cycle time（running → merged）"""
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
**时间范围**: 最近{since_days}天

## 📊 关键指标

| 指标 | 值 | 目标 | 状态 |
|-----|----|----|-----|
| 平均Cycle Time | {sum(cycle_times)/len(cycle_times):.1f}h | <72h | {'✅' if sum(cycle_times)/len(cycle_times) < 72 else '❌'} |
| 周Throughput | {sum(throughput_by_week.values())/len(throughput_by_week):.1f} | ≥5 | {'✅' if sum(throughput_by_week.values())/len(throughput_by_week) >= 5 else '⚠️'} |
| 当前WIP | {sum(1 for t in tasks.values() if t.get('status') == 'running')} | ≤3 | {'✅' if sum(1 for t in tasks.values() if t.get('status') == 'running') <= 3 else '❌'} |

## 📈 Cycle Time分布

```
{' '.join(['█' if ct < 24 else '▓' if ct < 72 else '░' for ct in cycle_times])}
```

## 🚀 每周Throughput

| 周 | 完成任务数 |
|---|----------|
{chr(10).join([f"| {week} | {count} |" for week, count in sorted(throughput_by_week.items())])}
"""
    
    # 写入文件
    output_path = Path(__file__).parents[1] / 'reports' / 'metrics_dashboard.md'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Metrics dashboard生成完成: {output_path}")

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

## Week 4 Tasks（第四周 - 长期优化）

### 🟡 P2-11: State接口抽象（Strangler Fig第一步）
**TaskCard**: B-14  
**预计工时**: 6小时  
**依赖**: 无

**Acceptance Criteria**:
- [ ] 创建`kernel/state_interface.py`定义抽象接口
- [ ] 实现YAMLStateStore和SQLiteStateStore（空实现）
- [ ] 修改state_store.py使用接口
- [ ] 测试可以切换backend: `pytest kernel/tests/test_state_backend.py`

**Implementation Steps**:
```python
# kernel/state_interface.py (新建文件)
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path

class StateStore(ABC):
    """状态存储抽象接口"""
    
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
        """列出所有任务"""
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
    """SQLite数据库存储（未来实现）"""
    
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
**依赖**: 无

**Acceptance Criteria**:
- [ ] 创建`scripts/check_blueprint_consistency.py`
- [ ] 检查docs/中的Markdown链接有效性
- [ ] 检查架构图引用的文件是否存在
- [ ] 检查ARCHITECTURE_PACK_INDEX中的blueprint状态与实际文件一致
- [ ] 生成报告: `reports/blueprint_consistency.md`

**Implementation Steps**:
```python
# scripts/check_blueprint_consistency.py (新建文件)
import re
from pathlib import Path

def check_markdown_links(docs_dir: Path):
    """检查Markdown文件中的链接有效性"""
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
                errors.append(f"❌ {md_file.name}: 断开的链接 {link}")
    
    return errors

def check_blueprint_status():
    """检查blueprint状态与实际文件一致性"""
    index_path = Path('docs/ARCHITECTURE_PACK_INDEX.md')
    content = index_path.read_text(encoding='utf-8')
    
    # 解析状态表格（简化版）
    errors = []
    # TODO: 实现完整的表格解析和验证
    
    return errors

if __name__ == '__main__':
    print("🔍 检查Blueprint一致性...\n")
    
    errors = []
    errors.extend(check_markdown_links(Path('docs')))
    errors.extend(check_blueprint_status())
    
    if not errors:
        print("✅ Blueprint一致性检查通过！")
    else:
        for err in errors:
            print(err)
        print(f"\n❌ 发现 {len(errors)} 个问题")
```

**Verification**:
```powershell
python scripts/check_blueprint_consistency.py
```

---

## 📌 立即执行的第一步（NEXT ACTION）

**选择**: 🔴 P0-1 State Store并发锁实现

**原因**:
1. 阻塞性最高 - 并发写入可能导致数据损坏
2. 无依赖 - 可以立即开始
3. 影响范围小 - 仅修改state_store.py
4. 风险可控 - 有明确的测试方案

**详细执行步骤见上方P0-1章节**

---

**Last Updated**: 2026-02-02  
**Next Review**: 每日standup时更新进度  
**Status**: 🟢 ACTIVE
