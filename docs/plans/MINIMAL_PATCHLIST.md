# 最小补丁列表（Minimal Patch List） - 2026-02-02

**文档ID**: MINIMAL_PATCHLIST_20260202  
**关联审计**: DRIFT_REPORT_20260202  
**目的**: 提供最小化、diff风格的修复清单，支持快速执行

---

## 使用说明

本文档按优先级列出了所有需要修复的漂移项，每项包含：
- **漂移ID**: 对应审计报告中的编号
- **文件路径**: 需要修改的文件
- **操作类型**: CREATE（创建）/ MODIFY（修改）/ DELETE（删除）
- **难度**: EASY（<1h）/ MEDIUM（1-4h）/ HARD（>4h）
- **依赖**: 前置任务（如有）
- **验证命令**: 修复后的验证方法

---

## P0 级别补丁（阻塞性 - 立即执行）

### PATCH-P0-01: 修复CI管道
**漂移ID**: D-P0-01  
**难度**: MEDIUM  
**预计工时**: 2小时

#### 操作1: 移除DGSF子模块依赖
**文件**: `.github/workflows/ci.yml`  
**类型**: MODIFY  
**当前行**: 
```yaml
- uses: actions/checkout@v3
  with:
    submodules: recursive
```
**修改为**:
```yaml
- uses: actions/checkout@v3
  # 移除 submodules: recursive，避免私有仓库导致失败
```

#### 操作2: 条件化DGSF检查
**文件**: `.github/workflows/ci.yml`  
**类型**: MODIFY  
**在 gate-g2-sanity job 中添加**:
```yaml
- name: Check DGSF availability
  run: |
    if [ -d "projects/dgsf" ]; then
      echo "DGSF project found"
    else
      echo "DGSF project not found, skipping DGSF-specific checks"
      exit 0
    fi
```

#### 操作3: 修复governance-check导入
**文件**: `.github/workflows/ci.yml`  
**类型**: MODIFY  
**当前行**:
```yaml
run: python -c "from governance_gate import verify_governance; ..."
```
**修改为**:
```yaml
run: python -c "from kernel.governance_gate import verify_governance; ..."
```

**验证命令**:
```bash
# 本地验证CI脚本
python -c "from kernel.governance_gate import verify_governance; verify_governance()"

# 推送后在GitHub Actions查看结果
git push origin main
# 访问 https://github.com/<用户>/AI-Workflow-OS/actions
```

**依赖**: 无  
**预期结果**: CI 全绿 ✅

---

### PATCH-P0-02: 实现Freeze和Acceptance操作
**漂移ID**: D-P0-02  
**难度**: HARD  
**预计工时**: 6小时

#### 操作1: 创建治理行动模块
**文件**: `kernel/governance_action.py`  
**类型**: CREATE  

```python
"""
Governance Action Module

Implements Freeze and Acceptance operations as defined in GOVERNANCE_INVARIANTS.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from kernel.paths import OPS_FREEZE_DIR, ROOT
from kernel.state_store import read_yaml, write_yaml


@dataclass
class FreezeRecord:
    """Record of an artifact freeze operation."""
    artifact_path: str
    frozen_at: datetime
    frozen_by: str
    content_hash: str  # SHA-256 of frozen content
    version: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "frozen_at": self.frozen_at.isoformat(),
            "frozen_by": self.frozen_by,
            "content_hash": self.content_hash,
            "version": self.version,
            "metadata": self.metadata,
        }


def freeze_artifact(
    artifact_path: Path,
    frozen_by: str,
    version: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> FreezeRecord:
    """
    Freeze an artifact, making it immutable.
    
    Creates a freeze record in ops/freeze/ with:
    - Original content snapshot
    - Cryptographic hash
    - Freeze metadata
    
    Args:
        artifact_path: Path to artifact to freeze (relative to ROOT)
        frozen_by: Identity of freezer (agent_id or user)
        version: Version identifier (e.g., "v1.0.0")
        metadata: Additional metadata
    
    Returns:
        FreezeRecord object
    
    Raises:
        FileNotFoundError: If artifact doesn't exist
        ValueError: If artifact already frozen at this version
    """
    full_path = ROOT / artifact_path
    if not full_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    
    # Compute content hash
    content = full_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()
    
    # Create freeze record
    record = FreezeRecord(
        artifact_path=str(artifact_path),
        frozen_at=datetime.now(timezone.utc),
        frozen_by=frozen_by,
        content_hash=content_hash,
        version=version,
        metadata=metadata or {},
    )
    
    # Save freeze record
    OPS_FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    freeze_file = OPS_FREEZE_DIR / f"{artifact_path.replace('/', '_')}_{version}.yaml"
    
    if freeze_file.exists():
        raise ValueError(f"Artifact already frozen at version {version}")
    
    write_yaml(freeze_file, record.to_dict())
    
    # Save frozen content snapshot
    snapshot_file = freeze_file.with_suffix(".snapshot")
    snapshot_file.write_bytes(content)
    
    return record


@dataclass
class AcceptanceRecord:
    """Record of an artifact acceptance operation."""
    artifact_path: str
    accepted_at: datetime
    accepted_by: str
    authority: str  # Who granted authority (e.g., "governance", "owner")
    content_hash: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "accepted_at": self.accepted_at.isoformat(),
            "accepted_by": self.accepted_by,
            "authority": self.authority,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


def accept_artifact(
    artifact_path: Path,
    accepted_by: str,
    authority: str = "owner",
    metadata: Optional[Dict[str, Any]] = None,
) -> AcceptanceRecord:
    """
    Accept an artifact, conferring it authority.
    
    Creates acceptance record and updates artifact status.
    
    Args:
        artifact_path: Path to artifact to accept
        accepted_by: Identity of acceptor
        authority: Authority source ("owner", "governance", "vote")
        metadata: Additional metadata
    
    Returns:
        AcceptanceRecord object
    """
    full_path = ROOT / artifact_path
    if not full_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")
    
    # Compute content hash
    content = full_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()
    
    # Create acceptance record
    record = AcceptanceRecord(
        artifact_path=str(artifact_path),
        accepted_at=datetime.now(timezone.utc),
        accepted_by=accepted_by,
        authority=authority,
        content_hash=content_hash,
        metadata=metadata or {},
    )
    
    # Save acceptance record
    acceptance_dir = ROOT / "ops" / "acceptance"
    acceptance_dir.mkdir(parents=True, exist_ok=True)
    
    acceptance_file = acceptance_dir / f"{artifact_path.replace('/', '_')}.yaml"
    write_yaml(acceptance_file, record.to_dict())
    
    return record


def is_frozen(artifact_path: Path, version: Optional[str] = None) -> bool:
    """Check if artifact is frozen at specified version."""
    if version:
        freeze_file = OPS_FREEZE_DIR / f"{artifact_path.replace('/', '_')}_{version}.yaml"
        return freeze_file.exists()
    else:
        # Check if any version is frozen
        pattern = f"{artifact_path.replace('/', '_')}_*.yaml"
        return any(OPS_FREEZE_DIR.glob(pattern))


def is_accepted(artifact_path: Path) -> bool:
    """Check if artifact has been accepted."""
    acceptance_file = ROOT / "ops" / "acceptance" / f"{artifact_path.replace('/', '_')}.yaml"
    return acceptance_file.exists()
```

#### 操作2: 集成到CLI
**文件**: `kernel/os.py`  
**类型**: MODIFY  
**在文件末尾添加命令**:

```python
def cmd_freeze(args: argparse.Namespace) -> None:
    """Freeze an artifact."""
    from kernel.governance_action import freeze_artifact
    
    artifact_path = Path(args.artifact)
    record = freeze_artifact(
        artifact_path=artifact_path,
        frozen_by=args.frozen_by or "cli_user",
        version=args.version,
        metadata={"reason": args.reason or "Manual freeze"},
    )
    print(f"✅ Frozen: {artifact_path} → v{record.version}")
    print(f"   Hash: {record.content_hash[:12]}...")
    print(f"   By: {record.frozen_by}")


def cmd_accept(args: argparse.Namespace) -> None:
    """Accept an artifact."""
    from kernel.governance_action import accept_artifact
    
    artifact_path = Path(args.artifact)
    record = accept_artifact(
        artifact_path=artifact_path,
        accepted_by=args.accepted_by or "cli_user",
        authority=args.authority or "owner",
        metadata={"reason": args.reason or "Manual acceptance"},
    )
    print(f"✅ Accepted: {artifact_path}")
    print(f"   Hash: {record.content_hash[:12]}...")
    print(f"   By: {record.accepted_by} (authority: {record.authority})")


# 在 build_parser() 中添加子命令
def build_parser() -> argparse.ArgumentParser:
    # ... 现有代码 ...
    
    # Freeze命令
    freeze_parser = subparsers.add_parser("freeze", help="Freeze an artifact")
    freeze_parser.add_argument("artifact", help="Path to artifact (relative to root)")
    freeze_parser.add_argument("version", help="Version identifier (e.g., v1.0.0)")
    freeze_parser.add_argument("--frozen-by", help="Freezer identity")
    freeze_parser.add_argument("--reason", help="Freeze reason")
    freeze_parser.set_defaults(func=cmd_freeze)
    
    # Accept命令
    accept_parser = subparsers.add_parser("accept", help="Accept an artifact")
    accept_parser.add_argument("artifact", help="Path to artifact (relative to root)")
    accept_parser.add_argument("--accepted-by", help="Acceptor identity")
    accept_parser.add_argument("--authority", help="Authority source (owner/governance)")
    accept_parser.add_argument("--reason", help="Acceptance reason")
    accept_parser.set_defaults(func=cmd_accept)
    
    return parser
```

#### 操作3: 添加测试
**文件**: `kernel/tests/test_governance_action.py`  
**类型**: CREATE  

```python
import pytest
from pathlib import Path
from datetime import datetime

from kernel.governance_action import (
    freeze_artifact,
    accept_artifact,
    is_frozen,
    is_accepted,
    FreezeRecord,
    AcceptanceRecord,
)


def test_freeze_artifact(tmp_path, monkeypatch):
    """Test artifact freezing."""
    monkeypatch.setattr("kernel.governance_action.ROOT", tmp_path)
    
    # Create test artifact
    artifact = tmp_path / "test.md"
    artifact.write_text("Test content")
    
    # Freeze it
    record = freeze_artifact(
        artifact_path=Path("test.md"),
        frozen_by="test_user",
        version="v1.0.0",
    )
    
    assert record.artifact_path == "test.md"
    assert record.version == "v1.0.0"
    assert record.frozen_by == "test_user"
    assert len(record.content_hash) == 64  # SHA-256
    
    # Verify freeze record exists
    freeze_dir = tmp_path / "ops" / "freeze"
    assert (freeze_dir / "test.md_v1.0.0.yaml").exists()
    assert (freeze_dir / "test.md_v1.0.0.snapshot").exists()


def test_accept_artifact(tmp_path, monkeypatch):
    """Test artifact acceptance."""
    monkeypatch.setattr("kernel.governance_action.ROOT", tmp_path)
    
    # Create test artifact
    artifact = tmp_path / "test.md"
    artifact.write_text("Test content")
    
    # Accept it
    record = accept_artifact(
        artifact_path=Path("test.md"),
        accepted_by="owner",
        authority="governance",
    )
    
    assert record.artifact_path == "test.md"
    assert record.accepted_by == "owner"
    assert record.authority == "governance"
    
    # Verify acceptance record
    acceptance_file = tmp_path / "ops" / "acceptance" / "test.md.yaml"
    assert acceptance_file.exists()


def test_is_frozen(tmp_path, monkeypatch):
    """Test frozen status check."""
    monkeypatch.setattr("kernel.governance_action.ROOT", tmp_path)
    
    artifact = tmp_path / "test.md"
    artifact.write_text("Test")
    
    assert not is_frozen(Path("test.md"))
    
    freeze_artifact(Path("test.md"), "user", "v1.0.0")
    
    assert is_frozen(Path("test.md"), "v1.0.0")
    assert is_frozen(Path("test.md"))  # Any version
```

**验证命令**:
```bash
# 运行新测试
python -m pytest kernel/tests/test_governance_action.py -v

# 测试CLI命令
python kernel/os.py freeze specs/canon/GOVERNANCE_INVARIANTS.md v1.0.0 --frozen-by governance
python kernel/os.py accept specs/canon/GOVERNANCE_INVARIANTS.md --accepted-by governance
```

**依赖**: 无  
**预期结果**: 测试通过，CLI命令可用

---

## P1 级别补丁（高价值 - 本周完成）

### PATCH-P1-01: 实现Artifact Locking
**漂移ID**: D-P1-01  
**难度**: MEDIUM  
**预计工时**: 3小时

#### 操作1: 扩展AgentSession
**文件**: `kernel/agent_auth.py`  
**类型**: MODIFY  
**在 AgentSession 类中添加字段**:

```python
@dataclass
class AgentSession:
    # ... 现有字段 ...
    pending_artifacts: Set[str] = field(default_factory=set)  # 已存在
    locked_artifacts: Set[str] = field(default_factory=set)  # 新增
```

**在 to_dict() 中添加**:
```python
def to_dict(self) -> Dict[str, Any]:
    return {
        # ... 现有字段 ...
        "locked_artifacts": list(self.locked_artifacts),
    }
```

**在 from_dict() 中添加**:
```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "AgentSession":
    return cls(
        # ... 现有字段 ...
        locked_artifacts=set(data.get("locked_artifacts", [])),
    )
```

#### 操作2: 实现锁管理方法
**文件**: `kernel/agent_auth.py`  
**类型**: MODIFY  
**在 AgentAuthManager 类中添加方法**:

```python
def lock_artifact(
    self,
    session_token: str,
    artifact_path: str,
    timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    """
    Acquire lock on artifact for session.
    
    Returns:
        {"success": bool, "session": AgentSession | None, "error": str | None}
    """
    session = self.get_session(session_token)
    if not session or not session.is_active:
        return {"success": False, "error": "Invalid or inactive session"}
    
    # Check if artifact already locked by another session
    for other_session in self.sessions.values():
        if other_session.session_token != session_token:
            if artifact_path in other_session.locked_artifacts:
                return {
                    "success": False,
                    "error": f"Artifact locked by session {other_session.session_token[:8]}",
                }
    
    # Acquire lock
    session.locked_artifacts.add(artifact_path)
    session.add_event("artifact_locked", {"artifact": artifact_path})
    
    self._persist_session(session)
    
    return {"success": True, "session": session, "error": None}


def unlock_artifact(
    self,
    session_token: str,
    artifact_path: str,
) -> Dict[str, Any]:
    """
    Release lock on artifact.
    
    Returns:
        {"success": bool, "session": AgentSession | None, "error": str | None}
    """
    session = self.get_session(session_token)
    if not session:
        return {"success": False, "error": "Session not found"}
    
    if artifact_path not in session.locked_artifacts:
        return {"success": False, "error": "Artifact not locked by this session"}
    
    # Release lock
    session.locked_artifacts.remove(artifact_path)
    session.add_event("artifact_unlocked", {"artifact": artifact_path})
    
    self._persist_session(session)
    
    return {"success": True, "session": session, "error": None}


def get_artifact_lock_holder(self, artifact_path: str) -> Optional[AgentSession]:
    """Get session that holds lock on artifact."""
    for session in self.sessions.values():
        if artifact_path in session.locked_artifacts:
            return session
    return None
```

#### 操作3: 暴露到MCP Server
**文件**: `kernel/mcp_server.py`  
**类型**: MODIFY  
**在工具列表中添加**:

```python
{
    "name": "agent_lock_artifact",
    "description": "Acquire exclusive lock on an artifact",
    "inputSchema": {
        "type": "object",
        "properties": {
            "session_token": {"type": "string"},
            "artifact_path": {"type": "string"},
        },
        "required": ["session_token", "artifact_path"],
    },
},
{
    "name": "agent_unlock_artifact",
    "description": "Release lock on an artifact",
    "inputSchema": {
        "type": "object",
        "properties": {
            "session_token": {"type": "string"},
            "artifact_path": {"type": "string"},
        },
        "required": ["session_token", "artifact_path"],
    },
},
```

**添加工具实现**:
```python
elif tool_name == "agent_lock_artifact":
    result = self.auth_manager.lock_artifact(
        session_token=arguments["session_token"],
        artifact_path=arguments["artifact_path"],
    )
    return result

elif tool_name == "agent_unlock_artifact":
    result = self.auth_manager.unlock_artifact(
        session_token=arguments["session_token"],
        artifact_path=arguments["artifact_path"],
    )
    return result
```

**验证命令**:
```bash
# 运行锁测试
python -m pytest kernel/tests/test_agent_auth.py::test_lock_artifact -v
python -m pytest kernel/tests/test_mcp_server.py -k lock -v
```

**依赖**: 无  
**预期结果**: 锁机制工作，测试通过

---

### PATCH-P1-02: 补充不变量验证
**漂移ID**: D-P1-03  
**难度**: MEDIUM  
**预计工时**: 4小时

#### 操作1: INV-2 WIP上限验证
**文件**: `scripts/check_wip_limit.py`  
**类型**: CREATE

```python
#!/usr/bin/env python3
"""
Check WIP Limit (SYSTEM_INVARIANTS INV-2)

Verifies that the number of running tasks does not exceed the configured limit.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.config import config
from kernel.state_store import read_yaml, get_running_tasks_count
from kernel.paths import TASKS_STATE_PATH


def check_wip_limit() -> bool:
    """Check if WIP limit is violated."""
    max_running = config.get("wip_limits", {}).get("max_running_tasks", 3)
    
    tasks_state = read_yaml(TASKS_STATE_PATH)
    running_count = get_running_tasks_count(tasks_state)
    
    print(f"WIP Limit Check")
    print(f"===============")
    print(f"Max allowed: {max_running}")
    print(f"Currently running: {running_count}")
    
    if running_count > max_running:
        print(f"❌ VIOLATION: {running_count} > {max_running}")
        return False
    else:
        print(f"✅ PASS: {running_count} <= {max_running}")
        return True


if __name__ == "__main__":
    passed = check_wip_limit()
    sys.exit(0 if passed else 1)
```

#### 操作2: INV-4 时间单调性验证
**文件**: `scripts/verify_state.py`  
**类型**: MODIFY  
**添加函数**:

```python
def check_timestamp_monotonicity(tasks_state: Dict[str, Any]) -> List[str]:
    """Check INV-4: Event timestamps must be monotonically increasing."""
    violations = []
    
    for task_id, task in tasks_state.get("tasks", {}).items():
        events = task.get("events", [])
        prev_time = None
        
        for i, event in enumerate(events):
            timestamp_str = event.get("timestamp")
            if not timestamp_str:
                violations.append(f"{task_id}: Event {i} missing timestamp")
                continue
            
            try:
                current_time = datetime.fromisoformat(timestamp_str)
            except ValueError:
                violations.append(f"{task_id}: Event {i} invalid timestamp format")
                continue
            
            if prev_time and current_time < prev_time:
                violations.append(
                    f"{task_id}: Event {i} timestamp {current_time} < previous {prev_time}"
                )
            
            prev_time = current_time
    
    return violations
```

**在 main() 中调用**:
```python
def main():
    # ... 现有检查 ...
    
    # Check timestamp monotonicity (INV-4)
    time_violations = check_timestamp_monotonicity(tasks_state)
    if time_violations:
        print("\n❌ Timestamp monotonicity violations (INV-4):")
        for violation in time_violations:
            print(f"  - {violation}")
    else:
        print("\n✅ All event timestamps are monotonic (INV-4)")
```

#### 操作3: INV-9 MCP接口一致性
**文件**: `scripts/check_mcp_interface.py`  
**类型**: CREATE

```python
#!/usr/bin/env python3
"""
Check MCP Interface Consistency (SYSTEM_INVARIANTS INV-9)

Verifies that MCP Server tools match mcp_server_manifest.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.mcp_server import create_server


def check_mcp_interface() -> bool:
    """Check if MCP Server tools match manifest."""
    manifest_path = ROOT / "mcp_server_manifest.json"
    
    if not manifest_path.exists():
        print("❌ mcp_server_manifest.json not found")
        return False
    
    with manifest_path.open() as f:
        manifest = json.load(f)
    
    manifest_tools = {tool["name"] for tool in manifest.get("tools", [])}
    
    # Get actual tools from server
    server = create_server()
    actual_tools = {tool["name"] for tool in server.list_tools()}
    
    print("MCP Interface Consistency Check")
    print("================================")
    print(f"Manifest tools: {len(manifest_tools)}")
    print(f"Actual tools: {len(actual_tools)}")
    
    missing = manifest_tools - actual_tools
    extra = actual_tools - manifest_tools
    
    if missing:
        print(f"\n❌ Tools in manifest but not implemented:")
        for tool in sorted(missing):
            print(f"  - {tool}")
    
    if extra:
        print(f"\n⚠️ Tools implemented but not in manifest:")
        for tool in sorted(extra):
            print(f"  - {tool}")
    
    if not missing and not extra:
        print("\n✅ All tools match manifest")
        return True
    else:
        return False


if __name__ == "__main__":
    passed = check_mcp_interface()
    sys.exit(0 if passed else 1)
```

**验证命令**:
```bash
# 运行各验证脚本
python scripts/check_wip_limit.py
python scripts/verify_state.py
python scripts/check_mcp_interface.py
```

**依赖**: 无  
**预期结果**: 所有验证通过

---

### PATCH-P1-03: 集成Gate G3-G6到CI
**漂移ID**: D-P1-04  
**难度**: EASY  
**预计工时**: 1小时

#### 操作: 扩展CI配置
**文件**: `.github/workflows/ci.yml`  
**类型**: MODIFY  
**在现有jobs后添加**:

```yaml
  gate-g3:
    runs-on: ubuntu-latest
    needs: [gate-g2-sanity]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-lock.txt
      - name: Run Gate G3 (Code Review)
        run: python scripts/run_gate_g3.py --output text
        continue-on-error: true  # G3为建议性门禁

  gate-g4:
    runs-on: ubuntu-latest
    needs: [gate-g3]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-lock.txt
      - name: Run Gate G4 (Architecture Check)
        run: python scripts/run_gate_g4.py --output text
        continue-on-error: true  # G4为建议性门禁

  gate-g5:
    runs-on: ubuntu-latest
    needs: [gate-g4]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-lock.txt
      - name: Run Gate G5 (Merge Ready)
        run: python scripts/run_gate_g5.py --output text
        # G5为阻塞性门禁，失败则阻止合并

  ci-summary:
    runs-on: ubuntu-latest
    needs: [policy-check, governance-check, gate-g1, gate-g2-sanity, gate-g3, gate-g4, gate-g5]
    if: always()
    steps:
      - name: CI Summary
        run: |
          echo "CI Pipeline Summary"
          echo "==================="
          echo "All critical gates completed"
```

**验证命令**:
```bash
# 本地模拟CI流程
python scripts/run_gate_g3.py --output text
python scripts/run_gate_g4.py --output text
python scripts/run_gate_g5.py --output text

# 推送后验证远端CI
git push origin main
```

**依赖**: PATCH-P0-01（CI基础修复）  
**预期结果**: CI包含所有门禁检查

---

## P2 级别补丁（改进 - 可延后）

### PATCH-P2-01: 补充文档索引
**漂移ID**: D-P2-01  
**难度**: EASY  
**预计工时**: 30分钟

#### 操作: 更新README
**文件**: `README.md`  
**类型**: MODIFY  
**在 "## Structure" 章节后添加**:

```markdown
## Architecture

This project follows a canonical architecture pack model:
- 📘 [Architecture Pack Index](docs/ARCHITECTURE_PACK_INDEX.md) - Complete architecture overview
- 📐 [Architecture Blueprint](docs/ARCH_BLUEPRINT_MASTER.mmd) - System structure
- 🔒 [Governance Invariants](specs/canon/GOVERNANCE_INVARIANTS.md) - Constitutional rules
- 🎭 [Role Mode Canon](specs/canon/ROLE_MODE_CANON.md) - Role-based authorization

## Documentation

- [MCP Usage Guide](docs/MCP_USAGE_GUIDE.md) - How to use the MCP Server
- [Pair Programming Guide](docs/PAIR_PROGRAMMING_GUIDE.md) - Code review process
- [System Invariants](docs/SYSTEM_INVARIANTS.md) - Verifiable system guarantees
```

**验证命令**:
```bash
# 检查链接有效性
python scripts/check_doc_links.py
```

**依赖**: 无  
**预期结果**: README包含所有关键文档链接

---

### PATCH-P2-02: 创建术语映射检查器
**漂移ID**: 一致性保证机制  
**难度**: MEDIUM  
**预计工时**: 3小时

#### 操作: 创建检查工具
**文件**: `scripts/check_terminology_mapping.py`  
**类型**: CREATE

```python
#!/usr/bin/env python3
"""
Terminology Mapping Checker

Verifies that terms defined in Canon specs have corresponding implementations.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent

# 术语定义：{术语: (定义位置, 预期实现模式)}
CANONICAL_TERMS: Dict[str, tuple] = {
    "RoleMode": ("ROLE_MODE_CANON", r"class RoleMode|enum RoleMode"),
    "AgentSession": ("AGENT_SESSION", r"class AgentSession"),
    "GovernanceGate": ("GOVERNANCE_INVARIANTS", r"class GovernanceGate"),
    "Freeze": ("GOVERNANCE_INVARIANTS", r"def freeze_artifact|class FreezeRecord"),
    "Acceptance": ("GOVERNANCE_INVARIANTS", r"def accept_artifact|class AcceptanceRecord"),
    "Artifact Lock": ("AGENT_SESSION", r"locked_artifacts|lock_artifact"),
    "Authority": ("GOVERNANCE_INVARIANTS", r"class Authority|authority_level"),
}


def search_term_in_code(term: str, pattern: str) -> List[Path]:
    """Search for term pattern in kernel/ code."""
    found_files = []
    kernel_dir = ROOT / "kernel"
    
    for py_file in kernel_dir.glob("*.py"):
        content = py_file.read_text()
        if re.search(pattern, content):
            found_files.append(py_file)
    
    return found_files


def check_terminology_mapping() -> bool:
    """Check all canonical terms have implementations."""
    print("Terminology Mapping Report")
    print("==========================\n")
    
    all_found = True
    
    for term, (spec, pattern) in CANONICAL_TERMS.items():
        found_files = search_term_in_code(term, pattern)
        
        if found_files:
            print(f"✅ {term}: FOUND")
            for f in found_files:
                print(f"   → {f.relative_to(ROOT)}")
        else:
            print(f"❌ {term}: NOT FOUND")
            print(f"   Defined in: {spec}")
            print(f"   Expected pattern: {pattern}")
            all_found = False
        print()
    
    return all_found


if __name__ == "__main__":
    passed = check_terminology_mapping()
    sys.exit(0 if passed else 1)
```

**验证命令**:
```bash
python scripts/check_terminology_mapping.py
```

**依赖**: 无  
**预期结果**: 生成术语映射报告

---

## 总结：补丁执行顺序

### 必须按顺序执行（有依赖）
1. PATCH-P0-01 (CI修复) → 为后续CI集成铺路
2. PATCH-P1-03 (Gate集成) → 依赖CI修复

### 可并行执行（无依赖）
- PATCH-P0-02 (Freeze/Acceptance)
- PATCH-P1-01 (Artifact Locking)
- PATCH-P1-02 (不变量验证)
- PATCH-P2-01 (文档索引)
- PATCH-P2-02 (术语检查器)

### 预计总工时
- **P0**: 8小时（阻塞性，立即执行）
- **P1**: 8小时（高价值，本周完成）
- **P2**: 3.5小时（改进，可延后）
- **总计**: 19.5小时 ≈ 2.5个工作日

### 验证检查点
执行完所有P0和P1补丁后，运行以下验证：

```bash
# 1. 测试套件
python -m pytest kernel/tests/ -v

# 2. 类型检查
python -m pyright kernel/

# 3. 所有不变量验证
python scripts/verify_state.py
python scripts/verify_state_transitions.py
python scripts/check_wip_limit.py
python scripts/check_mcp_interface.py

# 4. 所有Gate检查
python scripts/run_gate_g1.py --output text
python scripts/run_gate_g2.py --output text
python scripts/run_gate_g3.py --output text
python scripts/run_gate_g4.py --output text
python scripts/run_gate_g5.py --output text

# 5. CI验证（远端）
git push origin main
# 确认 GitHub Actions 全绿
```

---

**补丁列表结束**

本文档提供了可直接执行的修复步骤。建议按照优先级顺序实施，每个补丁完成后立即验证，确保系统保持稳定状态。
