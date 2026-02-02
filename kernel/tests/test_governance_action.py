"""
测试治理操作模块 (Freeze & Acceptance)

验证范围:
- freeze_artifact() 冻结操作
- accept_artifact() 接受操作
- is_frozen() / is_accepted() 状态查询
- get_freeze_record() / get_acceptance_record() 记录检索
- 错误处理和边界条件
"""
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kernel.governance_action import (
    AcceptanceRecord,
    FreezeRecord,
    accept_artifact,
    freeze_artifact,
    get_acceptance_record,
    get_freeze_record,
    is_accepted,
    is_frozen,
)
from kernel.paths import OPS_ACCEPTANCE_DIR, OPS_FREEZE_DIR


@pytest.fixture
def temp_artifact():
    """创建临时测试工件"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Artifact\nSample content for governance testing.\n")
        artifact_path = Path(f.name)
    
    yield artifact_path
    
    # Cleanup
    if artifact_path.exists():
        artifact_path.unlink()


@pytest.fixture
def cleanup_governance_dirs():
    """清理测试后的治理目录"""
    yield
    
    # Cleanup freeze and acceptance directories after test
    if OPS_FREEZE_DIR.exists():
        shutil.rmtree(OPS_FREEZE_DIR, ignore_errors=True)
    if OPS_ACCEPTANCE_DIR.exists():
        shutil.rmtree(OPS_ACCEPTANCE_DIR, ignore_errors=True)


def test_freeze_artifact_basic(temp_artifact, cleanup_governance_dirs):
    """测试基本冻结操作"""
    artifact_path = temp_artifact
    version = "v1.0.0"
    frozen_by = "test_user"
    reason = "Initial freeze for testing"
    
    # 执行冻结
    record = freeze_artifact(
        artifact_path=artifact_path,
        frozen_by=frozen_by,
        version=version,
        metadata={"reason": reason}
    )
    
    # 验证返回记录
    assert isinstance(record, FreezeRecord)
    assert record.artifact_path == str(artifact_path)
    assert record.version == version
    assert record.frozen_by == frozen_by
    assert record.metadata["reason"] == reason
    assert len(record.content_hash) == 64  # SHA-256 哈希长度
    assert isinstance(record.frozen_at, datetime)
    
    # 验证冻结状态
    assert is_frozen(artifact_path, version)
    
    # 验证记录检索
    retrieved_record = get_freeze_record(artifact_path, version)
    assert retrieved_record is not None
    assert retrieved_record.content_hash == record.content_hash


def test_freeze_artifact_snapshot_created(temp_artifact, cleanup_governance_dirs):
    """测试冻结快照文件创建"""
    artifact_path = temp_artifact
    version = "v2.0.0"
    
    # 执行冻结
    record = freeze_artifact(
        artifact_path=artifact_path,
        frozen_by="snapshot_tester",
        version=version,
        metadata={}
    )
    
    # 验证快照文件存在（使用规范化路径）
    safe_path = str(artifact_path).replace('/', '_').replace('\\', '_').replace(':', '_')
    snapshot_path = OPS_FREEZE_DIR / f"{safe_path}_{version}.snapshot"
    assert snapshot_path.exists()
    
    # 验证快照内容匹配原文件
    original_content = artifact_path.read_text(encoding="utf-8")
    snapshot_content = snapshot_path.read_text(encoding="utf-8")
    assert snapshot_content == original_content


def test_freeze_duplicate_version(temp_artifact, cleanup_governance_dirs):
    """测试重复版本冻结（应该覆盖）"""
    artifact_path = temp_artifact
    version = "v3.0.0"
    
    # 首次冻结
    record1 = freeze_artifact(
        artifact_path=artifact_path,
        frozen_by="user1",
        version=version,
        metadata={"reason": "First freeze"}
    )
    
    # 修改文件内容
    artifact_path.write_text("# Modified Content\nNew content.\n", encoding="utf-8")
    
    # 再次冻结相同版本（应该覆盖）
    record2 = freeze_artifact(
        artifact_path=artifact_path,
        frozen_by="user2",
        version=version,
        metadata={"reason": "Second freeze"}
    )
    
    # 验证新记录覆盖旧记录
    assert record2.content_hash != record1.content_hash
    assert record2.frozen_by == "user2"
    
    # 验证检索到的是最新记录
    retrieved = get_freeze_record(artifact_path, version)
    assert retrieved.frozen_by == "user2"


def test_accept_artifact_basic(temp_artifact, cleanup_governance_dirs):
    """测试基本接受操作"""
    artifact_path = temp_artifact
    accepted_by = "governance_reviewer"
    authority = "GOVERNANCE_INVARIANTS §1"
    reason = "Artifact meets all governance requirements"
    
    # 执行接受
    record = accept_artifact(
        artifact_path=artifact_path,
        accepted_by=accepted_by,
        authority=authority,
        metadata={"reason": reason}
    )
    
    # 验证返回记录
    assert isinstance(record, AcceptanceRecord)
    assert record.artifact_path == str(artifact_path)
    assert record.accepted_by == accepted_by
    assert record.authority == authority
    assert record.metadata["reason"] == reason
    assert len(record.content_hash) == 64
    assert isinstance(record.accepted_at, datetime)
    
    # 验证接受状态
    assert is_accepted(artifact_path)
    
    # 验证记录检索
    retrieved_record = get_acceptance_record(artifact_path)
    assert retrieved_record is not None
    assert retrieved_record.content_hash == record.content_hash


def test_accept_artifact_overwrite(temp_artifact, cleanup_governance_dirs):
    """测试重复接受（应该覆盖旧记录）"""
    artifact_path = temp_artifact
    
    # 首次接受
    record1 = accept_artifact(
        artifact_path=artifact_path,
        accepted_by="reviewer1",
        authority="owner",
        metadata={"reason": "Initial approval"}
    )
    
    # 再次接受（覆盖）
    record2 = accept_artifact(
        artifact_path=artifact_path,
        accepted_by="reviewer2",
        authority="governance",
        metadata={"reason": "Re-approval with higher authority"}
    )
    
    # 验证新记录覆盖旧记录
    assert record2.accepted_by == "reviewer2"
    assert record2.authority == "governance"
    
    # 验证检索到的是最新记录
    retrieved = get_acceptance_record(artifact_path)
    assert retrieved.accepted_by == "reviewer2"
    assert retrieved.authority == "governance"


def test_is_frozen_nonexistent_artifact():
    """测试查询不存在的冻结工件"""
    fake_path = Path("/nonexistent/fake_artifact.md")
    assert not is_frozen(fake_path, "v1.0.0")


def test_is_accepted_nonexistent_artifact():
    """测试查询不存在的接受工件"""
    fake_path = Path("/nonexistent/fake_artifact.md")
    assert not is_accepted(fake_path)


def test_get_freeze_record_nonexistent():
    """测试检索不存在的冻结记录"""
    fake_path = Path("/nonexistent/fake_artifact.md")
    record = get_freeze_record(fake_path, "v1.0.0")
    assert record is None


def test_get_acceptance_record_nonexistent():
    """测试检索不存在的接受记录"""
    fake_path = Path("/nonexistent/fake_artifact.md")
    record = get_acceptance_record(fake_path)
    assert record is None


def test_freeze_then_accept_workflow(temp_artifact, cleanup_governance_dirs):
    """测试完整治理工作流：冻结 → 接受"""
    artifact_path = temp_artifact
    version = "v4.0.0"
    
    # 步骤1: 冻结工件
    freeze_record = freeze_artifact(
        artifact_path=artifact_path,
        frozen_by="author",
        version=version,
        metadata={"reason": "Release candidate"}
    )
    
    assert is_frozen(artifact_path, version)
    
    # 步骤2: 接受工件
    accept_record = accept_artifact(
        artifact_path=artifact_path,
        accepted_by="governance_committee",
        authority="vote",
        metadata={"reason": "Passed governance review", "frozen_version": version}
    )
    
    assert is_accepted(artifact_path)
    
    # 验证两个记录都存在且哈希一致（假设文件未修改）
    assert freeze_record.content_hash == accept_record.content_hash


def test_freeze_multiple_versions(temp_artifact, cleanup_governance_dirs):
    """测试同一工件的多版本冻结"""
    artifact_path = temp_artifact
    
    # 冻结 v1.0.0
    record_v1 = freeze_artifact(
        artifact_path=artifact_path,
        frozen_by="user",
        version="v1.0.0",
        metadata={}
    )
    
    # 修改文件内容
    artifact_path.write_text("# Updated Content v2\n", encoding="utf-8")
    
    # 冻结 v2.0.0
    record_v2 = freeze_artifact(
        artifact_path=artifact_path,
        frozen_by="user",
        version="v2.0.0",
        metadata={}
    )
    
    # 验证两个版本都存在且哈希不同
    assert is_frozen(artifact_path, "v1.0.0")
    assert is_frozen(artifact_path, "v2.0.0")
    assert record_v1.content_hash != record_v2.content_hash


def test_freeze_with_special_characters_in_path(cleanup_governance_dirs):
    """测试包含特殊字符的路径（空格、中文）"""
    with tempfile.NamedTemporaryFile(
        mode="w", 
        suffix=" test 测试.md", 
        delete=False,
        dir=Path(tempfile.gettempdir())
    ) as f:
        f.write("Test content with special path\n")
        artifact_path = Path(f.name)
    
    try:
        # 冻结包含特殊字符的工件
        record = freeze_artifact(
            artifact_path=artifact_path,
            frozen_by="tester",
            version="v1.0.0",
            metadata={}
        )
        
        assert record.artifact_path == str(artifact_path)
        assert is_frozen(artifact_path, "v1.0.0")
        
    finally:
        if artifact_path.exists():
            artifact_path.unlink()
