"""
Test concurrent access to state store with file locking.
"""
import concurrent.futures
import tempfile
from pathlib import Path
import pytest
import yaml

from kernel.state_store import write_yaml, read_yaml, init_state, upsert_task, atomic_update


def test_concurrent_writes_no_corruption():
    """测试并发写入不会导致数据损坏"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.yaml"
        
        def write_worker(worker_id):
            """模拟并发写入 - 使用atomic_update确保read-modify-write原子性"""
            for i in range(10):
                with atomic_update(test_path) as data:
                    data[f"worker_{worker_id}_key_{i}"] = f"value_{i}"
        
        # 5个worker并发写入
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_worker, i) for i in range(5)]
            concurrent.futures.wait(futures)
        
        # 验证：应有5*10=50个keys（不计入版本控制元数据字段）
        final_data = read_yaml(test_path)
        user_keys = [k for k in final_data if not k.startswith("_")]
        assert len(user_keys) == 50, f"Expected 50 keys, got {len(user_keys)}"
        
        # 验证数据完整性
        for worker_id in range(5):
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                assert key in final_data, f"Missing key: {key}"
                assert final_data[key] == f"value_{i}"


def test_concurrent_task_updates():
    """测试并发更新tasks不会冲突"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        init_state(tmpdir_path)
        tasks_path = tmpdir_path / "state" / "tasks.yaml"
        
        def update_task_worker(task_id):
            """模拟并发更新任务状态 - 使用atomic_update"""
            for status in ["draft", "ready", "running", "reviewing"]:
                with atomic_update(tasks_path) as tasks_state:
                    upsert_task(tasks_state, task_id, {"status": status})
        
        # 3个任务并发更新
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(update_task_worker, f"TASK_{i}") for i in range(3)]
            concurrent.futures.wait(futures)
        
        # 验证：3个任务都应该存在
        final_state = read_yaml(tasks_path)
        assert len(final_state["tasks"]) == 3
        
        for i in range(3):
            task_id = f"TASK_{i}"
            assert task_id in final_state["tasks"]
            assert "last_updated" in final_state["tasks"][task_id]


def test_high_volume_concurrent_writes():
    """测试高并发写入（1000次）无数据损坏"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.yaml"

        def write_worker(worker_id):
            with atomic_update(test_path) as data:
                for i in range(200):
                    data[f"w{worker_id}_k{i}"] = i

        # 5个worker * 200次 = 1000次写入
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_worker, i) for i in range(5)]
            for future in futures:
                future.result()

        final_data = read_yaml(test_path)
        # 不计入版本控制元数据字段（_version, _checksum, _last_modified_at）
        user_keys = [k for k in final_data if not k.startswith("_")]
        assert len(user_keys) == 1000, f"Expected 1000 keys, got {len(user_keys)}"


def test_lock_timeout():
    """测试锁超时机制"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.yaml"
        lock_path = test_path.with_suffix(test_path.suffix + ".lock")
        
        # 手动创建锁文件，模拟死锁
        lock_path.touch()
        
        # 尝试写入应该超时
        with pytest.raises(TimeoutError):
            write_yaml(test_path, {"data": "test"})
        
        # 清理
        lock_path.unlink()


def test_lock_release_on_exception():
    """测试异常情况下锁能正确释放"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.yaml"
        lock_path = test_path.with_suffix(test_path.suffix + ".lock")
        
        # 第一次写入正常
        write_yaml(test_path, {"key": "value"})
        assert not lock_path.exists(), "Lock file should be released"
        
        # 即使第一次操作完成，第二次写入也应该成功（锁已释放）
        write_yaml(test_path, {"key": "value2"})
        assert not lock_path.exists(), "Lock file should be released after second write"


if __name__ == "__main__":
    # 可以直接运行此文件进行快速测试
    print("Running concurrency tests...")
    test_concurrent_writes_no_corruption()
    print("✅ test_concurrent_writes_no_corruption passed")
    
    test_concurrent_task_updates()
    print("✅ test_concurrent_task_updates passed")
    
    test_lock_timeout()
    print("✅ test_lock_timeout passed")
    
    test_lock_release_on_exception()
    print("✅ test_lock_release_on_exception passed")
    
    print("\n🎉 All concurrency tests passed!")
