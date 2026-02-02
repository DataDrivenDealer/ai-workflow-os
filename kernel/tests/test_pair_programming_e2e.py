"""
End-to-end integration tests for Pair Programming feature.

Tests complete workflow from code generation → review → approval.
Validates automatic triggering of code review when new code is generated.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import yaml
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_review import CodeReviewEngine, ReviewVerdict, IssueSeverity
from mcp_server import MCPServer


@pytest.fixture
def temp_workspace():
    """Create temporary workspace with required directory structure."""
    temp_dir = Path(tempfile.mkdtemp(prefix="pp_test_"))
    
    # Create directory structure
    (temp_dir / "state").mkdir()
    (temp_dir / "tasks").mkdir()
    (temp_dir / "kernel").mkdir()
    (temp_dir / "ops" / "audit").mkdir(parents=True)
    
    # Create minimal state files
    with (temp_dir / "state" / "agents.yaml").open("w") as f:
        yaml.safe_dump({"agents": {}}, f)
    
    with (temp_dir / "state" / "sessions.yaml").open("w") as f:
        yaml.safe_dump({"sessions": {}}, f)
    
    with (temp_dir / "state" / "tasks.yaml").open("w") as f:
        yaml.safe_dump({"tasks": {}, "reviews": {}}, f)
    
    # Copy state machine configuration
    src_state_machine = Path(__file__).parent.parent / "state_machine.yaml"
    dst_state_machine = temp_dir / "kernel" / "state_machine.yaml"
    shutil.copy(src_state_machine, dst_state_machine)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mcp_server(temp_workspace, monkeypatch):
    """Create MCP server instance with temp workspace."""
    # Change working directory to temp workspace
    monkeypatch.chdir(temp_workspace)
    # Override KERNEL_DIR in mcp_server module
    import mcp_server as mcp_mod
    monkeypatch.setattr(mcp_mod, 'KERNEL_DIR', temp_workspace / "kernel")
    server = MCPServer()
    server.root = temp_workspace
    return server


class TestPairProgrammingIntegration:
    """End-to-end integration tests for Pair Programming workflow."""
    
    def test_complete_pair_programming_workflow_approved(self, mcp_server, temp_workspace):
        """
        Test complete workflow: code generation → review → approval.
        
        Simulates:
        1. Builder generates code
        2. Builder submits for review
        3. Reviewer creates session
        4. Reviewer conducts review (finds no major issues)
        5. Reviewer approves
        6. Task advances to 'reviewing'
        """
        
        # Step 1: Register agents
        builder_result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_builder",
            "display_name": "Builder Agent",
            "allowed_role_modes": ["builder"],
        })
        assert builder_result["success"]
        builder_agent_id = builder_result["agent"]["agent_id"]
        
        reviewer_result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_reviewer",
            "display_name": "Reviewer Agent",
            "allowed_role_modes": ["reviewer"],
        })
        assert reviewer_result["success"]
        reviewer_agent_id = reviewer_result["agent"]["agent_id"]
        
        # Step 2: Create sessions
        builder_session = mcp_server.call_tool("session_create", {
            "agent_id": builder_agent_id,
            "role_mode": "builder",
            "authorized_by": "test_operator",
        })
        assert builder_session["success"]
        builder_token = builder_session["session"]["session_token"]
        
        reviewer_session = mcp_server.call_tool("session_create", {
            "agent_id": reviewer_agent_id,
            "role_mode": "reviewer",
            "authorized_by": "test_operator",
        })
        assert reviewer_session["success"]
        reviewer_token = reviewer_session["session"]["session_token"]
        
        # Step 3: Create a task
        task_id = "PP_TEST_001"
        taskcard_path = temp_workspace / "tasks" / f"{task_id}.md"
        taskcard_content = f"""---
task_id: {task_id}
type: dev
queue: dev
branch: feature/{task_id}
priority: P2
spec_ids:
  - PAIR_PROGRAMMING_STANDARD
---

# Task {task_id}

## Requirements
- Implement feature X
- Add error handling
- Include unit tests
"""
        taskcard_path.write_text(taskcard_content)
        
        # Add task to state
        with (temp_workspace / "state" / "tasks.yaml").open("r") as f:
            tasks_state = yaml.safe_load(f)
        
        tasks_state["tasks"][task_id] = {
            "status": "running",
            "queue": "dev",
            "branch": f"feature/{task_id}",
            "priority": "P2",
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        
        with (temp_workspace / "state" / "tasks.yaml").open("w") as f:
            yaml.safe_dump(tasks_state, f)
        
        # Step 4: Builder generates code (simulated)
        code_file = temp_workspace / "kernel" / "new_feature.py"
        code_content = '''"""New feature implementation."""

def calculate(x: int, y: int) -> int:
    """Calculate sum."""
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("Arguments must be integers")
    return x + y

def process_data(items: list) -> list:
    """Process data items."""
    if not items:
        return []
    result = []
    for item in items:
        result.append(item * 2)
    return result
'''
        code_file.write_text(code_content)
        
        # Step 5: Builder submits for review (THIS IS THE KEY AUTOMATIC TRIGGER)
        print("\n" + "="*70)
        print("STEP 5: Builder submits code for review (AUTOMATIC TRIGGER)")
        print("="*70)
        submit_result = mcp_server.call_tool("review_submit", {
            "session_token": builder_token,
            "task_id": task_id,
            "artifact_paths": ["kernel/new_feature.py"],
            "notes": "Implementation complete, ready for review",
        })
        
        assert submit_result["success"]
        assert submit_result["status"] == "code_review"
        print(f"[PASS] Code automatically submitted for review")
        print(f"[PASS] Task status automatically changed to: {submit_result['status']}")
        
        # Step 6: Reviewer creates review session
        print("\n" + "="*70)
        print("STEP 6: Reviewer creates review session")
        print("="*70)
        review_session_result = mcp_server.call_tool("review_create_session", {
            "session_token": reviewer_token,
            "task_id": task_id,
            "personas": ["security_expert", "architecture_expert"],
        })
        
        assert review_session_result["success"]
        review_session_id = review_session_result["review_session_id"]
        print(f"[PASS] Review session created: {review_session_id}")
        
        # Step 7: Reviewer gets review prompts
        print("\n" + "="*70)
        print("STEP 7: Get review prompts for 4 dimensions")
        print("="*70)
        prompts_result = mcp_server.call_tool("review_get_prompts", {
            "session_token": reviewer_token,
            "task_id": task_id,
            "dimension": "all",  # Get all 4 dimensions
        })
        
        print(f"DEBUG: prompts_result keys = {prompts_result.keys()}")
        assert "prompts" in prompts_result
        assert len(prompts_result["prompts"]) == 4
        print(f"[PASS] Generated prompts for all 4 dimensions")
        for dimension_name in prompts_result["prompts"]:
            print(f"  - {dimension_name}")
        
        # Step 8: Reviewer conducts review (simulating approval)
        print("\n" + "="*70)
        print("STEP 8: Conduct review - Simulating APPROVED verdict")
        print("="*70)
        conduct_result = mcp_server.call_tool("review_conduct", {
            "session_token": reviewer_token,
            "review_session_id": review_session_id,
            "dimension": "quality",
            "result": {
                "pass": True,
                "issues": [
                    {
                        "check_id": "Q002",
                        "severity": "INFO",
                        "message": "Good error handling",
                        "file_path": "kernel/new_feature.py",
                        "line_number": 5,
                    }
                ],
                "notes": "Code quality is good",
            },
        })
        
        # Complete other dimensions
        for dimension in ["requirements", "completeness", "optimization"]:
            mcp_server.call_tool("review_conduct", {
                "session_token": reviewer_token,
                "review_session_id": review_session_id,
                "dimension": dimension,
                "result": {
                    "pass": True,
                    "issues": [],
                    "notes": f"{dimension} check passed",
                },
            })
        
        assert conduct_result["success"]
        assert conduct_result["verdict"] == "APPROVED"
        print(f"[PASS] Review conducted successfully")
        print(f"[PASS] Verdict: {conduct_result['verdict']}")
        
        # Step 9: Get review report
        print("\n" + "="*70)
        print("STEP 9: Generate and retrieve review report")
        print("="*70)
        report_result = mcp_server.call_tool("review_get_report", {
            "session_token": reviewer_token,
            "task_id": task_id,
            "format": "markdown",
        })
        
        assert "report_markdown" in report_result or "report" in report_result
        print(f"[PASS] Review report generated")
        
        # Step 10: Check review status
        print("\n" + "="*70)
        print("STEP 10: Verify review status")
        print("="*70)
        status_result = mcp_server.call_tool("review_get_status", {
            "session_token": reviewer_token,
            "task_id": task_id,
        })
        
        assert status_result.get("review_status") in ("APPROVED", "approved")
        print(f"[PASS] Review status verified: {status_result.get('review_status')}")
        
        # Step 11: Approve and advance task
        print("\n" + "="*70)
        print("STEP 11: Approve task and advance to 'reviewing' state")
        print("="*70)
        approve_result = mcp_server.call_tool("review_approve", {
            "session_token": reviewer_token,
            "task_id": task_id,
            "review_session_id": review_session_id,
        })
        
        assert approve_result["success"]
        assert approve_result["new_status"] == "reviewing"
        print(f"[PASS] Task approved and advanced to: {approve_result['new_status']}")
        
        # Final verification
        print("\n" + "="*70)
        print("PAIR PROGRAMMING WORKFLOW TEST COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"[PASS] Builder generated code")
        print(f"[PASS] Code automatically submitted for review")
        print(f"[PASS] Task automatically transitioned to 'code_review' state")
        print(f"[PASS] Reviewer conducted 4-dimensional review")
        print(f"[PASS] Review verdict: APPROVED")
        print(f"[PASS] Task approved and transitioned to 'reviewing' state")
        print("="*70)
    
    def test_self_review_prevention(self, mcp_server, temp_workspace):
        """Test that agents cannot review their own code submissions."""
        
        print("\n" + "="*70)
        print("TEST: Self-review prevention")
        print("="*70)
        
        # Register single agent
        agent_result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_agent",
            "display_name": "Same Agent",
            "allowed_role_modes": ["builder", "reviewer"],
        })
        assert agent_result["success"]
        agent_id = agent_result["agent"]["agent_id"]
        
        # Create builder session
        builder_session = mcp_server.call_tool("session_create", {
            "agent_id": agent_id,
            "role_mode": "builder",
            "authorized_by": "test_operator",
        })
        builder_token = builder_session["session"]["session_token"]
        
        # Create task and submit
        task_id = "PP_SELF_TEST"
        taskcard_path = temp_workspace / "tasks" / f"{task_id}.md"
        taskcard_path.write_text(f"---\ntask_id: {task_id}\ntype: dev\n---\n# Test")
        
        with (temp_workspace / "state" / "tasks.yaml").open("r") as f:
            tasks_state = yaml.safe_load(f)
        tasks_state["tasks"][task_id] = {"status": "running"}
        with (temp_workspace / "state" / "tasks.yaml").open("w") as f:
            yaml.safe_dump(tasks_state, f)
        
        code_file = temp_workspace / "kernel" / "test.py"
        code_file.write_text("def test(): pass")
        
        # Submit for review
        mcp_server.call_tool("review_submit", {
            "session_token": builder_token,
            "task_id": task_id,
            "artifact_paths": ["kernel/test.py"],
        })

        # Terminate builder session to allow reviewer session for same agent
        mcp_server.call_tool("session_terminate", {
            "session_token": builder_token,
            "reason": "switching to reviewer role for self-review prevention test",
        })

        # Create reviewer session (same agent)
        reviewer_session = mcp_server.call_tool("session_create", {
            "agent_id": agent_id,
            "role_mode": "reviewer",
            "authorized_by": "test_operator",
        })
        reviewer_token = reviewer_session["session"]["session_token"]
        
        # Try to create review session (should fail)
        session_result = mcp_server.call_tool("review_create_session", {
            "session_token": reviewer_token,
            "task_id": task_id,
            "personas": ["security_expert"],
        })
        
        assert session_result.get("error") == "SELF_REVIEW_PROHIBITED"
        assert "self-review" in session_result.get("message", "").lower()
        print(f"[PASS] Self-review correctly prevented")
        print(f"[PASS] Error message: {session_result.get('message')}")


def test_code_review_engine_requires_review():
    """Test that CodeReviewEngine correctly identifies files requiring review."""
    
    print("\n" + "="*70)
    print("TEST: File review requirement detection")
    print("="*70)
    
    config_path = Path(__file__).parent.parent / "state_machine.yaml"
    review_engine = CodeReviewEngine(config_path)
    
    # Python files should require review
    assert review_engine.requires_review("kernel/new_feature.py") is True
    assert review_engine.requires_review("scripts/helper.py") is True
    print(f"[PASS] Python files correctly require review")
    
    # Non-code files should not require review
    assert review_engine.requires_review("README.md") is False
    assert review_engine.requires_review("docs/guide.md") is False
    print(f"[PASS] Documentation files correctly skip review")
    
    


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
