# Spec Evolution Workflow - E2E Validation Test
# Created: 2026-02-04
# Purpose: Validate the Skills + MCP + Hooks integration for spec evolution workflow
#
# Test Scenario: OOS Sharpe threshold is too lenient, needs to be increased from 1.0 to 1.5
#
# This test simulates the full workflow:
# 1. Problem Discovery (experiment failure)
# 2. Triage (spec_triage)
# 3. Research (existing /dgsf_research skill)
# 4. Proposal (spec_propose)
# 5. Approval (manual simulation)
# 6. Commit (spec_commit)
# 7. Verification (post-commit checks)

"""
E2E Test for Spec Evolution Workflow

Run with:
    pytest projects/dgsf/tests/test_spec_evolution_e2e.py -v

Prerequisites:
    - MCP Server running or importable
    - DGSF project structure in place
    - Valid session token (or mock)
"""

import pytest
import os
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestSpecEvolutionWorkflow:
    """End-to-end tests for the spec evolution workflow."""
    
    @pytest.fixture
    def mcp_server(self):
        """Create MCP server instance with mocked auth."""
        from kernel.mcp_server import MCPServer
        
        server = MCPServer()
        
        # Mock authentication for testing
        mock_session = Mock()
        mock_session.agent_id = "test-agent-001"
        mock_session.session_token = "test-session-token-12345"
        mock_session.role_mode = Mock()
        mock_session.role_mode.value = "planner"
        
        # Import RoleMode enum
        from kernel.agent_auth import RoleMode
        mock_session.role_mode = RoleMode.PLANNER
        
        server.auth_manager.get_session = Mock(return_value=mock_session)
        server.auth_manager.validate_session = Mock(return_value=True)
        
        return server
    
    @pytest.fixture
    def test_session_token(self):
        """Return a valid test session token."""
        return "test-session-token-12345"
    
    # =========================================================================
    # Phase 1: Spec Triage
    # =========================================================================
    
    def test_spec_triage_metric_deviation(self, mcp_server, test_session_token):
        """Test that spec_triage correctly classifies metric deviation problems."""
        
        result = mcp_server.call_tool("spec_triage", {
            "session_token": test_session_token,
            "problem_description": "Experiment t05_momentum has OOS Sharpe = 0.8, below threshold. OOS/IS ratio is only 0.5 indicating overfitting.",
            "source": "experiment",
            "context": {
                "experiment_id": "t05_momentum",
                "oos_sharpe": 0.8,
                "is_sharpe": 1.6
            }
        })
        
        assert "error" not in result, f"Triage failed: {result.get('error')}"
        assert result["triage_id"].startswith("TRI-")
        assert result["classification"]["category"] == "metric_deviation"
        assert result["classification"]["root_cause"] in ["spec_issue", "model_issue"]
        assert result["recommended_action"]["type"] in ["spec_research", "manual_investigation"]
        
        print(f"\n[TRIAGE RESULT]")
        print(f"  ID: {result['triage_id']}")
        print(f"  Category: {result['classification']['category']}")
        print(f"  Root Cause: {result['classification']['root_cause']}")
        print(f"  Priority: {result['priority']['level']}")
        print(f"  Recommended: {result['recommended_action']['command']}")
    
    def test_spec_triage_assertion_error(self, mcp_server, test_session_token):
        """Test that spec_triage correctly classifies assertion errors as spec issues."""
        
        result = mcp_server.call_tool("spec_triage", {
            "session_token": test_session_token,
            "problem_description": "AssertionError: Expected min_sharpe >= 1.5 but threshold is set to 1.0 in spec",
            "source": "test"
        })
        
        assert "error" not in result
        # Note: assertion errors with "threshold" keyword may be classified as metric_deviation
        # because the heuristic looks for threshold-related keywords
        assert result["classification"]["category"] in ["runtime_error", "metric_deviation"]
        assert result["classification"]["root_cause"] == "spec_issue"
        
    def test_spec_triage_code_bug(self, mcp_server, test_session_token):
        """Test that spec_triage correctly classifies runtime errors as code bugs."""
        
        result = mcp_server.call_tool("spec_triage", {
            "session_token": test_session_token,
            "problem_description": "TypeError: NoneType object is not subscriptable in model.py line 42",
            "source": "test"
        })
        
        assert "error" not in result
        assert result["classification"]["category"] == "runtime_error"
        assert result["classification"]["root_cause"] == "code_bug"
        assert result["recommended_action"]["command"] == "/dgsf_diagnose"
    
    # =========================================================================
    # Phase 2: Spec Read
    # =========================================================================
    
    def test_spec_read_existing_file(self, mcp_server, test_session_token, tmp_path):
        """Test reading an existing spec file."""
        
        # Create a temporary spec file
        spec_content = """
validation:
  min_sharpe_threshold: 1.0
  max_drawdown: 0.25
  oos_is_ratio_min: 0.8
"""
        spec_dir = tmp_path / "projects" / "dgsf" / "specs"
        spec_dir.mkdir(parents=True)
        spec_file = spec_dir / "TEST_SPEC.yaml"
        spec_file.write_text(spec_content)
        
        # Temporarily override root
        original_root = mcp_server.root
        mcp_server.root = tmp_path
        
        try:
            result = mcp_server.call_tool("spec_read", {
                "session_token": test_session_token,
                "spec_path": "projects/dgsf/specs/TEST_SPEC.yaml"
            })
            
            assert "error" not in result
            assert result["exists"] is True
            assert "min_sharpe_threshold" in result["content"]
            assert result["layer"] == "L2"
            assert result["editable_by_ai"] is False  # L2 requires approval
        finally:
            mcp_server.root = original_root
    
    def test_spec_read_with_section(self, mcp_server, test_session_token, tmp_path):
        """Test reading a specific section of a spec file."""
        
        spec_content = """
validation:
  thresholds:
    min_sharpe: 1.0
    max_drawdown: 0.25
  checks:
    - point_in_time
    - no_lookahead
"""
        spec_dir = tmp_path / "projects" / "dgsf" / "specs"
        spec_dir.mkdir(parents=True)
        spec_file = spec_dir / "TEST_SPEC.yaml"
        spec_file.write_text(spec_content)
        
        original_root = mcp_server.root
        mcp_server.root = tmp_path
        
        try:
            result = mcp_server.call_tool("spec_read", {
                "session_token": test_session_token,
                "spec_path": "projects/dgsf/specs/TEST_SPEC.yaml",
                "section": "validation.thresholds"
            })
            
            assert "error" not in result
            assert result["section"] == "validation.thresholds"
            assert result["section_content"]["min_sharpe"] == 1.0
        finally:
            mcp_server.root = original_root
    
    def test_spec_read_nonexistent(self, mcp_server, test_session_token):
        """Test reading a non-existent spec file."""
        
        result = mcp_server.call_tool("spec_read", {
            "session_token": test_session_token,
            "spec_path": "projects/dgsf/specs/NONEXISTENT.yaml"
        })
        
        assert result.get("error") == "SPEC_NOT_FOUND"
    
    # =========================================================================
    # Phase 3: Spec Propose
    # =========================================================================
    
    def test_spec_propose_valid(self, mcp_server, test_session_token, tmp_path):
        """Test creating a valid spec change proposal."""
        
        # Setup directories
        decisions_dir = tmp_path / "projects" / "dgsf" / "decisions"
        decisions_dir.mkdir(parents=True)
        
        original_root = mcp_server.root
        mcp_server.root = tmp_path
        
        try:
            result = mcp_server.call_tool("spec_propose", {
                "session_token": test_session_token,
                "spec_path": "projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml",
                "change_type": "modify",
                "rationale": "Increase min_sharpe threshold from 1.0 to 1.5 based on research showing industry standard is 1.5 for production SDF models.",
                "proposed_diff": """--- a/projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml
+++ b/projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml
@@ -45,7 +45,7 @@ validation:
-  min_sharpe_threshold: 1.0
+  min_sharpe_threshold: 1.5
   max_drawdown: 0.25""",
                "evidence_refs": [
                    "experiments/t03_threshold_study/results.json",
                    "research/sharpe_analysis.md"
                ]
            })
            
            assert "error" not in result, f"Proposal failed: {result.get('error')}"
            assert result["success"] is True
            assert result["proposal_id"].startswith("SCP-")
            assert result["status"] == "proposed"
            assert "Project Lead" in result["required_approval_from"]
            
            # Verify proposal file was created
            proposal_file = decisions_dir / f"{result['proposal_id']}.yaml"
            assert proposal_file.exists()
            
            print(f"\n[PROPOSAL CREATED]")
            print(f"  ID: {result['proposal_id']}")
            print(f"  File: {result['proposal_file']}")
            print(f"  Required Approval: {result['required_approval_from']}")
            
        finally:
            mcp_server.root = original_root
    
    def test_spec_propose_canon_blocked(self, mcp_server, test_session_token):
        """Test that proposing changes to Canon specs is blocked."""
        
        result = mcp_server.call_tool("spec_propose", {
            "session_token": test_session_token,
            "spec_path": "specs/canon/GOVERNANCE_INVARIANTS.md",
            "change_type": "modify",
            "rationale": "Trying to modify canon spec",
            "proposed_diff": "some diff"
        })
        
        assert result.get("error") == "CANON_PROTECTED"
    
    def test_spec_propose_role_violation(self, mcp_server, test_session_token, tmp_path):
        """Test that executor role cannot propose spec changes."""
        
        from kernel.agent_auth import RoleMode
        
        # Change mock session to executor role
        mock_session = mcp_server.auth_manager.get_session.return_value
        mock_session.role_mode = RoleMode.EXECUTOR
        
        decisions_dir = tmp_path / "projects" / "dgsf" / "decisions"
        decisions_dir.mkdir(parents=True)
        
        original_root = mcp_server.root
        mcp_server.root = tmp_path
        
        try:
            result = mcp_server.call_tool("spec_propose", {
                "session_token": test_session_token,
                "spec_path": "projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml",
                "change_type": "modify",
                "rationale": "Test rationale",
                "proposed_diff": "test diff"
            })
            
            assert result.get("error") == "ROLE_MODE_VIOLATION"
        finally:
            mcp_server.root = original_root
            mock_session.role_mode = RoleMode.PLANNER  # Restore
    
    # =========================================================================
    # Phase 4: Spec Commit
    # =========================================================================
    
    def test_spec_commit_with_approval(self, mcp_server, test_session_token, tmp_path):
        """Test committing an approved spec change."""
        
        import yaml
        
        # Setup directories and create a proposal
        decisions_dir = tmp_path / "projects" / "dgsf" / "decisions"
        decisions_dir.mkdir(parents=True)
        
        # Create the spec file that will be modified
        specs_dir = tmp_path / "projects" / "dgsf" / "specs"
        specs_dir.mkdir(parents=True)
        spec_file = specs_dir / "SDF_INTERFACE_CONTRACT.yaml"
        spec_file.write_text("validation:\n  min_sharpe_threshold: 1.0\n")
        
        proposal_id = "SCP-2026-02-04-001"
        proposal = {
            "id": proposal_id,
            "spec_path": "projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml",
            "change_type": "modify",
            "status": "proposed",
            "proposed_by": {"agent_id": "test", "role_mode": "planner"},
            "rationale": "Test rationale",
            "proposed_diff": "test diff",
            "approval": {"required_from": "Project Lead", "approved_by": None}
        }
        
        proposal_file = decisions_dir / f"{proposal_id}.yaml"
        with proposal_file.open("w") as f:
            yaml.dump(proposal, f)
        
        # Create the approval reference (simulating human approval)
        approval_ref = "APPROVED-by-project-lead"
        approval_file = decisions_dir / f"{approval_ref}.yaml"
        with approval_file.open("w") as f:
            yaml.dump({"approved": True, "by": "Project Lead", "date": "2026-02-04"}, f)
        
        original_root = mcp_server.root
        mcp_server.root = tmp_path
        
        try:
            result = mcp_server.call_tool("spec_commit", {
                "session_token": test_session_token,
                "proposal_id": proposal_id,
                "approval_ref": approval_ref,
                "run_hooks": False  # Skip hooks for unit test
            })
            
            assert "error" not in result, f"Commit failed: {result.get('error')}"
            assert result["success"] is True
            assert result["status"] == "committed"
            
            # Verify proposal status was updated
            with proposal_file.open() as f:
                updated_proposal = yaml.safe_load(f)
            assert updated_proposal["status"] == "committed"
            assert updated_proposal["approval"]["approved_by"] == approval_ref
            
            print(f"\n[COMMIT SUCCESSFUL]")
            print(f"  Proposal: {result['proposal_id']}")
            print(f"  Approval: {result['approval_ref']}")
            
        finally:
            mcp_server.root = original_root
    
    def test_spec_commit_no_approval(self, mcp_server, test_session_token, tmp_path):
        """Test that commit fails without approval."""
        
        import yaml
        
        decisions_dir = tmp_path / "projects" / "dgsf" / "decisions"
        decisions_dir.mkdir(parents=True)
        
        proposal_id = "SCP-2026-02-04-002"
        proposal = {
            "id": proposal_id,
            "spec_path": "projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml",
            "change_type": "modify",
            "status": "proposed",
        }
        
        proposal_file = decisions_dir / f"{proposal_id}.yaml"
        with proposal_file.open("w") as f:
            yaml.dump(proposal, f)
        
        original_root = mcp_server.root
        mcp_server.root = tmp_path
        
        try:
            result = mcp_server.call_tool("spec_commit", {
                "session_token": test_session_token,
                "proposal_id": proposal_id,
                "approval_ref": "nonexistent-approval"
            })
            
            assert result.get("error") == "APPROVAL_NOT_FOUND"
            
        finally:
            mcp_server.root = original_root
    
    # =========================================================================
    # Integration: Full Workflow Simulation
    # =========================================================================
    
    def test_full_workflow_simulation(self, mcp_server, test_session_token, tmp_path):
        """
        Simulate the complete spec evolution workflow:
        1. Triage → 2. Read → 3. Propose → 4. (Manual Approval) → 5. Commit
        """
        import yaml
        
        print("\n" + "="*60)
        print("FULL WORKFLOW SIMULATION: Increase min_sharpe threshold")
        print("="*60)
        
        # Setup
        decisions_dir = tmp_path / "projects" / "dgsf" / "decisions"
        decisions_dir.mkdir(parents=True)
        
        specs_dir = tmp_path / "projects" / "dgsf" / "specs"
        specs_dir.mkdir(parents=True)
        
        # Create the spec file
        spec_content = """
# SDF Interface Contract
validation:
  min_sharpe_threshold: 1.0
  max_drawdown: 0.25
"""
        spec_file = specs_dir / "SDF_INTERFACE_CONTRACT.yaml"
        spec_file.write_text(spec_content)
        
        original_root = mcp_server.root
        mcp_server.root = tmp_path
        
        try:
            # Step 1: Triage
            print("\n[Step 1] TRIAGE - Analyzing problem...")
            triage_result = mcp_server.call_tool("spec_triage", {
                "session_token": test_session_token,
                "problem_description": "OOS Sharpe of 0.8 is below industry standard of 1.5. Current threshold 1.0 is too lenient.",
                "source": "experiment"
            })
            assert "error" not in triage_result
            print(f"  → Category: {triage_result['classification']['category']}")
            print(f"  → Root Cause: {triage_result['classification']['root_cause']}")
            print(f"  → Recommended: {triage_result['recommended_action']['command']}")
            
            # Step 2: Read current spec
            print("\n[Step 2] READ - Checking current spec...")
            read_result = mcp_server.call_tool("spec_read", {
                "session_token": test_session_token,
                "spec_path": "projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml"
            })
            assert "error" not in read_result
            print(f"  → Layer: {read_result['layer']}")
            print(f"  → Editable by AI: {read_result['editable_by_ai']}")
            
            # Step 3: Propose change
            print("\n[Step 3] PROPOSE - Creating spec change proposal...")
            propose_result = mcp_server.call_tool("spec_propose", {
                "session_token": test_session_token,
                "spec_path": "projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml",
                "change_type": "modify",
                "rationale": "Increase min_sharpe from 1.0 to 1.5 based on industry standard.",
                "proposed_diff": "-  min_sharpe_threshold: 1.0\n+  min_sharpe_threshold: 1.5"
            })
            assert "error" not in propose_result
            proposal_id = propose_result["proposal_id"]
            print(f"  → Proposal ID: {proposal_id}")
            print(f"  → Awaiting: {propose_result['required_approval_from']}")
            
            # Step 4: Simulate human approval
            print("\n[Step 4] APPROVAL - Simulating human approval...")
            approval_ref = "manual-approval-001"
            approval_file = decisions_dir / f"{approval_ref}.yaml"
            with approval_file.open("w") as f:
                yaml.dump({
                    "approved": True,
                    "by": "Project Lead",
                    "date": "2026-02-04",
                    "proposal_id": proposal_id
                }, f)
            print(f"  → Approval created: {approval_ref}")
            
            # Step 5: Commit
            print("\n[Step 5] COMMIT - Applying spec change...")
            commit_result = mcp_server.call_tool("spec_commit", {
                "session_token": test_session_token,
                "proposal_id": proposal_id,
                "approval_ref": approval_ref,
                "run_hooks": False
            })
            assert "error" not in commit_result
            print(f"  → Status: {commit_result['status']}")
            print(f"  → Message: {commit_result['message']}")
            
            print("\n" + "="*60)
            print("✓ WORKFLOW COMPLETED SUCCESSFULLY")
            print("="*60)
            
        finally:
            mcp_server.root = original_root


class TestHooksIntegration:
    """Test the pre/post spec-change hooks."""
    
    def test_pre_spec_change_hook_exists(self):
        """Verify pre-spec-change hook file exists."""
        hook_path = REPO_ROOT / "hooks" / "pre-spec-change"
        assert hook_path.exists(), "pre-spec-change hook not found"
    
    def test_post_spec_change_hook_exists(self):
        """Verify post-spec-change hook file exists."""
        hook_path = REPO_ROOT / "hooks" / "post-spec-change"
        assert hook_path.exists(), "post-spec-change hook not found"


class TestSkillsExist:
    """Verify all required skill prompts exist."""
    
    @pytest.mark.parametrize("skill_name", [
        "dgsf_spec_propose",
        "dgsf_spec_commit", 
        "dgsf_spec_triage"
    ])
    def test_skill_prompt_exists(self, skill_name):
        """Verify skill prompt file exists."""
        prompt_path = REPO_ROOT / ".github" / "prompts" / f"{skill_name}.prompt.md"
        assert prompt_path.exists(), f"Skill prompt {skill_name} not found"
        
        # Verify it has required frontmatter (with explicit UTF-8 encoding)
        content = prompt_path.read_text(encoding='utf-8')
        assert "---" in content, "Missing YAML frontmatter"
        assert "description:" in content, "Missing description field"
        assert "mode: agent" in content, "Missing mode: agent field"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
