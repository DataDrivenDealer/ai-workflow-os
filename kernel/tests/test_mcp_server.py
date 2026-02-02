"""
Unit tests for kernel/mcp_server.py

Tests cover:
- MCPServer initialization and tool registration
- Tool dispatch via call_tool
- Individual tool implementations
- Session validation in tool calls

Author: 李质量 (QA Test Engineer)
Date: 2026-02-01
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server import MCPServer
from agent_auth import reset_auth_manager


class TestMCPServerInit:
    """Tests for MCPServer initialization."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    def test_server_creation(self, mcp_server: MCPServer):
        """MCPServer should initialize successfully."""
        assert mcp_server is not None
        assert mcp_server.auth_manager is not None
        assert mcp_server.governance_gate is not None
        assert mcp_server.root is not None


class TestMCPServerTools:
    """Tests for MCPServer tool definitions."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    def test_get_tools_returns_list(self, mcp_server: MCPServer):
        """get_tools should return a list of tool definitions."""
        tools = mcp_server.get_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_get_tools_expected_tools(self, mcp_server: MCPServer):
        """get_tools should include all expected tools."""
        tools = mcp_server.get_tools()
        tool_names = {t["name"] for t in tools}
        
        expected_tools = {
            "agent_register",
            "session_create",
            "session_validate",
            "session_terminate",
            "task_list",
            "task_get",
            "task_start",
            "task_finish",
            "governance_check",
            "artifact_read",
            "artifact_list",
            "spec_list",
        }
        
        assert expected_tools.issubset(tool_names)

    def test_tool_schema_structure(self, mcp_server: MCPServer):
        """Each tool should have proper schema structure."""
        tools = mcp_server.get_tools()
        
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]

    def test_tool_count(self, mcp_server: MCPServer):
        """get_tools should return 20 tools (12 original + 8 code review)."""
        tools = mcp_server.get_tools()
        assert len(tools) == 20


class TestMCPServerCallTool:
    """Tests for MCPServer.call_tool dispatch."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    def test_call_unknown_tool(self, mcp_server: MCPServer):
        """call_tool should return error for unknown tool."""
        result = mcp_server.call_tool("unknown_tool", {})
        
        # Should have error indicator
        assert "error" in result or result.get("success") is False

    def test_call_agent_register(self, mcp_server: MCPServer):
        """call_tool should dispatch agent_register correctly."""
        result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_test",
            "display_name": "Test Agent",
            "allowed_role_modes": ["executor"],
        })
        
        assert result.get("success") is True
        assert "agent" in result
        assert "agent_id" in result["agent"]


class TestAgentRegisterTool:
    """Tests for agent_register tool."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    def test_register_with_roles(self, mcp_server: MCPServer):
        """agent_register should work with specified roles."""
        result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_assistant",
            "display_name": "Test Assistant",
            "allowed_role_modes": ["executor"],
        })
        
        assert result.get("success") is True
        assert "agent" in result

    def test_register_with_multiple_roles(self, mcp_server: MCPServer):
        """agent_register should accept multiple roles."""
        result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_assistant",
            "display_name": "Multi-Role Agent",
            "allowed_role_modes": ["executor", "builder", "planner"],
        })
        
        assert result.get("success") is True
        agent = result["agent"]
        # Verify roles are stored
        assert len(agent["allowed_role_modes"]) == 3

    def test_register_missing_roles(self, mcp_server: MCPServer):
        """agent_register should fail without allowed_role_modes."""
        result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_assistant",
            "display_name": "Test Assistant",
        })
        
        # Should indicate error
        assert "error" in result


class TestSessionTools:
    """Tests for session management tools."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    @pytest.fixture
    def registered_agent_id(self, mcp_server: MCPServer) -> str:
        """Register an agent and return agent_id."""
        result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_test",
            "display_name": "Test Agent",
            "allowed_role_modes": ["executor", "builder"],
        })
        return result["agent"]["agent_id"]

    def test_session_create(self, mcp_server: MCPServer, registered_agent_id: str):
        """session_create should create new session."""
        result = mcp_server.call_tool("session_create", {
            "agent_id": registered_agent_id,
            "role_mode": "executor",
            "authorized_by": "test_harness",
        })
        
        assert result.get("success") is True
        assert "session" in result
        assert "session_token" in result["session"]

    def test_session_create_unknown_agent(self, mcp_server: MCPServer):
        """session_create should fail for unknown agent."""
        result = mcp_server.call_tool("session_create", {
            "agent_id": "nonexistent-agent",
            "role_mode": "executor",
            "authorized_by": "test_harness",
        })
        
        assert result.get("success") is False or "error" in result

    def test_session_validate_active(self, mcp_server: MCPServer, registered_agent_id: str):
        """session_validate should return valid for active session."""
        # Create session
        create_result = mcp_server.call_tool("session_create", {
            "agent_id": registered_agent_id,
            "role_mode": "executor",
            "authorized_by": "test",
        })
        session_token = create_result["session"]["session_token"]
        
        # Validate session
        result = mcp_server.call_tool("session_validate", {
            "session_token": session_token,
        })
        
        assert result.get("valid") is True

    def test_session_validate_invalid_token(self, mcp_server: MCPServer):
        """session_validate should return invalid for unknown token."""
        result = mcp_server.call_tool("session_validate", {
            "session_token": "invalid-token-12345",
        })
        
        assert result.get("valid") is False

    def test_session_terminate(self, mcp_server: MCPServer, registered_agent_id: str):
        """session_terminate should end session."""
        # Create session
        create_result = mcp_server.call_tool("session_create", {
            "agent_id": registered_agent_id,
            "role_mode": "executor",
            "authorized_by": "test",
        })
        session_token = create_result["session"]["session_token"]
        
        # Terminate session
        term_result = mcp_server.call_tool("session_terminate", {
            "session_token": session_token,
            "reason": "test_complete",
        })
        assert term_result.get("success") is True
        
        # Validate should now fail
        validate_result = mcp_server.call_tool("session_validate", {
            "session_token": session_token,
        })
        assert validate_result.get("valid") is False


class TestTaskTools:
    """Tests for task management tools."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    @pytest.fixture
    def active_session_token(self, mcp_server: MCPServer) -> str:
        """Create agent with active session and return token."""
        reg_result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_test",
            "display_name": "Test Agent",
            "allowed_role_modes": ["executor"],
        })
        sess_result = mcp_server.call_tool("session_create", {
            "agent_id": reg_result["agent"]["agent_id"],
            "role_mode": "executor",
            "authorized_by": "test",
        })
        return sess_result["session"]["session_token"]

    def test_task_list_requires_session(self, mcp_server: MCPServer, active_session_token: str):
        """task_list should work with valid session."""
        result = mcp_server.call_tool("task_list", {
            "session_token": active_session_token,
        })
        
        # Either returns tasks or indicates success
        assert "tasks" in result or result.get("success") is True

    def test_task_get_not_found(self, mcp_server: MCPServer, active_session_token: str):
        """task_get should return error for unknown task."""
        result = mcp_server.call_tool("task_get", {
            "task_id": "TASK_NONEXISTENT_9999",
            "session_token": active_session_token,
        })
        
        # Should indicate error or not found
        assert "error" in result or result.get("success") is False or result.get("task") is None

    def test_task_start_without_session(self, mcp_server: MCPServer):
        """task_start should require valid session."""
        result = mcp_server.call_tool("task_start", {
            "task_id": "TASK_TEST_001",
            "session_token": "invalid-session",
        })
        
        # Should fail due to invalid session
        assert "error" in result or result.get("success") is False


class TestGovernanceTool:
    """Tests for governance_check tool."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    def test_governance_check_exists(self, mcp_server: MCPServer):
        """governance_check tool should be available."""
        tools = mcp_server.get_tools()
        tool_names = {t["name"] for t in tools}
        assert "governance_check" in tool_names

    def test_governance_check_returns_result(self, mcp_server: MCPServer):
        """governance_check should return some result."""
        result = mcp_server.call_tool("governance_check", {
            "action": "test_action",
            "artifact_path": "tasks/TASK_001.md",
        })
        
        # Should return some structured result
        assert isinstance(result, dict)


class TestArtifactTools:
    """Tests for artifact management tools."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    @pytest.fixture
    def active_session_token(self, mcp_server: MCPServer) -> str:
        """Create agent with active session and return token."""
        reg_result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_test",
            "display_name": "Test Agent",
            "allowed_role_modes": ["executor"],
        })
        sess_result = mcp_server.call_tool("session_create", {
            "agent_id": reg_result["agent"]["agent_id"],
            "role_mode": "executor",
            "authorized_by": "test",
        })
        return sess_result["session"]["session_token"]

    def test_artifact_list(self, mcp_server: MCPServer, active_session_token: str):
        """artifact_list should return artifacts structure."""
        result = mcp_server.call_tool("artifact_list", {
            "artifact_type": "spec",
            "session_token": active_session_token,
        })
        
        # Should have some result structure
        assert isinstance(result, dict)


class TestSpecListTool:
    """Tests for spec_list tool."""

    @pytest.fixture
    def mcp_server(self) -> MCPServer:
        """Create a fresh MCP server for each test."""
        reset_auth_manager()
        return MCPServer()

    @pytest.fixture
    def active_session_token(self, mcp_server: MCPServer) -> str:
        """Create agent with active session and return token."""
        reg_result = mcp_server.call_tool("agent_register", {
            "agent_type": "ai_test",
            "display_name": "Test Agent",
            "allowed_role_modes": ["executor"],
        })
        sess_result = mcp_server.call_tool("session_create", {
            "agent_id": reg_result["agent"]["agent_id"],
            "role_mode": "executor",
            "authorized_by": "test",
        })
        return sess_result["session"]["session_token"]

    def test_spec_list(self, mcp_server: MCPServer, active_session_token: str):
        """spec_list should return specs structure."""
        result = mcp_server.call_tool("spec_list", {
            "session_token": active_session_token,
        })
        
        # Should have specs array
        assert isinstance(result, dict)
        assert "specs" in result

    def test_spec_list_by_category(self, mcp_server: MCPServer, active_session_token: str):
        """spec_list should support category filter."""
        result = mcp_server.call_tool("spec_list", {
            "category": "canon",
            "session_token": active_session_token,
        })
        
        # Should return filtered results
        assert isinstance(result, dict)
        assert "specs" in result
