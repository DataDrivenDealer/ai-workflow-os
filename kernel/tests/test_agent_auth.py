"""
Unit tests for kernel/agent_auth.py

Tests cover:
- AgentIdentity dataclass and serialization
- AgentSession lifecycle and state management
- AgentAuthManager registration, session, and concurrency control

Author: 李质量 (QA Test Engineer)
Date: 2026-02-01
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_auth import (
    AgentAuthManager,
    AgentIdentity,
    AgentSession,
    RoleMode,
    SessionState,
    reset_auth_manager,
)


class TestRoleMode:
    """Tests for RoleMode enum."""

    def test_role_mode_values(self):
        """RoleMode should have correct canonical values."""
        assert RoleMode.ARCHITECT.value == "architect"
        assert RoleMode.PLANNER.value == "planner"
        assert RoleMode.EXECUTOR.value == "executor"
        assert RoleMode.BUILDER.value == "builder"

    def test_role_mode_from_string(self):
        """RoleMode should be creatable from string."""
        assert RoleMode("architect") == RoleMode.ARCHITECT
        assert RoleMode("executor") == RoleMode.EXECUTOR


class TestSessionState:
    """Tests for SessionState enum."""

    def test_session_state_values(self):
        """SessionState should have correct values."""
        assert SessionState.ANONYMOUS.value == "anonymous"
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.SUSPENDED.value == "suspended"
        assert SessionState.TERMINATED.value == "terminated"


class TestAgentIdentity:
    """Tests for AgentIdentity dataclass."""

    def test_create_agent_identity(self):
        """AgentIdentity should be creatable with minimal args."""
        agent = AgentIdentity(
            agent_id="test-agent-001",
            agent_type="ai_test",
            display_name="Test Agent",
        )
        assert agent.agent_id == "test-agent-001"
        assert agent.agent_type == "ai_test"
        assert agent.display_name == "Test Agent"
        assert RoleMode.EXECUTOR in agent.allowed_role_modes

    def test_agent_identity_to_dict(self):
        """AgentIdentity.to_dict should serialize correctly."""
        agent = AgentIdentity(
            agent_id="test-agent-001",
            agent_type="ai_test",
            display_name="Test Agent",
            allowed_role_modes={RoleMode.EXECUTOR, RoleMode.BUILDER},
        )
        data = agent.to_dict()
        
        assert data["agent_id"] == "test-agent-001"
        assert data["agent_type"] == "ai_test"
        assert set(data["allowed_role_modes"]) == {"executor", "builder"}
        assert "registered_at" in data

    def test_agent_identity_from_dict(self):
        """AgentIdentity.from_dict should deserialize correctly."""
        data = {
            "agent_id": "test-agent-001",
            "agent_type": "ai_test",
            "display_name": "Test Agent",
            "registered_at": "2026-02-01T12:00:00+00:00",
            "allowed_role_modes": ["executor", "builder"],
            "metadata": {"key": "value"},
        }
        agent = AgentIdentity.from_dict(data)
        
        assert agent.agent_id == "test-agent-001"
        assert RoleMode.BUILDER in agent.allowed_role_modes
        assert agent.metadata["key"] == "value"


class TestAgentSession:
    """Tests for AgentSession dataclass."""

    def test_create_session(self):
        """AgentSession should be creatable."""
        session = AgentSession(
            session_token="sess-test-001",
            agent_id="test-agent-001",
            role_mode=RoleMode.EXECUTOR,
            state=SessionState.ACTIVE,
            started_at=datetime.now(timezone.utc),
            authorized_by="test_harness",
        )
        assert session.session_token == "sess-test-001"
        assert session.role_mode == RoleMode.EXECUTOR
        assert session.state == SessionState.ACTIVE

    def test_session_is_active(self):
        """is_active should return True for active, non-expired sessions."""
        session = AgentSession(
            session_token="sess-test-001",
            agent_id="test-agent-001",
            role_mode=RoleMode.EXECUTOR,
            state=SessionState.ACTIVE,
            started_at=datetime.now(timezone.utc),
            authorized_by="test",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert session.is_active is True

    def test_session_is_active_expired(self):
        """is_active should return False for expired sessions."""
        session = AgentSession(
            session_token="sess-test-001",
            agent_id="test-agent-001",
            role_mode=RoleMode.EXECUTOR,
            state=SessionState.ACTIVE,
            started_at=datetime.now(timezone.utc) - timedelta(hours=2),
            authorized_by="test",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),  # Already expired
        )
        assert session.is_active is False

    def test_session_is_active_terminated(self):
        """is_active should return False for terminated sessions."""
        session = AgentSession(
            session_token="sess-test-001",
            agent_id="test-agent-001",
            role_mode=RoleMode.EXECUTOR,
            state=SessionState.TERMINATED,
            started_at=datetime.now(timezone.utc),
            authorized_by="test",
        )
        assert session.is_active is False

    def test_session_add_event(self):
        """add_event should append to events list."""
        session = AgentSession(
            session_token="sess-test-001",
            agent_id="test-agent-001",
            role_mode=RoleMode.EXECUTOR,
            state=SessionState.ACTIVE,
            started_at=datetime.now(timezone.utc),
            authorized_by="test",
        )
        session.add_event("test_action", {"key": "value"})
        
        assert len(session.events) == 1
        assert session.events[0]["action"] == "test_action"
        assert session.events[0]["details"]["key"] == "value"

    def test_session_to_dict(self):
        """AgentSession.to_dict should serialize correctly."""
        session = AgentSession(
            session_token="sess-test-001",
            agent_id="test-agent-001",
            role_mode=RoleMode.EXECUTOR,
            state=SessionState.ACTIVE,
            started_at=datetime.now(timezone.utc),
            authorized_by="test",
            task_scope=["TASK_001"],
        )
        data = session.to_dict()
        
        assert data["session_token"] == "sess-test-001"
        assert data["role_mode"] == "executor"
        assert data["state"] == "active"
        assert "TASK_001" in data["task_scope"]

    def test_session_from_dict(self):
        """AgentSession.from_dict should deserialize correctly."""
        data = {
            "session_token": "sess-test-001",
            "agent_id": "test-agent-001",
            "role_mode": "builder",
            "state": "active",
            "started_at": "2026-02-01T12:00:00+00:00",
            "authorized_by": "test",
            "expires_at": None,
            "task_scope": ["TASK_001"],
            "parent_session": None,
            "events": [],
            "pending_artifacts": ["tasks/TASK_001.md"],
        }
        session = AgentSession.from_dict(data)
        
        assert session.session_token == "sess-test-001"
        assert session.role_mode == RoleMode.BUILDER
        assert "tasks/TASK_001.md" in session.pending_artifacts


class TestAgentAuthManager:
    """Tests for AgentAuthManager class."""

    @pytest.fixture
    def temp_state_dir(self, tmp_path: Path) -> Path:
        """Create a temporary state directory."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        return state_dir

    @pytest.fixture
    def auth_manager(self, temp_state_dir: Path) -> AgentAuthManager:
        """Create a fresh auth manager for each test."""
        reset_auth_manager()
        return AgentAuthManager(state_dir=temp_state_dir)

    # =========================================================================
    # Agent Registration Tests
    # =========================================================================

    def test_register_agent(self, auth_manager: AgentAuthManager):
        """register_agent should create new agent identity."""
        agent = auth_manager.register_agent(
            agent_type="ai_test",
            display_name="Test Agent",
            allowed_role_modes=["executor"],
        )
        
        assert agent.agent_id is not None
        assert agent.agent_type == "ai_test"
        assert agent.display_name == "Test Agent"
        assert RoleMode.EXECUTOR in agent.allowed_role_modes

    def test_register_agent_with_multiple_roles(self, auth_manager: AgentAuthManager):
        """register_agent should allow multiple role modes."""
        agent = auth_manager.register_agent(
            agent_type="ai_test",
            display_name="Multi-Role Agent",
            allowed_role_modes=["executor", "builder", "planner"],
        )
        
        assert RoleMode.EXECUTOR in agent.allowed_role_modes
        assert RoleMode.BUILDER in agent.allowed_role_modes
        assert RoleMode.PLANNER in agent.allowed_role_modes

    def test_get_agent(self, auth_manager: AgentAuthManager):
        """get_agent should retrieve registered agent."""
        agent = auth_manager.register_agent(
            agent_type="ai_test",
            display_name="Test Agent",
        )
        
        retrieved = auth_manager.get_agent(agent.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == agent.agent_id

    def test_get_agent_not_found(self, auth_manager: AgentAuthManager):
        """get_agent should return None for unknown agent."""
        result = auth_manager.get_agent("nonexistent-agent")
        assert result is None

    def test_list_agents(self, auth_manager: AgentAuthManager):
        """list_agents should return all registered agents."""
        auth_manager.register_agent("ai_test", "Agent 1")
        auth_manager.register_agent("ai_test", "Agent 2")
        
        agents = auth_manager.list_agents()
        assert len(agents) == 2

    # =========================================================================
    # Session Management Tests
    # =========================================================================

    def test_create_session(self, auth_manager: AgentAuthManager):
        """create_session should create new session."""
        agent = auth_manager.register_agent(
            agent_type="ai_test",
            display_name="Test Agent",
            allowed_role_modes=["executor"],
        )
        
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test_harness",
        )
        
        assert session.session_token is not None
        assert session.session_token.startswith("sess-")
        assert session.agent_id == agent.agent_id
        assert session.role_mode == RoleMode.EXECUTOR
        assert session.is_active is True

    def test_create_session_agent_not_found(self, auth_manager: AgentAuthManager):
        """create_session should raise for unknown agent."""
        with pytest.raises(ValueError, match="Agent not found"):
            auth_manager.create_session(
                agent_id="nonexistent",
                role_mode="executor",
                authorized_by="test",
            )

    def test_create_session_role_not_allowed(self, auth_manager: AgentAuthManager):
        """create_session should raise for disallowed role mode."""
        agent = auth_manager.register_agent(
            agent_type="ai_test",
            display_name="Test Agent",
            allowed_role_modes=["executor"],  # Only executor
        )
        
        with pytest.raises(ValueError, match="not allowed"):
            auth_manager.create_session(
                agent_id=agent.agent_id,
                role_mode="architect",  # Not allowed
                authorized_by="test",
            )

    def test_create_session_duplicate_active(self, auth_manager: AgentAuthManager):
        """create_session should prevent duplicate active sessions."""
        agent = auth_manager.register_agent(
            agent_type="ai_test",
            display_name="Test Agent",
            allowed_role_modes=["executor"],
        )
        
        auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        with pytest.raises(ValueError, match="already has an active session"):
            auth_manager.create_session(
                agent_id=agent.agent_id,
                role_mode="executor",
                authorized_by="test",
            )

    def test_validate_session(self, auth_manager: AgentAuthManager):
        """validate_session should return True for active session."""
        agent = auth_manager.register_agent("ai_test", "Test Agent")
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        assert auth_manager.validate_session(session.session_token) is True

    def test_validate_session_not_found(self, auth_manager: AgentAuthManager):
        """validate_session should return False for unknown token."""
        assert auth_manager.validate_session("invalid-token") is False

    def test_terminate_session(self, auth_manager: AgentAuthManager):
        """terminate_session should change session state."""
        agent = auth_manager.register_agent("ai_test", "Test Agent")
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        result = auth_manager.terminate_session(session.session_token, "test_complete")
        assert result is True
        
        # Session should no longer be active
        assert auth_manager.validate_session(session.session_token) is False
        
        # Session state should be TERMINATED
        updated = auth_manager.get_session(session.session_token)
        assert updated is not None
        assert updated.state == SessionState.TERMINATED

    def test_suspend_and_resume_session(self, auth_manager: AgentAuthManager):
        """suspend/resume should change session state correctly."""
        agent = auth_manager.register_agent("ai_test", "Test Agent")
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        # Suspend
        auth_manager.suspend_session(session.session_token)
        updated = auth_manager.get_session(session.session_token)
        assert updated is not None
        assert updated.state == SessionState.SUSPENDED
        assert updated.is_active is False
        
        # Resume
        auth_manager.resume_session(session.session_token)
        updated = auth_manager.get_session(session.session_token)
        assert updated is not None
        assert updated.state == SessionState.ACTIVE

    def test_get_active_sessions(self, auth_manager: AgentAuthManager):
        """get_active_sessions should return only active sessions."""
        agent1 = auth_manager.register_agent("ai_test", "Agent 1")
        agent2 = auth_manager.register_agent("ai_test", "Agent 2")
        
        session1 = auth_manager.create_session(
            agent_id=agent1.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        session2 = auth_manager.create_session(
            agent_id=agent2.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        # Terminate session1
        auth_manager.terminate_session(session1.session_token)
        
        active = auth_manager.get_active_sessions()
        assert len(active) == 1
        assert active[0].session_token == session2.session_token

    # =========================================================================
    # Role Mode Management Tests
    # =========================================================================

    def test_switch_role_mode(self, auth_manager: AgentAuthManager):
        """switch_role_mode should allow de-escalation."""
        agent = auth_manager.register_agent(
            agent_type="ai_test",
            display_name="Test Agent",
            allowed_role_modes=["executor", "builder"],
        )
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="builder",  # Start with higher role
            authorized_by="test",
        )
        
        # De-escalate to executor
        result = auth_manager.switch_role_mode(
            session.session_token,
            "executor",
            authorized_by="test",
        )
        assert result is True
        
        updated = auth_manager.get_session(session.session_token)
        assert updated is not None
        assert updated.role_mode == RoleMode.EXECUTOR

    def test_switch_role_mode_escalation_blocked(self, auth_manager: AgentAuthManager):
        """switch_role_mode should block escalation."""
        agent = auth_manager.register_agent(
            agent_type="ai_test",
            display_name="Test Agent",
            allowed_role_modes=["executor", "architect"],
        )
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        # Attempt escalation to architect
        with pytest.raises(ValueError, match="Escalation"):
            auth_manager.switch_role_mode(
                session.session_token,
                "architect",
                authorized_by="test",
            )

    # =========================================================================
    # Concurrency Control Tests (Artifact Locking)
    # =========================================================================

    def test_lock_artifact(self, auth_manager: AgentAuthManager):
        """lock_artifact should acquire lock and return success dict."""
        agent = auth_manager.register_agent("ai_test", "Test Agent")
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        result = auth_manager.lock_artifact(session.session_token, "tasks/TASK_001.md")
        assert result["success"] is True
        assert result["error"] is None
        assert result["session"] is not None
        
        # Verify lock is recorded
        updated = auth_manager.get_session(session.session_token)
        assert updated is not None
        assert "tasks/TASK_001.md" in updated.locked_artifacts

    def test_lock_artifact_conflict(self, auth_manager: AgentAuthManager):
        """lock_artifact should fail if already locked by another session."""
        agent1 = auth_manager.register_agent("ai_test", "Agent 1")
        agent2 = auth_manager.register_agent("ai_test", "Agent 2")
        
        session1 = auth_manager.create_session(
            agent_id=agent1.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        session2 = auth_manager.create_session(
            agent_id=agent2.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        # Session 1 locks artifact
        result1 = auth_manager.lock_artifact(session1.session_token, "tasks/TASK_001.md")
        assert result1["success"] is True
        
        # Session 2 cannot lock same artifact
        result2 = auth_manager.lock_artifact(session2.session_token, "tasks/TASK_001.md")
        assert result2["success"] is False
        assert "locked by session" in result2["error"]

    def test_unlock_artifact(self, auth_manager: AgentAuthManager):
        """unlock_artifact should release lock."""
        agent = auth_manager.register_agent("ai_test", "Test Agent")
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        auth_manager.lock_artifact(session.session_token, "tasks/TASK_001.md")
        result = auth_manager.unlock_artifact(session.session_token, "tasks/TASK_001.md")
        
        assert result["success"] is True
        assert result["error"] is None
        
        updated = auth_manager.get_session(session.session_token)
        assert updated is not None
        assert "tasks/TASK_001.md" not in updated.locked_artifacts

    def test_get_artifact_lock_holder(self, auth_manager: AgentAuthManager):
        """get_artifact_lock_holder should return locking session."""
        agent = auth_manager.register_agent("ai_test", "Test Agent")
        session = auth_manager.create_session(
            agent_id=agent.agent_id,
            role_mode="executor",
            authorized_by="test",
        )
        
        auth_manager.lock_artifact(session.session_token, "tasks/TASK_001.md")
        
        holder = auth_manager.get_artifact_lock_holder("tasks/TASK_001.md")
        assert holder is not None
        assert holder.session_token == session.session_token

    def test_get_artifact_lock_holder_none(self, auth_manager: AgentAuthManager):
        """get_artifact_lock_holder should return None if not locked."""
        holder = auth_manager.get_artifact_lock_holder("tasks/TASK_UNLOCKED.md")
        assert holder is None

    # =========================================================================
    # Persistence Tests
    # =========================================================================

    def test_state_persistence(self, temp_state_dir: Path):
        """State should persist across manager instances."""
        # Create manager and register agent
        manager1 = AgentAuthManager(state_dir=temp_state_dir)
        agent = manager1.register_agent(
            agent_type="ai_test",
            display_name="Persistent Agent",
            allowed_role_modes=["executor"],
        )
        
        # Create new manager instance (simulates restart)
        manager2 = AgentAuthManager(state_dir=temp_state_dir)
        
        # Agent should still exist
        retrieved = manager2.get_agent(agent.agent_id)
        assert retrieved is not None
        assert retrieved.display_name == "Persistent Agent"
