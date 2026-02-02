"""
Agent Authentication & Session Management Module

Implements agent authentication, session management, and Role Mode assignment
as defined in:
- ROLE_MODE_CANON
- AUTHORITY_CANON  
- MULTI_AGENT_CANON

This module provides:
- Agent identity registration
- Session lifecycle management
- Role Mode assignment and switching
- Concurrency control via session tokens
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class SessionState(Enum):
    """Session lifecycle states as per MULTI_AGENT_CANON."""
    ANONYMOUS = "anonymous"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class RoleMode(Enum):
    """Canonical Role Modes as per ROLE_MODE_CANON and PAIR_PROGRAMMING_STANDARD."""
    ARCHITECT = "architect"
    PLANNER = "planner"
    EXECUTOR = "executor"
    BUILDER = "builder"
    REVIEWER = "reviewer"  # NEW: Pair Programming code reviewer


@dataclass
class AgentIdentity:
    """Registered agent identity."""
    agent_id: str
    agent_type: str  # "human", "ai_claude", "ai_gpt", "ai_other", etc.
    display_name: str
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    allowed_role_modes: Set[RoleMode] = field(default_factory=lambda: {RoleMode.EXECUTOR})
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "display_name": self.display_name,
            "registered_at": self.registered_at.isoformat(),
            "allowed_role_modes": [rm.value for rm in self.allowed_role_modes],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentIdentity":
        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            display_name=data["display_name"],
            registered_at=datetime.fromisoformat(data["registered_at"]),
            allowed_role_modes={RoleMode(rm) for rm in data.get("allowed_role_modes", ["executor"])},
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentSession:
    """Active agent session with Role Mode binding."""
    session_token: str
    agent_id: str
    role_mode: RoleMode
    state: SessionState
    started_at: datetime
    authorized_by: str  # Who authorized this session
    expires_at: Optional[datetime] = None
    task_scope: List[str] = field(default_factory=list)  # Bound task IDs
    parent_session: Optional[str] = None  # For delegation tracking
    events: List[Dict[str, Any]] = field(default_factory=list)
    pending_artifacts: Set[str] = field(default_factory=set)  # For concurrency tracking
    locked_artifacts: Set[str] = field(default_factory=set)  # Artifact locking
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_token": self.session_token,
            "agent_id": self.agent_id,
            "role_mode": self.role_mode.value,
            "state": self.state.value,
            "started_at": self.started_at.isoformat(),
            "authorized_by": self.authorized_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "task_scope": self.task_scope,
            "parent_session": self.parent_session,
            "events": self.events,
            "pending_artifacts": list(self.pending_artifacts),
            "locked_artifacts": list(self.locked_artifacts),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSession":
        return cls(
            session_token=data["session_token"],
            agent_id=data["agent_id"],
            role_mode=RoleMode(data["role_mode"]),
            state=SessionState(data["state"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            authorized_by=data["authorized_by"],
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            task_scope=data.get("task_scope", []),
            parent_session=data.get("parent_session"),
            events=data.get("events", []),
            pending_artifacts=set(data.get("pending_artifacts", [])),
            locked_artifacts=set(data.get("locked_artifacts", [])),
        )
    
    @property
    def is_active(self) -> bool:
        """Check if session is active and not expired."""
        if self.state != SessionState.ACTIVE:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True
    
    def add_event(self, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to session audit trail."""
        self.events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details or {},
        })


class AgentAuthManager:
    """
    Manages agent authentication and session lifecycle.
    
    Responsibilities:
    - Agent registration
    - Session creation and authorization
    - Role Mode assignment and validation
    - Session termination
    - Concurrency control
    """
    
    def __init__(self, state_dir: Optional[Path] = None):
        """Initialize the auth manager with state directory."""
        self.state_dir = state_dir or Path(__file__).parent.parent / "state"
        self.agents_file = self.state_dir / "agents.yaml"
        self.sessions_file = self.state_dir / "sessions.yaml"
        
        self._agents: Dict[str, AgentIdentity] = {}
        self._sessions: Dict[str, AgentSession] = {}
        self._load_state()
    
    def _load_state(self) -> None:
        """Load agents and sessions from disk."""
        if self.agents_file.exists():
            with self.agents_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for agent_id, agent_data in data.get("agents", {}).items():
                    self._agents[agent_id] = AgentIdentity.from_dict(agent_data)
        
        if self.sessions_file.exists():
            with self.sessions_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                for token, session_data in data.get("sessions", {}).items():
                    self._sessions[token] = AgentSession.from_dict(session_data)
    
    def _save_state(self) -> None:
        """Persist agents and sessions to disk."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        agents_data = {
            "version": "0.1",
            "agents": {aid: agent.to_dict() for aid, agent in self._agents.items()},
        }
        with self.agents_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(agents_data, f, sort_keys=False, allow_unicode=True)
        
        sessions_data = {
            "version": "0.1",
            "sessions": {token: sess.to_dict() for token, sess in self._sessions.items()},
        }
        with self.sessions_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(sessions_data, f, sort_keys=False, allow_unicode=True)
    
    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        return f"sess-{secrets.token_hex(16)}"
    
    def _generate_agent_id(self, agent_type: str, display_name: str) -> str:
        """Generate a unique agent ID."""
        hash_input = f"{agent_type}:{display_name}:{datetime.now(timezone.utc).isoformat()}"
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"{agent_type}-{hash_suffix}"
    
    # =========================================================================
    # Agent Registration
    # =========================================================================
    
    def register_agent(
        self,
        agent_type: str,
        display_name: str,
        allowed_role_modes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentIdentity:
        """
        Register a new agent identity.
        
        Args:
            agent_type: Type of agent ("human", "ai_claude", "ai_gpt", etc.)
            display_name: Human-readable name
            allowed_role_modes: List of Role Mode names this agent can use
            metadata: Additional agent metadata
        
        Returns:
            Newly created AgentIdentity
        """
        agent_id = self._generate_agent_id(agent_type, display_name)
        
        role_modes = {RoleMode.EXECUTOR}  # Default
        if allowed_role_modes:
            role_modes = {RoleMode(rm) for rm in allowed_role_modes}
        
        agent = AgentIdentity(
            agent_id=agent_id,
            agent_type=agent_type,
            display_name=display_name,
            allowed_role_modes=role_modes,
            metadata=metadata or {},
        )
        
        self._agents[agent_id] = agent
        self._save_state()
        
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[AgentIdentity]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[AgentIdentity]:
        """List all registered agents."""
        return list(self._agents.values())
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def create_session(
        self,
        agent_id: str,
        role_mode: str,
        authorized_by: str,
        timeout_minutes: int = 480,
        task_scope: Optional[List[str]] = None,
    ) -> AgentSession:
        """
        Create a new authorized session for an agent.
        
        Args:
            agent_id: ID of the registered agent
            role_mode: Requested Role Mode
            authorized_by: Who is authorizing this session
            timeout_minutes: Session timeout (default 8 hours)
            task_scope: Optional list of task IDs this session is bound to
        
        Returns:
            Newly created AgentSession
        
        Raises:
            ValueError: If agent not found or role mode not allowed
        """
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        requested_mode = RoleMode(role_mode)
        if requested_mode not in agent.allowed_role_modes:
            raise ValueError(
                f"Role Mode '{role_mode}' not allowed for agent {agent_id}. "
                f"Allowed: {[rm.value for rm in agent.allowed_role_modes]}"
            )
        
        # Check for existing active sessions (max 1 per agent)
        for sess in self._sessions.values():
            if sess.agent_id == agent_id and sess.is_active:
                raise ValueError(
                    f"Agent {agent_id} already has an active session: {sess.session_token}"
                )
        
        session = AgentSession(
            session_token=self._generate_session_token(),
            agent_id=agent_id,
            role_mode=requested_mode,
            state=SessionState.ACTIVE,
            started_at=datetime.now(timezone.utc),
            authorized_by=authorized_by,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes),
            task_scope=task_scope or [],
        )
        
        session.add_event("session_created", {
            "role_mode": role_mode,
            "authorized_by": authorized_by,
        })
        
        self._sessions[session.session_token] = session
        self._save_state()
        
        return session
    
    def get_session(self, session_token: str) -> Optional[AgentSession]:
        """Get session by token."""
        return self._sessions.get(session_token)
    
    def validate_session(self, session_token: str) -> bool:
        """Check if session is valid and active."""
        session = self._sessions.get(session_token)
        return session is not None and session.is_active
    
    def get_active_sessions(self) -> List[AgentSession]:
        """Get all currently active sessions."""
        return [s for s in self._sessions.values() if s.is_active]
    
    def terminate_session(
        self,
        session_token: str,
        reason: str = "explicit_termination",
    ) -> bool:
        """
        Terminate a session.
        
        Args:
            session_token: Token of session to terminate
            reason: Reason for termination
        
        Returns:
            True if session was terminated, False if not found
        """
        session = self._sessions.get(session_token)
        if not session:
            return False
        
        session.state = SessionState.TERMINATED
        session.add_event("session_terminated", {"reason": reason})
        
        self._save_state()
        return True
    
    def suspend_session(self, session_token: str, reason: str = "suspended") -> bool:
        """Temporarily suspend a session."""
        session = self._sessions.get(session_token)
        if not session or session.state != SessionState.ACTIVE:
            return False
        
        session.state = SessionState.SUSPENDED
        session.add_event("session_suspended", {"reason": reason})
        
        self._save_state()
        return True
    
    def resume_session(self, session_token: str) -> bool:
        """Resume a suspended session."""
        session = self._sessions.get(session_token)
        if not session or session.state != SessionState.SUSPENDED:
            return False
        
        # Check if expired
        if session.expires_at and datetime.now(timezone.utc) > session.expires_at:
            session.state = SessionState.TERMINATED
            session.add_event("session_expired", {})
            self._save_state()
            return False
        
        session.state = SessionState.ACTIVE
        session.add_event("session_resumed", {})
        
        self._save_state()
        return True
    
    # =========================================================================
    # Role Mode Management
    # =========================================================================
    
    def switch_role_mode(
        self,
        session_token: str,
        new_role_mode: str,
        authorized_by: str,
    ) -> bool:
        """
        Switch the Role Mode of an active session.
        
        This requires explicit authorization and is subject to escalation rules.
        
        Args:
            session_token: Session to modify
            new_role_mode: Target Role Mode
            authorized_by: Who is authorizing the switch
        
        Returns:
            True if switch was successful
        
        Raises:
            ValueError: If switch violates escalation rules
        """
        session = self._sessions.get(session_token)
        if not session or not session.is_active:
            raise ValueError(f"Session not active: {session_token}")
        
        agent = self._agents.get(session.agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {session.agent_id}")
        
        new_mode = RoleMode(new_role_mode)
        
        # Check if role mode is allowed for this agent
        if new_mode not in agent.allowed_role_modes:
            raise ValueError(
                f"Role Mode '{new_role_mode}' not allowed for agent {agent.agent_id}"
            )
        
        # Check escalation rules
        current_mode = session.role_mode
        if self._is_escalation(current_mode, new_mode):
            raise ValueError(
                f"Escalation from '{current_mode.value}' to '{new_mode.value}' is prohibited. "
                "Escalation requires Project Owner authorization and a new session."
            )
        
        old_mode = session.role_mode
        session.role_mode = new_mode
        session.add_event("role_mode_switched", {
            "from": old_mode.value,
            "to": new_mode.value,
            "authorized_by": authorized_by,
        })
        
        self._save_state()
        return True
    
    def _is_escalation(self, from_mode: RoleMode, to_mode: RoleMode) -> bool:
        """Check if a role mode change constitutes escalation."""
        # Define authority levels (lower = less authority)
        authority_levels = {
            RoleMode.EXECUTOR: 1,
            RoleMode.BUILDER: 2,
            RoleMode.PLANNER: 3,
            RoleMode.ARCHITECT: 4,
        }
        return authority_levels.get(to_mode, 0) > authority_levels.get(from_mode, 0)
    
    # =========================================================================
    # Concurrency Control
    # =========================================================================
    
    def lock_artifact(
        self,
        session_token: str,
        artifact_path: str,
        timeout_seconds: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Acquire lock on artifact for session.
        
        Args:
            session_token: Session requesting the lock
            artifact_path: Path to artifact
            timeout_seconds: Lock timeout (currently unused, for future extension)
        
        Returns:
            {"success": bool, "session": AgentSession | None, "error": str | None}
        """
        session = self._sessions.get(session_token)
        if not session or not session.is_active:
            return {"success": False, "session": None, "error": "Invalid or inactive session"}
        
        # Check if artifact already locked by another session
        for other_token, other_session in self._sessions.items():
            if other_token == session_token:
                continue
            if other_session.is_active and artifact_path in other_session.locked_artifacts:
                return {
                    "success": False,
                    "session": None,
                    "error": f"Artifact locked by session {other_token[:8]}",
                }
        
        # Acquire lock
        session.locked_artifacts.add(artifact_path)
        session.add_event("artifact_locked", {"artifact": artifact_path})
        
        self._save_state()
        return {"success": True, "session": session, "error": None}
    
    def unlock_artifact(
        self,
        session_token: str,
        artifact_path: str,
    ) -> Dict[str, Any]:
        """
        Release lock on artifact.
        
        Args:
            session_token: Session releasing the lock
            artifact_path: Path to artifact
        
        Returns:
            {"success": bool, "session": AgentSession | None, "error": str | None}
        """
        session = self._sessions.get(session_token)
        if not session:
            return {"success": False, "session": None, "error": "Session not found"}
        
        if artifact_path not in session.locked_artifacts:
            return {"success": False, "session": None, "error": "Artifact not locked by this session"}
        
        # Release lock
        session.locked_artifacts.remove(artifact_path)
        session.add_event("artifact_unlocked", {"artifact": artifact_path})
        
        self._save_state()
        return {"success": True, "session": session, "error": None}
    
    def get_artifact_lock_holder(self, artifact_path: str) -> Optional[AgentSession]:
        """Get session that holds lock on artifact."""
        for session in self._sessions.values():
            if artifact_path in session.locked_artifacts:
                return session
        return None


# Global instance for convenience
_auth_manager: Optional[AgentAuthManager] = None


def get_auth_manager() -> AgentAuthManager:
    """Get the global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AgentAuthManager()
    return _auth_manager


def reset_auth_manager() -> None:
    """Reset the global auth manager (for testing)."""
    global _auth_manager
    _auth_manager = None


if __name__ == "__main__":
    # Example usage
    manager = AgentAuthManager()
    
    # Register an AI agent
    agent = manager.register_agent(
        agent_type="ai_claude",
        display_name="Claude Research Agent",
        allowed_role_modes=["executor", "builder"],
    )
    print(f"Registered agent: {agent.agent_id}")
    
    # Create a session
    session = manager.create_session(
        agent_id=agent.agent_id,
        role_mode="executor",
        authorized_by="project_owner",
    )
    print(f"Created session: {session.session_token}")
    print(f"Session active: {session.is_active}")
    
    # Lock an artifact
    result = manager.lock_artifact(session.session_token, "tasks/TASK_001.md")
    print(f"Artifact lock result: {result}")
    
    # Terminate session
    manager.terminate_session(session.session_token, "completed")
    print(f"Session terminated")
