"""
MCP Server for AI Workflow OS

Exposes kernel capabilities as MCP (Model Context Protocol) tools
for AI agents to invoke.

This server provides:
- Task lifecycle management (new, start, finish)
- Session management (create, validate, terminate)
- Governance gate verification
- Artifact operations (read, propose)

All operations are subject to:
- Role Mode permissions
- Governance gate checks
- Session-based access control
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add repo root to path for package imports
KERNEL_DIR = Path(__file__).parent
ROOT_DIR = KERNEL_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kernel.agent_auth import AgentAuthManager, RoleMode, SessionState, get_auth_manager
from kernel.governance_gate import GovernanceGate, verify_governance, has_violations, get_all_violations


class MCPServer:
    """
    MCP Server exposing AI Workflow OS capabilities as tools.
    
    Tool Categories:
    1. Session Management - Authentication and authorization
    2. Task Operations - Task lifecycle management
    3. Governance - Verification and compliance
    4. Artifacts - Read and propose changes
    """
    
    def __init__(self):
        """Initialize MCP Server with dependencies."""
        self.auth_manager = get_auth_manager()
        self.governance_gate = GovernanceGate()
        self.root = KERNEL_DIR.parent
    
    # =========================================================================
    # Tool Definitions (MCP Schema)
    # =========================================================================
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions."""
        return [
            # Agent Registration
            {
                "name": "agent_register",
                "description": "Register a new AI agent. Returns agent_id for use in session creation. Must be called before session_create.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_type": {
                            "type": "string",
                            "description": "Type of agent (e.g., 'ai_claude', 'ai_gpt')"
                        },
                        "display_name": {
                            "type": "string",
                            "description": "Human-readable name for the agent"
                        },
                        "allowed_role_modes": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["architect", "planner", "executor", "builder", "reviewer"]
                            },
                            "description": "Role modes this agent is allowed to use"
                        }
                    },
                    "required": ["agent_type", "display_name", "allowed_role_modes"]
                }
            },
            # Session Management
            {
                "name": "session_create",
                "description": "Create a new authorized session for an agent. Required before any other operations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Registered agent ID"
                        },
                        "role_mode": {
                            "type": "string",
                            "enum": ["architect", "planner", "executor", "builder", "reviewer"],
                            "description": "Requested Role Mode"
                        },
                        "authorized_by": {
                            "type": "string",
                            "description": "Who is authorizing this session"
                        },
                        "task_scope": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: Task IDs this session is scoped to"
                        }
                    },
                    "required": ["agent_id", "role_mode", "authorized_by"]
                }
            },
            {
                "name": "session_validate",
                "description": "Validate that a session is active and get its current state.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Session token to validate"
                        }
                    },
                    "required": ["session_token"]
                }
            },
            {
                "name": "session_terminate",
                "description": "Terminate a session. Required when work is complete.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Session token to terminate"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for termination"
                        }
                    },
                    "required": ["session_token"]
                }
            },
            # Task Operations
            {
                "name": "task_list",
                "description": "List all tasks with their current status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "status_filter": {
                            "type": "string",
                            "enum": ["draft", "ready", "running", "reviewing", "merged", "released", "blocked", "abandoned"],
                            "description": "Optional: Filter by status"
                        }
                    },
                    "required": ["session_token"]
                }
            },
            {
                "name": "task_get",
                "description": "Get details of a specific task including its TaskCard content.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to retrieve"
                        }
                    },
                    "required": ["session_token", "task_id"]
                }
            },
            {
                "name": "task_start",
                "description": "Start working on a task. Changes status to 'running'.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to start"
                        }
                    },
                    "required": ["session_token", "task_id"]
                }
            },
            {
                "name": "task_finish",
                "description": "Mark a task as finished. Changes status to 'reviewing'.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to finish"
                        }
                    },
                    "required": ["session_token", "task_id"]
                }
            },
            # Governance
            {
                "name": "governance_check",
                "description": "Run governance gate verification on proposed changes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "output_text": {
                            "type": "string",
                            "description": "Optional: Text output to check for authority claims"
                        },
                        "artifact_changes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "action": {"type": "string", "enum": ["create", "modify", "delete"]}
                                }
                            },
                            "description": "Optional: Proposed artifact changes"
                        }
                    },
                    "required": ["session_token"]
                }
            },
            # Artifacts
            {
                "name": "artifact_read",
                "description": "Read the content of an artifact (file).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path to artifact"
                        }
                    },
                    "required": ["session_token", "path"]
                }
            },
            {
                "name": "artifact_list",
                "description": "List artifacts in a directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative directory path"
                        }
                    },
                    "required": ["session_token", "path"]
                }
            },
            {
                "name": "spec_list",
                "description": "List all registered specs from spec_registry.yaml.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "scope_filter": {
                            "type": "string",
                            "enum": ["canon", "framework", "project"],
                            "description": "Optional: Filter by scope"
                        }
                    },
                    "required": ["session_token"]
                }
            },
            # =====================================================================
            # Spec Evolution Tools (Skills + MCP + Hooks Integration)
            # =====================================================================
            {
                "name": "spec_read",
                "description": "Read the content of a specification file. Returns full content or specific section.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "spec_path": {
                            "type": "string",
                            "description": "Relative path to spec file (e.g., 'projects/dgsf/specs/SDF_INTERFACE_CONTRACT.yaml')"
                        },
                        "section": {
                            "type": "string",
                            "description": "Optional: Specific YAML section to read (dot notation, e.g., 'validation.thresholds')"
                        }
                    },
                    "required": ["session_token", "spec_path"]
                }
            },
            {
                "name": "spec_propose",
                "description": "Propose a change to a specification. Creates a proposal record pending human approval.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "spec_path": {
                            "type": "string",
                            "description": "Relative path to spec file"
                        },
                        "change_type": {
                            "type": "string",
                            "enum": ["add", "modify", "deprecate"],
                            "description": "Type of change being proposed"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Explanation of why this change is needed (required)"
                        },
                        "proposed_diff": {
                            "type": "string",
                            "description": "Unified diff format showing the proposed changes"
                        },
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: References to research or experiment results"
                        }
                    },
                    "required": ["session_token", "spec_path", "change_type", "rationale", "proposed_diff"]
                }
            },
            {
                "name": "spec_commit",
                "description": "Commit an approved spec change. Requires approval reference. Triggers pre/post hooks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "proposal_id": {
                            "type": "string",
                            "description": "Proposal ID from spec_propose (format: SCP-YYYY-MM-DD-NNN)"
                        },
                        "approval_ref": {
                            "type": "string",
                            "description": "Approval reference (decision log path or PR number)"
                        },
                        "run_hooks": {
                            "type": "boolean",
                            "description": "Whether to run pre/post spec-change hooks (default: true)"
                        }
                    },
                    "required": ["session_token", "proposal_id", "approval_ref"]
                }
            },
            {
                "name": "spec_triage",
                "description": "Analyze a problem to determine if spec change is needed. Returns classification and recommended action.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "problem_description": {
                            "type": "string",
                            "description": "Description of the problem (error message, metric deviation, etc.)"
                        },
                        "source": {
                            "type": "string",
                            "enum": ["test", "experiment", "review", "monitoring", "manual"],
                            "description": "Where the problem was discovered"
                        },
                        "context": {
                            "type": "object",
                            "description": "Optional: Additional context (file paths, metric values, etc.)"
                        }
                    },
                    "required": ["session_token", "problem_description", "source"]
                }
            },
            # =====================================================================
            # Pair Programming / Code Review Tools
            # =====================================================================
            {
                "name": "review_submit",
                "description": "Submit code artifacts for Pair Programming review. Transitions task to 'code_review' state.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token (builder agent)"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID being submitted for review"
                        },
                        "artifact_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Relative paths to artifacts to review"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional notes for the reviewer"
                        }
                    },
                    "required": ["session_token", "task_id", "artifact_paths"]
                }
            },
            {
                "name": "review_create_session",
                "description": "Create a review session to begin Pair Programming review. Must be called by reviewer agent.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token (reviewer agent)"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to review"
                        },
                        "personas": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["security_expert", "performance_expert", "architecture_expert", "domain_expert", "testing_expert"]
                            },
                            "description": "Optional: Expert personas to apply"
                        }
                    },
                    "required": ["session_token", "task_id"]
                }
            },
            {
                "name": "review_conduct",
                "description": "Conduct a code review on submitted artifacts. Generates a comprehensive review report.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token (reviewer agent)"
                        },
                        "review_session_id": {
                            "type": "string",
                            "description": "Review session ID from review_create_session"
                        },
                        "quality_issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "check_id": {"type": "string"},
                                    "severity": {"type": "string", "enum": ["CRITICAL", "MAJOR", "MINOR", "SUGGESTION"]},
                                    "description": {"type": "string"},
                                    "file_path": {"type": "string"},
                                    "line_start": {"type": "integer"},
                                    "suggested_fix": {"type": "string"}
                                },
                                "required": ["check_id", "severity", "description", "file_path"]
                            },
                            "description": "Quality check issues found"
                        },
                        "requirements_issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "check_id": {"type": "string"},
                                    "severity": {"type": "string", "enum": ["CRITICAL", "MAJOR", "MINOR"]},
                                    "description": {"type": "string"},
                                    "file_path": {"type": "string"},
                                    "requirement_ref": {"type": "string"}
                                },
                                "required": ["check_id", "severity", "description", "file_path"]
                            },
                            "description": "Requirements check issues found"
                        },
                        "completeness_issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "check_id": {"type": "string"},
                                    "severity": {"type": "string", "enum": ["CRITICAL", "MAJOR", "MINOR"]},
                                    "description": {"type": "string"},
                                    "missing_item": {"type": "string"}
                                },
                                "required": ["check_id", "severity", "description"]
                            },
                            "description": "Completeness check issues found"
                        },
                        "optimization_suggestions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "check_id": {"type": "string"},
                                    "description": {"type": "string"},
                                    "file_path": {"type": "string"},
                                    "current_code": {"type": "string"},
                                    "suggested_code": {"type": "string"},
                                    "rationale": {"type": "string"},
                                    "impact": {"type": "string", "enum": ["minor", "moderate", "significant"]}
                                },
                                "required": ["check_id", "description"]
                            },
                            "description": "Optimization suggestions"
                        },
                        "requirements_coverage_pct": {
                            "type": "number",
                            "description": "Percentage of requirements covered (0-100)"
                        },
                        "completeness_pct": {
                            "type": "number",
                            "description": "Percentage of task completeness (0-100)"
                        }
                    },
                    "required": ["session_token", "review_session_id"]
                }
            },
            {
                "name": "review_get_status",
                "description": "Get the current status of a code review for a task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to check"
                        }
                    },
                    "required": ["session_token", "task_id"]
                }
            },
            {
                "name": "review_get_report",
                "description": "Get the latest review report for a task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID"
                        },
                        "report_id": {
                            "type": "string",
                            "description": "Optional: Specific report ID. If not provided, returns latest."
                        },
                        "format": {
                            "type": "string",
                            "enum": ["yaml", "markdown", "json"],
                            "description": "Output format (default: json)"
                        }
                    },
                    "required": ["session_token", "task_id"]
                }
            },
            {
                "name": "review_respond",
                "description": "Respond to a code review (builder accepts feedback or submits revision).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token (builder agent)"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID"
                        },
                        "action": {
                            "type": "string",
                            "enum": ["acknowledge", "revision_submitted", "request_clarification"],
                            "description": "Response action"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional response notes"
                        }
                    },
                    "required": ["session_token", "task_id", "action"]
                }
            },
            {
                "name": "review_approve",
                "description": "Approve a code review and advance task to 'reviewing' state. Only for reviewer agent.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token (reviewer agent)"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to approve"
                        },
                        "review_session_id": {
                            "type": "string",
                            "description": "Review session ID"
                        },
                        "final_notes": {
                            "type": "string",
                            "description": "Optional final review notes"
                        }
                    },
                    "required": ["session_token", "task_id", "review_session_id"]
                }
            },
            {
                "name": "review_get_prompts",
                "description": "Get review prompts for conducting a Pair Programming review.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to review"
                        },
                        "dimension": {
                            "type": "string",
                            "enum": ["quality", "requirements", "completeness", "optimization", "all"],
                            "description": "Which review dimension prompt to get"
                        },
                        "persona": {
                            "type": "string",
                            "enum": ["security_expert", "performance_expert", "architecture_expert", "domain_expert", "testing_expert"],
                            "description": "Optional: Expert persona prompt"
                        }
                    },
                    "required": ["session_token", "task_id", "dimension"]
                }
            },
            # Artifact Locking
            {
                "name": "agent_lock_artifact",
                "description": "Acquire exclusive lock on an artifact to prevent concurrent modifications.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "artifact_path": {
                            "type": "string",
                            "description": "Path to artifact to lock"
                        }
                    },
                    "required": ["session_token", "artifact_path"]
                }
            },
            {
                "name": "agent_unlock_artifact",
                "description": "Release lock on an artifact.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_token": {
                            "type": "string",
                            "description": "Active session token"
                        },
                        "artifact_path": {
                            "type": "string",
                            "description": "Path to artifact to unlock"
                        }
                    },
                    "required": ["session_token", "artifact_path"]
                }
            },
        ]
    
    # =========================================================================
    # Tool Implementations
    # =========================================================================
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call and return the result.
        
        Args:
            name: Tool name
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        """
        tool_map = {
            "agent_register": self._agent_register,
            "session_create": self._session_create,
            "session_validate": self._session_validate,
            "session_terminate": self._session_terminate,
            "task_list": self._task_list,
            "task_get": self._task_get,
            "task_start": self._task_start,
            "task_finish": self._task_finish,
            "governance_check": self._governance_check,
            "artifact_read": self._artifact_read,
            "artifact_list": self._artifact_list,
            "spec_list": self._spec_list,
            "spec_read": self._spec_read,
            "spec_propose": self._spec_propose,
            "spec_commit": self._spec_commit,
            "spec_triage": self._spec_triage,
            # Pair Programming / Code Review
            "review_submit": self._review_submit,
            "review_create_session": self._review_create_session,
            "review_conduct": self._review_conduct,
            "review_get_status": self._review_get_status,
            "review_get_report": self._review_get_report,
            "review_respond": self._review_respond,
            "review_approve": self._review_approve,
            "review_get_prompts": self._review_get_prompts,
            # Artifact Locking
            "agent_lock_artifact": self._agent_lock_artifact,
            "agent_unlock_artifact": self._agent_unlock_artifact,
        }
        
        if name not in tool_map:
            return {"error": f"Unknown tool: {name}"}
        
        try:
            return tool_map[name](arguments)
        except Exception as e:
            return {"error": str(e), "error_type": type(e).__name__}
    
    def _agent_register(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new AI agent."""
        agent_type = args.get("agent_type")
        display_name = args.get("display_name")
        allowed_role_modes = args.get("allowed_role_modes", [])
        
        if not agent_type or not display_name:
            return {"error": "INVALID_PARAMS", "message": "agent_type and display_name are required"}
        
        if not allowed_role_modes:
            return {"error": "INVALID_PARAMS", "message": "allowed_role_modes must not be empty"}
        
        try:
            agent = self.auth_manager.register_agent(
                agent_type=agent_type,
                display_name=display_name,
                allowed_role_modes=allowed_role_modes,
            )
            
            return {
                "success": True,
                "agent": {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "display_name": agent.display_name,
                    "allowed_role_modes": [rm.value for rm in agent.allowed_role_modes],
                    "registered_at": agent.registered_at.isoformat(),
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session and return error dict if invalid."""
        session = self.auth_manager.get_session(session_token)
        if not session:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        if not session.is_active:
            return {"error": "SESSION_INACTIVE", "message": f"Session is {session.state.value}"}
        return None
    
    # Session Management
    
    def _session_create(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new session."""
        try:
            session = self.auth_manager.create_session(
                agent_id=args["agent_id"],
                role_mode=args["role_mode"],
                authorized_by=args["authorized_by"],
                task_scope=args.get("task_scope"),
            )
            return {
                "success": True,
                "session": {
                    "session_token": session.session_token,
                    "agent_id": session.agent_id,
                    "role_mode": session.role_mode.value,
                    "state": session.state.value,
                    "started_at": session.started_at.isoformat(),
                    "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                }
            }
        except ValueError as e:
            return {"success": False, "error": str(e)}
    
    def _session_validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a session."""
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"valid": False, "error": "SESSION_NOT_FOUND"}
        
        return {
            "valid": session.is_active,
            "session": {
                "session_token": session.session_token,
                "agent_id": session.agent_id,
                "role_mode": session.role_mode.value,
                "state": session.state.value,
                "started_at": session.started_at.isoformat(),
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            }
        }
    
    def _session_terminate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Terminate a session."""
        success = self.auth_manager.terminate_session(
            session_token=args["session_token"],
            reason=args.get("reason", "explicit_termination"),
        )
        return {"success": success}
    
    # Task Operations
    
    def _task_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List tasks."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        import yaml
        tasks_file = self.root / "state" / "tasks.yaml"
        if not tasks_file.exists():
            return {"tasks": []}
        
        with tasks_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        tasks = []
        status_filter = args.get("status_filter")
        
        for task_id, task_data in data.get("tasks", {}).items():
            if status_filter and task_data.get("status") != status_filter:
                continue
            tasks.append({
                "task_id": task_id,
                "status": task_data.get("status"),
                "queue": task_data.get("queue"),
                "branch": task_data.get("branch"),
                "last_updated": task_data.get("last_updated"),
            })
        
        return {"tasks": tasks}
    
    def _task_get(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get task details."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        task_id = args["task_id"]
        taskcard_path = self.root / "tasks" / f"{task_id}.md"
        
        if not taskcard_path.exists():
            return {"error": "TASK_NOT_FOUND", "message": f"TaskCard not found: {task_id}"}
        
        with taskcard_path.open("r", encoding="utf-8") as f:
            content = f.read()
        
        # Also get state
        import yaml
        tasks_file = self.root / "state" / "tasks.yaml"
        state = {}
        if tasks_file.exists():
            with tasks_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                state = data.get("tasks", {}).get(task_id, {})
        
        return {
            "task_id": task_id,
            "taskcard_content": content,
            "state": state,
        }
    
    def _task_start(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Start a task."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        
        # Check Role Mode permissions
        if session.role_mode not in (RoleMode.EXECUTOR, RoleMode.BUILDER):
            return {
                "error": "ROLE_MODE_VIOLATION",
                "message": f"Role Mode '{session.role_mode.value}' cannot start tasks"
            }
        
        # Run governance check
        results = verify_governance(
            agent_session={
                "agent_id": session.agent_id,
                "role_mode": session.role_mode.value,
                "session_token": session.session_token,
            },
            state_mutations=[{
                "type": "task_state",
                "task_id": args["task_id"],
                "from": "draft",
                "to": "running",
                "artifact_backed": True,
            }],
        )
        
        if has_violations(results):
            violations = get_all_violations(results)
            return {
                "error": "GOVERNANCE_VIOLATION",
                "violations": [v.to_dict() for v in violations],
            }
        
        # Lock the task
        locked = self.auth_manager.lock_artifact(
            session.session_token,
            f"tasks/{args['task_id']}.md"
        )
        if not locked:
            holder = self.auth_manager.get_artifact_lock_holder(f"tasks/{args['task_id']}.md")
            return {
                "error": "ARTIFACT_LOCKED",
                "message": f"Task is locked by another session: {holder}"
            }
        
        # Actually update task state via os.py functions
        import argparse
        import sys
        sys.path.insert(0, str(KERNEL_DIR))
        
        try:
            # Import os.py functions (avoid naming conflict with stdlib os)
            import importlib.util
            spec = importlib.util.spec_from_file_location("kernel_os", KERNEL_DIR / "os.py")
            if spec is None or spec.loader is None:
                return {
                    "error": "KERNEL_IMPORT_FAILED",
                    "message": "Failed to load kernel/os.py module spec",
                }
            kernel_os = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kernel_os)
            
            # Create args namespace for cmd_task_start
            cmd_args = argparse.Namespace(task_id=args["task_id"])
            kernel_os.cmd_task_start(cmd_args)
            
            return {
                "success": True,
                "task_id": args["task_id"],
                "new_status": "running",
                "message": "Task started successfully via kernel/os.py.",
            }
        except Exception as e:
            # Unlock on failure
            self.auth_manager.unlock_artifact(
                session.session_token,
                f"tasks/{args['task_id']}.md"
            )
            return {
                "error": "TASK_START_FAILED",
                "message": str(e),
            }
    
    def _task_finish(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Finish a task."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        
        # Check Role Mode permissions
        if session.role_mode not in (RoleMode.EXECUTOR, RoleMode.BUILDER):
            return {
                "error": "ROLE_MODE_VIOLATION",
                "message": f"Role Mode '{session.role_mode.value}' cannot finish tasks"
            }
        
        # Unlock the task
        self.auth_manager.unlock_artifact(
            session.session_token,
            f"tasks/{args['task_id']}.md"
        )
        
        return {
            "success": True,
            "task_id": args["task_id"],
            "new_status": "reviewing",
            "message": "Task finished. Use kernel/os.py for full state update.",
        }
    
    # Governance
    
    def _governance_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run governance verification."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        
        results = verify_governance(
            output_text=args.get("output_text"),
            agent_session={
                "agent_id": session.agent_id,
                "role_mode": session.role_mode.value,
                "session_token": session.session_token,
            },
            artifact_changes=args.get("artifact_changes"),
        )
        
        return {
            "passed": not has_violations(results),
            "results": [r.to_dict() for r in results],
        }
    
    # Artifacts
    
    def _artifact_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read an artifact."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        try:
            requested = Path(args["path"])
            artifact_path = (self.root / requested).resolve()
            if self.root not in artifact_path.parents and artifact_path != self.root:
                return {"error": "INVALID_PATH", "message": "Artifact path escapes workspace root"}
        except Exception as e:
            return {"error": "INVALID_PATH", "message": str(e)}
        
        if not artifact_path.exists():
            return {"error": "NOT_FOUND", "message": f"Artifact not found: {args['path']}"}
        
        if not artifact_path.is_file():
            return {"error": "NOT_A_FILE", "message": f"Path is not a file: {args['path']}"}
        
        try:
            with artifact_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return {"path": args["path"], "content": content}
        except Exception as e:
            return {"error": "READ_ERROR", "message": str(e)}
    
    def _artifact_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List artifacts in a directory."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        try:
            requested = Path(args["path"])
            dir_path = (self.root / requested).resolve()
            if self.root not in dir_path.parents and dir_path != self.root:
                return {"error": "INVALID_PATH", "message": "Directory path escapes workspace root"}
        except Exception as e:
            return {"error": "INVALID_PATH", "message": str(e)}
        
        if not dir_path.exists():
            return {"error": "NOT_FOUND", "message": f"Directory not found: {args['path']}"}
        
        if not dir_path.is_dir():
            return {"error": "NOT_A_DIRECTORY", "message": f"Path is not a directory: {args['path']}"}
        
        items = []
        for item in sorted(dir_path.iterdir()):
            try:
                relative_path = item.resolve().relative_to(self.root)
            except Exception:
                continue
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "path": str(relative_path),
            })
        
        return {"path": args["path"], "items": items}
    
    def _spec_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List registered specs."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        import yaml
        registry_path = self.root / "spec_registry.yaml"
        if not registry_path.exists():
            return {"specs": []}
        
        with registry_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        specs = []
        scope_filter = args.get("scope_filter")
        
        for spec in data.get("specs", []):
            if scope_filter and spec.get("scope") != scope_filter:
                continue
            specs.append({
                "spec_id": spec.get("spec_id"),
                "title": spec.get("title"),
                "scope": spec.get("scope"),
                "status": spec.get("status"),
                "version": spec.get("version", {}).get("semver"),
                "path": spec.get("location", {}).get("path"),
            })
        
        return {"specs": specs}
    
    # =========================================================================
    # Spec Evolution Tool Implementations (Skills + MCP + Hooks Integration)
    # =========================================================================
    
    def _spec_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Read the content of a specification file."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        import yaml
        spec_path = self.root / args["spec_path"]
        
        if not spec_path.exists():
            return {"error": "SPEC_NOT_FOUND", "message": f"Spec file not found: {args['spec_path']}"}
        
        # Read file content
        with spec_path.open("r", encoding="utf-8") as f:
            content = f.read()
        
        result = {
            "spec_path": args["spec_path"],
            "exists": True,
            "content": content,
        }
        
        # If specific section requested and it's a YAML file
        section = args.get("section")
        if section and (spec_path.suffix in [".yaml", ".yml"]):
            try:
                data = yaml.safe_load(content)
                # Navigate dot notation (e.g., "validation.thresholds")
                for key in section.split("."):
                    if isinstance(data, dict) and key in data:
                        data = data[key]
                    else:
                        return {"error": "SECTION_NOT_FOUND", "message": f"Section '{section}' not found"}
                result["section"] = section
                result["section_content"] = data
            except yaml.YAMLError as e:
                return {"error": "YAML_PARSE_ERROR", "message": str(e)}
        
        # Determine spec layer from path
        if "specs/canon" in args["spec_path"]:
            result["layer"] = "L0"
            result["editable_by_ai"] = False
        elif "specs/framework" in args["spec_path"]:
            result["layer"] = "L1"
            result["editable_by_ai"] = False  # Propose only
        elif "/specs/" in args["spec_path"]:
            result["layer"] = "L2"
            result["editable_by_ai"] = False  # Propose only
        elif "/experiments/" in args["spec_path"] and "config.yaml" in args["spec_path"]:
            result["layer"] = "L3"
            result["editable_by_ai"] = True
        else:
            result["layer"] = "unknown"
            result["editable_by_ai"] = False
        
        return result
    
    def _spec_propose(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Propose a change to a specification."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        
        import yaml
        from datetime import datetime
        
        spec_path = args["spec_path"]
        change_type = args["change_type"]
        rationale = args["rationale"]
        proposed_diff = args["proposed_diff"]
        evidence_refs = args.get("evidence_refs", [])
        
        # Check Canon protection
        if "specs/canon" in spec_path:
            return {
                "error": "CANON_PROTECTED",
                "message": "Cannot propose changes to Canon specs (L0). Contact Project Owner."
            }
        
        # Validate role can propose
        if session.role_mode not in (RoleMode.ARCHITECT, RoleMode.PLANNER):
            return {
                "error": "ROLE_MODE_VIOLATION",
                "message": f"Role Mode '{session.role_mode.value}' cannot propose spec changes. Requires 'architect' or 'planner'."
            }
        
        # Generate proposal ID
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        
        # Find next sequence number
        proposals_dir = self.root / "projects" / "dgsf" / "decisions"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        
        existing = list(proposals_dir.glob(f"SCP-{date_str}-*.yaml"))
        seq_num = len(existing) + 1
        proposal_id = f"SCP-{date_str}-{seq_num:03d}"
        
        # Create proposal record
        proposal = {
            "id": proposal_id,
            "spec_path": spec_path,
            "change_type": change_type,
            "status": "proposed",
            "proposed_by": {
                "agent_id": session.agent_id,
                "session_token": session.session_token[:8] + "...",
                "role_mode": session.role_mode.value,
            },
            "timestamp": now.isoformat(),
            "rationale": rationale,
            "proposed_diff": proposed_diff,
            "evidence_refs": evidence_refs,
            "approval": {
                "required_from": self._get_required_approver(spec_path),
                "approved_by": None,
                "approved_at": None,
            }
        }
        
        # Write proposal file
        proposal_file = proposals_dir / f"{proposal_id}.yaml"
        with proposal_file.open("w", encoding="utf-8") as f:
            yaml.dump(proposal, f, default_flow_style=False, allow_unicode=True)
        
        return {
            "success": True,
            "proposal_id": proposal_id,
            "proposal_file": str(proposal_file.relative_to(self.root)),
            "status": "proposed",
            "required_approval_from": self._get_required_approver(spec_path),
            "message": f"Spec change proposal created. Awaiting approval from {self._get_required_approver(spec_path)}."
        }
    
    def _get_required_approver(self, spec_path: str) -> str:
        """Determine who must approve a spec change based on path."""
        if "specs/canon" in spec_path:
            return "Project Owner (freeze required)"
        elif "specs/framework" in spec_path:
            return "Platform Engineer"
        elif "/specs/" in spec_path:
            return "Project Lead"
        elif "/experiments/" in spec_path:
            return "Auto-approval (threshold verification)"
        return "Human Operator"
    
    def _spec_commit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Commit an approved spec change."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        
        import yaml
        import subprocess
        from datetime import datetime
        
        proposal_id = args["proposal_id"]
        approval_ref = args["approval_ref"]
        run_hooks = args.get("run_hooks", True)
        
        # Find proposal file
        proposals_dir = self.root / "projects" / "dgsf" / "decisions"
        proposal_file = proposals_dir / f"{proposal_id}.yaml"
        
        if not proposal_file.exists():
            return {"error": "PROPOSAL_NOT_FOUND", "message": f"Proposal {proposal_id} not found"}
        
        with proposal_file.open("r", encoding="utf-8") as f:
            proposal = yaml.safe_load(f)
        
        spec_path = proposal["spec_path"]
        
        # Verify approval (for L0-L2, check approval_ref; for L3, auto-approve)
        is_l3 = "/experiments/" in spec_path and "config.yaml" in spec_path
        if not is_l3:
            # Check if approval file or reference exists
            approval_file = self.root / "decisions" / f"{approval_ref}.yaml"
            if not approval_file.exists() and not approval_ref.startswith("PR#"):
                # Check in project decisions folder too
                project_approval = proposals_dir / f"{approval_ref}.yaml"
                if not project_approval.exists():
                    return {
                        "error": "APPROVAL_NOT_FOUND",
                        "message": f"Approval reference '{approval_ref}' not found. Provide decision log path or PR number."
                    }
        
        # Run pre-spec-change hook
        if run_hooks:
            hook_path = self.root / "hooks" / "pre-spec-change"
            if hook_path.exists():
                try:
                    result = subprocess.run(
                        ["sh", str(hook_path), spec_path, proposal["change_type"], approval_ref],
                        capture_output=True,
                        text=True,
                        cwd=str(self.root),
                        timeout=30
                    )
                    if result.returncode != 0:
                        return {
                            "error": "PRE_HOOK_FAILED",
                            "message": f"Pre-spec-change hook failed: {result.stderr}"
                        }
                except subprocess.TimeoutExpired:
                    return {"error": "HOOK_TIMEOUT", "message": "Pre-spec-change hook timed out"}
                except Exception as e:
                    # On Windows, sh might not be available - try PowerShell
                    pass
        
        # Apply the change (parse diff and apply)
        full_spec_path = self.root / spec_path
        if not full_spec_path.exists() and proposal["change_type"] != "add":
            return {"error": "SPEC_NOT_FOUND", "message": f"Spec file not found: {spec_path}"}
        
        # For now, just update status - actual diff application would need patch utility
        now = datetime.now(timezone.utc)
        proposal["status"] = "committed"
        proposal["approval"]["approved_by"] = approval_ref
        proposal["approval"]["approved_at"] = now.isoformat()
        
        with proposal_file.open("w", encoding="utf-8") as f:
            yaml.dump(proposal, f, default_flow_style=False, allow_unicode=True)
        
        # Run post-spec-change hook
        if run_hooks:
            hook_path = self.root / "hooks" / "post-spec-change"
            if hook_path.exists():
                try:
                    subprocess.run(
                        ["sh", str(hook_path), spec_path, proposal["change_type"]],
                        capture_output=True,
                        text=True,
                        cwd=str(self.root),
                        timeout=60
                    )
                except Exception:
                    pass  # Post-hook failure is non-blocking
        
        return {
            "success": True,
            "proposal_id": proposal_id,
            "spec_path": spec_path,
            "status": "committed",
            "approval_ref": approval_ref,
            "message": f"Spec change {proposal_id} committed successfully. Run /dgsf_git_ops to finalize."
        }
    
    def _spec_triage(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a problem to determine if spec change is needed."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        from datetime import datetime
        
        problem_description = args["problem_description"]
        source = args["source"]
        context = args.get("context", {})
        
        # Generate triage ID
        now = datetime.now(timezone.utc)
        triage_id = f"TRI-{now.strftime('%Y-%m-%d')}-{now.strftime('%H%M%S')}"
        
        # Simple heuristic classification (in production, this could be more sophisticated)
        classification = {
            "category": "unknown",
            "root_cause": "unknown",
            "confidence": "low",
            "reasoning": ""
        }
        
        problem_lower = problem_description.lower()
        
        # Pattern matching for classification
        if any(kw in problem_lower for kw in ["threshold", "sharpe", "drawdown", "oos", "metric"]):
            classification["category"] = "metric_deviation"
            if any(kw in problem_lower for kw in ["threshold", "too low", "too high", "should be"]):
                classification["root_cause"] = "spec_issue"
                classification["confidence"] = "medium"
                classification["reasoning"] = "Problem mentions thresholds or metric criteria, suggesting spec-defined values may need adjustment."
            else:
                classification["root_cause"] = "model_issue"
                classification["confidence"] = "medium"
                classification["reasoning"] = "Metric deviation may be due to model performance rather than spec."
        
        elif any(kw in problem_lower for kw in ["error", "exception", "failed", "crash", "traceback"]):
            classification["category"] = "runtime_error"
            if any(kw in problem_lower for kw in ["assertion", "assert", "expected", "validation"]):
                classification["root_cause"] = "spec_issue"
                classification["confidence"] = "medium"
                classification["reasoning"] = "Assertion errors often indicate spec-defined constraints are being violated."
            else:
                classification["root_cause"] = "code_bug"
                classification["confidence"] = "medium"
                classification["reasoning"] = "Runtime errors typically indicate code issues."
        
        elif any(kw in problem_lower for kw in ["interface", "contract", "api", "signature", "incompatible"]):
            classification["category"] = "design_issue"
            classification["root_cause"] = "spec_issue"
            classification["confidence"] = "high"
            classification["reasoning"] = "Interface/contract issues directly relate to specification definitions."
        
        # Determine recommended action
        if classification["root_cause"] == "spec_issue":
            recommended_action = {
                "type": "spec_research",
                "command": "/dgsf_research",
                "then": "/dgsf_spec_propose",
                "description": "Research the issue and propose spec changes."
            }
        elif classification["root_cause"] == "code_bug":
            recommended_action = {
                "type": "code_diagnose",
                "command": "/dgsf_diagnose",
                "description": "Diagnose and fix the code bug."
            }
        else:
            recommended_action = {
                "type": "manual_investigation",
                "command": None,
                "description": "Manual investigation required. Insufficient information to classify."
            }
        
        # Calculate priority score
        impact_keywords = {"critical": 5, "major": 4, "blocking": 5, "production": 5, "all": 4}
        frequency_keywords = {"always": 5, "every": 5, "often": 4, "sometimes": 3, "rarely": 2}
        
        impact_score = 3  # default
        frequency_score = 3  # default
        
        for kw, score in impact_keywords.items():
            if kw in problem_lower:
                impact_score = max(impact_score, score)
        
        for kw, score in frequency_keywords.items():
            if kw in problem_lower:
                frequency_score = max(frequency_score, score)
        
        priority_score = impact_score * frequency_score
        if priority_score >= 16:
            priority = "P1"
        elif priority_score >= 9:
            priority = "P2"
        else:
            priority = "P3"
        
        return {
            "triage_id": triage_id,
            "problem": problem_description[:200],
            "source": source,
            "timestamp": now.isoformat(),
            "classification": classification,
            "priority": {
                "impact": impact_score,
                "frequency": frequency_score,
                "score": priority_score,
                "level": priority
            },
            "recommended_action": recommended_action,
            "context_received": bool(context)
        }
    
    # =========================================================================
    # Pair Programming / Code Review Tool Implementations
    # =========================================================================
    
    def _review_submit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Submit code artifacts for Pair Programming review."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        task_id = args["task_id"]
        artifact_paths = args.get("artifact_paths", [])
        notes = args.get("notes", "")
        
        # Validate builder role
        if session.role_mode not in (RoleMode.EXECUTOR, RoleMode.BUILDER):
            return {
                "error": "ROLE_MODE_VIOLATION",
                "message": f"Role Mode '{session.role_mode.value}' cannot submit for review"
            }
        
        # Check task exists and is in running state
        import yaml
        tasks_file = self.root / "state" / "tasks.yaml"
        if not tasks_file.exists():
            return {"error": "TASK_NOT_FOUND", "message": f"Task {task_id} not found"}
        
        with tasks_file.open("r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f) or {}
        
        task = tasks_data.get("tasks", {}).get(task_id, {})
        if not task:
            return {"error": "TASK_NOT_FOUND", "message": f"Task {task_id} not found"}
        
        current_status = task.get("status", "draft")
        if current_status not in ("running", "revision_needed"):
            return {
                "error": "INVALID_STATE",
                "message": f"Task must be in 'running' or 'revision_needed' state, currently: {current_status}"
            }
        
        # Validate artifacts exist
        missing = []
        for path in artifact_paths:
            full_path = self.root / path
            if not full_path.exists():
                missing.append(path)
        
        if missing:
            return {
                "error": "ARTIFACTS_NOT_FOUND",
                "message": f"Artifacts not found: {', '.join(missing)}"
            }
        
        # Create review submission record
        from datetime import datetime, timezone
        review_data = tasks_data.setdefault("reviews", {}).setdefault(task_id, {})
        submission = {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "submitted_by": session.agent_id,
            "artifacts": artifact_paths,
            "notes": notes,
            "revision_number": review_data.get("revision_count", 0) + 1,
        }
        review_data["current_submission"] = submission
        review_data["revision_count"] = submission["revision_number"]
        review_data["status"] = "pending_review"
        
        # Transition task to code_review state
        task["status"] = "code_review"
        task["last_updated"] = datetime.now(timezone.utc).isoformat()
        tasks_data["tasks"][task_id] = task
        
        with tasks_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(tasks_data, f, default_flow_style=False, sort_keys=False)
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "code_review",
            "submission": submission,
            "message": f"Task {task_id} submitted for Pair Programming review (revision #{submission['revision_number']})"
        }
    
    def _review_create_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a review session for Pair Programming."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        task_id = args["task_id"]
        personas = args.get("personas", [])
        
        # Validate reviewer role
        if session.role_mode != RoleMode.REVIEWER:
            # Allow builder to also review (but not their own code)
            if session.role_mode not in (RoleMode.REVIEWER, RoleMode.ARCHITECT, RoleMode.PLANNER):
                return {
                    "error": "ROLE_MODE_VIOLATION",
                    "message": f"Role Mode '{session.role_mode.value}' cannot conduct reviews. Use 'reviewer' mode."
                }
        
        # Get task and review data
        import yaml
        tasks_file = self.root / "state" / "tasks.yaml"
        if not tasks_file.exists():
            return {"error": "TASK_NOT_FOUND", "message": f"Task {task_id} not found"}
        
        with tasks_file.open("r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f) or {}
        
        task = tasks_data.get("tasks", {}).get(task_id, {})
        review_data = tasks_data.get("reviews", {}).get(task_id, {})
        
        if not review_data.get("current_submission"):
            return {
                "error": "NO_SUBMISSION",
                "message": f"No code submission pending for task {task_id}"
            }
        
        # Check for self-review
        builder_id = review_data["current_submission"].get("submitted_by")
        if builder_id == session.agent_id:
            return {
                "error": "SELF_REVIEW_PROHIBITED",
                "message": "Self-review is prohibited. A different agent must conduct the review."
            }
        
        # Create review session
        from datetime import datetime, timezone
        import hashlib
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        hash_input = f"{task_id}-{session.agent_id}-{timestamp}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        review_session_id = f"RSESS-{task_id}-{short_hash}"
        
        review_session = {
            "review_session_id": review_session_id,
            "reviewer_agent_id": session.agent_id,
            "builder_agent_id": builder_id,
            "task_id": task_id,
            "artifacts": review_data["current_submission"]["artifacts"],
            "revision_number": review_data["current_submission"]["revision_number"],
            "personas": personas,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
        }
        
        review_data["current_session"] = review_session
        review_data["status"] = "under_review"
        
        with tasks_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(tasks_data, f, default_flow_style=False, sort_keys=False)
        
        # Get TaskCard content for review context
        taskcard_path = self.root / "tasks" / f"{task_id}.md"
        taskcard_content = ""
        if taskcard_path.exists():
            taskcard_content = taskcard_path.read_text(encoding="utf-8")
        
        return {
            "success": True,
            "review_session_id": review_session_id,
            "task_id": task_id,
            "artifacts": review_session["artifacts"],
            "revision_number": review_session["revision_number"],
            "builder_agent_id": builder_id,
            "taskcard_content": taskcard_content,
            "personas": personas,
            "message": "Review session created. Use review_conduct to submit findings."
        }
    
    def _review_conduct(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct a code review and generate report."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        review_session_id = args["review_session_id"]
        
        quality_issues = args.get("quality_issues", [])
        requirements_issues = args.get("requirements_issues", [])
        completeness_issues = args.get("completeness_issues", [])
        optimization_suggestions = args.get("optimization_suggestions", [])
        requirements_coverage = args.get("requirements_coverage_pct", 100.0)
        completeness_pct = args.get("completeness_pct", 100.0)
        
        # Get review data
        import yaml
        tasks_file = self.root / "state" / "tasks.yaml"
        with tasks_file.open("r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f) or {}
        
        # Find the task with this review session
        task_id = None
        review_data = None
        for tid, rdata in tasks_data.get("reviews", {}).items():
            if rdata.get("current_session", {}).get("review_session_id") == review_session_id:
                task_id = tid
                review_data = rdata
                break
        
        if not review_data:
            return {"error": "SESSION_NOT_FOUND", "message": f"Review session {review_session_id} not found"}
        
        current_session = review_data["current_session"]
        if current_session.get("reviewer_agent_id") != session.agent_id:
            return {
                "error": "UNAUTHORIZED",
                "message": "Only the assigned reviewer can submit review findings"
            }
        
        # Count issues by severity
        critical = sum(1 for i in quality_issues + requirements_issues + completeness_issues if i.get("severity") == "CRITICAL")
        major = sum(1 for i in quality_issues + requirements_issues + completeness_issues if i.get("severity") == "MAJOR")
        minor = sum(1 for i in quality_issues + requirements_issues + completeness_issues if i.get("severity") == "MINOR")
        suggestions = len(optimization_suggestions) + sum(1 for i in quality_issues if i.get("severity") == "SUGGESTION")
        
        # Determine verdict
        if critical > 0 or major > 0:
            verdict = "NEEDS_REVISION"
        else:
            verdict = "APPROVED"
        
        # Create report
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        report_id = f"REV-{task_id}-{timestamp}-R{current_session['revision_number']}"
        
        report = {
            "report_id": report_id,
            "task_id": task_id,
            "reviewer_agent_id": session.agent_id,
            "builder_agent_id": current_session["builder_agent_id"],
            "review_session_id": review_session_id,
            "revision_number": current_session["revision_number"],
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "summary": {
                "critical_count": critical,
                "major_count": major,
                "minor_count": minor,
                "suggestion_count": suggestions,
            },
            "quality_check": {
                "passed": not any(i.get("severity") in ("CRITICAL", "MAJOR") for i in quality_issues),
                "issues": quality_issues,
            },
            "requirements_check": {
                "passed": not any(i.get("severity") in ("CRITICAL", "MAJOR") for i in requirements_issues),
                "issues": requirements_issues,
                "coverage_percentage": requirements_coverage,
            },
            "completeness_check": {
                "passed": not any(i.get("severity") in ("CRITICAL", "MAJOR") for i in completeness_issues),
                "issues": completeness_issues,
                "completeness_percentage": completeness_pct,
            },
            "optimization_check": {
                "suggestions": optimization_suggestions,
            },
            "personas_used": current_session.get("personas", []),
        }
        
        # Save report
        current_session["report"] = report
        current_session["status"] = "completed"
        review_data["status"] = verdict.lower()
        review_data.setdefault("reports", []).append(report)
        
        # Update task status based on verdict
        task = tasks_data["tasks"][task_id]
        if verdict == "NEEDS_REVISION":
            task["status"] = "revision_needed"
        # If approved, keep in code_review until explicit approval
        task["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        with tasks_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(tasks_data, f, default_flow_style=False, sort_keys=False)
        
        return {
            "success": True,
            "report_id": report_id,
            "verdict": verdict,
            "summary": report["summary"],
            "message": f"Review completed with verdict: {verdict}"
        }
    
    def _review_get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get review status for a task."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        task_id = args["task_id"]
        
        import yaml
        tasks_file = self.root / "state" / "tasks.yaml"
        if not tasks_file.exists():
            return {"error": "TASK_NOT_FOUND"}
        
        with tasks_file.open("r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f) or {}
        
        task = tasks_data.get("tasks", {}).get(task_id, {})
        review_data = tasks_data.get("reviews", {}).get(task_id, {})
        
        return {
            "task_id": task_id,
            "task_status": task.get("status"),
            "review_status": review_data.get("status", "not_started"),
            "revision_count": review_data.get("revision_count", 0),
            "current_submission": review_data.get("current_submission"),
            "current_session": review_data.get("current_session"),
            "latest_verdict": review_data.get("reports", [{}])[-1].get("verdict") if review_data.get("reports") else None,
        }
    
    def _review_get_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get review report for a task."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        task_id = args["task_id"]
        report_id = args.get("report_id")
        output_format = args.get("format", "json")
        
        import yaml
        tasks_file = self.root / "state" / "tasks.yaml"
        with tasks_file.open("r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f) or {}
        
        review_data = tasks_data.get("reviews", {}).get(task_id, {})
        reports = review_data.get("reports", [])
        
        if not reports:
            return {"error": "NO_REPORTS", "message": f"No review reports for task {task_id}"}
        
        # Find specific report or get latest
        report = None
        if report_id:
            for r in reports:
                if r.get("report_id") == report_id:
                    report = r
                    break
            if not report:
                return {"error": "REPORT_NOT_FOUND", "message": f"Report {report_id} not found"}
        else:
            report = reports[-1]
        
        if output_format == "yaml":
            return {"report": yaml.safe_dump(report, default_flow_style=False)}
        elif output_format == "markdown":
            # Generate markdown format
            md = self._report_to_markdown(report)
            return {"report": md}
        else:
            return {"report": report}
    
    def _report_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to markdown format."""
        lines = [
            f"# Code Review Report: {report['report_id']}",
            "",
            f"**Task**: {report['task_id']}",
            f"**Reviewer**: {report['reviewer_agent_id']}",
            f"**Builder**: {report['builder_agent_id']}",
            f"**Revision**: #{report['revision_number']}",
            f"**Reviewed At**: {report['reviewed_at']}",
            "",
            "---",
            "",
            f"## Verdict: {report['verdict']}",
            "",
            "### Issue Summary",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| CRITICAL | {report['summary']['critical_count']} |",
            f"| MAJOR | {report['summary']['major_count']} |",
            f"| MINOR | {report['summary']['minor_count']} |",
            f"| SUGGESTION | {report['summary']['suggestion_count']} |",
            "",
        ]
        
        # Add dimension details
        for dim_name, dim_data in [
            ("Quality Check", report["quality_check"]),
            ("Requirements Check", report["requirements_check"]),
            ("Completeness Check", report["completeness_check"]),
        ]:
            status = "✅ PASSED" if dim_data.get("passed") else "❌ FAILED"
            lines.append(f"## {dim_name}: {status}")
            lines.append("")
            
            if "coverage_percentage" in dim_data:
                lines.append(f"**Coverage**: {dim_data['coverage_percentage']:.1f}%")
                lines.append("")
            if "completeness_percentage" in dim_data:
                lines.append(f"**Completeness**: {dim_data['completeness_percentage']:.1f}%")
                lines.append("")
            
            for issue in dim_data.get("issues", []):
                lines.append(f"- **[{issue.get('severity', 'UNKNOWN')}]** {issue.get('check_id', 'N/A')}: {issue.get('description', 'No description')}")
                if issue.get("file_path"):
                    lines.append(f"  - File: `{issue['file_path']}`")
                if issue.get("line_start"):
                    lines.append(f"  - Line: {issue['line_start']}")
                if issue.get("suggested_fix"):
                    lines.append(f"  - Fix: {issue['suggested_fix']}")
            lines.append("")
        
        # Optimization suggestions
        lines.append("## Optimization Suggestions")
        lines.append("")
        for sugg in report.get("optimization_check", {}).get("suggestions", []):
            lines.append(f"- **{sugg.get('check_id', 'O-???')}**: {sugg.get('description', 'No description')}")
            if sugg.get("rationale"):
                lines.append(f"  - Rationale: {sugg['rationale']}")
            if sugg.get("impact"):
                lines.append(f"  - Impact: {sugg['impact']}")
        
        return "\n".join(lines)
    
    def _review_respond(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Respond to a code review."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        task_id = args["task_id"]
        action = args["action"]
        notes = args.get("notes", "")
        
        import yaml
        from datetime import datetime, timezone
        
        tasks_file = self.root / "state" / "tasks.yaml"
        with tasks_file.open("r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f) or {}
        
        task = tasks_data.get("tasks", {}).get(task_id, {})
        review_data = tasks_data.get("reviews", {}).get(task_id, {})
        
        # Validate builder is responding
        builder_id = review_data.get("current_submission", {}).get("submitted_by")
        if builder_id != session.agent_id:
            return {
                "error": "UNAUTHORIZED",
                "message": "Only the original builder can respond to review feedback"
            }
        
        response = {
            "action": action,
            "responded_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        
        review_data.setdefault("responses", []).append(response)
        
        if action == "revision_submitted":
            # Reset for new review
            review_data["status"] = "pending_review"
            task["status"] = "code_review"
        
        task["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        with tasks_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(tasks_data, f, default_flow_style=False, sort_keys=False)
        
        return {
            "success": True,
            "action": action,
            "message": f"Response '{action}' recorded for task {task_id}"
        }
    
    def _review_approve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Approve a code review and advance task."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        session = self.auth_manager.get_session(args["session_token"])
        if session is None:
            return {"error": "SESSION_NOT_FOUND", "message": "Session not found"}
        task_id = args["task_id"]
        review_session_id = args["review_session_id"]
        final_notes = args.get("final_notes", "")
        
        import yaml
        from datetime import datetime, timezone
        
        tasks_file = self.root / "state" / "tasks.yaml"
        with tasks_file.open("r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f) or {}
        
        review_data = tasks_data.get("reviews", {}).get(task_id, {})
        current_session = review_data.get("current_session", {})
        
        # Validate reviewer
        if current_session.get("reviewer_agent_id") != session.agent_id:
            return {
                "error": "UNAUTHORIZED",
                "message": "Only the assigned reviewer can approve"
            }
        
        # Check verdict allows approval
        latest_report = review_data.get("reports", [{}])[-1]
        if latest_report.get("verdict") == "NEEDS_REVISION":
            return {
                "error": "CANNOT_APPROVE",
                "message": "Cannot approve - review verdict is NEEDS_REVISION. Issues must be addressed first."
            }
        
        # Transition task to reviewing
        task = tasks_data["tasks"][task_id]
        task["status"] = "reviewing"
        task["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        review_data["status"] = "approved"
        review_data["approved_at"] = datetime.now(timezone.utc).isoformat()
        review_data["final_notes"] = final_notes
        
        with tasks_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(tasks_data, f, default_flow_style=False, sort_keys=False)
        
        return {
            "success": True,
            "task_id": task_id,
            "new_status": "reviewing",
            "message": f"Pair Programming review approved. Task {task_id} advanced to 'reviewing' state."
        }
    
    def _review_get_prompts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get review prompts for Pair Programming."""
        error = self._validate_session(args["session_token"])
        if error:
            return error
        
        task_id = args["task_id"]
        dimension = args["dimension"]
        persona = args.get("persona")
        
        import yaml
        
        # Load state machine config for review checks
        state_machine_path = self.root / "kernel" / "state_machine.yaml"
        with state_machine_path.open("r", encoding="utf-8") as f:
            sm_config = yaml.safe_load(f) or {}
        
        pp_config = sm_config.get("pair_programming", {})
        dimensions_config = pp_config.get("review_dimensions", {})
        personas_config = pp_config.get("personas", {})
        
        # Get TaskCard content
        taskcard_path = self.root / "tasks" / f"{task_id}.md"
        taskcard_content = ""
        if taskcard_path.exists():
            taskcard_content = taskcard_path.read_text(encoding="utf-8")
        
        # Get artifacts to review
        tasks_file = self.root / "state" / "tasks.yaml"
        with tasks_file.open("r", encoding="utf-8") as f:
            tasks_data = yaml.safe_load(f) or {}
        
        review_data = tasks_data.get("reviews", {}).get(task_id, {})
        artifacts = review_data.get("current_submission", {}).get("artifacts", [])
        
        # Read artifact content
        artifact_contents = {}
        for path in artifacts:
            full_path = self.root / path
            if full_path.exists():
                artifact_contents[path] = full_path.read_text(encoding="utf-8")
        
        prompts = {}
        
        if dimension in ("quality", "all"):
            checks = dimensions_config.get("quality_check", {}).get("checks", [])
            prompts["quality"] = self._generate_quality_prompt(artifact_contents, taskcard_content, checks)
        
        if dimension in ("requirements", "all"):
            checks = dimensions_config.get("requirements_check", {}).get("checks", [])
            prompts["requirements"] = self._generate_requirements_prompt(artifact_contents, taskcard_content, checks)
        
        if dimension in ("completeness", "all"):
            checks = dimensions_config.get("completeness_check", {}).get("checks", [])
            prompts["completeness"] = self._generate_completeness_prompt(artifact_contents, taskcard_content, checks)
        
        if dimension in ("optimization", "all"):
            checks = dimensions_config.get("optimization_check", {}).get("checks", [])
            prompts["optimization"] = self._generate_optimization_prompt(artifact_contents, checks)
        
        if persona and persona in personas_config:
            prompts["persona"] = self._generate_persona_prompt(persona, personas_config[persona], artifact_contents)
        
        return {
            "task_id": task_id,
            "artifacts": list(artifact_contents.keys()),
            "prompts": prompts,
        }
    
    def _generate_quality_prompt(self, artifacts: Dict[str, str], taskcard: str, checks: list) -> str:
        """Generate quality review prompt."""
        check_list = "\n".join([f"- {c['id']}: {c['name'].replace('_', ' ').title()}" for c in checks])
        code_blocks = "\n\n".join([f"### File: `{path}`\n```\n{content}\n```" for path, content in artifacts.items()])
        
        return f"""You are a Senior Code Reviewer conducting a **Quality Check** (Q-Check).

## Code to Review
{code_blocks}

## Task Context
{taskcard}

## Quality Checklist
{check_list}

## Instructions
For each issue found, provide in JSON format:
{{
  "check_id": "Q-XXX",
  "severity": "CRITICAL|MAJOR|MINOR|SUGGESTION",
  "description": "Clear description",
  "file_path": "path/to/file",
  "line_start": 123,
  "suggested_fix": "How to fix"
}}

Focus on: syntax, types, error handling, security, performance, duplication."""

    def _generate_requirements_prompt(self, artifacts: Dict[str, str], taskcard: str, checks: list) -> str:
        """Generate requirements review prompt."""
        check_list = "\n".join([f"- {c['id']}: {c['name'].replace('_', ' ').title()}" for c in checks])
        code_blocks = "\n\n".join([f"### File: `{path}`\n```\n{content}\n```" for path, content in artifacts.items()])
        
        return f"""You are a Senior Code Reviewer conducting a **Requirements Check** (R-Check).

## Code to Review
{code_blocks}

## TaskCard (Requirements)
{taskcard}

## Requirements Checklist
{check_list}

## Instructions
Verify each requirement is correctly implemented. For issues, provide:
{{
  "check_id": "R-XXX",
  "severity": "CRITICAL|MAJOR|MINOR",
  "description": "What requirement is violated/missing",
  "file_path": "path/to/file",
  "requirement_ref": "Reference to the requirement"
}}

Calculate and report requirements_coverage_pct (0-100)."""

    def _generate_completeness_prompt(self, artifacts: Dict[str, str], taskcard: str, checks: list) -> str:
        """Generate completeness review prompt."""
        check_list = "\n".join([f"- {c['id']}: {c['name'].replace('_', ' ').title()}" for c in checks])
        code_blocks = "\n\n".join([f"### File: `{path}`\n```\n{content}\n```" for path, content in artifacts.items()])
        
        return f"""You are a Senior Code Reviewer conducting a **Completeness Check** (C-Check).

## Code to Review
{code_blocks}

## TaskCard (All Requirements)
{taskcard}

## Completeness Checklist
{check_list}

## Instructions
Ensure ALL requirements are fully addressed. For missing items:
{{
  "check_id": "C-XXX",
  "severity": "CRITICAL|MAJOR|MINOR",
  "description": "What is missing",
  "missing_item": "Specific missing functionality"
}}

Calculate and report completeness_pct (0-100)."""

    def _generate_optimization_prompt(self, artifacts: Dict[str, str], checks: list) -> str:
        """Generate optimization review prompt."""
        check_list = "\n".join([f"- {c['id']}: {c['name'].replace('_', ' ').title()}" for c in checks])
        code_blocks = "\n\n".join([f"### File: `{path}`\n```\n{content}\n```" for path, content in artifacts.items()])
        
        return f"""You are a Senior Code Reviewer conducting an **Optimization Review** (O-Check).

## Code to Review
{code_blocks}

## Optimization Checklist
{check_list}

## Instructions
Identify opportunities to improve while maintaining functionality. For each suggestion:
{{
  "check_id": "O-XXX",
  "description": "What can be improved",
  "file_path": "path/to/file",
  "current_code": "snippet",
  "suggested_code": "improved snippet",
  "rationale": "Why this is better",
  "impact": "minor|moderate|significant"
}}

These are SUGGESTIONS, not blocking issues."""

    def _generate_persona_prompt(self, persona: str, config: Dict[str, Any], artifacts: Dict[str, str]) -> str:
        """Generate expert persona prompt."""
        description = config.get("description", "")
        focus = config.get("focus", [])
        code_blocks = "\n\n".join([f"### File: `{path}`\n```\n{content}\n```" for path, content in artifacts.items()])
        
        return f"""You are a **{persona.replace('_', ' ').title()}** reviewing code.

## Your Expertise
{description}

## Focus Areas
{', '.join(focus)}

## Code to Review
{code_blocks}

## Instructions
Apply your specialized expertise. For each finding:
{{
  "check_id": "Related check ID",
  "severity": "CRITICAL|MAJOR|MINOR|SUGGESTION",
  "description": "Technical details",
  "file_path": "path",
  "remediation": "Steps to fix"
}}"""

    # =========================================================================
    # Artifact Locking
    # =========================================================================

    def _agent_lock_artifact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Acquire exclusive lock on an artifact."""
        session_token = args["session_token"]
        artifact_path = args["artifact_path"]
        
        result = self.auth_manager.lock_artifact(
            session_token=session_token,
            artifact_path=artifact_path,
        )
        
        return result

    def _agent_unlock_artifact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Release lock on an artifact."""
        session_token = args["session_token"]
        artifact_path = args["artifact_path"]
        
        result = self.auth_manager.unlock_artifact(
            session_token=session_token,
            artifact_path=artifact_path,
        )
        
        return result


def create_server() -> MCPServer:
    """Create and return an MCP server instance."""
    return MCPServer()


# MCP Protocol Handler (stdin/stdout JSON-RPC style)
def main():
    """Main entry point for MCP server."""
    import sys
    import json
    
    server = create_server()
    
    # Print available tools on startup
    print(json.dumps({
        "type": "tools",
        "tools": server.get_tools()
    }), file=sys.stderr)
    
    # Simple REPL for testing
    print("AI Workflow OS MCP Server Ready", file=sys.stderr)
    print("Enter tool calls as JSON: {\"tool\": \"name\", \"arguments\": {...}}", file=sys.stderr)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            tool_name = request.get("tool")
            arguments = request.get("arguments", {})
            
            result = server.call_tool(tool_name, arguments)
            print(json.dumps({"result": result}))
            sys.stdout.flush()
            
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
