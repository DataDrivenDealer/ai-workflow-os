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

# Add kernel to path
KERNEL_DIR = Path(__file__).parent
sys.path.insert(0, str(KERNEL_DIR))

from agent_auth import AgentAuthManager, RoleMode, SessionState, get_auth_manager
from governance_gate import GovernanceGate, verify_governance, has_violations, get_all_violations


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
                                "enum": ["architect", "planner", "executor", "builder"]
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
                            "enum": ["architect", "planner", "executor", "builder"],
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
        if not session:
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
        
        artifact_path = self.root / args["path"]
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
        
        dir_path = self.root / args["path"]
        if not dir_path.exists():
            return {"error": "NOT_FOUND", "message": f"Directory not found: {args['path']}"}
        
        if not dir_path.is_dir():
            return {"error": "NOT_A_DIRECTORY", "message": f"Path is not a directory: {args['path']}"}
        
        items = []
        for item in sorted(dir_path.iterdir()):
            items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "path": str(item.relative_to(self.root)),
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
