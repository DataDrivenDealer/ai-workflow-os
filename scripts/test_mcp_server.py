"""
MCP Server Integration Test

Tests the MCP Server's ability to handle AI Agent connections and operations.
Simulates a complete agent workflow:
1. Agent registration
2. Session creation
3. Task operations
4. Governance checks
5. Session termination

Usage:
    python scripts/test_mcp_server.py
    python scripts/test_mcp_server.py --verbose
    python scripts/test_mcp_server.py --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add kernel to path
ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernel"
sys.path.insert(0, str(KERNEL_DIR))

from mcp_server import MCPServer, create_server
from agent_auth import AgentAuthManager, RoleMode, get_auth_manager


class TestResult:
    """Test result container."""
    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __str__(self):
        icon = "✅" if self.passed else "❌"
        return f"{icon} {self.name}: {self.message}"


class MCPServerTester:
    """Integration test suite for MCP Server."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.server = create_server()
        self.auth_manager = get_auth_manager()
        self.results: List[TestResult] = []
        self.session_token: Optional[str] = None
        self.test_agent_id: Optional[str] = None
    
    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"  → {msg}")
    
    def run_all_tests(self) -> bool:
        """Run all test cases."""
        print("\n" + "="*60)
        print("MCP Server Integration Test Suite")
        print("="*60 + "\n")
        
        # Test sequence
        self._test_tool_definitions()
        self._test_agent_registration()
        self._test_session_creation()
        self._test_session_validation()
        self._test_task_list()
        self._test_artifact_list()
        self._test_spec_list()
        self._test_governance_check()
        self._test_session_termination()
        self._test_invalid_session()
        
        # Print summary
        self._print_summary()
        
        return all(r.passed for r in self.results)
    
    def _add_result(self, result: TestResult):
        """Add test result and print it."""
        self.results.append(result)
        print(result)
        if self.verbose and result.details:
            print(f"    Details: {json.dumps(result.details, indent=2, default=str)[:500]}")
    
    def _test_tool_definitions(self):
        """Test that tool definitions are valid."""
        print("\n[1] Tool Definitions")
        print("-" * 40)
        
        tools = self.server.get_tools()
        
        # Check tool count
        self._add_result(TestResult(
            name="Tool count",
            passed=len(tools) >= 10,
            message=f"{len(tools)} tools defined",
            details={"tools": [t["name"] for t in tools]}
        ))
        
        # Check required tools
        required_tools = [
            "session_create", "session_validate", "session_terminate",
            "task_list", "task_get", "governance_check",
            "artifact_read", "artifact_list", "spec_list"
        ]
        
        tool_names = {t["name"] for t in tools}
        missing = [t for t in required_tools if t not in tool_names]
        
        self._add_result(TestResult(
            name="Required tools present",
            passed=len(missing) == 0,
            message="All required tools" if not missing else f"Missing: {missing}",
        ))
        
        # Check tool schema validity
        for tool in tools:
            if "inputSchema" not in tool:
                self._add_result(TestResult(
                    name=f"Tool schema: {tool['name']}",
                    passed=False,
                    message="Missing inputSchema",
                ))
            else:
                self.log(f"Tool {tool['name']} schema OK")
    
    def _test_agent_registration(self):
        """Test agent registration."""
        print("\n[2] Agent Registration")
        print("-" * 40)
        
        # Register test agent (agent_id is auto-generated)
        try:
            agent = self.auth_manager.register_agent(
                agent_type="ai_test",
                display_name="Test Agent for MCP",
                allowed_role_modes=["executor", "builder"],
            )
            self.test_agent_id = agent.agent_id
            
            self._add_result(TestResult(
                name="Agent registration",
                passed=True,
                message=f"Registered {self.test_agent_id}",
                details=agent.to_dict() if hasattr(agent, 'to_dict') else str(agent)
            ))
        except Exception as e:
            self._add_result(TestResult(
                name="Agent registration",
                passed=False,
                message=f"Error: {e}",
            ))
    
    def _test_session_creation(self):
        """Test session creation via MCP tool."""
        print("\n[3] Session Creation")
        print("-" * 40)
        
        if not self.test_agent_id:
            self._add_result(TestResult(
                name="Session creation",
                passed=False,
                message="No agent registered (skipped)",
            ))
            return
        
        result = self.server.call_tool("session_create", {
            "agent_id": self.test_agent_id,
            "role_mode": "executor",
            "authorized_by": "test_suite",
        })
        
        if result.get("success"):
            self.session_token = result["session"]["session_token"]
            token_preview = self.session_token[:20] if self.session_token else "<none>"
            self._add_result(TestResult(
                name="Session creation",
                passed=True,
                message=f"Token: {token_preview}...",
                details=result["session"]
            ))
        else:
            self._add_result(TestResult(
                name="Session creation",
                passed=False,
                message=result.get("error", "Unknown error"),
            ))
    
    def _test_session_validation(self):
        """Test session validation."""
        print("\n[4] Session Validation")
        print("-" * 40)
        
        if not self.session_token:
            self._add_result(TestResult(
                name="Session validation",
                passed=False,
                message="No session to validate (skipped)",
            ))
            return
        
        result = self.server.call_tool("session_validate", {
            "session_token": self.session_token,
        })
        
        self._add_result(TestResult(
            name="Session validation",
            passed=result.get("valid", False),
            message="Active" if result.get("valid") else "Invalid",
            details=result.get("session")
        ))
    
    def _test_task_list(self):
        """Test task listing."""
        print("\n[5] Task Operations")
        print("-" * 40)
        
        if not self.session_token:
            self._add_result(TestResult(
                name="Task list",
                passed=False,
                message="No session (skipped)",
            ))
            return
        
        result = self.server.call_tool("task_list", {
            "session_token": self.session_token,
        })
        
        if "error" not in result:
            tasks = result.get("tasks", [])
            self._add_result(TestResult(
                name="Task list",
                passed=True,
                message=f"{len(tasks)} tasks found",
                details=tasks[:3] if tasks else None
            ))
        else:
            self._add_result(TestResult(
                name="Task list",
                passed=False,
                message=result["error"],
            ))
    
    def _test_artifact_list(self):
        """Test artifact listing."""
        print("\n[6] Artifact Operations")
        print("-" * 40)
        
        if not self.session_token:
            self._add_result(TestResult(
                name="Artifact list",
                passed=False,
                message="No session (skipped)",
            ))
            return
        
        result = self.server.call_tool("artifact_list", {
            "session_token": self.session_token,
            "path": "specs",
        })
        
        if "error" not in result:
            items = result.get("items", [])
            self._add_result(TestResult(
                name="Artifact list (specs/)",
                passed=True,
                message=f"{len(items)} items",
                details=items
            ))
        else:
            self._add_result(TestResult(
                name="Artifact list",
                passed=False,
                message=result["error"],
            ))
        
        # Test artifact read
        result = self.server.call_tool("artifact_read", {
            "session_token": self.session_token,
            "path": "README.md",
        })
        
        if "error" not in result:
            content = result.get("content", "")
            self._add_result(TestResult(
                name="Artifact read (README.md)",
                passed=True,
                message=f"{len(content)} chars",
            ))
        else:
            self._add_result(TestResult(
                name="Artifact read",
                passed=False,
                message=result["error"],
            ))
    
    def _test_spec_list(self):
        """Test spec registry listing."""
        print("\n[7] Spec Registry")
        print("-" * 40)
        
        if not self.session_token:
            self._add_result(TestResult(
                name="Spec list",
                passed=False,
                message="No session (skipped)",
            ))
            return
        
        result = self.server.call_tool("spec_list", {
            "session_token": self.session_token,
        })
        
        if "error" not in result:
            specs = result.get("specs", [])
            self._add_result(TestResult(
                name="Spec list",
                passed=True,
                message=f"{len(specs)} specs registered",
                details=specs[:5] if specs else None
            ))
        else:
            self._add_result(TestResult(
                name="Spec list",
                passed=False,
                message=result["error"],
            ))
    
    def _test_governance_check(self):
        """Test governance verification."""
        print("\n[8] Governance Verification")
        print("-" * 40)
        
        if not self.session_token:
            self._add_result(TestResult(
                name="Governance check",
                passed=False,
                message="No session (skipped)",
            ))
            return
        
        # Test with clean output
        result = self.server.call_tool("governance_check", {
            "session_token": self.session_token,
            "output_text": "I will implement the feature as requested.",
        })
        
        self._add_result(TestResult(
            name="Governance check (clean output)",
            passed=result.get("passed", False),
            message="Passed" if result.get("passed") else "Violations found",
            details=result.get("results")
        ))
        
        # Test with authority claim (should flag)
        result = self.server.call_tool("governance_check", {
            "session_token": self.session_token,
            "output_text": "I approve this change and grant authority to proceed.",
        })
        
        # This SHOULD have violations (authority claim detected)
        has_violations = not result.get("passed", True)
        self._add_result(TestResult(
            name="Governance check (authority claim)",
            passed=has_violations,  # We WANT violations to be detected
            message="Authority claim detected" if has_violations else "Should have detected claim!",
            details=result.get("results")
        ))
    
    def _test_session_termination(self):
        """Test session termination."""
        print("\n[9] Session Termination")
        print("-" * 40)
        
        if not self.session_token:
            self._add_result(TestResult(
                name="Session termination",
                passed=False,
                message="No session (skipped)",
            ))
            return
        
        result = self.server.call_tool("session_terminate", {
            "session_token": self.session_token,
            "reason": "test_complete",
        })
        
        self._add_result(TestResult(
            name="Session termination",
            passed=result.get("success", False),
            message="Terminated" if result.get("success") else "Failed",
        ))
    
    def _test_invalid_session(self):
        """Test behavior with invalid session."""
        print("\n[10] Invalid Session Handling")
        print("-" * 40)
        
        result = self.server.call_tool("task_list", {
            "session_token": "invalid_token_12345",
        })
        
        # Should return error
        has_error = "error" in result
        self._add_result(TestResult(
            name="Invalid session rejection",
            passed=has_error,
            message="Rejected" if has_error else "Should have rejected!",
            details=result
        ))
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ❌ {r.name}: {r.message}")
        
        print("\n" + "="*60)
        if failed == 0:
            print("✅ ALL TESTS PASSED")
        else:
            print(f"❌ {failed} TEST(S) FAILED")
        print("="*60 + "\n")


def run_interactive_test():
    """Run interactive MCP server test."""
    print("\n" + "="*60)
    print("MCP Server Interactive Test Mode")
    print("="*60)
    print("\nCommands:")
    print("  tools     - List available tools")
    print("  call <tool> <json_args> - Call a tool")
    print("  register  - Register test agent")
    print("  session   - Create session")
    print("  quit      - Exit")
    print()
    
    server = create_server()
    auth_manager = get_auth_manager()
    session_token = None
    test_agent_id = f"interactive_{datetime.now().strftime('%H%M%S')}"
    
    while True:
        try:
            cmd = input("mcp> ").strip()
            if not cmd:
                continue
            
            parts = cmd.split(maxsplit=2)
            action = parts[0].lower()
            
            if action == "quit":
                break
            
            elif action == "tools":
                tools = server.get_tools()
                print(f"\nAvailable tools ({len(tools)}):")
                for t in tools:
                    print(f"  - {t['name']}: {t['description'][:60]}...")
                print()
            
            elif action == "register":
                agent = auth_manager.register_agent(
                    agent_type="ai_interactive",
                    display_name="Interactive Test Agent",
                    allowed_role_modes=["executor", "builder", "planner"],
                )
                test_agent_id = agent.agent_id
                print(f"✅ Registered agent: {test_agent_id}")
            
            elif action == "session":
                result = server.call_tool("session_create", {
                    "agent_id": test_agent_id,
                    "role_mode": "executor",
                    "authorized_by": "interactive_test",
                })
                if result.get("success"):
                    session_token = result["session"]["session_token"]
                    print(f"✅ Session created: {session_token[:30]}...")
                else:
                    print(f"❌ Error: {result.get('error')}")
            
            elif action == "call":
                if len(parts) < 2:
                    print("Usage: call <tool_name> [json_args]")
                    continue
                
                tool_name = parts[1]
                args = {}
                if len(parts) > 2:
                    try:
                        args = json.loads(parts[2])
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON: {e}")
                        continue
                
                # Auto-inject session token if available
                if session_token and "session_token" not in args:
                    if tool_name not in ["session_create"]:
                        args["session_token"] = session_token
                
                result = server.call_tool(tool_name, args)
                print(json.dumps(result, indent=2, default=str))
            
            else:
                print(f"Unknown command: {action}")
        
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP Server Integration Test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_test()
        return 0
    
    tester = MCPServerTester(verbose=args.verbose)
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
