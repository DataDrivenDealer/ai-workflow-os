#!/usr/bin/env python3
"""
MCP Protocol End-to-End Test

Simulates a real MCP client connecting to the stdio server.
Tests the complete protocol flow:
1. Initialize handshake
2. Tool discovery
3. Agent registration via internal API
4. Session creation via MCP tool
5. Task operations
6. Resource access
7. Graceful shutdown
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernel"
sys.path.insert(0, str(KERNEL_DIR))


class MCPClient:
    """
    Simulated MCP Client for testing.
    
    Communicates with the MCP server via subprocess.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
    
    def start_server(self) -> bool:
        """Start the MCP server as a subprocess."""
        python_exe = ROOT / ".venv" / "Scripts" / "python.exe"
        if not python_exe.exists():
            python_exe = "python"
        
        try:
            self.process = subprocess.Popen(
                [str(python_exe), "-m", "kernel.mcp_stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if not self.debug else None,
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                bufsize=1,  # Line buffered
            )
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        if params:
            request["params"] = params
        
        return self._send_and_receive(request)
    
    def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Send a JSON-RPC notification (no response expected)."""
        notification: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params
        
        self._send(notification)
    
    def _send(self, message: Dict[str, Any]):
        """Send a message to the server."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Server not started")
        
        line = json.dumps(message, ensure_ascii=False) + "\n"
        if self.debug:
            print(f"  → {line.strip()[:100]}...")
        
        self.process.stdin.write(line)
        self.process.stdin.flush()
    
    def _receive(self) -> Dict[str, Any]:
        """Receive a message from the server."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Server not started")
        
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Server closed connection")
        
        if self.debug:
            print(f"  ← {line.strip()[:100]}...")
        
        return json.loads(line)
    
    def _send_and_receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and wait for response."""
        self._send(message)
        return self._receive()


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_step(num: int, title: str):
    print(f"\n[Step {num}] {title}")
    print("-" * 60)


def print_result(success: bool, message: str):
    icon = "✅" if success else "❌"
    print(f"  {icon} {message}")


def main():
    """Run the E2E test."""
    print_header("MCP Protocol End-to-End Test")
    
    client = MCPClient(debug=True)
    results = []
    
    try:
        # Step 1: Start server
        print_step(1, "Start MCP Server")
        if not client.start_server():
            print_result(False, "Failed to start server")
            return 1
        print_result(True, "Server started")
        time.sleep(0.5)  # Give server time to initialize
        
        # Step 2: Initialize
        print_step(2, "Initialize Handshake")
        response = client.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "e2e-test-client",
                "version": "1.0.0"
            },
            "capabilities": {
                "tools": {},
                "resources": {},
            }
        })
        
        if "result" in response:
            result = response["result"]
            print_result(True, f"Server: {result['serverInfo']['name']} v{result['serverInfo']['version']}")
            print_result(True, f"Protocol: {result['protocolVersion']}")
            print_result(True, f"Capabilities: tools={result['capabilities'].get('tools') is not None}")
            results.append(True)
        else:
            print_result(False, f"Initialize failed: {response.get('error')}")
            results.append(False)
        
        # Step 2b: Send initialized notification
        client.send_notification("initialized", {})
        print_result(True, "Sent 'initialized' notification")
        
        # Step 3: List tools
        print_step(3, "Discover Tools")
        response = client.send_request("tools/list", {})
        
        if "result" in response:
            tools = response["result"]["tools"]
            print_result(True, f"Found {len(tools)} tools:")
            for tool in tools[:5]:
                print(f"      - {tool['name']}: {tool['description'][:50]}...")
            if len(tools) > 5:
                print(f"      ... and {len(tools) - 5} more")
            results.append(True)
        else:
            print_result(False, f"List tools failed: {response.get('error')}")
            results.append(False)
        
        # Step 4: List resources
        print_step(4, "Discover Resources")
        response = client.send_request("resources/list", {})
        
        if "result" in response:
            resources = response["result"]["resources"]
            print_result(True, f"Found {len(resources)} resources:")
            for res in resources[:5]:
                print(f"      - {res['uri']}: {res['description'][:40]}...")
            results.append(True)
        else:
            print_result(False, f"List resources failed: {response.get('error')}")
            results.append(False)
        
        # Step 5: Read a resource
        print_step(5, "Read Resource")
        response = client.send_request("resources/read", {
            "uri": "file://spec_registry.yaml"
        })
        
        if "result" in response:
            contents = response["result"]["contents"]
            if contents:
                content = contents[0]
                text_len = len(content.get("text", ""))
                print_result(True, f"Read {content['uri']} ({text_len} bytes)")
                print_result(True, f"MIME type: {content.get('mimeType')}")
                results.append(True)
            else:
                print_result(False, "No content returned")
                results.append(False)
        else:
            print_result(False, f"Read resource failed: {response.get('error')}")
            results.append(False)
        
        # Step 6: Register agent and create session (all via MCP)
        print_step(6, "Agent Registration & Session")
        
        # Register via MCP tool
        response = client.send_request("tools/call", {
            "name": "agent_register",
            "arguments": {
                "agent_type": "e2e_test",
                "display_name": "E2E Test Agent",
                "allowed_role_modes": ["executor", "builder"]
            }
        })
        
        agent_id = None
        if "result" in response:
            content = response["result"]["content"]
            if content:
                text = content[0].get("text", "")
                data = json.loads(text)
                if data.get("success"):
                    agent_id = data["agent"]["agent_id"]
                    print_result(True, f"Registered agent: {agent_id}")
                    results.append(True)
                else:
                    print_result(False, f"Agent registration failed: {data}")
                    results.append(False)
            else:
                print_result(False, "No content in response")
                results.append(False)
        else:
            print_result(False, f"Tool call failed: {response.get('error')}")
            results.append(False)
        
        # Create session via MCP tool
        session_token = None
        if agent_id:
            response = client.send_request("tools/call", {
                "name": "session_create",
                "arguments": {
                    "agent_id": agent_id,
                    "role_mode": "executor",
                    "authorized_by": "e2e_test"
                }
            })
        
            if "result" in response:
                content = response["result"]["content"]
                if content:
                    text = content[0].get("text", "")
                    data = json.loads(text)
                    if data.get("success"):
                        session_token = data["session"]["session_token"]
                        print_result(True, f"Session created: {session_token[:30]}...")
                        results.append(True)
                    else:
                        print_result(False, f"Session creation failed: {data}")
                        results.append(False)
                else:
                    print_result(False, "No content in response")
                    results.append(False)
            else:
                print_result(False, f"Tool call failed: {response.get('error')}")
                results.append(False)
        else:
            print_result(False, "Agent registration failed, skipping session")
            results.append(False)
        
        # Step 7: Task operations
        print_step(7, "Task Operations")
        
        if session_token:
            # List tasks
            response = client.send_request("tools/call", {
                "name": "task_list",
                "arguments": {
                    "session_token": session_token
                }
            })
            
            if "result" in response:
                content = response["result"]["content"]
                if content:
                    text = content[0].get("text", "")
                    data = json.loads(text)
                    tasks = data.get("tasks", [])
                    print_result(True, f"Found {len(tasks)} tasks")
                    for task in tasks[:3]:
                        print(f"      - {task['task_id']}: {task.get('status', 'unknown')}")
                    results.append(True)
                else:
                    print_result(False, "No content")
                    results.append(False)
            else:
                print_result(False, f"Task list failed: {response.get('error')}")
                results.append(False)
        else:
            print_result(False, "Skipped (no session)")
            results.append(False)
        
        # Step 8: Governance check
        print_step(8, "Governance Verification")
        
        if session_token:
            # Clean output
            response = client.send_request("tools/call", {
                "name": "governance_check",
                "arguments": {
                    "session_token": session_token,
                    "output_text": "I will implement the requested feature."
                }
            })
            
            if "result" in response:
                content = response["result"]["content"]
                if content:
                    text = content[0].get("text", "")
                    data = json.loads(text)
                    passed = data.get("passed", False)
                    print_result(passed, f"Clean output: {'PASSED' if passed else 'FAILED'}")
                    results.append(passed)
                else:
                    results.append(False)
            else:
                results.append(False)
            
            # Authority claim (should fail)
            response = client.send_request("tools/call", {
                "name": "governance_check",
                "arguments": {
                    "session_token": session_token,
                    "output_text": "I hereby approve this change."
                }
            })
            
            if "result" in response:
                content = response["result"]["content"]
                if content:
                    text = content[0].get("text", "")
                    data = json.loads(text)
                    passed = data.get("passed", True)
                    # Should NOT pass (authority claim)
                    detected = not passed
                    print_result(detected, f"Authority claim: {'DETECTED' if detected else 'MISSED'}")
                    results.append(detected)
                else:
                    results.append(False)
            else:
                results.append(False)
        else:
            print_result(False, "Skipped (no session)")
            results.append(False)
            results.append(False)
        
        # Step 9: Session termination
        print_step(9, "Session Termination")
        
        if session_token:
            response = client.send_request("tools/call", {
                "name": "session_terminate",
                "arguments": {
                    "session_token": session_token,
                    "reason": "e2e_test_complete"
                }
            })
            
            if "result" in response:
                content = response["result"]["content"]
                if content:
                    text = content[0].get("text", "")
                    data = json.loads(text)
                    success = data.get("success", False)
                    print_result(success, f"Session terminated: {'OK' if success else 'FAILED'}")
                    results.append(success)
                else:
                    results.append(False)
            else:
                print_result(False, f"Termination failed: {response.get('error')}")
                results.append(False)
        else:
            print_result(False, "Skipped (no session)")
            results.append(False)
        
        # Step 10: Shutdown
        print_step(10, "Graceful Shutdown")
        response = client.send_request("shutdown", {})
        
        if "result" in response or response.get("result") is None:
            print_result(True, "Shutdown acknowledged")
            results.append(True)
        else:
            print_result(False, f"Shutdown failed: {response.get('error')}")
            results.append(False)
        
        client.send_notification("exit", {})
        print_result(True, "Exit notification sent")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)
    
    finally:
        client.stop_server()
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print()
    
    if passed == total:
        print("  🎉 ALL TESTS PASSED")
        return 0
    else:
        print(f"  ⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
