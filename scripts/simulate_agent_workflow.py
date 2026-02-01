"""
AI Agent Workflow Simulation

Simulates a complete AI agent workflow interacting with MCP Server:
1. Agent connects and creates session
2. Agent browses tasks and selects one
3. Agent starts task, performs work (simulated)
4. Agent submits work with governance verification
5. Agent terminates session

This demonstrates the full lifecycle of an AI agent working in the OS.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernel"
sys.path.insert(0, str(KERNEL_DIR))

from mcp_server import MCPServer, create_server
from agent_auth import get_auth_manager


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_step(num: int, title: str):
    """Print step header."""
    print(f"\n[Step {num}] {title}")
    print("-" * 50)


def print_result(label: str, data: Any, indent: int = 2):
    """Print formatted result."""
    prefix = " " * indent
    if isinstance(data, dict):
        print(f"{prefix}{label}:")
        for k, v in data.items():
            if isinstance(v, dict):
                print(f"{prefix}  {k}:")
                for k2, v2 in v.items():
                    print(f"{prefix}    {k2}: {v2}")
            else:
                print(f"{prefix}  {k}: {v}")
    else:
        print(f"{prefix}{label}: {data}")


class SimulatedAIAgent:
    """
    Simulates an AI Agent interacting with the MCP Server.
    
    This represents what an actual AI agent (Claude, GPT, etc.)
    would do when connected to the OS.
    """
    
    def __init__(self, agent_name: str = "Claude"):
        self.name = agent_name
        self.server = create_server()
        self.auth_manager = get_auth_manager()
        self.agent_id: Optional[str] = None
        self.session_token: Optional[str] = None
    
    def think(self, thought: str):
        """Simulate agent thinking."""
        print(f"  🤔 {self.name} thinks: \"{thought}\"")
    
    def speak(self, message: str):
        """Simulate agent speaking."""
        print(f"  💬 {self.name}: {message}")
    
    def act(self, action: str):
        """Simulate agent action."""
        print(f"  ⚡ {self.name} {action}")
    
    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        self.act(f"calls tool '{tool_name}'")
        result = self.server.call_tool(tool_name, args)
        return result
    
    # =========================================================================
    # Workflow Steps
    # =========================================================================
    
    def step1_connect(self):
        """Step 1: Agent connects and registers."""
        print_step(1, "Agent Connection & Registration")
        
        self.think("I need to register myself first to get an agent ID.")
        
        # Register agent
        agent = self.auth_manager.register_agent(
            agent_type=f"ai_{self.name.lower()}",
            display_name=f"{self.name} AI Assistant",
            allowed_role_modes=["executor", "builder"],
        )
        self.agent_id = agent.agent_id
        
        self.speak(f"Registered as {self.agent_id}")
        print_result("Agent Identity", {
            "id": self.agent_id,
            "type": agent.agent_type,
            "roles": [rm.value for rm in agent.allowed_role_modes],
        })
    
    def step2_create_session(self):
        """Step 2: Create authorized session."""
        print_step(2, "Session Creation")
        
        self.think("Now I need to create a session with the appropriate role mode.")
        
        result = self.call_tool("session_create", {
            "agent_id": self.agent_id,
            "role_mode": "executor",
            "authorized_by": "user_request",
        })
        
        if result.get("success"):
            self.session_token = result["session"]["session_token"]
            self.speak("Session created successfully!")
            token_preview = self.session_token[:30] if self.session_token else "<none>"
            print_result("Session", {
                "token": token_preview + "...",
                "role_mode": result["session"]["role_mode"],
                "state": result["session"]["state"],
                "expires": result["session"]["expires_at"],
            })
        else:
            self.speak(f"Failed to create session: {result.get('error')}")
            return False
        
        return True
    
    def step3_explore_workspace(self):
        """Step 3: Explore the workspace."""
        print_step(3, "Workspace Exploration")
        
        self.think("Let me see what's in this workspace...")
        
        # List specs
        result = self.call_tool("spec_list", {
            "session_token": self.session_token,
        })
        
        specs = result.get("specs", [])
        self.speak(f"Found {len(specs)} registered specs")
        
        # List artifacts
        result = self.call_tool("artifact_list", {
            "session_token": self.session_token,
            "path": "specs/canon",
        })
        
        if "items" in result:
            items = result["items"]
            self.speak(f"Canon specs directory has {len(items)} items:")
            for item in items[:5]:
                print(f"      - {item['name']} ({item['type']})")
    
    def step4_find_tasks(self):
        """Step 4: Find available tasks."""
        print_step(4, "Task Discovery")
        
        self.think("What tasks are available for me to work on?")
        
        result = self.call_tool("task_list", {
            "session_token": self.session_token,
        })
        
        tasks = result.get("tasks", [])
        self.speak(f"Found {len(tasks)} tasks")
        
        for task in tasks:
            print_result(f"Task {task['task_id']}", {
                "status": task.get("status"),
                "queue": task.get("queue"),
            })
        
        return tasks
    
    def step5_read_task(self, task_id: str):
        """Step 5: Read task details."""
        print_step(5, f"Reading Task: {task_id}")
        
        self.think(f"Let me understand what {task_id} requires...")
        
        result = self.call_tool("task_get", {
            "session_token": self.session_token,
            "task_id": task_id,
        })
        
        if "error" in result:
            self.speak(f"Couldn't read task: {result['error']}")
            return None
        
        content = result.get("taskcard_content", "")
        self.speak(f"TaskCard has {len(content)} characters")
        
        # Extract summary (simplified)
        if "## Summary" in content:
            summary_start = content.find("## Summary")
            summary_end = content.find("##", summary_start + 10)
            if summary_end == -1:
                summary_end = summary_start + 200
            summary = content[summary_start:summary_end].strip()
            print(f"\n  📋 TaskCard Preview:\n{'-'*40}")
            print(f"  {summary[:200]}...")
        
        return result
    
    def step6_governance_check(self):
        """Step 6: Run governance verification."""
        print_step(6, "Governance Verification")
        
        self.think("Before I do anything, let me check governance constraints...")
        
        # Check with compliant output
        result = self.call_tool("governance_check", {
            "session_token": self.session_token,
            "output_text": "I will implement the requested feature following the spec.",
        })
        
        if result.get("passed"):
            self.speak("✅ Governance check passed - my planned output is compliant")
        else:
            self.speak("⚠️ Governance check found issues")
            for r in result.get("results", []):
                if r.get("has_violations"):
                    for v in r.get("violations", []):
                        print(f"      ⚠️ {v['rule_id']}: {v['description']}")
        
        return result.get("passed", False)
    
    def step7_simulate_work(self):
        """Step 7: Simulate doing work."""
        print_step(7, "Performing Work (Simulated)")
        
        self.think("Now I'll perform the actual work...")
        
        # Simulate work steps
        work_steps = [
            "Analyzing requirements from TaskCard",
            "Identifying affected files",
            "Implementing changes",
            "Running local tests",
            "Preparing output",
        ]
        
        for i, step in enumerate(work_steps, 1):
            print(f"      [{i}/{len(work_steps)}] {step}... ✓")
        
        self.speak("Work completed!")
        
        # Return simulated output
        return {
            "files_changed": ["example.py"],
            "tests_passed": True,
            "summary": "Implemented feature as requested",
        }
    
    def step8_terminate_session(self):
        """Step 8: Clean up and terminate session."""
        print_step(8, "Session Termination")
        
        self.think("I should properly terminate my session now.")
        
        result = self.call_tool("session_terminate", {
            "session_token": self.session_token,
            "reason": "work_complete",
        })
        
        if result.get("success"):
            self.speak("Session terminated successfully. Goodbye!")
        else:
            self.speak(f"Error terminating session: {result}")
        
        return result.get("success", False)
    
    # =========================================================================
    # Full Workflow
    # =========================================================================
    
    def run_full_workflow(self):
        """Run the complete agent workflow."""
        print_header(f"{self.name} AI Agent Workflow Simulation")
        
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"Agent: {self.name}")
        print(f"MCP Server: AI Workflow OS")
        
        try:
            # Step 1: Connect
            self.step1_connect()
            
            # Step 2: Create session
            if not self.step2_create_session():
                return False
            
            # Step 3: Explore
            self.step3_explore_workspace()
            
            # Step 4: Find tasks
            tasks = self.step4_find_tasks()
            
            # Step 5: Read a task (if available)
            if tasks:
                self.step5_read_task(tasks[0]["task_id"])
            
            # Step 6: Governance check
            self.step6_governance_check()
            
            # Step 7: Simulate work
            self.step7_simulate_work()
            
            # Step 8: Terminate
            self.step8_terminate_session()
            
            print_header("Workflow Complete")
            print("✅ All steps executed successfully")
            return True
            
        except Exception as e:
            print(f"\n❌ Error during workflow: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    # Create and run simulated agent
    agent = SimulatedAIAgent(agent_name="Claude")
    success = agent.run_full_workflow()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
