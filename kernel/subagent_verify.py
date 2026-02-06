"""
Subagent Verification Module for Copilot Runtime OS.

Enforces subagent invocation requirements defined in execution queue tasks.
Gate-E0: Pre-execution check for required subagents.

Version: 1.0.0
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add kernel to path for imports
_KERNEL_DIR = Path(__file__).parent
_ROOT = _KERNEL_DIR.parent
sys.path.insert(0, str(_ROOT))

try:
    from kernel.paths import ROOT
except ImportError:
    ROOT = _ROOT


@dataclass
class SubagentRequirement:
    """A required subagent for a task."""
    subagent_id: str
    required: bool = True
    output_path: Optional[str] = None
    skip_justification: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of subagent requirement verification."""
    task_id: str
    all_satisfied: bool
    requirements: List[Dict[str, Any]]
    missing: List[str]
    message: str


def load_execution_queue() -> Optional[Dict[str, Any]]:
    """Load the execution queue from state file.
    
    Returns:
        Queue data if exists, None otherwise.
    """
    queue_path = ROOT / "state" / "execution_queue.yaml"
    if not queue_path.exists():
        return None
    
    with open(queue_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_subagent_output_dir() -> Path:
    """Get the directory where subagent outputs are stored."""
    return ROOT / "docs" / "subagents" / "runs"


def check_subagent_output_exists(subagent_id: str, task_id: str) -> Tuple[bool, Optional[str]]:
    """Check if a subagent has produced output for a task.
    
    Args:
        subagent_id: The subagent identifier.
        task_id: The task identifier.
    
    Returns:
        Tuple of (exists: bool, path: Optional[str])
    """
    output_dir = get_subagent_output_dir()
    
    if not output_dir.exists():
        return False, None
    
    # Look for any output directory matching the subagent
    # Format: {timestamp}_{subagent_id}/
    for entry in output_dir.iterdir():
        if entry.is_dir() and subagent_id in entry.name:
            summary_path = entry / "SUMMARY.md"
            if summary_path.exists():
                return True, str(entry.relative_to(ROOT))
    
    return False, None


def verify_task_subagents(task: Dict[str, Any]) -> VerificationResult:
    """Verify that all required subagents have been invoked for a task.
    
    Args:
        task: The task configuration from execution queue.
    
    Returns:
        VerificationResult indicating if all requirements are satisfied.
    """
    task_id = task.get("id", task.get("subtask_id", "unknown"))
    required_subagents = task.get("required_subagents", [])
    subagent_artifacts = task.get("subagent_artifacts", [])
    
    if not required_subagents:
        return VerificationResult(
            task_id=task_id,
            all_satisfied=True,
            requirements=[],
            missing=[],
            message="No subagent requirements for this task",
        )
    
    requirements = []
    missing = []
    
    for subagent_id in required_subagents:
        # Check if there's an artifact reference
        artifact = next(
            (a for a in subagent_artifacts if a.get("subagent_id") == subagent_id),
            None
        )
        
        if artifact and artifact.get("output_path"):
            # Has artifact reference - verify it exists
            output_path = ROOT / artifact["output_path"]
            summary_path = output_path / "SUMMARY.md" if output_path.is_dir() else output_path
            
            if summary_path.exists():
                requirements.append({
                    "subagent_id": subagent_id,
                    "status": "satisfied",
                    "output_path": artifact["output_path"],
                })
            else:
                requirements.append({
                    "subagent_id": subagent_id,
                    "status": "missing_output",
                    "expected_path": artifact["output_path"],
                })
                missing.append(subagent_id)
        else:
            # No artifact reference - check by convention
            exists, path = check_subagent_output_exists(subagent_id, task_id)
            
            if exists:
                requirements.append({
                    "subagent_id": subagent_id,
                    "status": "found_by_scan",
                    "output_path": path,
                })
            else:
                # Check for skip justification
                skip = task.get("skip_justifications", {}).get(subagent_id)
                if skip:
                    requirements.append({
                        "subagent_id": subagent_id,
                        "status": "skipped",
                        "skip_justification": skip,
                    })
                else:
                    requirements.append({
                        "subagent_id": subagent_id,
                        "status": "not_invoked",
                    })
                    missing.append(subagent_id)
    
    all_satisfied = len(missing) == 0
    
    if all_satisfied:
        message = f"All {len(required_subagents)} required subagents satisfied"
    else:
        message = f"Missing {len(missing)}/{len(required_subagents)} subagents: {', '.join(missing)}"
    
    return VerificationResult(
        task_id=task_id,
        all_satisfied=all_satisfied,
        requirements=requirements,
        missing=missing,
        message=message,
    )


def verify_queue_entry(entry_index: int = 0) -> VerificationResult:
    """Verify subagent requirements for a queue entry.
    
    Args:
        entry_index: Index of the queue entry (0 = next pending).
    
    Returns:
        VerificationResult for the entry.
    """
    queue = load_execution_queue()
    
    if not queue:
        return VerificationResult(
            task_id="none",
            all_satisfied=False,
            requirements=[],
            missing=[],
            message="No execution queue found",
        )
    
    items = queue.get("items", [])
    pending = [i for i in items if i.get("status") == "pending"]
    
    if not pending:
        return VerificationResult(
            task_id="none",
            all_satisfied=True,
            requirements=[],
            missing=[],
            message="No pending items in queue",
        )
    
    if entry_index >= len(pending):
        return VerificationResult(
            task_id="none",
            all_satisfied=False,
            requirements=[],
            missing=[],
            message=f"Entry index {entry_index} out of range (max: {len(pending)-1})",
        )
    
    task = pending[entry_index]
    return verify_task_subagents(task)


def gate_e0_check() -> Dict[str, Any]:
    """Execute Gate-E0: Pre-execution subagent check.
    
    Returns:
        Dict with 'passed' (bool), 'result' (VerificationResult), 'action' (str).
    """
    result = verify_queue_entry(0)
    
    if result.all_satisfied:
        return {
            "passed": True,
            "result": result,
            "action": "proceed",
            "message": f"✅ Gate-E0 PASSED: {result.message}",
        }
    else:
        return {
            "passed": False,
            "result": result,
            "action": "block",
            "message": f"❌ Gate-E0 BLOCKED: {result.message}",
            "missing_subagents": result.missing,
        }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for subagent verification."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Subagent Verification")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify subagent requirements")
    verify_parser.add_argument("--index", type=int, default=0, help="Queue entry index")
    verify_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # gate command
    gate_parser = subparsers.add_parser("gate", help="Run Gate-E0 check")
    gate_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # list command
    subparsers.add_parser("list", help="List all subagent requirements in queue")
    
    args = parser.parse_args()
    
    if args.command == "verify":
        result = verify_queue_entry(args.index)
        
        if args.json:
            print(json.dumps({
                "task_id": result.task_id,
                "all_satisfied": result.all_satisfied,
                "requirements": result.requirements,
                "missing": result.missing,
                "message": result.message,
            }, indent=2))
        else:
            icon = "✅" if result.all_satisfied else "❌"
            print(f"{icon} Task: {result.task_id}")
            print(f"   Status: {'SATISFIED' if result.all_satisfied else 'MISSING REQUIREMENTS'}")
            print(f"   {result.message}\n")
            
            for req in result.requirements:
                status = req["status"]
                if status == "satisfied":
                    print(f"   ✅ {req['subagent_id']}: {req.get('output_path', 'found')}")
                elif status == "found_by_scan":
                    print(f"   ✅ {req['subagent_id']}: {req.get('output_path', 'found')} (by scan)")
                elif status == "skipped":
                    print(f"   ⏭️ {req['subagent_id']}: skipped - {req.get('skip_justification', 'no reason')}")
                else:
                    print(f"   ❌ {req['subagent_id']}: not invoked")
        
        sys.exit(0 if result.all_satisfied else 1)
    
    elif args.command == "gate":
        check = gate_e0_check()
        
        if args.json:
            print(json.dumps({
                "passed": check["passed"],
                "action": check["action"],
                "message": check["message"],
                "missing_subagents": check.get("missing_subagents", []),
            }, indent=2))
        else:
            print(check["message"])
            if not check["passed"]:
                print(f"\n   Required actions:")
                for subagent_id in check.get("missing_subagents", []):
                    print(f"   - Invoke: {subagent_id}")
        
        sys.exit(0 if check["passed"] else 1)
    
    elif args.command == "list":
        queue = load_execution_queue()
        
        if not queue:
            print("📭 No execution queue found")
            return
        
        items = queue.get("items", [])
        print(f"📋 Subagent Requirements in Queue ({len(items)} items):\n")
        
        for i, item in enumerate(items):
            task_id = item.get("id", item.get("subtask_id", f"item-{i}"))
            status = item.get("status", "unknown")
            required = item.get("required_subagents", [])
            
            status_icon = {"pending": "⏸️", "in-progress": "▶️", "completed": "✅"}.get(status, "❓")
            print(f"  {status_icon} [{i}] {task_id}")
            
            if required:
                print(f"      Required: {', '.join(required)}")
            else:
                print(f"      Required: (none)")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
