"""
TaskCard Gate Validator

Validates that a TaskCard meets the gate requirements for its stage.
Reads the TaskCard markdown and checks gate completion status.

Usage:
    python scripts/taskcard_gate_validator.py tasks/TASK_DATA_2_001.md
    python scripts/taskcard_gate_validator.py --stage 2 --task-id DATA_2_001
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add kernel to path for imports
sys.path.insert(0, str(Path(__file__).parents[1]))
from kernel.paths import ROOT


@dataclass
class GateCheckItem:
    """A single gate check item from TaskCard."""
    name: str
    status: str  # pending, passed, failed
    actual_value: Optional[str] = None
    threshold: Optional[str] = None


@dataclass
class TaskCardGateInfo:
    """Extracted gate information from TaskCard."""
    task_id: str
    stage: int
    gate_id: Optional[str]
    checks: List[GateCheckItem]
    gate_result: Optional[str]  # PASS, FAIL, None


class TaskCardParser:
    """Parse TaskCard markdown to extract gate information."""
    
    STAGE_PATTERN = re.compile(r'Stage[:\s]*(\d+)', re.IGNORECASE)
    GATE_PATTERN = re.compile(r'Gate\s+(G\d+)', re.IGNORECASE)
    CHECK_ROW_PATTERN = re.compile(
        r'\|\s*([^|]+)\s*\|\s*`?(pending|passed|failed|✅|❌|⚠️)`?\s*\|',
        re.IGNORECASE
    )
    GATE_RESULT_PATTERN = re.compile(
        r'\[([xX ])\]\s*\*\*(PASS|FAIL|CONDITIONAL)',
        re.IGNORECASE
    )
    
    def parse(self, content: str, task_id: str) -> TaskCardGateInfo:
        """Parse TaskCard content and extract gate information."""
        
        # Extract stage
        stage_match = self.STAGE_PATTERN.search(content)
        stage = int(stage_match.group(1)) if stage_match else 0
        
        # Extract gate ID
        gate_match = self.GATE_PATTERN.search(content)
        gate_id = gate_match.group(1) if gate_match else None
        
        # Extract check items
        checks = []
        for match in self.CHECK_ROW_PATTERN.finditer(content):
            name = match.group(1).strip()
            status_raw = match.group(2).strip().lower()
            
            # Normalize status
            if status_raw in ('✅', 'passed'):
                status = 'passed'
            elif status_raw in ('❌', 'failed'):
                status = 'failed'
            else:
                status = 'pending'
            
            checks.append(GateCheckItem(name=name, status=status))
        
        # Extract gate result
        gate_result = None
        for match in self.GATE_RESULT_PATTERN.finditer(content):
            if match.group(1).lower() == 'x':
                gate_result = match.group(2).upper()
                break
        
        return TaskCardGateInfo(
            task_id=task_id,
            stage=stage,
            gate_id=gate_id,
            checks=checks,
            gate_result=gate_result,
        )


def validate_taskcard_gate(taskcard_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate TaskCard gate completion.
    
    Returns:
        (passed, messages)
    """
    if not taskcard_path.exists():
        return False, [f"TaskCard not found: {taskcard_path}"]
    
    with taskcard_path.open('r', encoding='utf-8') as f:
        content = f.read()
    
    task_id = taskcard_path.stem
    parser = TaskCardParser()
    info = parser.parse(content, task_id)
    
    messages = []
    
    # Check if gate is applicable
    stage_gate_map = {
        2: "G1",
        3: "G2",
        4: "G3",
        5: "G4",
        6: "G5",
    }
    
    expected_gate = stage_gate_map.get(info.stage)
    if not expected_gate:
        messages.append(f"ℹ️ Stage {info.stage} has no required gate")
        return True, messages
    
    if info.gate_id and info.gate_id != expected_gate:
        messages.append(f"⚠️ Gate mismatch: TaskCard has {info.gate_id}, expected {expected_gate}")
    
    # Check individual items
    pending_count = sum(1 for c in info.checks if c.status == 'pending')
    failed_count = sum(1 for c in info.checks if c.status == 'failed')
    passed_count = sum(1 for c in info.checks if c.status == 'passed')
    
    messages.append(f"Gate {expected_gate}: {passed_count} passed, {pending_count} pending, {failed_count} failed")
    
    for check in info.checks:
        if check.status == 'failed':
            messages.append(f"  ❌ {check.name}")
        elif check.status == 'pending':
            messages.append(f"  ⏳ {check.name}")
    
    # Determine pass/fail
    if info.gate_result == 'PASS':
        messages.append(f"✅ Gate {expected_gate} marked as PASSED")
        return True, messages
    elif info.gate_result == 'FAIL':
        messages.append(f"❌ Gate {expected_gate} marked as FAILED")
        return False, messages
    elif failed_count > 0:
        messages.append(f"❌ Gate {expected_gate} has {failed_count} failed checks")
        return False, messages
    elif pending_count > 0:
        messages.append(f"⏳ Gate {expected_gate} has {pending_count} pending checks")
        return False, messages
    else:
        messages.append(f"✅ All {passed_count} checks passed")
        return True, messages


def find_taskcard(task_id: str) -> Optional[Path]:
    """Find TaskCard by ID."""
    # Direct path
    direct = ROOT / "tasks" / f"{task_id}.md"
    if direct.exists():
        return direct
    
    # Search in subdirectories
    for subdir in ["inbox", "running", "done"]:
        path = ROOT / "tasks" / subdir / f"{task_id}.md"
        if path.exists():
            return path
    
    # Search by pattern
    for path in (ROOT / "tasks").rglob(f"*{task_id}*.md"):
        return path
    
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="TaskCard Gate Validator")
    parser.add_argument("taskcard", nargs="?", help="Path to TaskCard file")
    parser.add_argument("--task-id", help="Task ID to find")
    parser.add_argument("--stage", type=int, help="Expected stage (for validation)")
    args = parser.parse_args()
    
    if args.taskcard:
        taskcard_path = Path(args.taskcard)
    elif args.task_id:
        taskcard_path = find_taskcard(args.task_id)
        if not taskcard_path:
            print(f"❌ TaskCard not found for: {args.task_id}")
            return 1
    else:
        parser.print_help()
        return 1
    
    passed, messages = validate_taskcard_gate(taskcard_path)
    
    print(f"\n{'='*60}")
    print(f"TaskCard Gate Validation: {taskcard_path.name}")
    print(f"{'='*60}\n")
    
    for msg in messages:
        print(msg)
    
    print(f"\n{'='*60}")
    if passed:
        print("✅ GATE VALIDATION PASSED")
    else:
        print("❌ GATE VALIDATION FAILED")
    print(f"{'='*60}\n")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
