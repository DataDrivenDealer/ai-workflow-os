#!/usr/bin/env python
"""
Review Gate Check Script.

Enforces Gate-E4.5: "NO REVIEW, NO RUN"
Verifies that a pair review artifact exists before allowing test/backtest execution.

Usage:
    python scripts/check_review_gate.py --task-id TASK_ID
    python scripts/check_review_gate.py --auto  # Infer from branch name
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def get_current_branch() -> str:
    """Get the current Git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return ""


def extract_task_id_from_branch(branch: str) -> Optional[str]:
    """Extract task ID from branch name.
    
    Patterns:
    - feature/TASK-001-description -> TASK-001
    - SDF_FEATURE_ENG_001-something -> SDF_FEATURE_ENG_001
    - t05-validation -> T05
    """
    patterns = [
        r'([A-Z][A-Z0-9_-]+(?:_[A-Z0-9]+)+)',  # SDF_FEATURE_ENG_001
        r'([A-Z]+-\d+)',                         # TASK-001
        r'(t\d+)',                               # t05
    ]
    
    for pattern in patterns:
        match = re.search(pattern, branch, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None


def find_review_artifact(task_id: str) -> Tuple[bool, Optional[Path], Optional[str]]:
    """Find a review artifact for a task.
    
    Looks for docs/reviews/{task_id}/REVIEW_{N}.md where N >= 2.
    
    Returns:
        Tuple of (found: bool, path: Optional[Path], verdict: Optional[str])
    """
    review_dir = ROOT / "docs" / "reviews" / task_id
    
    if not review_dir.exists():
        return False, None, None
    
    # Look for REVIEW_2.md or higher (REVIEW_1 is initial, REVIEW_2+ is after patch)
    for n in range(10, 1, -1):
        review_file = review_dir / f"REVIEW_{n}.md"
        if review_file.exists():
            # Try to extract verdict
            content = review_file.read_text(encoding="utf-8")
            verdict_match = re.search(r'verdict:\s*["\']?(\w+)["\']?', content, re.IGNORECASE)
            verdict = verdict_match.group(1) if verdict_match else "UNKNOWN"
            return True, review_file, verdict
    
    return False, None, None


def check_review_gate(task_id: str) -> dict:
    """Check if the review gate is satisfied for a task.
    
    Args:
        task_id: The task identifier.
    
    Returns:
        Dict with 'passed', 'message', 'artifact_path', 'verdict'.
    """
    found, path, verdict = find_review_artifact(task_id)
    
    if not found:
        return {
            "passed": False,
            "message": f"No review artifact found for task {task_id}",
            "expected_path": f"docs/reviews/{task_id}/REVIEW_2.md",
            "artifact_path": None,
            "verdict": None,
        }
    
    if verdict and verdict.upper() == "APPROVED":
        return {
            "passed": True,
            "message": f"Review approved: {path.name}",
            "artifact_path": str(path.relative_to(ROOT)),
            "verdict": verdict,
        }
    elif verdict and verdict.upper() in ("REJECTED", "NEEDS_WORK"):
        return {
            "passed": False,
            "message": f"Review not approved (verdict: {verdict})",
            "artifact_path": str(path.relative_to(ROOT)),
            "verdict": verdict,
        }
    else:
        # Unknown verdict - warn but allow
        return {
            "passed": True,  # Allow with warning
            "message": f"Review found but verdict unclear: {verdict}",
            "artifact_path": str(path.relative_to(ROOT)),
            "verdict": verdict,
            "warning": True,
        }


def main():
    """CLI interface for review gate check."""
    parser = argparse.ArgumentParser(description="Check Review Gate (Gate-E4.5)")
    parser.add_argument("--task-id", help="Task ID to check")
    parser.add_argument("--auto", action="store_true", help="Auto-detect from branch")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Determine task ID
    task_id = args.task_id
    
    if not task_id and args.auto:
        branch = get_current_branch()
        task_id = extract_task_id_from_branch(branch)
        
        if not task_id:
            if not args.quiet:
                print("⚠️ Could not infer task ID from branch:", branch)
                print("   Use --task-id to specify explicitly")
            sys.exit(2)
    
    if not task_id:
        # Check environment variable
        task_id = os.environ.get("CURRENT_TASK_ID")
    
    if not task_id:
        if not args.quiet:
            print("❌ No task ID provided")
            print("   Use --task-id TASK_ID or --auto")
        sys.exit(2)
    
    # Run check
    result = check_review_gate(task_id)
    
    if result["passed"]:
        if result.get("warning"):
            if args.strict:
                if not args.quiet:
                    print(f"❌ Gate-E4.5 FAILED (strict): {result['message']}")
                sys.exit(1)
            else:
                if not args.quiet:
                    print(f"⚠️ Gate-E4.5 WARNING: {result['message']}")
                    print(f"   Artifact: {result.get('artifact_path', 'N/A')}")
                sys.exit(0)
        else:
            if not args.quiet:
                print(f"✅ Gate-E4.5 PASSED: {result['message']}")
                print(f"   Artifact: {result.get('artifact_path', 'N/A')}")
            sys.exit(0)
    else:
        if not args.quiet:
            print(f"❌ Gate-E4.5 FAILED: {result['message']}")
            if result.get("expected_path"):
                print(f"   Expected: {result['expected_path']}")
            print()
            print("   To satisfy this gate:")
            print("   1. Run /dgsf_pair_review skill")
            print("   2. Complete Coder ↔ Reviewer loop")
            print("   3. Ensure REVIEW_2.md has verdict: APPROVED")
        sys.exit(1)


if __name__ == "__main__":
    main()
