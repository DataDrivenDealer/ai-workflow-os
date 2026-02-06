#!/usr/bin/env python3
"""
PR Checklist Gate — Verify PR Checklist Before Merge

Purpose: Parse PR body for gate checklist and verify all items are checked.
         Integrates with GitHub Actions for automated gate enforcement.

Usage:
    python scripts/pr_checklist_gate.py --pr 15
    python scripts/pr_checklist_gate.py --pr 15 --strict
    python scripts/pr_checklist_gate.py --body-file pr_body.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Regex patterns for checklist items
CHECKLIST_PATTERN = re.compile(r"^\s*-\s*\[([ xX])\]\s*(.+)$", re.MULTILINE)

# Required gates for different PR types
REQUIRED_GATES = {
    "default": ["G2", "Review"],
    "feature": ["G1", "G2", "Review"],
    "experiment": ["G1", "G2", "G3", "Review"],
    "hotfix": ["G2"],
}

# Gate patterns to match in checklist text
GATE_PATTERNS = {
    "G1": re.compile(r"\*\*G1\*\*|G1[:\s]|data\s*quality", re.IGNORECASE),
    "G2": re.compile(r"\*\*G2\*\*|G2[:\s]|unit\s*tests?", re.IGNORECASE),
    "G3": re.compile(r"\*\*G3\*\*|G3[:\s]|performance", re.IGNORECASE),
    "G4": re.compile(r"\*\*G4\*\*|G4[:\s]|release", re.IGNORECASE),
    "Review": re.compile(r"\*\*Review\*\*|pair\s*review|review\s*completed", re.IGNORECASE),
}


@dataclass
class ChecklistItem:
    """A single checklist item."""
    text: str
    checked: bool
    gate_id: Optional[str] = None
    
    def __str__(self) -> str:
        check = "✅" if self.checked else "❌"
        gate = f"[{self.gate_id}]" if self.gate_id else ""
        return f"{check} {gate} {self.text}"


@dataclass
class ChecklistResult:
    """Result of checklist parsing."""
    items: list[ChecklistItem]
    total: int
    checked: int
    missing_gates: list[str]
    all_required_checked: bool
    
    @property
    def pass_rate(self) -> float:
        return self.checked / self.total if self.total > 0 else 0.0


def parse_checklist(body: str) -> list[ChecklistItem]:
    """Parse markdown checklist items from PR body."""
    items = []
    
    for match in CHECKLIST_PATTERN.finditer(body):
        checked = match.group(1).lower() == "x"
        text = match.group(2).strip()
        
        # Identify gate
        gate_id = None
        for gate, pattern in GATE_PATTERNS.items():
            if pattern.search(text):
                gate_id = gate
                break
        
        items.append(ChecklistItem(text=text, checked=checked, gate_id=gate_id))
    
    return items


def verify_checklist(
    items: list[ChecklistItem],
    required_gates: list[str],
) -> ChecklistResult:
    """Verify checklist against required gates."""
    # Count checked items
    checked = sum(1 for item in items if item.checked)
    
    # Find missing required gates
    found_gates = {item.gate_id for item in items if item.checked and item.gate_id}
    missing_gates = [g for g in required_gates if g not in found_gates]
    
    return ChecklistResult(
        items=items,
        total=len(items),
        checked=checked,
        missing_gates=missing_gates,
        all_required_checked=len(missing_gates) == 0,
    )


def get_pr_body(pr_number: int) -> tuple[bool, str]:
    """Fetch PR body from GitHub."""
    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(pr_number), "--json", "body,labels,title"],
            capture_output=True,
            text=True,
            cwd=_ROOT,
            check=False,
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return True, data.get("body", "")
        
        return False, result.stderr.strip()
    
    except FileNotFoundError:
        return False, "GitHub CLI (gh) not installed"
    except json.JSONDecodeError:
        return False, "Failed to parse PR data"
    except Exception as e:
        return False, str(e)


def detect_pr_type(body: str, title: str = "") -> str:
    """Detect PR type from body/title for required gate selection."""
    combined = f"{title} {body}".lower()
    
    if "hotfix" in combined or "urgent" in combined:
        return "hotfix"
    if "experiment" in combined or "exp/" in combined:
        return "experiment"
    if "feature" in combined or "feat/" in combined:
        return "feature"
    
    return "default"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify PR checklist gates before merge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check PR #15 against required gates
    python scripts/pr_checklist_gate.py --pr 15
    
    # Strict mode: fail if any checklist item is unchecked
    python scripts/pr_checklist_gate.py --pr 15 --strict
    
    # Check from file (for testing)
    python scripts/pr_checklist_gate.py --body-file pr_body.md
    
    # Override required gates
    python scripts/pr_checklist_gate.py --pr 15 --require G1 G2 Review
        """
    )
    
    parser.add_argument("--pr", type=int, help="PR number to check")
    parser.add_argument("--body-file", type=Path, help="Read body from file instead")
    parser.add_argument("--strict", action="store_true", 
                        help="Fail if any checklist item is unchecked")
    parser.add_argument("--require", nargs="*", 
                        help="Override required gates (e.g., G1 G2 Review)")
    parser.add_argument("--pr-type", choices=["default", "feature", "experiment", "hotfix"],
                        help="Override PR type for gate selection")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    # Get PR body
    if args.body_file:
        if not args.body_file.exists():
            print(f"❌ File not found: {args.body_file}")
            return 1
        body = args.body_file.read_text(encoding="utf-8")
        title = ""
    elif args.pr:
        success, body = get_pr_body(args.pr)
        if not success:
            print(f"❌ Failed to fetch PR #{args.pr}: {body}")
            return 1
        title = ""
    else:
        parser.print_help()
        return 1
    
    # Parse checklist
    items = parse_checklist(body)
    
    if not items:
        if args.json:
            print(json.dumps({"error": "No checklist found", "pass": False}))
        else:
            print("⚠️  No checklist items found in PR body")
            print("   Expected format: - [ ] or - [x]")
        return 1
    
    # Determine required gates
    if args.require:
        required_gates = args.require
    else:
        pr_type = args.pr_type or detect_pr_type(body, title)
        required_gates = REQUIRED_GATES.get(pr_type, REQUIRED_GATES["default"])
    
    # Verify
    result = verify_checklist(items, required_gates)
    
    # Determine pass/fail
    if args.strict:
        passed = result.checked == result.total
    else:
        passed = result.all_required_checked
    
    # Output
    if args.json:
        output = {
            "pass": passed,
            "total": result.total,
            "checked": result.checked,
            "pass_rate": result.pass_rate,
            "missing_gates": result.missing_gates,
            "required_gates": required_gates,
            "items": [
                {"text": item.text, "checked": item.checked, "gate": item.gate_id}
                for item in result.items
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        pr_label = f"PR #{args.pr}" if args.pr else "Checklist"
        print(f"{'✅' if passed else '❌'} {pr_label} Gate Check")
        print()
        print(f"📋 Checklist: {result.checked}/{result.total} checked ({result.pass_rate:.0%})")
        print(f"🎯 Required: {', '.join(required_gates)}")
        print()
        
        # Show items
        print("Items:")
        for item in result.items:
            print(f"   {item}")
        
        if result.missing_gates:
            print()
            print(f"❌ Missing Required Gates: {', '.join(result.missing_gates)}")
        
        if not passed:
            print()
            if args.strict:
                print("⛔ STRICT MODE: All checklist items must be checked")
            else:
                print("⛔ Required gates must be checked before merge")
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
