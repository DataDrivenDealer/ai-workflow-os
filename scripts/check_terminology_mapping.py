#!/usr/bin/env python3
"""
Terminology Mapping Checker

Verifies that terms defined in Canon specs have corresponding implementations.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# 术语定义：{术语: (定义位置, 预期实现模式)}
CANONICAL_TERMS: Dict[str, Tuple[str, str]] = {
    "RoleMode": ("ROLE_MODE_CANON", r"class RoleMode|enum RoleMode"),
    "AgentSession": ("AGENT_SESSION", r"class AgentSession"),
    "GovernanceGate": ("GOVERNANCE_INVARIANTS", r"class GovernanceGate"),
    "Freeze": ("GOVERNANCE_INVARIANTS", r"def freeze_artifact|class FreezeRecord"),
    "Acceptance": ("GOVERNANCE_INVARIANTS", r"def accept_artifact|class AcceptanceRecord"),
    "Artifact Lock": ("AGENT_SESSION", r"locked_artifacts|lock_artifact"),
    "Authority": ("GOVERNANCE_INVARIANTS", r"class Authority|authority"),
    "SessionState": ("AGENT_SESSION", r"class SessionState|enum SessionState"),
    "TaskState": ("STATE_MACHINE", r"states:|TaskStatus"),
}


def search_term_in_code(term: str, pattern: str) -> List[Path]:
    """Search for term pattern in kernel/ code."""
    found_files = []
    kernel_dir = ROOT / "kernel"
    
    for py_file in kernel_dir.glob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if re.search(pattern, content, re.IGNORECASE):
                found_files.append(py_file)
        except Exception:
            continue
    
    return found_files


def check_terminology_mapping() -> bool:
    """Check all canonical terms have implementations."""
    print("Terminology Mapping Report")
    print("==========================\n")
    
    all_found = True
    found_count = 0
    missing_count = 0
    
    for term, (spec, pattern) in CANONICAL_TERMS.items():
        found_files = search_term_in_code(term, pattern)
        
        if found_files:
            print(f"✅ {term}: FOUND")
            for f in found_files:
                print(f"   → {f.relative_to(ROOT)}")
            found_count += 1
        else:
            print(f"❌ {term}: NOT FOUND")
            print(f"   Defined in: {spec}")
            print(f"   Expected pattern: {pattern}")
            all_found = False
            missing_count += 1
        print()
    
    print("\n" + "="*60)
    print(f"Summary: {found_count} found, {missing_count} missing")
    print("="*60)
    
    return all_found


if __name__ == "__main__":
    success = check_terminology_mapping()
    sys.exit(0 if success else 1)
