#!/usr/bin/env python3
"""
Verify architectural dependency direction.

Ensures kernel/ does NOT import from projects/ (single-direction dependency).
AI Workflow OS is infrastructure; application projects must not leak back.

Exit codes:
  0 - no reverse dependencies
  1 - violations found
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT_DIR / "kernel"
PROJECTS_DIR = ROOT_DIR / "projects"


def extract_imports(file_path: Path) -> Set[str]:
    """
    Extract all import module names from a Python file using AST.
    
    Returns:
        Set of top-level module names (e.g., 'projects', 'projects.dgsf')
    """
    try:
        tree = ast.parse(file_path.read_text(encoding='utf-8'), filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"[WARN] Cannot parse {file_path}: {e}", file=sys.stderr)
        return set()
    
    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    
    return imports


def main() -> int:
    """
    Check for reverse dependencies (kernel/ → projects/).
    
    Returns:
        0 if no violations, 1 if violations found
    """
    if not KERNEL_DIR.exists():
        print("❌ kernel/ directory not found", file=sys.stderr)
        return 1
    
    violations: List[dict] = []
    
    # Scan all Python files in kernel/
    for py_file in KERNEL_DIR.rglob("*.py"):
        # Skip __pycache__ and other generated files
        if "__pycache__" in py_file.parts or py_file.name.startswith("."):
            continue
        
        imports = extract_imports(py_file)
        
        # Check for imports starting with 'projects.'
        for imp in imports:
            if imp.startswith("projects.") or imp == "projects":
                violations.append({
                    "file": str(py_file.relative_to(ROOT_DIR)),
                    "import": imp,
                    "line": None,  # AST doesn't easily provide line numbers
                })
    
    # Report results
    if violations:
        print(f"❌ Found {len(violations)} architectural boundary violations")
        print("   (kernel/ must NOT import from projects/)")
        print()
        for v in violations:
            print(f"  {v['file']}")
            print(f"    → import {v['import']}")
        return 1
    
    print("✅ No reverse dependencies detected")
    print("   (kernel/ → projects/ boundary is clean)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
