from __future__ import annotations

import ast
import sys
from pathlib import Path


def _get_local_modules() -> set[str]:
    kernel_dir = Path(__file__).resolve().parents[1]
    modules = set()
    for py_file in kernel_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        modules.add(py_file.stem)
    return modules


def test_kernel_imports_are_absolute() -> None:
    """验证所有kernel模块使用绝对导入（kernel.*）"""
    kernel_dir = Path(__file__).resolve().parents[1]
    local_modules = _get_local_modules()
    stdlib_modules = getattr(sys, "stdlib_module_names", set()) | set(sys.builtin_module_names)

    for py_file in kernel_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is None:
                    assert False, f"{py_file}: relative import is not allowed"

                if node.module.startswith("__future__"):
                    continue

                if node.module in local_modules and node.module not in stdlib_modules:
                    assert False, f"{py_file}: use 'kernel.{node.module}' instead of '{node.module}'"

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in local_modules and alias.name not in stdlib_modules:
                        assert False, f"{py_file}: use 'kernel.{alias.name}' instead of '{alias.name}'"
