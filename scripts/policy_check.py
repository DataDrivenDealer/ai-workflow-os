from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Set

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "spec_registry.yaml"

CANON_PREFIX = Path("specs") / "canon"
FRAMEWORK_PREFIX = Path("specs") / "framework"
PROJECT_SPECS_PREFIX = Path("projects")

PROPOSAL_DIRS = [Path("ops") / "proposals", Path("ops") / "decision-log"]
DEVIATION_DIRS = [Path("ops") / "deviations", Path("ops") / "decision-log"]


def run_git(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def get_changed_files(mode: str) -> Set[Path]:
    if mode == "precommit":
        result = run_git(["git", "diff", "--cached", "--name-only"])
    else:
        result = run_git(["git", "diff", "--name-only", "origin/main...HEAD"])
        if result.returncode != 0:
            result = run_git(["git", "diff", "--name-only", "HEAD~1...HEAD"])

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")

    return {Path(line.strip()) for line in result.stdout.splitlines() if line.strip()}


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise RuntimeError("spec_registry.yaml not found")

    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def has_artifacts(dirs: List[Path]) -> bool:
    for relative in dirs:
        target = ROOT / relative
        if not target.exists():
            continue
        for path in target.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                return True
    return False


def check_proposals(changed_files: Set[Path], registry: dict) -> List[str]:
    messages: List[str] = []
    touches_canon = any(path.is_relative_to(CANON_PREFIX) for path in changed_files)
    touches_framework = any(path.is_relative_to(FRAMEWORK_PREFIX) for path in changed_files)

    if not (touches_canon or touches_framework):
        return messages

    proposals = registry.get("proposals", [])
    proposal_files = has_artifacts(PROPOSAL_DIRS)

    if not proposals and not proposal_files:
        messages.append(
            "Blocking change: L0/L1 specs modified without a proposal. "
            "Add a proposal entry in spec_registry.yaml or a file under ops/proposals/ or ops/decision-log/."
        )
    return messages


def check_deviations(changed_files: Set[Path], registry: dict) -> List[str]:
    messages: List[str] = []
    touches_project_specs = any(
        len(path.parts) > 1 and path.parts[0] == PROJECT_SPECS_PREFIX.parts[0] and "specs" in path.parts
        for path in changed_files
    )

    if not touches_project_specs:
        return messages

    deviations = registry.get("deviations", [])
    deviation_files = has_artifacts(DEVIATION_DIRS)

    if not deviations and not deviation_files:
        messages.append(
            "Warning: project spec changes detected but no deviations declared. "
            "Add a deviation entry in spec_registry.yaml or a file under ops/deviations/ or ops/decision-log/."
        )
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Spec registry policy gate")
    parser.add_argument("--mode", choices=["precommit", "prepush", "ci"], default="precommit")
    args = parser.parse_args()

    changed_files = get_changed_files("precommit" if args.mode == "precommit" else "ci")
    registry = load_registry()

    errors = check_proposals(changed_files, registry)
    warnings = check_deviations(changed_files, registry)

    for warning in warnings:
        print(warning)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("Policy check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
