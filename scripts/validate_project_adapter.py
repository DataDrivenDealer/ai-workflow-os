#!/usr/bin/env python3
"""
Project Adapter Validator

Validates that a project's adapter.yaml correctly implements
the interface contract defined in configs/project_interface.yaml.

Part of AEP-1: Kernel-Project Decoupling

Usage:
    python scripts/validate_project_adapter.py dgsf
    python scripts/validate_project_adapter.py --all
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file safely."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_all_project_ids() -> List[str]:
    """Discover all project IDs from projects/ directory."""
    projects_dir = PROJECT_ROOT / "projects"
    if not projects_dir.exists():
        return []
    
    return [
        d.name for d in projects_dir.iterdir() 
        if d.is_dir() and (d / "adapter.yaml").exists()
    ]


def validate_adapter(project_id: str) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a project's adapter against the interface contract.
    
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    # Load interface contract
    interface_path = PROJECT_ROOT / "configs" / "project_interface.yaml"
    if not interface_path.exists():
        errors.append(f"Interface contract not found: {interface_path}")
        return False, errors, warnings
    
    interface = load_yaml_file(interface_path)
    
    # Load adapter
    adapter_path = PROJECT_ROOT / "projects" / project_id / "adapter.yaml"
    if not adapter_path.exists():
        errors.append(f"Adapter not found: {adapter_path}")
        return False, errors, warnings
    
    adapter = load_yaml_file(adapter_path)
    
    # ==========================================================================
    # Validate Identity
    # ==========================================================================
    identity = adapter.get("identity", {})
    
    if not identity.get("project_id"):
        errors.append("Missing required: identity.project_id")
    elif identity.get("project_id") != project_id:
        errors.append(
            f"identity.project_id mismatch: "
            f"expected '{project_id}', got '{identity.get('project_id')}'"
        )
    
    if not identity.get("project_name"):
        errors.append("Missing required: identity.project_name")
    
    if not identity.get("kernel_version"):
        errors.append("Missing required: identity.kernel_version")
    
    # ==========================================================================
    # Validate Thresholds
    # ==========================================================================
    thresholds = adapter.get("thresholds", {})
    
    if not thresholds.get("primary_metrics"):
        errors.append("Missing required: thresholds.primary_metrics")
    else:
        for metric_name, metric_def in thresholds["primary_metrics"].items():
            if "operator" not in metric_def:
                errors.append(f"thresholds.primary_metrics.{metric_name} missing 'operator'")
            if "value" not in metric_def:
                errors.append(f"thresholds.primary_metrics.{metric_name} missing 'value'")
    
    # ==========================================================================
    # Validate Paths
    # ==========================================================================
    paths = adapter.get("paths", {})
    
    required_paths = ["source", "tests", "experiments", "data_safe", "data_protected"]
    for path_key in required_paths:
        if path_key not in paths:
            errors.append(f"Missing required path: paths.{path_key}")
        else:
            # Check if path exists (relative to project directory)
            full_path = PROJECT_ROOT / "projects" / project_id / paths[path_key]
            if not full_path.exists():
                warnings.append(f"Path does not exist: paths.{path_key} -> {full_path}")
    
    # ==========================================================================
    # Validate Skills
    # ==========================================================================
    skills = adapter.get("skills", {})
    
    if not skills.get("prefix"):
        errors.append("Missing required: skills.prefix")
    
    # Check prompt files exist
    prompt_files = skills.get("prompt_files", {})
    required_skills = [
        "research", "plan", "execute", "verify", "diagnose",
        "abort", "decision_log", "state_update", "research_summary",
        "repo_scan", "git_ops"
    ]
    
    for skill in required_skills:
        if skill not in prompt_files:
            warnings.append(f"Missing skill prompt mapping: skills.prompt_files.{skill}")
        else:
            prompt_path = PROJECT_ROOT / prompt_files[skill]
            if not prompt_path.exists():
                errors.append(f"Prompt file not found: {prompt_files[skill]}")
    
    # ==========================================================================
    # Validate Rules
    # ==========================================================================
    rules = adapter.get("rules", {})
    
    if not rules.get("inherit_kernel_rules"):
        warnings.append("rules.inherit_kernel_rules not specified (will default to 'all')")
    
    # ==========================================================================
    # Validate Behavior
    # ==========================================================================
    behavior = adapter.get("behavior", {})
    
    if not behavior.get("scope_pattern"):
        errors.append("Missing required: behavior.scope_pattern")
    
    # ==========================================================================
    # Validate Experiment Format (optional but recommended)
    # ==========================================================================
    experiment = adapter.get("experiment", {})
    
    if not experiment.get("naming_pattern"):
        warnings.append("experiment.naming_pattern not specified")
    
    if not experiment.get("required_artifacts"):
        warnings.append("experiment.required_artifacts not specified")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate project adapter against interface contract"
    )
    parser.add_argument(
        "project_id",
        nargs="?",
        help="Project ID to validate (e.g., 'dgsf')"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Validate all projects"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    if args.all:
        project_ids = get_all_project_ids()
        if not project_ids:
            print("No projects found with adapter.yaml")
            sys.exit(1)
    elif args.project_id:
        project_ids = [args.project_id]
    else:
        parser.print_help()
        sys.exit(1)
    
    all_valid = True
    
    for project_id in project_ids:
        print(f"\n{'='*60}")
        print(f"Validating: {project_id}")
        print(f"{'='*60}")
        
        try:
            is_valid, errors, warnings = validate_adapter(project_id)
        except Exception as e:
            print(f"❌ Validation failed with exception: {e}")
            all_valid = False
            continue
        
        if errors:
            print("\n❌ ERRORS:")
            for error in errors:
                print(f"  - {error}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if args.strict and warnings:
            is_valid = False
        
        if is_valid:
            print(f"\n✅ {project_id}: VALID")
        else:
            print(f"\n❌ {project_id}: INVALID")
            all_valid = False
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ All validations passed")
        sys.exit(0)
    else:
        print("❌ Some validations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
