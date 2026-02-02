"""
Path management for AI Workflow OS.

This module provides centralized path constants to eliminate hardcoded
`Path(__file__).parents[1]` patterns throughout the codebase.

Usage:
    from kernel.paths import ROOT, STATE_DIR, TASKS_DIR
    
    # Access state files
    tasks_path = TASKS_STATE_PATH
    
    # Ensure directories exist
    ensure_dirs()
"""

from pathlib import Path
from typing import Optional

# ============================================================================
# Root Directories
# ============================================================================

# Project root (parent of kernel/)
ROOT = Path(__file__).resolve().parents[1]

# Primary directories
KERNEL_DIR = ROOT / "kernel"
STATE_DIR = ROOT / "state"
TASKS_DIR = ROOT / "tasks"
SPECS_DIR = ROOT / "specs"
CONFIGS_DIR = ROOT / "configs"
TEMPLATES_DIR = ROOT / "templates"
SCRIPTS_DIR = ROOT / "scripts"
OPS_DIR = ROOT / "ops"
DOCS_DIR = ROOT / "docs"
PROJECTS_DIR = ROOT / "projects"
REPORTS_DIR = ROOT / "reports"
HOOKS_DIR = ROOT / "hooks"
LEGACY_DIR = ROOT / "legacy"

# ============================================================================
# Configuration Files
# ============================================================================

STATE_MACHINE_PATH = KERNEL_DIR / "state_machine.yaml"
REGISTRY_PATH = ROOT / "spec_registry.yaml"
GATES_CONFIG_PATH = CONFIGS_DIR / "gates.yaml"
MCP_MANIFEST_PATH = ROOT / "mcp_server_manifest.json"
PYRIGHT_CONFIG_PATH = ROOT / "pyrightconfig.json"
REQUIREMENTS_PATH = ROOT / "requirements.txt"
REQUIREMENTS_LOCK_PATH = ROOT / "requirements-lock.txt"

# ============================================================================
# State Files
# ============================================================================

TASKS_STATE_PATH = STATE_DIR / "tasks.yaml"
AGENTS_STATE_PATH = STATE_DIR / "agents.yaml"
SESSIONS_STATE_PATH = STATE_DIR / "sessions.yaml"
PROJECT_STATE_PATH = STATE_DIR / "project.yaml"

# ============================================================================
# Template Files
# ============================================================================

TASKCARD_TEMPLATE_PATH = TEMPLATES_DIR / "TASKCARD_TEMPLATE.md"
TASKCARD_WITH_REVIEW_PATH = TEMPLATES_DIR / "TASKCARD_WITH_REVIEW.md"
REVIEW_REPORT_TEMPLATE_PATH = TEMPLATES_DIR / "REVIEW_REPORT_TEMPLATE.md"
DONE_CRITERIA_CHECKLIST_PATH = TEMPLATES_DIR / "DONE_CRITERIA_CHECKLIST.md"

# ============================================================================
# Operational Directories
# ============================================================================

OPS_AUDIT_DIR = OPS_DIR / "audit"
OPS_DECISION_LOG_DIR = OPS_DIR / "decision-log"
OPS_FREEZE_DIR = OPS_DIR / "freeze"
OPS_ACCEPTANCE_DIR = OPS_DIR / "acceptance"
OPS_DEVIATIONS_DIR = OPS_DIR / "deviations"
OPS_PROPOSALS_DIR = OPS_DIR / "proposals"

TASKS_DONE_DIR = TASKS_DIR / "done"
TASKS_INBOX_DIR = TASKS_DIR / "inbox"
TASKS_RUNNING_DIR = TASKS_DIR / "running"

DOCS_PLANS_DIR = DOCS_DIR / "plans"
DOCS_STATE_DIR = DOCS_DIR / "state"

REPORTS_GATES_DIR = REPORTS_DIR / "gates"

# ============================================================================
# Specs Directories
# ============================================================================

SPECS_CANON_DIR = SPECS_DIR / "canon"
SPECS_FRAMEWORK_DIR = SPECS_DIR / "framework"

# ============================================================================
# Utility Functions
# ============================================================================

def ensure_dirs() -> None:
    """
    Ensure all required directories exist.
    
    Creates directories with parents=True, exist_ok=True to avoid errors
    when running on fresh clones or after cleanup operations.
    """
    required_dirs = [
        STATE_DIR,
        TASKS_DIR,
        TASKS_DONE_DIR,
        TASKS_INBOX_DIR,
        TASKS_RUNNING_DIR,
        OPS_AUDIT_DIR,
        OPS_DECISION_LOG_DIR,
        OPS_FREEZE_DIR,
        OPS_DEVIATIONS_DIR,
        OPS_PROPOSALS_DIR,
        DOCS_PLANS_DIR,
        DOCS_STATE_DIR,
        REPORTS_GATES_DIR,
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_task_path(task_id: str, status: Optional[str] = None) -> Path:
    """
    Get the file path for a TaskCard.
    
    Args:
        task_id: Task identifier (e.g., "SDF_DEV_001")
        status: Optional status subdirectory ("done", "inbox", "running")
    
    Returns:
        Path to the TaskCard markdown file
        
    Example:
        >>> get_task_path("SDF_DEV_001")
        Path("tasks/SDF_DEV_001.md")
        >>> get_task_path("SDF_DEV_001", "done")
        Path("tasks/done/SDF_DEV_001.md")
    """
    filename = f"{task_id}.md"
    
    if status == "done":
        return TASKS_DONE_DIR / filename
    elif status == "inbox":
        return TASKS_INBOX_DIR / filename
    elif status == "running":
        return TASKS_RUNNING_DIR / filename
    else:
        return TASKS_DIR / filename


def get_ops_audit_path(execution_id: str) -> Path:
    """
    Get the file path for an execution audit log.
    
    Args:
        execution_id: Execution identifier (e.g., "EXEC_20260201_DGSF_INIT")
    
    Returns:
        Path to the audit markdown file
        
    Example:
        >>> get_ops_audit_path("EXEC_20260201_DGSF_INIT")
        Path("ops/audit/EXEC_20260201_DGSF_INIT.md")
    """
    return OPS_AUDIT_DIR / f"{execution_id}.md"


# ============================================================================
# Module Verification
# ============================================================================

if __name__ == "__main__":
    print("AI Workflow OS - Path Configuration")
    print("=" * 60)
    print(f"ROOT:        {ROOT}")
    print(f"KERNEL_DIR:  {KERNEL_DIR}")
    print(f"STATE_DIR:   {STATE_DIR}")
    print(f"TASKS_DIR:   {TASKS_DIR}")
    print(f"SPECS_DIR:   {SPECS_DIR}")
    print(f"CONFIGS_DIR: {CONFIGS_DIR}")
    print()
    print("Configuration Files:")
    print(f"  STATE_MACHINE_PATH: {STATE_MACHINE_PATH}")
    print(f"  REGISTRY_PATH:      {REGISTRY_PATH}")
    print(f"  GATES_CONFIG_PATH:  {GATES_CONFIG_PATH}")
    print()
    print("State Files:")
    print(f"  TASKS_STATE_PATH:    {TASKS_STATE_PATH}")
    print(f"  AGENTS_STATE_PATH:   {AGENTS_STATE_PATH}")
    print(f"  SESSIONS_STATE_PATH: {SESSIONS_STATE_PATH}")
    print()
    print("Verification:")
    print(f"  ROOT exists: {ROOT.exists()}")
    print(f"  STATE_DIR exists: {STATE_DIR.exists()}")
    print(f"  TASKS_DIR exists: {TASKS_DIR.exists()}")
