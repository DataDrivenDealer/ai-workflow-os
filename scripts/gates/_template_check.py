"""
Example gate check script template.

Each check script should:
1. Define a `run_check(config: dict) -> dict` function
2. Return a dict with: status, actual, message, evidence_path (optional)

This is a template - copy and modify for your specific check.
"""

from pathlib import Path
from typing import Any, Dict


def run_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the check and return result.
    
    Args:
        config: The check configuration from gates.yaml
        
    Returns:
        Dict with:
            - status: "pass", "fail", "warn", "error"
            - actual: The actual value found
            - message: Human-readable message
            - evidence_path: Optional path to evidence file
    """
    # Example: Check if a file exists
    # target_path = Path(config.get("target", ""))
    # if target_path.exists():
    #     return {
    #         "status": "pass",
    #         "actual": True,
    #         "message": f"File exists: {target_path}",
    #     }
    # else:
    #     return {
    #         "status": "fail",
    #         "actual": False,
    #         "message": f"File not found: {target_path}",
    #     }
    
    return {
        "status": "warn",
        "actual": None,
        "message": "This is a template - implement the actual check logic",
    }
