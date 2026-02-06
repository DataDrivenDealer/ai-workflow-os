"""
Check: Unit Tests Pass (G2.unit_tests_pass)

Runs pytest on kernel/tests and returns pass/fail status.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def run_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run unit tests and check if they pass.
    
    Args:
        config: Check configuration with optional 'command' override.
    
    Returns:
        Check result dict.
    """
    command = config.get("command", "pytest kernel/tests -v")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return {
                "status": "pass",
                "actual": True,
                "message": f"All tests passed",
                "evidence_path": None,
            }
        else:
            # Extract failure summary
            lines = result.stdout.split("\n")
            summary = [l for l in lines if "failed" in l.lower() or "error" in l.lower()]
            summary_text = summary[0] if summary else "Tests failed"
            
            return {
                "status": "fail",
                "actual": False,
                "message": summary_text,
            }
    
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "actual": None,
            "message": "Test execution timed out (>5 minutes)",
        }
    except Exception as e:
        return {
            "status": "error",
            "actual": None,
            "message": f"Failed to run tests: {str(e)}",
        }
