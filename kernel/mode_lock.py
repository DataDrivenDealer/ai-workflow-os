"""
Mode Lock Module for Copilot Runtime OS.

Provides deterministic enforcement of operating mode constraints.
When PLAN MODE is active, code execution and file writes are blocked.

Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add kernel to path for imports
_KERNEL_DIR = Path(__file__).parent
_ROOT = _KERNEL_DIR.parent
sys.path.insert(0, str(_ROOT))

try:
    from kernel.paths import ROOT
except ImportError:
    ROOT = _ROOT


class OperatingMode(Enum):
    """Operating modes with their enforcement levels."""
    EXECUTE = "EXECUTE"
    PLAN = "PLAN"
    REVIEW = "REVIEW"


@dataclass
class ModeLock:
    """Represents the current mode lock state."""
    mode: OperatingMode
    locked_at: datetime
    session_id: str
    prohibitions: List[str]
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if the lock has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_action_allowed(self, action: str) -> bool:
        """Check if an action is allowed under current mode."""
        if self.mode == OperatingMode.EXECUTE:
            return True
        return action not in self.prohibitions


# Default prohibitions for each mode
MODE_PROHIBITIONS = {
    OperatingMode.PLAN: [
        "write_code",
        "run_terminal",
        "run_python",
        "run_pytest", 
        "create_file",
        "edit_file",
        "run_data",
        "execute_task",
    ],
    OperatingMode.REVIEW: [
        "write_code",
        "run_data",
        "execute_task",
    ],
    OperatingMode.EXECUTE: [],
}

# File patterns blocked in PLAN mode
PLAN_MODE_BLOCKED_PATTERNS = [
    "*.py",
    "*.ts",
    "*.js",
    "*.jsx",
    "*.tsx",
    "*.yaml",  # except state files
    "*.yml",
]

# State files that ARE allowed in PLAN mode
PLAN_MODE_ALLOWED_STATE_FILES = [
    "state/",
    "docs/state/",
    "docs/subagents/",
    "decisions/",
]


def get_lock_path() -> Path:
    """Get the path to the mode lock file."""
    return ROOT / "state" / "mode_lock.yaml"


def read_mode_lock() -> Optional[ModeLock]:
    """Read the current mode lock from state file.
    
    Returns:
        ModeLock if lock exists and is valid, None otherwise.
    """
    lock_path = get_lock_path()
    if not lock_path.exists():
        return None
    
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data:
            return None
        
        mode = OperatingMode(data.get("mode", "EXECUTE"))
        locked_at = datetime.fromisoformat(data["locked_at"])
        session_id = data.get("session_id", "unknown")
        prohibitions = data.get("prohibitions", MODE_PROHIBITIONS.get(mode, []))
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])
        
        lock = ModeLock(
            mode=mode,
            locked_at=locked_at,
            session_id=session_id,
            prohibitions=prohibitions,
            expires_at=expires_at,
        )
        
        # Check expiration
        if lock.is_expired():
            release_mode_lock()
            return None
        
        return lock
    except Exception:
        return None


def acquire_mode_lock(
    mode: OperatingMode,
    session_id: str,
    duration_hours: Optional[float] = None,
) -> ModeLock:
    """Acquire a mode lock.
    
    Args:
        mode: The operating mode to lock to.
        session_id: Identifier for the current session.
        duration_hours: Optional lock duration in hours.
    
    Returns:
        The created ModeLock.
    """
    now = datetime.now(timezone.utc)
    expires_at = None
    if duration_hours:
        from datetime import timedelta
        expires_at = now + timedelta(hours=duration_hours)
    
    prohibitions = MODE_PROHIBITIONS.get(mode, [])
    
    lock = ModeLock(
        mode=mode,
        locked_at=now,
        session_id=session_id,
        prohibitions=prohibitions,
        expires_at=expires_at,
    )
    
    lock_path = get_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "mode": lock.mode.value,
        "locked_at": lock.locked_at.isoformat(),
        "session_id": lock.session_id,
        "prohibitions": lock.prohibitions,
        "expires_at": lock.expires_at.isoformat() if lock.expires_at else None,
    }
    
    with open(lock_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    return lock


def release_mode_lock() -> bool:
    """Release the current mode lock.
    
    Returns:
        True if lock was released, False if no lock existed.
    """
    lock_path = get_lock_path()
    if lock_path.exists():
        lock_path.unlink()
        return True
    return False


def check_action_allowed(action: str, target_path: Optional[str] = None) -> Dict[str, Any]:
    """Check if an action is allowed under the current mode lock.
    
    Args:
        action: The action being attempted (e.g., "write_code", "run_terminal").
        target_path: Optional path being targeted.
    
    Returns:
        Dict with 'allowed' (bool) and 'reason' (str if blocked).
    """
    lock = read_mode_lock()
    
    # No lock = EXECUTE mode = all allowed
    if lock is None:
        return {"allowed": True, "mode": "EXECUTE", "reason": None}
    
    # Check if action is in prohibitions
    if not lock.is_action_allowed(action):
        return {
            "allowed": False,
            "mode": lock.mode.value,
            "reason": f"Action '{action}' is prohibited in {lock.mode.value} MODE",
            "prohibitions": lock.prohibitions,
        }
    
    # Additional path-based checks for PLAN mode
    if lock.mode == OperatingMode.PLAN and target_path:
        # Check if it's an allowed state file
        is_state_file = any(
            allowed in target_path.replace("\\", "/")
            for allowed in PLAN_MODE_ALLOWED_STATE_FILES
        )
        
        if not is_state_file:
            # Check if it matches blocked patterns
            from fnmatch import fnmatch
            filename = Path(target_path).name
            for pattern in PLAN_MODE_BLOCKED_PATTERNS:
                if fnmatch(filename, pattern):
                    return {
                        "allowed": False,
                        "mode": lock.mode.value,
                        "reason": f"File pattern '{pattern}' is blocked in PLAN MODE",
                        "target": target_path,
                    }
    
    return {"allowed": True, "mode": lock.mode.value, "reason": None}


def get_current_mode() -> OperatingMode:
    """Get the current operating mode.
    
    Returns:
        Current OperatingMode (defaults to EXECUTE if no lock).
    """
    lock = read_mode_lock()
    if lock is None:
        return OperatingMode.EXECUTE
    return lock.mode


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for mode lock management."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Mode Lock Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show current mode lock status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # acquire command
    acquire_parser = subparsers.add_parser("acquire", help="Acquire a mode lock")
    acquire_parser.add_argument("mode", choices=["PLAN", "EXECUTE", "REVIEW"])
    acquire_parser.add_argument("--session", default="cli", help="Session ID")
    acquire_parser.add_argument("--hours", type=float, help="Lock duration in hours")
    
    # release command
    subparsers.add_parser("release", help="Release the mode lock")
    
    # check command
    check_parser = subparsers.add_parser("check", help="Check if an action is allowed")
    check_parser.add_argument("action", help="Action to check")
    check_parser.add_argument("--path", help="Target path")
    check_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.command == "status":
        lock = read_mode_lock()
        if lock is None:
            result = {"mode": "EXECUTE", "locked": False}
        else:
            result = {
                "mode": lock.mode.value,
                "locked": True,
                "locked_at": lock.locked_at.isoformat(),
                "session_id": lock.session_id,
                "prohibitions": lock.prohibitions,
                "expires_at": lock.expires_at.isoformat() if lock.expires_at else None,
            }
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["locked"]:
                print(f"🔒 Mode: {result['mode']}")
                print(f"   Locked at: {result['locked_at']}")
                print(f"   Session: {result['session_id']}")
                print(f"   Prohibitions: {', '.join(result['prohibitions'])}")
            else:
                print("🔓 Mode: EXECUTE (no lock)")
    
    elif args.command == "acquire":
        mode = OperatingMode(args.mode)
        lock = acquire_mode_lock(mode, args.session, args.hours)
        print(f"✅ Acquired {lock.mode.value} MODE lock")
        print(f"   Session: {lock.session_id}")
        if lock.expires_at:
            print(f"   Expires: {lock.expires_at.isoformat()}")
    
    elif args.command == "release":
        if release_mode_lock():
            print("✅ Mode lock released")
        else:
            print("ℹ️ No lock to release")
    
    elif args.command == "check":
        result = check_action_allowed(args.action, args.path)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["allowed"]:
                print(f"✅ Action '{args.action}' is ALLOWED in {result['mode']} MODE")
            else:
                print(f"❌ Action '{args.action}' is BLOCKED")
                print(f"   Reason: {result['reason']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
