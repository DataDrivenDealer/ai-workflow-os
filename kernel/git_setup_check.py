"""
Git Setup Check — Detects missing hooks and prompts for installation.

Called automatically during git operations to ensure the local environment
has all required Git hooks installed.  When hooks are missing, the user
is prompted with a single [Y/n] confirmation to install them.

Version: 1.0.0
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from kernel.paths import ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOOKS_SOURCE_DIR = ROOT / "hooks"
GIT_DIR = ROOT / ".git"
GIT_HOOKS_DIR = GIT_DIR / "hooks"

# All hooks that should be installed
REQUIRED_HOOKS = [
    "pre-commit",
    "pre-push",
    "pre-spec-change",
    "post-spec-change",
    "post-tag",
    "pre-destructive-op",
]

# State file to remember user's choice (avoids repeated prompts)
SETUP_STATE_FILE = ROOT / "state" / ".git_hooks_check"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SetupStatus:
    """Result of checking the Git environment setup."""
    hooks_installed: bool
    missing_hooks: List[str] = field(default_factory=list)
    installed_hooks: List[str] = field(default_factory=list)
    git_available: bool = True
    message: str = ""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def check_git_hooks() -> SetupStatus:
    """Check which required hooks are installed in .git/hooks/.

    Returns:
        SetupStatus describing what is present/missing.
    """
    if not GIT_DIR.exists():
        return SetupStatus(
            hooks_installed=False,
            git_available=False,
            message="Not a Git repository (.git/ not found).",
        )

    missing: List[str] = []
    installed: List[str] = []

    for hook_name in REQUIRED_HOOKS:
        source = HOOKS_SOURCE_DIR / hook_name
        target = GIT_HOOKS_DIR / hook_name

        if not source.exists():
            # Source hook doesn't exist — skip (not our problem)
            continue

        if target.exists():
            installed.append(hook_name)
        else:
            missing.append(hook_name)

    all_installed = len(missing) == 0
    if all_installed:
        msg = f"All {len(installed)} hooks installed."
    else:
        msg = (
            f"{len(missing)} hook(s) missing: {', '.join(missing)}. "
            f"{len(installed)} installed."
        )

    return SetupStatus(
        hooks_installed=all_installed,
        missing_hooks=missing,
        installed_hooks=installed,
        message=msg,
    )


def install_hooks(force: bool = False) -> SetupStatus:
    """Copy all hooks from hooks/ to .git/hooks/.

    Args:
        force: If True, overwrite existing hooks.

    Returns:
        SetupStatus after installation.
    """
    if not GIT_DIR.exists():
        return SetupStatus(
            hooks_installed=False,
            git_available=False,
            message="Cannot install hooks: .git/ not found.",
        )

    GIT_HOOKS_DIR.mkdir(parents=True, exist_ok=True)

    installed: List[str] = []
    errors: List[str] = []

    for hook_name in REQUIRED_HOOKS:
        source = HOOKS_SOURCE_DIR / hook_name
        target = GIT_HOOKS_DIR / hook_name

        if not source.exists():
            continue

        if target.exists() and not force:
            installed.append(hook_name)
            continue

        try:
            shutil.copy2(str(source), str(target))
            # Ensure executable on Unix-like systems
            if os.name != "nt":
                target.chmod(target.stat().st_mode | 0o111)
            installed.append(hook_name)
        except OSError as exc:
            errors.append(f"{hook_name}: {exc}")

    all_good = len(errors) == 0
    msg_parts = [f"Installed {len(installed)} hooks."]
    if errors:
        msg_parts.append(f"Errors: {'; '.join(errors)}")

    return SetupStatus(
        hooks_installed=all_good,
        installed_hooks=installed,
        missing_hooks=[h for h in REQUIRED_HOOKS if h not in installed],
        message=" ".join(msg_parts),
    )


def _user_declined_recently() -> bool:
    """Check if the user declined hook installation recently (within 24h)."""
    if not SETUP_STATE_FILE.exists():
        return False
    try:
        import time
        mtime = SETUP_STATE_FILE.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours > 24:
            SETUP_STATE_FILE.unlink(missing_ok=True)
            return False
        content = SETUP_STATE_FILE.read_text().strip()
        return content == "declined"
    except Exception:
        return False


def _record_decline() -> None:
    """Record that user declined hook installation."""
    SETUP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETUP_STATE_FILE.write_text("declined")


def _record_installed() -> None:
    """Record that hooks were installed."""
    if SETUP_STATE_FILE.exists():
        SETUP_STATE_FILE.unlink(missing_ok=True)


def prompt_and_install_hooks(
    status: Optional[SetupStatus] = None,
    interactive: bool = True,
) -> SetupStatus:
    """Check hooks and prompt user to install if missing.

    This is the main entry point called from git_ops workflows.

    Args:
        status: Pre-computed status (avoids re-checking).
        interactive: If False, skip user prompt (for CI).

    Returns:
        Final SetupStatus.
    """
    if status is None:
        status = check_git_hooks()

    if status.hooks_installed:
        return status

    if not status.git_available:
        return status

    # Check if user already declined recently
    if _user_declined_recently():
        return status

    if not interactive:
        return status

    # Display prompt
    print("")
    print("=" * 60)
    print("  ⚠️  GIT HOOKS NOT INSTALLED")
    print("=" * 60)
    print("")
    print(f"  Missing hooks: {', '.join(status.missing_hooks)}")
    print("")
    print("  Git hooks enforce branch naming, YAML validation,")
    print("  and pre-push gate checks automatically.")
    print("")

    try:
        answer = input("  Install hooks now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer in ("", "y", "yes"):
        result = install_hooks(force=True)
        _record_installed()
        print("")
        if result.hooks_installed:
            print("  ✅ Hooks installed successfully!")
            for h in result.installed_hooks:
                print(f"    - {h}")
        else:
            print(f"  ⚠️ {result.message}")
        print("")
        return result
    else:
        _record_decline()
        print("  Skipped. You can install later with:")
        print("    .\\scripts\\install_hooks.ps1")
        print("")
        return status


# ---------------------------------------------------------------------------
# Format for Copilot / Agent output
# ---------------------------------------------------------------------------

def format_setup_status(status: SetupStatus) -> str:
    """Format the setup status for display in Copilot output."""
    if status.hooks_installed:
        return f"✅ Git hooks: {len(status.installed_hooks)} installed"

    lines = [
        "⚠️ Git Hooks Status:",
        f"  Installed: {', '.join(status.installed_hooks) or 'none'}",
        f"  Missing:   {', '.join(status.missing_hooks)}",
        "",
        "  Run: .\\scripts\\install_hooks.ps1",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Git setup check")
    parser.add_argument("--install", action="store_true",
                        help="Install hooks without prompting")
    parser.add_argument("--check", action="store_true",
                        help="Check only, no prompt")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing hooks on install")
    args = parser.parse_args()

    if args.install:
        result = install_hooks(force=args.force)
        print(result.message)
        return 0 if result.hooks_installed else 1

    status = check_git_hooks()
    if args.check:
        print(format_setup_status(status))
        return 0 if status.hooks_installed else 1

    # Interactive prompt
    prompt_and_install_hooks(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
