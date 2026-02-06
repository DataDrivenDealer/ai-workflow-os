"""
Plan Mode Phase Persistence (PP-024)

Enforces phase state persistence for Plan Mode P0-P9 flow.
Ensures session resumption after interruption.

Usage:
    python kernel/plan_mode_phases.py start              # Start Plan Mode
    python kernel/plan_mode_phases.py complete P1        # Mark P1 complete
    python kernel/plan_mode_phases.py status             # Show current state
    python kernel/plan_mode_phases.py resume             # Get resume info
    python kernel/plan_mode_phases.py checkpoint P4      # Save checkpoint at P4
    python kernel/plan_mode_phases.py end                # End Plan Mode
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse

try:
    import yaml
except ImportError:
    yaml = None

# Find project root
_ROOT = Path(__file__).parent.parent
try:
    from kernel.paths import ROOT
except ImportError:
    ROOT = _ROOT

STATE_FILE = ROOT / "state" / "plan_mode_state.yaml"

# Phase definitions
PHASES = {
    "P0": {"name": "Owner Steering Parse", "required": True},
    "P0.5": {"name": "Escalation Check", "required": False},
    "P1": {"name": "Task/Problem Universe Scan", "required": True},
    "P2": {"name": "Transition to Canonical", "required": True},
    "P3": {"name": "Phase Gate Check", "required": True},
    "P4": {"name": "System Diagnostic", "required": True},
    "P5": {"name": "Problem Qualification", "required": True},
    "P6": {"name": "DRS Resolution", "required": False},
    "P7": {"name": "Research Governance", "required": False},
    "P8": {"name": "Write-back", "required": True},
    "P9": {"name": "Exit to Execute", "required": True},
}

PHASE_ORDER = ["P0", "P0.5", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]


def now_iso() -> str:
    """Return current time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    """Load current plan mode state."""
    if not yaml:
        print("[ERROR] PyYAML not installed")
        return {}
    
    if not STATE_FILE.exists():
        return {
            "plan_mode": {
                "active": False,
                "phases": {}
            }
        }
    
    with open(STATE_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"plan_mode": {"active": False, "phases": {}}}


def save_state(state: dict) -> None:
    """Save plan mode state."""
    if not yaml:
        print("[ERROR] PyYAML not installed")
        return
    
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Update last_updated
    if "plan_mode" in state:
        state["plan_mode"]["last_updated"] = now_iso()
    
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        yaml.dump(state, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"[OK] State saved to {STATE_FILE}")


def start_plan_mode() -> dict:
    """Start a new Plan Mode session."""
    state = load_state()
    
    if state.get("plan_mode", {}).get("active"):
        print("[WARN] Plan Mode already active, resuming...")
        return state
    
    state["plan_mode"] = {
        "active": True,
        "started_at": now_iso(),
        "last_updated": now_iso(),
        "current_phase": "P0",
        "owner_steering": "<EMPTY>",
        "steering_mode": "autonomous",
        "phases": {
            phase: {"status": "not-started"} for phase in PHASE_ORDER
        },
        "checkpoints": [],
        "interruptions": []
    }
    
    save_state(state)
    print(f"[OK] Plan Mode started at {state['plan_mode']['started_at']}")
    print(f"     Current phase: P0 - {PHASES['P0']['name']}")
    return state


def complete_phase(phase: str, metadata: dict = None) -> dict:
    """Mark a phase as complete and advance to next."""
    state = load_state()
    pm = state.get("plan_mode", {})
    
    if not pm.get("active"):
        print("[ERROR] Plan Mode not active. Start with: python kernel/plan_mode_phases.py start")
        return state
    
    # Normalize phase name
    phase_key = phase.upper().replace("P0_5", "P0.5").replace("P05", "P0.5")
    if phase_key not in PHASES:
        print(f"[ERROR] Unknown phase: {phase}")
        print(f"        Valid phases: {', '.join(PHASE_ORDER)}")
        return state
    
    # Update phase status
    phases = pm.setdefault("phases", {})
    phases[phase_key] = {
        "status": "completed",
        "completed_at": now_iso()
    }
    if metadata:
        phases[phase_key].update(metadata)
    
    # Determine next phase
    try:
        idx = PHASE_ORDER.index(phase_key)
        if idx < len(PHASE_ORDER) - 1:
            next_phase = PHASE_ORDER[idx + 1]
            pm["current_phase"] = next_phase
            print(f"[OK] {phase_key} completed")
            print(f"     Next phase: {next_phase} - {PHASES[next_phase]['name']}")
        else:
            pm["current_phase"] = "COMPLETE"
            print(f"[OK] {phase_key} completed - Plan Mode complete!")
    except ValueError:
        pm["current_phase"] = phase_key
    
    save_state(state)
    return state


def skip_phase(phase: str, reason: str) -> dict:
    """Mark a phase as skipped with reason."""
    state = load_state()
    pm = state.get("plan_mode", {})
    
    if not pm.get("active"):
        print("[ERROR] Plan Mode not active")
        return state
    
    phase_key = phase.upper()
    phases = pm.setdefault("phases", {})
    phases[phase_key] = {
        "status": "skipped",
        "skipped_at": now_iso(),
        "skip_reason": reason
    }
    
    # Advance to next
    try:
        idx = PHASE_ORDER.index(phase_key)
        if idx < len(PHASE_ORDER) - 1:
            next_phase = PHASE_ORDER[idx + 1]
            pm["current_phase"] = next_phase
    except ValueError:
        pass
    
    save_state(state)
    print(f"[OK] {phase_key} skipped: {reason}")
    return state


def checkpoint(phase: str, notes: str = None) -> dict:
    """Save a checkpoint for session resumption."""
    state = load_state()
    pm = state.get("plan_mode", {})
    
    if not pm.get("active"):
        print("[ERROR] Plan Mode not active")
        return state
    
    checkpoints = pm.setdefault("checkpoints", [])
    checkpoints.append({
        "phase": phase,
        "timestamp": now_iso(),
        "notes": notes or f"Checkpoint at {phase}"
    })
    
    save_state(state)
    print(f"[OK] Checkpoint saved at {phase}")
    return state


def record_interruption(reason: str = "session_ended") -> dict:
    """Record an interruption for audit."""
    state = load_state()
    pm = state.get("plan_mode", {})
    
    if not pm.get("active"):
        return state
    
    interruptions = pm.setdefault("interruptions", [])
    interruptions.append({
        "phase": pm.get("current_phase", "unknown"),
        "timestamp": now_iso(),
        "reason": reason
    })
    
    save_state(state)
    print(f"[WARN] Interruption recorded at {pm.get('current_phase')}: {reason}")
    return state


def get_resume_info() -> dict:
    """Get information needed to resume Plan Mode."""
    state = load_state()
    pm = state.get("plan_mode", {})
    
    if not pm.get("active"):
        print("Plan Mode is not active.")
        print("Start with: python kernel/plan_mode_phases.py start")
        return {"active": False}
    
    current = pm.get("current_phase", "P0")
    phases = pm.get("phases", {})
    
    # Calculate progress
    completed = [p for p, info in phases.items() if info.get("status") == "completed"]
    
    resume_info = {
        "active": True,
        "started_at": pm.get("started_at"),
        "current_phase": current,
        "phase_name": PHASES.get(current, {}).get("name", "Unknown"),
        "completed_phases": completed,
        "progress": f"{len(completed)}/{len(PHASES)}",
        "last_checkpoint": pm.get("checkpoints", [{}])[-1] if pm.get("checkpoints") else None,
        "interruptions": len(pm.get("interruptions", []))
    }
    
    print("\n=== PLAN MODE RESUME INFO ===")
    print(f"Started:    {resume_info['started_at']}")
    print(f"Current:    {current} - {resume_info['phase_name']}")
    print(f"Progress:   {resume_info['progress']} phases")
    print(f"Completed:  {', '.join(completed) or 'None'}")
    if resume_info['last_checkpoint']:
        print(f"Checkpoint: {resume_info['last_checkpoint']}")
    if resume_info['interruptions']:
        print(f"Interruptions: {resume_info['interruptions']}")
    print("=============================\n")
    
    return resume_info


def end_plan_mode(exit_trigger: str = "manual") -> dict:
    """End Plan Mode session."""
    state = load_state()
    pm = state.get("plan_mode", {})
    
    if not pm.get("active"):
        print("[WARN] Plan Mode already inactive")
        return state
    
    pm["active"] = False
    pm["ended_at"] = now_iso()
    pm["exit_trigger"] = exit_trigger
    pm["current_phase"] = "P9"
    
    # Mark P9 as complete
    phases = pm.setdefault("phases", {})
    phases["P9"] = {
        "status": "completed",
        "completed_at": now_iso(),
        "exit_trigger": exit_trigger
    }
    
    save_state(state)
    print(f"[OK] Plan Mode ended: {exit_trigger}")
    
    # Suggest next action
    print("\nNext: Switch to Execute Mode")
    print("      'Resume EXECUTE MODE' or 'Switch to EXECUTE MODE'")
    
    return state


def show_status() -> dict:
    """Display current Plan Mode status."""
    state = load_state()
    pm = state.get("plan_mode", {})
    
    print("\n=== PLAN MODE STATUS ===")
    print(f"Active: {pm.get('active', False)}")
    
    if not pm.get("active"):
        if pm.get("ended_at"):
            print(f"Last session ended: {pm.get('ended_at')}")
            print(f"Exit trigger: {pm.get('exit_trigger')}")
        print("========================\n")
        return state
    
    print(f"Started: {pm.get('started_at')}")
    print(f"Current Phase: {pm.get('current_phase')}")
    print(f"Steering Mode: {pm.get('steering_mode')}")
    print("\nPhase Progress:")
    
    phases = pm.get("phases", {})
    for phase in PHASE_ORDER:
        info = phases.get(phase, {"status": "not-started"})
        status = info.get("status", "not-started")
        
        if status == "completed":
            icon = "✅"
            extra = info.get("completed_at", "")[:10]
        elif status == "skipped":
            icon = "⏭️"
            extra = info.get("skip_reason", "")[:30]
        elif phase == pm.get("current_phase"):
            icon = "🔄"
            extra = "IN PROGRESS"
        else:
            icon = "⬜"
            extra = ""
        
        print(f"  {icon} {phase}: {PHASES[phase]['name'][:30]} {extra}")
    
    print("========================\n")
    return state


def main():
    parser = argparse.ArgumentParser(description="Plan Mode Phase Persistence (PP-024)")
    parser.add_argument("command", choices=["start", "complete", "skip", "checkpoint", 
                                            "interrupt", "resume", "status", "end"],
                       help="Command to execute")
    parser.add_argument("phase", nargs="?", help="Phase to operate on (e.g., P1, P4)")
    parser.add_argument("--reason", "-r", help="Reason for skip/interrupt")
    parser.add_argument("--notes", "-n", help="Notes for checkpoint")
    parser.add_argument("--trigger", "-t", default="Switch to EXECUTE MODE",
                       help="Exit trigger for end command")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_plan_mode()
    elif args.command == "complete":
        if not args.phase:
            print("[ERROR] Phase required: python kernel/plan_mode_phases.py complete P1")
            return
        complete_phase(args.phase)
    elif args.command == "skip":
        if not args.phase:
            print("[ERROR] Phase required: python kernel/plan_mode_phases.py skip P6 --reason 'No DRS needed'")
            return
        skip_phase(args.phase, args.reason or "Skipped by user")
    elif args.command == "checkpoint":
        checkpoint(args.phase or "unknown", args.notes)
    elif args.command == "interrupt":
        record_interruption(args.reason or "session_ended")
    elif args.command == "resume":
        get_resume_info()
    elif args.command == "status":
        show_status()
    elif args.command == "end":
        end_plan_mode(args.trigger)


if __name__ == "__main__":
    main()
