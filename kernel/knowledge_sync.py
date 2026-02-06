"""
Knowledge Sync Scheduler (PP-025)

Tracks and enforces QKB (Quant Knowledge Base) update frequency.
Generates reminders when knowledge sync is overdue.

Usage:
    python kernel/knowledge_sync.py status         # Check sync status
    python kernel/knowledge_sync.py record         # Record a sync event
    python kernel/knowledge_sync.py check          # Check if sync needed (exit 1 if overdue)
    python kernel/knowledge_sync.py schedule       # Show sync schedule
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
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

STATE_FILE = ROOT / "state" / "knowledge_sync_state.yaml"
QKB_FILE = ROOT / "configs" / "quant_knowledge_base.yaml"

# Sync schedule (from QKB definition)
SYNC_SCHEDULE = {
    "scan": {"interval_days": 7, "name": "Weekly Scan"},
    "review": {"interval_days": 30, "name": "Monthly Review"},
    "deep_audit": {"interval_days": 90, "name": "Quarterly Audit"}
}


def now_iso() -> str:
    """Return current time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    """Load sync state."""
    if not yaml:
        return {}
    
    if not STATE_FILE.exists():
        return {
            "syncs": [],
            "last_scan": None,
            "last_review": None,
            "last_deep_audit": None
        }
    
    with open(STATE_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_state(state: dict) -> None:
    """Save sync state."""
    if not yaml:
        print("[ERROR] PyYAML not installed")
        return
    
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        yaml.dump(state, f, default_flow_style=False, allow_unicode=True)


def get_qkb_meta() -> dict:
    """Get QKB metadata for version tracking."""
    if not yaml or not QKB_FILE.exists():
        return {}
    
    with open(QKB_FILE, encoding="utf-8") as f:
        qkb = yaml.safe_load(f) or {}
    
    return {
        "version": qkb.get("version"),
        "updated": qkb.get("updated"),
        "entries_count": len(qkb.get("frontier_research", [])) + 
                        len(qkb.get("methodologies", [])) +
                        len(qkb.get("tools_resources", []))
    }


def is_overdue(sync_type: str, state: dict) -> tuple:
    """Check if a sync type is overdue. Returns (is_overdue, days_since, days_allowed)."""
    schedule = SYNC_SCHEDULE.get(sync_type, {})
    interval_days = schedule.get("interval_days", 7)
    
    last_key = f"last_{sync_type}"
    last_sync_str = state.get(last_key)
    
    if not last_sync_str:
        return (True, 999, interval_days)
    
    try:
        # Parse ISO datetime
        last_sync = datetime.fromisoformat(last_sync_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days_since = (now - last_sync).days
        return (days_since > interval_days, days_since, interval_days)
    except Exception:
        return (True, 999, interval_days)


def record_sync(sync_type: str, notes: str = None) -> dict:
    """Record a sync event."""
    state = load_state()
    
    if sync_type not in SYNC_SCHEDULE:
        print(f"[ERROR] Unknown sync type: {sync_type}")
        print(f"        Valid types: {', '.join(SYNC_SCHEDULE.keys())}")
        return state
    
    timestamp = now_iso()
    
    # Update last sync
    state[f"last_{sync_type}"] = timestamp
    
    # Add to history
    syncs = state.setdefault("syncs", [])
    syncs.append({
        "type": sync_type,
        "timestamp": timestamp,
        "notes": notes,
        "qkb_meta": get_qkb_meta()
    })
    
    # Keep only last 50 syncs
    state["syncs"] = syncs[-50:]
    
    save_state(state)
    print(f"[OK] {SYNC_SCHEDULE[sync_type]['name']} recorded at {timestamp}")
    
    return state


def check_sync_needed() -> dict:
    """Check if any sync is overdue. Returns status and exit code."""
    state = load_state()
    
    result = {
        "overdue": [],
        "upcoming": [],
        "status": "ok"
    }
    
    for sync_type, schedule in SYNC_SCHEDULE.items():
        overdue, days_since, allowed = is_overdue(sync_type, state)
        
        info = {
            "type": sync_type,
            "name": schedule["name"],
            "days_since": days_since,
            "interval": allowed,
            "days_remaining": max(0, allowed - days_since)
        }
        
        if overdue:
            result["overdue"].append(info)
        elif info["days_remaining"] <= 2:
            result["upcoming"].append(info)
    
    if result["overdue"]:
        result["status"] = "overdue"
    elif result["upcoming"]:
        result["status"] = "upcoming"
    
    return result


def show_status() -> None:
    """Display current sync status."""
    state = load_state()
    result = check_sync_needed()
    
    print("\n=== KNOWLEDGE SYNC STATUS ===")
    print(f"State file: {STATE_FILE}")
    
    qkb = get_qkb_meta()
    if qkb:
        print(f"QKB Version: {qkb.get('version')}")
        print(f"QKB Entries: {qkb.get('entries_count')}")
    
    print("\nSync Schedule:")
    for sync_type, schedule in SYNC_SCHEDULE.items():
        overdue, days_since, allowed = is_overdue(sync_type, state)
        
        if days_since == 999:
            status = "⚠️ NEVER"
        elif overdue:
            status = f"❌ OVERDUE ({days_since - allowed}d)"
        elif allowed - days_since <= 2:
            status = f"⏳ SOON ({allowed - days_since}d)"
        else:
            status = f"✅ OK ({allowed - days_since}d remaining)"
        
        print(f"  {schedule['name']}: {status}")
        last = state.get(f"last_{sync_type}")
        if last and days_since != 999:
            print(f"    Last: {last[:10]} ({days_since}d ago)")
    
    print("\nRecent Syncs:")
    syncs = state.get("syncs", [])[-5:]
    if syncs:
        for s in reversed(syncs):
            print(f"  {s['timestamp'][:10]}: {s['type']} - {s.get('notes', 'No notes')}")
    else:
        print("  No syncs recorded yet.")
    
    print("=============================\n")


def show_schedule() -> None:
    """Display sync schedule and reminders."""
    state = load_state()
    
    print("\n=== KNOWLEDGE SYNC SCHEDULE ===")
    
    for sync_type, schedule in SYNC_SCHEDULE.items():
        overdue, days_since, allowed = is_overdue(sync_type, state)
        
        if days_since == 999:
            next_due = "IMMEDIATELY"
        else:
            next_due = f"in {max(0, allowed - days_since)} days"
        
        print(f"\n{schedule['name']}:")
        print(f"  Interval: Every {allowed} days")
        print(f"  Next due: {next_due}")
        
        if overdue:
            print(f"  ⚠️ ACTION REQUIRED: Run /dgsf_knowledge_sync")
    
    print("\n================================\n")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Sync Scheduler (PP-025)")
    parser.add_argument("command", choices=["status", "record", "check", "schedule"],
                       help="Command to execute")
    parser.add_argument("--type", "-t", choices=list(SYNC_SCHEDULE.keys()),
                       default="scan", help="Sync type for record command")
    parser.add_argument("--notes", "-n", help="Notes for record command")
    
    args = parser.parse_args()
    
    if args.command == "status":
        show_status()
    
    elif args.command == "record":
        record_sync(args.type, args.notes)
    
    elif args.command == "check":
        result = check_sync_needed()
        
        if result["status"] == "overdue":
            print("\n⚠️ KNOWLEDGE SYNC OVERDUE")
            for item in result["overdue"]:
                print(f"  - {item['name']}: {item['days_since']}d since last (max: {item['interval']}d)")
            print("\nRun: /dgsf_knowledge_sync")
            exit(1)
        elif result["status"] == "upcoming":
            print("\n⏳ KNOWLEDGE SYNC DUE SOON")
            for item in result["upcoming"]:
                print(f"  - {item['name']}: {item['days_remaining']}d remaining")
            exit(0)
        else:
            print("\n✅ All knowledge syncs up to date")
            exit(0)
    
    elif args.command == "schedule":
        show_schedule()


if __name__ == "__main__":
    main()
