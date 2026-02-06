"""
Strategic Debt Priority Scorer (PP-030)

Automatically calculates priority scores for SDL (Strategic Debt Ledger) items
based on age, impact, and other factors.

Usage:
    python kernel/debt_priority.py score              # Score all debt items
    python kernel/debt_priority.py top 5              # Show top 5 priority items
    python kernel/debt_priority.py update             # Update SDL with scores
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import re

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

SDL_FILE = ROOT / "configs" / "strategic_debt_ledger.yaml"
STATE_FILE = ROOT / "state" / "debt_priority_state.yaml"

# Scoring weights
WEIGHTS = {
    "age": 0.25,         # Older = higher priority
    "impact": 0.35,      # Higher impact = higher priority  
    "effort": -0.15,     # Higher effort = slightly lower priority
    "blocking": 0.25,    # Blocking other work = higher priority
}

# Impact level mapping
IMPACT_SCORES = {
    "critical": 100,
    "high": 75,
    "medium": 50,
    "low": 25,
    "minimal": 10
}

# Effort level mapping (inverse - high effort = lower score)
EFFORT_SCORES = {
    "trivial": 90,
    "low": 75,
    "medium": 50,
    "high": 25,
    "massive": 10
}


def load_sdl() -> dict:
    """Load Strategic Debt Ledger."""
    if not yaml or not SDL_FILE.exists():
        return {}
    
    with open(SDL_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_sdl(sdl: dict) -> None:
    """Save SDL with scores."""
    if not yaml:
        return
    
    with open(SDL_FILE, "w", encoding="utf-8") as f:
        yaml.dump(sdl, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def calculate_age_score(created_date: str) -> float:
    """Calculate age score (0-100). Older items get higher scores."""
    if not created_date:
        return 50  # Default middle score
    
    try:
        # Parse various date formats
        if 'T' in created_date:
            created = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
        else:
            created = datetime.strptime(created_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        age_days = (now - created).days
        
        # Score based on age buckets
        if age_days > 365:
            return 100  # Over a year old
        elif age_days > 180:
            return 85
        elif age_days > 90:
            return 70
        elif age_days > 30:
            return 55
        elif age_days > 7:
            return 40
        else:
            return 25  # Very recent
            
    except Exception:
        return 50


def calculate_blocking_score(item: dict) -> float:
    """Calculate blocking score based on dependencies."""
    blocks = item.get("blocks", [])
    blocked_by = item.get("blocked_by", [])
    
    if not blocks and not blocked_by:
        return 50  # Neutral
    
    # Items that block others get higher priority
    blocking_count = len(blocks) if isinstance(blocks, list) else 1 if blocks else 0
    
    # Items blocked by others get slightly lower priority (wait for blockers)
    blocked_count = len(blocked_by) if isinstance(blocked_by, list) else 1 if blocked_by else 0
    
    score = 50 + (blocking_count * 15) - (blocked_count * 5)
    return min(100, max(0, score))


def calculate_priority_score(item: dict) -> dict:
    """Calculate overall priority score for a debt item."""
    # Get individual scores
    age_score = calculate_age_score(item.get("created") or item.get("identified"))
    
    impact_str = (item.get("impact") or "medium").lower()
    impact_score = IMPACT_SCORES.get(impact_str, 50)
    
    effort_str = (item.get("effort") or "medium").lower()
    effort_score = EFFORT_SCORES.get(effort_str, 50)
    
    blocking_score = calculate_blocking_score(item)
    
    # Calculate weighted total
    total = (
        age_score * WEIGHTS["age"] +
        impact_score * WEIGHTS["impact"] +
        effort_score * abs(WEIGHTS["effort"]) +
        blocking_score * WEIGHTS["blocking"]
    )
    
    return {
        "total": round(total, 1),
        "components": {
            "age": round(age_score, 1),
            "impact": round(impact_score, 1),
            "effort": round(effort_score, 1),
            "blocking": round(blocking_score, 1)
        },
        "calculated_at": datetime.now(timezone.utc).isoformat()
    }


def score_all_items() -> list:
    """Score all debt items in SDL."""
    sdl = load_sdl()
    scored_items = []
    
    # Process different debt categories
    categories = ["technical_debt", "research_debt", "process_debt", "data_debt"]
    
    for category in categories:
        items = sdl.get(category, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    score_info = calculate_priority_score(item)
                    item_copy = item.copy()
                    item_copy["_category"] = category
                    item_copy["_priority_score"] = score_info
                    scored_items.append(item_copy)
    
    # Sort by total score (descending)
    scored_items.sort(key=lambda x: x.get("_priority_score", {}).get("total", 0), reverse=True)
    
    return scored_items


def update_sdl_with_scores() -> dict:
    """Update SDL file with calculated priority scores."""
    sdl = load_sdl()
    
    categories = ["technical_debt", "research_debt", "process_debt", "data_debt"]
    
    for category in categories:
        items = sdl.get(category, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    score_info = calculate_priority_score(item)
                    item["priority_score"] = score_info["total"]
                    item["priority_components"] = score_info["components"]
    
    # Add metadata
    sdl["_priority_meta"] = {
        "last_scored": datetime.now(timezone.utc).isoformat(),
        "weights": WEIGHTS
    }
    
    save_sdl(sdl)
    
    return sdl


def get_top_priorities(n: int = 10) -> list:
    """Get top N priority debt items."""
    scored = score_all_items()
    return scored[:n]


def show_scores() -> None:
    """Display all scored debt items."""
    scored = score_all_items()
    
    print("\n=== STRATEGIC DEBT PRIORITY SCORES ===")
    print(f"Total items: {len(scored)}")
    print(f"Weights: age={WEIGHTS['age']}, impact={WEIGHTS['impact']}, "
          f"effort={WEIGHTS['effort']}, blocking={WEIGHTS['blocking']}")
    print()
    
    for i, item in enumerate(scored, 1):
        score = item.get("_priority_score", {})
        print(f"{i}. [{score.get('total', 0):.1f}] {item.get('id', 'unknown')}: "
              f"{item.get('title', item.get('description', 'No title'))[:50]}")
        print(f"   Category: {item.get('_category')} | "
              f"Impact: {item.get('impact', 'unknown')} | "
              f"Effort: {item.get('effort', 'unknown')}")
        
        components = score.get("components", {})
        print(f"   Scores: age={components.get('age', 0)}, "
              f"impact={components.get('impact', 0)}, "
              f"effort={components.get('effort', 0)}, "
              f"blocking={components.get('blocking', 0)}")
        print()
    
    print("======================================\n")


def main():
    parser = argparse.ArgumentParser(description="Strategic Debt Priority Scorer (PP-030)")
    parser.add_argument("command", choices=["score", "top", "update"],
                       help="Command to execute")
    parser.add_argument("n", nargs="?", type=int, default=10,
                       help="Number of items to show (for top command)")
    
    args = parser.parse_args()
    
    if args.command == "score":
        show_scores()
    
    elif args.command == "top":
        top = get_top_priorities(args.n)
        
        print(f"\n=== TOP {args.n} PRIORITY DEBT ITEMS ===")
        for i, item in enumerate(top, 1):
            score = item.get("_priority_score", {}).get("total", 0)
            title = item.get("title", item.get("description", "No title"))[:50]
            print(f"{i}. [{score:.1f}] {item.get('id', 'unknown')}: {title}")
        print("=================================\n")
    
    elif args.command == "update":
        update_sdl_with_scores()
        print("[OK] SDL updated with priority scores")
        print(f"     File: {SDL_FILE}")


if __name__ == "__main__":
    main()
