"""
Spec Proposal Duplicate Detection (PP-023)

Checks for duplicate or similar spec proposals in decisions/ directory
before creating a new proposal.

Usage:
    python kernel/spec_duplicate_check.py check --diff "Add x parameter to Y interface"
    python kernel/spec_duplicate_check.py list                 # List recent proposals
    python kernel/spec_duplicate_check.py similar "query"      # Find similar proposals
"""

from pathlib import Path
from datetime import datetime, timedelta
import argparse
import hashlib
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

DECISIONS_DIR = ROOT / "decisions"
PROPOSALS_INDEX = ROOT / "state" / "spec_proposals_index.yaml"


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase, remove extra whitespace, remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_text_hash(text: str) -> str:
    """Get hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def extract_proposal_content(file_path: Path) -> dict:
    """Extract key content from a proposal file."""
    content = {
        "path": str(file_path),
        "filename": file_path.name,
        "date": None,
        "title": None,
        "description": "",
        "spec_changes": [],
        "hash": None
    }
    
    try:
        text = file_path.read_text(encoding="utf-8")
        
        # Extract date from filename (e.g., 2025-02-05_topic.md)
        date_match = re.match(r'(\d{4}-\d{2}-\d{2})', file_path.name)
        if date_match:
            content["date"] = date_match.group(1)
        
        # Extract title (first # heading)
        title_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        if title_match:
            content["title"] = title_match.group(1).strip()
        
        # Extract description (first paragraph after title)
        desc_match = re.search(r'^#.+?\n\n(.+?)\n\n', text, re.DOTALL)
        if desc_match:
            content["description"] = desc_match.group(1).strip()[:500]
        
        # Extract spec file references
        spec_refs = re.findall(r'`?(specs?/[\w/.-]+\.ya?ml)`?', text, re.IGNORECASE)
        content["spec_changes"] = list(set(spec_refs))
        
        # Compute content hash
        content["hash"] = get_text_hash(text)
        
    except Exception as e:
        content["error"] = str(e)
    
    return content


def load_proposals_index() -> dict:
    """Load or create proposals index."""
    if not yaml:
        return {"proposals": []}
    
    if PROPOSALS_INDEX.exists():
        with open(PROPOSALS_INDEX, encoding="utf-8") as f:
            return yaml.safe_load(f) or {"proposals": []}
    
    return {"proposals": []}


def save_proposals_index(index: dict) -> None:
    """Save proposals index."""
    if not yaml:
        return
    
    PROPOSALS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(PROPOSALS_INDEX, "w", encoding="utf-8") as f:
        yaml.dump(index, f, default_flow_style=False, allow_unicode=True)


def scan_decisions_directory() -> list:
    """Scan decisions directory for proposals."""
    proposals = []
    
    if not DECISIONS_DIR.exists():
        return proposals
    
    for file_path in DECISIONS_DIR.glob("*.md"):
        content = extract_proposal_content(file_path)
        proposals.append(content)
    
    # Sort by date, newest first
    proposals.sort(key=lambda x: x.get("date") or "0000-00-00", reverse=True)
    
    return proposals


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple word overlap similarity."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def check_duplicate(diff_description: str, threshold: float = 0.5) -> dict:
    """Check if a proposal with similar diff already exists."""
    proposals = scan_decisions_directory()
    
    result = {
        "is_duplicate": False,
        "similar_proposals": [],
        "recommendation": "proceed"
    }
    
    diff_normalized = normalize_text(diff_description)
    diff_hash = get_text_hash(diff_description)
    
    for proposal in proposals:
        # Check exact hash match
        if proposal.get("hash") == diff_hash:
            result["is_duplicate"] = True
            result["similar_proposals"].append({
                "file": proposal["filename"],
                "similarity": 1.0,
                "match_type": "exact_hash"
            })
            continue
        
        # Check title similarity
        title = proposal.get("title") or ""
        title_sim = calculate_similarity(diff_description, title)
        
        # Check description similarity
        desc = proposal.get("description") or ""
        desc_sim = calculate_similarity(diff_description, desc)
        
        max_sim = max(title_sim, desc_sim)
        
        if max_sim >= threshold:
            result["similar_proposals"].append({
                "file": proposal["filename"],
                "title": title[:60],
                "similarity": round(max_sim, 2),
                "match_type": "content_similarity"
            })
    
    # Set recommendation
    if result["is_duplicate"]:
        result["recommendation"] = "reject_duplicate"
    elif result["similar_proposals"]:
        if any(p["similarity"] >= 0.8 for p in result["similar_proposals"]):
            result["recommendation"] = "review_existing"
        else:
            result["recommendation"] = "proceed_with_caution"
    
    return result


def find_similar(query: str, top_n: int = 5) -> list:
    """Find proposals similar to query."""
    proposals = scan_decisions_directory()
    scored = []
    
    for proposal in proposals:
        title = proposal.get("title") or ""
        desc = proposal.get("description") or ""
        combined = f"{title} {desc}"
        
        sim = calculate_similarity(query, combined)
        if sim > 0.1:
            scored.append({
                "file": proposal["filename"],
                "title": title[:60],
                "date": proposal.get("date"),
                "similarity": round(sim, 2)
            })
    
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_n]


def list_recent(days: int = 30) -> list:
    """List recent proposals."""
    proposals = scan_decisions_directory()
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    
    recent = [
        p for p in proposals 
        if (p.get("date") or "0000-00-00") >= cutoff_str
    ]
    
    return recent


def main():
    parser = argparse.ArgumentParser(description="Spec Proposal Duplicate Detection (PP-023)")
    parser.add_argument("command", choices=["check", "list", "similar", "scan"],
                       help="Command to execute")
    parser.add_argument("query", nargs="?", help="Diff description or search query")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--days", "-d", type=int, default=30,
                       help="Days to look back for list command")
    parser.add_argument("--top", "-n", type=int, default=5,
                       help="Number of results for similar command")
    
    args = parser.parse_args()
    
    if args.command == "check":
        if not args.query:
            print("[ERROR] Diff description required")
            print("Usage: python kernel/spec_duplicate_check.py check 'Add x to Y interface'")
            return
        
        result = check_duplicate(args.query, args.threshold)
        
        print("\n=== DUPLICATE CHECK RESULT ===")
        print(f"Is Duplicate: {result['is_duplicate']}")
        print(f"Recommendation: {result['recommendation']}")
        
        if result['similar_proposals']:
            print("\nSimilar Proposals:")
            for p in result['similar_proposals']:
                print(f"  - {p['file']} (similarity: {p['similarity']}, type: {p['match_type']})")
                if 'title' in p:
                    print(f"    Title: {p['title']}")
        else:
            print("\n✅ No similar proposals found. Safe to proceed.")
        
        print("==============================\n")
        
        # Exit code for scripting
        if result['is_duplicate']:
            exit(1)
        elif result['recommendation'] == 'review_existing':
            exit(2)
    
    elif args.command == "list":
        recent = list_recent(args.days)
        
        print(f"\n=== RECENT PROPOSALS (last {args.days} days) ===")
        if not recent:
            print("No proposals found.")
        else:
            for p in recent:
                print(f"  {p.get('date', 'unknown')}: {p['filename']}")
                if p.get('title'):
                    print(f"    Title: {p['title'][:60]}")
        print("=====================================\n")
    
    elif args.command == "similar":
        if not args.query:
            print("[ERROR] Search query required")
            return
        
        results = find_similar(args.query, args.top)
        
        print(f"\n=== SIMILAR PROPOSALS ===")
        if not results:
            print("No similar proposals found.")
        else:
            for p in results:
                print(f"  [{p['similarity']}] {p['file']}")
                print(f"    {p.get('title', 'No title')}")
        print("=========================\n")
    
    elif args.command == "scan":
        proposals = scan_decisions_directory()
        
        # Update index
        index = {"proposals": proposals, "scanned_at": datetime.now().isoformat()}
        save_proposals_index(index)
        
        print(f"\n=== SCAN COMPLETE ===")
        print(f"Total proposals: {len(proposals)}")
        print(f"Index saved to: {PROPOSALS_INDEX}")
        print("=====================\n")


if __name__ == "__main__":
    main()
