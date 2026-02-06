#!/usr/bin/env python3
"""
Context Checkpoint Script — Session State Persistence

Purpose: Easy-to-use wrapper for creating and restoring context checkpoints.
         Designed for cross-session context handoff.

Usage:
    # Before ending a session, save context
    python scripts/context_checkpoint.py save --session SESSION_001 \\
        --summary "Completed Phase B of Prompt Prayer audit" \\
        --findings "8 P0 items enforced" "Gate framework created" \\
        --pending "Phase C implementation" "P1 items review"
    
    # At start of new session, restore context
    python scripts/context_checkpoint.py load --session SESSION_001
    
    # List all checkpoints
    python scripts/context_checkpoint.py list
    
    # Auto-assess and checkpoint if needed
    python scripts/context_checkpoint.py auto --session SESSION_001 \\
        --tokens 75000 --files 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))

try:
    from kernel.context_hygiene import (
        assess_context,
        create_checkpoint,
        restore_checkpoint,
        list_checkpoints,
        ContextMetrics,
        ContextStatus,
        DelegationAction,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure kernel/context_hygiene.py exists")
    sys.exit(1)


def cmd_save(args: argparse.Namespace) -> int:
    """Save a context checkpoint."""
    # Build metrics from args if provided
    metrics = ContextMetrics(
        estimated_tokens=args.tokens or 0,
        file_count=args.files or 0,
        call_depth=args.depth or 0,
        topics=args.topics or [],
    )
    
    checkpoint = create_checkpoint(
        session_id=args.session,
        metrics=metrics,
        summary=args.summary or "Manual checkpoint",
        files_snapshot=args.snapshot or [],
        key_findings=args.findings or [],
        pending_tasks=args.pending or [],
    )
    
    print(f"💾 Checkpoint Saved")
    print(f"   Session:    {checkpoint.session_id}")
    print(f"   Checkpoint: {checkpoint.checkpoint_id}")
    print(f"   Created:    {checkpoint.created_at[:19]}")
    print()
    print(f"📋 Summary: {checkpoint.summary}")
    
    if checkpoint.key_findings:
        print(f"\n🔍 Key Findings ({len(checkpoint.key_findings)}):")
        for i, f in enumerate(checkpoint.key_findings, 1):
            print(f"   {i}. {f}")
    
    if checkpoint.pending_tasks:
        print(f"\n📝 Pending Tasks ({len(checkpoint.pending_tasks)}):")
        for i, t in enumerate(checkpoint.pending_tasks, 1):
            print(f"   {i}. {t}")
    
    print()
    print(f"✅ To restore: python scripts/context_checkpoint.py load --session {args.session}")
    
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    """Load a context checkpoint."""
    checkpoint = restore_checkpoint(args.session, args.checkpoint_id)
    
    if not checkpoint:
        print(f"❌ No checkpoint found for session '{args.session}'")
        print(f"   Use 'python scripts/context_checkpoint.py list' to see available checkpoints")
        return 1
    
    print("=" * 70)
    print(f"📂 CONTEXT RESTORATION — {checkpoint.session_id}")
    print("=" * 70)
    print()
    print(f"🕐 Checkpoint: {checkpoint.checkpoint_id}")
    print(f"   Created:   {checkpoint.created_at[:19]}")
    print()
    print(f"📋 Summary:")
    print(f"   {checkpoint.summary}")
    print()
    
    if checkpoint.key_findings:
        print(f"🔍 Key Findings from Previous Session:")
        for i, f in enumerate(checkpoint.key_findings, 1):
            print(f"   {i}. {f}")
        print()
    
    if checkpoint.pending_tasks:
        print(f"📝 Pending Tasks to Resume:")
        for i, t in enumerate(checkpoint.pending_tasks, 1):
            print(f"   [ ] {i}. {t}")
        print()
    
    print(f"📊 Context at Checkpoint:")
    print(f"   • Estimated Tokens: {checkpoint.metrics.estimated_tokens}")
    print(f"   • Files in Context: {checkpoint.metrics.file_count}")
    print(f"   • Topics: {', '.join(checkpoint.metrics.topics) or 'N/A'}")
    print()
    
    if checkpoint.restoration_notes:
        print(f"📌 Restoration Notes:")
        print(f"   {checkpoint.restoration_notes}")
        print()
    
    print("=" * 70)
    print("✅ Context restored. Ready to continue.")
    print("=" * 70)
    
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available checkpoints."""
    checkpoints = list_checkpoints(args.session if hasattr(args, 'session') else None)
    
    if not checkpoints:
        print("📭 No checkpoints found.")
        print("   Create one with: python scripts/context_checkpoint.py save --session SESSION_ID")
        return 0
    
    print(f"📚 Available Checkpoints ({len(checkpoints)})")
    print()
    print(f"{'Session':<20} {'Checkpoint':<28} {'Created':<20}")
    print("-" * 70)
    
    for cp in checkpoints:
        print(f"{cp['session_id']:<20} {cp['checkpoint_id']:<28} {cp['created_at'][:19]}")
    
    print()
    print("💡 Load with: python scripts/context_checkpoint.py load --session <SESSION>")
    
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Auto-assess and checkpoint if needed."""
    metrics = ContextMetrics(
        estimated_tokens=args.tokens,
        file_count=args.files,
        call_depth=args.depth or 0,
        topics=args.topics or [],
    )
    
    assessment = assess_context(metrics)
    
    print(f"📊 Context Assessment")
    
    status_icons = {
        "healthy": "✅",
        "warning": "⚠️",
        "critical": "🔴",
        "overloaded": "💥",
    }
    icon = status_icons.get(assessment.status.value, "❓")
    print(f"   Status: {icon} {assessment.status.value.upper()}")
    print(f"   Tokens: ~{metrics.estimated_tokens}")
    print(f"   Files:  {metrics.file_count}")
    print()
    
    if assessment.reasons:
        print("⚠️  Issues Detected:")
        for r in assessment.reasons:
            print(f"   • {r}")
        print()
    
    if assessment.checkpoint_suggested or args.force:
        print("💾 Creating automatic checkpoint...")
        
        checkpoint = create_checkpoint(
            session_id=args.session,
            metrics=metrics,
            summary=args.summary or f"Auto-checkpoint: {assessment.status.value}",
            files_snapshot=[],
            key_findings=args.findings or [f"Context {assessment.status.value}"],
            pending_tasks=args.pending or [],
        )
        
        print(f"   ✅ Checkpoint created: {checkpoint.checkpoint_id}")
        
        if assessment.recommended_subagents:
            print()
            print("🤖 Recommended: Delegate to subagents:")
            for s in assessment.recommended_subagents:
                print(f"   • {s}")
        
        if assessment.status in (ContextStatus.CRITICAL, ContextStatus.OVERLOADED):
            print()
            print("🚨 CRITICAL: Consider delegating complex tasks to subagents")
            return 1
    else:
        print("✅ Context is healthy, no checkpoint needed.")
        if args.force:
            print("   (Use --force to checkpoint anyway)")
    
    return 0


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Context Checkpoint Manager - Cross-session state persistence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save checkpoint before ending session
  python scripts/context_checkpoint.py save --session S001 \\
      --summary "Completed feature X" \\
      --findings "Fixed bug Y" "Refactored Z" \\
      --pending "Test coverage" "Documentation"
  
  # Restore at start of new session
  python scripts/context_checkpoint.py load --session S001
  
  # Auto-checkpoint based on context metrics
  python scripts/context_checkpoint.py auto --session S001 --tokens 60000 --files 12
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # save
    save_parser = subparsers.add_parser("save", help="Save a context checkpoint")
    save_parser.add_argument("--session", "-s", required=True, help="Session ID")
    save_parser.add_argument("--summary", help="Session summary")
    save_parser.add_argument("--findings", nargs="*", help="Key findings")
    save_parser.add_argument("--pending", nargs="*", help="Pending tasks")
    save_parser.add_argument("--snapshot", nargs="*", help="Files to snapshot")
    save_parser.add_argument("--tokens", type=int, help="Estimated token count")
    save_parser.add_argument("--files", type=int, help="File count in context")
    save_parser.add_argument("--depth", type=int, help="Call depth")
    save_parser.add_argument("--topics", nargs="*", help="Current topics")
    
    # load
    load_parser = subparsers.add_parser("load", help="Load a checkpoint")
    load_parser.add_argument("--session", "-s", required=True, help="Session ID")
    load_parser.add_argument("--checkpoint-id", help="Specific checkpoint ID")
    
    # list
    list_parser = subparsers.add_parser("list", help="List available checkpoints")
    list_parser.add_argument("--session", "-s", help="Filter by session ID")
    
    # auto
    auto_parser = subparsers.add_parser("auto", help="Auto-assess and checkpoint if needed")
    auto_parser.add_argument("--session", "-s", required=True, help="Session ID")
    auto_parser.add_argument("--tokens", type=int, default=0, help="Estimated token count")
    auto_parser.add_argument("--files", type=int, default=0, help="File count in context")
    auto_parser.add_argument("--depth", type=int, default=0, help="Call depth")
    auto_parser.add_argument("--topics", nargs="*", help="Current topics")
    auto_parser.add_argument("--summary", help="Checkpoint summary if created")
    auto_parser.add_argument("--findings", nargs="*", help="Key findings")
    auto_parser.add_argument("--pending", nargs="*", help="Pending tasks")
    auto_parser.add_argument("--force", action="store_true", help="Force checkpoint creation")
    
    args = parser.parse_args()
    
    if args.command == "save":
        return cmd_save(args)
    elif args.command == "load":
        return cmd_load(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "auto":
        return cmd_auto(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
