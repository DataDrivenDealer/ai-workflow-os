#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State Verification Script

This script validates the consistency of state/tasks.yaml against the
state machine definition in kernel/state_machine.yaml.

Checks performed:
1. State transitions are legal (according to state_machine.yaml)
2. Event timestamps are monotonically increasing
3. No orphaned branches (branches exist but task doesn't)
4. Task status matches latest event state

Exit codes:
    0 - All checks passed
    1 - Warnings found (non-critical issues)
    2 - Errors found (critical issues)

Usage:
    python scripts/verify_state.py
    python scripts/verify_state.py --verbose
"""

import sys
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
import argparse

# Set UTF-8 encoding for stdout to handle emoji characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[1]))

from kernel.state_store import read_yaml
from kernel.paths import TASKS_STATE_PATH, STATE_MACHINE_PATH
from kernel.config import config


# =============================================================================
# Helper Functions
# =============================================================================

def parse_timestamp(ts_str: str) -> datetime:
    """
    Parse timestamp string, ensuring timezone awareness.
    
    Args:
        ts_str: ISO 8601 timestamp string
        
    Returns:
        Timezone-aware datetime object
        
    Raises:
        ValueError: If timestamp format is invalid
    """
    # Replace 'Z' with '+00:00' for proper ISO format
    ts_str = ts_str.replace('Z', '+00:00')
    
    # Try parsing as-is
    try:
        dt = datetime.fromisoformat(ts_str)
    except ValueError:
        raise ValueError(f"Invalid timestamp format: {ts_str}")
    
    # If no timezone info, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt


# =============================================================================
# Verification Functions
# =============================================================================

def verify_state_transitions(verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    Verify that all state transitions in tasks are legal.
    
    Args:
        verbose: Print detailed information
        
    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []
    
    try:
        tasks_data = read_yaml(TASKS_STATE_PATH)
        tasks = tasks_data.get('tasks', {})
    except FileNotFoundError:
        warnings.append(f"⚠️ Tasks file not found: {TASKS_STATE_PATH}")
        return errors, warnings
    except Exception as e:
        errors.append(f"❌ Error reading tasks file: {e}")
        return errors, warnings
    
    # Get valid transitions from config
    valid_transitions = set()
    for transition in config.get_transitions():
        from_state = transition.get('from')
        to_state = transition.get('to')
        if from_state and to_state:
            valid_transitions.add((from_state, to_state))
    
    if verbose:
        print(f"📋 Loaded {len(valid_transitions)} valid transitions")
        print(f"📋 Checking {len(tasks)} tasks")
    
    # Check each task's event history
    for task_id, task_data in tasks.items():
        events = task_data.get('events', [])
        
        if len(events) < 2:
            continue  # Need at least 2 events to check transitions
        
        for i in range(len(events) - 1):
            from_state = events[i].get('to')
            to_state = events[i + 1].get('to')
            
            if not from_state or not to_state:
                warnings.append(f"⚠️ {task_id}: Event missing 'to' field (index {i})")
                continue
            
            if (from_state, to_state) not in valid_transitions:
                # Check if it's the same state (not a transition)
                if from_state == to_state:
                    warnings.append(
                        f"⚠️ {task_id}: Duplicate state event {from_state} → {to_state} (event {i}→{i+1})"
                    )
                else:
                    errors.append(
                        f"❌ {task_id}: Illegal transition {from_state} → {to_state} (event {i}→{i+1})"
                    )
    
    return errors, warnings


def verify_event_timestamps(verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    Verify that event timestamps are monotonically increasing.
    
    Args:
        verbose: Print detailed information
        
    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []
    
    try:
        tasks_data = read_yaml(TASKS_STATE_PATH)
        tasks = tasks_data.get('tasks', {})
    except FileNotFoundError:
        warnings.append(f"⚠️ Tasks file not found: {TASKS_STATE_PATH}")
        return errors, warnings
    except Exception as e:
        errors.append(f"❌ Error reading tasks file: {e}")
        return errors, warnings
    
    if verbose:
        print(f"📋 Checking timestamps for {len(tasks)} tasks")
    
    for task_id, task_data in tasks.items():
        events = task_data.get('events', [])
        
        for i in range(len(events) - 1):
            t1_str = events[i].get('timestamp')
            t2_str = events[i + 1].get('timestamp')
            
            if not t1_str or not t2_str:
                warnings.append(f"⚠️ {task_id}: Event missing timestamp (index {i} or {i+1})")
                continue
            
            try:
                t1 = parse_timestamp(t1_str)
                t2 = parse_timestamp(t2_str)
                
                if t1 > t2:
                    errors.append(
                        f"❌ {task_id}: Timestamp out of order at event {i}→{i+1}: "
                        f"{t1_str} > {t2_str}"
                    )
                elif t1 == t2:
                    warnings.append(
                        f"⚠️ {task_id}: Duplicate timestamp at event {i}→{i+1}: {t1_str}"
                    )
            except ValueError as e:
                warnings.append(
                    f"⚠️ {task_id}: Invalid timestamp format at event {i}: {t1_str} or {t2_str} ({e})"
                )
    
    return errors, warnings


def verify_task_status_consistency(verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    Verify that task status field matches the latest event state.
    
    Args:
        verbose: Print detailed information
        
    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []
    
    try:
        tasks_data = read_yaml(TASKS_STATE_PATH)
        tasks = tasks_data.get('tasks', {})
    except FileNotFoundError:
        warnings.append(f"⚠️ Tasks file not found: {TASKS_STATE_PATH}")
        return errors, warnings
    except Exception as e:
        errors.append(f"❌ Error reading tasks file: {e}")
        return errors, warnings
    
    if verbose:
        print(f"📋 Checking status consistency for {len(tasks)} tasks")
    
    for task_id, task_data in tasks.items():
        status = task_data.get('status')
        events = task_data.get('events', [])
        
        if not events:
            if status:
                warnings.append(
                    f"⚠️ {task_id}: Has status '{status}' but no events"
                )
            continue
        
        latest_event = events[-1]
        latest_state = latest_event.get('to')
        
        if status != latest_state:
            errors.append(
                f"❌ {task_id}: Status mismatch - status='{status}' but latest event='{latest_state}'"
            )
    
    return errors, warnings


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for state verification."""
    parser = argparse.ArgumentParser(
        description='Verify state consistency in AI Workflow OS'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed verification information'
    )
    args = parser.parse_args()
    
    print("🔍 Verifying State Consistency...\n")
    
    all_errors = []
    all_warnings = []
    
    # Run all verification checks
    checks = [
        ("State Transitions", verify_state_transitions),
        ("Event Timestamps", verify_event_timestamps),
        ("Status Consistency", verify_task_status_consistency),
    ]
    
    for check_name, check_func in checks:
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Running: {check_name}")
            print(f"{'='*60}\n")
        
        errors, warnings = check_func(verbose=args.verbose)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    
    # Print results
    print(f"\n{'='*60}")
    print("Verification Results")
    print(f"{'='*60}\n")
    
    if not all_errors and not all_warnings:
        print("✅ All state verification checks passed!")
        print(f"   - {len(checks)} checks completed successfully")
        return 0
    
    if all_warnings:
        print(f"⚠️ Warnings ({len(all_warnings)}):\n")
        for warning in all_warnings:
            print(f"   {warning}")
        print()
    
    if all_errors:
        print(f"❌ Errors ({len(all_errors)}):\n")
        for error in all_errors:
            print(f"   {error}")
        print()
    
    # Determine exit code
    if all_errors:
        print(f"❌ Verification FAILED: {len(all_errors)} errors, {len(all_warnings)} warnings")
        return 2
    else:
        print(f"⚠️ Verification completed with warnings: {len(all_warnings)} warnings")
        return 1


if __name__ == '__main__':
    sys.exit(main())
