#!/usr/bin/env python3
"""
Check MCP Interface Consistency (SYSTEM_INVARIANTS INV-9)

Verifies that MCP Server tools match mcp_server_manifest.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kernel.mcp_server import create_server


def check_mcp_interface() -> bool:
    """Check if MCP Server tools match manifest."""
    manifest_path = ROOT / "mcp_server_manifest.json"
    
    if not manifest_path.exists():
        print("❌ mcp_server_manifest.json not found")
        return False
    
    with manifest_path.open() as f:
        manifest = json.load(f)
    
    manifest_tools = {tool["name"] for tool in manifest.get("tools", [])}
    
    # Get actual tools from server
    server = create_server()
    actual_tools = {tool["name"] for tool in server.get_tools()}
    
    print("MCP Interface Consistency Check")
    print("================================")
    print(f"Manifest tools: {len(manifest_tools)}")
    print(f"Actual tools: {len(actual_tools)}")
    
    missing = manifest_tools - actual_tools
    extra = actual_tools - manifest_tools
    
    if missing:
        print(f"\n❌ Tools in manifest but not implemented:")
        for tool in sorted(missing):
            print(f"  - {tool}")
    
    if extra:
        print(f"\n⚠️ Tools implemented but not in manifest:")
        for tool in sorted(extra):
            print(f"  - {tool}")
    
    if not missing and not extra:
        print("\n✅ All tools match manifest")
        return True
    else:
        return False


if __name__ == "__main__":
    passed = check_mcp_interface()
    sys.exit(0 if passed else 1)
