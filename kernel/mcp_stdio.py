#!/usr/bin/env python3
"""
MCP Stdio Protocol Handler

Implements the Model Context Protocol (MCP) over stdio transport.
This is the production entry point for connecting AI agents to the OS.

Protocol: JSON-RPC 2.0 over stdio
Transport: Line-delimited JSON (newline-separated)

Spec Reference: https://modelcontextprotocol.io/specification

Usage:
    python -m kernel.mcp_stdio

    Or pipe directly:
    echo '{"jsonrpc":"2.0","method":"initialize","id":1}' | python -m kernel.mcp_stdio
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure logging to stderr (stdout is for protocol messages)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_stdio")

# Add kernel to path
KERNEL_DIR = Path(__file__).parent
ROOT_DIR = KERNEL_DIR.parent
sys.path.insert(0, str(KERNEL_DIR))

from mcp_server import MCPServer, create_server

# =============================================================================
# MCP Protocol Constants
# =============================================================================

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ai-workflow-os"
SERVER_VERSION = "0.1.0"

# JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP-specific Error Codes
TOOL_NOT_FOUND = -32001
TOOL_EXECUTION_ERROR = -32002
RESOURCE_NOT_FOUND = -32003


# =============================================================================
# JSON-RPC 2.0 Message Types
# =============================================================================

class JSONRPCError(Exception):
    """JSON-RPC error with code and message."""
    
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC error object."""
        error = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            error["data"] = self.data
        return error


def make_response(id: Union[str, int, None], result: Any) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 response."""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    }


def make_error_response(
    id: Union[str, int, None], 
    code: int, 
    message: str, 
    data: Any = None
) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 error response."""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": error,
    }


def make_notification(method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 notification (no id, no response expected)."""
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params:
        msg["params"] = params
    return msg


# =============================================================================
# MCP Stdio Server
# =============================================================================

class MCPStdioServer:
    """
    MCP Server implementing stdio transport.
    
    Handles the full MCP protocol lifecycle:
    1. Initialize handshake
    2. Capability negotiation
    3. Tool discovery (tools/list)
    4. Tool execution (tools/call)
    5. Resource operations (resources/list, resources/read)
    6. Graceful shutdown
    """
    
    def __init__(self):
        self.mcp_server = create_server()
        self.initialized = False
        self.client_info: Optional[Dict[str, Any]] = None
        self.session_token: Optional[str] = None
        self._running = True
        
    # =========================================================================
    # Protocol Methods
    # =========================================================================
    
    def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'initialize' request.
        
        This is the first message in the MCP protocol.
        Client sends capabilities, server responds with its capabilities.
        """
        self.client_info = params.get("clientInfo", {})
        protocol_version = params.get("protocolVersion", MCP_PROTOCOL_VERSION)
        
        client_name = self.client_info.get('name', 'unknown') if self.client_info else 'unknown'
        logger.info(f"Initialize request from: {client_name}")
        logger.info(f"Client protocol version: {protocol_version}")
        
        # Server capabilities
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "listChanged": False,  # We don't dynamically change tools
                },
                "resources": {
                    "subscribe": False,
                    "listChanged": False,
                },
                "prompts": {
                    "listChanged": False,
                },
                "logging": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
            "instructions": (
                "AI Workflow OS MCP Server. "
                "First create a session with session_create, then use the session_token "
                "for all subsequent operations. Respect Role Mode permissions."
            ),
        }
    
    def handle_initialized(self, params: Dict[str, Any]) -> None:
        """
        Handle 'initialized' notification.
        
        Client confirms initialization is complete.
        This is a notification (no response).
        """
        self.initialized = True
        logger.info("Client confirmed initialization")
    
    def handle_shutdown(self, params: Dict[str, Any]) -> None:
        """
        Handle 'shutdown' request.
        
        Prepare for server termination.
        """
        logger.info("Shutdown requested")
        self._running = False
        return None
    
    def handle_exit(self, params: Dict[str, Any]) -> None:
        """
        Handle 'exit' notification.
        
        Actually exit the server process.
        """
        logger.info("Exit notification received")
        self._running = False
    
    # =========================================================================
    # Tools Methods
    # =========================================================================
    
    def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'tools/list' request.
        
        Returns the list of available tools with their schemas.
        """
        tools = self.mcp_server.get_tools()
        
        # Convert to MCP format
        mcp_tools = []
        for tool in tools:
            mcp_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {"type": "object"}),
            })
        
        logger.info(f"Listed {len(mcp_tools)} tools")
        return {"tools": mcp_tools}
    
    def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'tools/call' request.
        
        Execute a tool and return results.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            raise JSONRPCError(INVALID_PARAMS, "Missing 'name' parameter")
        
        logger.info(f"Tool call: {tool_name}")
        logger.debug(f"Arguments: {arguments}")
        
        try:
            result = self.mcp_server.call_tool(tool_name, arguments)
            
            # MCP tools/call response format
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2, ensure_ascii=False),
                    }
                ],
                "isError": "error" in result and not result.get("success", True),
            }
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"error": str(e)}, ensure_ascii=False),
                    }
                ],
                "isError": True,
            }
    
    # =========================================================================
    # Resources Methods
    # =========================================================================
    
    def handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'resources/list' request.
        
        List available resources (specs, tasks, artifacts).
        """
        resources = []
        
        # Add spec registry as a resource
        spec_registry = ROOT_DIR / "spec_registry.yaml"
        if spec_registry.exists():
            resources.append({
                "uri": "file://spec_registry.yaml",
                "name": "Spec Registry",
                "description": "Registry of all specifications",
                "mimeType": "application/x-yaml",
            })
        
        # Add canon specs
        canon_dir = ROOT_DIR / "specs" / "canon"
        if canon_dir.exists():
            for f in canon_dir.glob("*.md"):
                resources.append({
                    "uri": f"file://specs/canon/{f.name}",
                    "name": f.stem,
                    "description": f"Canon specification: {f.stem}",
                    "mimeType": "text/markdown",
                })
        
        # Add task files
        tasks_dir = ROOT_DIR / "tasks"
        if tasks_dir.exists():
            for f in tasks_dir.glob("TASK_*.md"):
                resources.append({
                    "uri": f"file://tasks/{f.name}",
                    "name": f.stem,
                    "description": f"TaskCard: {f.stem}",
                    "mimeType": "text/markdown",
                })
        
        logger.info(f"Listed {len(resources)} resources")
        return {"resources": resources}
    
    def handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'resources/read' request.
        
        Read a resource by URI.
        """
        uri = params.get("uri", "")
        
        if not uri.startswith("file://"):
            raise JSONRPCError(RESOURCE_NOT_FOUND, f"Unsupported URI scheme: {uri}")
        
        # Convert URI to path
        rel_path = uri[7:]  # Remove "file://"
        file_path = ROOT_DIR / rel_path
        
        if not file_path.exists():
            raise JSONRPCError(RESOURCE_NOT_FOUND, f"Resource not found: {uri}")
        
        # Security: ensure path is within workspace
        try:
            file_path.resolve().relative_to(ROOT_DIR.resolve())
        except ValueError:
            raise JSONRPCError(RESOURCE_NOT_FOUND, "Access denied: path outside workspace")
        
        # Read content
        content = file_path.read_text(encoding="utf-8")
        
        # Determine mime type
        mime_type = "text/plain"
        if file_path.suffix == ".md":
            mime_type = "text/markdown"
        elif file_path.suffix == ".yaml" or file_path.suffix == ".yml":
            mime_type = "application/x-yaml"
        elif file_path.suffix == ".json":
            mime_type = "application/json"
        elif file_path.suffix == ".py":
            mime_type = "text/x-python"
        
        logger.info(f"Read resource: {uri}")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": mime_type,
                    "text": content,
                }
            ]
        }
    
    # =========================================================================
    # Prompts Methods (Optional)
    # =========================================================================
    
    def handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'prompts/list' request.
        
        List available prompt templates.
        """
        prompts = [
            {
                "name": "create_task",
                "description": "Template for creating a new task",
                "arguments": [
                    {
                        "name": "task_title",
                        "description": "Title of the task",
                        "required": True,
                    },
                    {
                        "name": "pipeline_stage",
                        "description": "Pipeline stage (research, dev, eval, etc.)",
                        "required": True,
                    },
                ],
            },
            {
                "name": "governance_check",
                "description": "Verify output against governance rules",
                "arguments": [
                    {
                        "name": "output_text",
                        "description": "The text to verify",
                        "required": True,
                    },
                ],
            },
        ]
        
        return {"prompts": prompts}
    
    def handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'prompts/get' request.
        
        Get a prompt template with arguments filled in.
        """
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name == "create_task":
            return {
                "description": "Create a new task",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Create a new task with title: {arguments.get('task_title', 'Untitled')}\n"
                                    f"Pipeline stage: {arguments.get('pipeline_stage', 'dev')}\n"
                                    f"Use the session_create and task_* tools to create and manage the task.",
                        },
                    }
                ],
            }
        elif name == "governance_check":
            return {
                "description": "Verify governance compliance",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Please verify the following output against governance rules:\n\n"
                                    f"```\n{arguments.get('output_text', '')}\n```\n\n"
                                    f"Use the governance_check tool to verify compliance.",
                        },
                    }
                ],
            }
        else:
            raise JSONRPCError(INVALID_PARAMS, f"Unknown prompt: {name}")
    
    # =========================================================================
    # Logging Methods
    # =========================================================================
    
    def handle_logging_setLevel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'logging/setLevel' request.
        
        Set the logging level.
        """
        level = params.get("level", "info").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.info(f"Log level set to: {level}")
        return {}
    
    # =========================================================================
    # Message Router
    # =========================================================================
    
    def route_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Route a JSON-RPC message to the appropriate handler.
        
        Returns response dict or None for notifications.
        """
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")
        
        # Map methods to handlers
        handlers = {
            # Lifecycle
            "initialize": self.handle_initialize,
            "initialized": self.handle_initialized,
            "shutdown": self.handle_shutdown,
            "exit": self.handle_exit,
            # Tools
            "tools/list": self.handle_tools_list,
            "tools/call": self.handle_tools_call,
            # Resources
            "resources/list": self.handle_resources_list,
            "resources/read": self.handle_resources_read,
            # Prompts
            "prompts/list": self.handle_prompts_list,
            "prompts/get": self.handle_prompts_get,
            # Logging
            "logging/setLevel": self.handle_logging_setLevel,
        }
        
        if not method:
            if msg_id is not None:
                return make_error_response(msg_id, INVALID_REQUEST, "Method is required")
            return None
        
        handler = handlers.get(method)
        if not handler:
            if msg_id is not None:
                return make_error_response(msg_id, METHOD_NOT_FOUND, f"Unknown method: {method}")
            return None  # Ignore unknown notifications
        
        try:
            result = handler(params)
            
            # Notifications don't get responses
            if msg_id is None:
                return None
            
            return make_response(msg_id, result)
            
        except JSONRPCError as e:
            if msg_id is not None:
                return make_error_response(msg_id, e.code, e.message, e.data)
            return None
        except Exception as e:
            logger.exception(f"Error handling {method}")
            if msg_id is not None:
                return make_error_response(msg_id, INTERNAL_ERROR, str(e))
            return None
    
    # =========================================================================
    # Stdio Transport
    # =========================================================================
    
    def send_message(self, message: Dict[str, Any]):
        """Send a JSON-RPC message to stdout."""
        try:
            line = json.dumps(message, ensure_ascii=False)
            print(line, flush=True)
            logger.debug(f"Sent: {line[:200]}...")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Send a notification to the client."""
        self.send_message(make_notification(method, params))
    
    def process_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Process a single line of input.
        
        Returns response dict or None.
        """
        line = line.strip()
        if not line:
            return None
        
        logger.debug(f"Received: {line[:200]}...")
        
        try:
            message = json.loads(line)
        except json.JSONDecodeError as e:
            return make_error_response(None, PARSE_ERROR, f"Invalid JSON: {e}")
        
        # Validate JSON-RPC 2.0
        if message.get("jsonrpc") != "2.0":
            return make_error_response(
                message.get("id"),
                INVALID_REQUEST,
                "Missing or invalid 'jsonrpc' field (must be '2.0')"
            )
        
        if "method" not in message:
            return make_error_response(
                message.get("id"),
                INVALID_REQUEST,
                "Missing 'method' field"
            )
        
        return self.route_message(message)
    
    def run_sync(self):
        """
        Run the server synchronously (blocking).
        
        Reads from stdin, writes to stdout.
        """
        logger.info(f"MCP Stdio Server starting (version {SERVER_VERSION})")
        logger.info(f"Protocol version: {MCP_PROTOCOL_VERSION}")
        
        # Reconfigure for UTF-8
        if sys.platform == "win32":
            if hasattr(sys.stdin, 'reconfigure'):
                sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        
        try:
            for line in sys.stdin:
                if not self._running:
                    break
                
                response = self.process_line(line)
                if response:
                    self.send_message(response)
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception("Fatal error in main loop")
        finally:
            logger.info("Server shutting down")
    
    async def run_async(self):
        """
        Run the server asynchronously.
        
        Useful for integration with async frameworks.
        """
        logger.info(f"MCP Stdio Server starting (async mode)")
        
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        
        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)
        
        try:
            while self._running:
                line = await reader.readline()
                if not line:
                    break
                
                line_str = line.decode("utf-8")
                response = self.process_line(line_str)
                
                if response:
                    writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
                    writer.write(b"\n")
                    await writer.drain()
                    
        except asyncio.CancelledError:
            logger.info("Async server cancelled")
        finally:
            writer.close()
            logger.info("Async server shutdown complete")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MCP Stdio Protocol Handler for AI Workflow OS"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async mode"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick self-test"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    if args.test:
        run_self_test()
        return
    
    server = MCPStdioServer()
    
    if args.use_async:
        asyncio.run(server.run_async())
    else:
        server.run_sync()


def run_self_test():
    """Run a quick self-test of the protocol handler."""
    print("=" * 60)
    print("MCP Stdio Protocol Handler - Self Test")
    print("=" * 60)
    
    server = MCPStdioServer()
    
    # Test messages
    test_messages = [
        # Initialize
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                },
                "capabilities": {}
            }
        },
        # Initialized notification
        {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        },
        # List tools
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        },
        # List resources
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        },
        # Call a tool
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "spec_list",
                "arguments": {
                    "session_token": "test-token"  # Will fail but tests the flow
                }
            }
        },
        # Shutdown
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "shutdown",
            "params": {}
        },
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\n[Test {i}] {msg['method']}")
        print(f"  Request: {json.dumps(msg)[:80]}...")
        
        response = server.route_message(msg)
        
        if response:
            result_str = json.dumps(response, indent=2, ensure_ascii=False)
            # Truncate long results
            if len(result_str) > 500:
                result_str = result_str[:500] + "...\n  (truncated)"
            print(f"  Response:\n{result_str}")
        else:
            print("  Response: (notification, no response)")
    
    print("\n" + "=" * 60)
    print("✅ Self-test completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
