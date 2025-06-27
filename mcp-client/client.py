#!/usr/bin/env python3
"""
Simple MCP Client Example

A minimal MCP client demonstrating how to connect to MCP servers and call tools.
This client connects to servers via stdio transport and provides basic functionality
for listing tools/resources and making tool calls.
"""

import asyncio
import sys
from typing import Optional

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class SimpleMCPClient:
    """A simple MCP client for connecting to and interacting with MCP servers."""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio_client = None

    async def connect(
        self, server_command: list[str], server_env: Optional[dict] = None
    ):
        """Connect to an MCP server via stdio transport."""
        print(f"🔄 Connecting to server: {' '.join(server_command)}")

        server_params = StdioServerParameters(
            command=server_command[0], args=server_command[1:], env=server_env
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def list_tools(self):
        """List all available tools from the connected server."""
        if not self.session:
            print("❌ Not connected to any server")
            return

        try:
            tools_result = await self.session.list_tools()
            print(f"\n📋 Available Tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"  • {tool.name}: {tool.description}")
                if tool.inputSchema and "properties" in tool.inputSchema:
                    params = list(tool.inputSchema["properties"].keys())
                    print(f"    Parameters: {', '.join(params)}")
        except Exception as e:
            print(f"❌ Error listing tools: {e}")

    async def list_resources(self):
        """List all available resources from the connected server."""
        if not self.session:
            print("❌ Not connected to any server")
            return

        try:
            resources_result = await self.session.list_resources()
            print(f"\n📂 Available Resources ({len(resources_result.resources)}):")
            for resource in resources_result.resources:
                print(f"  • {resource.uri}: {resource.name}")
                if resource.description:
                    print(f"    {resource.description}")
        except Exception as e:
            print(f"❌ Error listing resources: {e}")

    async def call_tool(self, tool_name: str, arguments: dict = None):
        """Call a specific tool with the given arguments."""
        if not self.session:
            print("❌ Not connected to any server")
            return

        arguments = arguments or {}

        try:
            print(f"\n🔧 Calling tool '{tool_name}' with arguments: {arguments}")
            result = await self.session.call_tool(tool_name, arguments)

            print(f"✅ Tool call successful:")
            for content in result.content:
                if hasattr(content, 'type') and content.type == "text":
                    print(f"  {content.text}")
                elif hasattr(content, 'text'):
                    print(f"  {content.text}")
                else:
                    print(f"  {content}")
        except Exception as e:
            print(f"❌ Error calling tool '{tool_name}': {e}")

    async def get_resource(self, uri: str):
        """Read a specific resource from the server."""
        if not self.session:
            print("❌ Not connected to any server")
            return

        try:
            print(f"\n📖 Reading resource: {uri}")
            result = await self.session.read_resource(uri)

            print(f"✅ Resource content:")
            for content in result.contents:
                if hasattr(content, 'type') and content.type == "text":
                    print(f"  {content.text}")
                elif hasattr(content, 'text'):
                    print(f"  {content.text}")
                else:
                    print(f"  {content}")
        except Exception as e:
            print(f"❌ Error reading resource '{uri}': {e}")

    async def close(self):
        """Close the connection to the MCP server."""
        try:
            await self.exit_stack.aclose()
        except (RuntimeError, asyncio.CancelledError, Exception):
            # Suppress cleanup errors during shutdown
            pass


async def main():
    """Interactive CLI for testing MCP servers."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_command> [args...]")
        print("Examples:")
        print("  python client.py python ../llm-api-bridge/server.py")
        print("  python client.py python ../arxiv-downloader/server.py")
        return

    client = SimpleMCPClient()
    server_command = sys.argv[1:]

    try:
        await client.connect(server_command)

        # Show available capabilities
        await client.list_tools()
        await client.list_resources()

        print("\n" + "=" * 50)
        print("🚀 Interactive MCP Client")
        print("Commands:")
        print("  tools           - List available tools")
        print("  resources       - List available resources")
        print("  call <tool>     - Call a tool (will prompt for arguments)")
        print("  get <uri>       - Get a resource")
        print("  quit            - Exit the client")
        print("=" * 50)

        while True:
            try:
                command = input("\n> ").strip().split()
                if not command:
                    continue

                if command[0] == "quit":
                    break
                elif command[0] == "tools":
                    await client.list_tools()
                elif command[0] == "resources":
                    await client.list_resources()
                elif command[0] == "call" and len(command) > 1:
                    tool_name = command[1]
                    print(
                        f"Enter arguments for '{tool_name}' as JSON (or press Enter for no args):"
                    )
                    args_input = input("Arguments: ").strip()
                    arguments = {}
                    if args_input:
                        try:
                            import json

                            arguments = json.loads(args_input)
                        except json.JSONDecodeError:
                            print("❌ Invalid JSON format")
                            continue
                    await client.call_tool(tool_name, arguments)
                elif command[0] == "get" and len(command) > 1:
                    uri = command[1]
                    await client.get_resource(uri)
                else:
                    print("❌ Unknown command or missing arguments")

            except KeyboardInterrupt:
                break
            except EOFError:
                break

    except Exception as e:
        print(f"❌ Connection error: {e}")

    finally:
        await client.close()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
