#!/usr/bin/env python3
"""
Example usage of the Simple MCP Client

This script demonstrates how to use the SimpleMCPClient to connect to
and interact with the MCP servers in this repository.
"""

import asyncio
import sys
import os

# Add parent directory to path to import client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client import SimpleMCPClient


async def test_llm_api_bridge():
    """Test the LLM API Bridge server."""
    print("🤖 Testing LLM API Bridge Server")
    print("=" * 40)

    client = SimpleMCPClient()

    try:
        # Connect to the server
        server_path = os.path.abspath("../../llm-api-bridge/server.py")
        await client.connect(
            ["python", server_path],
            server_env={
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            },
        )

        # List capabilities
        await client.list_tools()
        await client.list_resources()

        # Test echo tool
        await client.call_tool("echo_test", {"message": "Hello from MCP client!"})

        # Test list models
        await client.call_tool("list_models", {"provider": "openai"})

        # Get server info resource
        await client.get_resource("info://server")

    except Exception as e:
        print(f"❌ Error testing LLM API Bridge: {e}")
    finally:
        await client.close()


async def test_arxiv_downloader():
    """Test the ArXiv Downloader server."""
    print("\n📚 Testing ArXiv Downloader Server")
    print("=" * 40)

    client = SimpleMCPClient()

    try:
        # Connect to the server
        server_path = os.path.abspath("../../arxiv-downloader/server.py")
        await client.connect(["python", server_path])

        # List capabilities
        await client.list_tools()
        await client.list_resources()

        # Test echo tool
        await client.call_tool("echo_test", {"message": "Hello from ArXiv client!"})

        # Search for papers
        await client.call_tool(
            "search_papers", {"query": "machine learning", "max_results": 3}
        )

        # List categories
        await client.call_tool("list_categories")

    except Exception as e:
        print(f"❌ Error testing ArXiv Downloader: {e}")
    finally:
        await client.close()


async def interactive_demo():
    """Interactive demo allowing user to choose which server to test."""
    print("🎯 Simple MCP Client Demo")
    print("=" * 30)
    print("Choose a server to test:")
    print("1. LLM API Bridge")
    print("2. ArXiv Downloader")
    print("3. Both servers")
    print("4. Custom server command")

    choice = input("Enter your choice (1-4): ").strip()

    if choice == "1":
        await test_llm_api_bridge()
    elif choice == "2":
        await test_arxiv_downloader()
    elif choice == "3":
        await test_llm_api_bridge()
        await test_arxiv_downloader()
    elif choice == "4":
        command = (
            input("Enter server command (e.g., 'python server.py'): ").strip().split()
        )
        client = SimpleMCPClient()
        try:
            await client.connect(command)
            await client.list_tools()
            await client.list_resources()
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            await client.close()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    asyncio.run(interactive_demo())
