# Simple MCP Client

A minimal MCP (Model Context Protocol) client example demonstrating how to connect to and interact with MCP servers. This client provides a practical reference for building custom MCP clients in Python.

## Features

- 🔌 **Stdio Transport**: Connect to MCP servers via stdio
- 🛠️ **Tool Calling**: List and call available tools with parameters
- 📂 **Resource Access**: List and read server resources
- 🎯 **Interactive CLI**: User-friendly command-line interface
- 📝 **Example Scripts**: Ready-to-run examples for existing servers

## Quick Start

### Installation

```bash
cd mcp-client
pip install -r requirements.txt
```

### Basic Usage

Connect to an MCP server and start interactive mode:

```bash
python client.py python ../llm-api-bridge/server.py
```

### Available Commands

Once connected, use these interactive commands:

- `tools` - List available tools
- `resources` - List available resources  
- `call <tool_name>` - Call a tool (prompts for JSON arguments)
- `get <uri>` - Read a resource
- `quit` - Exit the client

## Example Usage

### Testing LLM API Bridge

```bash
python client.py python ../llm-api-bridge/server.py
```

Then try these commands:
```
> tools
> call echo_test
> call list_models
> get info://server
```

### Testing ArXiv Downloader

```bash
python client.py python ../arxiv-downloader/server.py
```

Then try these commands:
```
> tools
> call search_papers
> call list_categories
> get info://server
```

## Programmatic Usage

```python
from client import SimpleMCPClient

async def example():
    client = SimpleMCPClient()
    
    # Connect to server
    await client.connect(["python", "server.py"])
    
    # List tools
    await client.list_tools()
    
    # Call a tool
    await client.call_tool("echo_test", {"message": "Hello!"})
    
    # Clean up
    await client.close()
```

## Examples

Run the example script to test both servers:

```bash
cd examples
python test_servers.py
```

This provides an interactive menu to test:
- LLM API Bridge server
- ArXiv Downloader server
- Both servers sequentially
- Custom server commands

## Architecture

The client implements these key components:

- **SimpleMCPClient**: Main client class handling connections and operations
- **Stdio Transport**: Communication with servers via stdin/stdout
- **Session Management**: MCP protocol session handling
- **Error Handling**: Comprehensive error reporting and recovery

## Use Cases

This client is useful for:

- **Development**: Testing MCP servers during development
- **Integration**: Building custom applications that consume MCP services
- **Learning**: Understanding MCP client implementation patterns
- **Debugging**: Troubleshooting server behavior and responses

## Dependencies

- `mcp>=1.0.0`: Official MCP Python library

## Related Projects

This client works with the MCP servers in this repository:

- [LLM API Bridge](../llm-api-bridge/) - Multi-provider LLM API access
- [ArXiv Downloader](../arxiv-downloader/) - Academic paper search and download

See the main [project README](../README.md) for complete documentation.