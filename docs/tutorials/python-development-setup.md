# Python Development Setup for MCP Servers

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- Virtual environment support

## Core Python Packages

### 1. Official MCP Python SDK

The primary package for MCP server development:

```bash
pip install mcp
```

**Key Features:**
- Official MCP server framework
- CLI utilities for development and testing
- FastMCP integration (high-level, Pythonic API)
- Complete toolkit for MCP ecosystem

**Latest Version:** mcp-1.9.4 (as of research date)

### 2. Additional Dependencies

For full functionality, install with CLI support:

```bash
pip install "mcp[cli]" httpx
```

**Recommended packages:**
- `httpx`: For HTTP client operations
- `uvicorn`: For running async servers
- `pydantic`: For data validation (often included)

## Development Environment Setup

### 1. Create Virtual Environment

```bash
python -m venv mcp-env
source mcp-env/bin/activate  # On Windows: mcp-env\Scripts\activate
```

### 2. Install Dependencies

Using pip:
```bash
pip install "mcp[cli]" httpx
```

Using uv (recommended for faster installs):
```bash
uv add "mcp[cli]" httpx
```

### 3. Project Structure

Basic MCP server structure:

```
my-mcp-server/
├── server.py          # Main server implementation
├── main.py           # Entry point
├── requirements.txt  # Dependencies
└── README.md        # Documentation
```

## Basic Server Template

### server.py
```python
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("my_server")

@mcp.tool()
def example_tool(text: str) -> str:
    """Example tool that processes text."""
    return f"Processed: {text}"

@mcp.resource("example://resource")
def example_resource() -> str:
    """Example resource that provides data."""
    return "This is example data from the resource"
```

### main.py
```python
from server import mcp

if __name__ == "__main__":
    mcp.run()
```

## Development Tools

### 1. Testing Your Server

Use the built-in development server:
```bash
mcp dev
```

### 2. MCP Inspector

For debugging and testing:
```bash
npx @modelcontextprotocol/inspector python main.py
```

### 3. Integration with Claude Desktop

Configure Claude Desktop to use your MCP server by adding to the configuration file.

## Project Creation Tool

Use the official project creator:

```bash
npx @modelcontextprotocol/create-server@latest my-server --template python
```

This creates a complete project structure with:
- Proper directory organization
- pyproject.toml configuration
- README with setup instructions
- Example implementations

## Alternative Frameworks

### FastMCP 2.0 (Standalone)

While FastMCP 1.0 is integrated into the official SDK, FastMCP 2.0 is available as a separate package:

```bash
pip install fastmcp
```

**Benefits:**
- More Pythonic API
- Decorator-based function registration
- Enhanced development experience

## Best Practices

1. **Use Virtual Environments**: Always isolate your MCP server dependencies
2. **Version Pinning**: Pin your MCP SDK version for consistency
3. **Type Hints**: Use Python type hints for better development experience
4. **Error Handling**: Implement proper error handling in your tools and resources
5. **Documentation**: Document your tools and resources clearly

## Next Steps

With this development environment set up, you're ready to:
1. Create your first MCP server
2. Implement custom tools and resources
3. Test with Claude Desktop or other MCP clients
4. Deploy your server for production use

## Troubleshooting

- **Python Version**: Ensure you're using Python 3.10+
- **Dependencies**: Use `pip install --upgrade mcp` to get the latest version
- **Virtual Environment**: Always activate your virtual environment before development
- **Testing**: Use `mcp dev` for local testing before integration