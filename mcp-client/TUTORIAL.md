# MCP Client Development Tutorial

This tutorial walks you through building a simple MCP (Model Context Protocol) client from scratch, using our working example as a reference.

## Prerequisites

- Python 3.8+
- Basic understanding of async/await in Python
- Familiarity with MCP concepts (see [MCP Overview](../docs/architecture/mcp-overview.md))

## Understanding MCP Client Architecture

An MCP client connects to MCP servers to access their tools and resources. The communication happens through various transports, with stdio being the most common for local development.

### Key Components

1. **Transport Layer**: How client and server communicate (stdio, HTTP, WebSocket)
2. **Session Management**: MCP protocol handling and message exchange
3. **Tool Calling**: Invoking server tools with parameters
4. **Resource Access**: Reading server-provided resources

## Step 1: Basic Client Structure

Start with the minimal client class:

```python
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class SimpleMCPClient:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        
    async def connect(self, server_command, server_env=None):
        # Connect to server via stdio
        server_params = StdioServerParameters(
            command=server_command[0], 
            args=server_command[1:], 
            env=server_env
        )
        
        # Use AsyncExitStack for proper cleanup
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        
        # Create and initialize session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()
```

## Step 2: Discovering Server Capabilities

Once connected, discover what the server offers:

```python
async def list_tools(self):
    tools_result = await self.session.list_tools()
    for tool in tools_result.tools:
        print(f"Tool: {tool.name} - {tool.description}")
        if tool.inputSchema and 'properties' in tool.inputSchema:
            params = list(tool.inputSchema['properties'].keys())
            print(f"  Parameters: {', '.join(params)}")

async def list_resources(self):
    resources_result = await self.session.list_resources()
    for resource in resources_result.resources:
        print(f"Resource: {resource.uri} - {resource.name}")
```

## Step 3: Calling Tools

Invoke server tools with parameters:

```python
async def call_tool(self, tool_name, arguments=None):
    arguments = arguments or {}
    result = await self.session.call_tool(tool_name, arguments)
    
    for content in result.content:
        if hasattr(content, 'type') and content.type == "text":
            print(content.text)
        elif hasattr(content, 'text'):
            print(content.text)
        else:
            print(content)
```

## Step 4: Reading Resources

Access server resources:

```python
async def get_resource(self, uri):
    result = await self.session.read_resource(uri)
    
    for content in result.contents:
        if hasattr(content, 'type') and content.type == "text":
            print(content.text)
        elif hasattr(content, 'text'):
            print(content.text)
        else:
            print(content)
```

## Step 5: Connection Management

Always properly close connections using AsyncExitStack:

```python
async def close(self):
    await self.exit_stack.aclose()
```

## Complete Example

Here's how to use the client:

```python
async def main():
    client = SimpleMCPClient()
    
    try:
        # Connect to server
        await client.connect(
            ["python", "/path/to/server.py"],
            server_env={"OPENAI_API_KEY": "your-key"}
        )
        
        # Discover capabilities
        await client.list_tools()
        await client.list_resources()
        
        # Use the server
        await client.call_tool("echo_test", {"message": "Hello!"})
        await client.get_resource("info://server")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing Your Client

### Test Against Local Servers

Use the servers in this repository:

```bash
# Test with LLM API Bridge
python client.py python ../llm-api-bridge/server.py

# Test with ArXiv Downloader  
python client.py python ../arxiv-downloader/server.py
```

### Interactive Testing

The example client provides an interactive mode:

1. **Connect**: Client establishes stdio connection
2. **Discover**: Lists available tools and resources
3. **Interact**: Use commands like `call echo_test` or `get info://server`
4. **Disconnect**: Clean shutdown with `quit`

## Key Learnings

### Content Handling

The MCP library structure has evolved. Content objects may not always have a `type` attribute, so use defensive programming:

```python
# Safe content handling
if hasattr(content, 'type') and content.type == "text":
    print(content.text)
elif hasattr(content, 'text'):
    print(content.text)
else:
    print(content)
```

### Connection Management

Use `AsyncExitStack` for proper resource cleanup:

```python
# Proper async resource management
self.exit_stack = AsyncExitStack()
stdio_transport = await self.exit_stack.enter_async_context(
    stdio_client(server_params)
)
```

### Server Parameters

Split command and arguments correctly:

```python
# Correct parameter structure
StdioServerParameters(
    command=server_command[0],    # "python"
    args=server_command[1:],      # ["/path/to/server.py"]
    env=server_env
)
```

## Common Patterns

### Server Discovery

```python
# List all server capabilities
await client.list_tools()
await client.list_resources()

# Check specific tool availability
tools = await client.session.list_tools()
has_echo = any(t.name == "echo_test" for t in tools.tools)
```

### Environment Variables

```python
# Pass environment to server
server_env = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
}
await client.connect(["python", "server.py"], server_env=server_env)
```

### Error Handling

```python
try:
    result = await self.session.call_tool(tool_name, arguments)
except Exception as e:
    print(f"Tool call failed: {e}")
```

## Troubleshooting

### Connection Issues

- Verify server command is correct
- Check server starts without errors
- Ensure proper Python environment
- Use absolute paths for server files

### Session Initialization Hangs

- Add timeouts to prevent infinite waits
- Check server responds to MCP protocol
- Verify server has proper `mcp.run()` call

### Content Access Errors

- Use defensive attribute checking
- Handle different content types gracefully
- Check MCP library version compatibility

## Advanced Example: Custom Chatbot with Tool Access

Here's a more advanced example that creates a chatbot with access to MCP tools, based on the [MCP Quickstart Client Structure](https://modelcontextprotocol.io/quickstart/client#basic-client-structure):

```python
import asyncio
import json
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPChatbot:
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.available_tools = []
    
    async def connect_to_server(self, server_command, server_env=None):
        """Connect to an MCP server and discover available tools."""
        server_params = StdioServerParameters(
            command=server_command[0],
            args=server_command[1:],
            env=server_env
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()
        
        # Discover available tools
        tools_response = await self.session.list_tools()
        self.available_tools = [tool.name for tool in tools_response.tools]
        print(f"Connected! Available tools: {', '.join(self.available_tools)}")
    
    async def chat_with_tools(self):
        """Interactive chat loop with tool calling capability."""
        print("\n🤖 MCP Chatbot with Tool Access")
        print("Commands:")
        print("  /tools - List available tools")
        print("  /call <tool> <json_args> - Call a tool")
        print("  /quit - Exit")
        print("  Or just chat normally!")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input == "/quit":
                break
            elif user_input == "/tools":
                print(f"Available tools: {', '.join(self.available_tools)}")
            elif user_input.startswith("/call "):
                await self.handle_tool_call(user_input[6:])
            else:
                # For normal chat, you could integrate with an LLM here
                # and let it decide when to call tools
                await self.handle_chat_message(user_input)
    
    async def handle_tool_call(self, command):
        """Handle direct tool calls from user."""
        try:
            parts = command.split(' ', 1)
            tool_name = parts[0]
            args = json.loads(parts[1]) if len(parts) > 1 else {}
            
            if tool_name not in self.available_tools:
                print(f"❌ Tool '{tool_name}' not available")
                return
            
            result = await self.session.call_tool(tool_name, args)
            print("🔧 Tool Result:")
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"  {content.text}")
                else:
                    print(f"  {content}")
                    
        except json.JSONDecodeError:
            print("❌ Invalid JSON arguments")
        except Exception as e:
            print(f"❌ Tool call failed: {e}")
    
    async def handle_chat_message(self, message):
        """Handle normal chat messages."""
        # This is where you'd integrate with an LLM that can:
        # 1. Understand the message
        # 2. Decide if tools are needed
        # 3. Call tools automatically
        # 4. Generate responses
        
        # For now, just echo with suggestions
        print(f"🤖 I received: '{message}'")
        
        # Suggest relevant tools based on keywords
        suggestions = []
        if "search" in message.lower() or "paper" in message.lower():
            if "search_papers" in self.available_tools:
                suggestions.append("search_papers")
        if "model" in message.lower() or "llm" in message.lower():
            if "list_models" in self.available_tools:
                suggestions.append("list_models")
        
        if suggestions:
            print(f"💡 You might want to try: {', '.join(suggestions)}")
            print("Use /call <tool> <args> to call them")
    
    async def close(self):
        """Clean up connections."""
        try:
            await self.exit_stack.aclose()
        except Exception:
            pass

async def main():
    """Example chatbot connecting to LLM API Bridge."""
    chatbot = MCPChatbot()
    
    try:
        # Connect to your MCP server
        await chatbot.connect_to_server(
            ["python", "/path/to/your/server.py"],
            server_env={
                "OPENAI_API_KEY": "your-key-here"
            }
        )
        
        # Start interactive chat
        await chatbot.chat_with_tools()
        
    finally:
        await chatbot.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Integrating with LLM APIs

To create a full chatbot that intelligently uses tools, you could extend this by:

1. **Adding LLM Integration**: Use OpenAI, Anthropic, or other APIs for natural language understanding
2. **Smart Tool Selection**: Let the LLM analyze user messages and decide which tools to call
3. **Function Calling**: Use the LLM's function calling capabilities to automatically invoke MCP tools
4. **Context Management**: Maintain conversation history and tool results

Example LLM integration:

```python
async def handle_chat_message_with_llm(self, message):
    """Handle chat with LLM that can call tools."""
    # 1. Send message to LLM with available tools
    # 2. LLM decides if tools are needed
    # 3. Execute tool calls if requested
    # 4. Send results back to LLM for final response
    
    system_prompt = f"""
    You are a helpful assistant with access to these tools:
    {', '.join(self.available_tools)}
    
    When users ask questions that could benefit from these tools,
    call them automatically and use the results in your response.
    """
    
    # Your LLM API call here...
    # Handle function calls by mapping to MCP tools
```

## Next Steps

1. **Custom Transports**: Implement HTTP or WebSocket clients
2. **Advanced Features**: Add prompt handling and streaming
3. **Integration**: Build clients into larger applications
4. **LLM Integration**: Connect with language models for intelligent tool usage
5. **Testing**: Create comprehensive test suites

## References

- [MCP Python SDK Documentation](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Quickstart](https://modelcontextprotocol.io/quickstart/client)
- [Project Architecture Guide](../docs/architecture/mcp-overview.md)

This tutorial provides the foundation for building MCP clients. The complete example in this directory demonstrates all these concepts in a working application that successfully connects to and interacts with MCP servers.