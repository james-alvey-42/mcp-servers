# Testing the LLM API Bridge Server

## MCP Inspector Testing

### Starting the Inspector
```bash
cd llm-api-bridge
mcp dev server.py
```

This will start the MCP Inspector at `http://localhost:6274` with an authentication token.

### Inspector Configuration
**Important:** In the MCP Inspector web interface, update the server configuration:
- **Command:** Change from `uv` to `mcp`
- **Arguments:** Change to `run server.py`

This tells the inspector how to properly start and connect to your MCP server.

### What to Test

#### 1. Tools Tab
- **echo_test**: Test basic tool functionality
  - Input: Any message string
  - Expected: "Echo: [your message]"

#### 2. Resources Tab
- **info://server**: Check server status and API key configuration
  - Expected: Server info with API key status (Set/Not set)

#### 3. Prompts Tab
- **test_prompt**: Test prompt templates
  - Input: Any topic string
  - Expected: Formatted explanation prompt

### Testing Workflow
1. Start inspector: `mcp dev server.py`
2. Open browser to provided URL with auth token
3. Configure command/arguments as noted above
4. Test each tool, resource, and prompt
5. Check that responses match expected outputs
6. Verify any error handling

### Stopping the Inspector
Press `Ctrl+C` in the terminal to stop the inspector and server.

## Claude Desktop Testing

### Installation
```bash
mcp install server.py --name "LLM API Bridge"
```

### Usage in Claude Desktop
Once installed, you can:
- Call tools directly by asking Claude to use them
- Reference resources in your conversations
- Use prompts as slash commands or templates

### Environment Variables
Ensure these are set before starting:
```bash
export OPENAI_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"
```