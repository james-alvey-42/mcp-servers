# LLM API Bridge MCP Server

A Model Context Protocol (MCP) server that provides unified access to multiple LLM APIs through a consistent interface. Call OpenAI, Gemini, and other LLM providers seamlessly from Claude Desktop or any MCP client.

## 🚀 Features

- **Unified Interface**: Call different LLM providers with the same API
- **Multiple Providers**: OpenAI (GPT models), Google Gemini (coming soon)
- **Structured Output**: Type-safe responses with usage statistics
- **Model Comparison**: Built-in tools for comparing different models
- **Secure Configuration**: Environment-based API key management

## 📋 Available Tools

- **`call_llm`**: Make calls to any supported LLM provider
- **`list_models`**: Get available models for a provider  
- **`echo_test`**: Test server connectivity

## 📊 Available Resources

- **`info://server`**: Server status and usage information
- **`providers://status`**: Detailed provider configuration

## 📝 Available Prompts

- **`compare_models`**: Template for systematic model comparison
- **`test_prompt`**: Basic prompt template for testing

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Choose One Method)

**Method A: Environment Variables (Recommended)**
```bash
# Add to ~/.zshrc (macOS) or ~/.bashrc (Linux)
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
source ~/.zshrc
```

**Method B: Create .env file**
```bash
# Create .env file (never commit to git!)
echo 'OPENAI_API_KEY="your-openai-key"' > .env
echo 'GEMINI_API_KEY="your-gemini-key"' >> .env
```

### 3. Test the Server
```bash
mcp dev server.py
```

**Note:** In the MCP Inspector web interface that opens:
- Change the default command from `uv` to `mcp`
- Change the arguments to `run server.py`
- This allows the inspector to properly connect to your server

### 4. Install in Claude Desktop
```bash
mcp install server.py --name "LLM API Bridge"
```

Restart Claude Desktop to load the server.

## 💡 Usage Examples

### Basic LLM Call
```json
{
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ]
}
```

### Model Comparison
Use the `compare_models` prompt:
```json
{
  "question": "Explain quantum computing",
  "models": "gpt-3.5-turbo,gpt-4,gemini-1.5-flash"
}
```

### List Available Models
```json
{
  "provider": "openai"
}
```

Or for Gemini:
```json
{
  "provider": "gemini"
}
```

## 🔧 Supported Providers

| Provider | Models | Status | API Key Required |
|----------|--------|--------|------------------|
| OpenAI | GPT-4, GPT-3.5-turbo, GPT-4-turbo | ✅ Available | `OPENAI_API_KEY` |
| Google Gemini | Gemini-1.5-Flash, Gemini-1.5-Pro, Gemini-2.0-Flash | ✅ Available | `GEMINI_API_KEY` |

## 📚 Documentation

- **[Complete User Guide](../docs/tutorials/llm-api-bridge-user-guide.md)** - Comprehensive usage documentation
- **[Claude Desktop Setup](../docs/tutorials/claude-desktop-setup.md)** - Step-by-step installation guide
- **[API Key Security](../docs/security/api-key-management.md)** - Secure configuration best practices
- **[Testing Guide](TESTING.md)** - How to test the server

## 🔒 Security

- Never hardcode API keys in your code
- Use environment variables or secure secret management
- See our [API Key Security Guide](../docs/security/api-key-management.md) for best practices

## 🐛 Troubleshooting

### Server Won't Start
- Check Python version: `python --version` (3.10+ required)
- Install dependencies: `pip install -r requirements.txt`
- Test imports: `python -c "import server; print('Success')"`

### API Key Issues
- Verify keys are set: `echo $OPENAI_API_KEY`
- Check server status: Use the `info://server` resource
- Restart Claude Desktop after setting environment variables

### Connection Issues
- Test basic connectivity with the `echo_test` tool
- Check firewall/proxy settings
- Verify network connectivity to provider APIs

## 🛠️ Development

For development and contributing information, see the [developer documentation](../docs/).

## 📝 License

This project is part of the MCP servers collection. See the main repository for license information.