# LLM API Bridge - Complete User Guide

## Overview

The LLM API Bridge is a Model Context Protocol (MCP) server that provides unified access to multiple Large Language Model APIs. It allows you to call different LLM providers (OpenAI, Gemini, etc.) through a consistent interface, making it easy to compare models and switch between providers.

## What You Can Do

### 🔧 Available Tools
- **`call_llm`**: Make calls to any supported LLM provider
- **`list_models`**: Get available models for a specific provider  
- **`echo_test`**: Test server connectivity

### 📊 Available Resources
- **`info://server`**: Server status and usage information
- **`providers://status`**: Detailed provider availability and configuration

### 📝 Available Prompts
- **`compare_models`**: Template for systematic model comparison
- **`test_prompt`**: Basic prompt template for testing

## Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- OpenAI API key (for OpenAI provider)
- MCP-compatible client (Claude Desktop, VS Code, etc.)

### 2. Installation
```bash
# Clone or download the server
cd llm-api-bridge

# Install dependencies
pip install -r requirements.txt

# Set your API keys
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Test the Server
```bash
# Test with MCP Inspector
mcp dev server.py

# In the inspector web interface:
# - Change command to: mcp
# - Change arguments to: run server.py
```

## Using the Tools

### Calling an LLM

**Basic Example:**
```json
{
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ]
}
```

**Advanced Example with Parameters:**
```json
{
  "provider": "openai", 
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing"}
  ],
  "temperature": 0.3,
  "max_tokens": 500
}
```

### Listing Available Models

```json
{
  "provider": "openai"
}
```

Returns a list like: `["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]`

### Model Comparison Workflow

1. **Use the compare_models prompt:**
   ```json
   {
     "question": "Explain artificial intelligence",
     "models": "gpt-3.5-turbo,gpt-4"
   }
   ```

2. **Follow the generated instructions to call each model**

3. **Compare responses based on:**
   - Quality and depth
   - Token usage efficiency
   - Response style
   - Accuracy

## Response Format

All LLM calls return a structured `LLMResponse` with:

```json
{
  "content": "The generated response text",
  "model": "gpt-3.5-turbo",
  "provider": "openai",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "raw_response": {...}
}
```

## Supported Providers

### OpenAI
- **Models**: GPT-4, GPT-3.5-turbo, GPT-4-turbo variants
- **Features**: Chat completions, model listing
- **API Key**: Set `OPENAI_API_KEY` environment variable

### Google Gemini (Coming Soon)
- **Models**: Gemini-Pro (planned)
- **API Key**: Set `GEMINI_API_KEY` environment variable

## Common Use Cases

### 1. Model Evaluation
Compare different models on the same task to choose the best one for your use case.

### 2. Cost Optimization
Use token usage data to optimize costs by selecting efficient models.

### 3. A/B Testing
Test different prompts or parameters across multiple models.

### 4. Fallback Systems
Implement fallback logic where if one provider fails, you can try another.

### 5. Research and Development
Experiment with different LLM providers and models in a consistent environment.

## Error Handling

The server provides clear error messages for common issues:

- **Missing API Key**: `ValueError: OPENAI_API_KEY environment variable is required`
- **Invalid Provider**: `ValueError: Unsupported provider: unknown_provider`
- **API Errors**: `Exception: OpenAI API error (status 401): Invalid API key`
- **Network Issues**: `Exception: Request timeout or connection error`

## Performance Tips

1. **Provider Caching**: Providers are cached after first use for efficiency
2. **Concurrent Calls**: The server handles multiple simultaneous requests
3. **Timeout Handling**: Reasonable timeouts prevent hanging requests
4. **Structured Output**: Use Pydantic models for type safety and validation

## Troubleshooting

### Server Won't Start
- Check Python version (3.10+ required)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check for port conflicts with `lsof -i :6274`

### API Key Issues
- Verify environment variables are set: `echo $OPENAI_API_KEY`
- Check API key validity by testing directly with provider
- Ensure no extra spaces or quotes in environment variables

### Connection Issues
- Test basic connectivity with `echo_test` tool
- Check network connectivity to provider APIs
- Verify firewall/proxy settings

### Model Not Available
- Use `list_models` to see available models
- Check if your API key has access to premium models
- Verify model name spelling and formatting

## Best Practices

1. **Security**: Never hardcode API keys in scripts
2. **Cost Management**: Monitor token usage with the usage statistics
3. **Error Handling**: Always handle potential API failures gracefully
4. **Rate Limiting**: Be aware of provider rate limits
5. **Testing**: Use `echo_test` to verify connectivity before important calls

## Next Steps

- Try the server with different MCP clients
- Experiment with different models and parameters
- Use the comparison tools to evaluate models for your specific use cases
- Set up environment variables for seamless operation

For technical details and advanced configuration, see the developer documentation in the `docs/` directory.