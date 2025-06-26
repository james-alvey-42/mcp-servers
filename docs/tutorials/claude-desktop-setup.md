# Claude Desktop Setup Guide for LLM API Bridge

## Overview

This guide shows you how to install and configure the LLM API Bridge server with Claude Desktop, including secure API key management that doesn't require copying keys around.

## Prerequisites

- Claude Desktop app installed ([download here](https://claude.ai/download))
- Python 3.10+ installed
- OpenAI API key
- Terminal/command line access

## Method 1: Environment Variables (Recommended)

This method keeps your API keys in your system environment, so they're available to all processes without copying them to configuration files.

### Step 1: Set Up Environment Variables

#### On macOS/Linux:

**Option A: Add to your shell profile (persistent)**
```bash
# Add to ~/.zshrc (macOS) or ~/.bashrc (Linux)
echo 'export OPENAI_API_KEY="your-openai-api-key-here"' >> ~/.zshrc
echo 'export GEMINI_API_KEY="your-gemini-api-key-here"' >> ~/.zshrc

# Reload your shell
source ~/.zshrc
```

**Option B: Set for current session only**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export GEMINI_API_KEY="your-gemini-api-key-here"
```

#### On Windows:

**Option A: System Environment Variables (persistent)**
1. Open Settings → System → About → Advanced system settings
2. Click "Environment Variables"
3. Under "User variables", click "New"
4. Variable name: `OPENAI_API_KEY`
5. Variable value: your OpenAI API key
6. Repeat for `GEMINI_API_KEY`
7. Restart Claude Desktop

**Option B: Command prompt (current session)**
```cmd
set OPENAI_API_KEY=your-openai-api-key-here
set GEMINI_API_KEY=your-gemini-api-key-here
```

### Step 2: Verify Environment Variables

```bash
# Check that your keys are set
echo $OPENAI_API_KEY
echo $GEMINI_API_KEY

# They should display your API keys (not empty)
```

### Step 3: Install the Server in Claude Desktop

```bash
# Navigate to the server directory
cd /path/to/mcp-servers/llm-api-bridge

# Install in Claude Desktop
mcp install server.py --name "LLM API Bridge"
```

### Step 4: Restart Claude Desktop

Close and reopen Claude Desktop to load the new server configuration.

## Method 2: Configuration File with Environment Variables

If you prefer to use Claude Desktop's configuration file but still want to reference environment variables:

### Step 1: Set Environment Variables (same as Method 1)

### Step 2: Find Claude Desktop Config

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

### Step 3: Edit Configuration File

```json
{
  "mcpServers": {
    "llm-api-bridge": {
      "command": "python",
      "args": ["/full/path/to/mcp-servers/llm-api-bridge/server.py"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

**Important:** Replace `/full/path/to/mcp-servers/llm-api-bridge/server.py` with the actual absolute path to your server.py file.

## Method 3: Direct Configuration (Less Secure)

⚠️ **Warning**: This method puts API keys directly in configuration files. Only use if other methods don't work.

### Edit Claude Desktop Config

```json
{
  "mcpServers": {
    "llm-api-bridge": {
      "command": "python",
      "args": ["/full/path/to/mcp-servers/llm-api-bridge/server.py"],
      "env": {
        "OPENAI_API_KEY": "your-actual-openai-key-here",
        "GEMINI_API_KEY": "your-actual-gemini-key-here"
      }
    }
  }
}
```

## Verification

### Step 1: Check Server Status in Claude Desktop

1. Open Claude Desktop
2. Look for the LLM API Bridge server in the server list
3. It should show as "Connected" with a green indicator

### Step 2: Test Basic Functionality

Ask Claude to:
```
Use the echo_test tool with the message "Hello from Claude Desktop"
```

You should see: `Echo: Hello from Claude Desktop`

### Step 3: Test API Key Configuration

Ask Claude to:
```
Read the info://server resource to check API key status
```

You should see:
- ✅ Set for OPENAI_API_KEY
- ✅ Set (or ❌ Not set) for GEMINI_API_KEY

### Step 4: Test LLM Calls

Ask Claude to:
```
Use the call_llm tool to ask GPT-3.5-turbo: "What is 2+2?"
```

You should get a structured response with the answer and usage statistics.

## Troubleshooting

### Server Shows as "Disconnected"

1. **Check Python Path**: Ensure `python` command works in terminal
2. **Check File Path**: Verify the absolute path to server.py is correct
3. **Check Dependencies**: Run `pip install -r requirements.txt` in the server directory
4. **Check Logs**: Look at Claude Desktop logs for error messages

### API Key Not Found Errors

1. **Environment Variables**: Verify keys are set with `echo $OPENAI_API_KEY`
2. **Restart Required**: After setting environment variables, restart Claude Desktop
3. **Shell Profile**: Ensure you added exports to the correct shell profile file
4. **Case Sensitivity**: Environment variable names are case-sensitive

### Permission Errors

1. **File Permissions**: Ensure server.py is readable: `chmod +r server.py`
2. **Directory Access**: Ensure Claude Desktop can access the server directory
3. **Python Installation**: Verify Python is in your PATH: `which python`

### Import Errors

```bash
# Test imports manually
cd llm-api-bridge
python -c "import server; print('Success')"

# If this fails, check dependencies
pip install -r requirements.txt
```

## Security Best Practices

1. **Never Commit API Keys**: Don't put API keys in version control
2. **Use Environment Variables**: Keep keys in environment, not config files
3. **Restrict Key Permissions**: Use least-privilege API keys when possible
4. **Regular Rotation**: Rotate API keys periodically
5. **Monitor Usage**: Keep track of API usage and costs

## Advanced Configuration

### Custom Server Name
```bash
mcp install server.py --name "My Custom LLM Bridge"
```

### Multiple Environments
```bash
# Development
export OPENAI_API_KEY="dev-key"
mcp install server.py --name "LLM Bridge (Dev)"

# Production  
export OPENAI_API_KEY="prod-key"
mcp install server.py --name "LLM Bridge (Prod)"
```

### Server Updates

When you update the server code:
```bash
# Reinstall to pick up changes
mcp install server.py --name "LLM API Bridge"
# Restart Claude Desktop
```

## Using the Server in Claude Desktop

Once installed, you can:

1. **Ask Claude to use tools directly:**
   - "Use the call_llm tool to ask GPT-4 about quantum computing"
   - "Check which OpenAI models are available"

2. **Reference resources in conversation:**
   - "What's the current server status?"
   - "Show me the provider configurations"

3. **Use prompts for complex workflows:**
   - "Use the compare_models prompt to compare GPT-3.5 and GPT-4 on explaining AI"

The server integrates seamlessly with Claude's natural conversation flow, making it easy to access multiple LLM providers without leaving Claude Desktop.

## Next Steps

- Try different models and compare their responses
- Experiment with different temperature and max_tokens settings
- Use the model comparison features for evaluation
- Monitor token usage for cost optimization

For technical details and troubleshooting, see the complete user guide and developer documentation.