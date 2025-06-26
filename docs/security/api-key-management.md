# API Key Security and Management

## Overview

Properly managing API keys is crucial for security and cost control when using the LLM API Bridge. This guide covers secure methods for storing, configuring, and managing your LLM provider API keys.

## Security Principles

### 1. Never Hardcode Keys
❌ **Bad:**
```python
api_key = "sk-1234567890abcdef"  # Never do this!
```

✅ **Good:**
```python
api_key = os.getenv("OPENAI_API_KEY")
```

### 2. Use Environment Variables
Environment variables keep secrets separate from code and configuration files.

### 3. Principle of Least Privilege
Use API keys with minimal necessary permissions.

### 4. Regular Rotation
Rotate API keys periodically and when team members leave.

## Secure Configuration Methods

### Method 1: Shell Profile (Recommended for Development)

#### macOS/Linux - Persistent Setup
```bash
# Add to ~/.zshrc (macOS) or ~/.bashrc (Linux)
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.zshrc
echo 'export GEMINI_API_KEY="your-gemini-key-here"' >> ~/.zshrc

# Reload shell
source ~/.zshrc

# Verify
echo $OPENAI_API_KEY
```

#### Advantages:
- Keys available to all applications
- Persistent across sessions
- Easy to update

#### Disadvantages:
- Visible in shell history if set manually
- Shared with all processes

### Method 2: .env Files (Development)

Create a `.env` file in your project directory:

```bash
# .env file (never commit this to git!)
OPENAI_API_KEY=sk-your-key-here
GEMINI_API_KEY=your-gemini-key-here
```

Load in your application:
```python
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
```

#### Advantages:
- Project-specific configuration
- Easy to manage multiple environments
- Can be loaded programmatically

#### Disadvantages:
- Risk of accidentally committing to version control
- Must be replicated across environments

**Important:** Always add `.env` to your `.gitignore`:
```gitignore
.env
.env.local
.env.*.local
```

### Method 3: System Environment Variables (Production)

#### macOS/Linux:
```bash
# Add to /etc/environment (system-wide)
sudo echo 'OPENAI_API_KEY="sk-your-key-here"' >> /etc/environment

# Or use systemctl for services
sudo systemctl edit your-service
# Add:
# [Service]
# Environment="OPENAI_API_KEY=sk-your-key-here"
```

#### Windows:
1. System Properties → Advanced → Environment Variables
2. Add system-wide variables
3. Restart required services

### Method 4: Cloud Provider Secret Management

#### AWS Secrets Manager
```python
import boto3

def get_secret(secret_name, region_name="us-west-2"):
    session = boto3.session.Session()
    client = session.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

api_key = get_secret("openai-api-key")
```

#### Azure Key Vault
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://vault.vault.azure.net/", credential=credential)
api_key = client.get_secret("openai-api-key").value
```

#### Google Secret Manager
```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
response = client.access_secret_version(request={"name": name})
api_key = response.payload.data.decode("UTF-8")
```

## Platform-Specific Setup

### macOS Keychain Integration

```bash
# Store in keychain
security add-generic-password -a "$USER" -s "openai-api-key" -w "sk-your-key-here"

# Retrieve from keychain
OPENAI_API_KEY=$(security find-generic-password -a "$USER" -s "openai-api-key" -w)
export OPENAI_API_KEY
```

### Windows Credential Manager

```powershell
# Store credential
cmdkey /add:openai-api-key /user:user /pass:sk-your-key-here

# Use in script
$credential = Get-StoredCredential -Target "openai-api-key"
$env:OPENAI_API_KEY = $credential.GetNetworkCredential().Password
```

### Linux Secret Service

```bash
# Using secret-tool (part of libsecret)
secret-tool store --label="OpenAI API Key" service openai username user
# Enter key when prompted

# Retrieve
OPENAI_API_KEY=$(secret-tool lookup service openai username user)
export OPENAI_API_KEY
```

## MCP-Specific Configuration

### Claude Desktop Configuration

**Secure (recommended):**
```json
{
  "mcpServers": {
    "llm-api-bridge": {
      "command": "python",
      "args": ["/path/to/server.py"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

**Less secure (but functional):**
```json
{
  "mcpServers": {
    "llm-api-bridge": {
      "command": "python", 
      "args": ["/path/to/server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-actual-key-here",
        "GEMINI_API_KEY": "actual-gemini-key-here"
      }
    }
  }
}
```

### Using `mcp install` Command

The `mcp install` command automatically picks up environment variables:

```bash
# Keys from environment
export OPENAI_API_KEY="sk-your-key"
mcp install server.py --name "LLM API Bridge"

# Or specify environment file
mcp install server.py --name "LLM API Bridge" -f .env

# Or individual variables  
mcp install server.py --name "LLM API Bridge" -v OPENAI_API_KEY=sk-your-key
```

## API Key Validation

The LLM API Bridge includes built-in validation:

```python
# Check if keys are configured
@mcp.resource("providers://status")
def providers_status():
    return {
        "openai": {
            "api_key": "configured" if os.getenv('OPENAI_API_KEY') else "not_configured",
            "status": "available" if os.getenv('OPENAI_API_KEY') else "missing_api_key"
        }
    }
```

Test key validity:
```bash
# Use the server info resource
# In Claude Desktop: "Check the server status"
# In MCP Inspector: Read the info://server resource
```

## Development vs Production

### Development Environment
- Use `.env` files or shell environment variables
- Include `.env` in `.gitignore`
- Use development/test API keys with limited quotas
- Consider using mock/stub providers for testing

### Production Environment
- Use system environment variables or cloud secret management
- Implement key rotation procedures
- Monitor API usage and costs
- Use production API keys with appropriate rate limits

## Best Practices Checklist

### ✅ Security
- [ ] Never commit API keys to version control
- [ ] Use environment variables or secret management systems
- [ ] Implement least-privilege access
- [ ] Rotate keys regularly
- [ ] Monitor for key exposure in logs

### ✅ Development
- [ ] Use `.env` files for local development
- [ ] Add `.env` to `.gitignore`
- [ ] Test with development/sandbox API keys
- [ ] Validate key configuration before deployment

### ✅ Production
- [ ] Use system environment variables or cloud secrets
- [ ] Implement automated key rotation
- [ ] Monitor API usage and costs
- [ ] Set up alerting for unusual usage patterns

### ✅ Monitoring
- [ ] Track API usage and costs
- [ ] Set up billing alerts
- [ ] Monitor for API errors and rate limits
- [ ] Log access patterns (without logging keys!)

## Common Mistakes to Avoid

1. **Committing keys to git** - Always check before committing
2. **Logging API keys** - Never log sensitive credentials
3. **Sharing keys in chat/email** - Use secure channels
4. **Using production keys in development** - Use separate dev keys
5. **Ignoring key rotation** - Rotate regularly
6. **Overprivileged keys** - Use minimal necessary permissions

## Troubleshooting

### Key Not Found Errors
```bash
# Check if environment variable is set
echo $OPENAI_API_KEY

# Check if it's available to the process
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"

# For Claude Desktop, restart after setting environment variables
```

### Invalid Key Errors
- Verify key format (OpenAI keys start with 'sk-')
- Check key hasn't been revoked
- Ensure key has necessary permissions
- Test key directly with provider's API

### Permission Errors
- Check if key has access to required models
- Verify billing/quota status
- Test with basic API calls first

## Key Rotation Procedure

1. **Generate new key** at provider's dashboard
2. **Test new key** in development environment
3. **Update environment variables** or secret storage
4. **Restart services** to pick up new key
5. **Verify functionality** with test calls
6. **Revoke old key** after successful transition
7. **Update documentation** and team

This ensures zero-downtime key rotation and maintains security.

## Summary

Secure API key management is essential for the LLM API Bridge. Use environment variables, avoid hardcoding, implement rotation procedures, and follow platform-specific best practices. The methods outlined here provide multiple options for different environments and security requirements.