#!/usr/bin/env python3
"""
LLM API Bridge MCP Server

A Model Context Protocol server that provides unified access to multiple LLM APIs.
Supports OpenAI, Google Gemini, and other providers through a consistent interface.
"""

import os
from typing import List, Dict, Optional
from mcp.server.fastmcp import FastMCP

# Import our provider system
from providers import LLMProvider, LLMResponse, LLMMessage, OpenAIProvider, GeminiProvider

# Create the FastMCP server instance
# The name "LLM API Bridge" will be displayed in MCP clients
mcp = FastMCP("LLM API Bridge")

# Global dictionary to store initialized providers
# This avoids recreating providers on each call
_providers: Dict[str, LLMProvider] = {}


def _get_provider(provider_name: str) -> LLMProvider:
    """
    Get or initialize a provider instance.
    
    This function handles the logic of creating provider instances with API keys
    from environment variables. It caches providers so we don't recreate them
    on every call.
    
    Args:
        provider_name: Name of the provider ("openai", "gemini", etc.)
        
    Returns:
        Initialized provider instance
        
    Raises:
        ValueError: If provider is not supported or API key is missing
    """
    # Check if we already have this provider initialized
    if provider_name in _providers:
        return _providers[provider_name]
    
    # Initialize the provider based on the name
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        provider = OpenAIProvider(api_key)
    elif provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini provider")
        provider = GeminiProvider(api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: openai, gemini")
    
    # Cache the provider for future use
    _providers[provider_name] = provider
    return provider


@mcp.tool()
async def call_llm(
    provider: str,
    model: str, 
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> LLMResponse:
    """
    Call an LLM API with a unified interface.
    
    This is the main tool that allows users to call any supported LLM provider
    through a consistent interface. It handles provider initialization, 
    message format conversion, and response standardization.
    
    Args:
        provider: LLM provider to use ("openai", "gemini")
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
        messages: List of conversation messages with 'role' and 'content' keys
        temperature: Randomness in response (0.0 to 1.0)
        max_tokens: Maximum tokens to generate (optional)
        
    Returns:
        LLMResponse with generated content, usage stats, and metadata
        
    Example:
        call_llm(
            provider="openai",
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
    """
    # Get the provider instance
    llm_provider = _get_provider(provider)
    
    # Convert dict messages to our LLMMessage format
    llm_messages = [LLMMessage(role=msg["role"], content=msg["content"]) for msg in messages]
    
    # Make the API call
    response = await llm_provider.call(
        model=model,
        messages=llm_messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response


@mcp.tool()
async def list_models(provider: str) -> List[str]:
    """
    Get available models for a specific provider.
    
    This tool allows users to discover what models are available
    for each LLM provider, making it easier to choose the right
    model for their use case.
    
    Args:
        provider: LLM provider name ("openai", "gemini")
        
    Returns:
        List of available model names for this provider
        
    Example:
        list_models("openai") -> ["gpt-4", "gpt-3.5-turbo", ...]
    """
    llm_provider = _get_provider(provider)
    return await llm_provider.list_models()


@mcp.tool()
def echo_test(message: str) -> str:
    """
    A simple echo tool to test that our MCP server is working.
    
    This tool just returns the message you send it, prefixed with "Echo: ".
    It's useful for testing that the MCP server is properly connected and
    responding to tool calls.
    
    Args:
        message: The message to echo back
        
    Returns:
        The message prefixed with "Echo: "
    """
    return f"Echo: {message}"


@mcp.resource("info://server")
def server_info() -> str:
    """
    Provides comprehensive information about this MCP server.
    
    This resource tells clients what providers are available,
    their API key status, and how to use the server.
    """
    openai_key_status = "✅ Set" if os.getenv('OPENAI_API_KEY') else "❌ Not set"
    gemini_key_status = "✅ Set" if os.getenv('GEMINI_API_KEY') else "❌ Not set"
    
    return f"""
LLM API Bridge MCP Server

🚀 CAPABILITIES:
This server provides unified access to multiple LLM APIs through standardized tools.

🔧 AVAILABLE TOOLS:
- call_llm: Make calls to any supported LLM provider
- list_models: Get available models for a provider  
- echo_test: Simple test tool for connectivity

🌐 SUPPORTED PROVIDERS:
- OpenAI (GPT models) - API Key: {openai_key_status}
- Google Gemini (Gemini models) - API Key: {gemini_key_status}

📋 USAGE EXAMPLES:
Call OpenAI:
  provider: "openai"
  model: "gpt-3.5-turbo"
  messages: [{{"role": "user", "content": "Hello!"}}]

Call Gemini:
  provider: "gemini"
  model: "gemini-1.5-flash"
  messages: [{{"role": "user", "content": "Hello!"}}]

🔑 ENVIRONMENT VARIABLES:
- OPENAI_API_KEY: {openai_key_status}
- GEMINI_API_KEY: {gemini_key_status}

💡 TIP: Use list_models("openai") or list_models("gemini") to see available models.
"""


@mcp.resource("providers://status")
def providers_status() -> Dict[str, Dict[str, str]]:
    """
    Detailed status of all LLM providers.
    
    Returns structured data about each provider's availability,
    API key status, and supported features.
    """
    return {
        "openai": {
            "status": "available" if os.getenv('OPENAI_API_KEY') else "missing_api_key",
            "api_key": "configured" if os.getenv('OPENAI_API_KEY') else "not_configured",
            "models_supported": "gpt-4, gpt-3.5-turbo, gpt-4-turbo",
            "features": "chat_completions, model_listing"
        },
        "gemini": {
            "status": "available" if os.getenv('GEMINI_API_KEY') else "missing_api_key",
            "api_key": "configured" if os.getenv('GEMINI_API_KEY') else "not_configured", 
            "models_supported": "gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash",
            "features": "chat_completions, model_listing"
        }
    }


@mcp.prompt()
def compare_models(question: str, models: str = "gpt-3.5-turbo,gpt-4") -> str:
    """
    Generate a prompt for comparing responses from different LLM models.
    
    This prompt template helps you systematically compare how different
    models respond to the same question, which is useful for evaluation
    and model selection.
    
    Args:
        question: The question to ask each model
        models: Comma-separated list of models to compare
        
    Returns:
        A formatted prompt for model comparison
    """
    model_list = [model.strip() for model in models.split(",")]
    
    return f"""
Model Comparison Analysis

Question: {question}

Please use the call_llm tool to get responses from these models:
{chr(10).join(f"- {model}" for model in model_list)}

For each model:
1. Call: call_llm(provider="openai", model="{model_list[0]}", messages=[{{"role": "user", "content": "{question}"}}])
2. Note the response quality, style, and token usage
3. Compare factual accuracy and helpfulness

Then provide a summary comparing:
- Response quality and depth
- Token efficiency (usage.total_tokens)  
- Response style and tone
- Which model is best for this type of question
"""


@mcp.prompt()
def test_prompt(topic: str) -> str:
    """
    A simple prompt template for testing.
    
    This creates a basic prompt that can be used to test the prompt
    functionality of our MCP server.
    
    Args:
        topic: The topic to create a prompt about
        
    Returns:
        A formatted prompt string
    """
    return f"Please explain the topic '{topic}' in simple terms with examples."


# This is the standard pattern for running an MCP server
if __name__ == "__main__":
    # When run directly, start the MCP server
    # The server will communicate via stdio (standard input/output)
    # which is how MCP clients connect to local servers
    mcp.run()