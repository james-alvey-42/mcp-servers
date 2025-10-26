#!/usr/bin/env python3
"""
LLM API Bridge MCP Server

A Model Context Protocol server that provides unified access to multiple LLM APIs.
Supports OpenAI, Google Gemini, and other providers through a consistent interface.
"""

import os
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from mcp.server.fastmcp import FastMCP

# Import our provider system
from providers import (
    LLMProvider,
    LLMResponse,
    LLMMessage,
    ContentPart,
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider,
)

# Create the FastMCP server instance
# The name "LLM API Bridge" will be displayed in MCP clients
mcp = FastMCP("LLM API Bridge")

# Global dictionary to store initialized providers
# This avoids recreating providers on each call
_providers: Dict[str, LLMProvider] = {}


def _process_content_item(item: Union[str, Dict[str, Any]]) -> ContentPart:
    """
    Process a single content item, which can be:
    - A file path (str ending with image/audio/video extension)
    - A text string
    - A dict with type/text/url/data fields (backward compatibility)

    Args:
        item: Content item to process

    Returns:
        ContentPart object ready for the provider

    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If file type is not supported
    """
    # If it's already a dict with 'type' field, use backward compatible path
    if isinstance(item, dict) and "type" in item:
        return ContentPart(**item)

    # If it's a string, determine if it's a file path or text
    if isinstance(item, str):
        # Check if it looks like a file path
        path = Path(item)

        # Check if it's an existing file
        if path.exists() and path.is_file():
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(path))

            if not mime_type:
                # Try to guess based on extension
                ext = path.suffix.lower()
                mime_map = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp',
                    '.bmp': 'image/bmp',
                    '.mp3': 'audio/mpeg',
                    '.wav': 'audio/wav',
                    '.mp4': 'video/mp4',
                    '.webm': 'video/webm',
                }
                mime_type = mime_map.get(ext)

            if not mime_type:
                raise ValueError(f"Unsupported file type for: {item}")

            # Read and encode the file
            with open(path, 'rb') as f:
                file_data = f.read()
                base64_data = base64.b64encode(file_data).decode('utf-8')

            # Return as image_base64 content part
            return ContentPart(
                type="image_base64",
                data=base64_data,
                mime_type=mime_type
            )
        else:
            # It's just a text string
            return ContentPart(type="text", text=item)

    # If it's a dict without 'type', try to infer
    if isinstance(item, dict):
        raise ValueError(f"Invalid content item format: {item}. Must have 'type' field or be a string/file path.")

    raise ValueError(f"Unsupported content item type: {type(item)}")


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
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI provider"
            )
        provider = OpenAIProvider(api_key)
    elif provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini provider"
            )
        provider = GeminiProvider(api_key)
    elif provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )
        provider = AnthropicProvider(api_key)
    else:
        raise ValueError(
            f"Unsupported provider: {provider_name}. Supported providers: openai, gemini, anthropic"
        )

    # Cache the provider for future use
    _providers[provider_name] = provider
    return provider


@mcp.tool()
async def call_llm(
    provider: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """
    Call an LLM API with a unified interface.

    This is the main tool that allows users to call any supported LLM provider
    through a consistent interface. It handles provider initialization,
    message format conversion, and response standardization.

    Supports both text-only and multi-modal messages (text + images, PDFs, audio).

    **MULTI-MODAL SUPPORT**: For multi-modal queries (images, PDFs, audio), use gemini-2.5-pro.
    OpenAI provider currently has a serialization bug with multi-modal content.

    **SIMPLIFIED FILE HANDLING**: Just pass file paths directly in the content list!
    The tool automatically detects files, reads them, and encodes them properly.

    Args:
        provider: LLM provider to use ("openai", "gemini", "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3-5-sonnet-20241022", "gemini-2.5-pro")
        messages: List of conversation messages with 'role' and 'content' keys.
                 Content can be:
                 - A string for text-only messages
                 - A list mixing text strings and file paths for multi-modal
        temperature: Randomness in response (0.0 to 1.0)
        max_tokens: Maximum tokens to generate (optional)

    Returns:
        LLMResponse with generated content, usage stats, and metadata

    Examples:
        # Text-only message
        call_llm(
            provider="gemini",
            model="gemini-2.5-pro",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )

        # Multi-modal with file path (EASY WAY!)
        call_llm(
            provider="gemini",
            model="gemini-2.5-pro",
            messages=[
                {
                    "role": "user",
                    "content": ["platypus.png", "What animal is this?"]
                }
            ]
        )

        # Multiple files
        call_llm(
            provider="gemini",
            model="gemini-2.5-pro",
            messages=[
                {
                    "role": "user",
                    "content": [
                        "diagram.png",
                        "photo.jpg",
                        "Compare these two images"
                    ]
                }
            ]
        )
    """
    # Get the provider instance
    llm_provider = _get_provider(provider)

    # Convert dict messages to our LLMMessage format
    llm_messages = []
    for msg in messages:
        content = msg["content"]

        # Handle multi-modal content (list of parts)
        if isinstance(content, list):
            # Process each item in the list (can be files, text, or dict objects)
            content_parts = [_process_content_item(item) for item in content]
            llm_messages.append(LLMMessage(role=msg["role"], content=content_parts))
        else:
            # Simple text content (backward compatible)
            # Still check if it's a file path
            content_part = _process_content_item(content)
            llm_messages.append(LLMMessage(role=msg["role"], content=[content_part]))

    # Make the API call
    response = await llm_provider.call(
        model=model,
        messages=llm_messages,
        temperature=temperature,
        max_tokens=max_tokens,
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
        provider: LLM provider name ("openai", "gemini", "anthropic")

    Returns:
        List of available model names for this provider

    Example:
        list_models("openai") -> ["gpt-4o", "gpt-4-turbo", ...]
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
    openai_key_status = "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set"
    gemini_key_status = "✅ Set" if os.getenv("GEMINI_API_KEY") else "❌ Not set"
    anthropic_key_status = "✅ Set" if os.getenv("ANTHROPIC_API_KEY") else "❌ Not set"

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
- Anthropic (Claude models) - API Key: {anthropic_key_status}

📋 USAGE EXAMPLES:
Call OpenAI:
  provider: "openai"
  model: "gpt-4o"
  messages: [{{"role": "user", "content": "Hello!"}}]

Call Gemini:
  provider: "gemini"
  model: "gemini-2.5-pro"
  messages: [{{"role": "user", "content": "Hello!"}}]

Call Anthropic:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  messages: [{{"role": "user", "content": "Hello!"}}]

🔑 ENVIRONMENT VARIABLES:
- OPENAI_API_KEY: {openai_key_status}
- GEMINI_API_KEY: {gemini_key_status}
- ANTHROPIC_API_KEY: {anthropic_key_status}

💡 TIP: Use list_models("openai"), list_models("gemini"), or list_models("anthropic") to see available models.
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
            "status": "available" if os.getenv("OPENAI_API_KEY") else "missing_api_key",
            "api_key": (
                "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"
            ),
            "models_supported": "gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, o3, o3-mini, gpt-4o, gpt-4",
            "features": "chat_completions, model_listing",
        },
        "gemini": {
            "status": "available" if os.getenv("GEMINI_API_KEY") else "missing_api_key",
            "api_key": (
                "configured" if os.getenv("GEMINI_API_KEY") else "not_configured"
            ),
            "models_supported": "gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash, gemini-2.0-flash-lite",
            "features": "chat_completions, model_listing",
        },
        "anthropic": {
            "status": (
                "available" if os.getenv("ANTHROPIC_API_KEY") else "missing_api_key"
            ),
            "api_key": (
                "configured" if os.getenv("ANTHROPIC_API_KEY") else "not_configured"
            ),
            "models_supported": "claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5, claude-sonnet-4-0, claude-opus-4-0, claude-3-7-sonnet-latest, claude-3-5-haiku-latest",
            "features": "chat_completions, model_listing",
        },
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
