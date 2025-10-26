"""
Anthropic Provider Implementation

This module implements the LLM provider interface for Anthropic's Claude API.
It handles converting our standard format to Anthropic's format and back.
"""

import httpx
from typing import List, Optional, Dict, Any
from .base import LLMProvider, LLMResponse, LLMMessage, LLMUsage


class AnthropicProvider(LLMProvider):
    """
    Anthropic API provider implementation.

    This class handles all communication with Anthropic's API, including:
    - Converting our standard message format to Anthropic's format
    - Making HTTP requests to Anthropic's endpoints
    - Converting Anthropic responses back to our standard format
    """

    def __init__(self, api_key: str):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (usually from ANTHROPIC_API_KEY environment variable)
        """
        super().__init__(api_key)
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def _convert_messages_to_anthropic(
        self, messages: List[LLMMessage]
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """
        Convert our standard message format to Anthropic's format.

        Anthropic has a special "system" parameter separate from messages,
        and requires alternating user/assistant messages.

        Args:
            messages: List of our standard LLMMessage objects

        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_message = None
        anthropic_messages = []

        for message in messages:
            if message.role == "system":
                # Anthropic uses a separate system parameter
                if system_message is None:
                    system_message = message.content
                else:
                    # Combine multiple system messages
                    system_message += f"\n\n{message.content}"
            elif message.role in ["user", "assistant"]:
                anthropic_messages.append(
                    {"role": message.role, "content": message.content}
                )

        return system_message, anthropic_messages

    async def call(
        self,
        model: str,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Make a call to Anthropic's Messages API.

        This method:
        1. Converts our messages to Anthropic format
        2. Builds the request payload
        3. Makes the HTTP request
        4. Processes the response
        5. Returns our standardized LLMResponse

        Args:
            model: Anthropic model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            messages: Conversation messages
            temperature: Randomness (0.0 to 1.0 for Anthropic)
            max_tokens: Maximum tokens to generate (required by Anthropic, defaults to 4096)
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            LLMResponse with the generated content and metadata

        Raises:
            Exception: If the API call fails or returns an error
        """
        # Convert our messages to Anthropic format
        system_message, anthropic_messages = self._convert_messages_to_anthropic(
            messages
        )

        # Anthropic requires max_tokens, default to 4096 if not specified
        if max_tokens is None:
            max_tokens = 4096

        # Build the request payload
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message

        # Add any additional Anthropic-specific parameters
        payload.update(kwargs)

        # Make the API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload,
                timeout=60.0,  # 60 second timeout
            )

            # Check if the request was successful
            if response.status_code != 200:
                error_detail = response.text
                raise Exception(
                    f"Anthropic API error (status {response.status_code}): {error_detail}"
                )

            # Parse the response
            data = response.json()

            # Handle API errors in the response
            if "error" in data:
                raise Exception(f"Anthropic API error: {data['error']['message']}")

            # Extract the generated content
            if not data.get("content") or len(data["content"]) == 0:
                raise Exception("Anthropic API returned no content")

            # Anthropic returns content as a list of content blocks
            # We'll concatenate text blocks
            content_blocks = data["content"]
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")

            # Extract usage information
            usage_data = data.get("usage", {})
            usage = LLMUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0),
            )

            # Return our standardized response
            return LLMResponse(
                content=content,
                model=model,
                provider="anthropic",
                usage=usage,
                raw_response=data,  # Keep original response for debugging
            )

    async def list_models(self) -> List[str]:
        """
        Get a list of available Anthropic models.

        Note: Anthropic doesn't have a models endpoint, so we return a curated list
        of current Claude models as of January 2025.

        Returns:
            List of available model names
        """
        # Anthropic doesn't have a public models API endpoint
        # Return the current list of available Claude models
        return [
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-haiku-4-5",
            "claude-sonnet-4-0",
            "claude-opus-4-0",
            "claude-haiku-4-0",
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]
