"""
OpenAI Provider Implementation

This module implements the LLM provider interface for OpenAI's API.
It handles converting our standard format to OpenAI's format and back.
"""

import httpx
from typing import List, Optional, Dict, Any
from .base import LLMProvider, LLMResponse, LLMMessage, LLMUsage


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider implementation.
    
    This class handles all communication with OpenAI's API, including:
    - Converting our standard message format to OpenAI's format
    - Making HTTP requests to OpenAI's endpoints
    - Converting OpenAI responses back to our standard format
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key (usually from OPENAI_API_KEY environment variable)
        """
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _convert_messages_to_openai(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """
        Convert our standard message format to OpenAI's format.
        
        OpenAI expects messages as a list of dicts with 'role' and 'content' keys.
        Our LLMMessage format maps directly to this.
        
        Args:
            messages: List of our standard LLMMessage objects
            
        Returns:
            List of dicts in OpenAI's expected format
        """
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    async def call(
        self, 
        model: str, 
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Make a call to OpenAI's chat completions API.
        
        This method:
        1. Converts our messages to OpenAI format
        2. Builds the request payload
        3. Makes the HTTP request
        4. Processes the response
        5. Returns our standardized LLMResponse
        
        Args:
            model: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            messages: Conversation messages
            temperature: Randomness (0.0 to 2.0 for OpenAI)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            LLMResponse with the generated content and metadata
            
        Raises:
            Exception: If the API call fails or returns an error
        """
        # Convert our messages to OpenAI format
        openai_messages = self._convert_messages_to_openai(messages)
        
        # Build the request payload
        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        # Add any additional OpenAI-specific parameters
        payload.update(kwargs)
        
        # Make the API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60.0  # 60 second timeout
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                error_detail = response.text
                raise Exception(f"OpenAI API error (status {response.status_code}): {error_detail}")
            
            # Parse the response
            data = response.json()
            
            # Handle API errors in the response
            if "error" in data:
                raise Exception(f"OpenAI API error: {data['error']['message']}")
            
            # Extract the generated content
            if not data.get("choices") or len(data["choices"]) == 0:
                raise Exception("OpenAI API returned no choices")
            
            content = data["choices"][0]["message"]["content"]
            
            # Extract usage information
            usage_data = data.get("usage", {})
            usage = LLMUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
            
            # Return our standardized response
            return LLMResponse(
                content=content,
                model=model,
                provider="openai",
                usage=usage,
                raw_response=data  # Keep original response for debugging
            )
    
    async def list_models(self) -> List[str]:
        """
        Get a list of available OpenAI models.
        
        This queries OpenAI's models endpoint and returns a list of model IDs
        that can be used for chat completions.
        
        Returns:
            List of available model names
            
        Raises:
            Exception: If the API call fails
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                error_detail = response.text
                raise Exception(f"OpenAI models API error (status {response.status_code}): {error_detail}")
            
            data = response.json()
            
            # Extract model IDs and filter for chat models
            # OpenAI returns many models, but we only want the ones suitable for chat
            all_models = [model["id"] for model in data.get("data", [])]
            
            # Filter for common chat models (you can expand this list)
            chat_models = [
                model for model in all_models 
                if any(chat_prefix in model for chat_prefix in [
                    "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"
                ])
            ]
            
            return sorted(chat_models)