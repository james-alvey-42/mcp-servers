"""
Base classes for LLM providers.

This module defines the abstract interface that all LLM providers must implement.
By using a common interface, we can easily switch between providers or add new ones.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """
    Represents a single message in a conversation.
    
    This is our standardized message format that works across all providers.
    Each provider will convert this to their specific format.
    """
    role: str = Field(description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(description="The text content of the message")


class LLMUsage(BaseModel):
    """
    Token usage information from the LLM API call.
    
    Different providers return usage stats in different formats,
    so we normalize them to this standard structure.
    """
    prompt_tokens: int = Field(description="Number of tokens in the input")
    completion_tokens: int = Field(description="Number of tokens in the output")
    total_tokens: int = Field(description="Total tokens used")


class LLMResponse(BaseModel):
    """
    Standardized response from any LLM provider.
    
    This ensures all providers return data in the same format,
    making it easy to use the responses in our MCP tools.
    """
    content: str = Field(description="The generated text response")
    model: str = Field(description="The specific model that generated this response")
    provider: str = Field(description="Which provider was used (openai, gemini, etc.)")
    usage: LLMUsage = Field(description="Token usage statistics")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this response was generated")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Original provider response for debugging")


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Each provider (OpenAI, Gemini, etc.) must implement these methods.
    This ensures they all work the same way from the perspective of our MCP server.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the provider with an API key.
        
        Args:
            api_key: The API key for this provider
        """
        self.api_key = api_key
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    async def call(
        self, 
        model: str, 
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Make a call to the LLM API.
        
        This is the main method that each provider must implement.
        It should convert our standard format to the provider's format,
        make the API call, and convert the response back to our standard format.
        
        Args:
            model: The model name to use (e.g., "gpt-4", "gemini-pro")
            messages: List of messages in the conversation
            temperature: Randomness in the response (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with the generated content and metadata
            
        Raises:
            Exception: If the API call fails
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """
        Get a list of available models for this provider.
        
        Returns:
            List of model names that can be used with this provider
        """
        pass
    
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.
        
        Returns:
            Provider name (e.g., "openai", "gemini")
        """
        return self.provider_name