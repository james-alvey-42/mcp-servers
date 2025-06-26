"""
LLM Provider Abstraction Layer

This module provides a unified interface for interacting with different LLM APIs.
Each provider (OpenAI, Gemini, etc.) implements the same interface, making it
easy to switch between providers or compare outputs.
"""

from .base import LLMProvider, LLMResponse, LLMMessage
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider

__all__ = [
    "LLMProvider",
    "LLMResponse", 
    "LLMMessage",
    "OpenAIProvider",
    "GeminiProvider",
]