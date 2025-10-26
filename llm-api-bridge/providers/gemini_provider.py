"""
Google Gemini Provider Implementation

This module implements the LLM provider interface for Google's Gemini API.
It handles converting our standard format to Gemini's format and back.
"""

import httpx
from typing import List, Optional, Dict, Any, Union
from .base import LLMProvider, LLMResponse, LLMMessage, LLMUsage, ContentPart


class GeminiProvider(LLMProvider):
    """
    Google Gemini API provider implementation.
    
    This class handles all communication with Google's Gemini API, including:
    - Converting our standard message format to Gemini's format
    - Making HTTP requests to Gemini's endpoints
    - Converting Gemini responses back to our standard format
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Google Gemini API key (usually from GEMINI_API_KEY environment variable)
        """
        super().__init__(api_key)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.headers = {
            "Content-Type": "application/json",
        }
    
    def _convert_content_to_parts(self, content: Union[str, List[ContentPart]]) -> List[Dict[str, Any]]:
        """
        Convert message content to Gemini parts format.

        Handles both simple text strings and multi-modal content with images.

        Args:
            content: Either a string or list of ContentPart objects

        Returns:
            List of Gemini part objects
        """
        # Simple text content (backward compatible)
        if isinstance(content, str):
            return [{"text": content}]

        # Multi-modal content
        parts = []
        for part in content:
            if part.type == "text":
                parts.append({"text": part.text})

            elif part.type == "image_url":
                # For image URLs, fetch and convert to inline data
                # Note: Gemini prefers inline data over URLs
                parts.append({
                    "text": f"[Image from URL: {part.url}]"
                })
                # TODO: Could fetch URL and convert to inline_data

            elif part.type == "image_base64":
                # Inline base64 image
                parts.append({
                    "inline_data": {
                        "mime_type": part.mime_type or "image/png",
                        "data": part.data
                    }
                })

        return parts

    def _convert_messages_to_gemini(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """
        Convert our standard message format to Gemini's format.

        Gemini expects a "contents" array with "parts" that can contain text,
        inline images, and other media types.

        Supports multi-modal content including images.

        Args:
            messages: List of our standard LLMMessage objects

        Returns:
            List of content objects in Gemini's expected format
        """
        gemini_contents = []

        for message in messages:
            # Convert content to parts (handles both text and multi-modal)
            parts = self._convert_content_to_parts(message.content)

            # Gemini uses "user" and "model" roles, and system messages are handled differently
            if message.role == "system":
                # For system messages, prepend "System instructions:" to make it clear
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": f"System instructions: {parts[0].get('text', '')}"}]
                })
            elif message.role == "user":
                gemini_contents.append({
                    "role": "user",
                    "parts": parts
                })
            elif message.role == "assistant":
                gemini_contents.append({
                    "role": "model",
                    "parts": parts
                })

        return gemini_contents
    
    async def call(
        self,
        model: str,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Make a call to Google's Gemini API.
        
        This method:
        1. Converts our messages to Gemini format
        2. Builds the request payload
        3. Makes the HTTP request
        4. Processes the response
        5. Returns our standardized LLMResponse
        
        Args:
            model: Gemini model name (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            messages: Conversation messages
            temperature: Randomness (0.0 to 2.0 for Gemini)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            LLMResponse with the generated content and metadata
            
        Raises:
            Exception: If the API call fails or returns an error
        """
        # Convert our messages to Gemini format
        gemini_contents = self._convert_messages_to_gemini(messages)
        
        # Build the request payload for Gemini's generateContent endpoint
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
            }
        }
        
        # Add max_tokens if specified (Gemini calls this maxOutputTokens)
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        
        # Add any additional Gemini-specific parameters
        if kwargs:
            payload["generationConfig"].update(kwargs)
        
        # Build the API endpoint URL
        url = f"{self.base_url}/models/{model}:generateContent"
        params = {"key": self.api_key}
        
        # Make the API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=self.headers,
                json=payload,
                params=params,
                timeout=60.0,  # 60 second timeout
            )
            
            # Check if the request was successful
            if response.status_code != 200:
                error_detail = response.text
                raise Exception(
                    f"Gemini API error (status {response.status_code}): {error_detail}"
                )
            
            # Parse the response
            data = response.json()
            
            # Handle API errors in the response
            if "error" in data:
                raise Exception(f"Gemini API error: {data['error']['message']}")
            
            # Extract the generated content from Gemini's response format
            if not data.get("candidates") or len(data["candidates"]) == 0:
                raise Exception("Gemini API returned no candidates")
            
            candidate = data["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"]:
                raise Exception("Gemini API returned invalid response structure")
            
            # Extract text from the first part
            parts = candidate["content"]["parts"]
            if not parts or "text" not in parts[0]:
                raise Exception("Gemini API returned no text content")
            
            content = parts[0]["text"]
            
            # Extract usage information (Gemini provides this in usageMetadata)
            usage_data = data.get("usageMetadata", {})
            usage = LLMUsage(
                prompt_tokens=usage_data.get("promptTokenCount", 0),
                completion_tokens=usage_data.get("candidatesTokenCount", 0),
                total_tokens=usage_data.get("totalTokenCount", 0),
            )
            
            # Return our standardized response
            return LLMResponse(
                content=content,
                model=model,
                provider="gemini",
                usage=usage,
                raw_response=data,  # Keep original response for debugging
            )
    
    async def list_models(self) -> List[str]:
        """
        Get a list of available Gemini models.
        
        This queries Gemini's models endpoint and returns a list of model names
        that can be used for text generation.
        
        Returns:
            List of available model names
            
        Raises:
            Exception: If the API call fails
        """
        url = f"{self.base_url}/models"
        params = {"key": self.api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, 
                headers=self.headers, 
                params=params,
                timeout=30.0
            )
            
            if response.status_code != 200:
                error_detail = response.text
                raise Exception(
                    f"Gemini models API error (status {response.status_code}): {error_detail}"
                )
            
            data = response.json()
            
            # Extract model names from Gemini's response
            # Gemini returns models with full names like "models/gemini-1.5-flash"
            if "models" not in data:
                return []
            
            all_models = []
            for model in data["models"]:
                # Extract just the model name part after "models/"
                model_name = model.get("name", "")
                if model_name.startswith("models/"):
                    model_name = model_name[7:]  # Remove "models/" prefix
                
                # Filter for text generation models (exclude embedding models, etc.)
                if (model_name and
                    "generateContent" in model.get("supportedGenerationMethods", []) and
                    any(gen_model in model_name.lower() for gen_model in ["gemini"])):
                    all_models.append(model_name)

            # Sort with newest/most capable first
            # Priority order: gemini-2.5 > gemini-2.0 > gemini-1.5 > gemini-1.0
            def model_sort_key(name):
                if "2.5" in name:
                    return (2.5, "flash-thinking" in name, "pro" in name)
                elif "2.0" in name:
                    return (2.0, "flash-thinking" in name, "pro" in name)
                elif "1.5" in name:
                    return (1.5, "flash" in name, "pro" in name)
                else:
                    return (1.0, False, False)

            return sorted(all_models, key=model_sort_key, reverse=True)