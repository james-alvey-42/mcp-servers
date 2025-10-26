# Multi-Modal Support in LLM API Bridge

The LLM API Bridge now supports multi-modal inputs, allowing you to send images along with text to supported models.

## Supported Providers

Currently, **Gemini** is the primary provider with full multi-modal support:
- ✅ **Gemini 2.5 Pro** - Full multi-modal support (text + images)
- ✅ **Gemini 2.5 Flash** - Full multi-modal support (text + images)
- ✅ **Gemini 2.0 Flash** - Full multi-modal support (text + images)

## Content Types

The system supports the following content types:

### 1. Text Content (Backward Compatible)
Simple string content for text-only messages.

### 2. Image Content
- **`image_base64`**: Inline base64-encoded images
  - Supported formats: PNG, JPEG, WebP, HEIC, HEIF
  - Maximum size: 20MB per image
  - Maximum images: 16 per request

- **`image_url`**: Images from URLs (planned - currently converts to placeholder)

## Usage Examples

### Example 1: Text-Only Message (Backward Compatible)

```python
from mcp__llm_bridge__call_llm import call_llm

response = call_llm(
    provider="gemini",
    model="gemini-2.5-pro",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
```

### Example 2: Image with Text

```python
import base64

# Read and encode image
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = call_llm(
    provider="gemini",
    model="gemini-2.5-pro",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What objects do you see in this image?"},
                {
                    "type": "image_base64",
                    "data": image_data,
                    "mime_type": "image/png"
                }
            ]
        }
    ]
)
```

### Example 3: Multiple Images

```python
import base64

# Load multiple images
with open("photo1.jpg", "rb") as f:
    image1 = base64.b64encode(f.read()).decode("utf-8")

with open("photo2.jpg", "rb") as f:
    image2 = base64.b64encode(f.read()).decode("utf-8")

response = call_llm(
    provider="gemini",
    model="gemini-2.5-pro",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images:"},
                {"type": "image_base64", "data": image1, "mime_type": "image/jpeg"},
                {"type": "image_base64", "data": image2, "mime_type": "image/jpeg"},
                {"type": "text", "text": "What are the main differences?"}
            ]
        }
    ]
)
```

### Example 4: Multi-Turn Conversation with Images

```python
import base64

# First message with image
with open("diagram.png", "rb") as f:
    diagram_data = base64.b64encode(f.read()).decode("utf-8")

response1 = call_llm(
    provider="gemini",
    model="gemini-2.5-pro",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain this diagram:"},
                {"type": "image_base64", "data": diagram_data, "mime_type": "image/png"}
            ]
        }
    ]
)

# Continue conversation with text only
response2 = call_llm(
    provider="gemini",
    model="gemini-2.5-pro",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain this diagram:"},
                {"type": "image_base64", "data": diagram_data, "mime_type": "image/png"}
            ]
        },
        {"role": "assistant", "content": response1.content},
        {"role": "user", "content": "Can you provide more detail about the components?"}
    ]
)
```

## Using in Claude Code

When using the MCP server in Claude Code, you can leverage multi-modal capabilities like this:

```python
# Claude Code will have access to the mcp__llm-bridge__call_llm tool
# You can ask Claude Code to analyze images:

"Can you use the Gemini API to analyze this image at /path/to/image.png?"

# Claude Code will:
# 1. Read the image file
# 2. Base64 encode it
# 3. Call the MCP tool with multi-modal content
# 4. Return the analysis
```

## Content Part Schema

### Text Part
```python
{
    "type": "text",
    "text": "Your text content here"
}
```

### Base64 Image Part
```python
{
    "type": "image_base64",
    "data": "base64_encoded_data_here",
    "mime_type": "image/png"  # or "image/jpeg", "image/webp", etc.
}
```

### Image URL Part (Planned)
```python
{
    "type": "image_url",
    "url": "https://example.com/image.png"
}
```

## Best Practices

1. **Image Size**: Keep images under 5MB for optimal performance
2. **Image Format**: Use PNG or JPEG for best compatibility
3. **Resolution**: Resize large images before encoding to reduce data transfer
4. **MIME Types**: Always specify the correct MIME type
5. **Context**: Provide clear text instructions with images for better results

## Future Modalities

The architecture supports expansion to additional modalities:

### Video (Planned)
- Video analysis and understanding
- Frame extraction and analysis
- Motion and activity recognition

### Audio (Planned)
- Speech recognition
- Audio classification
- Sound analysis

### PDF Documents (Planned)
- Direct PDF processing
- Multi-page document analysis
- Table and figure extraction

## Technical Details

### Architecture

The multi-modal support is built on a flexible content part system:

1. **`ContentPart`** class in `base.py` defines individual content pieces
2. **`LLMMessage`** supports both `str` (text-only) and `List[ContentPart]` (multi-modal)
3. **Provider abstraction** handles conversion to provider-specific formats
4. **Backward compatibility** maintained for existing text-only code

### Provider-Specific Implementation

#### Gemini Provider
- Converts `ContentPart` objects to Gemini's `parts` format
- Supports `inline_data` for base64 images
- Handles multiple images per message
- Maintains conversation context across multi-modal exchanges

## Troubleshooting

### Common Issues

1. **"Invalid base64 data"**
   - Ensure data is properly base64 encoded
   - Check that you're not including the data URL prefix (`data:image/png;base64,`)

2. **"Image too large"**
   - Resize images before encoding
   - Maximum 20MB per image

3. **"Unsupported MIME type"**
   - Stick to: `image/png`, `image/jpeg`, `image/webp`

4. **"Too many images"**
   - Maximum 16 images per request
   - Consider splitting into multiple requests

## Examples Repository

See the `examples/` directory for complete working examples:
- `image_analysis.py` - Basic image analysis
- `multi_image_comparison.py` - Compare multiple images
- `document_ocr.py` - Extract text from images
- `visual_qa.py` - Visual question answering

## API Reference

### ContentPart Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | str | Yes | Content type: "text", "image_base64", "image_url" |
| `text` | str | Conditional | Text content (required if type="text") |
| `data` | str | Conditional | Base64 data (required if type="image_base64") |
| `mime_type` | str | Conditional | MIME type (required if type="image_base64") |
| `url` | str | Conditional | Image URL (required if type="image_url") |

### Supported MIME Types

- `image/png`
- `image/jpeg`
- `image/webp`
- `image/heic`
- `image/heif`

## Performance Considerations

1. **Token Usage**: Images consume tokens based on resolution
   - Higher resolution = more tokens
   - Gemini uses ~258 tokens per image (1024x1024)

2. **Latency**: Multi-modal requests take longer than text-only
   - Expected: 2-5 seconds for single image
   - Multiple images: Add ~1 second per additional image

3. **Cost**: Multi-modal requests cost more than text-only
   - Check current Gemini pricing for vision capabilities

## Security Notes

1. **Data Privacy**: Images are sent to the provider's API
2. **API Keys**: Keep your API keys secure
3. **Image Content**: Be mindful of sensitive information in images
4. **Rate Limits**: Provider rate limits apply to multi-modal requests

## Migration Guide

### From Text-Only to Multi-Modal

Before (text-only):
```python
messages=[
    {"role": "user", "content": "Describe a sunset"}
]
```

After (with image):
```python
messages=[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this sunset:"},
            {"type": "image_base64", "data": sunset_image_data, "mime_type": "image/jpeg"}
        ]
    }
]
```

**Note**: Text-only messages still work with the original format - no migration required for existing code.
