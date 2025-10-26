# LLM Chat GUI - Simple Multi-Provider Interface

A lightweight Tkinter-based GUI application for chatting with multiple LLM providers locally.

## Features

✅ **Multi-Provider Support**
- OpenAI (GPT-4, GPT-3.5-turbo, GPT-4-turbo)
- Google Gemini (Gemini-2.5-Pro, Gemini-2.0-Flash, Gemini-1.5-Pro, Gemini-1.5-Flash)

✅ **Chat Mode**
- Maintains full conversation history
- Sends entire context with each message
- Clear conversation history option

✅ **Simple Interface**
- Provider selection dropdown
- Model selection dropdown
- Temperature control (0.0 - 1.0)
- Color-coded messages (user/assistant/system)
- Token usage statistics

✅ **Zero External Dependencies**
- Uses Python's built-in Tkinter
- Reuses existing provider architecture from llm-api-bridge

## Quick Start

### 1. Set API Keys

```bash
export OPENAI_API_KEY='your-openai-key-here'
export GEMINI_API_KEY='your-gemini-key-here'
```

You only need to set the keys for providers you want to use.

### 2. Run the GUI

```bash
cd llm-api-bridge
python chat_gui.py
```

### 3. Use the Interface

1. **Select Provider**: Choose OpenAI or Gemini from the dropdown
2. **Select Model**: Choose your desired model (updates automatically when provider changes)
3. **Adjust Temperature**: Set randomness level (default: 0.7)
4. **Type Message**: Enter your message in the input box
5. **Send**: Click "Send" or press Ctrl+Enter
6. **View Response**: See the AI response with token usage stats

## Interface Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Provider: [OpenAI ▼] Model: [gpt-3.5-turbo ▼] Temp: [0.7]  │
│ [Clear Chat]                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Chat Display Area                                          │
│  - Blue: User messages                                      │
│  - Green: Assistant responses                               │
│  - Gray: System messages and timestamps                     │
│                                                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Your message:                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Type your message here...                               │ │
│ └─────────────────────────────────────────────────────────┘ │
│                     [Send (Ctrl+Enter)]                     │
└─────────────────────────────────────────────────────────────┘
```

## Keyboard Shortcuts

- **Ctrl+Enter**: Send message

## Features Explained

### Conversation History
The application maintains the full conversation history in memory. Each time you send a message, the entire conversation context is sent to the LLM, enabling coherent multi-turn conversations.

### Provider Switching
You can switch providers and models at any time during a conversation. The conversation history is preserved, allowing you to compare responses from different models on the same context.

### Token Usage
After each response, you'll see token usage statistics:
- Prompt tokens: Input tokens (your message + conversation history)
- Completion tokens: Output tokens (AI response)
- Total tokens: Sum of both

### Temperature Control
Adjust the randomness of responses:
- `0.0` - Deterministic, focused responses
- `0.7` - Balanced (default)
- `1.0` - More creative, varied responses

## Troubleshooting

### "Provider not initialized" Error
**Cause**: API key not set for the selected provider

**Solution**:
```bash
export OPENAI_API_KEY='your-key'
# or
export GEMINI_API_KEY='your-key'
```
Then restart the application.

### Import Errors
**Cause**: Running from wrong directory

**Solution**: Make sure you're in the `llm-api-bridge` directory:
```bash
cd llm-api-bridge
python chat_gui.py
```

### Slow Responses
**Cause**: Normal for LLM API calls (2-10 seconds typical)

**Note**: The GUI may appear frozen during API calls. This is normal - wait for the response.

## Technical Details

### Architecture
- **GUI Framework**: Tkinter (Python built-in)
- **Provider System**: Reuses `providers/` from llm-api-bridge
- **Async Handling**: Uses `asyncio.run()` for async provider calls
- **Message Format**: Uses `LLMMessage` from provider base classes

### File Structure
```
llm-api-bridge/
├── chat_gui.py           # Main GUI application (this file)
├── server.py             # MCP server (separate)
├── providers/
│   ├── base.py          # LLMProvider, LLMMessage, LLMResponse
│   ├── openai_provider.py
│   └── gemini_provider.py
└── requirements.txt      # Python dependencies
```

### Dependencies
All dependencies are already installed for llm-api-bridge:
- `openai` - OpenAI API client
- `google-generativeai` - Gemini API client
- `pydantic` - Data validation
- `tkinter` - Built into Python

## Comparison with MCP Server

| Feature | MCP Server | GUI Application |
|---------|-----------|----------------|
| Interface | MCP protocol | Tkinter GUI |
| Use case | Claude Desktop integration | Standalone local chat |
| Conversation | Managed by MCP client | Managed by GUI |
| Setup | Claude Desktop config | Just run Python script |
| Dependencies | FastMCP | Tkinter (built-in) |

Both use the same provider architecture, ensuring consistent behavior.

## Future Enhancements (Ideas)

- [ ] Save/load conversation history
- [ ] System message configuration
- [ ] Max tokens control
- [ ] Streaming responses
- [ ] Copy message button
- [ ] Export conversation to markdown
- [ ] Dark mode theme
- [ ] Multi-window support for comparing models side-by-side

## License

Same as llm-api-bridge project.
