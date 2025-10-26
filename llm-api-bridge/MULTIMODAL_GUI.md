# Multimodal Chat GUI Guide

## Overview

The Platypus Chat GUI now supports **multimodal messages** with **live message structure preview**! You can attach images, audio, and video files to your messages, control the exact order they appear, and insert text at specific positions. This gives you complete control over how your multimodal prompt is structured.

## Features

### 🎯 Live Message Structure Preview
- See exactly how your message will be structured before sending
- Visual preview shows the order of all content parts (files and text)
- Reorder parts using ▲/▼ buttons
- Insert text markers at specific positions in your message
- Perfect for complex prompts like "Compare this: [image1] to this: [image2]"

### 🎨 Drag-and-Drop Support
- Simply drag image, audio, or video files directly into the text input area
- Files are added to the message structure in the order you drop them
- Rearrange them after dropping if needed

### 📎 File Picker
- Click the 📎 button to open a file picker dialog
- Select one or multiple files to attach
- Supported file types are filtered for easy selection

### 🖼️ Supported File Types

**Images:**
- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- GIF (`.gif`)
- WebP (`.webp`)
- BMP (`.bmp`)

**Audio:**
- MP3 (`.mp3`)
- WAV (`.wav`)

**Video:**
- MP4 (`.mp4`)
- WebM (`.webm`)

## Installation

1. **Install the new dependency:**
   ```bash
   cd mcp-servers/llm-api-bridge
   pip install -r requirements.txt
   ```

   This will install `tkinterdnd2` which provides drag-and-drop support.

2. **Run the GUI:**
   ```bash
   python chat_gui.py
   ```

## Usage

### Basic Workflow

1. **Attach files** by dragging them into the input area or clicking 📎
2. **Preview appears** showing the message structure
3. **Reorder or insert text markers** as needed
4. **Type your message** in the text input
5. **Send** - your message is structured exactly as shown in the preview!

### Attaching Files

**Method 1: Drag and Drop**
1. Find the file(s) you want to attach in your file explorer
2. Drag them into the text input area
3. They appear in the "Message Structure" preview

**Method 2: File Picker Button**
1. Click the 📎 button above the Send button
2. Select files in the dialog
3. Click "Open" to attach them

### Controlling Message Order

The **Message Structure** preview shows exactly how your message will be sent:

**Example structure:**
```
Message Structure (in order):
1. 🖼️ cat.jpg
2. 📝 [Your text message]
3. 🖼️ dog.jpg
```

**Reordering:**
- Click ▲ to move a part up
- Click ▼ to move a part down
- Click ✕ to remove a part

**Adding text markers:**
- Click "+ Insert Text Marker" to add a placeholder
- Your typed message will replace the marker(s) when sent
- Multiple markers? Your text replaces ALL of them (useful for repeated context)

### Advanced Examples

**Example 1: Compare two images**
```
1. 📝 [Your text message]  → "Compare these two images:"
2. 🖼️ image1.jpg
3. 📝 [Your text message]  → "versus:"
4. 🖼️ image2.jpg
```
Type: "Compare these two images: versus:"
Result: Text is split at the markers!

**Example 2: Image in the middle**
```
1. 📝 [Your text message]
2. 🖼️ diagram.jpg
3. 📝 [Your text message]
```
Type: "Here is a diagram: What does this show?"
(Note: Currently all text goes to one marker, so use this for simple cases)

**Example 3: Just images with caption**
```
1. 🖼️ photo1.jpg
2. 🖼️ photo2.jpg
3. 📝 [Your text message]
```
Type: "Describe the differences between these images"

### Sending Multimodal Messages

1. Structure your message in the preview
2. Type your text in the input area
3. Click "Send" or press Ctrl+Return

The files will be sent to the LLM in the exact order shown. The server will automatically:
- Read the files from disk
- Encode them as base64
- Detect the MIME type
- Send them in the proper format for the selected provider

### Visual Feedback

**In the Message Structure:**
- 🖼️ Image files
- 🎵 Audio files
- 🎬 Video files
- 📄 Other files
- 📝 Text markers (placeholders for your message)

Each part shows:
- Order number (1., 2., 3., etc.)
- File type icon and name OR text marker
- Reorder buttons (▲ ▼)
- Remove button (✕)

**Status Messages:**
- "Attached X file(s)" - shown when files are attached
- File names are displayed in the chat when sent

## Provider Support

### Gemini (Recommended for Multimodal)
✅ **Full support** for images, audio, and video
- Use `gemini-2.5-pro` or `gemini-2.5-flash`
- Handles all file types natively

### OpenAI
⚠️ **Limited support** - OpenAI provider currently has serialization issues with multimodal content
- Stick to text-only for now with OpenAI models

### Anthropic
⚠️ **Limited support** - Similar to OpenAI
- Text-only recommended for Claude models via the GUI

## Technical Details

### How It Works

1. **File Attachment:**
   - Files are stored as file paths in `self.attached_files`
   - Visual chips are created using Tkinter frames

2. **Message Building:**
   - When you send a message with files, the content is built as a list:
     ```python
     content = [file1_path, file2_path, "your text message"]
     ```

3. **Processing:**
   - The `LLMMessage` is created with this list as content
   - The provider's `_process_content_item()` method handles each item
   - Files are automatically read, encoded, and formatted
   - Text remains as simple text parts

4. **Display:**
   - In the chat display, files are shown as `[Attached: filename1, filename2]`
   - This keeps the chat history readable

### Example Message Structure

**Simple text message:**
```python
LLMMessage(role="user", content="What is this?")
```

**Multimodal message:**
```python
LLMMessage(role="user", content=[
    "/path/to/image.png",
    "What animal is in this image?"
])
```

The server's `_process_content_item()` function in `server.py` automatically detects which items are files and processes them accordingly.

## Tips

1. **Start with Gemini:** The best experience for multimodal is with Gemini models
2. **Multiple Files:** You can attach multiple images to ask comparison questions
3. **File Size:** Be mindful of large files - they increase token usage
4. **Clear Files:** Files are automatically cleared after sending, or you can remove them manually

## Troubleshooting

**Drag-and-drop not working?**
- Make sure `tkinterdnd2` is installed: `pip install tkinterdnd2`
- Try the file picker button (📎) instead

**Files not being processed?**
- Check the file type is supported (see list above)
- Verify the file path is valid and accessible
- Check console for error messages

**Provider errors?**
- Use Gemini for multimodal content
- OpenAI and Anthropic providers may have limitations in the GUI

## Future Enhancements

Possible improvements:
- Image thumbnails in the file chips
- File size validation and warnings
- Support for more file types
- Better multimodal support for OpenAI/Anthropic providers
- Paste image from clipboard
