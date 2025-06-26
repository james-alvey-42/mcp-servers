# LLM API Bridge MCP Server

A Model Context Protocol (MCP) server that provides unified access to multiple LLM APIs.

## Features

- Unified interface for multiple LLM providers (OpenAI, Google Gemini)
- Structured output with type safety
- Usage tracking and statistics
- Configuration management

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export GEMINI_API_KEY="your-gemini-key"
   ```

3. Test the server:
   ```bash
   mcp dev server.py
   ```
   
   **Note:** In the MCP Inspector web interface that opens:
   - Change the default command from `uv` to `mcp`
   - Change the arguments to `run server.py`
   - This allows the inspector to properly connect to your server

4. Install in Claude Desktop:
   ```bash
   mcp install server.py --name "LLM API Bridge"
   ```

## Usage

This server provides tools to call different LLM APIs through a unified interface, making it easy to compare responses and manage multiple AI models from within Claude Desktop or other MCP clients.