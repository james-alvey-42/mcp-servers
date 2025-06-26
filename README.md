# MCP Servers Collection

A comprehensive suite of Model Context Protocol (MCP) servers for personal and research contexts. This project provides unified access to multiple LLM APIs, research tools, and productivity integrations through the MCP protocol.

## 🚀 Available Servers

### LLM API Bridge ✅ Complete
**Location:** `llm-api-bridge/`

A unified interface for multiple LLM APIs with built-in comparison and evaluation tools.

**Features:**
- Call OpenAI, Google Gemini, and other LLM providers with consistent API
- Model comparison and evaluation workflows  
- Usage tracking and cost optimization
- Secure API key management

**Quick Start:**
```bash
cd llm-api-bridge
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
mcp install server.py --name "LLM API Bridge"
```

**Tools:** `call_llm`, `list_models`, `echo_test`  
**Resources:** Server info, provider status  
**Prompts:** Model comparison templates

---

### ArXiv Paper Downloader 🚧 Planned
**Location:** `arxiv-downloader/` (coming soon)

Download and manage academic papers from ArXiv with full metadata and contextual detail.

**Planned Features:**
- Search ArXiv by keywords, authors, categories
- Download PDFs with metadata extraction
- Citation management and bibliography tools
- Local paper database with full-text search

---

### Apple Reminders Interface 🚧 Planned  
**Location:** `apple-reminders/` (coming soon)

Direct management of Apple Reminders through MCP for seamless task integration.

**Planned Features:**
- Create, read, update, delete reminders
- Manage reminder lists and categories
- Location-based and time-based reminders
- Integration with Apple's EventKit

---

### Calendar-Reminders Integration 🚧 Planned
**Location:** `calendar-integration/` (coming soon)

Bridge between Calendar and Reminders for intelligent work planning and overcommitment prevention.

**Planned Features:**
- Analyze calendar availability
- Suggest optimal scheduling times
- Workload analysis and warnings
- Time blocking for focused work

## 📚 Documentation

**[Complete Documentation](docs/)** - Comprehensive guides and references

### Quick Links
- **[LLM API Bridge User Guide](docs/tutorials/llm-api-bridge-user-guide.md)** - How to use the LLM API Bridge
- **[Claude Desktop Setup](docs/tutorials/claude-desktop-setup.md)** - Installation and configuration
- **[API Key Security](docs/security/api-key-management.md)** - Secure configuration best practices
- **[MCP Overview](docs/architecture/mcp-overview.md)** - Understanding the Model Context Protocol

## 🏃‍♂️ Quick Start

### 1. Choose a Server
Start with the **LLM API Bridge** for immediate LLM access, or explore the planned servers for future capabilities.

### 2. Follow Setup Guide
Each server includes detailed setup instructions. For the LLM API Bridge:
- Install Python dependencies
- Configure API keys securely
- Test with MCP Inspector
- Install in Claude Desktop

### 3. Explore Documentation
Review the comprehensive documentation in `docs/` for advanced usage, security best practices, and development guides.

## 🛠️ Development Status

### Phase 1: Foundation ✅ Complete
- [x] MCP architecture research and documentation
- [x] FastMCP implementation patterns
- [x] LLM API Bridge server with OpenAI provider
- [x] Comprehensive documentation and security guides

### Phase 2: Multi-Provider Support ✅ Complete
- [x] Google Gemini provider implementation
- [ ] Additional LLM providers (Claude, local models)
- [ ] Enhanced caching and optimization

### Phase 3: Research Tools 📅 Planned
- [ ] ArXiv Paper Downloader server
- [ ] Academic research workflow integration
- [ ] Citation and bibliography management

### Phase 4: Productivity Integration 📅 Planned
- [ ] Apple Reminders interface
- [ ] Calendar integration features
- [ ] Cross-platform task management

## 🔧 Technical Architecture

### MCP Protocol
Built on the Model Context Protocol standard for LLM-application integration:
- **Tools:** Execute actions and functions
- **Resources:** Provide contextual data
- **Prompts:** Reusable interaction templates

### FastMCP Framework
Uses the high-level FastMCP Python framework:
- Decorator-based API (`@mcp.tool()`, `@mcp.resource()`, `@mcp.prompt()`)
- Structured output with Pydantic models
- Built-in development and testing tools

### Provider Abstraction
Consistent interface across different services:
- Unified API for multiple LLM providers
- Standardized response formats
- Pluggable provider architecture

## 🔒 Security

- **Environment-based configuration** for API keys
- **No hardcoded secrets** in any codebase
- **Comprehensive security documentation**
- **Best practices for production deployment**

See [API Key Management Guide](docs/security/api-key-management.md) for detailed security practices.

## 🤝 Contributing

### Getting Started
1. Review [MCP Overview](docs/architecture/mcp-overview.md)
2. Set up [Python Development Environment](docs/tutorials/python-development-setup.md)
3. Follow [FastMCP Implementation Guide](docs/tutorials/fastmcp-implementation-guide.md)

### Development Workflow
- Each server follows the established patterns
- Comprehensive testing with MCP Inspector
- Documentation for all features
- Security review for API integrations

## 📞 Support

### Documentation First
- Check the relevant server README
- Review troubleshooting sections in documentation
- Test basic connectivity with echo tools

### Common Issues
- **Server won't start:** Check Python version (3.10+) and dependencies
- **API key errors:** Verify environment variable configuration  
- **Connection issues:** Test with MCP Inspector first

## 🌟 Project Vision

This project demonstrates the power of the Model Context Protocol for creating unified, secure, and extensible LLM integrations. By providing multiple servers with consistent interfaces, we enable:

- **Seamless LLM provider switching** and comparison
- **Research workflow automation** with academic tools
- **Productivity enhancement** through calendar and task integration
- **Educational resources** for MCP development

The goal is to create a comprehensive toolkit that bridges the gap between AI capabilities and practical daily workflows, all while maintaining security and ease of use.

---

**Latest Update:** Current as of latest commit  
**MCP SDK Version:** 1.9.4+  
**Python Version:** 3.10+
