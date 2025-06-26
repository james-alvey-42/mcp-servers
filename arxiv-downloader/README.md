# ArXiv Paper Downloader MCP Server

A Model Context Protocol (MCP) server for searching and downloading academic papers from ArXiv. Seamlessly search, analyze, and download research papers from Claude Desktop or any MCP client.

## 🚀 Features

- **Advanced Search**: Query ArXiv with keywords, authors, titles, and subject categories
- **Metadata Extraction**: Get comprehensive paper information including abstracts, authors, and citations
- **PDF Download**: Download papers to local storage with organized file management
- **Citation Formatting**: Generate properly formatted citations in multiple styles (APA, MLA, IEEE, BibTeX)
- **Category Filtering**: Search within specific research domains (AI, ML, Physics, Math, etc.)
- **Analysis Tools**: Built-in prompts for systematic paper analysis

## 📋 Available Tools

- **`search_papers`**: Search ArXiv for papers matching query terms
- **`get_paper_info`**: Get detailed metadata for a specific ArXiv paper
- **`download_paper`**: Download PDF files to local storage
- **`list_categories`**: Get available ArXiv subject categories
- **`echo_test`**: Test server connectivity

## 📊 Available Resources

- **`arxiv://server-info`**: Server configuration and API endpoints
- **`arxiv://download-status`**: Download directory status and recent downloads

## 📝 Available Prompts

- **`paper-analysis`**: Structured template for analyzing research papers
- **`citation-format`**: Generate properly formatted citations in various styles

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Optional: Configure Download Directory
```bash
# Set custom download location (optional)
export ARXIV_DOWNLOAD_DIR="/path/to/your/papers"

# Or use default: ./papers/ directory
```

### 3. Test the Server
```bash
mcp dev server.py
```

**Note:** In the MCP Inspector web interface that opens:
- Change the default command from `uv` to `mcp`
- Change the arguments to `run server.py`
- Test with: `echo_test()`

### 4. Install in Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "arxiv-downloader": {
      "command": "mcp",
      "args": ["run", "/Users/james/Dropbox/JimDex/60-69 Library/60 Packages/60.19 MCP/mcp-servers/arxiv-downloader/server.py"],
      "env": {
        "ARXIV_DOWNLOAD_DIR": "/Users/james/Downloads/ArXiv-Papers"
      }
    }
  }
}
```

**Important:** Replace the paths with your actual locations:
- Update the server.py path to match your installation
- Set `ARXIV_DOWNLOAD_DIR` to a writable directory like:
  - `/Users/yourusername/Downloads/ArXiv-Papers`
  - `/Users/yourusername/Documents/Research/Papers` 
  - `/Users/yourusername/Desktop/Papers`

## 📚 Usage Examples

### Search for Papers
```python
# Basic search
search_papers("machine learning")

# Search with category filter
search_papers("neural networks", category="cs.LG", max_results=20)

# Author search
search_papers("au:Hinton", sort_by="lastUpdatedDate")

# Advanced query with field specifiers
search_papers("ti:transformer AND abs:attention")
```

### Download Papers
```python
# Download with default filename
download_paper("2301.00001")

# Download with custom filename
download_paper("1706.03762", filename="attention_is_all_you_need")

# Get paper metadata first
get_paper_info("2301.00001")
```

### Citation Generation
```python
# Generate APA citation
citation_format_prompt("1706.03762", style="apa")

# Generate BibTeX entry
citation_format_prompt("1706.03762", style="bibtex")
```

### Paper Analysis
```python
# General analysis
paper_analysis_prompt("1706.03762", focus="general")

# Focus on methodology
paper_analysis_prompt("1706.03762", focus="methodology")
```

## 🔧 Advanced Usage

### Search Query Syntax

The `search_papers` tool supports ArXiv's advanced query syntax:

- **Title search**: `ti:"machine learning"`
- **Author search**: `au:Smith`
- **Abstract search**: `abs:neural`
- **Category search**: `cat:cs.LG`
- **Boolean operators**: `ti:transformer AND abs:attention`
- **Date ranges**: `submittedDate:[2020 TO 2023]`

### Available Categories

Common ArXiv categories include:
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning  
- `cs.CL` - Computation and Language
- `cs.CV` - Computer Vision
- `stat.ML` - Machine Learning (Statistics)
- `math.ST` - Statistics Theory
- `physics.data-an` - Data Analysis

Use `list_categories()` to see all available categories.

### File Organization

Downloaded papers are organized in the specified directory:
```
papers/
├── 2301.00001.pdf
├── 1706.03762.pdf
└── attention_is_all_you_need.pdf
```

## 🔒 Security & Privacy

- **No API Keys Required**: ArXiv is an open repository
- **Local Storage**: All downloads are stored locally on your machine
- **Rate Limiting**: Built-in respect for ArXiv's usage guidelines
- **Error Handling**: Comprehensive error handling and validation

## 🛠️ Development

### Project Structure
```
arxiv-downloader/
├── server.py              # Main FastMCP server
├── requirements.txt       # Python dependencies
├── providers/             # Provider abstraction layer
│   ├── __init__.py
│   ├── base.py           # Abstract base classes
│   └── arxiv_provider.py # ArXiv API implementation
├── README.md             # This file
└── TESTING.md            # Testing procedures
```

### Testing
See [TESTING.md](TESTING.md) for comprehensive testing procedures.

### Adding New Providers
The provider abstraction system makes it easy to add support for other academic databases:
1. Implement the `PaperProvider` abstract base class
2. Add provider initialization to `server.py`
3. Update tool functions to support the new provider

## 📞 Troubleshooting

### Common Issues

**Server won't start**
- Check Python version (3.10+ required)
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Search returns no results**
- Verify ArXiv connectivity with `echo_test()`
- Check query syntax and try simpler terms
- Ensure category codes are valid

**Download fails**
- Check internet connectivity
- Verify ArXiv ID format (e.g., '2301.00001')
- Ensure write permissions for download directory

**MCP Inspector connection issues**
- Use `mcp` command instead of `uv`
- Use `run server.py` as arguments
- Check that FastMCP is properly installed

### Debug Mode
Run with additional logging:
```bash
python server.py --verbose
```

## 🌟 Use Cases

### Research Workflows
- **Literature Review**: Search and download papers for systematic reviews
- **Reference Management**: Generate formatted citations for papers
- **Paper Analysis**: Use structured prompts to analyze methodology and results
- **Category Exploration**: Discover papers in specific research domains

### Academic Writing
- **Citation Generation**: Create properly formatted references
- **Background Research**: Find relevant papers for introduction sections
- **Methodology Review**: Analyze approaches used in similar work
- **Result Comparison**: Compare findings across multiple papers

### Teaching & Learning
- **Course Preparation**: Download papers for course reading lists
- **Student Research**: Help students find and analyze relevant papers
- **Assignment Review**: Quick access to papers referenced in student work

## 📈 Performance

- **Search Speed**: Typically 1-3 seconds for standard queries
- **Download Speed**: Varies by paper size and network connection
- **Local Storage**: No cloud dependencies for downloaded papers
- **Rate Limits**: Automatically respects ArXiv's usage guidelines

## 🤝 Contributing

This server follows the established MCP development patterns:
- Provider abstraction for easy extension
- Comprehensive error handling and validation
- Type-safe interfaces with Pydantic models
- Extensive documentation and examples

See the main project documentation for development guidelines and contribution instructions.

---

**ArXiv Paper Downloader** - Part of the MCP Servers Collection  
**Status**: ✅ Production Ready  
**Last Updated**: 2025-01-26  
**MCP SDK Version**: 1.9.4+