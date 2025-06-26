# Testing the ArXiv Paper Downloader Server

This document provides comprehensive testing procedures for the ArXiv Paper Downloader MCP Server.

## MCP Inspector Testing

### Starting the Inspector
```bash
cd arxiv-downloader
mcp dev server.py
```

This will start the MCP Inspector at `http://localhost:6274` with an authentication token.

### Inspector Configuration
**Important:** In the MCP Inspector web interface, update the server configuration:
- **Command:** Change from `uv` to `mcp`
- **Arguments:** Change to `run server.py`

This tells the inspector how to properly start and connect to your MCP server.

### What to Test

#### 1. Tools Tab

**echo_test**
- Purpose: Test basic server connectivity
- Input: `{"message": "Testing ArXiv server"}`
- Expected: JSON response with echo message and server status

**search_papers**
- Purpose: Test ArXiv search functionality
- Test Cases:
  - Basic search: `{"query": "machine learning", "max_results": 5}`
  - Category search: `{"query": "neural networks", "category": "cs.LG", "max_results": 3}`
  - Author search: `{"query": "au:Hinton", "max_results": 10}`
  - Sort by date: `{"query": "transformer", "sort_by": "lastUpdatedDate"}`
- Expected: JSON with paper results including titles, authors, abstracts

**get_paper_info**
- Purpose: Test metadata retrieval for specific papers
- Test Cases:
  - Classic paper: `{"arxiv_id": "1706.03762"}` (Attention Is All You Need)
  - Recent paper: `{"arxiv_id": "2301.00001"}`
  - Old format ID: `{"arxiv_id": "cs/0001001"}`
- Expected: Detailed paper metadata with full abstract and author info

**download_paper**
- Purpose: Test PDF download functionality
- Test Cases:
  - Default filename: `{"arxiv_id": "1706.03762"}`
  - Custom filename: `{"arxiv_id": "1706.03762", "filename": "transformer_paper"}`
  - Invalid ID: `{"arxiv_id": "invalid.id"}` (should handle gracefully)
- Expected: Download result with local path and file size, or error message

**list_categories**
- Purpose: Test category listing
- Input: No parameters `{}`
- Expected: List of ArXiv categories with descriptions

#### 2. Resources Tab

**arxiv://server-info**
- Purpose: Check server configuration
- Expected: JSON with server name, provider info, download directory, and supported features

**arxiv://download-status**
- Purpose: Check download directory status
- Expected: JSON with directory info, file count, and recent downloads (if any)

#### 3. Prompts Tab

**paper-analysis**
- Purpose: Test paper analysis prompt generation
- Test Cases:
  - General analysis: `{"arxiv_id": "1706.03762", "focus": "general"}`
  - Methodology focus: `{"arxiv_id": "1706.03762", "focus": "methodology"}`
  - Results focus: `{"arxiv_id": "1706.03762", "focus": "results"}`
- Expected: Structured analysis prompt with paper metadata

**citation-format**
- Purpose: Test citation generation
- Test Cases:
  - APA style: `{"arxiv_id": "1706.03762", "style": "apa"}`
  - BibTeX format: `{"arxiv_id": "1706.03762", "style": "bibtex"}`
  - IEEE style: `{"arxiv_id": "1706.03762", "style": "ieee"}`
- Expected: Properly formatted citations

### Testing Workflow
1. Start inspector: `mcp dev server.py`
2. Open browser to provided URL with auth token
3. Configure command/arguments as noted above
4. Test each tool systematically
5. Verify resources provide expected information
6. Test prompts with different parameters

## Manual Testing

### Basic Functionality Test
```bash
# Test server import and basic startup
python -c "from server import mcp; print('Server imports successfully')"

# Test provider initialization
python -c "from providers.arxiv_provider import ArXivProvider; print('Provider imports successfully')"
```

### Download Directory Test
```bash
# Create test download directory
mkdir -p ./test_papers

# Set environment variable
export ARXIV_DOWNLOAD_DIR="./test_papers"

# Run server and test download
python server.py
```

### Network Connectivity Test
Test ArXiv API accessibility:
```bash
# Test ArXiv API directly
curl "http://export.arxiv.org/api/query?search_query=all:machine+learning&max_results=1"
```

## Claude Desktop Integration Testing

### Configuration Setup
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv-downloader": {
      "command": "mcp",
      "args": ["run", "/full/path/to/arxiv-downloader/server.py"],
      "env": {
        "ARXIV_DOWNLOAD_DIR": "/path/to/your/papers"
      }
    }
  }
}
```

### Testing in Claude Desktop

1. **Server Connection Test**
   - Ask Claude: "Can you test the ArXiv server connection?"
   - Expected: Claude should use `echo_test` and confirm connection

2. **Paper Search Test**
   - Ask Claude: "Search for papers about 'transformer neural networks' and show me the top 3 results"
   - Expected: Claude should use `search_papers` and display formatted results

3. **Paper Download Test**
   - Ask Claude: "Download the 'Attention Is All You Need' paper (ArXiv ID: 1706.03762)"
   - Expected: Claude should use `download_paper` and confirm successful download

4. **Citation Generation Test**
   - Ask Claude: "Generate an APA citation for ArXiv paper 1706.03762"
   - Expected: Claude should use `citation_format_prompt` and provide formatted citation

5. **Paper Analysis Test**
   - Ask Claude: "Analyze the methodology of paper 1706.03762"
   - Expected: Claude should use `paper_analysis_prompt` with methodology focus

## Performance Testing

### Search Performance
- Test various query complexities
- Measure response times for different result set sizes
- Test category filtering performance

### Download Performance
- Test download of papers with different file sizes
- Verify download resumption (if interrupted)
- Test concurrent downloads

### Error Handling
- Test with invalid ArXiv IDs
- Test with network interruptions
- Test with permission errors for download directory

## Validation Checklist

### Core Functionality
- [ ] Server starts without errors
- [ ] All tools are properly registered and callable
- [ ] Resources provide expected information
- [ ] Prompts generate properly formatted output

### Search Functionality
- [ ] Basic keyword search works
- [ ] Category filtering works
- [ ] Author search works  
- [ ] Date sorting works
- [ ] Advanced query syntax works

### Download Functionality
- [ ] PDF downloads complete successfully
- [ ] Custom filenames work
- [ ] Download directory is created if needed
- [ ] File sizes are reported correctly
- [ ] Error handling for failed downloads

### Integration Testing
- [ ] MCP Inspector connection successful
- [ ] Claude Desktop integration works
- [ ] All tools accessible from Claude
- [ ] Resources and prompts work in Claude

### Error Scenarios
- [ ] Invalid ArXiv IDs handled gracefully
- [ ] Network errors handled properly
- [ ] File system errors handled appropriately
- [ ] Malformed queries handled correctly

## Common Issues and Solutions

### Server Won't Start
- **Issue**: ImportError or module not found
- **Solution**: Ensure all dependencies installed with `pip install -r requirements.txt`

### ArXiv Search Fails
- **Issue**: Network connection or API errors
- **Solution**: Check internet connectivity and try with simpler queries

### Download Permission Errors
- **Issue**: Cannot write to download directory
- **Solution**: Check directory permissions or set different `ARXIV_DOWNLOAD_DIR`

### MCP Inspector Connection Issues
- **Issue**: Server not connecting in inspector
- **Solution**: Verify command is `mcp` and arguments are `run server.py`

### Claude Desktop Not Recognizing Server
- **Issue**: Server not appearing in Claude
- **Solution**: Check JSON configuration syntax and file paths in config

## Debugging Tips

### Enable Verbose Logging
```bash
# Add debug output to server
python server.py --debug
```

### Test Individual Components
```bash
# Test provider directly
python -c "
import asyncio
from providers.arxiv_provider import ArXivProvider

async def test():
    provider = ArXivProvider()
    result = await provider.search('machine learning', max_results=1)
    print(result)

asyncio.run(test())
"
```

### Network Debugging
```bash
# Test ArXiv connectivity
curl -v "http://export.arxiv.org/api/query?search_query=all:test&max_results=1"
```

This comprehensive testing approach ensures the ArXiv Paper Downloader server works correctly across all supported platforms and integration scenarios.