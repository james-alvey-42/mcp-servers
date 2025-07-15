Integrating Literature RAG with MCP (Model Context Protocol)

  This guide shows how to create an MCP server that exposes your ChromaDB literature database, enabling
   any MCP-compatible client to query your research corpus through a standardized interface.

  🏗️ MCP Server Architecture

  High-Level Design

  MCP Client (Claude Desktop, etc.)
      ↓ (MCP Protocol)
  MCP Server (Your Implementation)
      ↓ (Direct Access)
  ChromaDB Database (Your Literature Corpus)
      ↓ (API Calls)
  OpenAI API (Embeddings + Generation)

  📁 Implementation Structure

  Directory Layout

  scirag-mcp-server/
  ├── src/
  │   ├── __init__.py
  │   ├── server.py              # Main MCP server
  │   ├── rag_manager.py         # RAG system interface
  │   └── config.py              # Configuration
  ├── pyproject.toml             # Project dependencies
  ├── README.md                  # Server documentation
  └── mcp_config.json           # MCP client configuration

  🔧 Core Implementation

  1. MCP Server Implementation (src/server.py)

  """
  MCP Server for Literature RAG System
  Provides tools for querying research literature corpus via ChromaDB
  """
  import asyncio
  import json
  from typing import Any, Sequence
  from pathlib import Path

  from mcp.server import Server, NotificationOptions
  from mcp.server.models import InitializationOptions
  import mcp.server.stdio
  import mcp.types as types

  from .rag_manager import LiteratureRAGManager
  from .config import Config

  class LiteratureRAGServer:
      def __init__(self):
          self.server = Server("literature-rag")
          self.rag_manager = None
          self.config = Config()

          # Register handlers
          self._register_handlers()

      def _register_handlers(self):
          """Register MCP handlers"""

          @self.server.list_tools()
          async def handle_list_tools() -> list[types.Tool]:
              """List available RAG tools"""
              return [
                  types.Tool(
                      name="query_literature",
                      description="Query the research literature corpus using semantic search",
                      inputSchema={
                          "type": "object",
                          "properties": {
                              "question": {
                                  "type": "string",
                                  "description": "Research question to query against the literature"
                              },
                              "max_results": {
                                  "type": "integer",
                                  "description": "Maximum number of relevant chunks to retrieve
  (default: 5)",
                                  "default": 5,
                                  "minimum": 1,
                                  "maximum": 20
                              },
                              "include_sources": {
                                  "type": "boolean",
                                  "description": "Whether to include source document information
  (default: true)",
                                  "default": True
                              },
                              "response_style": {
                                  "type": "string",
                                  "enum": ["concise", "detailed", "technical", "summary"],
                                  "description": "Style of response generation (default: detailed)",
                                  "default": "detailed"
                              }
                          },
                          "required": ["question"]
                      }
                  ),
                  types.Tool(
                      name="explore_corpus",
                      description="Explore the structure and content of the literature corpus",
                      inputSchema={
                          "type": "object",
                          "properties": {
                              "action": {
                                  "type": "string",
                                  "enum": ["list_papers", "corpus_stats", "search_papers",
  "get_paper_info"],
                                  "description": "Type of corpus exploration to perform"
                              },
                              "query": {
                                  "type": "string",
                                  "description": "Search query for paper titles/content (when
  action=search_papers)"
                              },
                              "paper_id": {
                                  "type": "string",
                                  "description": "Specific paper ID to get info about (when
  action=get_paper_info)"
                              }
                          },
                          "required": ["action"]
                      }
                  ),
                  types.Tool(
                      name="analyze_literature",
                      description="Perform advanced analysis on the literature corpus",
                      inputSchema={
                          "type": "object",
                          "properties": {
                              "analysis_type": {
                                  "type": "string",
                                  "enum": ["methodology_comparison", "concept_evolution",
  "gap_analysis", "trend_analysis"],
                                  "description": "Type of literature analysis to perform"
                              },
                              "focus_area": {
                                  "type": "string",
                                  "description": "Specific research area or concept to focus the
  analysis on"
                              },
                              "comparison_papers": {
                                  "type": "array",
                                  "items": {"type": "string"},
                                  "description": "Specific papers to compare (optional)"
                              }
                          },
                          "required": ["analysis_type"]
                      }
                  )
              ]

          @self.server.call_tool()
          async def handle_call_tool(
              name: str, arguments: dict[str, Any] | None
          ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
              """Handle tool calls"""

              # Initialize RAG manager if not already done
              if self.rag_manager is None:
                  self.rag_manager = LiteratureRAGManager(self.config)
                  await self.rag_manager.initialize()

              if name == "query_literature":
                  return await self._handle_query_literature(arguments or {})
              elif name == "explore_corpus":
                  return await self._handle_explore_corpus(arguments or {})
              elif name == "analyze_literature":
                  return await self._handle_analyze_literature(arguments or {})
              else:
                  raise ValueError(f"Unknown tool: {name}")

      async def _handle_query_literature(self, args: dict[str, Any]) -> list[types.TextContent]:
          """Handle literature query requests"""
          question = args.get("question", "")
          max_results = args.get("max_results", 5)
          include_sources = args.get("include_sources", True)
          response_style = args.get("response_style", "detailed")

          if not question:
              return [types.TextContent(type="text", text="Error: Question is required")]

          try:
              result = await self.rag_manager.query(
                  question=question,
                  max_results=max_results,
                  response_style=response_style
              )

              # Format response
              response_text = f"**Question:** {question}\n\n"
              response_text += f"**Answer:** {result['answer']}\n\n"

              if include_sources and result.get('sources'):
                  response_text += f"**Sources:** {', '.join(result['sources'])}\n\n"

              if result.get('context_snippets'):
                  response_text += "**Retrieved Context:**\n"
                  for i, snippet in enumerate(result['context_snippets'][:3], 1):
                      response_text += f"{i}. {snippet[:200]}...\n\n"

              return [types.TextContent(type="text", text=response_text)]

          except Exception as e:
              return [types.TextContent(type="text", text=f"Error querying literature: {str(e)}")]

      async def _handle_explore_corpus(self, args: dict[str, Any]) -> list[types.TextContent]:
          """Handle corpus exploration requests"""
          action = args.get("action", "")

          try:
              if action == "list_papers":
                  papers = await self.rag_manager.list_papers()
                  response = "**Available Papers in Corpus:**\n\n"
                  for i, paper in enumerate(papers, 1):
                      response += f"{i}. {paper}\n"

              elif action == "corpus_stats":
                  stats = await self.rag_manager.get_corpus_stats()
                  response = f"""**Corpus Statistics:**

  - **Total Papers:** {stats['total_papers']}
  - **Total Chunks:** {stats['total_chunks']}
  - **Average Chunks per Paper:** {stats['avg_chunks_per_paper']:.1f}
  - **Corpus Size:** {stats['total_size_mb']:.1f} MB
  - **Last Updated:** {stats['last_updated']}
  """

              elif action == "search_papers":
                  query = args.get("query", "")
                  if not query:
                      response = "Error: Query required for paper search"
                  else:
                      papers = await self.rag_manager.search_papers(query)
                      response = f"**Papers matching '{query}':**\n\n"
                      for paper in papers:
                          response += f"- {paper}\n"

              elif action == "get_paper_info":
                  paper_id = args.get("paper_id", "")
                  if not paper_id:
                      response = "Error: Paper ID required"
                  else:
                      info = await self.rag_manager.get_paper_info(paper_id)
                      response = f"""**Paper Information: {paper_id}**

  - **Chunks:** {info['chunk_count']}
  - **Size:** {info['size_kb']:.1f} KB
  - **Preview:** {info['preview'][:300]}...
  """
              else:
                  response = f"Error: Unknown action '{action}'"

              return [types.TextContent(type="text", text=response)]

          except Exception as e:
              return [types.TextContent(type="text", text=f"Error exploring corpus: {str(e)}")]

      async def _handle_analyze_literature(self, args: dict[str, Any]) -> list[types.TextContent]:
          """Handle advanced literature analysis"""
          analysis_type = args.get("analysis_type", "")
          focus_area = args.get("focus_area", "")

          try:
              result = await self.rag_manager.analyze_literature(
                  analysis_type=analysis_type,
                  focus_area=focus_area,
                  comparison_papers=args.get("comparison_papers", [])
              )

              response = f"**{analysis_type.replace('_', ' ').title()} Analysis**\n\n"
              response += result['analysis']

              if result.get('key_findings'):
                  response += "\n\n**Key Findings:**\n"
                  for finding in result['key_findings']:
                      response += f"- {finding}\n"

              return [types.TextContent(type="text", text=response)]

          except Exception as e:
              return [types.TextContent(type="text", text=f"Error analyzing literature: {str(e)}")]

  async def main():
      """Main server entry point"""
      server_instance = LiteratureRAGServer()

      async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
          await server_instance.server.run(
              read_stream,
              write_stream,
              InitializationOptions(
                  server_name="literature-rag",
                  server_version="1.0.0",
                  capabilities=server_instance.server.get_capabilities(
                      notification_options=NotificationOptions(),
                      experimental_capabilities={},
                  ),
              ),
          )

  if __name__ == "__main__":
      asyncio.run(main())

  2. RAG Manager Interface (src/rag_manager.py)

  """
  RAG Manager for MCP Server
  Interfaces with existing ChromaDB setup and OpenAI API
  """
  import os
  import asyncio
  from pathlib import Path
  from typing import Dict, List, Any
  from datetime import datetime

  import chromadb
  from openai import AsyncOpenAI

  class LiteratureRAGManager:
      def __init__(self, config):
          self.config = config
          self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
          self.chroma_client = None
          self.collection = None

      async def initialize(self):
          """Initialize ChromaDB connection"""
          self.chroma_client = chromadb.PersistentClient(
              path=str(self.config.chromadb_path)
          )

          try:
              self.collection = self.chroma_client.get_collection(
                  self.config.collection_name
              )
          except Exception:
              raise RuntimeError(f"ChromaDB collection '{self.config.collection_name}' not found.
  Please run the setup script first.")

      async def query(self, question: str, max_results: int = 5, response_style: str = "detailed") ->
  Dict[str, Any]:
          """Query the literature corpus"""

          # Create embedding for question
          embedding_response = await self.client.embeddings.create(
              input=[question],
              model=self.config.embedding_model
          )
          query_embedding = embedding_response.data[0].embedding

          # Search ChromaDB
          results = self.collection.query(
              query_embeddings=[query_embedding],
              n_results=max_results
          )

          # Format context
          context = self._format_context(results)
          sources = list(set([meta['source'] for meta in results['metadatas'][0]]))

          # Generate response with style-specific prompts
          prompt = self._build_prompt(question, context, response_style)

          chat_response = await self.client.chat.completions.create(
              model=self.config.generation_model,
              messages=[
                  {"role": "system", "content": self._get_system_prompt(response_style)},
                  {"role": "user", "content": prompt}
              ],
              temperature=self.config.temperature,
              max_tokens=self._get_max_tokens(response_style)
          )

          return {
              'question': question,
              'answer': chat_response.choices[0].message.content,
              'sources': sources,
              'context_snippets': results['documents'][0],
              'style': response_style
          }

      async def list_papers(self) -> List[str]:
          """List all papers in the corpus"""
          results = self.collection.get()
          sources = set()
          for metadata in results['metadatas']:
              sources.add(metadata['source'])
          return sorted(list(sources))

      async def get_corpus_stats(self) -> Dict[str, Any]:
          """Get corpus statistics"""
          results = self.collection.get()
          sources = set()
          total_chars = 0

          for metadata, doc in zip(results['metadatas'], results['documents']):
              sources.add(metadata['source'])
              total_chars += len(doc)

          return {
              'total_papers': len(sources),
              'total_chunks': len(results['documents']),
              'avg_chunks_per_paper': len(results['documents']) / len(sources) if sources else 0,
              'total_size_mb': total_chars / (1024 * 1024),
              'last_updated': datetime.now().isoformat()
          }

      async def search_papers(self, query: str) -> List[str]:
          """Search for papers by title/content"""
          # Simple keyword search through paper sources
          all_papers = await self.list_papers()
          matching_papers = [
              paper for paper in all_papers
              if query.lower() in paper.lower()
          ]
          return matching_papers

      async def get_paper_info(self, paper_id: str) -> Dict[str, Any]:
          """Get information about a specific paper"""
          results = self.collection.get(
              where={"source": {"$eq": f"{paper_id}.md"}}
          )

          if not results['documents']:
              raise ValueError(f"Paper {paper_id} not found")

          total_chars = sum(len(doc) for doc in results['documents'])
          preview = results['documents'][0][:500] if results['documents'] else ""

          return {
              'chunk_count': len(results['documents']),
              'size_kb': total_chars / 1024,
              'preview': preview
          }

      async def analyze_literature(self, analysis_type: str, focus_area: str = "", comparison_papers:
  List[str] = None) -> Dict[str, Any]:
          """Perform advanced literature analysis"""

          if analysis_type == "methodology_comparison":
              query = f"Compare the methodological approaches used for {focus_area}"
          elif analysis_type == "concept_evolution":
              query = f"How has the concept of {focus_area} evolved across the papers?"
          elif analysis_type == "gap_analysis":
              query = f"What research gaps exist in {focus_area}?"
          elif analysis_type == "trend_analysis":
              query = f"What are the current trends and future directions in {focus_area}?"
          else:
              raise ValueError(f"Unknown analysis type: {analysis_type}")

          # Use enhanced query for analysis
          result = await self.query(query, max_results=10, response_style="technical")

          # Extract key findings (simple implementation)
          key_findings = self._extract_key_findings(result['answer'])

          return {
              'analysis': result['answer'],
              'key_findings': key_findings,
              'analysis_type': analysis_type,
              'focus_area': focus_area
          }

      def _format_context(self, results) -> str:
          """Format retrieved context for LLM"""
          context = ""
          for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
              context += f"\n\n--- Source {i+1}: {metadata['source']} ---\n{doc}"
          return context

      def _build_prompt(self, question: str, context: str, style: str) -> str:
          """Build style-specific prompts"""
          base_prompt = f"""Based on the following context from research papers, answer the question.

  Context:
  {context}

  Question: {question}

  Answer:"""

          if style == "concise":
              return f"Provide a brief, direct answer.\n\n{base_prompt}"
          elif style == "technical":
              return f"Provide a detailed technical response with specific methods and
  results.\n\n{base_prompt}"
          elif style == "summary":
              return f"Provide a comprehensive summary covering multiple
  perspectives.\n\n{base_prompt}"
          else:  # detailed
              return f"Provide a thorough response with proper citations.\n\n{base_prompt}"

      def _get_system_prompt(self, style: str) -> str:
          """Get style-specific system prompts"""
          base = "You are a research assistant specializing in scientific literature analysis."

          style_prompts = {
              "concise": f"{base} Provide brief, direct answers.",
              "detailed": f"{base} Provide comprehensive responses with citations.",
              "technical": f"{base} Focus on technical details, methods, and quantitative results.",
              "summary": f"{base} Provide broad overviews synthesizing multiple sources."
          }

          return style_prompts.get(style, style_prompts["detailed"])

      def _get_max_tokens(self, style: str) -> int:
          """Get token limits by style"""
          return {
              "concise": 200,
              "detailed": 800,
              "technical": 1000,
              "summary": 600
          }.get(style, 800)

      def _extract_key_findings(self, analysis_text: str) -> List[str]:
          """Extract key findings from analysis (simple implementation)"""
          # Simple extraction - could be enhanced with NLP
          sentences = analysis_text.split('. ')
          key_sentences = [
              s.strip() for s in sentences
              if any(keyword in s.lower() for keyword in ['important', 'significant', 'key', 'main',
  'primary'])
          ]
          return key_sentences[:5]  # Return top 5

  3. Configuration (src/config.py)

  """
  Configuration for Literature RAG MCP Server
  """
  from pathlib import Path
  import os

  class Config:
      def __init__(self):
          # Paths (adjust to your local_test setup)
          self.base_path = Path(os.environ.get("SCIRAG_BASE_PATH", ""))
          self.chromadb_path = self.base_path / "embeddings" / "chromadb"

          # ChromaDB settings
          self.collection_name = "local_literature_collection"

          # OpenAI settings
          self.embedding_model = "text-embedding-3-large"
          self.generation_model = "gpt-4o-mini"
          self.temperature = 0.1

          # Validate paths
          if not self.chromadb_path.exists():
              raise RuntimeError(f"ChromaDB path does not exist: {self.chromadb_path}")

  4. Project Configuration (pyproject.toml)

  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"

  [project]
  name = "literature-rag-mcp-server"
  version = "1.0.0"
  description = "MCP Server for Literature RAG System"
  authors = [
      {name = "Your Name", email = "your.email@example.com"},
  ]
  dependencies = [
      "mcp>=1.0.0",
      "chromadb>=0.4.0",
      "openai>=1.0.0",
      "asyncio",
  ]

  [project.scripts]
  literature-rag-server = "src.server:main"

  🔗 MCP Client Configuration

  Claude Desktop Integration (mcp_config.json)

  {
    "mcpServers": {
      "literature-rag": {
        "command": "uv",
        "args": [
          "--directory", "/path/to/scirag-mcp-server",
          "run", "literature-rag-server"
        ],
        "env": {
          "OPENAI_API_KEY": "your-openai-api-key",
          "SCIRAG_BASE_PATH": "/path/to/scirag/local_test"
        }
      }
    }
  }

  🚀 Usage Examples

  Basic Literature Query

  User: Use the literature RAG to query "What are the main challenges in simulation-based inference?"

  Assistant uses query_literature tool:
  {
    "question": "What are the main challenges in simulation-based inference?",
    "max_results": 5,
    "response_style": "detailed"
  }

  Corpus Exploration

  User: Show me what papers are available in the literature corpus

  Assistant uses explore_corpus tool:
  {
    "action": "list_papers"
  }

  Advanced Analysis

  User: Analyze the evolution of robustness concepts in the literature

  Assistant uses analyze_literature tool:
  {
    "analysis_type": "concept_evolution",
    "focus_area": "robustness in cosmological simulations"
  }

  🎯 Advanced Features

  Custom Prompt Templates

  # In rag_manager.py
  ANALYSIS_PROMPTS = {
      "methodology_comparison": """
      Compare and contrast the methodological approaches across papers.
      Focus on:
      1. Data collection methods
      2. Analysis techniques
      3. Validation approaches
      4. Limitations and assumptions
      """,

      "gap_analysis": """
      Identify research gaps and unexplored areas.
      Consider:
      1. Methodological gaps
      2. Theoretical limitations
      3. Empirical needs
      4. Future research directions
      """
  }

  Query Enhancement

  async def enhanced_query(self, question: str, filters: Dict[str, Any] = None):
      """Enhanced query with filtering and ranking"""

      # Apply filters (date range, paper type, etc.)
      where_clause = self._build_where_clause(filters)

      results = self.collection.query(
          query_embeddings=[query_embedding],
          n_results=max_results,
          where=where_clause
      )

      # Re-rank results based on relevance scores
      ranked_results = self._rerank_results(results, question)

      return ranked_results

  Integration Benefits

  1. Seamless Access: Query your literature from any MCP-compatible client
  2. Standardized Interface: Consistent API across different applications
  3. Rich Functionality: Advanced analysis beyond simple search
  4. Scalable Architecture: Easy to extend with new tools and capabilities
  5. Cross-Platform: Works with Claude Desktop, CLI tools, and custom clients

  This MCP integration transforms your local RAG system into a powerful, accessible research tool that
  can be used across your entire workflow.