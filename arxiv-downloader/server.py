#!/usr/bin/env python3
"""
ArXiv Paper Downloader MCP Server

A Model Context Protocol server for searching and downloading academic papers from ArXiv.
Provides unified access to ArXiv's search API and PDF download capabilities.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from mcp.server.fastmcp import FastMCP

# Import our provider system
from providers.base import PaperProvider, PaperMetadata, SearchResult, DownloadResult
from providers.arxiv_provider import ArXivProvider

# Create the FastMCP server instance
mcp = FastMCP("ArXiv Paper Downloader")

# Global provider instance
_provider: Optional[PaperProvider] = None


def _get_provider() -> PaperProvider:
    """
    Get or initialize the ArXiv provider instance.
    
    Returns:
        Initialized ArXiv provider
    """
    global _provider
    if _provider is None:
        # Get download directory from environment or use default
        download_dir = os.getenv("ARXIV_DOWNLOAD_DIR", "./papers")
        _provider = ArXivProvider(download_dir=Path(download_dir))
    return _provider


@mcp.tool()
async def search_papers(
    query: str,
    max_results: int = 10,
    category: Optional[str] = None,
    sort_by: str = "relevance"
) -> Dict[str, Any]:
    """
    Search ArXiv for academic papers matching the given query.
    
    Args:
        query: Search terms (keywords, author names, titles, etc.)
               Can include field specifiers like 'ti:machine learning' for title search
        max_results: Maximum number of papers to return (1-100, default: 10)
        category: ArXiv category to search within (e.g., 'cs.AI', 'stat.ML')
        sort_by: Sort order - 'relevance', 'lastUpdatedDate', 'submittedDate' (default: 'relevance')
    
    Returns:
        Dictionary containing search results with paper metadata
    
    Examples:
        search_papers("machine learning")
        search_papers("au:Hinton", max_results=20)
        search_papers("neural networks", category="cs.LG")
        search_papers("transformer", sort_by="lastUpdatedDate")
    """
    try:
        provider = _get_provider()
        result = await provider.search(
            query=query,
            max_results=max_results,
            category=category,
            sort_by=sort_by
        )
        
        return {
            "success": True,
            "query": result.query,
            "total_results": result.total_results,
            "returned_results": len(result.papers),
            "papers": [
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract,
                    "categories": paper.categories,
                    "published_date": paper.published_date.isoformat(),
                    "pdf_url": paper.pdf_url,
                    "doi": paper.doi,
                    "journal": paper.journal
                }
                for paper in result.papers
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


@mcp.tool()
async def get_paper_info(arxiv_id: str) -> Dict[str, Any]:
    """
    Get detailed metadata for a specific ArXiv paper.
    
    Args:
        arxiv_id: ArXiv identifier (e.g., '2301.00001', 'cs/0001001')
    
    Returns:
        Dictionary containing detailed paper metadata
    
    Examples:
        get_paper_info("2301.00001")
        get_paper_info("cs/0001001")
    """
    try:
        provider = _get_provider()
        paper = await provider.get_paper_metadata(arxiv_id)
        
        return {
            "success": True,
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": [
                {
                    "name": author.name,
                    "affiliation": author.affiliation
                }
                for author in paper.authors
            ],
            "abstract": paper.abstract,
            "categories": paper.categories,
            "published_date": paper.published_date.isoformat(),
            "updated_date": paper.updated_date.isoformat() if paper.updated_date else None,
            "pdf_url": paper.pdf_url,
            "doi": paper.doi,
            "journal": paper.journal,
            "comment": paper.comment
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "arxiv_id": arxiv_id
        }


@mcp.tool()
async def download_paper(
    arxiv_id: str, 
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download the PDF for an ArXiv paper to local storage.
    
    Args:
        arxiv_id: ArXiv identifier (e.g., '2301.00001', 'cs/0001001')
        filename: Optional custom filename (without .pdf extension)
                 If not provided, uses the ArXiv ID as filename
    
    Returns:
        Dictionary containing download result and file information
    
    Examples:
        download_paper("2301.00001")
        download_paper("2301.00001", filename="attention_is_all_you_need")
    """
    try:
        provider = _get_provider()
        result = await provider.download_paper(arxiv_id, filename)
        
        if result.success:
            return {
                "success": True,
                "arxiv_id": result.arxiv_id,
                "local_path": str(result.local_path),
                "file_size": result.file_size,
                "download_time": result.timestamp.isoformat()
            }
        else:
            return {
                "success": False,
                "error": result.error_message,
                "arxiv_id": result.arxiv_id
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "arxiv_id": arxiv_id
        }


@mcp.tool()
async def list_categories() -> Dict[str, Any]:
    """
    Get a list of available ArXiv subject categories for searching.
    
    Returns:
        Dictionary containing available categories and their descriptions
    
    Examples:
        list_categories()
    """
    try:
        provider = _get_provider()
        categories = await provider.list_categories()
        
        # Category descriptions for common subjects
        category_descriptions = {
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.LG": "Machine Learning",
            "cs.NE": "Neural and Evolutionary Computing",
            "cs.RO": "Robotics",
            "math.CO": "Combinatorics",
            "math.GT": "Geometric Topology",
            "math.NT": "Number Theory",
            "math.OC": "Optimization and Control",
            "math.PR": "Probability",
            "math.ST": "Statistics Theory",
            "physics.app-ph": "Applied Physics",
            "physics.data-an": "Data Analysis, Statistics and Probability",
            "physics.comp-ph": "Computational Physics",
            "stat.ML": "Machine Learning (Statistics)",
            "stat.ME": "Methodology (Statistics)",
            "stat.TH": "Statistics Theory",
            "q-bio.BM": "Biomolecules",
            "q-bio.CB": "Cell Behavior",
            "q-bio.GN": "Genomics",
            "q-bio.MN": "Molecular Networks",
            "econ.EM": "Econometrics",
            "econ.GN": "General Economics",
            "econ.TH": "Theoretical Economics"
        }
        
        return {
            "success": True,
            "categories": [
                {
                    "code": cat,
                    "description": category_descriptions.get(cat, "")
                }
                for cat in categories
            ],
            "total_categories": len(categories)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def echo_test(message: str = "Hello from ArXiv server!") -> Dict[str, str]:
    """
    Simple echo test to verify the server is working correctly.
    
    Args:
        message: Message to echo back (default: "Hello from ArXiv server!")
    
    Returns:
        Dictionary containing the echoed message and server info
    
    Examples:
        echo_test()
        echo_test("Testing ArXiv connection")
    """
    return {
        "echo": message,
        "server": "ArXiv Paper Downloader",
        "status": "operational"
    }


@mcp.resource("arxiv://server-info")
async def get_server_info() -> str:
    """
    Get information about the ArXiv server configuration and status.
    
    Returns:
        JSON string containing server information
    """
    provider = _get_provider()
    
    info = {
        "server_name": "ArXiv Paper Downloader",
        "provider": provider.get_provider_name(),
        "download_directory": str(provider.download_dir),
        "supported_features": [
            "Paper search with advanced queries",
            "Metadata extraction",
            "PDF download and local storage",
            "Category-based filtering",
            "Multiple sort options"
        ],
        "api_endpoints": {
            "search": "https://export.arxiv.org/api/query",
            "pdf_download": "https://arxiv.org/pdf"
        }
    }
    
    return json.dumps(info, indent=2)


@mcp.resource("arxiv://download-status")
async def get_download_status() -> str:
    """
    Get status of the download directory and previously downloaded papers.
    
    Returns:
        JSON string containing download directory status
    """
    provider = _get_provider()
    download_dir = provider.download_dir
    
    # Check if directory exists and get file count
    if download_dir.exists():
        pdf_files = list(download_dir.glob("*.pdf"))
        total_size = sum(f.stat().st_size for f in pdf_files if f.is_file())
        
        status = {
            "download_directory": str(download_dir),
            "directory_exists": True,
            "total_papers": len(pdf_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "recent_downloads": [
                {
                    "filename": f.name,
                    "size_bytes": f.stat().st_size,
                    "modified": f.stat().st_mtime
                }
                for f in sorted(pdf_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            ]
        }
    else:
        status = {
            "download_directory": str(download_dir),
            "directory_exists": False,
            "total_papers": 0,
            "note": "Directory will be created on first download"
        }
    
    return json.dumps(status, indent=2)


@mcp.prompt("paper-analysis")
async def paper_analysis_prompt(
    arxiv_id: str,
    focus: str = "general"
) -> str:
    """
    Generate a structured prompt for analyzing an ArXiv paper.
    
    Args:
        arxiv_id: ArXiv identifier of the paper to analyze
        focus: Analysis focus - 'general', 'methodology', 'results', 'related-work'
    
    Returns:
        Formatted prompt for paper analysis
    """
    # Get paper metadata first
    provider = _get_provider()
    try:
        paper = await provider.get_paper_metadata(arxiv_id)
        
        focus_instructions = {
            "general": "Provide a comprehensive analysis covering methodology, key contributions, and significance.",
            "methodology": "Focus specifically on the methods, algorithms, and experimental design used.",
            "results": "Concentrate on the results, findings, and their interpretation.",
            "related-work": "Analyze how this work relates to and builds upon previous research."
        }
        
        instruction = focus_instructions.get(focus, focus_instructions["general"])
        
        return f"""Please analyze the following ArXiv paper with a focus on {focus}:

**Paper Information:**
- Title: {paper.title}
- Authors: {', '.join([author.name for author in paper.authors])}
- ArXiv ID: {paper.arxiv_id}
- Categories: {', '.join(paper.categories)}
- Published: {paper.published_date.strftime('%Y-%m-%d')}

**Abstract:**
{paper.abstract}

**Analysis Instructions:**
{instruction}

Please provide:
1. **Summary**: A concise summary of the paper's main contribution
2. **Key Points**: 3-5 most important findings or methodological innovations
3. **Strengths**: What makes this work valuable or novel
4. **Limitations**: Any limitations or areas for improvement you identify
5. **Impact**: Potential significance for the field

If you need to reference specific details from the paper, note that the PDF should be downloaded first using the download_paper tool."""
        
    except Exception as e:
        return f"""Error retrieving paper metadata for {arxiv_id}: {str(e)}

Please ensure the ArXiv ID is correct and try again. You can search for papers using the search_papers tool first."""


@mcp.prompt("citation-format")
async def citation_format_prompt(
    arxiv_id: str,
    style: str = "apa"
) -> str:
    """
    Generate properly formatted citations for an ArXiv paper.
    
    Args:
        arxiv_id: ArXiv identifier of the paper to cite
        style: Citation style - 'apa', 'mla', 'chicago', 'ieee', 'bibtex'
    
    Returns:
        Formatted citation in the requested style
    """
    provider = _get_provider()
    try:
        paper = await provider.get_paper_metadata(arxiv_id)
        
        authors_list = [author.name for author in paper.authors]
        year = paper.published_date.year
        
        if style.lower() == "apa":
            authors_apa = ", ".join(authors_list[:3]) + (" et al." if len(authors_list) > 3 else "")
            return f"""{authors_apa} ({year}). {paper.title}. arXiv preprint arXiv:{paper.arxiv_id}."""
            
        elif style.lower() == "mla":
            first_author = authors_list[0] if authors_list else "Unknown"
            others = " et al." if len(authors_list) > 1 else ""
            return f"""{first_author}{others} "{paper.title}." arXiv preprint arXiv:{paper.arxiv_id} ({year})."""
            
        elif style.lower() == "ieee":
            authors_ieee = ", ".join(authors_list[:6]) + (" et al." if len(authors_list) > 6 else "")
            return f"""{authors_ieee}, "{paper.title}," arXiv preprint arXiv:{paper.arxiv_id}, {year}."""
            
        elif style.lower() == "bibtex":
            safe_title = paper.title.replace("{", "").replace("}", "")
            key = paper.arxiv_id.replace("/", "").replace(".", "")
            return f"""@article{{{key},
  title={{{safe_title}}},
  author={{{" and ".join(authors_list)}}},
  journal={{arXiv preprint arXiv:{paper.arxiv_id}}},
  year={{{year}}}
}}"""
        else:
            return f"""Supported citation styles: APA, MLA, IEEE, BibTeX

**Paper:** {paper.title}
**Authors:** {', '.join(authors_list)}
**ArXiv ID:** {paper.arxiv_id}
**Year:** {year}

Please specify one of the supported styles for formatting."""
            
    except Exception as e:
        return f"""Error retrieving paper metadata for {arxiv_id}: {str(e)}

Please ensure the ArXiv ID is correct and try again."""


if __name__ == "__main__":
    mcp.run()