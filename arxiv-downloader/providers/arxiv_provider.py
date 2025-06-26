"""
ArXiv provider implementation for paper search and download.

This module implements the ArXiv API interface for searching and downloading academic papers.
It uses the ArXiv RSS/Atom API for search and direct HTTP requests for PDF downloads.
"""

import re
import httpx
import feedparser
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from .base import (
    PaperProvider, 
    PaperMetadata, 
    PaperAuthor, 
    SearchResult, 
    DownloadResult
)


class ArXivProvider(PaperProvider):
    """
    ArXiv provider for searching and downloading academic papers.
    
    Uses the ArXiv API (https://export.arxiv.org/api/) for search operations
    and direct HTTPS requests for PDF downloads.
    """
    
    BASE_URL = "https://export.arxiv.org/api/query"
    PDF_BASE_URL = "https://arxiv.org/pdf"
    
    # ArXiv subject categories
    CATEGORIES = [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO",  # Computer Science
        "math.CO", "math.GT", "math.NT", "math.OC", "math.PR", "math.ST",  # Mathematics
        "physics.app-ph", "physics.data-an", "physics.comp-ph",  # Physics
        "stat.ML", "stat.ME", "stat.TH",  # Statistics
        "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN",  # Quantitative Biology
        "econ.EM", "econ.GN", "econ.TH",  # Economics
        "q-fin.CP", "q-fin.EC", "q-fin.MF", "q-fin.PM", "q-fin.RM", "q-fin.ST", "q-fin.TR"  # Quantitative Finance
    ]
    
    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize the ArXiv provider.
        
        Args:
            download_dir: Directory where PDFs should be downloaded
        """
        super().__init__(download_dir)
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    
    async def search(
        self, 
        query: str,
        max_results: int = 10,
        category: Optional[str] = None,
        sort_by: str = "relevance",
        **kwargs
    ) -> SearchResult:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search terms (can include keywords, authors, titles)
            max_results: Maximum number of papers to return (max 100)
            category: ArXiv category to search within (e.g., 'cs.AI')
            sort_by: Sort order - 'relevance', 'lastUpdatedDate', 'submittedDate'
            
        Returns:
            SearchResult with matching papers
        """
        # Construct the search query
        search_query = self._build_search_query(query, category)
        
        # Map sort_by to ArXiv API format
        sort_order = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate"
        }.get(sort_by, "relevance")
        
        # Build API request parameters
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 100),  # ArXiv limits to 100 per request
            "sortBy": sort_order,
            "sortOrder": "descending"
        }
        
        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            # Parse the Atom feed response
            feed = feedparser.parse(response.text)
            
            papers = []
            for entry in feed.entries:
                paper = self._parse_entry(entry)
                papers.append(paper)
            
            return SearchResult(
                papers=papers,
                total_results=int(getattr(feed.feed, 'opensearch_totalresults', len(papers))),
                query=query
            )
            
        except Exception as e:
            raise Exception(f"ArXiv search failed: {str(e)}")
    
    async def get_paper_metadata(self, arxiv_id: str) -> PaperMetadata:
        """
        Get detailed metadata for a specific ArXiv paper.
        
        Args:
            arxiv_id: ArXiv identifier (e.g., '2301.00001' or 'cs/0001001')
            
        Returns:
            PaperMetadata with full paper information
        """
        # Normalize ArXiv ID
        normalized_id = self._normalize_arxiv_id(arxiv_id)
        
        params = {
            "id_list": normalized_id,
            "max_results": 1
        }
        
        try:
            response = await self.client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            feed = feedparser.parse(response.text)
            
            if not feed.entries:
                raise Exception(f"Paper not found: {arxiv_id}")
            
            return self._parse_entry(feed.entries[0])
            
        except Exception as e:
            raise Exception(f"Failed to get paper metadata: {str(e)}")
    
    async def download_paper(self, arxiv_id: str, filename: Optional[str] = None) -> DownloadResult:
        """
        Download the PDF for an ArXiv paper.
        
        Args:
            arxiv_id: ArXiv identifier
            filename: Optional custom filename (without extension)
            
        Returns:
            DownloadResult with download status and file information
        """
        normalized_id = self._normalize_arxiv_id(arxiv_id)
        pdf_url = f"{self.PDF_BASE_URL}/{normalized_id}.pdf"
        local_path = self.get_download_path(arxiv_id, filename)
        
        try:
            # Download the PDF
            response = await self.client.get(pdf_url, follow_redirects=True)
            response.raise_for_status()
            
            # Ensure the download directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the PDF to disk
            with open(local_path, "wb") as f:
                f.write(response.content)
            
            return DownloadResult(
                success=True,
                local_path=local_path,
                file_size=len(response.content),
                arxiv_id=arxiv_id
            )
            
        except Exception as e:
            return DownloadResult(
                success=False,
                arxiv_id=arxiv_id,
                error_message=f"Download failed: {str(e)}"
            )
    
    async def list_categories(self) -> List[str]:
        """
        Get a list of ArXiv subject categories.
        
        Returns:
            List of category codes
        """
        return self.CATEGORIES.copy()
    
    def _build_search_query(self, query: str, category: Optional[str] = None) -> str:
        """
        Build a search query string for the ArXiv API.
        
        Args:
            query: User search terms
            category: Optional category filter
            
        Returns:
            Formatted search query string
        """
        # If the query looks like it contains field specifiers, use it as-is
        if any(field in query.lower() for field in ['ti:', 'au:', 'abs:', 'cat:']):
            search_query = query
        else:
            # Default to searching in title, abstract, and comments
            search_query = f"all:{quote(query)}"
        
        # Add category filter if specified
        if category and category in self.CATEGORIES:
            search_query = f"cat:{category} AND ({search_query})"
        
        return search_query
    
    def _parse_entry(self, entry) -> PaperMetadata:
        """
        Parse a feedparser entry into a PaperMetadata object.
        
        Args:
            entry: feedparser entry object
            
        Returns:
            PaperMetadata object
        """
        # Extract ArXiv ID from the ID URL
        arxiv_id = entry.id.split('/')[-1].replace('v1', '').replace('v2', '').replace('v3', '')
        if 'v' in arxiv_id:
            arxiv_id = arxiv_id.split('v')[0]
        
        # Parse authors
        authors = []
        for author in entry.authors:
            authors.append(PaperAuthor(name=author.name))
        
        # Parse categories
        categories = []
        if hasattr(entry, 'tags'):
            categories = [tag.term for tag in entry.tags]
        elif hasattr(entry, 'arxiv_primary_category'):
            categories = [entry.arxiv_primary_category['term']]
        
        # Parse dates
        published_date = datetime(*entry.published_parsed[:6])
        updated_date = None
        if hasattr(entry, 'updated_parsed'):
            updated_date = datetime(*entry.updated_parsed[:6])
        
        # Extract DOI if present
        doi = None
        if hasattr(entry, 'arxiv_doi'):
            doi = entry.arxiv_doi
        
        # Extract journal reference if present
        journal = None
        if hasattr(entry, 'arxiv_journal_ref'):
            journal = entry.arxiv_journal_ref
        
        # Extract comment if present
        comment = None
        if hasattr(entry, 'arxiv_comment'):
            comment = entry.arxiv_comment
        
        # Build PDF URL
        pdf_url = f"{self.PDF_BASE_URL}/{arxiv_id}.pdf"
        
        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=entry.title.strip(),
            authors=authors,
            abstract=entry.summary.strip(),
            categories=categories,
            published_date=published_date,
            updated_date=updated_date,
            doi=doi,
            journal=journal,
            pdf_url=pdf_url,
            comment=comment
        )
    
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """
        Normalize an ArXiv ID to the standard format.
        
        Args:
            arxiv_id: Raw ArXiv ID from user input
            
        Returns:
            Normalized ArXiv ID
        """
        # Remove any version numbers (v1, v2, etc.)
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # Handle old-style IDs (e.g., cs/0001001)
        if '/' in arxiv_id:
            return arxiv_id
        
        # Handle new-style IDs (e.g., 2301.00001)
        if re.match(r'^\d{4}\.\d{4,5}$', arxiv_id):
            return arxiv_id
        
        # If it doesn't match expected patterns, return as-is and let ArXiv handle it
        return arxiv_id
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()