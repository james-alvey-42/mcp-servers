"""
Base classes for ArXiv paper providers.

This module defines the abstract interface for academic paper search and download providers.
By using a common interface, we can potentially support multiple academic databases in the future.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field


class PaperAuthor(BaseModel):
    """Represents a paper author with name and optional affiliation."""
    name: str = Field(description="Author's full name")
    affiliation: Optional[str] = Field(default=None, description="Author's institutional affiliation")


class PaperMetadata(BaseModel):
    """
    Standardized metadata for an academic paper.
    
    This provides a consistent structure regardless of the source database.
    """
    arxiv_id: str = Field(description="ArXiv identifier (e.g., '2301.00001')")
    title: str = Field(description="Paper title")
    authors: List[PaperAuthor] = Field(description="List of paper authors")
    abstract: str = Field(description="Paper abstract")
    categories: List[str] = Field(description="ArXiv subject categories")
    published_date: datetime = Field(description="Date the paper was published")
    updated_date: Optional[datetime] = Field(default=None, description="Date the paper was last updated")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    journal: Optional[str] = Field(default=None, description="Journal or conference name")
    pdf_url: str = Field(description="URL to download the PDF")
    comment: Optional[str] = Field(default=None, description="Additional comments or notes")


class SearchResult(BaseModel):
    """
    Results from a paper search query.
    """
    papers: List[PaperMetadata] = Field(description="List of papers matching the search")
    total_results: int = Field(description="Total number of results available")
    query: str = Field(description="The original search query")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this search was performed")


class DownloadResult(BaseModel):
    """
    Result of downloading a paper PDF.
    """
    success: bool = Field(description="Whether the download was successful")
    local_path: Optional[Path] = Field(default=None, description="Local file path where PDF was saved")
    file_size: Optional[int] = Field(default=None, description="Size of downloaded file in bytes")
    arxiv_id: str = Field(description="ArXiv ID of the downloaded paper")
    error_message: Optional[str] = Field(default=None, description="Error message if download failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this download was performed")


class PaperProvider(ABC):
    """
    Abstract base class for academic paper providers.
    
    Each provider (ArXiv, PubMed, etc.) must implement these methods.
    This ensures consistent interfaces for paper search and download operations.
    """
    
    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize the provider.
        
        Args:
            download_dir: Directory where PDFs should be downloaded. Defaults to ./papers/
        """
        self.download_dir = download_dir or Path("./papers")
        self.download_dir.mkdir(exist_ok=True)
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    async def search(
        self, 
        query: str,
        max_results: int = 10,
        category: Optional[str] = None,
        **kwargs
    ) -> SearchResult:
        """
        Search for papers matching the given query.
        
        Args:
            query: Search terms (keywords, author names, etc.)
            max_results: Maximum number of papers to return
            category: Specific subject category to search within
            **kwargs: Provider-specific search parameters
            
        Returns:
            SearchResult containing matching papers and metadata
            
        Raises:
            Exception: If the search fails
        """
        pass
    
    @abstractmethod
    async def get_paper_metadata(self, arxiv_id: str) -> PaperMetadata:
        """
        Get detailed metadata for a specific paper.
        
        Args:
            arxiv_id: The ArXiv identifier for the paper
            
        Returns:
            PaperMetadata with full paper information
            
        Raises:
            Exception: If the paper cannot be found or retrieved
        """
        pass
    
    @abstractmethod
    async def download_paper(self, arxiv_id: str, filename: Optional[str] = None) -> DownloadResult:
        """
        Download the PDF for a specific paper.
        
        Args:
            arxiv_id: The ArXiv identifier for the paper
            filename: Optional custom filename (without extension)
            
        Returns:
            DownloadResult with download status and file information
            
        Raises:
            Exception: If the download fails
        """
        pass
    
    @abstractmethod
    async def list_categories(self) -> List[str]:
        """
        Get a list of available subject categories.
        
        Returns:
            List of category codes that can be used for searching
        """
        pass
    
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.
        
        Returns:
            Provider name (e.g., "arxiv", "pubmed")
        """
        return self.provider_name
    
    def get_download_path(self, arxiv_id: str, filename: Optional[str] = None) -> Path:
        """
        Generate the local file path for a downloaded paper.
        
        Args:
            arxiv_id: The ArXiv identifier
            filename: Optional custom filename
            
        Returns:
            Path where the file should be saved
        """
        if filename:
            return self.download_dir / f"{filename}.pdf"
        else:
            # Use ArXiv ID as filename, replacing problematic characters
            safe_id = arxiv_id.replace("/", "_").replace(":", "_")
            return self.download_dir / f"{safe_id}.pdf"