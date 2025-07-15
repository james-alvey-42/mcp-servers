"""
Local configuration for your custom literature RAG system
Based on the SciRag framework
"""
import os
from pathlib import Path
from openai import OpenAI

# Define paths relative to this file
LOCAL_TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = LOCAL_TEST_DIR.parent  # Points to main scirag directory

# Local project paths
markdown_files_path = LOCAL_TEST_DIR / "markdowns"
datasets_path = LOCAL_TEST_DIR / "datasets"
embeddings_path = LOCAL_TEST_DIR / "embeddings"
txt_files_path = LOCAL_TEST_DIR / "txt_files"
results_path = LOCAL_TEST_DIR / "results"
arxiv_pdfs_path = LOCAL_TEST_DIR / "arxiv_pdfs"

# Dataset configuration
DATASET = "local_literature_corpus.parquet"

# Document processing parameters
CHUNK_SIZE = 3000  # Smaller chunks for better precision
CHUNK_OVERLAP = 300  # Higher overlap for better context

# Retrieval parameters
TOP_K = 15  # Slightly fewer chunks for more focused results
DISTANCE_THRESHOLD = 0.4  # Lower threshold for more relevant results

# Model configuration
OPENAI_GEN_MODEL = "gpt-4o-mini"  # Cost-effective model
TEMPERATURE = 0.1  # Low temperature for consistent scientific responses

# OpenAI client setup (assumes API key is in environment)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
OpenAI_Embedding_Model = "text-embedding-3-large"

# ChromaDB configuration
CHROMA_COLLECTION_NAME = "local_literature_collection"
CHROMA_DB_PATH = str(embeddings_path / "chromadb_local")

# Display names
corpus_name = "local_literature_corpus"
assistant_name = "local_rag_agent"

# Pricing configuration (for cost tracking)
OAI_PRICE1K = {
    "gpt-4o-mini": (0.000150, 0.000600),  # (input, output) per 1K tokens
    "gpt-4o": (0.005, 0.015),
    "text-embedding-3-large": 0.00013,  # per 1K tokens
}

print(f"Local test directory: {LOCAL_TEST_DIR}")
print(f"Markdown files will be stored in: {markdown_files_path}")
print(f"Embeddings will be stored in: {embeddings_path}")