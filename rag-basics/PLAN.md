### Plan: Building Your Own Literature RAG System with SciRag

  A) Dataset Format Requirements

  1. Primary Dataset Structure

  The library expects a parquet file with the following columns:
  - question: Evaluation questions about your literature
  - ideal: Expected/ground truth answers
  - author: Paper authors
  - source file: ArXiv URLs or DOI links
  - doi: Digital Object Identifiers
  - Location: Page/section references
  - key passage: Relevant text excerpts

  2. Document Format

  - Input: Markdown files (.md) containing your literature
  - Location: markdowns/ directory
  - Processing: The system uses LangChain's TextLoader with UTF-8 encoding
  - Metadata: Each document gets enriched with source file, file type, and filename

  3. Alternative Formats Supported

  - PDF files (via OCR processing using scirag_ocr.py)
  - Text files (.txt format)
  - The library can handle various academic paper formats

  B) How to Configure the Library for Your Corpus

  1. Environment Setup

  # Required API keys
  export OPENAI_API_KEY="your-openai-key"
  export GOOGLE_APPLICATION_CREDENTIALS="path/to/google-creds.json"

  2. Core Configuration (scirag/config.py)

  # Update these paths for your corpus
  REPO_DIR = Path("your/literature/directory")
  markdown_files_path = REPO_DIR / "markdowns"
  datasets_path = REPO_DIR / "datasets"
  embeddings_path = REPO_DIR / "embeddings"

  # Your custom dataset name
  DATASET = "YourLiteratureCorpus.parquet"

  # Chunking parameters (adjust based on your documents)
  CHUNK_SIZE = 5000  # Character count
  CHUNK_OVERLAP = 250  # Overlap between chunks

  # Retrieval parameters
  TOP_K = 20  # Number of chunks to retrieve
  DISTANCE_THRESHOLD = 0.5  # Similarity threshold

  3. Choose Your Implementation

  # Option 1: OpenAI with ChromaDB (Recommended for local use)
  from scirag import SciRagOpenAI
  rag = SciRagOpenAI(
      vector_db_backend="chromadb",
      chroma_collection_name="your_literature_collection",
      chroma_db_path="path/to/your/chromadb"
  )

  # Option 2: Pure OpenAI Vector Store
  rag = SciRagOpenAI(vector_db_backend="openai")

  # Option 3: Google Vertex AI (if using Google Cloud)
  from scirag import SciRagVertexAI
  rag = SciRagVertexAI()

  C) Best Practices for Implementation

  1. Document Preparation Workflow

  # Step 1: Convert PDFs to Markdown
  from scirag import MistralOCRProcessor
  ocr = MistralOCRProcessor()
  ocr.process_pdf_directory("your/pdfs/", "markdowns/")

  # Step 2: Initialize your RAG system
  from scirag import SciRagOpenAI
  rag = SciRagOpenAI(
      markdown_files_path=Path("markdowns/"),
      corpus_name="your_literature_corpus",
      vector_db_backend="chromadb"
  )

  # Step 3: Process documents and create embeddings
  rag.load_markdown_files()
  rag.split_documents()
  rag._create_chromadb_embeddings()

  2. Optimal Chunking Strategy

  # For academic papers, these settings work well:
  CHUNK_SIZE = 3000  # Shorter chunks for precise answers
  CHUNK_OVERLAP = 300  # Higher overlap for context preservation

  # For longer documents:
  CHUNK_SIZE = 5000  # Standard setting
  CHUNK_OVERLAP = 250  # Standard overlap

  3. Cost-Effective Approach

  # Use OpenAI with ChromaDB for local storage
  # Avoids OpenAI vector store costs
  rag = SciRagOpenAI(
      vector_db_backend="chromadb",
      gen_model="gpt-4o-mini",  # Cheaper model
      chroma_db_path="local_chromadb"
  )

  4. Query and Evaluation

  # Query your corpus
  response = rag.query("What are the main findings about X?")

  # Evaluate with your dataset
  from scirag import SciRagDataSet
  dataset = SciRagDataSet("YourLiteratureCorpus.parquet")
  evaluation_data = dataset.load_dataset()

  # Run evaluation
  from scirag import SingleRAGEvaluationSystem
  evaluator = SingleRAGEvaluationSystem(rag)
  results = evaluator.evaluate(evaluation_data)

  5. Performance Optimization

  # Adjust retrieval parameters
  TOP_K = 10  # Fewer chunks for faster processing
  DISTANCE_THRESHOLD = 0.3  # Lower threshold for more relevant results

  # Use batch processing for large corpora
  batch_size = 100
  embeddings = rag._create_embeddings_batch(texts, batch_size)

  6. Directory Structure

  your_literature_project/
  ├── markdowns/           # Your converted literature files
  ├── datasets/           # Evaluation datasets
  ├── embeddings/         # Vector database storage
  ├── txt_files/          # OCR output (if using PDFs)
  ├── results/            # Evaluation results
  └── config.py           # Custom configuration

  7. Integration Example

  # Complete setup for your corpus
  import os
  from pathlib import Path
  from scirag import SciRagOpenAI, SciRagDataSet

  # Set up paths
  project_root = Path("your_literature_project")
  markdown_path = project_root / "markdowns"
  dataset_path = project_root / "datasets"

  # Initialize RAG system
  rag = SciRagOpenAI(
      markdown_files_path=markdown_path,
      corpus_name="your_literature",
      vector_db_backend="chromadb",
      chroma_collection_name="your_collection"
  )

  # Load and process your literature
  print("Loading documents...")
  rag.load_markdown_files()
  rag.split_documents()

  # Create embeddings (one-time setup)
  print("Creating embeddings...")
  rag._create_chromadb_embeddings()

  # Query your corpus
  response = rag.query("What are the key methodologies in field X?")
  print(response)

  8. Advanced Features

  - Hybrid Search: Combine semantic and keyword search
  - Cost Tracking: Monitor API usage and costs
  - Evaluation Framework: Systematic assessment of answer quality
  - Multi-modal Support: Handle images and tables in papers

  This plan provides a comprehensive approach to adapting the SciRag library for your specific
  literature corpus while maintaining the robust evaluation and cost-tracking capabilities of the
  original system.