# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SciRag is a scientific RAG (Retrieval-Augmented Generation) system for astronomy papers, designed to evaluate different RAG approaches including OpenAI, Gemini, VertexAI, and PaperQA2 implementations. The project is structured as a research comparison tool for ML4ASTRO ICML Workshop.

## Development Commands

### Python Package Installation
```bash
pip install -e .
```

### Running Notebooks
Navigate to the `notebooks/` directory and run Jupyter notebooks:
- `scirag_openai.ipynb` - OpenAI-based RAG implementation
- `scirag_gemini.ipynb` - Gemini-based RAG implementation  
- `scirag_vertex.ipynb` - VertexAI-based RAG implementation
- `scirag_paperqa2.ipynb` - PaperQA2-based RAG implementation
- `eval.ipynb` - Evaluation framework
- `human_eval.ipynb` - Human evaluation interface
- `ai_eval.ipynb` - AI-based evaluation system

### Data Processing
- OCR processing: `notebooks/scirag_ocr.ipynb`
- Dataset handling: Use `SciRagDataSet` class from `scirag.dataset`

## Architecture

### Core Components

1. **Base RAG Class (`scirag/scirag.py`)**
   - Abstract base class for all RAG implementations
   - Handles document chunking, embedding, and retrieval
   - Uses Google Drive and GCS integration for document management

2. **Implementation Classes**
   - `SciRagOpenAI` - OpenAI GPT-4 with ChromaDB or OpenAI vector store
   - `SciRagVertexAI` - Google VertexAI with Vertex RAG store
   - `SciRagGemini` - Gemini with grounded generation
   - `SciRagPaperQA2` - PaperQA2 framework integration
   - `SciRagHybrid` - Hybrid approach combining multiple methods

3. **Evaluation Framework (`scirag/scirag_evaluator.py`)**
   - `AIEvaluator` - AutoGen-based AI evaluation system
   - `GeminiEvaluator` - Gemini-based evaluation
   - `SingleRAGEvaluationSystem` - Comprehensive evaluation pipeline
   - Binary accuracy scoring (0 or 100) for response evaluation

4. **Configuration (`scirag/config.py`)**
   - Central configuration for all models, embeddings, and parameters
   - Pricing information for cost analysis (`OAI_PRICE1K`, `GEMINI_PRICE1K`)
   - Google Cloud and authentication setup
   - Paper processing settings for PaperQA2

### Data Flow

1. **Document Processing**: PDF → OCR → Markdown → Text chunks
2. **Embedding Generation**: Text chunks → Vector embeddings (OpenAI/Gemini)
3. **Storage**: ChromaDB, OpenAI Vector Store, or Vertex RAG Store
4. **Retrieval**: Semantic search with configurable TOP_K and DISTANCE_THRESHOLD
5. **Generation**: LLM response with structured output format (`AnswerFormat`)

### Key Configuration Parameters

- `TOP_K = 20` - Number of retrieved documents
- `DISTANCE_THRESHOLD = 0.5` - Similarity threshold for retrieval
- `CHUNK_SIZE = 5000` - Document chunk size
- `CHUNK_OVERLAP = 250` - Overlap between chunks
- `TEMPERATURE = 0.01` - Low temperature for consistent scientific responses

## Dataset Structure

- `datasets/CosmoPaperQA.parquet` - Main evaluation dataset
- `markdowns/` - Processed paper content in markdown format
- `embeddings/` - Precomputed embeddings for different models
- `results/` - Evaluation results and performance metrics

## Authentication Requirements

The system requires several API keys and credentials:
- OpenAI API key (`OPENAI_API_KEY`)
- Google Cloud credentials for VertexAI and Gemini
- Google Drive API credentials for document management
- Perplexity API key for comparative evaluation

## Evaluation Metrics

The evaluation system uses binary accuracy scoring:
- 100: Answer contains correct factual content matching ideal answer
- 0: Answer is fundamentally wrong or contradicts ideal answer

Results are stored in `rag_evaluation_results/` with timestamped CSV files for each model and approach.

## File Structure Notes

- Git merge conflicts present in `__init__.py` (lines 9-18) - resolve before development
- Jupyter notebooks contain the main experimental workflows
- Python modules in `scirag/` package provide reusable components
- OCR output stored in `txt_files/` for PaperQA2 processing