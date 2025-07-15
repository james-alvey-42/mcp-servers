"""
Step 2: Simple RAG System Implementation
Using OpenAI embeddings and ChromaDB for local literature corpus
"""
import os
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI
import json
from typing import List, Dict, Any
import pandas as pd

# Import local configuration
from local_config import *

class SimpleRAGSystem:
    """
    A simplified RAG system using OpenAI embeddings and ChromaDB
    """
    
    def __init__(self):
        print("🔧 Initializing Simple RAG System")
        
        # Check OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.chroma_client = None
        self.collection = None
        self.documents_loaded = False
        
    def load_documents(self):
        """Load markdown documents and split them into chunks"""
        print("📚 Loading documents from markdowns...")
        
        markdown_files = list(markdown_files_path.glob("*.md"))
        if not markdown_files:
            print("❌ No markdown files found. Run step1_convert_pdfs.py first.")
            return False
            
        print(f"Found {len(markdown_files)} markdown files")
        
        self.documents = []
        self.chunks = []
        
        for md_file in markdown_files:
            print(f"  Loading {md_file.name}...")
            
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple chunking: split by double newlines and filter short chunks
            raw_chunks = content.split('\n\n')
            file_chunks = []
            
            for i, chunk in enumerate(raw_chunks):
                # Skip very short chunks
                if len(chunk.strip()) < 100:
                    continue
                    
                chunk_data = {
                    'content': chunk.strip(),
                    'source': md_file.name,
                    'chunk_id': f"{md_file.stem}_chunk_{i}",
                    'file_stem': md_file.stem
                }
                file_chunks.append(chunk_data)
                
            self.chunks.extend(file_chunks)
            print(f"    Created {len(file_chunks)} chunks")
            
        print(f"✅ Total chunks created: {len(self.chunks)}")
        self.documents_loaded = True
        return True
    
    def create_embeddings(self):
        """Create embeddings for all chunks using OpenAI"""
        if not self.documents_loaded:
            print("❌ Documents not loaded. Run load_documents() first.")
            return False
            
        print("🧮 Creating embeddings with OpenAI...")
        
        # Extract text content
        texts = [chunk['content'] for chunk in self.chunks]
        
        print(f"Creating embeddings for {len(texts)} chunks...")
        
        # Create embeddings in batches
        batch_size = 50  # Smaller batches to avoid rate limits
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=OpenAI_Embedding_Model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                import time
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error creating embeddings for batch {i//batch_size + 1}: {e}")
                return False
        
        # Store embeddings in chunks
        for chunk, embedding in zip(self.chunks, embeddings):
            chunk['embedding'] = embedding
            
        print("✅ Embeddings created successfully")
        return True
    
    def setup_chromadb(self):
        """Setup ChromaDB collection"""
        print("🗄️ Setting up ChromaDB...")
        
        # Create ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(embeddings_path / "chromadb"))
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(CHROMA_COLLECTION_NAME)
            print(f"  ✅ Loaded existing collection: {CHROMA_COLLECTION_NAME}")
        except:
            print(f"  Creating new collection: {CHROMA_COLLECTION_NAME}")
            self.collection = self.chroma_client.create_collection(CHROMA_COLLECTION_NAME)
        
        return True
    
    def store_in_chromadb(self):
        """Store chunks and embeddings in ChromaDB"""
        if not self.collection:
            print("❌ ChromaDB not set up. Run setup_chromadb() first.")
            return False
            
        if not self.chunks or 'embedding' not in self.chunks[0]:
            print("❌ Chunks or embeddings not ready.")
            return False
            
        print("💾 Storing documents in ChromaDB...")
        
        # Check if collection already has documents
        count = self.collection.count()
        if count > 0:
            print(f"  Collection already has {count} documents. Clearing...")
            self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(CHROMA_COLLECTION_NAME)
        
        # Prepare data for ChromaDB
        ids = [chunk['chunk_id'] for chunk in self.chunks]
        documents = [chunk['content'] for chunk in self.chunks]
        embeddings = [chunk['embedding'] for chunk in self.chunks]
        metadatas = [
            {
                'source': chunk['source'],
                'file_stem': chunk['file_stem']
            }
            for chunk in self.chunks
        ]
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            print(f"  Stored batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
        
        print(f"✅ Stored {len(ids)} documents in ChromaDB")
        return True
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.collection:
            print("❌ ChromaDB not set up.")
            return None
            
        print(f"🔍 Querying: {question}")
        
        # Create embedding for the question
        try:
            response = self.client.embeddings.create(
                input=[question],
                model=OpenAI_Embedding_Model
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"❌ Error creating query embedding: {e}")
            return None
        
        # Search ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
        except Exception as e:
            print(f"❌ Error querying ChromaDB: {e}")
            return None
        
        # Format context from retrieved documents
        context = ""
        sources = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context += f"\n\n--- Source {i+1}: {metadata['source']} ---\n{doc}"
            sources.append(metadata['source'])
        
        # Generate response using OpenAI
        try:
            prompt = f"""Based on the following context from research papers, answer the question. 
Be specific and cite the sources when possible.

Context:
{context}

Question: {question}

Answer:"""
            
            chat_response = self.client.chat.completions.create(
                model=OPENAI_GEN_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant. Answer questions based on the provided context from research papers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=500
            )
            
            answer = chat_response.choices[0].message.content
            
            result = {
                'question': question,
                'answer': answer,
                'sources': list(set(sources)),  # Remove duplicates
                'context_snippets': results['documents'][0]
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return None
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 Running Complete RAG Setup")
        print("=" * 50)
        
        steps = [
            ("Load Documents", self.load_documents),
            ("Create Embeddings", self.create_embeddings),
            ("Setup ChromaDB", self.setup_chromadb),
            ("Store in ChromaDB", self.store_in_chromadb),
        ]
        
        for step_name, step_func in steps:
            print(f"\n--- {step_name} ---")
            success = step_func()
            if not success:
                print(f"❌ Setup failed at: {step_name}")
                return False
                
        print("\n🎉 RAG System Setup Complete!")
        return True

def main():
    """Main function to demonstrate the RAG system"""
    rag = SimpleRAGSystem()
    
    # Run setup
    if not rag.run_setup():
        print("❌ Setup failed")
        return None
    
    # Test queries
    test_questions = [
        "What is simulation-based inference?",
        "What are the main challenges in cosmological simulations?",
        "What methods are used for galaxy clustering analysis?",
        "What is the role of machine learning in these papers?",
        "What are the key findings about robustness?",
    ]
    
    print("\n🧪 Testing RAG System")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test Query {i}: {question}")
        print("-" * 40)
        
        result = rag.query(question)
        if result:
            print(f"💬 Answer: {result['answer']}")
            print(f"📄 Sources: {', '.join(result['sources'])}")
        
        print()
    
    return rag

if __name__ == "__main__":
    rag_system = main()