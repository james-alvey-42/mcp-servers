"""
Local RAG Implementation Example
Step-by-step implementation using SciRag framework
"""
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import scirag modules
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import local configuration
from local_config import *

# Import SciRag components
from scirag import SciRagOpenAI, SciRagDataSet
from scirag.ocr import MistralOCRProcessor

class LocalRAGSystem:
    """
    Custom RAG system for your literature corpus
    """
    
    def __init__(self):
        """Initialize the local RAG system"""
        print("🚀 Initializing Local RAG System")
        print(f"Working directory: {LOCAL_TEST_DIR}")
        
        # Check if OpenAI API key is available
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print("✅ OpenAI API key found")
        
        self.rag_system = None
        self.documents_processed = False
        
    def step1_convert_pdfs_to_markdown(self):
        """
        Step 1: Convert PDF files to markdown using OCR
        """
        print("\n📄 Step 1: Converting PDFs to Markdown")
        
        # Check if PDFs exist
        pdf_files = list(arxiv_pdfs_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        
        if not pdf_files:
            print("❌ No PDF files found!")
            return False
            
        # For this example, we'll use a simple text extraction approach
        # The original SciRag uses Mistral OCR, but we'll implement a simpler version
        try:
            import PyPDF2
            print("Using PyPDF2 for text extraction...")
            
            for pdf_file in pdf_files:
                print(f"Processing {pdf_file.name}...")
                
                # Extract text from PDF
                text_content = self._extract_text_from_pdf(pdf_file)
                
                # Save as markdown
                markdown_file = markdown_files_path / f"{pdf_file.stem}.md"
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {pdf_file.stem}\n\n")
                    f.write(text_content)
                
                print(f"  ✅ Converted to {markdown_file.name}")
                
        except ImportError:
            print("⚠️  PyPDF2 not installed. Installing...")
            os.system("pip install PyPDF2")
            return self.step1_convert_pdfs_to_markdown()
            
        print(f"✅ Successfully converted {len(pdf_files)} PDFs to markdown")
        return True
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyPDF2"""
        import PyPDF2
        
        text_content = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += f"\n\n## Page {page_num + 1}\n\n"
                    text_content += page.extract_text()
                    
        except Exception as e:
            print(f"  ⚠️  Error extracting text from {pdf_path.name}: {e}")
            text_content = f"Error extracting text: {e}"
            
        return text_content
    
    def step2_initialize_rag_system(self):
        """
        Step 2: Initialize RAG system with ChromaDB
        """
        print("\n🔧 Step 2: Initializing RAG System")
        
        try:
            self.rag_system = SciRagOpenAI(
                client=openai_client,
                markdown_files_path=markdown_files_path,
                corpus_name=corpus_name,
                gen_model=OPENAI_GEN_MODEL,
                vector_db_backend="chromadb",
                chroma_collection_name=CHROMA_COLLECTION_NAME,
                chroma_db_path=CHROMA_DB_PATH,
            )
            print("✅ RAG system initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            return False
    
    def step3_process_documents(self):
        """
        Step 3: Load and process documents
        """
        print("\n📚 Step 3: Processing Documents")
        
        if not self.rag_system:
            print("❌ RAG system not initialized")
            return False
            
        try:
            # Check if markdown files exist
            markdown_files = list(markdown_files_path.glob("*.md"))
            print(f"Found {len(markdown_files)} markdown files")
            
            if not markdown_files:
                print("❌ No markdown files found. Run step1 first.")
                return False
            
            # Note: The SciRagOpenAI with ChromaDB backend automatically
            # loads documents and creates embeddings in its __init__ method
            print("✅ Documents processed and embeddings created")
            self.documents_processed = True
            return True
            
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            return False
    
    def step4_test_queries(self):
        """
        Step 4: Test the RAG system with sample queries
        """
        print("\n🔍 Step 4: Testing RAG System")
        
        if not self.rag_system or not self.documents_processed:
            print("❌ System not ready. Complete previous steps first.")
            return False
            
        # Sample queries to test the system
        test_queries = [
            "What are the main research topics in these papers?",
            "What methodologies are used in these studies?",
            "What are the key findings or conclusions?",
            "What datasets or experiments are mentioned?",
        ]
        
        print("Running test queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            try:
                response = self.rag_system.query(query)
                print(f"💬 Response: {response}")
                print("-" * 80)
                
            except Exception as e:
                print(f"❌ Error with query {i}: {e}")
        
        return True
    
    def step5_create_sample_evaluation(self):
        """
        Step 5: Create a sample evaluation dataset
        """
        print("\n📊 Step 5: Creating Sample Evaluation Dataset")
        
        # Create a simple evaluation dataset based on your PDFs
        import pandas as pd
        
        # Sample evaluation data
        eval_data = {
            'question': [
                'What research methods are discussed in these papers?',
                'What are the main contributions of these studies?',
                'What future work is suggested?'
            ],
            'ideal': [
                'The papers discuss various research methodologies including experimental studies, theoretical analysis, and computational approaches.',
                'The main contributions include novel algorithms, improved performance metrics, and theoretical insights.',
                'Future work suggestions include expanding datasets, improving algorithms, and conducting additional experiments.'
            ],
            'author': ['Multiple Authors'] * 3,
            'source_file': ['Combined ArXiv Papers'] * 3,
            'doi': ['Various DOIs'] * 3,
            'location': ['Multiple Sections'] * 3,
            'key_passage': [
                'Research methodology sections across papers',
                'Contribution and results sections',
                'Future work and conclusion sections'
            ]
        }
        
        df = pd.DataFrame(eval_data)
        dataset_file = datasets_path / DATASET
        df.to_parquet(dataset_file, index=False)
        
        print(f"✅ Sample evaluation dataset created: {dataset_file}")
        return True
    
    def run_complete_pipeline(self):
        """
        Run the complete RAG setup pipeline
        """
        print("🚀 Starting Complete RAG Pipeline")
        print("=" * 60)
        
        steps = [
            ("Convert PDFs to Markdown", self.step1_convert_pdfs_to_markdown),
            ("Initialize RAG System", self.step2_initialize_rag_system),
            ("Process Documents", self.step3_process_documents),
            ("Test Queries", self.step4_test_queries),
            ("Create Evaluation Dataset", self.step5_create_sample_evaluation),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            success = step_func()
            if not success:
                print(f"❌ Pipeline stopped at: {step_name}")
                return False
                
        print("\n🎉 Pipeline completed successfully!")
        print("You can now use your RAG system to query your literature corpus.")
        return True

def main():
    """Main function to run the example"""
    try:
        # Create the local RAG system
        local_rag = LocalRAGSystem()
        
        # Run the complete pipeline
        success = local_rag.run_complete_pipeline()
        
        if success:
            print("\n🎯 Next Steps:")
            print("1. Use local_rag.rag_system.query('your question') to ask questions")
            print("2. Explore the created files in the local_test directory")
            print("3. Modify the configuration in local_config.py as needed")
            
        return local_rag
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None

if __name__ == "__main__":
    # Run the implementation
    rag_system = main()