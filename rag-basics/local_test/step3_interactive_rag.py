"""
Step 3: Interactive RAG Interface
Interactive query interface for your literature corpus
"""
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Import the simple RAG system
from step2_simple_rag import SimpleRAGSystem
from local_config import datasets_path, DATASET

class InteractiveRAG:
    """Interactive interface for the RAG system"""
    
    def __init__(self):
        print("🚀 Loading Interactive RAG System...")
        self.rag = SimpleRAGSystem()
        
        # Try to load existing ChromaDB collection
        if not self.rag.setup_chromadb():
            print("❌ Failed to setup ChromaDB")
            return
            
        # Check if collection has data
        if self.rag.collection.count() == 0:
            print("📚 No data found in ChromaDB. Setting up...")
            if not self.rag.run_setup():
                print("❌ Failed to setup RAG system")
                return
        else:
            print(f"✅ Loaded existing collection with {self.rag.collection.count()} documents")
        
        self.query_history = []
        
    def query_interactive(self):
        """Interactive query loop"""
        print("\n🔍 Interactive RAG Query Interface")
        print("=" * 50)
        print("Type your questions about the literature corpus.")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the interface")
        print("  'history' - Show query history")
        print("  'eval' - Create evaluation dataset from queries")
        print("  'papers' - List available papers")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n📝 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                    
                elif question.lower() == 'history':
                    self.show_history()
                    continue
                    
                elif question.lower() == 'eval':
                    self.create_evaluation_dataset()
                    continue
                    
                elif question.lower() == 'papers':
                    self.list_papers()
                    continue
                    
                elif not question:
                    print("❓ Please enter a question.")
                    continue
                
                # Process the query
                print(f"\n🔍 Searching for: {question}")
                result = self.rag.query(question, n_results=5)
                
                if result:
                    print(f"\n💬 Answer:")
                    print(f"{result['answer']}")
                    print(f"\n📄 Sources: {', '.join(result['sources'])}")
                    
                    # Save to history
                    self.query_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'question': question,
                        'answer': result['answer'],
                        'sources': result['sources']
                    })
                    
                else:
                    print("❌ Sorry, I couldn't process your question.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_history(self):
        """Show query history"""
        if not self.query_history:
            print("📊 No queries in history yet.")
            return
            
        print(f"\n📊 Query History ({len(self.query_history)} queries)")
        print("-" * 50)
        
        for i, query in enumerate(self.query_history[-5:], 1):  # Show last 5
            print(f"{i}. {query['question']}")
            print(f"   Sources: {', '.join(query['sources'])}")
            print()
    
    def list_papers(self):
        """List available papers in the corpus"""
        # Get unique sources from collection
        try:
            results = self.rag.collection.get()
            sources = set()
            for metadata in results['metadatas']:
                sources.add(metadata['source'])
            
            print(f"\n📚 Available Papers ({len(sources)} papers)")
            print("-" * 50)
            for i, source in enumerate(sorted(sources), 1):
                paper_id = source.replace('.md', '')
                print(f"{i}. {paper_id}")
                
        except Exception as e:
            print(f"❌ Error listing papers: {e}")
    
    def create_evaluation_dataset(self):
        """Create evaluation dataset from query history"""
        if len(self.query_history) < 3:
            print("❓ Need at least 3 queries to create evaluation dataset.")
            print("Ask more questions first!")
            return
            
        print(f"\n📊 Creating evaluation dataset from {len(self.query_history)} queries...")
        
        # Create evaluation data structure
        eval_data = {
            'question': [],
            'ideal': [],
            'author': [],
            'source_file': [],
            'doi': [],
            'location': [],
            'key_passage': []
        }
        
        for query in self.query_history:
            eval_data['question'].append(query['question'])
            eval_data['ideal'].append(query['answer'])
            eval_data['author'].append('Multiple Authors')
            eval_data['source_file'].append(', '.join(query['sources']))
            eval_data['doi'].append('ArXiv Papers')
            eval_data['location'].append('Various Sections')
            eval_data['key_passage'].append('Generated from RAG query')
        
        # Create DataFrame and save
        df = pd.DataFrame(eval_data)
        dataset_file = datasets_path / DATASET
        df.to_parquet(dataset_file, index=False)
        
        print(f"✅ Evaluation dataset created: {dataset_file}")
        print(f"   Contains {len(df)} question-answer pairs")
        
        # Also save as JSON for easy viewing
        json_file = datasets_path / f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(self.query_history, f, indent=2)
        
        print(f"✅ Query history saved: {json_file}")

def predefined_demo():
    """Run a demonstration with predefined queries"""
    print("🎯 Running Predefined Demo")
    print("=" * 50)
    
    rag = SimpleRAGSystem()
    
    # Setup if needed
    if not rag.setup_chromadb():
        return
        
    if rag.collection.count() == 0:
        print("📚 Setting up RAG system...")
        if not rag.run_setup():
            return
    else:
        print(f"✅ Using existing collection with {rag.collection.count()} documents")
    
    # Predefined interesting questions
    demo_questions = [
        "What are the main differences between IllustrisTNG and SIMBA simulations?",
        "How does simulation-based inference compare to traditional methods?",
        "What are the key challenges in multifidelity simulation methods?",
        "What role does machine learning play in cosmological parameter estimation?",
        "What evidence is there for model misspecification in cosmological simulations?",
        "How do researchers handle out-of-distribution data in their analyses?",
        "What improvements in parameter constraints are achieved by these methods?",
    ]
    
    results = []
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📝 Demo Query {i}: {question}")
        print("-" * 60)
        
        result = rag.query(question, n_results=3)
        if result:
            print(f"💬 Answer: {result['answer'][:300]}...")
            print(f"📄 Sources: {', '.join(result['sources'])}")
            results.append(result)
        
        print()
    
    print(f"\n🎉 Demo completed! Processed {len(results)} queries successfully.")
    return results

def main():
    """Main function"""
    print("🔬 Local Literature RAG System")
    print("Choose an option:")
    print("1. Interactive mode")
    print("2. Predefined demo")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            interactive_rag = InteractiveRAG()
            interactive_rag.query_interactive()
        elif choice == "2":
            predefined_demo()
        else:
            print("❓ Invalid choice. Running demo...")
            predefined_demo()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()