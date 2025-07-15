"""
Step 1: Convert PDFs to Markdown
Simple PDF to text conversion for local testing
"""
import sys
from pathlib import Path
import PyPDF2

# Import local configuration
from local_config import arxiv_pdfs_path, markdown_files_path

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2"""
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

def convert_pdfs_to_markdown():
    """
    Convert PDF files to markdown format
    """
    print("📄 Step 1: Converting PDFs to Markdown")
    print(f"Source directory: {arxiv_pdfs_path}")
    print(f"Output directory: {markdown_files_path}")
    
    # Check if PDFs exist
    pdf_files = list(arxiv_pdfs_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    if not pdf_files:
        print("❌ No PDF files found!")
        return False
    
    # Convert each PDF
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file.name}...")
        
        # Extract text from PDF
        text_content = extract_text_from_pdf(pdf_file)
        
        # Save as markdown
        markdown_file = markdown_files_path / f"{pdf_file.stem}.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(f"# {pdf_file.stem}\n\n")
            f.write(f"Source: {pdf_file.name}\n\n")
            f.write(text_content)
        
        print(f"  ✅ Converted to {markdown_file.name}")
        
        # Show first 200 characters as preview
        preview = text_content.replace('\n', ' ')[:200]
        print(f"  📝 Preview: {preview}...")
    
    print(f"\n✅ Successfully converted {len(pdf_files)} PDFs to markdown")
    return True

if __name__ == "__main__":
    convert_pdfs_to_markdown()