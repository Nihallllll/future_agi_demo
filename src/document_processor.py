"""Document processing module for loading and chunking documents."""

import os
from pathlib import Path
from typing import List, Dict, Any
import tiktoken


class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def load_documents(self, directory: str | Path) -> List[Dict[str, Any]]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to the documents directory
            
        Returns:
            List of document dictionaries with text and metadata
        """
        directory = Path(directory)
        documents = []
        
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist")
            return documents
        
        for file_path in directory.glob("**/*"):
            if file_path.is_file() and file_path.suffix in ['.txt', '.pdf']:
                try:
                    if file_path.suffix == '.txt':
                        text = self._load_text_file(file_path)
                    elif file_path.suffix == '.pdf':
                        text = self._load_pdf_file(file_path)
                    else:
                        continue
                    
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": str(file_path),
                            "filename": file_path.name,
                        }
                    })
                    print(f"✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"✗ Error loading {file_path.name}: {str(e)}")
        
        return documents
    
    def _load_text_file(self, file_path: Path) -> str:
        """Load a text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_pdf_file(self, file_path: Path) -> str:
        """Load a PDF file."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            print("pypdf not installed. Install with: pip install pypdf")
            return ""
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks based on token count.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        
        for doc in documents:
            text = doc["text"]
            tokens = self.tokenizer.encode(text)
            
            # Split into chunks
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": len(chunks),
                        "chunk_size": len(chunk_tokens),
                    }
                })
        
        print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
