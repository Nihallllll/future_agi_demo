"""Vector store module for semantic search using ChromaDB."""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
import google.generativeai as genai


class VectorStore:
    """Manages vector embeddings and semantic search using ChromaDB."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "models/text-embedding-004"
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Gemini embedding model to use
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for text using Gemini.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = []
        for i, text in enumerate(documents):
            if i % 10 == 0:
                print(f"  Embedding chunk {i+1}/{len(documents)}...")
            embeddings.append(self.embed_text(text))
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching documents with metadata and scores
        """
        # Generate query embedding
        query_embedding = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def reset(self) -> None:
        """Reset the vector store (delete all data)."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Reset collection: {self.collection_name}")
    
    def get_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
