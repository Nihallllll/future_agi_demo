"""Helper script to create sample documents for testing the RAG system."""

from pathlib import Path


def create_sample_documents():
    """Create sample documents in the data/documents directory."""
    
    # Create directory if it doesn't exist
    docs_dir = Path(__file__).parent / "data" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample Document 1: AI Basics
    ai_basics = """Artificial Intelligence (AI) Overview

Artificial Intelligence is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.

Machine Learning
Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning from labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through trial and error

Deep Learning
Deep Learning is a subset of machine learning based on artificial neural networks. It's particularly effective for:
- Image recognition
- Natural language processing
- Speech recognition
- Autonomous vehicles

AI Applications
Modern AI applications include:
- Virtual assistants (Siri, Alexa)
- Recommendation systems (Netflix, Amazon)
- Autonomous vehicles
- Medical diagnosis
- Financial trading
- Fraud detection

The future of AI holds tremendous potential for solving complex problems and improving human life.
"""
    
    # Sample Document 2: Python Programming
    python_guide = """Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python has become one of the most popular programming languages.

Key Features:
- Easy to learn and read
- Interpreted language
- Dynamically typed
- Object-oriented
- Extensive standard library

Python Syntax Basics
Python uses indentation to define code blocks. Variables don't need to be declared with a specific type.

Example:
def greet(name):
    return f"Hello, {name}!"

Data Types
Python supports various data types:
- int: Integer numbers
- float: Decimal numbers
- str: Text strings
- list: Ordered collections
- dict: Key-value pairs
- tuple: Immutable sequences
- set: Unordered unique elements

Popular Python Libraries:
- NumPy: Numerical computing
- Pandas: Data manipulation
- Matplotlib: Data visualization
- TensorFlow: Machine learning
- Django: Web development
- Flask: Lightweight web framework

Python is widely used in:
- Web development
- Data science and analytics
- Artificial intelligence
- Scientific computing
- Automation and scripting
- Game development

The simplicity and versatility of Python make it an excellent choice for both beginners and experienced programmers.
"""
    
    # Sample Document 3: RAG Systems
    rag_systems = """Retrieval-Augmented Generation (RAG) Systems

RAG is a technique that enhances Large Language Models (LLMs) by retrieving relevant information from external knowledge bases before generating responses.

How RAG Works:
1. Document Ingestion: Load and process documents
2. Chunking: Split documents into manageable pieces
3. Embedding: Convert text into vector representations
4. Storage: Store embeddings in a vector database
5. Retrieval: Find relevant chunks for a query
6. Generation: Use LLM to generate answer with context

Components of a RAG System:

Vector Database
A specialized database that stores and retrieves vector embeddings efficiently. Popular options include:
- ChromaDB
- Pinecone
- Weaviate
- Qdrant
- FAISS

Embedding Models
Models that convert text into numerical vectors:
- OpenAI Embeddings
- Google Gemini Embeddings
- Sentence Transformers
- Cohere Embeddings

Language Models
LLMs used for generation:
- GPT-4
- Claude
- Google Gemini
- Llama
- Mistral

Benefits of RAG:
- Reduces hallucinations
- Provides source attribution
- Enables up-to-date information
- Domain-specific knowledge
- Cost-effective compared to fine-tuning

Best Practices:
1. Use appropriate chunk sizes (300-800 tokens)
2. Include chunk overlap (10-20%)
3. Implement hybrid search (semantic + keyword)
4. Add metadata for filtering
5. Monitor and evaluate quality
6. Implement caching for efficiency

RAG is particularly useful for:
- Question answering systems
- Customer support chatbots
- Document analysis
- Knowledge management
- Research assistants

The combination of retrieval and generation makes RAG a powerful technique for building reliable AI applications.
"""
    
    # Write documents
    files = {
        "ai_basics.txt": ai_basics,
        "python_guide.txt": python_guide,
        "rag_systems.txt": rag_systems
    }
    
    print("\n" + "="*60)
    print("📝 Creating Sample Documents")
    print("="*60 + "\n")
    
    for filename, content in files.items():
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"✓ Created: {filename}")
    
    print(f"\n✅ Successfully created {len(files)} sample documents")
    print(f"📁 Location: {docs_dir}")
    print("\nYou can now run 'python main.py' to start the RAG system!")
    print("="*60 + "\n")


if __name__ == "__main__":
    create_sample_documents()
