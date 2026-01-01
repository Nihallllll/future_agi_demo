"""Main application entry point for the Gemini RAG system."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
import google.generativeai as genai

from config import get_settings
from src import (
    DocumentProcessor,
    VectorStore,
    RAGEngine,
    QualityEvaluator,
    create_interface
)

# Future AGI imports
try:
    from fi_instrumentation import register, Transport
    from fi_instrumentation.fi_types import ProjectType
    FI_AVAILABLE = True
except ImportError:
    FI_AVAILABLE = False
    print("⚠️  fi_instrumentation not available")


def initialize_system():
    """Initialize all system components."""
    print("\n" + "="*60)
    print("🚀 Initializing Gemini RAG Document Q&A System")
    print("="*60 + "\n")
    
    # Load environment variables
    load_dotenv()
    settings = get_settings()
    
    # Step 1: Configure Google API
    print("📌 Step 1/7: Configuring Google Gemini API...")
    google_api_key = os.getenv("GOOGLE_API_KEY") or settings.google_api_key
    if not google_api_key or google_api_key == "your_google_api_key_here":
        print("❌ ERROR: GOOGLE_API_KEY not found!")
        print("Please set your API key in the .env file")
        print("Get your free key at: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    genai.configure(api_key=google_api_key)
    print("✓ Google Gemini API configured\n")
    
    # Step 2: Configure Future AGI (optional)
    print("📌 Step 2/7: Configuring Future AGI Observability...")
    fi_api_key = os.getenv("FI_API_KEY") or settings.fi_api_key
    fi_secret_key = os.getenv("FI_SECRET_KEY") or settings.fi_secret_key
    trace_provider = None
    
    if fi_api_key and fi_api_key != "your_future_agi_api_key_here" and FI_AVAILABLE:
        os.environ["FI_API_KEY"] = fi_api_key
        if fi_secret_key and fi_secret_key != "your_future_agi_secret_key_here":
            os.environ["FI_SECRET_KEY"] = fi_secret_key
        
        try:
            # Register project with Future AGI platform
            trace_provider = register(
                project_type=ProjectType.OBSERVE,
                project_name=settings.project_name,
                transport=Transport.GRPC  # Use GRPC transport for traces
            )
            print("✓ Future AGI project registered")
            print(f"  Project: {settings.project_name}")
            print(f"  Transport: GRPC")
            print(f"  Dashboard: https://app.futureagi.com\n")
        except Exception as e:
            print(f"⚠️  Could not register with Future AGI: {e}")
            print("  Continuing without observability...\n")
            trace_provider = None
    else:
        print("⚠ Future AGI credentials not found (observability disabled)\n")
    
    # Step 3: Initialize Document Processor
    print("📌 Step 3/7: Initializing Document Processor...")
    doc_processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    print("✓ Document Processor initialized\n")
    
    # Step 4: Initialize Vector Store
    print("📌 Step 4/7: Initializing Vector Store...")
    vector_store = VectorStore(
        collection_name="documents",
        persist_directory=str(settings.chroma_dir),
        embedding_model=settings.embedding_model
    )
    print(f"✓ Vector Store initialized ({vector_store.get_count()} documents loaded)\n")
    
    # Step 5: Load and process documents
    print("📌 Step 5/7: Loading documents...")
    documents = doc_processor.load_documents(settings.data_dir)
    
    if documents:
        print(f"✓ Loaded {len(documents)} documents")
        
        # Only add documents if vector store is empty
        if vector_store.get_count() == 0:
            print("  Processing and indexing documents...")
            chunks = doc_processor.chunk_documents(documents)
            vector_store.add_documents(chunks)
        else:
            print("  Using existing vector store (documents already indexed)")
    else:
        print("⚠ No documents found!")
        print(f"  Please add .txt or .pdf files to: {settings.data_dir}")
        print("  You can run 'python add_sample_docs.py' to create sample documents")
    
    print()
    
    # Step 6: Initialize RAG Engine
    print("📌 Step 6/7: Initializing RAG Engine...")
    rag_engine = RAGEngine(
        vector_store=vector_store,
        generation_model=settings.generation_model,
        trace_provider=trace_provider
    )
    print("✓ RAG Engine initialized\n")
    
    # Step 7: Initialize Evaluator
    print("📌 Step 7/7: Initializing Quality Evaluator...")
    evaluator = QualityEvaluator(
        enable_evaluation=settings.enable_evaluation
    )
    print("✓ Quality Evaluator initialized\n")
    
    print("="*60)
    print("✅ System initialization complete!")
    print("="*60 + "\n")
    
    return rag_engine, evaluator, settings


def main():
    """Main application entry point."""
    try:
        # Initialize system
        rag_engine, evaluator, settings = initialize_system()
        
        # Create and launch interface
        print("🌐 Starting web interface...")
        print(f"📊 Model: {settings.generation_model}")
        print(f"📁 Documents: {settings.data_dir}")
        print(f"💾 Vector DB: {settings.chroma_dir}")
        print()
        
        interface = create_interface(rag_engine, evaluator)
        
        print("="*60)
        print("🎉 Application is running!")
        print("="*60)
        print()
        print("📱 Access the interface at: http://localhost:7860")
        print("🔍 Ask questions about your documents")
        print("📊 View quality evaluations and source citations")
        print()
        print("Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
