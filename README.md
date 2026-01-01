# 🤖 Gemini RAG Document Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** system built with **Google Gemini 2.0 Flash**, featuring automatic quality evaluation and full observability.

## ✨ Features

- 🚀 **FREE Google Gemini 2.0 Flash** - High-quality LLM at no cost (within rate limits)
- 📊 **TraceAI Observability** - Automatic instrumentation for all LLM calls
- ✅ **Future AGI Evaluation** - Automated quality checks (hallucination, relevance, toxicity, tone)
- 🔍 **Semantic Search** - ChromaDB vector database with Gemini embeddings
- 🌐 **Gradio Web Interface** - Beautiful, easy-to-use UI
- 📚 **Multi-format Support** - TXT, PDF document processing
- 💰 **Cost Tracking** - Real-time token usage and cost monitoring
- 🎯 **Source Citations** - Answers include references to source documents

## 🚀 Quick Start

### 1. Clone or Download

```bash
cd Future_agi_RAG
```

### 2. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required: Get your FREE Gemini API key
GOOGLE_API_KEY=your_google_api_key_here

# Optional: For evaluation features
FI_API_KEY=your_future_agi_api_key_here
FI_SECRET_KEY=your_future_agi_secret_key_here
```

**Get API Keys:**
- **Google Gemini** (FREE): https://makersuite.google.com/app/apikey
- **Future AGI** (optional): https://futureagi.com

### 4. Create Sample Documents

```bash
python add_sample_docs.py
```

This creates 3 sample documents about AI, Python, and RAG systems.

### 5. Run the Application

```bash
python main.py
```

Access the interface at: **http://localhost:7860**

## 📁 Project Structure

```
Future_agi_RAG/
├── main.py                 # Application entry point
├── config.py               # Configuration management
├── add_sample_docs.py      # Sample document generator
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
├── .env.example            # Environment template
├── .env                    # Your API keys (create this)
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # Document loading & chunking
│   ├── vector_store.py         # ChromaDB vector database
│   ├── rag_engine.py           # RAG orchestration
│   ├── evaluator.py            # Quality evaluation
│   └── ui.py                   # Gradio interface
├── data/
│   └── documents/              # Your documents go here
│       ├── ai_basics.txt
│       ├── python_guide.txt
│       └── rag_systems.txt
└── chroma_db/                  # Vector database (auto-created)
```

## 🔧 How It Works

1. **Document Processing**: Load TXT/PDF files and split into chunks (500 tokens with 100 overlap)
2. **Embedding**: Convert chunks to vectors using Gemini text-embedding-004
3. **Storage**: Store vectors in ChromaDB for efficient retrieval
4. **Retrieval**: Find top-k most relevant chunks for user questions
5. **Generation**: Use Gemini 2.0 Flash to generate answers with context
6. **Evaluation**: Automatically assess quality (hallucination, relevance, toxicity, tone)

## 💡 Usage Examples

### Ask Questions

```
Question: What is machine learning?
→ System retrieves relevant chunks from documents
→ Gemini generates answer with source citations
→ Quality evaluation runs automatically
→ Shows token usage and sources
```

### Adjust Retrieval

Use the slider to control how many context chunks to retrieve (1-10).
- **More chunks**: Better context, higher token usage
- **Fewer chunks**: Faster, lower cost, but might miss information

## 📊 Evaluation Metrics

The system automatically evaluates every response:

- **Hallucination**: Does the answer stay grounded in context?
- **Relevance**: Does it answer the question?
- **Toxicity**: Is the language safe and appropriate?
- **Tone**: Is the tone professional and helpful?

Each metric gets a score and severity rating (Good/Moderate/Poor).

## 💰 Cost Information

**Gemini 2.0 Flash is FREE** within rate limits:
- **Input**: FREE (up to rate limits)
- **Output**: FREE (up to rate limits)
- **Rate Limits**: 15 RPM, 1 million TPM, 1,500 RPD

Perfect for development and testing!

## 🛠️ Advanced Configuration

Edit `config.py` to customize:

```python
# Model settings
generation_model = "gemini-2.0-flash-exp"
embedding_model = "models/text-embedding-004"

# RAG parameters
chunk_size = 500          # Tokens per chunk
chunk_overlap = 100       # Overlap between chunks
top_k_results = 5         # Default context chunks

# Evaluation
enable_evaluation = True
evaluation_templates = ["hallucination", "relevance", "toxicity", "tone"]
```

## 📝 Adding Your Own Documents

1. Place TXT or PDF files in `data/documents/`
2. Restart the application
3. Documents will be automatically processed and indexed

## 🐛 Troubleshooting

**"GOOGLE_API_KEY not found"**
- Create a `.env` file (copy from `.env.example`)
- Add your Gemini API key

**"No documents found"**
- Run `python add_sample_docs.py` to create samples
- Or add your own files to `data/documents/`

**Python not found**
- Make sure Python 3.12+ is installed
- Or use `uv` which manages Python automatically

**Import errors**
- Install dependencies: `uv sync` or `pip install -r requirements.txt`

## 🤝 Support

- **Gemini Docs**: https://ai.google.dev/docs
- **Future AGI**: https://futureagi.com
- **ChromaDB**: https://docs.trychroma.com

## 📄 License

This project is open source and available under the MIT License.

## 🌟 Features Coming Soon

- [ ] Multi-language support
- [ ] Advanced filtering and search
- [ ] Conversation history
- [ ] Document management UI
- [ ] Export capabilities
- [ ] API endpoint

---

**Built with:** Google Gemini 2.0 Flash | TraceAI | Future AGI | ChromaDB | Gradio

**Happy RAG-ing! 🚀**
