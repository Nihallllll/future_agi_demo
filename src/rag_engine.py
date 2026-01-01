"""RAG engine for question answering with observability."""

from typing import List, Dict, Any, Optional
import google.generativeai as genai
from traceai_google_genai import GoogleGenAIInstrumentor


class RAGEngine:
    """Retrieval-Augmented Generation engine with TraceAI instrumentation."""
    
    def __init__(
        self,
        vector_store,
        generation_model: str = "gemini-2.0-flash-exp",
        trace_provider = None
    ):
        """
        Initialize the RAG engine.
        
        Args:
            vector_store: VectorStore instance
            generation_model: Gemini model to use for generation
            trace_provider: OpenTelemetry trace provider from Future AGI register()
        """
        self.vector_store = vector_store
        self.generation_model = generation_model
        self.model = genai.GenerativeModel(generation_model)
        
        # Initialize TraceAI instrumentation with trace provider
        if trace_provider is not None:
            try:
                self.instrumentor = GoogleGenAIInstrumentor()
                self.instrumentor.instrument(tracer_provider=trace_provider)
                print("✓ TraceAI instrumentation enabled (linked to Future AGI)")
            except Exception as e:
                print(f"Warning: Could not enable instrumentation: {e}")
    
    def generate_answer(
        self,
        question: str,
        top_k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer to a question using RAG.
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant context
        search_results = self.vector_store.search(query=question, top_k=top_k)
        
        if not search_results:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": [],
                "context_used": 0,
                "model": self.generation_model
            }
        
        # Format context
        context = self._format_context(search_results)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        # Generate answer
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": usage.prompt_token_count,
                    "completion_tokens": usage.candidates_token_count,
                    "total_tokens": usage.total_token_count
                }
            
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            token_usage = {}
        
        # Format sources
        sources = []
        if include_sources:
            sources = [
                {
                    "text": result["text"][:200] + "...",
                    "source": result["metadata"].get("source", "Unknown"),
                    "score": round(result["score"], 3)
                }
                for result in search_results
            ]
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(search_results),
            "model": self.generation_model,
            "token_usage": token_usage
        }
    
    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context string."""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result["metadata"].get("filename", "Unknown")
            text = result["text"]
            context_parts.append(f"[Source {i}: {source}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create the prompt for the LLM."""
        return f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer: Please provide a clear, concise answer based on the context above. If you reference specific information, mention which source it comes from."""

    def calculate_cost(self, token_usage: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate the cost of a request.
        
        Note: Gemini 2.0 Flash is FREE up to rate limits.
        
        Args:
            token_usage: Dictionary with token counts
            
        Returns:
            Dictionary with cost information
        """
        # Gemini 2.0 Flash pricing (FREE tier)
        # Input: $0 per 1M tokens
        # Output: $0 per 1M tokens
        
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        return {
            "prompt_cost": 0.0,
            "completion_cost": 0.0,
            "total_cost": 0.0,
            "note": "Gemini 2.0 Flash is FREE (within rate limits)"
        }
