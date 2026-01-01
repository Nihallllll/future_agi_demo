"""Gradio web interface for the RAG system."""

import gradio as gr
from typing import Tuple, Dict, Any


def create_interface(rag_engine, evaluator) -> gr.Blocks:
    """
    Create a Gradio interface for the RAG system.
    
    Args:
        rag_engine: RAGEngine instance
        evaluator: QualityEvaluator instance
        
    Returns:
        Gradio Blocks interface
    """
    
    def process_question(question: str, top_k: int) -> Tuple[str, str, str, str]:
        """Process a question and return formatted results."""
        if not question.strip():
            return "Please enter a question.", "", "", ""
        
        # Generate answer
        result = rag_engine.generate_answer(
            question=question,
            top_k=top_k,
            include_sources=True
        )
        
        answer = result["answer"]
        sources = result.get("sources", [])
        token_usage = result.get("token_usage", {})
        
        # Format sources
        sources_text = "## Retrieved Sources\n\n"
        context_for_eval = ""
        
        if sources:
            for i, source in enumerate(sources, 1):
                sources_text += f"**Source {i}** (Score: {source['score']})\n"
                sources_text += f"*File: {source['source']}*\n"
                sources_text += f"{source['text']}\n\n"
                context_for_eval += source['text'] + "\n\n"
        else:
            sources_text += "No sources found.\n"
        
        # Evaluate response
        evaluation_text = "## Quality Evaluation\n\n"
        
        if evaluator.enable_evaluation and answer and "Error" not in answer:
            eval_result = evaluator.evaluate_response(
                question=question,
                answer=answer,
                context=context_for_eval
            )
            
            if eval_result.get("enabled"):
                if "results" in eval_result:
                    for metric, data in eval_result["results"].items():
                        evaluation_text += f"**{metric.title()}:** {data['severity']}\n"
                        if data.get('score') is not None:
                            evaluation_text += f"  - Score: {data['score']:.2f}\n"
                    
                    evaluation_text += f"\n**Summary:** {eval_result.get('summary', 'N/A')}\n"
                elif "error" in eval_result:
                    evaluation_text += f"⚠ Evaluation error: {eval_result['error']}\n"
            else:
                evaluation_text += "Evaluation is disabled.\n"
        else:
            evaluation_text += "No evaluation performed.\n"
        
        # Format metadata
        metadata_text = "## Request Metadata\n\n"
        metadata_text += f"**Model:** {result['model']}\n"
        metadata_text += f"**Context Chunks Used:** {result['context_used']}\n"
        
        if token_usage:
            metadata_text += f"**Tokens Used:** {token_usage.get('total_tokens', 'N/A')}\n"
            metadata_text += f"  - Prompt: {token_usage.get('prompt_tokens', 'N/A')}\n"
            metadata_text += f"  - Completion: {token_usage.get('completion_tokens', 'N/A')}\n"
            
            # Calculate cost
            cost_info = rag_engine.calculate_cost(token_usage)
            metadata_text += f"**Cost:** ${cost_info['total_cost']:.4f}\n"
            metadata_text += f"  - {cost_info['note']}\n"
        
        return answer, sources_text, evaluation_text, metadata_text
    
    # Create the interface
    with gr.Blocks(title="Gemini RAG Document Q&A", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 Gemini RAG Document Q&A System
        
        Ask questions about your documents. The system uses **Google Gemini 2.0 Flash** with automatic 
        quality evaluation and observability.
        
        ### Features:
        - 🔍 Semantic search across your documents
        - 🤖 AI-powered answers with source citations
        - ✅ Automatic quality & safety evaluation
        - 📊 Full observability with TraceAI
        - 💰 **FREE** (Gemini 2.0 Flash within rate limits)
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know?",
                    lines=3
                )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of context chunks to retrieve"
                    )
                
                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("📝 Answer"):
                        answer_output = gr.Markdown(label="Answer")
                    
                    with gr.Tab("📚 Sources"):
                        sources_output = gr.Markdown(label="Sources")
                    
                    with gr.Tab("✅ Evaluation"):
                        evaluation_output = gr.Markdown(label="Quality Evaluation")
                    
                    with gr.Tab("📊 Metadata"):
                        metadata_output = gr.Markdown(label="Metadata")
        
        # Set up the event handler
        submit_btn.click(
            fn=process_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, sources_output, evaluation_output, metadata_output]
        )
        
        # Also trigger on Enter key
        question_input.submit(
            fn=process_question,
            inputs=[question_input, top_k_slider],
            outputs=[answer_output, sources_output, evaluation_output, metadata_output]
        )
        
        gr.Markdown("""
        ---
        ### 💡 Tips:
        - Ask specific questions for better results
        - Check the Sources tab to see where information came from
        - Review the Evaluation tab for quality metrics
        - Adjust the number of context chunks if needed
        
        **Powered by:** Google Gemini 2.0 Flash | TraceAI | Future AGI Evaluation
        """)
    
    return interface
