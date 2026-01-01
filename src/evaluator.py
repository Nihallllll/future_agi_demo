"""Quality evaluation module using Future AGI SDK."""

from typing import Dict, Any, List

# Try to import Evaluation from Future AGI SDK
try:
    from futureagi import FutureAGI
    EVALUATION_AVAILABLE = True
except ImportError:
    try:
        from fi.client import Client as FutureAGI
        EVALUATION_AVAILABLE = True
    except ImportError:
        FutureAGI = None
        EVALUATION_AVAILABLE = False


class QualityEvaluator:
    """Evaluates RAG responses for quality, hallucinations, and safety."""
    
    def __init__(self, enable_evaluation: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            enable_evaluation: Whether to enable evaluation
        """
        self.enable_evaluation = enable_evaluation and EVALUATION_AVAILABLE
        self.evaluator = None
        
        if self.enable_evaluation:
            try:
                # Initialize Future AGI client for evaluations
                # For now, evaluation is integrated via the UI, not the SDK
                # The SDK's evaluation features are accessed through the web platform
                print("INFO: Future AGI SDK available - evaluation via web platform")
                print("  Visit https://app.futureagi.com for evaluation features")
                # Disable for now since Evaluation class is not directly available
                self.enable_evaluation = False
            except Exception as e:
                print(f"Warning: Could not initialize evaluator: {e}")
                self.enable_evaluation = False
        
        if not EVALUATION_AVAILABLE:
            print("INFO: Future AGI evaluation SDK not fully configured")
    
    def evaluate_response(
        self,
        question: str,
        answer: str,
        context: str = "",
        templates: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG response for quality and safety.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Context used for generation
            templates: List of evaluation templates to use
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.enable_evaluation:
            return {
                "enabled": False,
                "message": "Evaluation is disabled"
            }
        
        if templates is None:
            templates = ["hallucination", "relevance", "toxicity", "tone"]
        
        results = {}
        
        try:
            # Hallucination check
            if "hallucination" in templates and context:
                hallucination = self.evaluator.evaluate(
                    template="hallucination",
                    input={"context": context, "response": answer}
                )
                results["hallucination"] = self._format_result(hallucination)
            
            # Relevance check
            if "relevance" in templates:
                relevance = self.evaluator.evaluate(
                    template="relevance",
                    input={"query": question, "response": answer}
                )
                results["relevance"] = self._format_result(relevance)
            
            # Toxicity check
            if "toxicity" in templates:
                toxicity = self.evaluator.evaluate(
                    template="toxicity",
                    input={"response": answer}
                )
                results["toxicity"] = self._format_result(toxicity)
            
            # Tone check
            if "tone" in templates:
                tone = self.evaluator.evaluate(
                    template="tone",
                    input={"response": answer}
                )
                results["tone"] = self._format_result(tone)
            
            return {
                "enabled": True,
                "results": results,
                "summary": self._create_summary(results)
            }
            
        except Exception as e:
            return {
                "enabled": True,
                "error": str(e),
                "message": "Evaluation failed"
            }
    
    def _format_result(self, result: Any) -> Dict[str, Any]:
        """Format an evaluation result."""
        if hasattr(result, 'score'):
            return {
                "score": result.score,
                "severity": self._get_severity(result.score),
                "details": str(result)
            }
        return {
            "score": None,
            "severity": "unknown",
            "details": str(result)
        }
    
    def _get_severity(self, score: float) -> str:
        """Get severity level from score."""
        if score >= 0.8:
            return "✓ Good"
        elif score >= 0.6:
            return "⚠ Moderate"
        else:
            return "✗ Poor"
    
    def _create_summary(self, results: Dict[str, Any]) -> str:
        """Create a summary of evaluation results."""
        summary_parts = []
        
        for metric, data in results.items():
            if isinstance(data, dict) and "severity" in data:
                summary_parts.append(f"{metric.title()}: {data['severity']}")
        
        return " | ".join(summary_parts) if summary_parts else "No evaluation data"
