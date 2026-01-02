"""
Future AGI + Gemini Integration - Complete Code Examples
Author: Code Examples for Testing
Date: 2026-01-02

This module demonstrates how to use Future AGI features with Google Gemini API
Each example is standalone and can be tested independently
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
import time
from dotenv import load_dotenv
load_dotenv()
# Install requirements:
# pip install google-generativeai futureagi python-dotenv

try:
    import google.generativeai as genai
except ImportError:
    print("Install: pip install google-generativeai")

try:
    from futureagi import Client
except ImportError:
    print("Install: pip install futureagi")


# ==================== SETUP ====================
def setup_clients():
    """Initialize Future AGI and Gemini clients"""
    
    # Get API keys from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    fi_api_key = os.getenv("FI_API_KEY")
    fi_secret_key = os.getenv("FI_SECRET_KEY")
    
    if not all([gemini_api_key, fi_api_key, fi_secret_key]):
        raise ValueError(
            "Missing API keys. Set: GEMINI_API_KEY, FI_API_KEY, FI_SECRET_KEY"
        )
    
    # Initialize Gemini
    genai.configure(api_key=gemini_api_key)
    
    # Initialize Future AGI (if available)
    try:
        fi_client = Client(
            api_key=fi_api_key,
            secret_key=fi_secret_key
        )
    except:
        fi_client = None
        print("Future AGI client not available - using Gemini only")
    
    return genai, fi_client


# ==================== EXAMPLE 1: TRACING ====================
class TracingExample:
    """
    EXAMPLE 1: TRACING - Monitor LLM calls, latency, and costs
    
    What it does:
    - Logs every API call (generation, tool use, evaluation)
    - Measures latency for each operation
    - Tracks token usage and cost
    - Exports data to Future AGI dashboard
    
    Setup required:
    - GEMINI_API_KEY
    - FI_API_KEY, FI_SECRET_KEY
    
    Output differences:
    - WITHOUT tracing: No visibility into execution
    - WITH tracing: Full observability (latency, tokens, cost)
    """
    
    def __init__(self):
        self.traces = []
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def traced_call(self, prompt: str, operation_name: str = "gemini_call"):
        """
        Make a Gemini call with tracing
        
        Logic:
        1. Record start time
        2. Call Gemini API
        3. Record end time (latency)
        4. Extract token usage
        5. Calculate cost (rough estimate)
        6. Store trace data
        """
        
        trace_data = {
            "operation": operation_name,
            "timestamp": datetime.now().isoformat(),
            "start_time": time.time()
        }

        try:
            # Make API call
            response = self.model.generate_content(prompt)
            trace_data["end_time"] = time.time()
            trace_data["latency_ms"] = (trace_data["end_time"] - trace_data["start_time"]) * 1000
            
            # Extract token info
            if hasattr(response, 'usage_metadata'):
                trace_data["input_tokens"] = response.usage_metadata.prompt_token_count
                trace_data["output_tokens"] = response.usage_metadata.candidates_token_count
                trace_data["total_tokens"] = (
                    response.usage_metadata.prompt_token_count +
                    response.usage_metadata.candidates_token_count
                )
            
            # Rough cost estimation (adjust pricing as needed)
            # Gemini 2.0 Flash: ~$0.075/1M input tokens, ~$0.3/1M output tokens
            trace_data["estimated_cost_usd"] = (
                (trace_data.get("input_tokens", 0) * 0.075) / 1_000_000 +
                (trace_data.get("output_tokens", 0) * 0.3) / 1_000_000
            )
            
            trace_data["status"] = "success"
            trace_data["response_preview"] = str(response.text)[:100]
            
            return response, trace_data
            
        except Exception as e:
            trace_data["end_time"] = time.time()
            trace_data["latency_ms"] = (trace_data["end_time"] - trace_data["start_time"]) * 1000
            trace_data["status"] = "error"
            trace_data["error"] = str(e)
            raise
        
        finally:
            self.traces.append(trace_data)
    
    def untraced_call(self, prompt: str):
        """Make a Gemini call WITHOUT tracing for comparison"""
        response = self.model.generate_content(prompt)
        return response
    
    def compare_outputs(self):
        """
        Show the difference between traced and untraced calls
        
        WITH TRACING:
        - Latency: 250ms (visible)
        - Input tokens: 15 (visible)
        - Output tokens: 87 (visible)
        - Cost: $0.00015 (visible)
        - Full audit trail
        
        WITHOUT TRACING:
        - Latency: Unknown
        - Token usage: Unknown
        - Cost: Unknown
        - No debugging capability
        """
        
        print("\n" + "="*60)
        print("TRACING COMPARISON")
        print("="*60)
        
        print("\n[WITH TRACING - Full Visibility]")
        for trace in self.traces:
            print(f"  Operation: {trace['operation']}")
            print(f"  Status: {trace['status']}")
            print(f"  Latency: {trace['latency_ms']:.2f}ms")
            print(f"  Input Tokens: {trace.get('input_tokens', 'N/A')}")
            print(f"  Output Tokens: {trace.get('output_tokens', 'N/A')}")
            print(f"  Estimated Cost: ${trace.get('estimated_cost_usd', 0):.6f}")
        
        print("\n[WITHOUT TRACING - No Visibility]")
        print("  Operation: gemini_call")
        print("  Status: Unknown")
        print("  Latency: Unknown")
        print("  Input Tokens: Unknown")
        print("  Output Tokens: Unknown")
        print("  Estimated Cost: Unknown")
        print("  → Cannot debug, monitor, or optimize!")
        
        return self.traces
    
    def export_traces(self, filename="traces.json"):
        """Export traces to JSON for analysis or Future AGI dashboard"""
        with open(filename, 'w') as f:
            json.dump(self.traces, f, indent=2)
        print(f"\nTraces exported to {filename}")
        return self.traces


# ==================== EXAMPLE 2: DATASET MANAGEMENT ====================
class DatasetExample:
    """
    EXAMPLE 2: DATASET MANAGEMENT - Create and manage evaluation datasets
    
    What it does:
    - Create datasets programmatically
    - Add columns and rows
    - Support multiple import methods
    - Version control datasets
    
    Methods demonstrated:
    1. Programmatic creation (SDK)
    2. CSV import
    3. JSON import
    4. Synthetic data generation
    """
    
    def __init__(self):
        self.datasets = {}
    
    def create_dataset_programmatically(self, name: str, data: List[Dict]):
        """
        Create dataset using SDK
        
        Logic:
        1. Define schema (column names and types)
        2. Create dataset
        3. Add rows of data
        4. Store reference
        """
        dataset_info = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "rows": len(data),
            "columns": list(data[0].keys()) if data else [],
            "data": data,
            "method": "programmatic"
        }
        
        self.datasets[name] = dataset_info
        print(f"✓ Created dataset: {name}")
        print(f"  Rows: {len(data)}, Columns: {len(dataset_info['columns'])}")
        return dataset_info
    
    def import_from_csv(self, name: str, csv_content: str):
        """
        Import dataset from CSV
        
        Logic:
        1. Parse CSV content
        2. Extract headers and rows
        3. Create dataset structure
        """
        lines = csv_content.strip().split('\n')
        headers = lines[0].split(',')
        data = []
        
        for line in lines[1:]:
            values = line.split(',')
            row = {headers[i]: values[i] for i in range(len(headers))}
            data.append(row)
        
        dataset_info = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "rows": len(data),
            "columns": headers,
            "data": data,
            "method": "csv_import"
        }
        
        self.datasets[name] = dataset_info
        print(f"✓ Imported CSV dataset: {name}")
        return dataset_info
    
    def import_from_json(self, name: str, json_content: str):
        """Import dataset from JSON"""
        data = json.loads(json_content)
        
        if isinstance(data, list):
            rows = len(data)
            columns = list(data[0].keys()) if data else []
        else:
            data = [data]
            rows = 1
            columns = list(data[0].keys())
        
        dataset_info = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "rows": rows,
            "columns": columns,
            "data": data,
            "method": "json_import"
        }
        
        self.datasets[name] = dataset_info
        print(f"✓ Imported JSON dataset: {name}")
        return dataset_info
    
    def create_synthetic_dataset(self, name: str, size: int = 10):
        """
        Generate synthetic dataset using Gemini
        
        Logic:
        1. Define data schema
        2. Use Gemini to generate realistic data
        3. Parse and structure
        """
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Generate {size} rows of realistic customer support data in JSON format.
        Each row should have: customer_name, issue_type, message, sentiment
        Return ONLY valid JSON array, no markdown or extra text.
        """
        
        response = model.generate_content(prompt)
        
        try:
            # Extract JSON from response
            json_str = response.text
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            
            data = json.loads(json_str)
            
            dataset_info = {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "rows": len(data),
                "columns": list(data[0].keys()) if data else [],
                "data": data,
                "method": "synthetic_generation"
            }
            
            self.datasets[name] = dataset_info
            print(f"✓ Generated synthetic dataset: {name}")
            return dataset_info
        
        except Exception as e:
            print(f"✗ Failed to generate synthetic data: {e}")
            return None
    
    def list_datasets(self):
        """List all datasets"""
        print("\n" + "="*60)
        print("DATASETS MANAGEMENT")
        print("="*60)
        
        for name, info in self.datasets.items():
            print(f"\n📊 {name}")
            print(f"   Method: {info['method']}")
            print(f"   Rows: {info['rows']}")
            print(f"   Columns: {', '.join(info['columns'])}")
            print(f"   Created: {info['created_at']}")
        
        return self.datasets
    
    def get_dataset(self, name: str):
        """Retrieve dataset"""
        return self.datasets.get(name)


# ==================== EXAMPLE 3: EVALUATION ====================
class EvaluationExample:
    """
    EXAMPLE 3: EVALUATION - Evaluate AI outputs against metrics
    
    What it does:
    - Define custom evaluation rules
    - Evaluate generated content
    - Score outputs (pass/fail, percentage, multi-choice)
    - Generate evaluation reports
    
    Evaluation types:
    1. Accuracy - does output match expected?
    2. Safety - is output free of toxins, PII?
    3. Relevance - is output relevant to input?
    4. Compliance - does output follow rules?
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.evaluations = []
    
    def evaluate_accuracy(self, prompt: str, expected: str, actual: str) -> Dict:
        """
        Evaluate accuracy using LLM as judge
        
        Logic:
        1. Use Gemini to compare expected vs actual
        2. Score on 0-100 scale
        3. Return detailed feedback
        """
        
        eval_prompt = f"""
        Evaluate if this output matches the expected result.
        
        Expected: {expected}
        Actual: {actual}
        
        Provide:
        1. Score (0-100)
        2. Match percentage
        3. Key differences
        4. Pass/Fail
        
        Return JSON format.
        """
        
        response = self.model.generate_content(eval_prompt)
        
        evaluation = {
            "type": "accuracy",
            "prompt": prompt,
            "expected": expected,
            "actual": actual,
            "timestamp": datetime.now().isoformat(),
            "model_feedback": response.text
        }
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def evaluate_safety(self, content: str) -> Dict:
        """
        Evaluate content safety
        
        Checks for:
        - Toxicity
        - PII (personal information)
        - Bias
        - Hate speech
        """
        
        eval_prompt = f"""
        Evaluate this content for safety issues.
        
        Content: {content}
        
        Check for:
        1. Toxicity (0-10)
        2. PII (0-10)
        3. Bias (0-10)
        4. Safety rating (Safe/Unsafe)
        
        Return JSON format.
        """
        
        response = self.model.generate_content(eval_prompt)
        
        evaluation = {
            "type": "safety",
            "content": content[:100],
            "timestamp": datetime.now().isoformat(),
            "model_feedback": response.text
        }
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def evaluate_relevance(self, query: str, response: str) -> Dict:
        """Evaluate if response is relevant to query"""
        
        eval_prompt = f"""
        Rate the relevance of this response to the query.
        
        Query: {query}
        Response: {response}
        
        Provide:
        1. Relevance score (0-100)
        2. Key points addressed
        3. Missing information
        
        Return JSON format.
        """
        
        response_obj = self.model.generate_content(eval_prompt)
        
        evaluation = {
            "type": "relevance",
            "query": query,
            "response": response[:100],
            "timestamp": datetime.now().isoformat(),
            "model_feedback": response_obj.text
        }
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def run_custom_evaluation(self, rule: str, content: str) -> Dict:
        """
        Run custom evaluation rule
        
        Logic:
        1. Define custom rule
        2. Apply to content
        3. Score output
        """
        
        eval_prompt = f"""
        Evaluate this content against the rule.
        
        Rule: {rule}
        Content: {content}
        
        Provide:
        1. Pass/Fail
        2. Score (0-100)
        3. Explanation
        
        Return JSON format.
        """
        
        response = self.model.generate_content(eval_prompt)
        
        evaluation = {
            "type": "custom",
            "rule": rule,
            "content": content[:100],
            "timestamp": datetime.now().isoformat(),
            "model_feedback": response.text
        }
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def generate_report(self):
        """Generate evaluation report"""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"\nTotal Evaluations: {len(self.evaluations)}")
        
        by_type = {}
        for eval_data in self.evaluations:
            eval_type = eval_data['type']
            by_type[eval_type] = by_type.get(eval_type, 0) + 1
        
        print("\nEvaluations by Type:")
        for eval_type, count in by_type.items():
            print(f"  {eval_type}: {count}")
        
        return self.evaluations


# ==================== EXAMPLE 4: PROMPT OPTIMIZATION ====================
class PromptOptimizationExample:
    """
    EXAMPLE 4: PROMPT OPTIMIZATION - Improve prompts automatically
    
    What it does:
    - Generate variations of prompts
    - Evaluate each variation
    - Select best performing version
    - Track optimization history
    
    Logic:
    1. Define initial prompt
    2. Generate variations using teacher model
    3. Evaluate each variation
    4. Compare results
    5. Return best prompt
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.optimization_history = []
    
    def generate_variations(self, original_prompt: str, num_variations: int = 3) -> List[str]:
        """
        Generate prompt variations using Gemini
        
        Logic:
        1. Request Gemini to create variations
        2. Parse and return variations
        """
        
        variation_prompt = f"""
        Generate {num_variations} different versions of this prompt that should produce better results.
        Make them progressively more detailed and specific.
        
        Original: {original_prompt}
        
        Return ONLY the variations as a numbered list, nothing else.
        """
        
        response = self.model.generate_content(variation_prompt)
        
        variations = []
        for line in response.text.split('\n'):
            if line.strip() and line[0].isdigit():
                # Remove numbering
                variation = line.split('. ', 1)[-1] if '. ' in line else line
                variations.append(variation.strip())
        
        return variations[:num_variations]
    
    def evaluate_prompt(self, prompt: str, test_query: str, evaluation_criteria: str) -> float:
        """
        Evaluate a prompt by testing it
        
        Logic:
        1. Use prompt to generate response
        2. Evaluate response against criteria
        3. Return score (0-1)
        """
        
        # Generate content with prompt
        response = self.model.generate_content(f"{prompt}\n\nQuestion: {test_query}")
        
        # Evaluate response
        eval_prompt = f"""
        Evaluate this response against the criteria.
        
        Criteria: {evaluation_criteria}
        Response: {response.text}
        
        Score from 0-1, where 1 is perfect.
        Return ONLY a number like: 0.85
        """
        
        eval_response = self.model.generate_content(eval_prompt)
        
        try:
            score = float(eval_response.text.strip())
            return min(1.0, max(0.0, score))
        except:
            return 0.5
    
    def optimize(self, original_prompt: str, test_query: str, evaluation_criteria: str):
        """
        Run full optimization loop
        
        Returns: Optimized prompt with score improvement
        """
        
        print("\n" + "="*60)
        print("PROMPT OPTIMIZATION")
        print("="*60)
        
        # Evaluate original
        original_score = self.evaluate_prompt(original_prompt, test_query, evaluation_criteria)
        print(f"\n📝 Original Prompt Score: {original_score:.3f}")
        print(f"   Prompt: {original_prompt[:60]}...")
        
        # Generate variations
        print(f"\n🔄 Generating {3} variations...")
        variations = self.generate_variations(original_prompt, num_variations=3)
        
        # Evaluate each variation
        best_prompt = original_prompt
        best_score = original_score
        
        print("\n📊 Evaluating variations:")
        for i, variation in enumerate(variations, 1):
            score = self.evaluate_prompt(variation, test_query, evaluation_criteria)
            print(f"   Variation {i}: {score:.3f}")
            print(f"     {variation[:60]}...")
            
            if score > best_score:
                best_score = score
                best_prompt = variation
        
        # Report results
        improvement = ((best_score - original_score) / original_score * 100) if original_score > 0 else 0
        
        print(f"\n✅ Optimization Complete")
        print(f"   Best Score: {best_score:.3f}")
        print(f"   Improvement: {improvement:+.1f}%")
        print(f"   Best Prompt: {best_prompt[:80]}...")
        
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "original_score": original_score,
            "best_score": best_score,
            "improvement_percent": improvement,
            "original_prompt": original_prompt,
            "best_prompt": best_prompt,
            "variations_count": len(variations)
        }
        
        self.optimization_history.append(history_entry)
        return best_prompt, best_score, improvement


# ==================== EXAMPLE 5: GUARDRAILS / CONTENT FILTERING ====================
class GuardrailsExample:
    """
    EXAMPLE 5: GUARDRAILS - Filter unsafe content
    
    What it does:
    - Screen for PII (emails, phone numbers, SSN)
    - Detect toxicity
    - Check for policy violations
    - Filter and redact content
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.filtered_count = 0
    
    def check_pii(self, content: str) -> Dict:
        """Detect PII in content"""
        
        check_prompt = f"""
        Scan this content for Personally Identifiable Information (PII).
        
        Content: {content}
        
        Find:
        1. Email addresses
        2. Phone numbers
        3. Social Security Numbers
        4. Credit card numbers
        5. Names
        
        Return JSON format with findings.
        """
        
        response = self.model.generate_content(check_prompt)
        
        return {
            "type": "pii_check",
            "content": content[:100],
            "timestamp": datetime.now().isoformat(),
            "findings": response.text
        }
    
    def check_toxicity(self, content: str) -> Dict:
        """Check for toxic content"""
        
        check_prompt = f"""
        Evaluate this content for toxicity, hate speech, and inappropriate language.
        
        Content: {content}
        
        Rate:
        1. Toxicity level (0-10)
        2. Hate speech present (Yes/No)
        3. Appropriate for all audiences (Yes/No)
        
        Return JSON format.
        """
        
        response = self.model.generate_content(check_prompt)
        
        return {
            "type": "toxicity_check",
            "content": content[:100],
            "timestamp": datetime.now().isoformat(),
            "findings": response.text
        }
    
    def filter_content(self, content: str, rules: List[str]) -> str:
        """
        Apply filtering rules to content
        
        Logic:
        1. Define rules (policy violations)
        2. Check content against rules
        3. Redact or reject if violations found
        """
        
        filter_prompt = f"""
        Filter this content against the rules. Redact any violations.
        
        Rules: {', '.join(rules)}
        Content: {content}
        
        Return filtered content with [REDACTED] for violations.
        """
        
        response = self.model.generate_content(filter_prompt)
        
        self.filtered_count += 1
        
        return response.text
    
    def safe_content_pipeline(self, content: str) -> Dict:
        """
        Full safety pipeline
        
        Logic:
        1. Check for PII
        2. Check for toxicity
        3. Apply policy filters
        4. Return safe content
        """
        
        print("\n" + "="*60)
        print("CONTENT SAFETY PIPELINE")
        print("="*60)
        
        print(f"\n🔍 Original Content: {content[:60]}...")
        
        # Check PII
        pii_check = self.check_pii(content)
        print(f"\n📋 PII Check: {pii_check['findings'][:100]}...")
        
        # Check toxicity
        toxicity_check = self.check_toxicity(content)
        print(f"\n⚠️  Toxicity Check: {toxicity_check['findings'][:100]}...")
        
        # Apply filters
        rules = [
            "No external links",
            "No promotional content",
            "No misleading information"
        ]
        
        filtered = self.filter_content(content, rules)
        print(f"\n✅ Filtered Content: {filtered[:60]}...")
        
        return {
            "original": content,
            "filtered": filtered,
            "pii_check": pii_check,
            "toxicity_check": toxicity_check,
            "safe": filtered
        }


# ==================== MAIN EXECUTION ====================
def main():
    """Run all examples"""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     FUTURE AGI + GEMINI INTEGRATION EXAMPLES              ║
    ║              Complete Feature Demonstration               ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    try:
        genai_module, fi_client = setup_clients()
        print("✓ Clients initialized successfully\n")
    except ValueError as e:
        print(f"✗ Setup failed: {e}")
        print("\nSet environment variables:")
        print("  export GEMINI_API_KEY='your_key'")
        print("  export FI_API_KEY='your_key'")
        print("  export FI_SECRET_KEY='your_key'")
        return
    
    # # Example 1: Tracing
    # print("\n" + "="*60)
    # print("EXAMPLE 1: TRACING")
    # print("="*60)
    # try:
    #     tracer = TracingExample()
        
    #     # Traced call
    #     response, trace_data = tracer.traced_call(
    #         "Explain quantum computing in one sentence",
    #         "quantum_explanation"
    #     )
        
    #     # Show comparison
    #     tracer.compare_outputs()
        
    #     # Export
    #     tracer.export_traces()
        
    # except Exception as e:
    #     print(f"✗ Tracing example failed: {e}")
    
    # Example 2: Dataset Management
    print("\n" + "="*60)
    print("EXAMPLE 2: DATASET MANAGEMENT")
    print("="*60)
    try:
        dataset_mgr = DatasetExample()
        
        # Create dataset programmatically
        sample_data = [
            {"text": "Great product!", "sentiment": "positive"},
            {"text": "Terrible experience", "sentiment": "negative"},
            {"text": "It's okay", "sentiment": "neutral"}
        ]
        dataset_mgr.create_dataset_programmatically("feedback", sample_data)
        
        # Import from CSV
        csv_data = """name,age,city
John,25,NYC
Jane,30,LA
Bob,35,Chicago"""
        dataset_mgr.import_from_csv("customers", csv_data)
        
        # List datasets
        dataset_mgr.list_datasets()
        
    except Exception as e:
        print(f"✗ Dataset example failed: {e}")
    
    # Example 3: Evaluation
    # print("\n" + "="*60)
    # print("EXAMPLE 3: EVALUATION")
    # print("="*60)
    # try:
    #     evaluator = EvaluationExample()
        
    #     # Accuracy evaluation
    #     evaluator.evaluate_accuracy(
    #         "What is 2+2?",
    #         expected="4",
    #         actual="The sum of 2 and 2 is 4"
    #     )
        
    #     # Safety evaluation
    #     evaluator.evaluate_safety(
    #         "This product is amazing! Contact us at 555-1234"
    #     )
        
    #     # Relevance evaluation
    #     evaluator.evaluate_relevance(
    #         "How do I reset my password?",
    #         "Click on 'Forgot Password' on the login page"
    #     )
        
    #     # Generate report
    #     evaluator.generate_report()
        
    # except Exception as e:
    #     print(f"✗ Evaluation example failed: {e}")
    
    # Example 4: Prompt Optimization
    # print("\n" + "="*60)
    # print("EXAMPLE 4: PROMPT OPTIMIZATION")
    # print("="*60)
    # try:
    #     optimizer = PromptOptimizationExample()
        
    #     optimizer.optimize(
    #         original_prompt="Write about AI",
    #         test_query="What is the impact of AI on society?",
    #         evaluation_criteria="Comprehensive, balanced, well-structured"
    #     )
        
    # except Exception as e:
    #     print(f"✗ Optimization example failed: {e}")
    
    # Example 5: Guardrails
    # print("\n" + "="*60)
    # print("EXAMPLE 5: GUARDRAILS / CONTENT FILTERING")
    # print("="*60)
    # try:
    #     guardrails = GuardrailsExample()
        
    #     unsafe_content = """
    #     Call John at 555-123-4567 or email john@email.com.
    #     This service is absolutely terrible and disgraceful!
    #     """
        
    #     result = guardrails.safe_content_pipeline(unsafe_content)
        
    # except Exception as e:
    #     print(f"✗ Guardrails example failed: {e}")
    
    # print("\n" + "="*60)
    # print("✅ ALL EXAMPLES COMPLETED")
    # print("="*60)


if __name__ == "__main__":
    main()