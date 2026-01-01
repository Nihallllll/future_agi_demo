"""Test script to demonstrate Future AGI tracing in action."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

print("="*60)
print("🧪 Future AGI Tracing Demonstration")
print("="*60)
print()

# Test 1: Without Instrumentation
print("📌 Test 1: Normal API Call (No Tracing)")
print("-" * 60)
model = genai.GenerativeModel("gemini-2.0-flash-exp")
response = model.generate_content("What is 2+2?")
print(f"Response: {response.text}")
print("Status: ✓ Completed")
print("Visibility: ❌ Not visible in Future AGI dashboard")
print()

# Test 2: With Instrumentation
print("📌 Test 2: Instrumented API Call (With Tracing)")
print("-" * 60)
try:
    from traceai_google_genai import GoogleGenAIInstrumentor
    
    # Enable instrumentation
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.instrument()
    print("✓ TraceAI instrumentation enabled")
    print()
    
    # Make the same call
    response = model.generate_content("What is 3+3?")
    print(f"Response: {response.text}")
    print("Status: ✓ Completed")
    print("Visibility: ✅ VISIBLE in Future AGI dashboard!")
    print()
    print("🔍 Next Steps:")
    print("1. Go to: https://app.futureagi.com")
    print("2. Navigate to 'Traces' or 'Observability'")
    print("3. You should see the trace for: 'What is 3+3?'")
    print()
    print("📊 What you'll see in the dashboard:")
    print("   - Request timestamp")
    print("   - Model: gemini-2.0-flash-exp")
    print("   - Input prompt: 'What is 3+3?'")
    print("   - Output response")
    print("   - Token usage (prompt + completion)")
    print("   - Latency (response time)")
    print("   - Cost (if applicable)")
    
except ImportError:
    print("❌ TraceAI not installed")

print()
print("="*60)
print("💡 Key Insight:")
print("="*60)
print("Without Future AGI: You only see the final answer")
print("With Future AGI: You see EVERYTHING that happened:")
print("  - Every API call")
print("  - Complete request/response")
print("  - Performance metrics")
print("  - Cost tracking")
print("  - Error traces (if any)")
print("="*60)
