"""Test script to verify Future AGI tracing is working."""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()

print("="*70)
print("🧪 Future AGI Trace Verification Test")
print("="*70)
print()

# Step 1: Check API keys
print("📌 Step 1: Checking API Keys...")
fi_api_key = os.getenv("FI_API_KEY")
fi_secret_key = os.getenv("FI_SECRET_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not fi_api_key or not fi_secret_key:
    print("❌ Missing Future AGI credentials!")
    exit(1)

if not google_api_key:
    print("❌ Missing Google API key!")
    exit(1)

print(f"✓ FI_API_KEY: {fi_api_key[:8]}...")
print(f"✓ FI_SECRET_KEY: {fi_secret_key[:8]}...")
print(f"✓ GOOGLE_API_KEY: {google_api_key[:8]}...")
print()

# Step 2: Register with Future AGI
print("📌 Step 2: Registering with Future AGI...")
try:
    from fi_instrumentation import register, Transport
    from fi_instrumentation.fi_types import ProjectType
    
    os.environ["FI_API_KEY"] = fi_api_key
    os.environ["FI_SECRET_KEY"] = fi_secret_key
    
    trace_provider = register(
        project_type=ProjectType.OBSERVE,
        project_name="test-tracing-verification",
        transport=Transport.GRPC
    )
    print("✓ Successfully registered with Future AGI")
    print(f"  Project: test-tracing-verification")
    print(f"  Transport: GRPC")
    print()
except Exception as e:
    print(f"❌ Registration failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Configure Google GenAI
print("📌 Step 3: Configuring Google GenAI...")
genai.configure(api_key=google_api_key)
print("✓ Google GenAI configured")
print()

# Step 4: Instrument with Future AGI
print("📌 Step 4: Enabling TraceAI Instrumentation...")
try:
    from traceai_google_genai import GoogleGenAIInstrumentor
    
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.instrument(tracer_provider=trace_provider)
    print("✓ TraceAI instrumentation enabled")
    print()
except Exception as e:
    print(f"❌ Instrumentation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Make test API calls
print("📌 Step 5: Making Test API Calls...")
print("-" * 70)

try:
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    # Test call 1
    print("\n🔹 Test Call 1: Simple question")
    response = model.generate_content("What is 2+2? Answer in one word.")
    print(f"   Response: {response.text}")
    
    # Test call 2
    print("\n🔹 Test Call 2: Another question")
    response = model.generate_content("What is the capital of France? Answer in one word.")
    print(f"   Response: {response.text}")
    
    print("\n✓ Both API calls completed successfully")
    
except Exception as e:
    print(f"❌ API call failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("="*70)
print("✅ Test Complete!")
print("="*70)
print()
print("📊 Next Steps:")
print()
print("1. Go to: https://app.futureagi.com")
print("2. Navigate to: Dashboard → Projects")
print("3. Look for project: 'test-tracing-verification'")
print("   OR click on 'Traces' in the sidebar")
print()
print("4. You should see TWO traces:")
print("   • 'What is 2+2?'")
print("   • 'What is the capital of France?'")
print()
print("5. Click on a trace to see:")
print("   • Request timestamp")
print("   • Model: gemini-2.0-flash-exp")
print("   • Full prompt")
print("   • Full response")
print("   • Token usage")
print("   • Latency")
print()
print("⏱️  Traces may take 5-30 seconds to appear in the dashboard")
print()
print("❓ If you don't see traces:")
print("   • Refresh the page")
print("   • Check if you're in the right project")
print("   • Verify API keys are correct")
print("   • Check 'All Projects' or 'All Sessions' filter")
print()
print("="*70)
