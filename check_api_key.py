"""Check if Groq API key is configured correctly"""
try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ Error: python-dotenv is not installed")
    print("\nPlease install it:")
    print("  pip install python-dotenv")
    print("\nOr use the Anaconda Python environment where packages are installed:")
    print("  (Make sure you're in the Anaconda base environment)")
    exit(1)

import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL_NAME", "mixtral-8x7b-32768")

print("=" * 60)
print("Groq API Key Configuration Check")
print("=" * 60)

if not api_key:
    print("❌ GROQ_API_KEY not found in environment")
    print("\nPlease set it in your .env file:")
    print("  GROQ_API_KEY=your-actual-api-key-here")
elif api_key == "your-groq-api-key-here":
    print("❌ GROQ_API_KEY is still set to placeholder value!")
    print("\n⚠️  You need to replace 'your-groq-api-key-here' with your actual API key")
    print("\nTo fix:")
    print("1. Open the .env file")
    print("2. Replace 'your-groq-api-key-here' with your actual Groq API key")
    print("3. Get your API key from: https://console.groq.com/")
    print("4. The key should start with 'gsk_' and be about 50+ characters long")
else:
    print(f"✅ GROQ_API_KEY found")
    print(f"   Key length: {len(api_key)} characters")
    print(f"   Key starts with: {api_key[:10]}...")
    print(f"   Key ends with: ...{api_key[-4:]}")
    
    # Test if it's a valid format (Groq keys usually start with 'gsk_')
    if api_key.startswith('gsk_'):
        print("   ✅ Key format looks correct (starts with 'gsk_')")
    else:
        print("   ⚠️  Key format might be incorrect (should start with 'gsk_')")

print(f"\nModel Name: {model_name}")
print("=" * 60)

