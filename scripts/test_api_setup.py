#!/usr/bin/env python3
"""
API Setup Verification Script

This script tests if your API keys are properly configured for the UNFCCC QA system.
Run this after setting up your API keys to verify everything is working.

Usage:
    python test_api_setup.py
"""

import os
import sys
from pathlib import Path

# Ensure stdout can display emoji on Windows CMD
if sys.stdout.encoding is None or "UTF-8" not in sys.stdout.encoding.upper():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def test_environment_variables():
    """Test if environment variables are properly set."""
    print("🔍 Testing Environment Variables...")
    
    openai_env = os.getenv('OPENAI_API_KEY')
    azure_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
    azure_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
    
    # Test OpenAI key
    if openai_env:
        if openai_env.startswith('sk-proj-') and 'your-' not in openai_env:
            print("  ✅ OPENAI_API_KEY environment variable: Properly configured")
        elif 'your-' in openai_env:
            print("  ❌ OPENAI_API_KEY environment variable: Contains template value")
            return False
        else:
            print("  ⚠️  OPENAI_API_KEY environment variable: Set but format unclear")
    else:
        print("  ⚠️  OPENAI_API_KEY environment variable: Not set")
    
    # Test Azure (optional)
    if azure_name and azure_key:
        print("  ✅ Azure credentials: Set in environment")
    else:
        print("  ℹ️  Azure credentials: Not set (optional)")
    
    return True

def test_streamlit_secrets():
    """Test if Streamlit secrets are properly configured."""
    print("\n🔍 Testing Streamlit Secrets...")
    
    secrets_file = Path('.streamlit/secrets.toml')
    if not secrets_file.exists():
        print("  ❌ .streamlit/secrets.toml: File not found")
        print("     Create the file and add your API keys")
        return False
    
    try:
        with open(secrets_file, 'r') as f:
            content = f.read()
        
        # Check for OpenAI key
        if 'OPENAI_API_KEY' in content:
            if 'your-openai-api-key-here' in content:
                print("  ❌ .streamlit/secrets.toml: Contains template value for OPENAI_API_KEY")
                print("     Replace 'your-openai-api-key-here' with your actual API key")
                return False
            elif 'sk-proj-' in content:
                print("  ✅ .streamlit/secrets.toml: OPENAI_API_KEY appears properly configured")
            else:
                print("  ⚠️  .streamlit/secrets.toml: OPENAI_API_KEY set but format unclear")
        else:
            print("  ❌ .streamlit/secrets.toml: OPENAI_API_KEY not found")
            return False
        
        # Check for Azure (optional)
        if 'AZURE_STORAGE_ACCOUNT_NAME' in content and 'AZURE_STORAGE_ACCOUNT_KEY' in content:
            print("  ✅ .streamlit/secrets.toml: Azure credentials found")
        else:
            print("  ℹ️  .streamlit/secrets.toml: Azure credentials not found (optional)")
        
    except Exception as e:
        print(f"  ❌ .streamlit/secrets.toml: Error reading file: {e}")
        return False
    
    return True

def test_config_loading():
    """Test if the config system loads the keys correctly."""
    print("\n🔍 Testing Config System...")
    
    try:
        from config import OPENAI_API_KEY, AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY
        
        # Test OpenAI key loading
        if OPENAI_API_KEY:
            if OPENAI_API_KEY.startswith('sk-proj-') and 'your-' not in OPENAI_API_KEY:
                print("  ✅ Config system: OPENAI_API_KEY loaded correctly")
            elif 'your-' in OPENAI_API_KEY:
                print("  ❌ Config system: OPENAI_API_KEY loaded template value")
                print("     Check .streamlit/secrets.toml for template values")
                return False
            else:
                print("  ⚠️  Config system: OPENAI_API_KEY loaded but format unclear")
        else:
            print("  ❌ Config system: OPENAI_API_KEY not loaded")
            return False
        
        # Test Azure (optional)
        if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
            print("  ✅ Config system: Azure credentials loaded")
        else:
            print("  ℹ️  Config system: Azure credentials not loaded (optional)")
        
    except Exception as e:
        print(f"  ❌ Config system: Error importing config: {e}")
        return False
    
    return True

def test_openai_client():
    """Test if the OpenAI client initializes properly."""
    print("\n🔍 Testing OpenAI Client...")
    
    try:
        from rag_engine import client
        
        if client:
            print("  ✅ OpenAI client: Initialized successfully")
            return True
        else:
            print("  ❌ OpenAI client: Failed to initialize")
            print("     Check your OPENAI_API_KEY in .streamlit/secrets.toml")
            return False
            
    except Exception as e:
        print(f"  ❌ OpenAI client: Error importing rag_engine: {e}")
        return False

def test_api_connection():
    """Test actual API connection (optional, requires credits)."""
    print("\n🔍 Testing API Connection (Optional)...")
    
    try:
        from config import OPENAI_API_KEY
        import openai
        
        if not OPENAI_API_KEY or 'your-' in OPENAI_API_KEY:
            print("  ⏭️  Skipping API test: No valid API key")
            return True
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Simple test request
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, respond with just 'OK'"}],
            max_tokens=5
        )
        
        if response.choices[0].message.content.strip().upper() == 'OK':
            print("  ✅ API connection: Successful test request")
            return True
        else:
            print("  ⚠️  API connection: Unexpected response")
            return True
            
    except Exception as e:
        if "401" in str(e):
            print("  ❌ API connection: Authentication failed (invalid API key)")
            return False
        elif "rate_limit" in str(e).lower():
            print("  ⚠️  API connection: Rate limited (API key is valid)")
            return True
        else:
            print(f"  ⚠️  API connection: Error testing connection: {e}")
            return True  # Don't fail on connection issues

def main():
    """Run all tests and provide summary."""
    print("🔐 UNFCCC API Setup Verification")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Environment Variables", test_environment_variables()))
    results.append(("Streamlit Secrets", test_streamlit_secrets()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("OpenAI Client", test_openai_client()))
    results.append(("API Connection", test_api_connection()))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API setup is working correctly.")
        print("   You can now run: streamlit run cluster_qa_app.py")
    elif passed >= 3:
        print("\n⚠️  Setup is mostly working, but some issues detected.")
        print("   Check the failed tests above and review SECURITY_SETUP.md")
    else:
        print("\n❌ Setup issues detected. Please review the following:")
        print("   1. Check SECURITY_SETUP.md for detailed instructions")
        print("   2. Ensure you have the correct API keys")
        print("   3. Verify .streamlit/secrets.toml contains your real keys")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 