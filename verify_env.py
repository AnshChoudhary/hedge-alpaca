#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

def check_env_variables():
    """Check if required environment variables are set."""
    
    # First, load from .env file if it exists
    load_dotenv()
    
    print("\n=== Environment Variable Check ===\n")
    
    # Required variables
    required_vars = {
        "ALPACA_API_KEY_ID": "Required for Alpaca trading",
        "ALPACA_API_SECRET_KEY": "Required for Alpaca trading",
    }
    
    # At least one of these API keys is required
    api_key_vars = {
        "LITELLM_API_KEY": "Preferred API key for LiteLLM",
        "OPENAI_API_KEY": "Alternative API key (will use LITELLM_API_KEY if both are set)"
    }
    
    # Optional variables with defaults
    optional_vars = {
        "OPENAI_BASE_URL": "https://litellm.deriv.ai/v1"
    }
    
    missing_required = False
    
    # Check required variables
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: NOT SET - {description}")
            missing_required = True
    
    # Check API keys (at least one must be set)
    api_key_found = False
    for var, description in api_key_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: Set")
            api_key_found = True
        else:
            print(f"⚠️ {var}: Not set - {description}")
    
    if not api_key_found:
        print(f"❌ API KEY ERROR: Either LITELLM_API_KEY or OPENAI_API_KEY must be set")
        missing_required = True
    
    # Check optional variables
    for var, default in optional_vars.items():
        value = os.environ.get(var, default)
        print(f"ℹ️ {var}: {value}")
    
    print("\n===================================\n")
    
    if missing_required:
        print("ERROR: Some required environment variables are missing. Please set them in your .env file or environment.")
        return False
    else:
        print("All required environment variables are set.")
        return True

if __name__ == "__main__":
    if not check_env_variables():
        sys.exit(1) 