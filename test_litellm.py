#!/usr/bin/env python3
import os
import argparse
import logging
import json
import requests
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("litellm_tester")

def verify_litellm_connection(api_key: str, base_url: str = "https://litellm.deriv.ai/v1") -> Dict[str, Any]:
    """
    Verify connection to LiteLLM API.
    
    Args:
        api_key: LiteLLM API key
        base_url: Base URL for the LiteLLM API
        
    Returns:
        Dictionary with test results
    """
    results = {
        "success": False,
        "models_available": False,
        "claude_available": False,
        "error": None,
        "models": []
    }
    
    # Test API access
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Try to access models endpoint
        models_url = f"{base_url}/models"
        logger.info(f"Testing connection to {models_url}")
        
        response = requests.get(models_url, headers=headers)
        
        if response.status_code == 200:
            models_data = response.json()
            results["success"] = True
            results["models_available"] = True
            
            if "data" in models_data:
                results["models"] = [model["id"] for model in models_data["data"]]
                logger.info(f"Found {len(results['models'])} models available")
                
                # Check if Claude model is available
                claude_models = [model for model in results["models"] 
                               if "claude" in model.lower()]
                results["claude_available"] = len(claude_models) > 0
                results["claude_models"] = claude_models
                
                if results["claude_available"]:
                    logger.info(f"Claude models available: {claude_models}")
                else:
                    logger.warning("No Claude models found")
            else:
                logger.warning("Models data format unexpected")
        else:
            results["error"] = f"HTTP {response.status_code}: {response.text}"
            logger.error(f"Failed to connect: {results['error']}")
            
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Connection test failed: {str(e)}")
    
    # If the connection was successful, try a simple completion
    if results["success"]:
        try:
            # Use a model that should be available (adjust if needed)
            test_model = "gpt-3.5-turbo"
            if results["claude_available"] and results["claude_models"]:
                test_model = results["claude_models"][0]
                
            completion_url = f"{base_url}/chat/completions"
            logger.info(f"Testing completion with model {test_model}")
            
            payload = {
                "model": test_model,
                "messages": [{"role": "user", "content": "Say hello!"}],
                "max_tokens": 10
            }
            
            response = requests.post(completion_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                completion_data = response.json()
                results["completion_success"] = True
                results["completion_response"] = completion_data
                logger.info("Completion test successful")
            else:
                results["completion_success"] = False
                results["completion_error"] = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Completion test failed: {results['completion_error']}")
                
        except Exception as e:
            results["completion_success"] = False
            results["completion_error"] = str(e)
            logger.error(f"Completion test failed: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test LiteLLM API Connection')
    parser.add_argument('--api-key', type=str, help='LiteLLM API Key (defaults to OPENAI_API_KEY env var)')
    parser.add_argument('--base-url', type=str, default="https://litellm.deriv.ai/v1", 
                      help='LiteLLM API Base URL')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key from arguments or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: API key is required. Provide it with --api-key or set OPENAI_API_KEY environment variable.")
        return
    
    # Test the connection
    print(f"Testing LiteLLM connection to: {args.base_url}")
    results = verify_litellm_connection(api_key, args.base_url)
    
    # Display results
    if results["success"]:
        print("\n✅ Successfully connected to LiteLLM API")
        
        if results["models_available"]:
            print(f"\nModels available ({len(results['models'])}):")
            for model in results["models"]:
                print(f"  - {model}")
        
        if results["claude_available"]:
            print("\n✅ Claude models are available:")
            for model in results["claude_models"]:
                print(f"  - {model}")
        else:
            print("\n❌ No Claude models found")
        
        if results.get("completion_success"):
            print("\n✅ Completion test successful")
        else:
            print(f"\n❌ Completion test failed: {results.get('completion_error', 'Unknown error')}")
    else:
        print(f"\n❌ Failed to connect to LiteLLM API: {results['error']}")
        print("\nTroubleshooting tips:")
        print("1. Verify your API key")
        print("2. Check if the API endpoint is correct")
        print("3. Ensure your internet connection is stable")
        print("4. Check if LiteLLM service is available")

if __name__ == "__main__":
    main() 