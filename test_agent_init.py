#!/usr/bin/env python3
import os
import logging
import traceback
import asyncio
from dotenv import load_dotenv

from agents import Agent, ModelSettings, function_tool, Runner
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent_init_test")

@function_tool
def echo(text: str) -> str:
    """Echo the input text."""
    return text

def create_test_agent():
    """Create and return a test agent using the same pattern as crypto_agent.py."""
    # Configure API settings
    base_url = os.environ.get("OPENAI_BASE_URL", "https://litellm.deriv.ai/v1")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set")
    
    logger.info(f"Creating agent with base URL: {base_url}")
    
    # Initialize the OpenAI client with LiteLLM configuration
    external_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Create the agent with the model settings
    agent = Agent(
        name="TestAgent",
        instructions="You are a helpful assistant. You will echo back text provided to you.",
        model=OpenAIChatCompletionsModel(
            model="claude-3-7-sonnet-latest",
            openai_client=external_client,
        ),
        model_settings=ModelSettings(temperature=0.2),
        tools=[echo]
    )
    return agent

async def test_agent():
    """Test that the agent can be initialized and run."""
    try:
        # Create the agent
        agent = create_test_agent()
        logger.info("Agent initialized successfully!")
        
        # Test a simple echo query
        logger.info("Testing agent with a simple query...")
        result = await Runner.run(agent, "Echo this text: Hello, world!")
        
        if result and hasattr(result, 'final_output'):
            logger.info(f"Agent response: {result.final_output}")
            return True
        else:
            logger.error("Agent did not return a valid response")
            return False
            
    except Exception as e:
        logger.error(f"Error testing agent: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def main():
    # Try to load environment variables from .env file
    try:
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    except Exception as e:
        logger.warning(f"Failed to load .env file: {str(e)}")
    
    # Check if API key is set
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No API key found. Set either LITELLM_API_KEY or OPENAI_API_KEY in your environment.")
        return
    else:
        # Use the API key from LITELLM_API_KEY if set, otherwise use OPENAI_API_KEY
        if os.environ.get("LITELLM_API_KEY"):
            logger.info("Using LITELLM_API_KEY for authentication")
            os.environ["OPENAI_API_KEY"] = os.environ.get("LITELLM_API_KEY")
    
    logger.info("Testing agent initialization and execution...")
    success = await test_agent()
    
    if success:
        logger.info("✅ Test passed! Agent was initialized and executed correctly.")
    else:
        logger.error("❌ Test failed! Could not initialize or execute agent.")

if __name__ == "__main__":
    asyncio.run(main()) 