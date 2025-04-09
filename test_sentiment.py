#!/usr/bin/env python3
import asyncio
import logging
import sys
import json
import os
from datetime import datetime
import argparse
import traceback

# Import the sentiment analysis agent
from sentiment_agent import SentimentAnalysisAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("test_sentiment")

async def test_sentiment():
    """Test the sentiment analysis agent with various cryptocurrency symbols."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Cryptocurrency Sentiment Analysis')
    parser.add_argument('--symbol', type=str, default='BTC/USD', 
                      help='Cryptocurrency symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='24h',
                      help='Timeframe for sentiment analysis (e.g., 24h, 7d)')
    parser.add_argument('--model', type=str, default='sonar-pro',
                      help='Model to use for sentiment analysis')
    parser.add_argument('--detailed', action='store_true',
                      help='Generate a detailed report across multiple timeframes')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--save', action='store_true',
                      help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Load environment variables if .env file exists
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            logger.info("Loaded environment variables from .env file")
        else:
            logger.warning("No .env file found or it's empty")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env loading")
    
    # Check for required environment variables
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No API key found. Set either LITELLM_API_KEY or OPENAI_API_KEY in your environment or .env file")
        return
    
    logger.info(f"Testing sentiment analysis with model: {args.model}")
    logger.info(f"Analyzing symbol: {args.symbol}")
    
    try:
        # Initialize the sentiment analysis agent
        start_time = datetime.now()
        agent = SentimentAnalysisAgent(model=args.model)
        logger.info(f"Agent initialized in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        
        # Run sentiment analysis
        if args.detailed:
            logger.info(f"Running detailed sentiment analysis across multiple timeframes...")
            start_time = datetime.now()
            report = await agent.get_detailed_sentiment_report(args.symbol, ["24h", "7d"])
            duration = (datetime.now() - start_time).total_seconds()
            
            # Display results
            print("\n" + "="*70)
            print(f"📊 DETAILED SENTIMENT ANALYSIS FOR {args.symbol}")
            print("="*70)
            
            print(f"\nOVERALL SENTIMENT: {report['overall_sentiment'].upper()}")
            print(f"SENTIMENT SCORE: {report['sentiment_score']:.2f}")
            print(f"CONFIDENCE: {report['confidence']:.2f}")
            print(f"TREND: {report['sentiment_trend'].upper()}")
            
            print("\nKEY FACTORS:")
            for factor in report.get('key_factors', [])[:5]:
                print(f"- {factor}")
                
            print("\nTRADING INSIGHTS:")
            insights = report['trading_insights']
            print(f"- Recommended Action: {insights['recommended_action'].upper()}")
            print(f"- Position Size Factor: {insights['position_size_factor']:.2f}")
            print(f"- Suggested Stop Loss: {insights['suggested_stop_loss_pct']*100:.1f}%")
            print(f"- Suggested Take Profit: {insights['suggested_take_profit_pct']*100:.1f}%")
            print(f"- Entry Timing: {insights['entry_timing'].upper()}")
            
            print(f"\nSUMMARY: {report.get('summary', '')}")
            
            logger.info(f"Analysis completed in {duration:.2f} seconds")
            
            # Save results if requested
            if args.save:
                filename = f"sentiment_{args.symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Results saved to {filename}")
            
        else:
            logger.info(f"Running basic sentiment analysis for {args.timeframe} timeframe...")
            start_time = datetime.now()
            result = await agent.analyze_sentiment(args.symbol, args.timeframe)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Display results
            print("\n" + "="*70)
            print(f"📊 SENTIMENT ANALYSIS FOR {args.symbol} ({args.timeframe})")
            print("="*70)
            
            print(f"\nSENTIMENT: {result['overall_sentiment'].upper()}")
            print(f"SCORE: {result['sentiment_score']:.2f}")
            print(f"CONFIDENCE: {result['confidence']:.2f}")
            
            if 'key_factors' in result:
                print("\nKEY FACTORS:")
                for factor in result['key_factors'][:5]:
                    print(f"- {factor}")
                    
            if 'significant_events' in result:
                print("\nSIGNIFICANT EVENTS:")
                for event in result['significant_events'][:3]:
                    print(f"- {event}")
                    
            if 'summary' in result:
                print(f"\nSUMMARY: {result['summary']}")
                
            logger.info(f"Analysis completed in {duration:.2f} seconds")
            
            # Save results if requested
            if args.save:
                filename = f"sentiment_{args.symbol.replace('/', '_')}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to {filename}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        logger.error(f"Error running sentiment analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test function
    asyncio.run(test_sentiment()) 