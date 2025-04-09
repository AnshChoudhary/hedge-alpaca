#!/usr/bin/env python3
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

# Import OpenAI for LiteLLM integration
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("sentiment_agent")

class SentimentAnalysisAgent:
    """Agent that analyzes market sentiment from news, social media, and other sources."""
    
    def __init__(self, model="sonar-pro"):
        """Initialize the sentiment analysis agent.
        
        Args:
            model: The LiteLLM model to use for sentiment analysis
        """
        self.model = model
        self.client = None
        self.sources = {
            "news": True,
            "reddit": True,
            "twitter": True,
            "blogs": True
        }
        self.sentiment_cache = {}  # Cache sentiment results by symbol and timestamp
        self.cache_duration = timedelta(hours=4)  # Cache valid for 4 hours
        
        # Initialize the client
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the AsyncOpenAI client with LiteLLM configuration."""
        try:
            # Try to load .env file
            try:
                from dotenv import load_dotenv
                load_dotenv()
                logger.info("Loaded environment variables from .env file")
            except Exception as e:
                logger.warning(f"Failed to load .env file: {str(e)}")
                
            # Get API key from environment variables
            api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No API key found. Set either LITELLM_API_KEY or OPENAI_API_KEY")
                
            # Get API base URL
            api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.deriv.ai/v1")
            
            # Initialize the client
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base
            )
            
            logger.info(f"Initialized sentiment analysis agent with model: {self.model}")
            logger.info(f"Using API base URL: {api_base}")
            
        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def analyze_sentiment(self, symbol: str, timeframe: str = "24h") -> Dict[str, Any]:
        """Analyze sentiment for a cryptocurrency symbol.
        
        Args:
            symbol: The cryptocurrency symbol to analyze (e.g., BTC/USD)
            timeframe: The timeframe to analyze sentiment for (e.g., 24h, 7d)
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            # Check if we have a recent cached result
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.sentiment_cache:
                cached_result = self.sentiment_cache[cache_key]
                cached_time = cached_result.get("timestamp")
                if cached_time:
                    cache_age = datetime.now() - datetime.fromisoformat(cached_time)
                    if cache_age < self.cache_duration:
                        logger.info(f"Using cached sentiment result for {symbol} ({timeframe})")
                        return cached_result
            
            # Format symbol for better search results
            search_symbol = symbol.replace("/", "")  # BTC/USD -> BTCUSD
            crypto_name = self._get_crypto_name(symbol)  # BTC/USD -> Bitcoin
            
            # Build the prompt for the Sonar Pro model
            prompt = self._build_sentiment_analysis_prompt(symbol, search_symbol, crypto_name, timeframe)
            
            # Call the model to generate the sentiment analysis
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency analyst with specialization in market sentiment analysis. Your task is to analyze the current sentiment around a cryptocurrency by reviewing recent news, social media, and market analysis from credible sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            # Parse the response to extract sentiment data
            analysis_result = self._parse_sentiment_response(response, symbol, timeframe)
            
            # Cache the result
            self.sentiment_cache[cache_key] = analysis_result
            
            logger.info(f"Completed sentiment analysis for {symbol} ({timeframe}): {analysis_result['overall_sentiment']}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "sources_analyzed": 0,
                "error": str(e)
            }
    
    def _get_crypto_name(self, symbol: str) -> str:
        """Get the full name of a cryptocurrency from its symbol."""
        crypto_names = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "XRP": "Ripple",
            "LTC": "Litecoin",
            "BCH": "Bitcoin Cash",
            "ADA": "Cardano",
            "DOT": "Polkadot",
            "SOL": "Solana",
            "DOGE": "Dogecoin",
            "LINK": "Chainlink",
            "UNI": "Uniswap",
            "AVAX": "Avalanche",
            "MATIC": "Polygon",
            "ATOM": "Cosmos"
        }
        
        # Extract the base symbol (e.g., BTC from BTC/USD)
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        
        return crypto_names.get(base_symbol, base_symbol)
    
    def _build_sentiment_analysis_prompt(self, symbol: str, search_symbol: str, crypto_name: str, timeframe: str) -> str:
        """Build a prompt for sentiment analysis."""
        time_phrase = "past 24 hours" if timeframe == "24h" else f"past {timeframe}"
        
        prompt = f"""
        Please analyze the market sentiment for {crypto_name} ({symbol}) over the {time_phrase}.

        1. Search for and analyze recent credible news articles about {crypto_name} from major financial and cryptocurrency news sources
        2. Examine discussions about {search_symbol} and {crypto_name} on Reddit's cryptocurrency subreddits
        3. Review any significant Twitter/X discussions from crypto influencers and analysts about {crypto_name}
        4. Analyze any recent market reports or analyst opinions on {crypto_name}

        For your analysis, please:
        - Identify the overall sentiment (bullish, bearish, or neutral)
        - Quantify the sentiment on a scale from -1.0 (extremely bearish) to +1.0 (extremely bullish)
        - Determine your confidence level in this assessment (0.0 to 1.0)
        - Identify key factors driving the current sentiment
        - Note any significant events, news, or developments
        - Detect any sentiment shifts or trends
        - Analyze how the current sentiment might impact price action in the short term

        Provide your analysis in a structured JSON format with the following fields:
        - overall_sentiment: "bullish", "somewhat bullish", "neutral", "somewhat bearish", or "bearish"
        - sentiment_score: numerical score from -1.0 to 1.0
        - confidence: your confidence in this assessment from 0.0 to 1.0
        - key_factors: array of factors driving sentiment
        - significant_events: array of important events
        - sentiment_trends: description of how sentiment is shifting
        - price_impact: likely impact on price in short term
        - summary: brief summary of findings
        - sources: array of source types analyzed (e.g., "news", "reddit", "twitter", "analyst_reports")

        Respond ONLY with the JSON object and no additional text.
        """
        
        return prompt
    
    def _parse_sentiment_response(self, response, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Parse the response from the sentiment analysis model."""
        try:
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON if it's wrapped in markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            
            # Parse JSON response
            sentiment_data = json.loads(response_text)
            
            # Add metadata
            sentiment_data["symbol"] = symbol
            sentiment_data["timeframe"] = timeframe
            sentiment_data["timestamp"] = datetime.now().isoformat()
            sentiment_data["model_used"] = self.model
            
            return sentiment_data
            
        except json.JSONDecodeError:
            logger.error(f"Could not parse JSON from response: {response.choices[0].message.content}")
            
            # Attempt to extract sentiment heuristically if JSON parsing fails
            response_text = response.choices[0].message.content.lower()
            
            # Determine sentiment based on keywords
            sentiment = "neutral"
            if "bullish" in response_text:
                sentiment = "somewhat bullish" if "somewhat" in response_text else "bullish"
            elif "bearish" in response_text:
                sentiment = "somewhat bearish" if "somewhat" in response_text else "bearish"
            
            # Create a basic sentiment result
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "overall_sentiment": sentiment,
                "sentiment_score": 0.5 if sentiment == "bullish" else (-0.5 if sentiment == "bearish" else 0.0),
                "confidence": 0.5,
                "summary": "Error parsing structured response, basic sentiment extracted from text.",
                "model_used": self.model,
                "parse_error": True
            }
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a default neutral sentiment
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "model_used": self.model,
                "error": str(e)
            }

    async def get_detailed_sentiment_report(self, symbol: str, timeframes: List[str] = ["24h", "7d"]) -> Dict[str, Any]:
        """Get a detailed sentiment report across multiple timeframes.
        
        Args:
            symbol: The cryptocurrency symbol to analyze
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary containing detailed sentiment analysis across timeframes
        """
        try:
            # Analyze sentiment for each timeframe
            results = {}
            for timeframe in timeframes:
                results[timeframe] = await self.analyze_sentiment(symbol, timeframe)
            
            # Calculate trend and momentum
            trend = "stable"
            if len(timeframes) >= 2:
                short_score = results[timeframes[0]].get("sentiment_score", 0)
                long_score = results[timeframes[-1]].get("sentiment_score", 0)
                
                if short_score > long_score + 0.3:
                    trend = "rapidly improving"
                elif short_score > long_score + 0.1:
                    trend = "improving"
                elif short_score < long_score - 0.3:
                    trend = "rapidly deteriorating"
                elif short_score < long_score - 0.1:
                    trend = "deteriorating"
            
            # Compile the comprehensive report
            latest_sentiment = results[timeframes[0]]
            
            report = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "overall_sentiment": latest_sentiment.get("overall_sentiment", "neutral"),
                "sentiment_score": latest_sentiment.get("sentiment_score", 0.0),
                "confidence": latest_sentiment.get("confidence", 0.0),
                "sentiment_trend": trend,
                "timeframe_analysis": results,
                "summary": latest_sentiment.get("summary", ""),
                "key_factors": latest_sentiment.get("key_factors", []),
                "trading_insights": self._generate_trading_insights(latest_sentiment, trend),
                "model_used": self.model
            }
            
            logger.info(f"Generated detailed sentiment report for {symbol}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating detailed sentiment report for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "error": str(e),
                "model_used": self.model
            }
    
    def _generate_trading_insights(self, sentiment: Dict[str, Any], trend: str) -> Dict[str, Any]:
        """Generate trading insights based on sentiment analysis."""
        score = sentiment.get("sentiment_score", 0.0)
        confidence = sentiment.get("confidence", 0.0)
        
        # Determine trade action recommendation
        action = "hold"
        if score > 0.7 and confidence > 0.7:
            action = "strong buy"
        elif score > 0.3 and confidence > 0.5:
            action = "buy"
        elif score < -0.7 and confidence > 0.7:
            action = "strong sell"
        elif score < -0.3 and confidence > 0.5:
            action = "sell"
            
        # Determine position sizing recommendation (0.0-1.0)
        position_size = 0.0
        if action in ["buy", "strong buy"]:
            position_size = min(0.9, max(0.1, (abs(score) * confidence))) 
        
        # Determine stop loss and take profit recommendations
        stop_loss = 0.05  # Default 5%
        take_profit = 0.1  # Default 10%
        
        if action == "strong buy":
            stop_loss = 0.07
            take_profit = 0.15
        elif action == "buy":
            stop_loss = 0.05
            take_profit = 0.1
            
        # Generate insights
        insights = {
            "recommended_action": action,
            "confidence": confidence,
            "position_size_factor": float(f"{position_size:.2f}"),
            "suggested_stop_loss_pct": float(f"{stop_loss:.2f}"),
            "suggested_take_profit_pct": float(f"{take_profit:.2f}"),
            "market_momentum": trend,
            "entry_timing": "immediate" if action in ["strong buy"] else "gradual" if action == "buy" else "wait"
        }
        
        return insights


async def main():
    """Main function to test the sentiment analysis agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cryptocurrency Sentiment Analysis')
    parser.add_argument('--symbol', type=str, default='BTC/USD', 
                      help='Cryptocurrency symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='24h',
                      help='Timeframe for sentiment analysis (e.g., 24h, 7d)')
    parser.add_argument('--detailed', action='store_true',
                      help='Generate a detailed report across multiple timeframes')
    
    args = parser.parse_args()
    
    try:
        sentiment_agent = SentimentAnalysisAgent()
        
        print(f"\n📊 SENTIMENT ANALYSIS FOR {args.symbol}")
        print("="*50)
        
        if args.detailed:
            report = await sentiment_agent.get_detailed_sentiment_report(args.symbol, ["24h", "7d"])
            
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
            
        else:
            result = await sentiment_agent.analyze_sentiment(args.symbol, args.timeframe)
            
            print(f"\nTIMEFRAME: {args.timeframe}")
            print(f"SENTIMENT: {result['overall_sentiment'].upper()}")
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
            
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error running sentiment analysis: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 