#!/usr/bin/env python3
import os
import asyncio
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, Optional, List
import json
import traceback
import yfinance as yf

# Import OpenAI Agents SDK
from agents import Agent, ModelSettings, Runner, function_tool, RunContextWrapper
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI, OpenAI

# Import our existing functionality
from alpaca_crypto import AlpacaCryptoTrader, normalize_crypto_symbol
from crypto_indicators import prepare_analysis_data, generate_trading_signals
from crypto_agent import TradeAction
# Import the sentiment analysis agent
from sentiment_agent import SentimentAnalysisAgent

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_trading_loop.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("crypto_trading_loop")

class CryptoTradingBot:
    """A continuous cryptocurrency trading bot that makes decisions every 5 minutes."""
    
    def __init__(self, symbols=['BTC/USD'], interval_minutes=5, paper_trading=True, aggressive=True):
        """Initialize the trading bot.
        
        Args:
            symbols: List of cryptocurrency symbols to trade
            interval_minutes: How often to analyze and make decisions (in minutes)
            paper_trading: Whether to use paper trading (True) or live trading (False)
            aggressive: Whether to use aggressive position sizing for higher profits
        """
        self.symbols = symbols
        self.interval_minutes = interval_minutes
        self.paper_trading = paper_trading
        self.aggressive = aggressive
        self.trader = AlpacaCryptoTrader(paper=paper_trading)
        self.agent = None
        self.sentiment_agent = None
        self.running = False
        self.trade_history = []
        self.last_action = {}  # Track last action for each symbol
        self.portfolio_allocation = {}  # Track portfolio allocation percentages
        self.max_position_size = 0.50 if aggressive else 0.02  # 50% of portfolio per position if aggressive
        self.profit_target_pct = 0.15 if aggressive else 0.05  # 15% profit target if aggressive
        self.use_sentiment = True  # Flag to enable/disable sentiment analysis
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Loaded environment variables from .env file")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {str(e)}")
        
        # Check environment variables
        self._check_environment_variables()
        
        # Initialize trade history file
        self.trade_history_file = "trade_history.json"
        self._load_trade_history()
        
        logger.info(f"Initialized trading bot in {'AGGRESSIVE' if aggressive else 'CONSERVATIVE'} mode")
        if aggressive:
            logger.info(f"Max position size: {self.max_position_size*100}% of portfolio")
            logger.info(f"Profit target: {self.profit_target_pct*100}%")
        
    def _check_environment_variables(self):
        """Check if necessary environment variables are set."""
        if not os.environ.get('ALPACA_API_KEY_ID') or not os.environ.get('ALPACA_API_SECRET_KEY'):
            raise ValueError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables must be set")
        
        # Check if LiteLLM API key is set
        api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Either LITELLM_API_KEY or OPENAI_API_KEY environment variable must be set")
        else:
            # Use the API key from LITELLM_API_KEY if set, otherwise use OPENAI_API_KEY
            if os.environ.get("LITELLM_API_KEY"):
                logger.info("Using LITELLM_API_KEY for authentication")
                os.environ["OPENAI_API_KEY"] = os.environ.get("LITELLM_API_KEY")
    
    def _load_trade_history(self):
        """Load trading history from file if it exists."""
        try:
            if os.path.exists(self.trade_history_file):
                with open(self.trade_history_file, 'r') as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} trade records from history file")
        except Exception as e:
            logger.error(f"Error loading trade history: {str(e)}")
            self.trade_history = []
    
    def _save_trade_history(self):
        """Save trading history to file."""
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info(f"Saved {len(self.trade_history)} trade records to history file")
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
    
    def _record_trade(self, symbol, action, quantity, price, timestamp, reason):
        """Record a trade action in the history."""
        trade_record = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "reason": reason
        }
        self.trade_history.append(trade_record)
        self._save_trade_history()
        
        # Update last action
        self.last_action[symbol] = {
            "action": action,
            "price": price,
            "timestamp": timestamp,
            "quantity": quantity
        }
    
    @function_tool
    def fetch_crypto_data_yfinance(self, symbol: str, period: str = "30d", interval: str = "1d") -> Dict[str, Any]:
        """
        Fetch historical cryptocurrency data from Yahoo Finance.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., BTC-USD, ETH-USD)
            period: The time period to fetch data for (e.g., 1d, 7d, 30d, 90d, 1y, max)
            interval: The data interval (e.g., 1m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)
        
        Returns:
            Dictionary containing the historical price data and metadata
        """
        try:
            # Convert to Yahoo Finance format (e.g., BTC/USD -> BTC-USD)
            yf_symbol = symbol.replace('/', '-')
            
            logger.info(f"Fetching yfinance data for {yf_symbol} (period: {period}, interval: {interval})")
            
            # Fetch data from Yahoo Finance
            data = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            
            if data.empty:
                logger.warning(f"No data returned from yfinance for {yf_symbol}")
                return {"error": f"No data available for {symbol}"}
            
            # Reset index to make Date a column and convert to records format
            data.reset_index(inplace=True)
            
            # Convert timestamps to string format for JSON serialization
            data['Date'] = data['Date'].astype(str)
            
            # Convert DataFrame to list of dictionaries
            records = data.to_dict('records')
            
            logger.info(f"Successfully fetched {len(records)} records for {symbol}")
            
            # Calculate some basic statistics - safely handle Series objects
            latest_price_series = data['Close'].iloc[-1]
            latest_price = float(latest_price_series.iloc[0] if hasattr(latest_price_series, 'iloc') else latest_price_series)
            
            first_price_series = data['Close'].iloc[0]
            first_price = float(first_price_series.iloc[0] if hasattr(first_price_series, 'iloc') else first_price_series)
            
            price_change = ((latest_price - first_price) / first_price) * 100 if first_price > 0 else 0
            
            # Safely calculate volume change if volume data is available
            vol_change = 0
            if 'Volume' in data.columns and not data['Volume'].iloc[0] is None:
                try:
                    first_vol_series = data['Volume'].iloc[0]
                    first_volume = float(first_vol_series.iloc[0] if hasattr(first_vol_series, 'iloc') else first_vol_series)
                    
                    latest_vol_series = data['Volume'].iloc[-1]
                    latest_volume = float(latest_vol_series.iloc[0] if hasattr(latest_vol_series, 'iloc') else latest_vol_series)
                    
                    if first_volume > 0:
                        vol_change = ((latest_volume - first_volume) / first_volume) * 100
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not calculate volume change: {str(e)}")
            
            # Return data and statistics
            return {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data_points": len(records),
                "latest_price": latest_price,
                "price_change_pct": price_change,
                "volume_change_pct": vol_change,
                "start_date": data['Date'].iloc[0],
                "end_date": data['Date'].iloc[-1],
                "prices": records
            }
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Failed to fetch data: {str(e)}"}
    
    @function_tool
    def analyze_crypto_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cryptocurrency data and generate trading signals.
        
        Args:
            data: Dictionary containing historical price data from fetch_crypto_data_yfinance
            
        Returns:
            Dictionary containing analysis results and trading signals
        """
        try:
            if "error" in data:
                return {"error": data["error"]}
            
            symbol = data["symbol"]
            logger.info(f"Analyzing data for {symbol}")
            
            # Convert the price records back to a DataFrame
            df = pd.DataFrame(data["prices"])
            
            # Prepare dataframe for technical analysis
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            
            # Generate technical indicators
            analysis_df = prepare_analysis_data(df)
            
            # Generate trading signals
            signals = generate_trading_signals(analysis_df)
            
            # Extract key indicators for the latest data point
            latest = analysis_df.iloc[-1]
            
            # Create a summary of indicators
            indicators = {
                "ma_50": float(latest.get('MA_50', 0)),
                "ma_100": float(latest.get('MA_100', 0)),
                "ma_200": float(latest.get('MA_200', 0)),
                "rsi": float(latest.get('RSI', 0)),
                "macd": float(latest.get('MACD', 0)),
                "macd_signal": float(latest.get('MACD_Signal', 0)),
                "bollinger_upper": float(latest.get('BB_upper', 0)),
                "bollinger_lower": float(latest.get('BB_lower', 0)),
                "adx": float(latest.get('ADX', 0)),
                "plus_di": float(latest.get('Plus_DI', 0)),
                "minus_di": float(latest.get('Minus_DI', 0)),
                "obv": float(latest.get('OBV', 0)) if 'OBV' in latest else 0,
                "latest_close": float(latest.get('Close', 0))
            }
            
            # Extract trading signals
            signal_summary = {
                "ma_signal": signals['ma_signal'][-1] if len(signals['ma_signal']) > 0 else "neutral",
                "rsi_signal": signals['rsi_signal'][-1] if len(signals['rsi_signal']) > 0 else "neutral",
                "macd_signal": signals['macd_signal'][-1] if len(signals['macd_signal']) > 0 else "neutral",
                "bollinger_signal": signals['bollinger_signal'][-1] if len(signals['bollinger_signal']) > 0 else "neutral",
                "adx_signal": signals['adx_signal'][-1] if len(signals['adx_signal']) > 0 else "neutral",
                "overall_signal": signals['overall_signal'][-1] if len(signals['overall_signal']) > 0 else "neutral"
            }
            
            # Calculate support and resistance levels
            last_price = float(df['Close'].iloc[-1])
            pivot = (float(df['High'].iloc[-1]) + float(df['Low'].iloc[-1]) + float(df['Close'].iloc[-1])) / 3
            
            support_1 = round((2 * pivot) - float(df['High'].iloc[-1]), 2)
            support_2 = round(pivot - (float(df['High'].iloc[-1]) - float(df['Low'].iloc[-1])), 2)
            
            resistance_1 = round((2 * pivot) - float(df['Low'].iloc[-1]), 2)
            resistance_2 = round(pivot + (float(df['High'].iloc[-1]) - float(df['Low'].iloc[-1])), 2)
            
            levels = {
                "last_price": last_price,
                "pivot": float(pivot),
                "support_1": float(support_1),
                "support_2": float(support_2),
                "resistance_1": float(resistance_1),
                "resistance_2": float(resistance_2)
            }
            
            # Prepare the response with all analysis results
            return {
                "symbol": symbol,
                "indicators": indicators,
                "signals": signal_summary,
                "price_levels": levels,
                "overall_recommendation": signal_summary["overall_signal"].upper()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Failed to analyze data: {str(e)}"}
    
    @function_tool
    def analyze_portfolio(self) -> Dict[str, Any]:
        """
        Analyze the entire portfolio and provide recommendations for optimal asset allocation.
        
        Returns:
            Dictionary containing portfolio analysis and recommendations
        """
        try:
            # Get account information
            account = self.trader.get_account()
            total_equity = float(account.equity)
            cash = float(account.cash)
            
            # Get current positions
            positions = self.trader.get_positions()
            formatted_positions = []
            
            # Calculate total position value and allocation
            total_position_value = 0
            for pos in positions:
                position_value = float(pos.market_value)
                total_position_value += position_value
                
                # Calculate unrealized profit/loss
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                qty = float(pos.qty)
                unrealized_pl = (current_price - entry_price) * qty
                unrealized_pl_pct = (unrealized_pl / (entry_price * qty)) * 100 if entry_price * qty > 0 else 0
                
                formatted_positions.append({
                    "symbol": pos.symbol,
                    "quantity": float(pos.qty),
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "market_value": position_value,
                    "allocation_pct": (position_value / total_equity) * 100 if total_equity > 0 else 0,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_pl_pct": unrealized_pl_pct
                })
            
            # Sort positions by allocation percentage (descending)
            formatted_positions.sort(key=lambda x: x["allocation_pct"], reverse=True)
            
            # Calculate cash allocation
            cash_allocation = (cash / total_equity) * 100 if total_equity > 0 else 0
            
            # Calculate portfolio metrics
            portfolio_exposure = (total_position_value / total_equity) * 100 if total_equity > 0 else 0
            
            # Calculate portfolio diversification score (higher is more diversified)
            if len(formatted_positions) > 1:
                # Use Herfindahl-Hirschman Index (HHI) to measure concentration
                hhi = sum((pos["allocation_pct"] / 100) ** 2 for pos in formatted_positions)
                diversification_score = (1 - hhi) * 100  # Convert to percentage
            else:
                diversification_score = 0  # No diversification with 0-1 positions
            
            # Prepare the portfolio summary
            portfolio = {
                "total_equity": total_equity,
                "cash": cash,
                "cash_allocation_pct": cash_allocation,
                "portfolio_exposure_pct": portfolio_exposure,
                "diversification_score": diversification_score,
                "total_positions": len(formatted_positions),
                "positions": formatted_positions
            }
            
            # Determine if portfolio is overexposed to any asset
            max_allocation = self.max_position_size * 100  # Convert to percentage
            overexposed_assets = [
                pos["symbol"] for pos in formatted_positions 
                if pos["allocation_pct"] > max_allocation
            ]
            
            # Determine if there's too much cash
            excess_cash = cash_allocation > 30  # Flag if more than 30% in cash
            
            # Prepare portfolio recommendations
            recommendations = []
            
            # Check for overexposure
            if overexposed_assets:
                recommendations.append({
                    "type": "reduce_exposure",
                    "assets": overexposed_assets,
                    "reason": f"Assets exceed maximum allocation of {max_allocation:.1f}%"
                })
            
            # Check for cash utilization
            if excess_cash and self.aggressive:
                recommendations.append({
                    "type": "increase_exposure",
                    "allocation_available": cash,
                    "reason": f"Excess cash ({cash_allocation:.1f}%) could be deployed for higher returns"
                })
            
            # Check for diversification
            if diversification_score < 50 and len(formatted_positions) < 3 and self.aggressive:
                recommendations.append({
                    "type": "diversify",
                    "current_score": diversification_score,
                    "reason": "Portfolio concentration is high, consider adding uncorrelated assets"
                })
            
            # Add portfolio recommendations to the response
            portfolio["recommendations"] = recommendations
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Failed to analyze portfolio: {str(e)}"}
    
    async def initialize_agent(self):
        """Initialize the crypto trading agent."""
        try:
            logger.info("Initializing trading agent...")
            self.agent = await self._create_crypto_agent_with_tools()
            
            # Initialize sentiment analysis agent
            if self.use_sentiment:
                logger.info("Initializing sentiment analysis agent...")
                self.sentiment_agent = SentimentAnalysisAgent(model="sonar-pro")
            
            logger.info("Trading agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def _create_crypto_agent_with_tools(self):
        """Create a crypto trading agent with yfinance data fetching tools."""
        try:
            # Create model settings for the agent using Claude
            model_name = "claude-3-7-sonnet-latest"
            logger.info(f"Using model: {model_name}")
            
            # Define the API base URL
            api_base = os.environ.get("OPENAI_BASE_URL", "https://litellm.deriv.ai/v1")
            logger.info(f"Using API base URL: {api_base}")
            
            # Ensure API key is set in environment variables
            api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No API key found. Set either LITELLM_API_KEY or OPENAI_API_KEY in your environment.")
            
            # Set OPENAI_API_KEY from LITELLM_API_KEY if needed
            if os.environ.get("LITELLM_API_KEY"):
                os.environ["OPENAI_API_KEY"] = os.environ.get("LITELLM_API_KEY")
            
            # Initialize the OpenAI client with LiteLLM configuration
            external_client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
            )
            
            # Determine trading style based on aggressive setting
            trading_style = "aggressive, high-risk, high-reward" if self.aggressive else "balanced, moderate-risk"
            position_sizing = f"up to {int(self.max_position_size*100)}% of portfolio per position" if self.aggressive else "conservative (1-2% per trade)"
            
            # Create the agent with the model settings - following pattern from crypto_agent.py
            agent = Agent(
                name="CryptoTradingAgent",
                instructions=f"""
                You are an expert cryptocurrency trading AI, specialized in analyzing market data 
                and making informed trading decisions. Your role is to analyze cryptocurrency data, 
                identify trading opportunities, and provide clear buy, sell, or hold recommendations.
                
                TRADING STYLE: {trading_style}
                POSITION SIZING: {position_sizing}
                
                When analyzing data, you should:
                1. Use the fetch_crypto_data_yfinance tool to get historical data
                2. Use the analyze_crypto_data tool to calculate technical indicators and signals
                3. Use the get_market_sentiment tool to incorporate market sentiment into your analysis
                4. Consider multiple timeframes for a comprehensive view
                5. Evaluate risk-reward ratios for any potential trades
                6. Set appropriate stop-loss and price targets
                7. Don't be afraid to recommend larger position sizes when confidence is high
                
                MARKET SENTIMENT ANALYSIS:
                - Always use the get_market_sentiment tool to incorporate news and social media sentiment
                - Give more weight to sentiment when the confidence score is high (>0.7)
                - Look for sentiment trends (improving, deteriorating) as early signals
                - When sentiment and technical analysis align, increase your confidence
                - When they conflict, explain the discrepancy and which you trust more
                
                Specifically, your portfolio management approach should be:
                - Manage the entire account portfolio across all symbols
                - Allocate up to {int(self.max_position_size*100)}% of portfolio to high-conviction trades
                - Target {int(self.profit_target_pct*100)}% profit per trade
                - Scale in/out of positions rather than using all-or-nothing trades
                - Consider market correlation between crypto assets
                
                Your final output should be a TradeAction object with:
                - action: "buy", "sell", or "hold"
                - confidence: a decimal between 0 and 1 indicating confidence level
                - reason: detailed explanation for your recommendation
                - suggested_quantity: recommended quantity to trade (can be aggressive for high-confidence trades)
                """,
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(temperature=0.2),
                output_type=TradeAction,
                tools=[
                    self.fetch_crypto_data_yfinance,
                    self.analyze_crypto_data,
                    self.analyze_portfolio,
                    self.get_portfolio_allocation,
                    self.get_market_sentiment
                ]
            )
            
            return agent
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @function_tool
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """Get the current portfolio allocation across assets."""
        try:
            return self._get_portfolio_allocation_internal()
        except Exception as e:
            logger.error(f"Error getting portfolio allocation: {str(e)}")
            return {"error": str(e)}
    
    def _get_portfolio_allocation_internal(self) -> Dict[str, float]:
        """Internal method to get portfolio allocation that can be called directly."""
        try:
            # Get account information
            account = self.trader.get_account()
            total_equity = float(account.equity)
            
            # Get current positions
            positions = self.trader.get_positions()
            
            # Calculate allocation
            allocation = {}
            for pos in positions:
                symbol = pos.symbol
                market_value = float(pos.market_value)
                percentage = (market_value / total_equity) * 100 if total_equity > 0 else 0
                allocation[symbol] = percentage
            
            # Calculate cash percentage
            cash = float(account.cash)
            cash_percentage = (cash / total_equity) * 100 if total_equity > 0 else 0
            allocation["CASH"] = cash_percentage
            
            # Store for later reference
            self.portfolio_allocation = allocation
            
            return allocation
        except Exception as e:
            logger.error(f"Error getting portfolio allocation: {str(e)}")
            return {"error": str(e)}
    
    @function_tool
    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get market sentiment analysis for a cryptocurrency symbol.
        
        Args:
            symbol: The cryptocurrency symbol to analyze (e.g., BTC/USD)
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            # This is a synchronous function tool wrapper around the async sentiment analysis
            sentiment_result = asyncio.run(self._get_market_sentiment_async(symbol))
            return sentiment_result
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            return {
                "symbol": symbol,
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _get_market_sentiment_async(self, symbol: str) -> Dict[str, Any]:
        """Async implementation of market sentiment analysis."""
        if not self.sentiment_agent or not self.use_sentiment:
            logger.warning("Sentiment analysis agent not initialized or disabled")
            return {
                "symbol": symbol,
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "not_available": True
            }
        
        try:
            logger.info(f"Getting market sentiment for {symbol}...")
            report = await self.sentiment_agent.get_detailed_sentiment_report(symbol, ["24h", "7d"])
            
            # Log sentiment results
            logger.info(f"Sentiment for {symbol}: {report['overall_sentiment']} (score: {report['sentiment_score']:.2f}, confidence: {report['confidence']:.2f})")
            
            # Return a simplified version for the agent to use
            return {
                "symbol": symbol,
                "overall_sentiment": report["overall_sentiment"],
                "sentiment_score": report["sentiment_score"],
                "confidence": report["confidence"],
                "sentiment_trend": report["sentiment_trend"],
                "key_factors": report.get("key_factors", [])[:3],
                "trading_insights": report["trading_insights"],
                "summary": report.get("summary", "")
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "symbol": symbol,
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def analyze_and_trade(self, symbol):
        """Analyze market data and execute trades for a specific symbol."""
        if not self.agent:
            logger.error("Agent not initialized. Cannot analyze or trade.")
            return
            
        try:
            # Get account information for position sizing
            account_info = await self._get_account_info()
            if not account_info:
                logger.error("Failed to get account information")
                return
                
            # Get current positions
            positions = await self._get_positions()
            portfolio_allocation = self._get_portfolio_allocation_internal()
            
            # Check if we already have a position in this symbol
            current_position = next((pos for pos in positions if pos["symbol"] == symbol.replace("/", "")), None)
            position_size = float(current_position["qty"]) if current_position else 0
            position_value = float(current_position["market_value"]) if current_position else 0
            position_pct = (position_value / account_info["equity"]) * 100 if account_info["equity"] > 0 else 0
            
            # Determine max allocation for this symbol based on confidence
            max_allocation = account_info["equity"] * self.max_position_size
            
            # Get current market data
            logger.info(f"Fetching market data for {symbol}...")
            
            # Get market sentiment if available
            if self.use_sentiment and self.sentiment_agent:
                try:
                    logger.info(f"Analyzing market sentiment for {symbol}...")
                    sentiment_data = await self._get_market_sentiment_async(symbol)
                    sentiment_info = (
                        f"Market Sentiment: {sentiment_data['overall_sentiment'].upper()}\n"
                        f"Sentiment Score: {sentiment_data['sentiment_score']:.2f} (confidence: {sentiment_data['confidence']:.2f})\n"
                        f"Sentiment Trend: {sentiment_data['sentiment_trend']}\n"
                        f"Key factors: {', '.join(sentiment_data.get('key_factors', []))[:150]}...\n"
                    )
                    
                    # Print sentiment information
                    print(f"\n📊 SENTIMENT ANALYSIS FOR {symbol}")
                    print(f"SENTIMENT: {sentiment_data['overall_sentiment'].upper()} ({sentiment_data['sentiment_score']:.2f})")
                    print(f"TREND: {sentiment_data['sentiment_trend'].upper()}")
                    if 'trading_insights' in sentiment_data:
                        insights = sentiment_data['trading_insights']
                        print(f"SUGGESTED ACTION: {insights['recommended_action'].upper()}")
                        
                except Exception as e:
                    logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
                    sentiment_info = "Market Sentiment: Not available due to error\n"
            else:
                sentiment_info = "Market Sentiment: Analysis not enabled\n"
            
            # Comprehensive query that ensures data fetching happens first
            user_query = (
                f"You are a professional cryptocurrency trader analyzing {symbol}. Follow these steps in order:\n\n"
                f"1. FIRST use the fetch_crypto_data_yfinance tool to fetch data for these timeframes:\n"
                f"   - Short-term: 7d with 1h interval\n"
                f"   - Medium-term: 30d with 1d interval\n"
                f"   - Long-term: 90d with 1d interval\n\n"
                f"2. THEN use the get_market_sentiment tool to get the latest market sentiment for {symbol}.\n\n"
                f"3. THEN analyze each dataset thoroughly using the analyze_crypto_data tool.\n\n"
                f"4. FINALLY, based on your complete analysis of both technical indicators and market sentiment,\n"
                f"   decide whether to buy, sell, or hold {symbol}.\n\n"
                f"Provide a comprehensive trading recommendation with:\n"
                f"- Multi-timeframe analysis of key indicators (RSI, MACD, EMAs, Bollinger Bands)\n"
                f"- Support and resistance levels\n"
                f"- Volume analysis\n"
                f"- How market sentiment influences your decision\n"
                f"- Specific entry/exit points and stop-loss levels\n"
                f"- Take-profit targets with specific price levels\n"
                f"- Risk-reward ratio for any trades\n"
                f"- Clear reasoning for your final decision\n\n"
                f"{sentiment_info}\n"
                f"Current portfolio allocation: {portfolio_allocation}\n"
                f"Current position size: {position_size} units (${position_value:.2f}, {position_pct:.2f}% of portfolio).\n"
                f"Account equity: ${account_info['equity']}.\n"
                f"Maximum allocation permitted: ${max_allocation:.2f} ({self.max_position_size*100:.0f}% of equity)\n\n"
                f"IMPORTANT: Complete all analysis steps before making your final recommendation.\n"
                f"Provide detailed reasoning based on technical analysis, market sentiment, and risk management principles.\n"
                f"ALWAYS include specific stop-loss and take-profit levels in your recommendation."
            )
            
            logger.info(f"Running comprehensive analysis for {symbol}...")
            result = await Runner.run(self.agent, user_query, context=self.trader)
            
            if not result or not hasattr(result, 'final_output'):
                logger.error(f"No result returned from agent for {symbol}")
                return
            
            trade_action = result.final_output
            
            # Validate that we have a proper recommendation, not just a planning message
            if not trade_action.reason or len(trade_action.reason) < 50 or "need to" in trade_action.reason.lower():
                logger.warning(f"Received incomplete reasoning from agent: {trade_action.reason}")
                # Try again with a more explicit request
                retry_query = (
                    f"You have already fetched and analyzed the data for {symbol}. Based on that data,\n"
                    f"please provide your FINAL recommendation (buy, sell, or hold) with detailed reasoning.\n"
                    f"Your reasoning should include technical analysis findings from multiple timeframes,\n"
                    f"sentiment analysis insights, support/resistance levels, and risk management considerations.\n"
                    f"MUST include specific stop-loss price and take-profit target price.\n"
                    f"DO NOT say you need to analyze the data first - you've already done that step."
                )
                retry_result = await Runner.run(self.agent, retry_query, context=self.trader)
                if retry_result and hasattr(retry_result, 'final_output'):
                    trade_action = retry_result.final_output
            
            logger.info(f"Agent recommendation for {symbol}: {trade_action.action}")
            logger.info(f"Confidence: {trade_action.confidence:.2f}")
            logger.info(f"Reasoning: {trade_action.reason}")
            
            # Extract stop-loss and take-profit levels from the reasoning if not explicitly provided
            stop_loss_price = trade_action.stop_loss
            take_profit_price = trade_action.price_target
            
            if not stop_loss_price or not take_profit_price:
                # Try to extract from the reasoning text
                reasoning_lower = trade_action.reason.lower()
                
                # Extract stop-loss
                if not stop_loss_price:
                    stop_loss_patterns = [
                        r"stop[ -]?loss:?\s*\$?([0-9,.]+)",
                        r"stop[ -]?loss\s*(?:price|level):?\s*\$?([0-9,.]+)",
                        r"place\s*(?:a)?\s*stop[ -]?loss\s*at\s*\$?([0-9,.]+)"
                    ]
                    for pattern in stop_loss_patterns:
                        import re
                        match = re.search(pattern, reasoning_lower)
                        if match:
                            try:
                                stop_loss_price = float(match.group(1).replace(',', ''))
                                logger.info(f"Extracted stop-loss price from reasoning: ${stop_loss_price}")
                                break
                            except (ValueError, IndexError):
                                pass
                
                # Extract take-profit
                if not take_profit_price:
                    take_profit_patterns = [
                        r"take[ -]?profit:?\s*\$?([0-9,.]+)",
                        r"take[ -]?profit\s*(?:price|level|target):?\s*\$?([0-9,.]+)",
                        r"price\s*target:?\s*\$?([0-9,.]+)",
                        r"target\s*price:?\s*\$?([0-9,.]+)"
                    ]
                    for pattern in take_profit_patterns:
                        match = re.search(pattern, reasoning_lower)
                        if match:
                            try:
                                take_profit_price = float(match.group(1).replace(',', ''))
                                logger.info(f"Extracted take-profit price from reasoning: ${take_profit_price}")
                                break
                            except (ValueError, IndexError):
                                pass
            
            # Get current price for the symbol using yfinance
            current_price = await self._get_current_price_yfinance(symbol)
            if not current_price:
                logger.error(f"Failed to get current price for {symbol}")
                return
                
            # Validate stop-loss and take-profit based on action and current price
            if trade_action.action.lower() == "buy":
                # For buy actions, stop-loss should be below current price and take-profit above
                if stop_loss_price and stop_loss_price >= current_price:
                    logger.warning(f"Invalid stop-loss (${stop_loss_price}) for buy action - must be below current price (${current_price})")
                    # Set a default stop-loss 5% below current price
                    stop_loss_price = round(current_price * 0.95, 2)
                    logger.info(f"Using default stop-loss at ${stop_loss_price} (5% below current price)")
                    
                if take_profit_price and take_profit_price <= current_price:
                    logger.warning(f"Invalid take-profit (${take_profit_price}) for buy action - must be above current price (${current_price})")
                    # Set a default take-profit 10% above current price
                    take_profit_price = round(current_price * 1.10, 2)
                    logger.info(f"Using default take-profit at ${take_profit_price} (10% above current price)")
            
            elif trade_action.action.lower() == "sell":
                # For sell actions, stop-loss should be above current price and take-profit below (if provided)
                if stop_loss_price and stop_loss_price <= current_price:
                    logger.warning(f"Invalid stop-loss (${stop_loss_price}) for sell action - must be above current price (${current_price})")
                    # Set a default stop-loss 5% above current price
                    stop_loss_price = round(current_price * 1.05, 2)
                    logger.info(f"Using default stop-loss at ${stop_loss_price} (5% above current price)")
                    
                if take_profit_price and take_profit_price >= current_price:
                    logger.warning(f"Invalid take-profit (${take_profit_price}) for sell action - must be below current price (${current_price})")
                    # Set a default take-profit 10% below current price
                    take_profit_price = round(current_price * 0.90, 2)
                    logger.info(f"Using default take-profit at ${take_profit_price} (10% below current price)")
            
            # Execute the recommended action
            if trade_action.action.lower() == "buy":
                # Determine position size using risk management and confidence
                if trade_action.suggested_quantity:
                    qty = trade_action.suggested_quantity
                else:
                    # Scale position size based on confidence
                    confidence_factor = trade_action.confidence * 2 if self.aggressive else trade_action.confidence
                    # Cap at max allocation percentage
                    allocation_pct = min(confidence_factor * self.max_position_size, self.max_position_size)
                    
                    # Calculate position size
                    allocation_amount = account_info["equity"] * allocation_pct
                    # If we already have a position, calculate additional amount to buy
                    additional_amount = max(0, allocation_amount - position_value)
                    
                    # Calculate quantity based on current price
                    qty = additional_amount / current_price if current_price > 0 else 0
                    
                # Only execute if qty is significant
                if qty > 0.0001:  # Minimum quantity threshold
                    # Execute the buy order with stop-loss and take-profit
                    logger.info(f"Executing BUY order for {symbol} - {qty} units at ~${current_price}")
                    logger.info(f"Stop-loss: ${stop_loss_price}, Take-profit: ${take_profit_price}")
                    
                    try:
                        order_result = await self._execute_trade(
                            symbol=symbol, 
                            action="buy", 
                            qty=qty,
                            stop_loss=stop_loss_price,
                            take_profit=take_profit_price
                        )
                        
                        if order_result and "error" not in order_result:
                            logger.info(f"BUY order executed: {order_result}")
                            self._record_trade(
                                symbol=symbol,
                                action="buy",
                                quantity=qty,
                                price=current_price,
                                timestamp=datetime.now().isoformat(),
                                reason=trade_action.reason
                            )
                            # Notify
                            print(f"\n🟢 BUY EXECUTED: {qty} units of {symbol} at ${current_price}")
                            print(f"Position value: ${(position_value + (qty * current_price)):.2f}")
                            print(f"Confidence: {trade_action.confidence:.2f}")
                            if stop_loss_price:
                                print(f"Stop-loss: ${stop_loss_price} ({((stop_loss_price / current_price) - 1) * 100:.1f}%)")
                            if take_profit_price:
                                print(f"Take-profit: ${take_profit_price} ({((take_profit_price / current_price) - 1) * 100:.1f}%)")
                            print(f"Reason: {trade_action.reason[:150]}...")
                        else:
                            logger.error(f"Failed to execute BUY order: {order_result}")
                    except Exception as e:
                        logger.error(f"Error executing BUY order: {str(e)}")
                else:
                    logger.info(f"Buy quantity too small ({qty}), skipping trade")
            
            elif trade_action.action.lower() == "sell":
                # Only sell if we have a position
                if position_size > 0:
                    # Determine quantity to sell based on confidence
                    if trade_action.suggested_quantity:
                        qty = min(trade_action.suggested_quantity, position_size)
                    else:
                        # Scale sell percentage based on confidence
                        sell_percentage = trade_action.confidence
                        qty = position_size * sell_percentage
                    
                    # Execute the sell order
                    logger.info(f"Executing SELL order for {symbol} - {qty} units at ~${current_price}")
                    
                    try:
                        order_result = await self._execute_trade(
                            symbol=symbol, 
                            action="sell", 
                            qty=qty
                        )
                        if order_result and "error" not in order_result:
                            logger.info(f"SELL order executed: {order_result}")
                            self._record_trade(
                                symbol=symbol,
                                action="sell",
                                quantity=qty,
                                price=current_price,
                                timestamp=datetime.now().isoformat(),
                                reason=trade_action.reason
                            )
                            # Notify
                            print(f"\n🔴 SELL EXECUTED: {qty} units of {symbol} at ${current_price}")
                            print(f"Remaining position: {position_size - qty} units")
                            print(f"Confidence: {trade_action.confidence:.2f}")
                            print(f"Reason: {trade_action.reason[:150]}...")
                        else:
                            logger.error(f"Failed to execute SELL order: {order_result}")
                    except Exception as e:
                        logger.error(f"Error executing SELL order: {str(e)}")
                else:
                    logger.info(f"No position in {symbol} to sell")
            
            else:  # Hold
                # Record the hold decision for tracking
                self._record_trade(
                    symbol=symbol,
                    action="hold",
                    quantity=0,
                    price=current_price,
                    timestamp=datetime.now().isoformat(),
                    reason=trade_action.reason
                )
                # Notify of hold decision
                if position_size > 0:
                    print(f"\n🔵 HOLD DECISION: Maintaining position of {position_size} units of {symbol} at ${current_price}")
                    print(f"Position value: ${position_value:.2f} ({position_pct:.2f}% of portfolio)")
                    print(f"Reason: {trade_action.reason[:150]}...")
                else:
                    print(f"\n⚪ NO POSITION: Continuing to monitor {symbol} at ${current_price}")
                    print(f"Reason: {trade_action.reason[:150]}...")
            
            # Return the action taken
            return {
                "symbol": symbol,
                "action": trade_action.action,
                "confidence": trade_action.confidence,
                "price": current_price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "reason": trade_action.reason
            }
            
        except Exception as e:
            logger.error(f"Error analyzing and trading {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def _get_account_info(self):
        """Get account information from Alpaca."""
        try:
            account = self.trader.get_account()
            return {
                "account_id": account.id,
                "status": account.status,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power)
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None
    
    async def _get_positions(self):
        """Get current positions from Alpaca."""
        try:
            positions = self.trader.get_positions()
            formatted_positions = []
            for pos in positions:
                formatted_positions.append({
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc) * 100
                })
            return formatted_positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    
    async def _get_current_price_yfinance(self, symbol):
        """Get the current price for a symbol using yfinance."""
        try:
            # Convert to Yahoo Finance format
            yf_symbol = symbol.replace('/', '-')
            
            logger.info(f"Getting current price for {yf_symbol} from yfinance")
            
            # Fetch the latest data (1 day)
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period="1d")
            
            if data.empty:
                logger.error(f"No price data available for {symbol} from yfinance")
                return None
            
            # Get the most recent closing price and safely convert to float
            if 'Close' in data.columns and len(data) > 0:
                close_series = data['Close'].iloc[-1]
                current_price = float(close_series.iloc[0] if hasattr(close_series, 'iloc') else close_series)
                logger.info(f"Current price for {symbol}: ${current_price:.2f}")
                return current_price
            else:
                logger.error(f"No Close column found in data for {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol} from yfinance: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    async def _execute_trade(self, symbol, action, qty=None, notional=None, stop_loss=None, take_profit=None):
        """Execute a trade on Alpaca with optional stop-loss and take-profit orders.
        
        Args:
            symbol: The cryptocurrency symbol to trade
            action: The trade action ("buy" or "sell")
            qty: The quantity to trade (optional)
            notional: The notional value to trade (optional)
            stop_loss: Stop-loss price level (optional)
            take_profit: Take-profit price level (optional)
            
        Returns:
            Dictionary with order details or error message
        """
        try:
            if action.lower() not in ["buy", "sell"]:
                return f"Error: Invalid action '{action}'. Must be 'buy' or 'sell'."
            
            if not (qty or notional):
                return "Error: Either quantity (qty) or notional value must be provided."
            
            # For buy orders with stop_loss or take_profit, use bracket orders
            if action.lower() == "buy" and (stop_loss or take_profit):
                try:
                    logger.info(f"Creating bracket order for {symbol} - {qty} units")
                    logger.info(f"Stop-loss: ${stop_loss}, Take-profit: ${take_profit}")
                    
                    # Create a bracket order following Alpaca's documentation
                    order = self.trader.submit_crypto_order(
                        symbol=symbol,
                        side=action.lower(),
                        qty=qty,
                        notional=notional,
                        order_type="market",
                        time_in_force="gtc",
                        order_class="bracket",
                        take_profit={
                            "limit_price": take_profit
                        } if take_profit else None,
                        stop_loss={
                            "stop_price": stop_loss
                        } if stop_loss else None
                    )
                    
                    # Return order details for bracket order
                    order_details = {
                        "order_id": order.id,
                        "symbol": order.symbol,
                        "side": order.side.name,
                        "type": order.type.name,
                        "qty": order.qty,
                        "status": order.status.name,
                        "created_at": str(order.created_at),
                        "order_class": "bracket",
                        "stop_loss_price": stop_loss,
                        "take_profit_price": take_profit
                    }
                    
                    logger.info(f"Bracket order created: {order.id}")
                    return order_details
                    
                except Exception as e:
                    logger.error(f"Error creating bracket order: {str(e)}")
                    
                    # Fall back to regular order if bracket order fails
                    logger.info(f"Falling back to regular order for {symbol}")
                    return await self._execute_simple_order(symbol, action, qty, notional)
            
            # For sell orders or orders without stop-loss/take-profit, use simple orders
            return await self._execute_simple_order(symbol, action, qty, notional)
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _execute_simple_order(self, symbol, action, qty=None, notional=None):
        """Execute a simple market order without stop-loss or take-profit."""
        try:
            # Execute the primary order
            order = self.trader.submit_crypto_order(
                symbol=symbol,
                side=action.lower(),
                qty=qty,
                notional=notional,
                order_type="market",
                time_in_force="gtc"
            )
            
            # Return order details
            return {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.name,
                "type": order.type.name,
                "qty": order.qty,
                "status": order.status.name,
                "created_at": str(order.created_at),
                "order_class": "simple"
            }
        except Exception as e:
            logger.error(f"Error executing simple order: {str(e)}")
            return f"Error: {str(e)}"
    
    async def display_portfolio_status(self):
        """Display current portfolio status and performance."""
        try:
            account_info = await self._get_account_info()
            positions = await self._get_positions()
            
            print("\n" + "="*50)
            print(f"PORTFOLIO STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*50)
            
            print(f"\nAccount Equity: ${account_info['equity']:,.2f}")
            print(f"Cash Balance: ${account_info['cash']:,.2f}")
            print(f"Buying Power: ${account_info['buying_power']:,.2f}")
            
            if positions:
                print("\nOpen Positions:")
                print("-"*50)
                for pos in positions:
                    pl_emoji = "🟢" if pos["unrealized_pl"] >= 0 else "🔴"
                    print(f"{pl_emoji} {pos['symbol']}: {float(pos['qty']):,.8f} units @ ${pos['avg_entry_price']:,.2f}")
                    print(f"   Current Price: ${pos['current_price']:,.2f}")
                    print(f"   Market Value: ${pos['market_value']:,.2f}")
                    print(f"   P&L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:,.2f}%)")
                    print("-"*50)
            else:
                print("\nNo open positions.")
            
            # Calculate trading statistics
            if self.trade_history:
                buys = [t for t in self.trade_history if t["action"] == "buy"]
                sells = [t for t in self.trade_history if t["action"] == "sell"]
                holds = [t for t in self.trade_history if t["action"] == "hold"]
                
                print("\nTrading Statistics:")
                print(f"Total Trades: {len(buys) + len(sells)}")
                print(f"Buy Decisions: {len(buys)}")
                print(f"Sell Decisions: {len(sells)}")
                print(f"Hold Decisions: {len(holds)}")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Error displaying portfolio status: {str(e)}")
    
    async def run(self):
        """Run the trading loop at specified intervals."""
        # Initialize the agent
        if not await self.initialize_agent():
            logger.error("Failed to initialize agent. Cannot start trading loop.")
            return
        
        self.running = True
        logger.info(f"Starting trading loop for symbols: {self.symbols}")
        logger.info(f"Trading interval: {self.interval_minutes} minutes")
        logger.info(f"Paper trading mode: {self.paper_trading}")
        
        try:
            while self.running:
                cycle_start_time = time.time()
                
                logger.info(f"Starting trading cycle at {datetime.now()}")
                
                # Display portfolio status at the beginning of each cycle
                await self.display_portfolio_status()
                
                # First analyze the portfolio as a whole
                try:
                    logger.info("Analyzing complete portfolio...")
                    portfolio_query = (
                        "First analyze the entire portfolio using the analyze_portfolio tool.\n"
                        "Then provide portfolio-wide recommendations for optimal asset allocation.\n"
                        "What assets should be increased or decreased? What's our current exposure?\n"
                        "Finally, summarize the overall portfolio strategy given current market conditions."
                    )
                    portfolio_result = await Runner.run(self.agent, portfolio_query, context=self.trader)
                    if portfolio_result and hasattr(portfolio_result, 'final_output'):
                        logger.info("Portfolio analysis completed")
                        # This output is in TradeAction format but contains portfolio overview
                        portfolio_action = portfolio_result.final_output
                        print("\n📊 PORTFOLIO ANALYSIS")
                        print(f"Strategy: {portfolio_action.action}")
                        print(f"Rationale: {portfolio_action.reason[:300]}...")
                except Exception as e:
                    logger.error(f"Error in portfolio analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        logger.info(f"Processing symbol: {symbol}")
                        result = await self.analyze_and_trade(symbol)
                        if result:
                            logger.info(f"Trading cycle completed for {symbol}: {result['action']}")
                    except Exception as e:
                        logger.error(f"Error processing symbol {symbol}: {str(e)}")
                        logger.error(traceback.format_exc())
                
                # Calculate time to wait until next cycle
                elapsed_time = time.time() - cycle_start_time
                wait_time = max(0, self.interval_minutes * 60 - elapsed_time)
                
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.2f} seconds until next trading cycle")
                    await asyncio.sleep(wait_time)
                
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
            self.running = False
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            logger.error(traceback.format_exc())
            self.running = False
        finally:
            logger.info("Trading loop stopped")
            # Final portfolio status
            await self.display_portfolio_status()

async def main():
    """Main function to start the trading bot."""
    # Load command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Automated Cryptocurrency Trading Bot')
    parser.add_argument('--symbols', type=str, default='BTC/USD', 
                      help='Comma-separated cryptocurrency symbols to trade')
    parser.add_argument('--interval', type=int, default=5,
                      help='Trading interval in minutes')
    parser.add_argument('--paper', action='store_true', default=True,
                      help='Use paper trading mode (default: True)')
    parser.add_argument('--live', action='store_true',
                      help='Use live trading (overrides --paper)')
    parser.add_argument('--aggressive', action='store_true', default=False,
                      help='Use aggressive position sizing (up to 50% per position)')
    parser.add_argument('--no-sentiment', action='store_true', default=False,
                      help='Disable sentiment analysis')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Determine trading mode
    paper_trading = not args.live
    
    # Print banner
    print("\n" + "="*70)
    print(f"  CRYPTO TRADING BOT - {'PAPER TRADING' if paper_trading else 'LIVE TRADING'}")
    print(f"  Mode: {'AGGRESSIVE' if args.aggressive else 'CONSERVATIVE'}")
    print(f"  Sentiment Analysis: {'DISABLED' if args.no_sentiment else 'ENABLED'}")
    print("="*70)
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Interval: {args.interval} minutes")
    print(f"  Started at: {datetime.now()}")
    print("="*70 + "\n")
    
    if not paper_trading:
        print("\n⚠️  WARNING: LIVE TRADING MODE ENABLED ⚠️")
        print("This will place REAL trades with REAL money.")
        confirmation = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirmation != "CONFIRM":
            print("Live trading not confirmed. Exiting.")
            return
    
    if args.aggressive:
        print("\n🔥 AGGRESSIVE MODE ENABLED 🔥")
        print("This will use up to 50% of your portfolio for a single position.")
        print("Profit targets and risk tolerance are increased.")
        if not paper_trading:
            confirmation = input("Type 'AGGRESSIVE' to confirm you understand the higher risk: ")
            if confirmation != "AGGRESSIVE":
                print("Aggressive mode not confirmed. Exiting.")
                return
    
    # Create and run the trading bot
    bot = CryptoTradingBot(
        symbols=symbols,
        interval_minutes=args.interval,
        paper_trading=paper_trading,
        aggressive=args.aggressive
    )
    
    # Set sentiment analysis flag
    bot.use_sentiment = not args.no_sentiment
    
    await bot.run()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 