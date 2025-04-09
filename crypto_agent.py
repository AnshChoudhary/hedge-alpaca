#!/usr/bin/env python3
import os
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

# Import OpenAI Agents SDK
from agents import Agent, ModelSettings, Runner, function_tool, RunContextWrapper

# Import our existing Alpaca crypto trading functionality
from alpaca_crypto import AlpacaCryptoTrader, normalize_crypto_symbol

# Import our technical indicators
from crypto_indicators import (
    prepare_analysis_data,
    generate_trading_signals,
    calculate_sma,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    detect_support_resistance,
    calculate_fibonacci_retracement
)

# Additional imports for the updated approach
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_agent.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Configure LiteLLM settings - use the correct environment variables
os.environ["OPENAI_BASE_URL"] = "https://litellm.deriv.ai/v1"
# Make sure we're not using the default OpenAI API URL
if "OPENAI_API_BASE" in os.environ:
    del os.environ["OPENAI_API_BASE"]

# Additional debugging to verify settings
logger.info(f"Using API base URL: {os.environ.get('OPENAI_BASE_URL')}")
logger.info(f"Using model: claude-3-7-sonnet-latest")

# Define model classes for our agent output
class TradeAction(BaseModel):
    symbol: str = Field(..., description="The cryptocurrency symbol (e.g., BTC/USD)")
    action: str = Field(..., description="The trade action to take (buy, sell, hold)")
    reason: str = Field(..., description="Explanation for the trade decision")
    confidence: float = Field(..., description="Confidence level in the decision (0-1)")
    suggested_quantity: Optional[float] = Field(None, description="Suggested quantity to trade if applicable")
    price_target: Optional[float] = Field(None, description="Target price for the trade")
    stop_loss: Optional[float] = Field(None, description="Suggested stop loss price")

# Define function tools

@function_tool
async def get_crypto_data_from_yahoo(symbol: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetch historical cryptocurrency data from Yahoo Finance.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., 'BTC-USD' for Bitcoin)
        period: The time period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: The data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        JSON string of historical data including: date, open, high, low, close, volume
    """
    try:
        # Ensure the symbol is in Yahoo Finance format (BTC-USD instead of BTC/USD)
        if '/' in symbol:
            symbol = symbol.replace('/', '-')
        
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period, interval=interval)
        
        # Reset index to make date a column and convert to json
        history_reset = history.reset_index()
        # Convert date column to string to make it JSON serializable
        history_reset['Date'] = history_reset['Date'].astype(str)
        
        # Convert to records format and then to JSON string
        result = history_reset.to_dict(orient='records')
        
        logger.info(f"Successfully retrieved {len(result)} data points for {symbol}")
        return result
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        return f"Error: {str(e)}"

@function_tool
async def get_crypto_data_from_alpaca(ctx: RunContextWrapper[AlpacaCryptoTrader], symbol: str, days: int = 30, timeframe: str = "1D") -> str:
    """
    Fetch historical cryptocurrency data from Alpaca.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., 'BTC/USD' for Bitcoin)
        days: Number of days of historical data to fetch
        timeframe: The data timeframe (1m, 1h, 1d)
        
    Returns:
        JSON string of historical data including: date, open, high, low, close, volume
    """
    try:
        trader = ctx.context
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Normalize the symbol format
        normalized = normalize_crypto_symbol(symbol)
        
        # Get the bar data from Alpaca
        bars = trader.get_crypto_bars([normalized["with_slash"]], timeframe, start=start_date, end=end_date)
        
        if bars.empty:
            return f"No data available for {symbol}"
        
        # Check if symbol is in the index and extract the relevant data
        if 'symbol' in bars.index.names:
            available_symbols = bars.index.get_level_values('symbol').unique()
            found_symbol = None
            
            for available in available_symbols:
                if available == normalized["with_slash"] or available == normalized["without_slash"]:
                    found_symbol = available
                    break
            
            if found_symbol:
                symbol_data = bars.xs(found_symbol, level='symbol')
            else:
                symbol_data = bars
        else:
            symbol_data = bars
        
        # Reset index to make date a column and convert to json
        symbol_data_reset = symbol_data.reset_index()
        # Convert timestamp to string to make it JSON serializable
        symbol_data_reset['timestamp'] = symbol_data_reset['timestamp'].astype(str)
        
        # Convert to records format
        result = symbol_data_reset.to_dict(orient='records')
        
        logger.info(f"Successfully retrieved {len(result)} data points for {symbol} from Alpaca")
        return result
    except Exception as e:
        logger.error(f"Error fetching data from Alpaca: {str(e)}")
        return f"Error: {str(e)}"

@function_tool
async def analyze_crypto_data(data: List[Dict], calculate_indicators: bool = True) -> str:
    """
    Analyze cryptocurrency data and generate trading signals.
    
    Args:
        data: List of dictionaries containing price data
        calculate_indicators: Whether to calculate technical indicators
        
    Returns:
        Dictionary with technical analysis results and trading signals
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Standardize column names (handle different formats from Yahoo and Alpaca)
        column_mapping = {}
        if 'Date' in df.columns:
            column_mapping['Date'] = 'date'
        if 'timestamp' in df.columns:
            column_mapping['timestamp'] = 'date'
        if 'Open' in df.columns:
            column_mapping['Open'] = 'open'
        if 'High' in df.columns:
            column_mapping['High'] = 'high'
        if 'Low' in df.columns:
            column_mapping['Low'] = 'low'
        if 'Close' in df.columns:
            column_mapping['Close'] = 'close'
        if 'Volume' in df.columns:
            column_mapping['Volume'] = 'volume'
            
        # Rename columns if needed
        if column_mapping:
            df = df.rename(columns=column_mapping)
            
        # Make sure we have the required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"Error: Missing required columns: {missing_columns}"
            
        # Calculate technical indicators
        if calculate_indicators:
            df = prepare_analysis_data(df)
            
        # Generate trading signals
        signals = generate_trading_signals(df)
        
        # Get the current price and recent statistics
        current_price = df['close'].iloc[-1]
        day_change = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
        week_high = df['high'].iloc[-7:].max() if len(df) >= 7 else df['high'].max()
        week_low = df['low'].iloc[-7:].min() if len(df) >= 7 else df['low'].min()
        
        # Get support and resistance levels
        support_resistance = detect_support_resistance(df)
        
        # Calculate Fibonacci retracement levels
        fib_levels = calculate_fibonacci_retracement(df)
        
        # Compile the analysis results
        analysis = {
            "current_price": current_price,
            "day_change_percent": day_change,
            "week_high": week_high,
            "week_low": week_low,
            "latest_rsi": df['rsi'].iloc[-1],
            "latest_macd": df['macd_line'].iloc[-1],
            "latest_signal": df['signal_line'].iloc[-1],
            "latest_histogram": df['macd_histogram'].iloc[-1],
            "volume": df['volume'].iloc[-1] if 'volume' in df.columns else None,
            "sma_20": df['sma_20'].iloc[-1],
            "sma_50": df['sma_50'].iloc[-1],
            "sma_200": df['sma_200'].iloc[-1] if 'sma_200' in df.columns else None,
            "bollinger_upper": df['bb_upper'].iloc[-1],
            "bollinger_middle": df['bb_middle'].iloc[-1],
            "bollinger_lower": df['bb_lower'].iloc[-1],
            "trading_signals": signals,
            "support_levels": support_resistance['support'],
            "resistance_levels": support_resistance['resistance'],
            "fibonacci_levels": fib_levels
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error analyzing data: {str(e)}"

@function_tool
async def calculate_risk_reward(
    current_price: float, 
    target_price: float, 
    stop_loss: float
) -> str:
    """
    Calculate risk-reward ratio and potential profit/loss.
    
    Args:
        current_price: Current price of the cryptocurrency
        target_price: Target price for take profit
        stop_loss: Stop loss price
        
    Returns:
        Dictionary with risk-reward analysis
    """
    try:
        # Calculate risk (potential loss)
        risk = abs(current_price - stop_loss)
        
        # Calculate reward (potential gain)
        reward = abs(target_price - current_price)
        
        # Calculate risk-reward ratio
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Calculate percentage moves
        profit_percent = ((target_price / current_price) - 1) * 100
        loss_percent = ((stop_loss / current_price) - 1) * 100
        
        return {
            "current_price": current_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "risk_amount": risk,
            "reward_amount": reward,
            "risk_reward_ratio": risk_reward_ratio,
            "profit_percent": profit_percent,
            "loss_percent": loss_percent,
            "is_favorable": risk_reward_ratio >= 2  # Generally, 1:2 risk-reward is considered good
        }
    except Exception as e:
        logger.error(f"Error calculating risk-reward: {str(e)}")
        return f"Error: {str(e)}"

@function_tool
async def execute_trade(ctx: RunContextWrapper[AlpacaCryptoTrader], symbol: str, action: str, qty: float = None, notional: float = None) -> str:
    """
    Execute a trade on Alpaca.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., 'BTC/USD')
        action: Trade action ('buy' or 'sell')
        qty: Quantity to trade (optional)
        notional: Dollar amount to trade (optional)
        
    Returns:
        JSON string with the result of the trade execution
    """
    try:
        trader = ctx.context
        
        if action.lower() not in ["buy", "sell"]:
            return f"Error: Invalid action '{action}'. Must be 'buy' or 'sell'."
        
        if not (qty or notional):
            return "Error: Either quantity (qty) or notional value must be provided."
        
        # Execute the order
        order = trader.submit_crypto_order(
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
            "created_at": str(order.created_at)
        }
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        return f"Error: {str(e)}"

@function_tool
async def get_account_info(ctx: RunContextWrapper[AlpacaCryptoTrader]) -> str:
    """
    Get Alpaca account information.
    
    Returns:
        JSON string with account details
    """
    try:
        trader = ctx.context
        account = trader.get_account()
        
        # Return account details
        return {
            "account_id": account.id,
            "status": account.status,
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "initial_margin": float(account.initial_margin),
            "maintenance_margin": float(account.maintenance_margin)
        }
    except Exception as e:
        logger.error(f"Error getting account info: {str(e)}")
        return f"Error: {str(e)}"

@function_tool
async def get_positions(ctx: RunContextWrapper[AlpacaCryptoTrader]) -> str:
    """
    Get current positions from Alpaca.
    
    Returns:
        JSON string with current positions
    """
    try:
        trader = ctx.context
        positions = trader.get_positions()
        
        # Format positions
        formatted_positions = []
        for pos in positions:
            formatted_positions.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc) * 100  # Convert to percentage
            })
        
        return formatted_positions
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        return f"Error: {str(e)}"

@function_tool
async def calculate_position_size(
    ctx: RunContextWrapper[AlpacaCryptoTrader],
    symbol: str,
    risk_percent: float = 1.0,
    stop_loss_percent: float = 2.0
) -> str:
    """
    Calculate appropriate position size based on account equity and risk tolerance.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., 'BTC/USD')
        risk_percent: Percentage of account equity to risk on this trade (e.g., 1.0 for 1%)
        stop_loss_percent: Stop loss percentage below entry price (e.g., 2.0 for 2%)
        
    Returns:
        Dictionary with position sizing details
    """
    try:
        trader = ctx.context
        account = trader.get_account()
        
        # Get the current price for the symbol
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        # Normalize the symbol format
        normalized = normalize_crypto_symbol(symbol)
        
        # Get the bar data from Alpaca
        bars = trader.get_crypto_bars([normalized["with_slash"]], "1D", start=start_date, end=end_date)
        
        if bars.empty:
            return f"Error: No price data available for {symbol}"
        
        # Get the current price
        if 'symbol' in bars.index.names:
            symbols = bars.index.get_level_values('symbol').unique()
            symbol_to_use = None
            for s in symbols:
                if s == normalized["with_slash"] or s == normalized["without_slash"]:
                    symbol_to_use = s
                    break
                    
            if symbol_to_use:
                current_price = bars.xs(symbol_to_use, level='symbol')['close'].iloc[-1]
            else:
                return f"Error: Symbol {symbol} not found in data"
        else:
            current_price = bars['close'].iloc[-1]
        
        # Calculate the account equity
        equity = float(account.equity)
        
        # Calculate the dollar amount to risk (based on risk_percent)
        risk_amount = equity * (risk_percent / 100)
        
        # Calculate the stop loss price
        stop_loss_price = current_price * (1 - stop_loss_percent / 100)
        
        # Calculate the dollar risk per unit
        risk_per_unit = current_price - stop_loss_price
        
        # Calculate the position size in units
        position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        # Calculate position size in dollars
        position_size_dollars = position_size * current_price
        
        # Calculate percentage of account
        account_percentage = (position_size_dollars / equity) * 100
        
        return {
            "account_equity": equity,
            "risk_amount_dollars": risk_amount,
            "current_price": current_price,
            "stop_loss_price": stop_loss_price,
            "risk_per_unit": risk_per_unit,
            "position_size_units": position_size,
            "position_size_dollars": position_size_dollars,
            "account_percentage": account_percentage
        }
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        return f"Error: {str(e)}"

# Create the Crypto Agent
def create_crypto_agent():
    """Create and return a Crypto Trading Agent."""
    # Configure extra settings for the API
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
        name="CryptoTradingAgent",
        instructions="""You are a cryptocurrency trading and financial analysis AI.
        Your primary goal is to analyze cryptocurrency data, identify trading opportunities, and execute trades at the right time to maximize profits and minimize losses.
        
        You have access to historical price data and can execute trades through the Alpaca API.
        
        When analyzing cryptocurrency data:
        1. Look for patterns and trends in price movements
        2. Identify support and resistance levels
        3. Monitor volume to confirm price moves
        4. Consider market sentiment and volatility
        5. Apply technical indicators (moving averages, RSI, MACD, etc.)
        
        Your trading strategy should follow these risk management rules:
        - Never risk more than 1-2% of the portfolio on a single trade
        - Always use stop losses to protect capital
        - Aim for at least a 1:2 risk-reward ratio (preferably 1:3)
        - Cut losses quickly (don't let losing positions get worse)
        - Let winners run (don't sell profitable positions too early)
        - Consider overall market conditions before trading
        - Be patient - only trade when high-probability setups appear
        
        You should always explain your reasoning for making trade decisions, including:
        - The technical indicators that influenced your decision
        - Current price relative to key moving averages
        - Support and resistance levels
        - Overall market trend and conditions
        - Risk-reward calculation
        - Position sizing recommendations
        
        Remember that preservation of capital is more important than making profits. It's okay to recommend "hold" if there are no clear trading signals.
        """,
        model=OpenAIChatCompletionsModel(
            model="claude-3-7-sonnet-latest",
            openai_client=external_client,
        ),
        model_settings=ModelSettings(temperature=0.2),
        output_type=TradeAction,
        tools=[
            get_crypto_data_from_yahoo,
            get_crypto_data_from_alpaca,
            analyze_crypto_data,
            calculate_risk_reward,
            calculate_position_size,
            execute_trade,
            get_account_info,
            get_positions
        ]
    )
    return agent

async def main():
    # Initialize Alpaca trader
    trader = AlpacaCryptoTrader(paper=True)  # Use paper trading
    
    # Create the crypto agent
    try:
        crypto_agent = create_crypto_agent()
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        print(f"Error: Failed to create agent: {str(e)}")
        print("Please check your API key and connection settings.")
        return
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='CryptoTrading AI Agent')
    parser.add_argument('--symbol', type=str, default='BTC/USD', help='Cryptocurrency symbol (e.g., BTC/USD)')
    parser.add_argument('--period', type=str, default='30d', help='Analysis period (e.g., 7d, 30d, 90d)')
    parser.add_argument('--action', type=str, choices=['analyze', 'trade'], default='analyze', 
                        help='Action to perform: analyze or trade')
    parser.add_argument('--trade-amount', type=float, help='Dollar amount to trade if action is trade')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["OPENAI_LOG"] = "debug"
        logger.info("Verbose logging enabled")
    
    # Determine the query based on action
    if args.action == 'analyze':
        user_query = f"Analyze {args.symbol} for the last {args.period} and tell me if I should buy, sell, or hold. " \
                    f"Consider price trends, volume, support/resistance levels, and technical indicators. " \
                    f"If recommending a trade, suggest position size and stop loss."
    else:  # args.action == 'trade'
        if not args.trade_amount:
            print("Error: --trade-amount is required when action is 'trade'")
            return
            
        user_query = f"Analyze {args.symbol} and execute a trade if there's a clear signal. " \
                    f"Use ${args.trade_amount} as the trade amount. " \
                    f"Only trade if there's a strong signal with good risk-reward ratio. " \
                    f"Set appropriate stop loss and explain your decision."
    
    # Display what we're doing
    print(f"Running AI agent to {args.action} {args.symbol}...")
    
    try:
        # Run the agent
        result = await Runner.run(crypto_agent, user_query, context=trader)
        
        # Print the final output
        print("\nAgent's recommendation:")
        print(f"Symbol: {result.final_output.symbol}")
        print(f"Action: {result.final_output.action}")
        print(f"Reason: {result.final_output.reason}")
        print(f"Confidence: {result.final_output.confidence:.2f}")
        
        if result.final_output.suggested_quantity:
            print(f"Suggested quantity: {result.final_output.suggested_quantity}")
        if result.final_output.price_target:
            print(f"Price target: ${result.final_output.price_target:,.2f}")
        if result.final_output.stop_loss:
            print(f"Stop loss: ${result.final_output.stop_loss:,.2f}")
    
    except Exception as e:
        logger.error(f"Error during agent execution: {str(e)}")
        print(f"\nError during AI agent execution: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check that your LiteLLM API key is valid")
        print("2. Ensure the base URL is correctly set: https://litellm.deriv.ai/v1")
        print("3. Try running with verbose logging: --verbose")
        print("4. Check if LiteLLM service is available")
        
        if "401" in str(e):
            print("\nAuthentication error detected. Please verify your API key.")
        elif "connection" in str(e).lower():
            print("\nConnection error detected. Please check your internet connection and API endpoint.")

if __name__ == "__main__":
    # Try to load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")
    except Exception as e:
        logger.warning(f"Failed to load .env file: {str(e)}")
    
    # Check environment variables
    if not os.environ.get('ALPACA_API_KEY_ID') or not os.environ.get('ALPACA_API_SECRET_KEY'):
        print("Error: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables must be set.")
        print("Example: export ALPACA_API_KEY_ID='your-key'")
        print("         export ALPACA_API_SECRET_KEY='your-secret'")
        exit(1)
    
    # Check if LiteLLM API key is set
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Either LITELLM_API_KEY or OPENAI_API_KEY environment variable must be set.")
        print("Example: export LITELLM_API_KEY='your-litellm-key'")
        print("         export OPENAI_API_KEY='your-litellm-key'")
        exit(1)
    else:
        # Use the API key from LITELLM_API_KEY if set, otherwise use OPENAI_API_KEY
        if os.environ.get("LITELLM_API_KEY"):
            logger.info("Using LITELLM_API_KEY for authentication")
            os.environ["OPENAI_API_KEY"] = os.environ.get("LITELLM_API_KEY")
    
    # Run the main function
    asyncio.run(main()) 