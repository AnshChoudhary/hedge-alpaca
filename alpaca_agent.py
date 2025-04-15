import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from google.adk.agents import Agent

# Alpaca API configuration - using paper trading
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY") 
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")

# Check if we have valid API keys before creating clients
has_valid_credentials = (ALPACA_API_KEY != "YOUR_ALPACA_API_KEY" and 
                        ALPACA_SECRET_KEY != "YOUR_ALPACA_SECRET_KEY")

# Initialize clients only if we have valid credentials
if has_valid_credentials:
    try:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    except Exception as e:
        print(f"Error initializing Alpaca clients: {e}")
        has_valid_credentials = False
else:
    print("No valid Alpaca API credentials found. Alpaca trading features will be unavailable.")

def get_market_data(ticker: str, period: str = "1mo") -> dict:
    """Retrieves market data for a specified stock or cryptocurrency ticker.

    Args:
        ticker (str): The ticker symbol (e.g., 'AAPL', 'BTC-USD').
        period (str): The time period for data (e.g., '1d', '1mo', '1y'). Default is '1mo'.

    Returns:
        dict: status and result or error msg.
    """
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period=period)
        
        if hist.empty:
            return {
                "status": "error",
                "error_message": f"No data found for ticker '{ticker}'."
            }
        
        current_price = hist['Close'].iloc[-1]
        price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0]
        percent_change = (price_change / hist['Close'].iloc[0]) * 100
        
        summary = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "percent_change": round(percent_change, 2),
            "period": period,
            "volume": int(hist['Volume'].mean())
        }
        
        return {
            "status": "success",
            "report": f"Market data for {ticker} over {period}: Current price: ${current_price:.2f}, Change: ${price_change:.2f} ({percent_change:.2f}%), Avg Volume: {int(hist['Volume'].mean())}",
            "data": summary
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error retrieving data for '{ticker}': {str(e)}"
        }


def calculate_correlation(ticker1: str, ticker2: str, period: str = "1y") -> dict:
    """Calculates correlation between two tickers.

    Args:
        ticker1 (str): First ticker symbol.
        ticker2 (str): Second ticker symbol.
        period (str): The time period for data. Default is '1y'.

    Returns:
        dict: status and result or error msg.
    """
    try:
        # Get historical data
        data1 = yf.Ticker(ticker1).history(period=period)['Close']
        data2 = yf.Ticker(ticker2).history(period=period)['Close']
        
        # Align the data (only use dates where both tickers have data)
        combined = pd.DataFrame({ticker1: data1, ticker2: data2})
        combined = combined.dropna()
        
        if combined.empty or len(combined) < 5:
            return {
                "status": "error",
                "error_message": f"Insufficient data to calculate correlation between {ticker1} and {ticker2}."
            }
        
        # Calculate correlations
        pearson_corr = combined[ticker1].corr(combined[ticker2], method='pearson')
        spearman_corr = combined[ticker1].corr(combined[ticker2], method='spearman')
        
        return {
            "status": "success",
            "report": f"Correlation between {ticker1} and {ticker2} over {period}: Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}",
            "data": {
                "pearson": round(pearson_corr, 4),
                "spearman": round(spearman_corr, 4),
                "period": period,
                "sample_size": len(combined)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error calculating correlation: {str(e)}"
        }


def get_account_info() -> dict:
    """Retrieves the current Alpaca paper trading account information.

    Returns:
        dict: status and result or error msg.
    """
    if not has_valid_credentials:
        return {
            "status": "error",
            "error_message": "Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        }
    
    try:
        account = trading_client.get_account()
        
        info = {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "initial_margin": float(account.initial_margin),
            "daytrade_count": account.daytrade_count,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
            "paper_only": True  # Always True since we're only allowing paper trading
        }
        
        return {
            "status": "success",
            "report": f"Account Summary: Cash: ${info['cash']:.2f}, Portfolio Value: ${info['portfolio_value']:.2f}, Buying Power: ${info['buying_power']:.2f}",
            "data": info
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error retrieving account information: {str(e)}"
        }


def get_positions() -> dict:
    """Retrieves all current positions in the Alpaca paper trading account.

    Returns:
        dict: status and result or error msg.
    """
    if not has_valid_credentials:
        return {
            "status": "error",
            "error_message": "Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        }
    
    try:
        positions = trading_client.get_all_positions()
        
        if not positions:
            return {
                "status": "success",
                "report": "You currently have no open positions in your paper trading account.",
                "data": []
            }
        
        position_list = []
        position_summary = []
        
        for position in positions:
            pos_data = {
                "symbol": position.symbol,
                "qty": float(position.qty),
                "market_value": float(position.market_value),
                "avg_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc) * 100,  # Convert to percentage
                "side": position.side
            }
            
            position_list.append(pos_data)
            position_summary.append(
                f"{pos_data['symbol']}: {pos_data['qty']} shares at ${pos_data['current_price']:.2f} " +
                f"(P/L: ${pos_data['unrealized_pl']:.2f}, {pos_data['unrealized_plpc']:.2f}%)"
            )
        
        return {
            "status": "success",
            "report": "Current positions:\n" + "\n".join(position_summary),
            "data": position_list
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error retrieving positions: {str(e)}"
        }


def place_order(ticker: str, qty: float, side: str) -> dict:
    """Places a market order to buy or sell a stock/crypto through Alpaca paper trading.

    Args:
        ticker (str): The ticker symbol (e.g., 'AAPL', 'BTC/USD').
        qty (float): The quantity to buy or sell.
        side (str): Either 'buy' or 'sell'.

    Returns:
        dict: status and result or error msg.
    """
    if not has_valid_credentials:
        return {
            "status": "error",
            "error_message": "Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        }
    
    try:
        # Validate input
        if side.lower() not in ['buy', 'sell']:
            return {
                "status": "error",
                "error_message": "Order side must be either 'buy' or 'sell'."
            }
        
        # Prepare order request
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        
        # Determine if this is a crypto ticker (contains a slash)
        is_crypto = '/' in ticker
        
        # Use different time_in_force for crypto vs stocks
        # For crypto, we need to use GTC (Good Till Canceled)
        if is_crypto:
            time_in_force = TimeInForce.GTC
        else:
            time_in_force = TimeInForce.DAY
        
        market_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=time_in_force
        )
        
        # Submit order
        order = trading_client.submit_order(market_order_data)
        
        # Ensure all values are JSON serializable (convert UUID to string)
        order_info = {
            "id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": side.lower(),
            "order_type": order.order_type,
            "status": order.status,
            "submitted_at": str(order.submitted_at)
        }
        
        return {
            "status": "success",
            "report": f"Successfully placed a {side.lower()} order for {qty} shares of {ticker}. Order ID: {str(order.id)}",
            "data": order_info
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error placing order: {str(e)}"
        }


def get_order_by_id(order_id: str) -> dict:
    """Retrieves the status of a specific order by ID.

    Args:
        order_id (str): The ID of the order to check.

    Returns:
        dict: status and result or error msg.
    """
    if not has_valid_credentials:
        return {
            "status": "error",
            "error_message": "Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        }
    
    try:
        # Get specific order
        order = trading_client.get_order_by_id(order_id)
        
        # Ensure all values are JSON serializable
        order_info = {
            "id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value,
            "order_type": order.order_type,
            "status": order.status.value,
            "submitted_at": str(order.submitted_at),
            "filled_at": str(order.filled_at) if order.filled_at else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0
        }
        
        return {
            "status": "success",
            "report": f"Order status for {order_id}: {order.status.value}",
            "data": order_info
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error retrieving order information: {str(e)}"
        }


def get_recent_orders() -> dict:
    """Retrieves the status of recent orders (last 5).

    Returns:
        dict: status and result or error msg.
    """
    if not has_valid_credentials:
        return {
            "status": "error",
            "error_message": "Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        }
    
    try:
        # Get recent orders (last 5)
        request_params = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=5
        )
        orders = trading_client.get_orders(request_params)
        
        if not orders:
            return {
                "status": "success",
                "report": "No recent orders found.",
                "data": []
            }
        
        order_list = []
        order_summary = []
        
        for order in orders:
            # Ensure all values are JSON serializable
            order_data = {
                "id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "order_type": order.order_type,
                "status": order.status.value,
                "submitted_at": str(order.submitted_at),
                "filled_at": str(order.filled_at) if order.filled_at else None
            }
            
            order_list.append(order_data)
            order_summary.append(
                f"{order.symbol}: {order.side.value} {order.qty} shares, Status: {order.status.value}"
            )
        
        return {
            "status": "success",
            "report": "Recent orders:\n" + "\n".join(order_summary),
            "data": order_list
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error retrieving order information: {str(e)}"
        }


def get_order_status(order_id: Optional[str] = None) -> dict:
    """Retrieves the status of a specific order or recent orders if no ID is provided.

    Args:
        order_id (Optional[str], optional): The ID of the order to check. If None, returns recent orders.

    Returns:
        dict: status and result or error msg.
    """
    try:
        if order_id:
            # Get specific order
            order = trading_client.get_order_by_id(order_id)
            
            order_info = {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "order_type": order.order_type,
                "status": order.status.value,
                "submitted_at": str(order.submitted_at),
                "filled_at": str(order.filled_at) if order.filled_at else None,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0
            }
            
            return {
                "status": "success",
                "report": f"Order status for {order_id}: {order.status.value}",
                "data": order_info
            }
        else:
            # Get recent orders (last 5)
            request_params = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=5
            )
            orders = trading_client.get_orders(request_params)
            
            if not orders:
                return {
                    "status": "success",
                    "report": "No recent orders found.",
                    "data": []
                }
            
            order_list = []
            order_summary = []
            
            for order in orders:
                order_data = {
                    "id": order.id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side.value,
                    "order_type": order.order_type,
                    "status": order.status.value,
                    "submitted_at": str(order.submitted_at),
                    "filled_at": str(order.filled_at) if order.filled_at else None
                }
                
                order_list.append(order_data)
                order_summary.append(
                    f"{order.symbol}: {order.side.value} {order.qty} shares, Status: {order.status.value}"
                )
            
            return {
                "status": "success",
                "report": "Recent orders:\n" + "\n".join(order_summary),
                "data": order_list
            }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error retrieving order information: {str(e)}"
        }


root_agent = Agent(
    name="finance_market_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to analyze stocks and cryptocurrencies, providing market data, correlation analysis, "
        "and paper trading capabilities through Alpaca."
    ),
    instruction=(
        "You are a financial expert agent who can provide market data for stocks and cryptocurrencies, "
        "analyze correlations between different assets, and execute paper trades through Alpaca. "
        "When users ask about a ticker, use get_market_data. "
        "When they want to compare two assets, use calculate_correlation. "
        "When they want to check their account or positions, use get_account_info or get_positions. "
        "When they want to place trades, use place_order with paper trading only. "
        "When they want to check a specific order status, use get_order_by_id. "
        "When they want to see recent orders, use get_recent_orders. "
        "Always clarify that you are using paper trading only, not real money. "
        "Be informative about market trends when providing data, but be cautious about giving financial advice."
    ),
    tools=[
        get_market_data, 
        calculate_correlation, 
        get_account_info, 
        get_positions, 
        place_order, 
        get_order_by_id,
        get_recent_orders
    ],
)