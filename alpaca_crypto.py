#!/usr/bin/env python3
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import json

# Import Alpaca libraries
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetAssetsRequest, MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, AssetClass, AssetStatus
from alpaca.trading.models import Order
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def normalize_crypto_symbol(symbol):
    """
    Normalize a cryptocurrency symbol to the format expected by Alpaca API.
    
    Handles both formats like "BTC/USD" and "BTCUSD".
    Returns both versions of the symbol for maximum compatibility.
    """
    if '/' in symbol:
        # Format is like "BTC/USD"
        base, quote = symbol.split('/')
        without_slash = f"{base}{quote}"
        with_slash = symbol
    else:
        # Format is like "BTCUSD"
        if len(symbol) >= 6:
            # Assume first 3 chars are base currency, rest is quote currency
            # This is a simplistic approach, might need refinement for symbols
            # like "DOGEUSDT" where base isn't exactly 3 chars
            base = symbol[0:3]
            quote = symbol[3:]
            with_slash = f"{base}/{quote}"
            without_slash = symbol
        else:
            # If symbol is too short, keep as is
            with_slash = symbol
            without_slash = symbol
    
    return {
        "with_slash": with_slash,
        "without_slash": without_slash
    }

class AlpacaCryptoTrader:
    """Class to manage Alpaca API for cryptocurrency trading."""
    
    def __init__(self, paper=True):
        """Initialize the Alpaca clients with API credentials."""
        # Get API keys from environment variables
        self.api_key = os.environ.get('ALPACA_API_KEY_ID')
        self.api_secret = os.environ.get('ALPACA_API_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables must be set")
        
        # Initialize Alpaca clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=paper)
        self.data_client = CryptoHistoricalDataClient(self.api_key, self.api_secret)
        
        logger.info("AlpacaCryptoTrader initialized")
    
    def get_account(self):
        """Get account information."""
        try:
            account = self.trading_client.get_account()
            logger.info("Account information retrieved successfully")
            return account
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            raise
    
    def get_positions(self):
        """Get all open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            logger.info(f"Retrieved {len(positions)} open positions")
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise
    
    def get_orders(self, status="open", limit=50):
        """Get orders with the specified status."""
        try:
            request_params = GetOrdersRequest(
                status=status,
                limit=limit
            )
            orders = self.trading_client.get_orders(request_params)
            logger.info(f"Retrieved {len(orders)} {status} orders")
            return orders
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            raise
    
    def get_crypto_assets(self):
        """Get available cryptocurrency assets."""
        try:
            request_params = GetAssetsRequest(
                asset_class=AssetClass.CRYPTO,
                status=AssetStatus.ACTIVE
            )
            assets = self.trading_client.get_all_assets(request_params)
            logger.info(f"Retrieved {len(assets)} crypto assets")
            return assets
        except Exception as e:
            logger.error(f"Error getting crypto assets: {str(e)}")
            raise
    
    def get_crypto_bars(self, symbols, timeframe, start=None, end=None, limit=None):
        """Get historical bar data for cryptocurrencies."""
        try:
            # Default to last 7 days if not specified
            if not start:
                end = datetime.now()
                start = end - timedelta(days=7)
            elif not end:
                end = datetime.now()
            
            # Convert timeframe string to TimeFrame enum
            if isinstance(timeframe, str):
                if timeframe.lower() == "1m":
                    timeframe = TimeFrame.Minute
                elif timeframe.lower() == "1h":
                    timeframe = TimeFrame.Hour
                elif timeframe.lower() == "1d":
                    timeframe = TimeFrame.Day
                else:
                    # Default to 1 day if unrecognized
                    timeframe = TimeFrame.Day
            
            # Process symbols to ensure correct format
            processed_symbols = []
            for symbol in symbols:
                # Try both formats as Alpaca API might expect different formats
                # depending on the endpoint and environment
                normalized = normalize_crypto_symbol(symbol)
                # Use the version with slash as it's the most common format
                processed_symbols.append(normalized["with_slash"])
                
            logger.info(f"Requesting data for symbols: {processed_symbols}")
            
            # Build request parameters
            request_params = CryptoBarsRequest(
                symbol_or_symbols=processed_symbols,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit,
                adjustment=Adjustment.ALL
            )
            
            # Get the bars data
            bars = self.data_client.get_crypto_bars(request_params)
            
            # Convert to DataFrame and return
            if not bars.data:
                logger.warning(f"No data returned for {processed_symbols}")
                return pd.DataFrame()
                
            df = bars.df
            logger.info(f"Retrieved {len(df)} bars for {len(processed_symbols)} symbols")
            
            # Log the structure of the returned dataframe for debugging
            logger.info(f"DataFrame index names: {df.index.names}")
            if 'symbol' in df.index.names:
                unique_symbols = df.index.get_level_values('symbol').unique()
                logger.info(f"Unique symbols in data: {unique_symbols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting crypto bars: {str(e)}")
            logger.error(f"Requested symbols: {symbols}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def submit_crypto_order(self, symbol, side, qty=None, notional=None, order_type="market", 
                            limit_price=None, time_in_force="gtc"):
        """Submit an order for a cryptocurrency."""
        try:
            # Normalize the crypto symbol format
            normalized = normalize_crypto_symbol(symbol)
            # For trading, use the symbol format with slash (e.g., "BTC/USD")
            trading_symbol = normalized["with_slash"]
            
            # Convert parameters to appropriate enums
            side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            tif_enum = TimeInForce.GTC
            if time_in_force.lower() == "day":
                tif_enum = TimeInForce.DAY
            elif time_in_force.lower() == "ioc":
                tif_enum = TimeInForce.IOC
            
            # Create appropriate request object based on order type
            if order_type.lower() == "market":
                # Market order
                if qty is not None:
                    request = MarketOrderRequest(
                        symbol=trading_symbol,
                        qty=float(qty),
                        side=side_enum,
                        time_in_force=tif_enum,
                    )
                elif notional is not None:
                    request = MarketOrderRequest(
                        symbol=trading_symbol,
                        notional=float(notional),
                        side=side_enum,
                        time_in_force=tif_enum,
                    )
                else:
                    raise ValueError("Either qty or notional must be provided")
                    
            elif order_type.lower() == "limit":
                # Limit order
                if limit_price is None:
                    raise ValueError("Limit price must be provided for limit orders")
                    
                if qty is not None:
                    request = LimitOrderRequest(
                        symbol=trading_symbol,
                        qty=float(qty),
                        limit_price=float(limit_price),
                        side=side_enum,
                        time_in_force=tif_enum,
                    )
                elif notional is not None:
                    request = LimitOrderRequest(
                        symbol=trading_symbol,
                        notional=float(notional),
                        limit_price=float(limit_price),
                        side=side_enum,
                        time_in_force=tif_enum,
                    )
                else:
                    raise ValueError("Either qty or notional must be provided")
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Submit the order using the request object
            order = self.trading_client.submit_order(request)
            
            logger.info(f"Order submitted: {order.id} for {trading_symbol} {side} {qty or notional}")
            return order
            
        except Exception as e:
            logger.error(f"Error submitting order: {str(e)}")
            raise
    
    def get_position(self, symbol):
        """Get position for a specific cryptocurrency."""
        try:
            # Convert symbol format if needed for positions
            normalized = normalize_crypto_symbol(symbol)
            # For positions, Alpaca typically uses the format without slash (e.g., "BTCUSD")
            position_symbol = normalized["without_slash"]
            
            logger.info(f"Getting position for {position_symbol}")
            position = self.trading_client.get_open_position(position_symbol)
            logger.info(f"Position retrieved for {position_symbol}")
            return position
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            return None
    
    def close_position(self, symbol):
        """Close an open position for a cryptocurrency."""
        try:
            # Convert symbol format if needed for positions
            normalized = normalize_crypto_symbol(symbol)
            # For positions, Alpaca typically uses the format without slash (e.g., "BTCUSD")
            position_symbol = normalized["without_slash"]
            
            logger.info(f"Closing position for {position_symbol}")
            response = self.trading_client.close_position(position_symbol)
            logger.info(f"Position closed for {position_symbol}")
            return response
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            raise
    
    def close_all_positions(self):
        """Close all open positions."""
        try:
            response = self.trading_client.close_all_positions()
            logger.info("All positions closed")
            return response
        except Exception as e:
            logger.error(f"Error closing all positions: {str(e)}")
            raise
    
    def get_clock(self):
        """Get the current market clock."""
        try:
            clock = self.trading_client.get_clock()
            return clock
        except Exception as e:
            logger.error(f"Error getting market clock: {str(e)}")
            raise

class CryptoPortfolioAnalyzer:
    """Class to analyze cryptocurrency portfolio performance and metrics."""
    
    def __init__(self, trader):
        """Initialize with an AlpacaCryptoTrader instance."""
        self.trader = trader
        logger.info("CryptoPortfolioAnalyzer initialized")
    
    def get_portfolio_summary(self):
        """Get a summary of the current portfolio."""
        try:
            account = self.trader.get_account()
            positions = self.trader.get_positions()
            
            equity = float(account.equity)
            cash = float(account.cash)
            buying_power = float(account.buying_power)
            
            position_value = 0
            position_allocation = {}
            
            # Calculate position values and allocations
            if positions:
                for pos in positions:
                    symbol = pos.symbol
                    market_value = float(pos.market_value)
                    position_value += market_value
                    
                    # Calculate allocation percentage
                    if equity > 0:
                        allocation = (market_value / equity) * 100
                        position_allocation[symbol] = allocation
            
            # Calculate cash and position allocations as percentages
            if equity > 0:
                cash_allocation = (cash / equity) * 100
                position_allocation_total = (position_value / equity) * 100
            else:
                cash_allocation = 100.0
                position_allocation_total = 0.0
            
            summary = {
                "equity": equity,
                "cash": cash,
                "buying_power": buying_power,
                "position_value": position_value,
                "position_count": len(positions),
                "cash_allocation": cash_allocation,
                "position_allocation": position_allocation_total,
                "positions": position_allocation
            }
            
            logger.info("Portfolio summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {str(e)}")
            return {"error": str(e)}
    
    def get_equity_history(self, timeframe="1D", days=30):
        """Get the equity history for the portfolio."""
        # Note: As of now, Alpaca does not provide a direct API for historical equity values.
        # This would typically involve storing your own equity values over time.
        # This method is a placeholder and would return a simulated equity curve.
        
        # Placeholder implementation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get account info for current equity
        account = self.trader.get_account()
        current_equity = float(account.equity)
        
        # Create a date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # For now, just create a simulated equity curve based on random fluctuations
        # In a real implementation, you would retrieve actual historical equity values
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.001, 0.02, size=len(dates))
        equity_values = [current_equity]
        
        # Work backwards to generate equity history
        for ret in reversed(daily_returns[:-1]):
            prev_equity = equity_values[0] / (1 + ret)
            equity_values.insert(0, prev_equity)
        
        equity_history = pd.DataFrame({
            'date': dates,
            'equity': equity_values
        })
        equity_history.set_index('date', inplace=True)
        
        logger.info(f"Generated equity history for the last {days} days")
        return equity_history
    
    def calculate_portfolio_metrics(self):
        """Calculate performance metrics for the portfolio."""
        try:
            # Get equity history (simulated for now)
            equity_history = self.get_equity_history(days=90)
            
            if equity_history.empty:
                return {"error": "No equity history available"}
            
            # Calculate daily returns
            equity_history['daily_return'] = equity_history['equity'].pct_change()
            
            # Drop NaN values (first day)
            equity_history = equity_history.dropna()
            
            # Calculate metrics
            current_equity = equity_history['equity'].iloc[-1]
            starting_equity = equity_history['equity'].iloc[0]
            
            # Returns
            total_return = (current_equity / starting_equity) - 1
            
            # Annualize based on trading days
            trading_days = len(equity_history)
            annual_factor = 252 / trading_days
            annual_return = ((1 + total_return) ** annual_factor) - 1
            
            # Volatility (annualized standard deviation of returns)
            daily_volatility = equity_history['daily_return'].std()
            annual_volatility = daily_volatility * np.sqrt(252)
            
            # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            avg_daily_return = equity_history['daily_return'].mean()
            sharpe_ratio = (avg_daily_return * 252) / annual_volatility if annual_volatility > 0 else 0
            
            # Drawdown
            equity_history['cumulative_return'] = (1 + equity_history['daily_return']).cumprod()
            equity_history['rolling_max'] = equity_history['cumulative_return'].cummax()
            equity_history['drawdown'] = equity_history['rolling_max'] - equity_history['cumulative_return']
            equity_history['drawdown_pct'] = equity_history['drawdown'] / equity_history['rolling_max']
            max_drawdown = equity_history['drawdown_pct'].max()
            
            # Count positive and negative days
            positive_days = (equity_history['daily_return'] > 0).sum()
            negative_days = (equity_history['daily_return'] < 0).sum()
            win_rate = positive_days / trading_days if trading_days > 0 else 0
            
            metrics = {
                "current_equity": current_equity,
                "starting_equity": starting_equity,
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "avg_daily_return": avg_daily_return,
                "total_days": trading_days,
                "positive_days": positive_days,
                "negative_days": negative_days
            }
            
            logger.info("Portfolio metrics calculated")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {"error": str(e)}
    
    def plot_equity_curve(self, days=30, save_path=None):
        """Plot the equity curve for the portfolio."""
        try:
            equity_history = self.get_equity_history(days=days)
            
            if equity_history.empty:
                logger.warning("No equity data available for plotting")
                return False
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(equity_history.index, equity_history['equity'], label='Portfolio Equity')
            plt.title(f'Portfolio Equity Curve (Last {days} Days)')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.legend()
            
            # Format the date on x-axis
            date_format = DateFormatter("%Y-%m-%d")
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gcf().autofmt_xdate()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Equity curve saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            return False


class CryptoTradingStrategies:
    """Class implementing various cryptocurrency trading strategies."""
    
    def __init__(self, trader):
        """Initialize with an AlpacaCryptoTrader instance."""
        self.trader = trader
        logger.info("CryptoTradingStrategies initialized")
    
    def _calculate_position_size(self, symbol, risk_percentage=1.0, stop_loss_percentage=2.0):
        """Calculate position size based on risk management rules."""
        try:
            # Get account equity
            account = self.trader.get_account()
            equity = float(account.equity)
            
            # Calculate dollar amount to risk
            risk_amount = equity * (risk_percentage / 100)
            
            # Get current price
            bars = self.trader.get_crypto_bars([symbol], "1D", limit=1)
            if bars.empty:
                return 0
                
            # Extract the price for this symbol
            symbol_data = bars.loc[(slice(None), symbol), :]
            current_price = symbol_data['close'].iloc[-1]
            
            # Calculate stop loss price
            stop_loss = current_price * (1 - stop_loss_percentage / 100)
            
            # Calculate how many units to buy
            price_risk = current_price - stop_loss
            position_size = risk_amount / price_risk if price_risk > 0 else 0
            
            # Ensure position size is not zero
            if position_size <= 0:
                return 0
                
            # Calculate equivalent dollar amount
            position_value = position_size * current_price
            
            # Make sure we're not exceeding max position size (e.g., 10% of portfolio)
            max_position_value = equity * 0.1
            if position_value > max_position_value:
                position_size = max_position_value / current_price
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def moving_average_crossover_strategy(self, symbols, short_period=20, long_period=50, days=100):
        """Implement a simple moving average crossover strategy."""
        signals = []
        
        try:
            for symbol in symbols:
                # Get historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                bars = self.trader.get_crypto_bars([symbol], "1D", start=start_date, end=end_date)
                
                if bars.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Filter for just this symbol
                symbol_data = bars.loc[(slice(None), symbol), :]
                
                if len(symbol_data) < long_period:
                    logger.warning(f"Not enough data for {symbol}, need at least {long_period} days")
                    continue
                
                # Calculate moving averages
                symbol_data['short_ma'] = symbol_data['close'].rolling(window=short_period).mean()
                symbol_data['long_ma'] = symbol_data['close'].rolling(window=long_period).mean()
                
                # Drop NaN values
                symbol_data = symbol_data.dropna()
                
                if symbol_data.empty:
                    logger.warning(f"No valid data for {symbol} after calculating moving averages")
                    continue
                
                # Get latest data point
                latest = symbol_data.iloc[-1]
                previous = symbol_data.iloc[-2]
                
                # Check for buy signal - short MA crosses above long MA
                if (previous['short_ma'] <= previous['long_ma']) and (latest['short_ma'] > latest['long_ma']):
                    # Calculate position size
                    qty = self._calculate_position_size(symbol)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'price': latest['close'],
                        'qty': qty,
                        'short_ma': latest['short_ma'],
                        'long_ma': latest['long_ma']
                    })
                    
                    logger.info(f"Buy signal for {symbol} at {latest['close']}")
                
                # Check for sell signal - short MA crosses below long MA
                elif (previous['short_ma'] >= previous['long_ma']) and (latest['short_ma'] < latest['long_ma']):
                    # Get current position
                    position = self.trader.get_position(symbol)
                    
                    if position:
                        qty = float(position.qty)
                        
                        signals.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'price': latest['close'],
                            'qty': qty,
                            'short_ma': latest['short_ma'],
                            'long_ma': latest['long_ma']
                        })
                        
                        logger.info(f"Sell signal for {symbol} at {latest['close']}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error executing moving average strategy: {str(e)}")
            return []
    
    def rsi_strategy(self, symbols, period=14, oversold=30, overbought=70, days=100):
        """Implement a Relative Strength Index (RSI) based strategy."""
        signals = []
        
        try:
            for symbol in symbols:
                # Get historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                bars = self.trader.get_crypto_bars([symbol], "1D", start=start_date, end=end_date)
                
                if bars.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Filter for just this symbol
                symbol_data = bars.loc[(slice(None), symbol), :]
                
                if len(symbol_data) < period + 1:
                    logger.warning(f"Not enough data for {symbol}, need at least {period + 1} days")
                    continue
                
                # Calculate RSI
                close_prices = symbol_data['close']
                delta = close_prices.diff()
                
                # Make two series: one for gains and one for losses
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # Calculate the average gain and loss
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                # Calculate the Relative Strength (RS)
                rs = avg_gain / avg_loss
                
                # Calculate the RSI
                rsi = 100 - (100 / (1 + rs))
                
                # Add RSI to the dataframe
                symbol_data['rsi'] = rsi
                
                # Drop NaN values
                symbol_data = symbol_data.dropna()
                
                if symbol_data.empty:
                    logger.warning(f"No valid data for {symbol} after calculating RSI")
                    continue
                
                # Get latest data point
                latest = symbol_data.iloc[-1]
                previous = symbol_data.iloc[-2]
                
                # Check for buy signal - RSI crosses above oversold threshold
                if (previous['rsi'] <= oversold) and (latest['rsi'] > oversold):
                    # Calculate position size
                    qty = self._calculate_position_size(symbol)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'price': latest['close'],
                        'qty': qty,
                        'rsi': latest['rsi']
                    })
                    
                    logger.info(f"Buy signal for {symbol} at {latest['close']} with RSI {latest['rsi']}")
                
                # Check for sell signal - RSI crosses below overbought threshold
                elif (previous['rsi'] >= overbought) and (latest['rsi'] < overbought):
                    # Get current position
                    position = self.trader.get_position(symbol)
                    
                    if position:
                        qty = float(position.qty)
                        
                        signals.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'price': latest['close'],
                            'qty': qty,
                            'rsi': latest['rsi']
                        })
                        
                        logger.info(f"Sell signal for {symbol} at {latest['close']} with RSI {latest['rsi']}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error executing RSI strategy: {str(e)}")
            return []


def show_progress(message, total=10, delay=0.1):
    """Show a simple progress bar for API calls."""
    try:
        import sys
        import time
        
        # Print message
        sys.stdout.write(f"{message} [")
        sys.stdout.flush()
        
        # Show progress
        for i in range(total):
            time.sleep(delay)
            sys.stdout.write("=")
            sys.stdout.flush()
        
        # Finish
        sys.stdout.write("] Done\n")
        sys.stdout.flush()
    except:
        # In case of any error, just continue without showing progress
        pass


if __name__ == "__main__":
    # Sample usage
    try:
        trader = AlpacaCryptoTrader()
        
        # Get account info
        account = trader.get_account()
        print(f"Account: {account.id}")
        print(f"Cash: ${float(account.cash)}")
        print(f"Portfolio Value: ${float(account.equity)}")
        
        # List crypto assets
        assets = trader.get_crypto_assets()
        print(f"\nAvailable Crypto Assets ({len(assets)}):")
        for asset in assets[:5]:  # Print first 5
            print(f"- {asset.symbol}: {asset.name}")
        
        # Get market data
        print("\nGetting BTC/USD data for past 7 days...")
        bars = trader.get_crypto_bars(["BTC/USD"], "1D", limit=7)
        if not bars.empty:
            print(bars)
        
    except Exception as e:
        logger.error(f"Error in sample usage: {str(e)}") 