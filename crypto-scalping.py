import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pytz

# Configuration
API_KEY = ""
SECRET_KEY = ""
SYMBOL = "BTC-USD"  # Yahoo Finance symbol for Bitcoin/USD
ALPACA_SYMBOL = "BTC/USD"  # Alpaca's symbol format
TIMEFRAME = "1m"  # 1-minute data
LOOKBACK_PERIOD = 30  # Reduced lookback period (was 100)
QUANTITY = 0.01  # Trade size in BTC (smaller amount for more frequent trades)

# Technical indicator parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 40  # Updated to match get_trading_signals
RSI_OVERBOUGHT = 60  # Updated to match get_trading_signals
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Trading strategy parameters
MAX_POSITION = 0.05  # Maximum position size (was 5000)
TAKE_PROFIT_PCT = 0.0005  # 0.05% (reduced for scalping)
STOP_LOSS_PCT = 0.0003  # 0.03% (reduced for scalping)

# Initialize Alpaca trading client
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)  # Use paper=True for paper trading

def fetch_historical_data():
    """Fetch historical forex data using yfinance"""
    end = datetime.now()
    # For scalping, we need more recent data but less history
    start = end - timedelta(hours=12)  # Get just last 12 hours instead of days
    
    # Download data from Yahoo Finance
    df = yf.download(SYMBOL, start=start, end=end, interval=TIMEFRAME)
    
    # Ensure we have enough data and it's recent
    if len(df) < LOOKBACK_PERIOD:
        print(f"Warning: Only retrieved {len(df)} data points, requested {LOOKBACK_PERIOD}")
    
    # Check for stale data (important during market closures)
    last_data_time = df.index[-1].to_pydatetime()
    time_difference = datetime.now(pytz.UTC) - last_data_time.replace(tzinfo=pytz.UTC)
    
    if time_difference > timedelta(minutes=5):
        print(f"Warning: Most recent data point is {time_difference} old. Data may be stale.")
        
    return df.tail(LOOKBACK_PERIOD)

def calculate_indicators(df):
    """Calculate technical indicators on the dataframe"""
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['sma'] = df['Close'].rolling(window=BOLLINGER_PERIOD).mean()
    df['std'] = df['Close'].rolling(window=BOLLINGER_PERIOD).std()
    df['upper_band'] = df['sma'] + (df['std'] * BOLLINGER_STD)
    df['lower_band'] = df['sma'] - (df['std'] * BOLLINGER_STD)
    
    # Calculate MACD
    df['ema_fast'] = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    df['ema_slow'] = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    return df

def get_trading_signals(df):
    """Generate buy/sell signals based on indicators"""
    # Convert all critical values to Python scalars to avoid Series comparison issues
    try:
        # Access the most recent values and convert to scalars
        close_price = float(df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].iloc[-1])
        lower_band = float(df['lower_band'].iloc[-1].item() if hasattr(df['lower_band'].iloc[-1], 'item') else df['lower_band'].iloc[-1])
        upper_band = float(df['upper_band'].iloc[-1].item() if hasattr(df['upper_band'].iloc[-1], 'item') else df['upper_band'].iloc[-1])
        current_rsi = float(df['rsi'].iloc[-1].item() if hasattr(df['rsi'].iloc[-1], 'item') else df['rsi'].iloc[-1])
        current_macd = float(df['macd_histogram'].iloc[-1].item() if hasattr(df['macd_histogram'].iloc[-1], 'item') else df['macd_histogram'].iloc[-1])
        prev_macd = float(df['macd_histogram'].iloc[-2].item() if hasattr(df['macd_histogram'].iloc[-2], 'item') else df['macd_histogram'].iloc[-2])
        is_high_volatility = bool(df['high_volatility'].iloc[-1].item() if hasattr(df['high_volatility'].iloc[-1], 'item') else df['high_volatility'].iloc[-1])
        
        # Check for NaN values
        if (pd.isna(close_price) or pd.isna(lower_band) or pd.isna(upper_band) or 
            pd.isna(current_rsi) or pd.isna(current_macd) or pd.isna(prev_macd)):
            print("Warning: NaN values detected in indicators, cannot generate signals")
            return False, False
            
        # More lenient sell conditions for scalping
        # Check if RSI is high enough and MACD histogram is decreasing
        buy_signal = (
            (current_rsi <= 40) and  # Less strict RSI threshold (was 30)
            (current_macd > prev_macd) and  # MACD histogram is increasing
            not is_high_volatility  # Not during high volatility
        )
        
        # More lenient sell conditions for scalping
        # Check if RSI is high enough and MACD histogram is decreasing
        sell_signal = (
            (current_rsi >= 60) and  # Less strict RSI threshold (was 70)
            (current_macd < prev_macd) and  # MACD histogram is decreasing
            not is_high_volatility  # Not during high volatility
        )
        
        return buy_signal, sell_signal
        
    except Exception as e:
        print(f"Error generating trading signals: {e}")
        return False, False

def check_positions():
    """Check current position in the account"""
    try:
        positions = trading_client.get_all_positions()
        for position in positions:
            if position.symbol == ALPACA_SYMBOL:
                return float(position.qty)
        return 0
    except Exception as e:
        print(f"Error checking positions: {e}")
        return 0

def place_order(side, quantity, current_price):
    """Place a market order with Alpaca"""
    try:
        order_data = MarketOrderRequest(
            symbol=ALPACA_SYMBOL,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        
        # Place the order
        order = trading_client.submit_order(order_data=order_data)
        print(f"{side} order placed for {quantity} {ALPACA_SYMBOL} at {current_price}")
        
        # Create bracket order for take profit and stop loss
        # Note: In a real implementation, you would use bracket orders
        # or create separate take profit and stop loss orders here
        if side == OrderSide.BUY:
            take_profit_price = current_price * (1 + TAKE_PROFIT_PCT)
            stop_loss_price = current_price * (1 - STOP_LOSS_PCT)
            print(f"  Take profit target: ${take_profit_price:.2f}")
            print(f"  Stop loss level: ${stop_loss_price:.2f}")
        else:
            take_profit_price = current_price * (1 - TAKE_PROFIT_PCT)
            stop_loss_price = current_price * (1 + STOP_LOSS_PCT)
            print(f"  Take profit target: ${take_profit_price:.2f}")
            print(f"  Stop loss level: ${stop_loss_price:.2f}")
        
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

def add_volatility_filter(df):
    """Add a volatility filter to avoid trading during extreme volatility"""
    # Calculate average true range (ATR) as a measure of volatility
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Normalize ATR as a percentage of price
    # Ensure indices align and calculate separately
    atr_series = df['atr'].squeeze()
    close_series = df['Close'].squeeze()
    # Reindex atr_series to match close_series index just in case
    atr_series = atr_series.reindex(close_series.index)
    
    # --- Debugging --- 
    # print(f"DEBUG: Type of atr_series: {type(atr_series)}, Shape: {atr_series.shape}")
    # print(f"DEBUG: Type of close_series: {type(close_series)}, Shape: {close_series.shape}")
    # --- End Debugging ---

    calculated_atr_pct = atr_series / close_series * 100
    df['atr_pct'] = calculated_atr_pct
    
    # Flag periods of extreme volatility
    df['high_volatility'] = df['atr_pct'] > df['atr_pct'].rolling(30).mean() * 1.5
    
    return df

def run_strategy():
    """Main function to run the scalping strategy"""
    print(f"Starting BTC/USD scalping strategy with yfinance data at {datetime.now()}")
    
    while True:
        try:
            # Fetch historical data using yfinance
            df = fetch_historical_data()
            
            if len(df) < LOOKBACK_PERIOD / 2:  # If we have less than half the required data
                print(f"Insufficient data available. Waiting 5 minutes to retry.")
                time.sleep(300)
                continue
                
            # Calculate technical indicators
            df = calculate_indicators(df)
            
            # Add volatility filter
            df = add_volatility_filter(df)
            
            # Get current position
            current_position = check_positions()
            
            # Get current price
            current_price = float(df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].iloc[-1])
            
            # Get trading signals (volatility check is now handled inside this function)
            buy_signal, sell_signal = get_trading_signals(df)
            
            # Execute trades based on signals
            if buy_signal and current_position < MAX_POSITION:
                place_order(OrderSide.BUY, QUANTITY, current_price)
            elif sell_signal and current_position > -MAX_POSITION:
                place_order(OrderSide.SELL, QUANTITY, current_price)
            
            # Print current status
            print(f"Time: {datetime.now()}, BTC Price: ${current_price:.2f}, Position: {current_position}")
            
            # Use try/except for each indicator to handle potential Series issues
            try:
                rsi_value = float(df['rsi'].iloc[-1].item() if hasattr(df['rsi'].iloc[-1], 'item') else df['rsi'].iloc[-1])
                print(f"RSI: {rsi_value:.2f}", end=", ")
            except Exception:
                print("RSI: N/A", end=", ")
                
            try:
                macd_hist = float(df['macd_histogram'].iloc[-1].item() if hasattr(df['macd_histogram'].iloc[-1], 'item') else df['macd_histogram'].iloc[-1])
                print(f"MACD Hist: {macd_hist:.6f}")
            except Exception:
                print("MACD Hist: N/A")
                
            try:
                upper = float(df['upper_band'].iloc[-1].item() if hasattr(df['upper_band'].iloc[-1], 'item') else df['upper_band'].iloc[-1])
                lower = float(df['lower_band'].iloc[-1].item() if hasattr(df['lower_band'].iloc[-1], 'item') else df['lower_band'].iloc[-1])
                print(f"BB Upper: ${upper:.2f}, BB Lower: ${lower:.2f}")
            except Exception:
                print("BB: N/A")
                
            try:
                atr_pct = float(df['atr_pct'].iloc[-1].item() if hasattr(df['atr_pct'].iloc[-1], 'item') else df['atr_pct'].iloc[-1])
                high_vol = bool(df['high_volatility'].iloc[-1].item() if hasattr(df['high_volatility'].iloc[-1], 'item') else df['high_volatility'].iloc[-1])
                print(f"Volatility (ATR%): {atr_pct:.2f}%, High Volatility: {high_vol}")
            except Exception:
                print("Volatility: N/A")
                
            print("-" * 50)
            
            # Sleep between cycles - check more frequently for scalping
            time.sleep(30)  # 30 seconds instead of 5 minutes
            
        except Exception as e:
            print(f"Error in strategy execution: {e}")
            time.sleep(30)  # Wait before retrying

def calculate_strategy_performance(backtest_days=30):
    """Calculate historical performance of the strategy (backtest)"""
    print(f"Running backtest over the past {backtest_days} days...")
    
    # Get historical data
    end = datetime.now()
    start = end - timedelta(days=backtest_days)
    df = yf.download(SYMBOL, start=start, end=end, interval=TIMEFRAME)
    
    # Calculate indicators
    df = calculate_indicators(df)
    df = add_volatility_filter(df)
    
    # Initialize tracking variables
    position = 0
    entry_price = 0
    trades = []
    
    # Loop through data and simulate trades
    for i in range(LOOKBACK_PERIOD, len(df)):
        # Access data directly using .iloc[i] to avoid potential Series label issues
        try:
            # Convert all values to Python scalars using .item()
            current_close = float(df['Close'].iloc[i].item() if hasattr(df['Close'].iloc[i], 'item') else df['Close'].iloc[i])
            lower_band = float(df['lower_band'].iloc[i].item() if hasattr(df['lower_band'].iloc[i], 'item') else df['lower_band'].iloc[i])
            upper_band = float(df['upper_band'].iloc[i].item() if hasattr(df['upper_band'].iloc[i], 'item') else df['upper_band'].iloc[i])
            current_rsi = float(df['rsi'].iloc[i].item() if hasattr(df['rsi'].iloc[i], 'item') else df['rsi'].iloc[i])
            current_macd_hist = float(df['macd_histogram'].iloc[i].item() if hasattr(df['macd_histogram'].iloc[i], 'item') else df['macd_histogram'].iloc[i])
            prev_macd_hist = float(df['macd_histogram'].iloc[i-1].item() if hasattr(df['macd_histogram'].iloc[i-1], 'item') else df['macd_histogram'].iloc[i-1])
            
            # Convert boolean to scalar explicitly
            is_high_volatility = bool(df['high_volatility'].iloc[i].item() if hasattr(df['high_volatility'].iloc[i], 'item') else df['high_volatility'].iloc[i])
            
            # Check for NaN values in required indicators for this row
            if (pd.isna(current_close) or pd.isna(lower_band) or pd.isna(upper_band) or 
                pd.isna(current_rsi) or pd.isna(current_macd_hist) or pd.isna(prev_macd_hist)):
                continue # Skip this iteration if any required data is NaN
                
            # Generate signals using direct values (now guaranteed scalar Python values)
            buy_signal = (
                (current_close <= lower_band) and
                (current_rsi <= RSI_OVERSOLD) and
                (current_macd_hist > prev_macd_hist) and
                not is_high_volatility
            )
            
            sell_signal = (
                (current_close >= upper_band) and
                (current_rsi >= RSI_OVERBOUGHT) and
                (current_macd_hist < prev_macd_hist) and
                not is_high_volatility
            )
            
        except Exception as e:
            print(f"Error in backtesting at index {i}: {e}")
            continue  # Skip this problematic data point
        
        # Execute simulated trades
        if buy_signal and position <= 0:
            if position < 0:  # Close short position
                trades.append({
                    'type': 'exit_short',
                    'price': current_close, # Use direct value
                    'time': df.index[i], # Use df.index[i] for timestamp
                    'profit': (entry_price - current_close) * abs(position)
                })
            
            # Open long position
            position = QUANTITY
            entry_price = current_close # Use direct value
            trades.append({
                'type': 'enter_long',
                'price': entry_price,
                'time': df.index[i], # Use df.index[i]
                'size': position
            })
            
        elif sell_signal and position >= 0:
            if position > 0:  # Close long position
                trades.append({
                    'type': 'exit_long',
                    'price': current_close, # Use direct value
                    'time': df.index[i], # Use df.index[i]
                    'profit': (current_close - entry_price) * position
                })
            
            # Open short position
            position = -QUANTITY
            entry_price = current_close # Use direct value
            trades.append({
                'type': 'enter_short',
                'price': entry_price,
                'time': df.index[i], # Use df.index[i]
                'size': abs(position)
            })
    
    # Close any open position at the end
    if position != 0:
        last_price = df['Close'].iloc[-1]
        if position > 0:
            profit = (last_price - entry_price) * position
            trades.append({
                'type': 'exit_long',
                'price': last_price,
                'time': df.index[-1],
                'profit': profit
            })
        else:
            profit = (entry_price - last_price) * abs(position)
            trades.append({
                'type': 'exit_short',
                'price': last_price,
                'time': df.index[-1],
                'profit': profit
            })
    
    # Calculate performance metrics
    if not trades:
        print("No trades were generated in the backtest period.")
        return
    
    total_trades = len([t for t in trades if t['type'] in ['enter_long', 'enter_short']])
    profitable_trades = len([t for t in trades if t.get('profit', 0) > 0])
    total_profit = sum([t.get('profit', 0) for t in trades])
    
    print(f"Backtest Results:")
    print(f"Total Trades: {total_trades}")
    print(f"Profitable Trades: {profitable_trades} ({profitable_trades/total_trades*100:.1f}%)")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Profit per Trade: ${total_profit/total_trades if total_trades else 0:.2f}")
    
    return trades

if __name__ == "__main__":
    # Verify account is set up for paper trading
    account = trading_client.get_account()
    if account.status != 'ACTIVE':
        raise Exception(f"Account is not active. Current status: {account.status}")
    
    print(f"Trading account ready. Cash available: ${account.cash}")
    
    # Run optional backtest before starting
    run_backtest = input("Run strategy backtest before starting? (y/n): ")
    if run_backtest.lower() == 'y':
        backtest_days = int(input("Enter number of days to backtest: ") or "30")
        calculate_strategy_performance(backtest_days)
    
    # Start the live trading strategy
    run_strategy()