import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
import logging
import signal
import sys
import ta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mstr_btc_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration
API_KEY = ""
API_SECRET = ""

BTC_SYMBOL = "BTC-USD"  # Yahoo Finance symbol for Bitcoin/USD
MSTR_SYMBOL = "MSTR"    # MicroStrategy stock
ALPACA_SYMBOL = MSTR_SYMBOL  # Alpaca's symbol format

# Technical indicator parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Strategy parameters
MIN_CORRELATION = 0.5          # Minimum correlation to consider trading
MAX_POSITION_SIZE = 0.1        # Maximum position size as percentage of portfolio
STOP_LOSS_PCT = 0.005          # 0.5% stop loss
TAKE_PROFIT_PCT = 0.01         # 1% take profit
MIN_BTC_MOVEMENT_PCT = 0.003   # 0.3% minimum BTC movement to trigger entry
MAX_TRADE_DURATION_MIN = 60    # Maximum time to hold a position in minutes

# Initialize Alpaca API using the newer SDK
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

# Global variables for position tracking
current_position = 0
entry_price = 0
entry_time = None
session_pnl = 0
trades_history = []
running = True

def check_market_open():
    """Check if the US stock market is currently open"""
    clock = trading_client.get_clock()
    return clock.is_open

def get_trading_hours():
    """Get today's market open and close times"""
    calendar = trading_client.get_calendar(
        start=datetime.now().strftime('%Y-%m-%d'),
        end=datetime.now().strftime('%Y-%m-%d')
    )
    
    if not calendar:
        logger.error("Could not get market calendar")
        return None, None
        
    market_open = calendar[0].open.replace(tzinfo=pytz.timezone('America/New_York'))
    market_close = calendar[0].close.replace(tzinfo=pytz.timezone('America/New_York'))
    
    return market_open, market_close

def fetch_historical_data(symbol, timeframe='1m', lookback_days=3):
    """Fetch historical data using yfinance"""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    
    # Download data from Yahoo Finance
    df = yf.download(symbol, start=start, end=end, interval=timeframe)
    
    # Check for stale data
    if len(df) < 10:
        logger.warning(f"Limited data available for {symbol}: {len(df)} bars")
    
    return df

def fetch_latest_data():
    """Fetch the latest data for both BTC and MSTR"""
    # For real-time analysis, we need some historical context
    btc_data = fetch_historical_data(BTC_SYMBOL, timeframe='1m', lookback_days=1)
    mstr_data = fetch_historical_data(MSTR_SYMBOL, timeframe='1m', lookback_days=1)
    
    # Ensure we have data for both assets
    if btc_data.empty or mstr_data.empty:
        logger.error("Failed to fetch data for one or both assets")
        return None, None
    
    return btc_data, mstr_data

def calculate_indicators(df):
    """Calculate technical indicators on the dataframe"""
    if len(df) < 30:  # Need minimum data points
        logger.warning("Not enough data to calculate indicators")
        return df
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=RSI_PERIOD).rsi()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=BOLLINGER_PERIOD, window_dev=BOLLINGER_STD)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    
    # MACD
    macd = ta.trend.MACD(df['Close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Volatility (ATR)
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['atr_pct'] = df['atr'] / df['Close'] * 100
    
    return df

def calculate_correlation(btc_data, mstr_data, window=60):
    """Calculate correlation between BTC and MSTR"""
    # Ensure both dataframes have the same index
    common_index = btc_data.index.intersection(mstr_data.index)
    if len(common_index) < window:
        logger.warning(f"Limited overlapping data points: {len(common_index)}")
        return 0.5  # Default to moderate correlation when data is insufficient
    
    # Align dataframes to common index
    btc_aligned = btc_data.loc[common_index]['Close']
    mstr_aligned = mstr_data.loc[common_index]['Close']
    
    # Calculate rolling correlation, use the most recent value
    rolling_corr = btc_aligned.rolling(window=window).corr(mstr_aligned)
    current_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else 0.5
    
    return current_corr

def calculate_correlation_stability(btc_data, mstr_data, window=60):
    """Calculate the stability of correlation (standard deviation of correlation)"""
    # Ensure both dataframes have the same index
    common_index = btc_data.index.intersection(mstr_data.index)
    if len(common_index) < window*2:
        return 0.1  # Default value when not enough data
    
    # Align dataframes to common index
    btc_aligned = btc_data.loc[common_index]['Close']
    mstr_aligned = mstr_data.loc[common_index]['Close']
    
    # Calculate rolling correlation
    rolling_corr = btc_aligned.rolling(window=window).corr(mstr_aligned)
    
    # Calculate standard deviation of recent correlation values
    if len(rolling_corr) > window:
        corr_stability = rolling_corr.tail(window).std()
        return corr_stability
    else:
        return 0.1  # Default value

def detect_btc_movement(btc_data):
    """Detect significant BTC price movements"""
    if len(btc_data) < 3:
        return 0, "neutral"
    
    # Calculate percentage changes for different timeframes
    latest_price = btc_data['Close'].iloc[-1]
    prev_price_1min = btc_data['Close'].iloc[-2]
    
    # 1-minute change
    pct_change_1min = (latest_price / prev_price_1min - 1)
    
    # 3-minute change (if available)
    if len(btc_data) >= 4:
        prev_price_3min = btc_data['Close'].iloc[-4]
        pct_change_3min = (latest_price / prev_price_3min - 1)
    else:
        pct_change_3min = pct_change_1min
        
    # 5-minute change (if available)
    if len(btc_data) >= 6:
        prev_price_5min = btc_data['Close'].iloc[-6]
        pct_change_5min = (latest_price / prev_price_5min - 1)
    else:
        pct_change_5min = pct_change_3min
    
    # Calculate the average percentage change (weighted more towards 1-min)
    weighted_pct_change = (0.6 * pct_change_1min) + (0.25 * pct_change_3min) + (0.15 * pct_change_5min)
    
    # Calculate the z-score relative to recent volatility (if we have enough data)
    if len(btc_data) > 30 and 'atr_pct' in btc_data.columns:
        recent_volatility = btc_data['atr_pct'].iloc[-1] / 100
        if recent_volatility > 0:
            z_score = abs(weighted_pct_change) / recent_volatility
        else:
            z_score = abs(weighted_pct_change) / 0.001  # Default when volatility is near zero
    else:
        z_score = abs(weighted_pct_change) / 0.003  # Use fixed denominator if ATR not available
    
    # Determine movement direction and significance
    if weighted_pct_change > MIN_BTC_MOVEMENT_PCT:
        movement = "bullish"
        significance = z_score
    elif weighted_pct_change < -MIN_BTC_MOVEMENT_PCT:
        movement = "bearish"
        significance = z_score
    else:
        movement = "neutral"
        significance = 0
    
    logger.info(f"BTC movement: {movement}, Change: {weighted_pct_change:.4f}, Z-score: {significance:.2f}")
    return significance, movement

def calculate_position_size(correlation, mstr_data, btc_movement_significance):
    """Calculate position size based on correlation strength and technical indicators"""
    global MAX_POSITION_SIZE
    
    # Start with base position size
    position_size = MAX_POSITION_SIZE
    
    # Adjust for correlation
    if correlation > 0.85:
        corr_factor = 1.0
    elif correlation > 0.7:
        corr_factor = 0.8
    elif correlation > 0.5:
        corr_factor = 0.5
    else:
        corr_factor = 0.2
    
    # Adjust for BTC movement significance
    if btc_movement_significance > 2.0:
        movement_factor = 1.5
    elif btc_movement_significance > 1.5:
        movement_factor = 1.25
    elif btc_movement_significance > 1.0:
        movement_factor = 1.0
    else:
        movement_factor = 0.75
    
    # Adjust for RSI (if available)
    rsi_factor = 1.0
    if 'rsi' in mstr_data.columns and not pd.isna(mstr_data['rsi'].iloc[-1]):
        rsi = mstr_data['rsi'].iloc[-1]
        if rsi < 30:
            rsi_factor = 1.0  # Full position for oversold (bullish)
        elif rsi < 40:
            rsi_factor = 0.75
        elif rsi < 50:
            rsi_factor = 0.5
        else:
            rsi_factor = 0.25
    
    # Combine all factors
    adjusted_position_size = position_size * corr_factor * movement_factor * rsi_factor
    
    # Get account value to calculate dollar position
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        dollar_position = equity * adjusted_position_size
        
        # Convert dollar position to shares
        latest_price = mstr_data['Close'].iloc[-1]
        shares = int(dollar_position / latest_price)
        
        logger.info(f"Position sizing: correlation_factor={corr_factor:.2f}, movement_factor={movement_factor:.2f}, rsi_factor={rsi_factor:.2f}")
        logger.info(f"Calculated position: {shares} shares (${dollar_position:.2f})")
        
        return max(1, shares)  # Minimum of 1 share
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 1  # Default to 1 share on error

def check_entry_conditions(btc_data, mstr_data, correlation, movement_significance, movement_direction):
    """Check if entry conditions are met"""
    if len(mstr_data) < 20 or 'rsi' not in mstr_data.columns:
        logger.warning("Not enough data for entry analysis")
        return False, 0
    
    # Extract relevant indicators
    rsi = mstr_data['rsi'].iloc[-1]
    current_price = mstr_data['Close'].iloc[-1]
    bb_lower = mstr_data['bb_lower'].iloc[-1]
    bb_upper = mstr_data['bb_upper'].iloc[-1]
    bb_middle = mstr_data['bb_middle'].iloc[-1]
    
    # MACD indicators
    macd_hist = mstr_data['macd_histogram'].iloc[-1]
    prev_macd_hist = mstr_data['macd_histogram'].iloc[-2] if len(mstr_data) > 2 else 0
    macd_direction = "bullish" if macd_hist > prev_macd_hist else "bearish"
    
    # Default entry size (will be adjusted)
    entry_size = 0
    
    # Check for long entry
    if movement_direction == "bullish" and movement_significance >= 1.0:
        # Base RSI conditions for long
        if rsi < 30:
            rsi_condition = True
            entry_strength = 1.0
        elif rsi < 40:
            rsi_condition = True
            entry_strength = 0.75
        elif rsi < 50:
            rsi_condition = correlation > 0.7  # Higher correlation needed for moderate RSI
            entry_strength = 0.5
        else:
            rsi_condition = correlation > 0.85 and movement_significance > 2.0  # Very high bar for high RSI
            entry_strength = 0.25
        
        # Bollinger Band condition
        bb_condition = current_price < bb_middle
        
        # MACD confirmation
        macd_condition = macd_direction == "bullish"
        
        # Combine conditions
        if rsi_condition and (bb_condition or macd_condition):
            logger.info(f"Long entry conditions met - RSI: {rsi:.2f}, MACD Direction: {macd_direction}")
            
            # Calculate position size based on entry strength
            position_size = calculate_position_size(correlation, mstr_data, movement_significance)
            entry_size = max(1, int(position_size * entry_strength))
            
            return True, entry_size
    
    # Check for short entry
    elif movement_direction == "bearish" and movement_significance >= 1.0:
        # Base RSI conditions for short
        if rsi > 70:
            rsi_condition = True
            entry_strength = 1.0
        elif rsi > 60:
            rsi_condition = True
            entry_strength = 0.75
        elif rsi > 50:
            rsi_condition = correlation > 0.7  # Higher correlation needed for moderate RSI
            entry_strength = 0.5
        else:
            rsi_condition = correlation > 0.85 and movement_significance > 2.0  # Very high bar for low RSI
            entry_strength = 0.25
        
        # Bollinger Band condition
        bb_condition = current_price > bb_middle
        
        # MACD confirmation
        macd_condition = macd_direction == "bearish"
        
        # Combine conditions
        if rsi_condition and (bb_condition or macd_condition):
            logger.info(f"Short entry conditions met - RSI: {rsi:.2f}, MACD Direction: {macd_direction}")
            
            # Calculate position size based on entry strength
            position_size = calculate_position_size(correlation, mstr_data, movement_significance)
            entry_size = max(1, int(position_size * entry_strength))
            
            return True, -entry_size  # Negative for short positions
    
    return False, 0

def check_exit_conditions(btc_data, mstr_data, position, entry_price, entry_time):
    """Check if exit conditions are met for current position"""
    if position == 0 or entry_price == 0:
        return False
    
    current_price = mstr_data['Close'].iloc[-1]
    current_time = datetime.now(pytz.UTC)
    position_duration = (current_time - entry_time).total_seconds() / 60  # in minutes
    
    # Calculate current profit/loss
    if position > 0:  # Long position
        pnl_pct = (current_price / entry_price - 1)
    else:  # Short position
        pnl_pct = (entry_price / current_price - 1)
    
    # 1. Technical exit conditions
    # RSI-based exit
    rsi = mstr_data['rsi'].iloc[-1] if 'rsi' in mstr_data.columns else 50
    
    rsi_exit = False
    if position > 0 and rsi > RSI_OVERBOUGHT:  # Exit long when overbought
        rsi_exit = True
    elif position < 0 and rsi < RSI_OVERSOLD:  # Exit short when oversold
        rsi_exit = True
    
    # 2. Stop-loss
    stop_loss_triggered = pnl_pct < -STOP_LOSS_PCT
    
    # 3. Take-profit
    take_profit_triggered = pnl_pct > TAKE_PROFIT_PCT
    
    # 4. Time-based exit
    time_exit = position_duration > MAX_TRADE_DURATION_MIN
    
    # 5. BTC reversal
    _, btc_direction = detect_btc_movement(btc_data)
    btc_reversal = (position > 0 and btc_direction == "bearish") or (position < 0 and btc_direction == "bullish")
    
    # Logging for monitoring
    logger.info(f"Position evaluation - PnL: {pnl_pct:.4f}, Duration: {position_duration:.1f}min, " +
               f"RSI: {rsi:.2f}, BTC Direction: {btc_direction}")
    
    # Decision logic
    if stop_loss_triggered:
        logger.info(f"⚠️ Stop loss triggered at {pnl_pct:.4f}")
        return True
    elif take_profit_triggered:
        logger.info(f"💰 Take profit triggered at {pnl_pct:.4f}")
        return True
    elif time_exit:
        logger.info(f"⏱️ Time-based exit after {position_duration:.1f} minutes")
        return True
    elif rsi_exit:
        logger.info(f"📊 RSI-based exit at {rsi:.2f}")
        return True
    elif btc_reversal and position_duration > 5:  # Only consider BTC reversal after 5 minutes
        logger.info(f"↩️ BTC reversal exit, direction now {btc_direction}")
        return True
    
    return False

def check_positions():
    """Check current position in the account"""
    try:
        positions = trading_client.get_all_positions()
        for position in positions:
            if position.symbol == ALPACA_SYMBOL:
                return float(position.qty)
        return 0
    except Exception as e:
        logger.error(f"Error checking positions: {e}")
        return 0

def place_order(side, quantity, current_price=None):
    """Place a market order with Alpaca"""
    try:
        if side == 'buy':
            order_side = OrderSide.BUY
        else:
            order_side = OrderSide.SELL

        order_data = MarketOrderRequest(
            symbol=ALPACA_SYMBOL,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.GTC
        )
        
        # Place the order
        order = trading_client.submit_order(order_data=order_data)
        logger.info(f"{side.upper()} order placed for {quantity} {ALPACA_SYMBOL}")
        
        # Create bracket order for take profit and stop loss (for demo/informational purposes)
        if current_price and side == 'buy':
            take_profit_price = current_price * (1 + TAKE_PROFIT_PCT)
            stop_loss_price = current_price * (1 - STOP_LOSS_PCT)
            logger.info(f"  Take profit target: ${take_profit_price:.2f}")
            logger.info(f"  Stop loss level: ${stop_loss_price:.2f}")
        elif current_price and side == 'sell':
            take_profit_price = current_price * (1 - TAKE_PROFIT_PCT)
            stop_loss_price = current_price * (1 + STOP_LOSS_PCT)
            logger.info(f"  Take profit target: ${take_profit_price:.2f}")
            logger.info(f"  Stop loss level: ${stop_loss_price:.2f}")
        
        return order
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return None

def manage_positions(btc_data, mstr_data, correlation):
    """Manage trading positions based on strategy signals"""
    global current_position, entry_price, entry_time, session_pnl, trades_history
    
    # Current MSTR price
    current_price = mstr_data['Close'].iloc[-1]
    
    # Check if our position tracking matches Alpaca's records
    alpaca_position = check_positions()
    if alpaca_position != current_position:
        logger.warning(f"Position mismatch: tracking {current_position} shares but Alpaca reports {alpaca_position}")
        current_position = alpaca_position
        
        # If we have a position in Alpaca but not in our tracking, initialize tracking
        if current_position != 0 and entry_price == 0:
            entry_price = current_price
            entry_time = datetime.now(pytz.UTC)
    
    # 1. Check for exit if we have a position
    if current_position != 0:
        if check_exit_conditions(btc_data, mstr_data, current_position, entry_price, entry_time):
            # Close position
            order = place_order('sell' if current_position > 0 else 'buy', abs(current_position), current_price)
            if order:
                # Calculate P&L
                if current_position > 0:  # Long position
                    trade_pnl = (current_price - entry_price) * current_position
                    pnl_pct = (current_price / entry_price - 1) * 100
                else:  # Short position
                    trade_pnl = (entry_price - current_price) * abs(current_position)
                    pnl_pct = (entry_price / current_price - 1) * 100
                
                # Update session P&L
                session_pnl += trade_pnl
                
                # Record trade
                trade_record = {
                    'entry_time': entry_time,
                    'exit_time': datetime.now(pytz.UTC),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': current_position,
                    'pnl': trade_pnl,
                    'pnl_pct': pnl_pct
                }
                trades_history.append(trade_record)
                
                logger.info(f"Closed position - P&L: ${trade_pnl:.2f} ({pnl_pct:.2f}%)")
                
                # Reset position tracking
                current_position = 0
                entry_price = 0
                entry_time = None
                
                return
    
    # 2. Check for entry only if we don't already have a position
    if current_position == 0:
        # Check BTC movement
        movement_significance, movement_direction = detect_btc_movement(btc_data)
        
        # Check if we should enter a position
        should_enter, position_size = check_entry_conditions(
            btc_data, mstr_data, correlation, movement_significance, movement_direction
        )
        
        if should_enter and position_size != 0:
            # Place order
            order_side = 'buy' if position_size > 0 else 'sell'
            order = place_order(order_side, abs(position_size), current_price)
            
            if order:
                # Update position tracking
                current_position = position_size
                entry_price = current_price
                entry_time = datetime.now(pytz.UTC)
                
                logger.info(f"Entered new position: {position_size} shares at ${entry_price:.2f}")

def display_session_summary():
    """Display a summary of the trading session"""
    global session_pnl, trades_history
    
    logger.info("=" * 60)
    logger.info("TRADING SESSION SUMMARY")
    logger.info("=" * 60)
    
    # Basic statistics
    total_trades = len(trades_history)
    winning_trades = len([t for t in trades_history if t['pnl'] > 0])
    losing_trades = len([t for t in trades_history if t['pnl'] <= 0])
    
    if total_trades > 0:
        win_rate = winning_trades / total_trades * 100
        avg_win = np.mean([t['pnl'] for t in trades_history if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades_history if t['pnl'] <= 0]) if losing_trades > 0 else 0
        avg_pnl = np.mean([t['pnl'] for t in trades_history])
        
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Average Win: ${avg_win:.2f}")
        logger.info(f"Average Loss: ${avg_loss:.2f}")
        logger.info(f"Average P&L per Trade: ${avg_pnl:.2f}")
        logger.info(f"Total Session P&L: ${session_pnl:.2f}")
    else:
        logger.info("No trades executed during this session.")
    
    logger.info("=" * 60)

def handle_exit():
    """Handle clean exit when user interrupts the program"""
    global running, current_position
    
    logger.info("Shutting down... Closing any open positions.")
    
    # Check current position from Alpaca to ensure accuracy
    alpaca_position = check_positions()
    if alpaca_position != 0:
        try:
            order_side = 'sell' if alpaca_position > 0 else 'buy'
            order = place_order(order_side, abs(alpaca_position))
            if order:
                logger.info(f"Closed position of {abs(alpaca_position)} shares on exit.")
        except Exception as e:
            logger.error(f"Error closing position on exit: {e}")
    
    # Get pending orders and cancel them
    try:
        open_orders = trading_client.get_orders(GetOrdersRequest(status=OrderStatus.OPEN))
        for order in open_orders:
            trading_client.cancel_order_by_id(order.id)
            logger.info(f"Canceled pending order: {order.id}")
    except Exception as e:
        logger.error(f"Error canceling orders: {e}")
    
    # Display session summary
    display_session_summary()
    
    running = False

def signal_handler(sig, frame):
    """Signal handler for graceful shutdown"""
    logger.info("Received shutdown signal. Starting clean exit...")
    handle_exit()
    sys.exit(0)

def main():
    """Main trading loop"""
    global running
    
    # Register signal handlers for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting MSTR-BTC lead-lag trading strategy")
    logger.info("Alpaca API: Paper trading environment")
    
    # Check account status
    try:
        account = trading_client.get_account()
        logger.info(f"Trading account ready. Cash available: ${float(account.cash):.2f}")
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {e}")
        return
    
    # Main trading loop
    while running:
        try:
            # Check if market is open
            if not check_market_open():
                market_open, market_close = get_trading_hours()
                now = datetime.now(pytz.UTC)
                
                # If market closed, sleep until next market open
                if now < market_open:
                    time_to_open = (market_open - now).total_seconds()
                    logger.info(f"Market closed. Next open at {market_open.astimezone(pytz.timezone('America/New_York')).strftime('%H:%M:%S')} ET ({time_to_open/60:.1f} minutes)")
                    
                    # If more than 10 minutes to open, sleep for 5 minutes and check again
                    if time_to_open > 600:
                        time.sleep(300)
                    else:
                        time.sleep(60)  # Check every minute when close to open
                    continue
                else:
                    # After market close
                    logger.info("Market closed for the day. Displaying session summary...")
                    display_session_summary()
                    
                    # Sleep until next update
                    time.sleep(300)
                    continue
            
            # Fetch latest data
            btc_data, mstr_data = fetch_latest_data()
            if btc_data is None or mstr_data is None or btc_data.empty or mstr_data.empty:
                logger.warning("Failed to fetch necessary data, retrying in 60 seconds")
                time.sleep(60)
                continue
            
            # Calculate indicators
            btc_data = calculate_indicators(btc_data)
            mstr_data = calculate_indicators(mstr_data)
            
            # Calculate correlation
            correlation = calculate_correlation(btc_data, mstr_data)
            correlation_stability = calculate_correlation_stability(btc_data, mstr_data)
            
            logger.info(f"Current correlation: {correlation:.4f}, Stability: {correlation_stability:.4f}")
            
            # Manage positions based on strategy
            manage_positions(btc_data, mstr_data, correlation)
            
            # Sleep until next check (1 minute for 1-minute bars)
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Sleep on error to prevent rapid retries

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("User interrupted program. Starting clean exit...")
        handle_exit()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        # Try to close positions on error
        handle_exit() 
