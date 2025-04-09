#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional

def calculate_sma(data: pd.DataFrame, period: int = 20, price_col: str = 'close') -> pd.Series:
    """Calculate Simple Moving Average (SMA).
    
    Args:
        data: DataFrame with price data
        period: Period for SMA calculation
        price_col: Column name for price data
        
    Returns:
        Series with SMA values
    """
    return data[price_col].rolling(window=period).mean()

def calculate_ema(data: pd.DataFrame, period: int = 20, price_col: str = 'close') -> pd.Series:
    """Calculate Exponential Moving Average (EMA).
    
    Args:
        data: DataFrame with price data
        period: Period for EMA calculation
        price_col: Column name for price data
        
    Returns:
        Series with EMA values
    """
    return data[price_col].ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
    """Calculate Relative Strength Index (RSI).
    
    Args:
        data: DataFrame with price data
        period: Period for RSI calculation
        price_col: Column name for price data
        
    Returns:
        Series with RSI values
    """
    # Calculate price changes
    delta = data[price_col].diff()
    
    # Calculate gains and losses
    gain = delta.copy()
    gain[gain < 0] = 0
    loss = delta.copy()
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                  signal_period: int = 9, price_col: str = 'close') -> Dict[str, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data: DataFrame with price data
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        price_col: Column name for price data
        
    Returns:
        Dictionary with MACD line, signal line, and histogram
    """
    # Calculate fast and slow EMAs
    fast_ema = data[price_col].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data[price_col].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                             price_col: str = 'close') -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands.
    
    Args:
        data: DataFrame with price data
        period: Period for moving average
        std_dev: Number of standard deviations
        price_col: Column name for price data
        
    Returns:
        Dictionary with upper band, middle band, and lower band
    """
    # Calculate middle band (SMA)
    middle_band = data[price_col].rolling(window=period).mean()
    
    # Calculate standard deviation
    std = data[price_col].rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return {
        'upper_band': upper_band,
        'middle_band': middle_band,
        'lower_band': lower_band
    }

def calculate_stochastic_oscillator(data: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                                  price_col: str = 'close') -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator.
    
    Args:
        data: DataFrame with price data
        k_period: Period for %K line
        d_period: Period for %D line
        price_col: Column name for price data
        
    Returns:
        Dictionary with %K and %D lines
    """
    # Calculate %K
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    
    k = 100 * ((data[price_col] - low_min) / (high_max - low_min))
    
    # Calculate %D
    d = k.rolling(window=d_period).mean()
    
    return {
        'k': k,
        'd': d
    }

def calculate_fibonacci_retracement(data: pd.DataFrame, high_col: str = 'high', 
                                   low_col: str = 'low') -> Dict[str, float]:
    """Calculate Fibonacci Retracement levels.
    
    Args:
        data: DataFrame with price data
        high_col: Column name for high price
        low_col: Column name for low price
        
    Returns:
        Dictionary with Fibonacci retracement levels
    """
    # Get the highest high and lowest low in the period
    highest_high = data[high_col].max()
    lowest_low = data[low_col].min()
    
    # Calculate the difference
    difference = highest_high - lowest_low
    
    # Calculate Fibonacci retracement levels
    levels = {
        '0.0': lowest_low,
        '0.236': lowest_low + 0.236 * difference,
        '0.382': lowest_low + 0.382 * difference,
        '0.5': lowest_low + 0.5 * difference,
        '0.618': lowest_low + 0.618 * difference,
        '0.786': lowest_low + 0.786 * difference,
        '1.0': highest_high
    }
    
    return levels

def detect_support_resistance(data: pd.DataFrame, price_col: str = 'close', 
                            window: int = 5) -> Dict[str, List[float]]:
    """Detect support and resistance levels.
    
    Args:
        data: DataFrame with price data
        price_col: Column name for price data
        window: Window size for peak detection
        
    Returns:
        Dictionary with support and resistance levels
    """
    # Initialize lists for support and resistance levels
    support_levels = []
    resistance_levels = []
    
    # Convert to numpy array for easier manipulation
    prices = data[price_col].values
    
    # Loop through prices to find local minima (support) and local maxima (resistance)
    for i in range(window, len(prices) - window):
        # Check if current price is a local minimum (support)
        if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] <= prices[i+j] for j in range(1, window+1)):
            support_levels.append(prices[i])
        
        # Check if current price is a local maximum (resistance)
        if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] >= prices[i+j] for j in range(1, window+1)):
            resistance_levels.append(prices[i])
    
    return {
        'support': support_levels,
        'resistance': resistance_levels
    }

def generate_trading_signals(data: pd.DataFrame) -> Dict[str, str]:
    """Generate trading signals using multiple indicators.
    
    Args:
        data: DataFrame with price data and calculated indicators
        
    Returns:
        Dictionary with trading signals for different strategies
    """
    signals = {}
    
    # Make sure we have enough data
    if len(data) < 50:
        return {"error": "Not enough data for reliable signals"}
    
    # Get the most recent data
    current_data = data.iloc[-1]
    prev_data = data.iloc[-2]
    
    # Calculate indicators if they don't exist
    if 'sma_20' not in data.columns:
        data['sma_20'] = calculate_sma(data, period=20)
    if 'sma_50' not in data.columns:
        data['sma_50'] = calculate_sma(data, period=50)
    if 'rsi' not in data.columns:
        data['rsi'] = calculate_rsi(data)
    
    # MACD signal
    if 'macd_line' not in data.columns or 'signal_line' not in data.columns:
        macd_data = calculate_macd(data)
        data['macd_line'] = macd_data['macd_line']
        data['signal_line'] = macd_data['signal_line']
        data['macd_histogram'] = macd_data['histogram']
    
    # Check for MACD crossover
    if data['macd_line'].iloc[-2] < data['signal_line'].iloc[-2] and \
       data['macd_line'].iloc[-1] > data['signal_line'].iloc[-1]:
        signals['macd'] = 'buy'
    elif data['macd_line'].iloc[-2] > data['signal_line'].iloc[-2] and \
         data['macd_line'].iloc[-1] < data['signal_line'].iloc[-1]:
        signals['macd'] = 'sell'
    else:
        signals['macd'] = 'hold'
    
    # Check for Golden Cross / Death Cross (SMA)
    if data['sma_20'].iloc[-2] < data['sma_50'].iloc[-2] and \
       data['sma_20'].iloc[-1] > data['sma_50'].iloc[-1]:
        signals['sma_cross'] = 'buy'  # Golden Cross
    elif data['sma_20'].iloc[-2] > data['sma_50'].iloc[-2] and \
         data['sma_20'].iloc[-1] < data['sma_50'].iloc[-1]:
        signals['sma_cross'] = 'sell'  # Death Cross
    else:
        signals['sma_cross'] = 'hold'
    
    # Check RSI for overbought/oversold conditions
    if data['rsi'].iloc[-1] < 30:
        signals['rsi'] = 'buy'  # Oversold
    elif data['rsi'].iloc[-1] > 70:
        signals['rsi'] = 'sell'  # Overbought
    else:
        signals['rsi'] = 'hold'  # Neutral
    
    # Price action signal
    if data['close'].iloc[-1] > data['sma_20'].iloc[-1]:
        signals['price_action'] = 'bullish'
    else:
        signals['price_action'] = 'bearish'
    
    # Volume signal
    if 'volume' in data.columns:
        avg_volume = data['volume'].iloc[-10:].mean()
        if data['volume'].iloc[-1] > avg_volume * 1.5:
            signals['volume'] = 'high'
        elif data['volume'].iloc[-1] < avg_volume * 0.5:
            signals['volume'] = 'low'
        else:
            signals['volume'] = 'normal'
    
    # Overall signal (simple strategy combining indicators)
    buy_signals = sum(1 for signal in signals.values() if signal in ['buy', 'bullish', 'high'])
    sell_signals = sum(1 for signal in signals.values() if signal in ['sell', 'bearish'])
    
    if buy_signals > sell_signals + 1:
        signals['overall'] = 'strong_buy'
    elif buy_signals > sell_signals:
        signals['overall'] = 'buy'
    elif sell_signals > buy_signals + 1:
        signals['overall'] = 'strong_sell'
    elif sell_signals > buy_signals:
        signals['overall'] = 'sell'
    else:
        signals['overall'] = 'hold'
    
    return signals

def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for analysis by calculating all indicators.
    
    Args:
        df: DataFrame with price data (open, high, low, close, volume)
        
    Returns:
        DataFrame with all indicators calculated
    """
    # Ensure the DataFrame has the necessary columns
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate SMAs
    data['sma_20'] = calculate_sma(data, period=20)
    data['sma_50'] = calculate_sma(data, period=50)
    data['sma_200'] = calculate_sma(data, period=200)
    
    # Calculate EMAs
    data['ema_12'] = calculate_ema(data, period=12)
    data['ema_26'] = calculate_ema(data, period=26)
    
    # Calculate RSI
    data['rsi'] = calculate_rsi(data)
    
    # Calculate MACD
    macd_data = calculate_macd(data)
    data['macd_line'] = macd_data['macd_line']
    data['signal_line'] = macd_data['signal_line']
    data['macd_histogram'] = macd_data['histogram']
    
    # Calculate Bollinger Bands
    bb_data = calculate_bollinger_bands(data)
    data['bb_upper'] = bb_data['upper_band']
    data['bb_middle'] = bb_data['middle_band']
    data['bb_lower'] = bb_data['lower_band']
    
    # Calculate Stochastic Oscillator
    stoch_data = calculate_stochastic_oscillator(data)
    data['stoch_k'] = stoch_data['k']
    data['stoch_d'] = stoch_data['d']
    
    # Calculate price percent change
    data['daily_return'] = data['close'].pct_change() * 100
    
    # Calculate Average True Range (ATR)
    data['tr1'] = abs(data['high'] - data['low'])
    data['tr2'] = abs(data['high'] - data['close'].shift())
    data['tr3'] = abs(data['low'] - data['close'].shift())
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr'] = data['true_range'].rolling(window=14).mean()
    
    # Clean up temporary columns
    data = data.drop(['tr1', 'tr2', 'tr3'], axis=1)
    
    return data 