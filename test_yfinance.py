#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("yfinance_test")

def get_crypto_data(symbol, period="30d", interval="1d", plot=False):
    """
    Fetch cryptocurrency data using yfinance and optionally plot it.
    
    Args:
        symbol: Cryptocurrency symbol (e.g. BTC-USD, ETH-USD)
        period: Time period (e.g. 1d, 7d, 30d, 90d, 1y, max)
        interval: Data interval (e.g. 1m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)
        plot: Whether to generate a price chart
    
    Returns:
        DataFrame with historical data
    """
    # Convert to Yahoo Finance format if necessary
    if '/' in symbol:
        symbol = symbol.replace('/', '-')
        
    logger.info(f"Fetching data for {symbol} (period: {period}, interval: {interval})")
    
    try:
        # Fetch data
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if data.empty:
            logger.error(f"No data available for {symbol}")
            return None
        
        # Print data info
        logger.info(f"Retrieved {len(data)} data points from {data.index[0]} to {data.index[-1]}")
        
        # Safely format price values - properly handle Series objects
        latest_price = float(data['Close'].iloc[-1].iloc[0] if hasattr(data['Close'].iloc[-1], 'iloc') else data['Close'].iloc[-1])
        logger.info(f"Latest price: ${latest_price:.2f}")
        
        # Calculate price change
        first_price = float(data['Close'].iloc[0].iloc[0] if hasattr(data['Close'].iloc[0], 'iloc') else data['Close'].iloc[0])
        price_change = ((latest_price - first_price) / first_price) * 100 if first_price > 0 else 0
        logger.info(f"Price change over period: {price_change:.2f}%")
        
        # Plot if requested
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['Close'])
            plt.title(f"{symbol} Price ({period}, {interval})")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            filename = f"{symbol.replace('-', '')}_chart_{period}_{interval}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename)
            logger.info(f"Chart saved as {filename}")
            
            try:
                plt.show()
            except:
                logger.warning("Could not display plot in current environment")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_multiple_timeframes(symbol):
    """
    Test fetching data in multiple timeframes for a symbol.
    
    Args:
        symbol: Cryptocurrency symbol to test
    """
    print("\n" + "="*50)
    print(f"TESTING MULTIPLE TIMEFRAMES FOR {symbol}")
    print("="*50)
    
    # Test different timeframes
    timeframes = [
        ("1d", "5m"),   # Short-term (1 day with 5-minute intervals)
        ("7d", "1h"),   # Short-term (1 week with 1-hour intervals)
        ("30d", "1d"),  # Medium-term (1 month with daily intervals)
        ("90d", "1d"),  # Long-term (3 months with daily intervals)
        ("1y", "1wk")   # Very long-term (1 year with weekly intervals)
    ]
    
    for period, interval in timeframes:
        print(f"\nTimeframe: {period} with {interval} intervals")
        print("-"*50)
        
        data = get_crypto_data(symbol, period=period, interval=interval)
        
        if data is not None:
            print(f"Data shape: {data.shape}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            # Safely format the latest data point
            try:
                # Extract the latest row safely
                latest_row = data.iloc[-1]
                
                # Convert to dictionary safely
                if isinstance(latest_row, pd.Series):
                    if isinstance(latest_row.index, pd.MultiIndex):
                        # Handle multi-index Series
                        formatted_row = {}
                        for idx, val in latest_row.items():
                            formatted_row[str(idx)] = f"{float(val):.2f}"
                    else:
                        # Standard Series
                        formatted_row = {k: f"{float(v):.2f}" for k, v in latest_row.items()}
                else:
                    formatted_row = {str(k): f"{float(v):.2f}" for k, v in latest_row.to_dict().items()}
                    
                print(f"Latest OHLCV: {formatted_row}")
            except Exception as e:
                print(f"Error formatting latest data: {str(e)}")
                print(f"Raw latest data: {data.iloc[-1]}")
        else:
            print(f"Failed to retrieve data for {period}/{interval}")
        
        print("-"*50)

def main():
    parser = argparse.ArgumentParser(description='Test yfinance cryptocurrency data fetching')
    parser.add_argument('--symbol', type=str, default='BTC-USD', 
                        help='Cryptocurrency symbol (e.g. BTC-USD, ETH-USD)')
    parser.add_argument('--period', type=str, default='30d',
                        help='Time period (e.g. 1d, 7d, 30d, 90d, 1y, max)')
    parser.add_argument('--interval', type=str, default='1d',
                        help='Data interval (e.g. 1m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate price chart')
    parser.add_argument('--test-all', action='store_true',
                        help='Test all timeframes')
    
    args = parser.parse_args()
    
    # Process symbol format if needed
    symbol = args.symbol
    
    print("\n" + "="*70)
    print(f"  YFINANCE CRYPTOCURRENCY DATA TEST")
    print(f"  Symbol: {symbol}")
    print("="*70 + "\n")
    
    if args.test_all:
        test_multiple_timeframes(symbol)
    else:
        data = get_crypto_data(symbol, args.period, args.interval, args.plot)
        if data is not None:
            # Print the first and last 5 rows
            print("\nFirst 5 rows:")
            print(data.head())
            print("\nLast 5 rows:")
            print(data.tail())

if __name__ == "__main__":
    main() 