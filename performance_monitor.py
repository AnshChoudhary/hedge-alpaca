#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def load_trade_history(file_path="trade_history.json"):
    """Load the trading history from the JSON file."""
    if not os.path.exists(file_path):
        print(f"Error: History file {file_path} not found.")
        return None
    
    with open(file_path, 'r') as f:
        try:
            history = json.load(f)
            print(f"Loaded {len(history)} trade records.")
            return history
        except json.JSONDecodeError:
            print(f"Error: Could not parse {file_path} as JSON.")
            return None

def create_trade_dataframe(history):
    """Convert trade history to a pandas DataFrame with proper date formatting."""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    # Convert timestamp strings to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df

def analyze_performance(df):
    """Analyze the trading performance from the DataFrame."""
    if df is None or df.empty:
        print("No trade data available for analysis.")
        return
    
    # Filter for actual buys and sells (exclude holds)
    trades_df = df[df['action'].isin(['buy', 'sell'])]
    
    # Statistics by symbol
    by_symbol = df.groupby('symbol')
    
    print("\n==== PERFORMANCE SUMMARY ====\n")
    
    # Overall statistics
    total_trades = len(trades_df)
    total_buys = len(df[df['action'] == 'buy'])
    total_sells = len(df[df['action'] == 'sell'])
    total_holds = len(df[df['action'] == 'hold'])
    
    print(f"Total Records: {len(df)}")
    print(f"Total Trades: {total_trades}")
    print(f"Buy Actions: {total_buys}")
    print(f"Sell Actions: {total_sells}")
    print(f"Hold Actions: {total_holds}")
    
    # Calculate profit/loss for completed trades
    symbols = df['symbol'].unique()
    total_profit = 0
    total_cost = 0
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_trades = symbol_df[symbol_df['action'].isin(['buy', 'sell'])]
        
        print(f"\n---- {symbol} ----")
        print(f"Records: {len(symbol_df)}")
        print(f"Buys: {len(symbol_df[symbol_df['action'] == 'buy'])}")
        print(f"Sells: {len(symbol_df[symbol_df['action'] == 'sell'])}")
        print(f"Holds: {len(symbol_df[symbol_df['action'] == 'hold'])}")
        
        # Skip if not enough trades
        if len(symbol_trades) < 2:
            print("Not enough trades to calculate P&L")
            continue
        
        # Calculate profit/loss
        buys = symbol_df[symbol_df['action'] == 'buy']
        sells = symbol_df[symbol_df['action'] == 'sell']
        
        if buys.empty or sells.empty:
            print("No complete trades (buy and sell)")
            continue
        
        # Calculate total cost and revenue
        total_buy_cost = (buys['price'] * buys['quantity']).sum()
        total_sell_revenue = (sells['price'] * sells['quantity']).sum()
        
        # Add to running total
        total_cost += total_buy_cost
        total_profit += (total_sell_revenue - total_buy_cost)
        
        # Calculate P&L
        profit = total_sell_revenue - total_buy_cost
        profit_percent = (profit / total_buy_cost * 100) if total_buy_cost > 0 else 0
        
        print(f"Total Buy Cost: ${total_buy_cost:.2f}")
        print(f"Total Sell Revenue: ${total_sell_revenue:.2f}")
        print(f"Profit/Loss: ${profit:.2f} ({profit_percent:.2f}%)")
    
    # Overall P&L
    if total_cost > 0:
        overall_profit_percent = (total_profit / total_cost * 100)
        print(f"\n==== OVERALL P&L ====")
        print(f"Total Profit/Loss: ${total_profit:.2f} ({overall_profit_percent:.2f}%)")
    
    return df

def plot_performance(df):
    """Plot performance metrics from the trading data."""
    if df is None or df.empty:
        print("No data available for plotting.")
        return
    
    # Set up the plots
    plt.figure(figsize=(12, 16))
    
    # 1. Action counts by symbol
    plt.subplot(4, 1, 1)
    action_counts = df.groupby(['symbol', 'action']).size().unstack()
    action_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Actions by Symbol')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # 2. Price history with buy/sell markers
    plt.subplot(4, 1, 2)
    
    symbols = df['symbol'].unique()
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol]
        
        # Plot price history
        plt.plot(symbol_df['timestamp'], symbol_df['price'], label=symbol, alpha=0.7)
        
        # Mark buys and sells
        buys = symbol_df[symbol_df['action'] == 'buy']
        sells = symbol_df[symbol_df['action'] == 'sell']
        
        plt.scatter(buys['timestamp'], buys['price'], color='green', marker='^', s=100, label=f'{symbol} Buy' if len(buys) > 0 else None)
        plt.scatter(sells['timestamp'], sells['price'], color='red', marker='v', s=100, label=f'{symbol} Sell' if len(sells) > 0 else None)
    
    plt.title('Price History with Buy/Sell Markers')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Trading activity over time
    plt.subplot(4, 1, 3)
    activity = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'action']).size().unstack().fillna(0)
    activity.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Trading Activity Over Time')
    plt.ylabel('Number of Actions')
    plt.grid(True, alpha=0.3)
    
    # 4. Price distribution
    plt.subplot(4, 1, 4)
    for symbol in symbols:
        symbol_prices = df[df['symbol'] == symbol]['price']
        if not symbol_prices.empty:
            symbol_prices.plot(kind='hist', alpha=0.5, bins=20, label=symbol)
    
    plt.title('Price Distribution')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    print("Performance chart saved as 'performance_analysis.png'")
    
    # Display the plot if not in a headless environment
    try:
        plt.show()
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='Analyze trading bot performance')
    parser.add_argument('--file', type=str, default='trade_history.json',
                        help='Path to the trade history JSON file')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting (text analysis only)')
    
    args = parser.parse_args()
    
    # Load and analyze the trade history
    history = load_trade_history(args.file)
    if history:
        df = create_trade_dataframe(history)
        analyze_performance(df)
        
        if not args.no_plot:
            try:
                plot_performance(df)
            except Exception as e:
                print(f"Error creating plots: {str(e)}")
                print("Try running with --no-plot if you're in a headless environment.")

if __name__ == "__main__":
    main() 