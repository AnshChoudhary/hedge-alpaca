#!/usr/bin/env python3
import os
import argparse
import json
import time
import logging
import sys
import textwrap
from tabulate import tabulate
from typing import Dict, List
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd

# Import custom modules
from alpaca_crypto import AlpacaCryptoTrader, CryptoPortfolioAnalyzer, CryptoTradingStrategies, normalize_crypto_symbol

# Try to import colorful terminal libraries if available
try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

# Constants for colorized output
GREEN = Fore.GREEN if HAS_COLOR else ""
RED = Fore.RED if HAS_COLOR else ""
YELLOW = Fore.YELLOW if HAS_COLOR else ""
BLUE = Fore.BLUE if HAS_COLOR else ""
RESET = Style.RESET_ALL if HAS_COLOR else ""
BOLD = Style.BRIGHT if HAS_COLOR else ""

def setup_logger():
    """Set up the logger for CLI output."""
    logger = logging.getLogger("crypto_cli")
    logger.setLevel(logging.INFO)
    
    # Create a custom formatter with colors if available
    if HAS_COLOR:
        class ColoredFormatter(logging.Formatter):
            FORMATS = {
                logging.DEBUG: f"{BLUE}%(asctime)s - %(name)s - %(levelname)s - %(message)s{RESET}",
                logging.INFO: f"%(asctime)s - %(name)s - {GREEN}%(levelname)s{RESET} - %(message)s",
                logging.WARNING: f"%(asctime)s - %(name)s - {YELLOW}%(levelname)s{RESET} - %(message)s",
                logging.ERROR: f"%(asctime)s - %(name)s - {RED}%(levelname)s{RESET} - %(message)s",
                logging.CRITICAL: f"%(asctime)s - %(name)s - {RED}{BOLD}%(levelname)s{RESET} - %(message)s",
            }

            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt)
                return formatter.format(record)
                
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(ColoredFormatter())
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()

def check_credentials():
    """Check if the required environment variables are set."""
    key_id = os.environ.get('ALPACA_API_KEY_ID')
    secret_key = os.environ.get('ALPACA_API_SECRET_KEY')
    
    if not key_id or not secret_key:
        logger.error("Error: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables must be set.")
        logger.error("Example: export ALPACA_API_KEY_ID='your-key'")
        logger.error("         export ALPACA_API_SECRET_KEY='your-secret'")
        sys.exit(1)
    
    return True

def display_account_info(trader: AlpacaCryptoTrader):
    """Display account information in a table format."""
    print_header("Account Information")
    
    show_spinner("Fetching account information...", 0.1)
    account = trader.get_account()
    
    # Prepare data for tabulation
    data = [
        ["Account ID", account.id],
        ["Status", f"{GREEN if account.status == 'ACTIVE' else RED}{account.status}{RESET}" if HAS_COLOR else account.status],
        ["Currency", account.currency],
        ["Cash", f"{GREEN if float(account.cash) > 0 else RED}${float(account.cash):,.2f}{RESET}" if HAS_COLOR else f"${float(account.cash):,.2f}"],
        ["Portfolio Value", f"{GREEN if float(account.equity) > 0 else RED}${float(account.equity):,.2f}{RESET}" if HAS_COLOR else f"${float(account.equity):,.2f}"],
        ["Buying Power", f"{GREEN}${float(account.buying_power):,.2f}{RESET}" if HAS_COLOR else f"${float(account.buying_power):,.2f}"],
        ["Initial Margin", f"${float(account.initial_margin):,.2f}"],
        ["Maintenance Margin", f"${float(account.maintenance_margin):,.2f}"],
        ["Daytrade Count", account.daytrade_count],
        ["Last Equity", f"${float(account.last_equity):,.2f}"],
        ["Pattern Day Trader", f"{RED}Yes{RESET}" if account.pattern_day_trader and HAS_COLOR else ("Yes" if account.pattern_day_trader else "No")],
        ["Trading Blocked", f"{RED}Yes{RESET}" if account.trading_blocked and HAS_COLOR else ("Yes" if account.trading_blocked else "No")],
        ["Account Blocked", f"{RED}Yes{RESET}" if account.account_blocked and HAS_COLOR else ("Yes" if account.account_blocked else "No")],
        ["Created", account.created_at]
    ]
    
    print(tabulate(data, tablefmt="grid"))

def display_positions(trader: AlpacaCryptoTrader):
    """Display current positions in a table format."""
    print_header("Crypto Positions")
    
    show_spinner("Fetching positions...", 0.1)
    positions = trader.get_positions()
    
    if not positions:
        print_warning("No open positions.")
        return
    
    # Prepare data for tabulation
    headers = ["Symbol", "Qty", "Entry Price", "Current Price", "Market Value", "Cost Basis", "Unrealized P&L", "P&L %"]
    data = []
    
    for pos in positions:
        symbol = pos.symbol
        qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price)
        current_price = float(pos.current_price)
        market_value = float(pos.market_value)
        cost_basis = float(pos.cost_basis)
        unrealized_pl = float(pos.unrealized_pl)
        unrealized_plpc = float(pos.unrealized_plpc) * 100  # Convert to percentage
        
        # Format crypto symbol with slash for display
        if len(symbol) >= 6 and '/' not in symbol:
            base_currency = symbol[0:3]
            quote_currency = symbol[3:]
            display_symbol = f"{base_currency}/{quote_currency}"
        else:
            display_symbol = symbol

        # Apply colors to P&L if available
        if HAS_COLOR:
            pl_color = GREEN if unrealized_pl >= 0 else RED
            pl_pct_color = GREEN if unrealized_plpc >= 0 else RED
            
            pl_formatted = f"{pl_color}${unrealized_pl:,.2f}{RESET}"
            pl_pct_formatted = f"{pl_pct_color}{unrealized_plpc:,.2f}%{RESET}"
        else:
            pl_formatted = f"${unrealized_pl:,.2f}"
            pl_pct_formatted = f"{unrealized_plpc:,.2f}%"
        
        data.append([
            f"{BOLD}{display_symbol}{RESET}" if HAS_COLOR else display_symbol,
            f"{qty:,.8f}",  # More decimal places for crypto
            f"${entry_price:,.2f}",
            f"${current_price:,.2f}",
            f"${market_value:,.2f}",
            f"${cost_basis:,.2f}",
            pl_formatted,
            pl_pct_formatted
        ])
    
    print(tabulate(data, headers=headers, tablefmt="grid"))

def display_orders(trader: AlpacaCryptoTrader, status="open"):
    """Display orders in a table format."""
    orders = trader.get_orders(status=status, limit=50)
    
    if not orders:
        print(f"\n=== {status.capitalize()} Orders ===")
        print(f"No {status} orders.")
        return
    
    # Prepare data for tabulation
    headers = ["Symbol", "Side", "Type", "Qty", "Filled Qty", "Price", "Status", "Created At"]
    data = []
    
    for order in orders:
        if not hasattr(order, 'symbol') or not order.symbol:
            continue  # Skip orders without symbol
            
        symbol = order.symbol
        side = order.side.name
        order_type = order.type.name
        qty = order.qty
        filled_qty = order.filled_qty
        limit_price = order.limit_price if hasattr(order, 'limit_price') and order.limit_price else 'N/A'
        status = order.status.name
        created_at = order.created_at
        
        data.append([
            symbol,
            side,
            order_type,
            qty,
            filled_qty,
            limit_price,
            status,
            created_at
        ])
    
    print(f"\n=== {status.capitalize()} Orders ===")
    print(tabulate(data, headers=headers, tablefmt="grid"))

def display_portfolio_summary(portfolio_analyzer: CryptoPortfolioAnalyzer):
    """Display portfolio summary in a table format."""
    summary = portfolio_analyzer.get_portfolio_summary()
    
    if 'error' in summary:
        print("\n=== Portfolio Summary ===")
        print(summary['error'])
        return
    
    # Prepare data for tabulation
    data = [
        ["Equity", f"${summary['equity']:,.2f}"],
        ["Cash", f"${summary['cash']:,.2f}"],
        ["Buying Power", f"${summary['buying_power']:,.2f}"],
        ["Position Value", f"${summary['position_value']:,.2f}"],
        ["Number of Positions", summary['position_count']],
        ["Cash Allocation", f"{summary['cash_allocation']:,.2f}%"],
        ["Position Allocation", f"{summary['position_allocation']:,.2f}%"]
    ]
    
    print("\n=== Crypto Portfolio Summary ===")
    print(tabulate(data, tablefmt="grid"))
    
    if summary['position_count'] > 0:
        headers = ["Symbol", "Allocation %"]
        
        # Format positions with display symbols
        allocation_data = []
        for symbol, alloc in summary['positions'].items():
            # Format crypto symbol with slash for display
            if len(symbol) >= 6 and '/' not in symbol:
                base_currency = symbol[0:3]
                quote_currency = symbol[3:]
                display_symbol = f"{base_currency}/{quote_currency}"
            else:
                display_symbol = symbol
                
            allocation_data.append([display_symbol, f"{alloc:.2f}%"])
        
        print("\n=== Position Allocations ===")
        print(tabulate(allocation_data, headers=headers, tablefmt="grid"))

def display_portfolio_metrics(portfolio_analyzer: CryptoPortfolioAnalyzer):
    """Display portfolio performance metrics in a table format."""
    metrics = portfolio_analyzer.calculate_portfolio_metrics()
    
    if 'error' in metrics:
        print("\n=== Portfolio Metrics ===")
        print(metrics['error'])
        return
    
    # Prepare data for tabulation
    data = [
        ["Current Equity", f"${metrics['current_equity']:,.2f}"],
        ["Starting Equity", f"${metrics['starting_equity']:,.2f}"],
        ["Total Return", f"{metrics['total_return']*100:,.2f}%"],
        ["Annual Return", f"{metrics['annual_return']*100:,.2f}%"],
        ["Annual Volatility", f"{metrics['annual_volatility']*100:,.2f}%"],
        ["Sharpe Ratio", f"{metrics['sharpe_ratio']:,.2f}"],
        ["Max Drawdown", f"{metrics['max_drawdown']*100:,.2f}%"],
        ["Win Rate", f"{metrics['win_rate']*100:,.2f}%"],
        ["Avg Daily Return", f"{metrics['avg_daily_return']*100:,.6f}%"],
        ["Total Trading Days", metrics['total_days']],
        ["Positive Days", metrics['positive_days']],
        ["Negative Days", metrics['negative_days']]
    ]
    
    print("\n=== Crypto Portfolio Performance Metrics ===")
    print(tabulate(data, tablefmt="grid"))

def execute_ma_strategy(trader: AlpacaCryptoTrader, strategy: CryptoTradingStrategies, symbols: List[str]):
    """Execute moving average crossover strategy for crypto."""
    print(f"\n=== Executing Moving Average Crossover Strategy on {len(symbols)} crypto symbols ===")
    signals = strategy.moving_average_crossover_strategy(symbols)
    
    if not signals:
        print("No trading signals generated.")
        return
    
    headers = ["Symbol", "Action", "Price", "Quantity", "Short MA", "Long MA"]
    data = []
    
    for signal in signals:
        data.append([
            signal['symbol'],
            signal['action'],
            f"${signal['price']:,.2f}",
            f"{signal['qty']:,.8f}",
            f"{signal['short_ma']:,.2f}",
            f"{signal['long_ma']:,.2f}"
        ])
    
    print("\n=== Trading Signals ===")
    print(tabulate(data, headers=headers, tablefmt="grid"))

def execute_rsi_strategy(trader: AlpacaCryptoTrader, strategy: CryptoTradingStrategies, symbols: List[str]):
    """Execute RSI strategy for crypto."""
    print(f"\n=== Executing RSI Strategy on {len(symbols)} crypto symbols ===")
    signals = strategy.rsi_strategy(symbols)
    
    if not signals:
        print("No trading signals generated.")
        return
    
    headers = ["Symbol", "Action", "Price", "Quantity", "RSI"]
    data = []
    
    for signal in signals:
        data.append([
            signal['symbol'],
            signal['action'],
            f"${signal['price']:,.2f}",
            f"{signal['qty']:,.8f}",
            f"{signal['rsi']:,.2f}"
        ])
    
    print("\n=== Trading Signals ===")
    print(tabulate(data, headers=headers, tablefmt="grid"))

def place_order(trader: AlpacaCryptoTrader, symbol: str, side: str, qty: float = None, 
               notional: float = None, order_type: str = "market", limit_price: float = None):
    """Place a crypto order."""
    try:
        if not (qty or notional):
            print_error("Either quantity or notional value must be provided.")
            return
        
        # Get current price for the symbol
        try:
            bars = trader.get_crypto_bars([symbol], "1D", limit=1)
            if not bars.empty:
                # Extract the current price
                if 'symbol' in bars.index.names:
                    available_symbols = bars.index.get_level_values('symbol').unique()
                    found_symbol = None
                    
                    for available in available_symbols:
                        normalized = normalize_crypto_symbol(symbol)
                        if available == normalized["with_slash"] or available == normalized["without_slash"]:
                            found_symbol = available
                            break
                    
                    if found_symbol:
                        current_price = bars.xs(found_symbol, level='symbol')['close'].iloc[-1]
                    else:
                        current_price = None
                else:
                    current_price = bars['close'].iloc[-1]
                
                # Calculate order value
                if qty:
                    value = qty * current_price if current_price else "Unknown"
                    value_str = f"${value:,.2f}" if isinstance(value, (int, float)) else value
                    order_size = f"{qty} {symbol} (approx. {value_str})"
                else:
                    value_str = f"${notional:,.2f}"
                    if current_price:
                        est_qty = notional / current_price
                        order_size = f"${notional:,.2f} (approx. {est_qty:,.8f} {symbol})"
                    else:
                        order_size = f"${notional:,.2f}"
            else:
                order_size = f"{qty} {symbol}" if qty else f"${notional:,.2f}"
                current_price = None
        except Exception as e:
            logger.error(f"Error getting price for confirmation: {str(e)}")
            order_size = f"{qty} {symbol}" if qty else f"${notional:,.2f}"
            current_price = None

        # Confirm the order with the user
        side_display = f"{GREEN}BUY{RESET}" if side.lower() == "buy" and HAS_COLOR else (
                      f"{RED}SELL{RESET}" if side.lower() == "sell" and HAS_COLOR else side.upper())
        
        order_info = (f"Order Summary:\n"
                     f"  Symbol: {BOLD}{symbol}{RESET if HAS_COLOR else ''}\n"
                     f"  Side: {side_display}\n"
                     f"  Type: {order_type.upper()}\n"
                     f"  Size: {BOLD}{order_size}{RESET if HAS_COLOR else ''}")
        
        if order_type.lower() == "limit" and limit_price:
            order_info += f"\n  Limit Price: ${limit_price:,.2f}"
            
        if current_price:
            order_info += f"\n  Current Market Price: ${current_price:,.2f}"
            
        print(order_info)
        
        # Ask for confirmation
        if not confirm_action("Do you want to place this order?"):
            print_warning("Order cancelled.")
            return
            
        # Show processing indicator
        show_spinner("Submitting order...", 0.1)
            
        order = trader.submit_crypto_order(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            time_in_force="gtc"
        )
        
        print_header("Order Placed")
        
        data = [
            ["Order ID", order.id],
            ["Symbol", order.symbol],
            ["Side", f"{GREEN}{order.side.name}{RESET}" if order.side.name == "BUY" and HAS_COLOR else 
                   (f"{RED}{order.side.name}{RESET}" if order.side.name == "SELL" and HAS_COLOR else order.side.name)],
            ["Type", order.type.name],
            ["Qty", order.qty],
            ["Status", f"{GREEN}{order.status.name}{RESET}" if HAS_COLOR else order.status.name],
            ["Created At", order.created_at]
        ]
        
        if hasattr(order, 'limit_price') and order.limit_price:
            data.append(["Limit Price", f"${float(order.limit_price):,.2f}"])
            
        print(tabulate(data, tablefmt="grid"))
        print_success("Order submitted successfully!")
        
    except Exception as e:
        print_error(f"Error placing order: {str(e)}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def get_crypto_data(trader: AlpacaCryptoTrader, symbol: str, days: int = 7):
    """Get and display historical crypto data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Normalize the symbol format
        normalized = normalize_crypto_symbol(symbol)
        with_slash = normalized["with_slash"]
        without_slash = normalized["without_slash"]
        
        print(f"\n=== Getting historical data for {with_slash} (last {days} days) ===")
        
        # Try to get data using the symbol with slash
        bars = trader.get_crypto_bars([with_slash], "1D", start=start_date, end=end_date)
        
        # If no data was returned, try using the symbol without slash
        if bars.empty:
            print(f"No data found for {with_slash}, trying alternative format {without_slash}...")
            bars = trader.get_crypto_bars([without_slash], "1D", start=start_date, end=end_date)
        
        if bars.empty:
            print(f"No data available for {symbol} (tried both {with_slash} and {without_slash})")
            return
        
        # Check the structure of the DataFrame's index
        print(f"DataFrame index levels: {bars.index.names}")
        
        # Get the symbols available in the returned data
        if 'symbol' in bars.index.names:
            available_symbols = bars.index.get_level_values('symbol').unique()
            print(f"Available symbols in data: {available_symbols}")
            
            # Try to find the symbol in different formats
            found_symbol = None
            for available in available_symbols:
                if available == with_slash or available == without_slash:
                    found_symbol = available
                    break
            
            if found_symbol:
                print(f"Found data for symbol: {found_symbol}")
                # The index is ordered (symbol, timestamp), use xs to select by level name
                symbol_data = bars.xs(found_symbol, level='symbol')
            else:
                print(f"Symbol not found in data. Using all available data.")
                symbol_data = bars
        else:
            # If 'symbol' is not in the index, use the entire dataframe
            print("Symbol not in index structure. Using all available data.")
            symbol_data = bars
        
        # Calculate some basic stats
        if not symbol_data.empty:
            # Get the most recent price data
            try:
                latest_data = symbol_data.iloc[-1]
                if isinstance(latest_data, pd.Series):
                    current_price = latest_data['close']
                else:
                    # If latest_data is not a Series, we may need to handle differently
                    print(f"Data format unexpected. Type: {type(latest_data)}")
                    current_price = symbol_data['close'].iloc[-1]
                
                high = symbol_data['high'].max()
                low = symbol_data['low'].min()
                vol = symbol_data['volume'].sum()
                
                print(f"Current price: ${current_price:,.2f}")
                print(f"High: ${high:,.2f}")
                print(f"Low: ${low:,.2f}")
                print(f"Volume (last {days} days): {vol:,.0f}")
                print("\n=== Daily Data ===")
                
                # Format the data for display
                data = []
                headers = ["Date", "Open", "High", "Low", "Close", "Volume"]
                
                # Handle different index structures
                for idx, row in symbol_data.iterrows():
                    # If the index is a multiindex, the first level is typically the timestamp
                    if isinstance(idx, tuple):
                        date = idx[0].strftime('%Y-%m-%d')
                    else:
                        # If it's a simple index, use the index directly
                        date = idx.strftime('%Y-%m-%d')
                    
                    data.append([
                        date,
                        f"${row['open']:,.2f}",
                        f"${row['high']:,.2f}",
                        f"${row['low']:,.2f}",
                        f"${row['close']:,.2f}",
                        f"{row['volume']:,.0f}"
                    ])
                
                print(tabulate(data, headers=headers, tablefmt="grid"))
            except Exception as e:
                print(f"Error processing data: {str(e)}")
                # Print DataFrame information to help debug
                print("\nDataFrame structure:")
                print(f"Columns: {symbol_data.columns}")
                print(f"Shape: {symbol_data.shape}")
                print(f"First 3 rows:\n{symbol_data.head(3)}")
                
                import traceback
                traceback.print_exc()
        else:
            print(f"No data available for {symbol}")
            
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        import traceback
        traceback.print_exc()

def list_crypto_assets(trader: AlpacaCryptoTrader):
    """List available crypto assets."""
    try:
        assets = trader.get_crypto_assets()
        
        if not assets:
            print("No crypto assets available.")
            return
            
        print(f"\n=== Available Crypto Assets ({len(assets)}) ===")
        
        headers = ["Symbol", "Name", "Exchange", "Status"]
        data = []
        
        for asset in assets:
            data.append([
                asset.symbol,
                asset.name,
                asset.exchange,
                asset.status.name
            ])
        
        print(tabulate(data, headers=headers, tablefmt="grid"))
        
    except Exception as e:
        print(f"Error listing crypto assets: {str(e)}")

def print_header(title):
    """Print a stylized header."""
    width = 80
    if HAS_COLOR:
        print(f"\n{BLUE}{BOLD}{'=' * width}{RESET}")
        print(f"{BLUE}{BOLD}{title.center(width)}{RESET}")
        print(f"{BLUE}{BOLD}{'=' * width}{RESET}\n")
    else:
        print(f"\n{'=' * width}")
        print(f"{title.center(width)}")
        print(f"{'=' * width}\n")

def print_success(message):
    """Print a success message."""
    if HAS_COLOR:
        print(f"{GREEN}{message}{RESET}")
    else:
        print(f"SUCCESS: {message}")

def print_warning(message):
    """Print a warning message."""
    if HAS_COLOR:
        print(f"{YELLOW}{message}{RESET}")
    else:
        print(f"WARNING: {message}")

def print_error(message):
    """Print an error message."""
    if HAS_COLOR:
        print(f"{RED}{message}{RESET}")
    else:
        print(f"ERROR: {message}")

def confirm_action(message):
    """Ask user to confirm an action."""
    if HAS_COLOR:
        prompt = f"{YELLOW}{message} (y/n): {RESET}"
    else:
        prompt = f"{message} (y/n): "
    
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def show_spinner(message, delay=0.1):
    """Show a simple spinner while processing."""
    if delay <= 0:
        return
        
    symbols = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    
    # Print initial message
    sys.stdout.write(f"{message} {symbols[i]} ")
    sys.stdout.flush()
    
    # Simulate processing
    for _ in range(10):
        time.sleep(delay)
        # Clear the previous symbol
        sys.stdout.write('\b\b')
        i = (i + 1) % len(symbols)
        # Print the next symbol
        sys.stdout.write(f"{symbols[i]} ")
        sys.stdout.flush()
    
    # Clear the spinner
    sys.stdout.write('\b\b\b\b')
    sys.stdout.write(" " * 4)
    sys.stdout.write('\b\b\b\b')
    sys.stdout.flush()

def print_command_help(commands):
    """Print command help in a more friendly format."""
    print_header("Available Commands")
    
    for command, description in commands.items():
        if HAS_COLOR:
            print(f"{BOLD}{BLUE}{command}{RESET}: {description}")
        else:
            print(f"{command}: {description}")
    print()

def interactive_menu():
    """Display interactive menu for easier CLI usage."""
    print_header("Alpaca Crypto Trading CLI")
    
    commands = {
        "account": "View account information",
        "positions": "Show current positions",
        "orders": "View orders",
        "portfolio": "Show portfolio summary and metrics",
        "strategy": "Run trading strategies",
        "order": "Place a crypto order",
        "data": "Get historical price data",
        "assets": "List available crypto assets",
        "clock": "Show market clock",
        "exit": "Exit the program"
    }
    
    trader = None
    portfolio_analyzer = None
    strategies = None
    
    try:
        # Check credentials
        if not check_credentials():
            return
        
        # Initialize clients
        trader = AlpacaCryptoTrader()
        portfolio_analyzer = CryptoPortfolioAnalyzer(trader)
        strategies = CryptoTradingStrategies(trader)
        
        while True:
            print("\n" + "=" * 30)
            for command, description in commands.items():
                if HAS_COLOR:
                    print(f"{BLUE}{command}{RESET}: {description}")
                else:
                    print(f"{command}: {description}")
            print("=" * 30)
            
            choice = input("\nEnter command: ").strip().lower()
            
            if choice == "exit":
                print_success("Exiting the program. Goodbye!")
                break
            
            elif choice == "account":
                display_account_info(trader)
                
            elif choice == "positions":
                display_positions(trader)
                
            elif choice == "orders":
                status_choice = input("Enter order status (open/closed/all) [default: open]: ").strip().lower()
                if not status_choice:
                    status_choice = "open"
                if status_choice not in ["open", "closed", "all"]:
                    print_warning("Invalid status. Using 'open' as default.")
                    status_choice = "open"
                display_orders(trader, status_choice)
                
            elif choice == "portfolio":
                display_portfolio_summary(portfolio_analyzer)
                if confirm_action("Do you want to see portfolio metrics?"):
                    display_portfolio_metrics(portfolio_analyzer)
                    
            elif choice == "strategy":
                strategy_menu(trader, strategies)
                
            elif choice == "order":
                place_order_interactive(trader)
                
            elif choice == "data":
                symbol = input("Enter crypto symbol (e.g., BTC/USD): ").strip()
                try:
                    days = int(input("Enter number of days of data to retrieve [default: 7]: ") or "7")
                except ValueError:
                    days = 7
                    print_warning("Invalid number of days. Using default of 7 days.")
                get_crypto_data(trader, symbol, days)
                
            elif choice == "assets":
                list_crypto_assets(trader)
                
            elif choice == "clock":
                show_spinner("Fetching market clock...", 0.1)
                clock = trader.get_clock()
                print_header("Market Clock")
                clock_data = [
                    ["Current Time", clock.timestamp],
                    ["Market Is Open", f"{GREEN}Yes{RESET}" if clock.is_open and HAS_COLOR else 
                                      (f"{RED}No{RESET}" if not clock.is_open and HAS_COLOR else 
                                       "Yes" if clock.is_open else "No")],
                    ["Next Open", clock.next_open],
                    ["Next Close", clock.next_close]
                ]
                print(tabulate(clock_data, tablefmt="grid"))
                
            else:
                print_error(f"Unknown command: {choice}")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting...")
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
def strategy_menu(trader, strategies):
    """Submenu for trading strategies."""
    print_header("Trading Strategies")
    
    strategies_list = {
        "ma": "Moving Average Crossover Strategy",
        "rsi": "Relative Strength Index (RSI) Strategy",
        "back": "Go back to main menu"
    }
    
    for code, name in strategies_list.items():
        if HAS_COLOR:
            print(f"{BLUE}{code}{RESET}: {name}")
        else:
            print(f"{code}: {name}")
    
    choice = input("\nSelect strategy: ").strip().lower()
    
    if choice == "back":
        return
    
    elif choice == "ma":
        print_header("Moving Average Crossover Strategy")
        symbols_input = input("Enter comma-separated list of crypto symbols (e.g., BTC/USD,ETH/USD): ").strip()
        if not symbols_input:
            print_warning("No symbols provided. Returning to menu.")
            return
            
        symbols = [s.strip() for s in symbols_input.split(',')]
        
        try:
            short_period = int(input("Enter short MA period [default: 20]: ") or "20")
            long_period = int(input("Enter long MA period [default: 50]: ") or "50")
        except ValueError:
            print_warning("Invalid input. Using default values: short=20, long=50")
            short_period = 20
            long_period = 50
            
        print(f"Using symbols: {symbols}")
        execute_ma_strategy(trader, strategies, symbols)
        
    elif choice == "rsi":
        print_header("RSI Strategy")
        symbols_input = input("Enter comma-separated list of crypto symbols (e.g., BTC/USD,ETH/USD): ").strip()
        if not symbols_input:
            print_warning("No symbols provided. Returning to menu.")
            return
            
        symbols = [s.strip() for s in symbols_input.split(',')]
        
        try:
            period = int(input("Enter RSI period [default: 14]: ") or "14")
            oversold = int(input("Enter oversold threshold [default: 30]: ") or "30")
            overbought = int(input("Enter overbought threshold [default: 70]: ") or "70")
        except ValueError:
            print_warning("Invalid input. Using default values: period=14, oversold=30, overbought=70")
            period = 14
            oversold = 30
            overbought = 70
            
        print(f"Using symbols: {symbols}")
        execute_rsi_strategy(trader, strategies, symbols)
        
    else:
        print_error(f"Unknown strategy: {choice}")
        
def place_order_interactive(trader):
    """Interactive menu for placing orders."""
    print_header("Place Crypto Order")
    
    symbol = input("Enter crypto symbol (e.g., BTC/USD): ").strip()
    if not symbol:
        print_warning("No symbol provided. Returning to menu.")
        return
        
    side = input("Enter order side (buy/sell): ").strip().lower()
    if side not in ["buy", "sell"]:
        print_error("Invalid side. Must be 'buy' or 'sell'.")
        return
        
    order_type = input("Enter order type (market/limit) [default: market]: ").strip().lower()
    if not order_type:
        order_type = "market"
    if order_type not in ["market", "limit"]:
        print_warning("Invalid order type. Using 'market' as default.")
        order_type = "market"
        
    size_type = input("Specify order size by quantity or dollar amount (qty/notional) [default: notional]: ").strip().lower()
    if not size_type:
        size_type = "notional"
    if size_type not in ["qty", "notional"]:
        print_warning("Invalid size type. Using 'notional' as default.")
        size_type = "notional"
        
    qty = None
    notional = None
    
    if size_type == "qty":
        try:
            qty = float(input("Enter quantity: ").strip())
        except ValueError:
            print_error("Invalid quantity. Must be a number.")
            return
    else:
        try:
            notional = float(input("Enter dollar amount: $").strip())
        except ValueError:
            print_error("Invalid dollar amount. Must be a number.")
            return
            
    limit_price = None
    if order_type == "limit":
        try:
            limit_price = float(input("Enter limit price: $").strip())
        except ValueError:
            print_error("Invalid limit price. Must be a number.")
            return
            
    place_order(
        trader, 
        symbol=symbol, 
        side=side, 
        qty=qty, 
        notional=notional, 
        order_type=order_type, 
        limit_price=limit_price
    )

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Crypto Trading System for Alpaca',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          python crypto_cli.py interactive   # Start interactive mode
          python crypto_cli.py account       # Show account information
          python crypto_cli.py positions     # Show current positions
          python crypto_cli.py data --symbol BTC/USD --days 7  # Get BTC price history
          python crypto_cli.py order --symbol BTC/USD --side buy --notional 100  # Buy $100 of BTC
          
        For more information and examples, see README.md
        ''')
    )
    
    # Add interactive mode argument
    parser.add_argument('--interactive', '-i', action='store_true', 
                      help='Start in interactive menu mode')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Account command
    account_parser = subparsers.add_parser('account', help='Show account information')
    
    # Positions command
    positions_parser = subparsers.add_parser('positions', help='Show current positions')
    
    # Orders command
    orders_parser = subparsers.add_parser('orders', help='Show orders')
    orders_parser.add_argument('--status', choices=['open', 'closed', 'all'], default='open',
                             help='Filter orders by status')
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Show portfolio information')
    portfolio_parser.add_argument('--metrics', action='store_true', help='Show portfolio metrics')
    
    # Strategy commands
    ma_parser = subparsers.add_parser('ma_strategy', help='Run moving average crossover strategy')
    ma_parser.add_argument('--symbols', required=True, help='Comma-separated list of crypto symbols (e.g., BTC/USD,ETH/USD)')
    ma_parser.add_argument('--short', type=int, default=20, help='Short MA period')
    ma_parser.add_argument('--long', type=int, default=50, help='Long MA period')
    
    rsi_parser = subparsers.add_parser('rsi_strategy', help='Run RSI strategy')
    rsi_parser.add_argument('--symbols', required=True, help='Comma-separated list of crypto symbols (e.g., BTC/USD,ETH/USD)')
    rsi_parser.add_argument('--period', type=int, default=14, help='RSI period')
    rsi_parser.add_argument('--oversold', type=int, default=30, help='Oversold threshold')
    rsi_parser.add_argument('--overbought', type=int, default=70, help='Overbought threshold')
    
    # Order command
    order_parser = subparsers.add_parser('order', help='Place a crypto order')
    order_parser.add_argument('--symbol', required=True, help='Crypto symbol (e.g., BTC/USD)')
    order_parser.add_argument('--side', choices=['buy', 'sell'], required=True, help='Order side')
    order_parser.add_argument('--qty', type=float, help='Quantity of crypto to trade')
    order_parser.add_argument('--notional', type=float, help='Dollar amount to trade')
    order_parser.add_argument('--type', choices=['market', 'limit'], default='market', help='Order type')
    order_parser.add_argument('--limit_price', type=float, help='Limit price (for limit orders)')
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Get historical crypto data')
    data_parser.add_argument('--symbol', required=True, help='Crypto symbol (e.g., BTC/USD or BTCUSD)')
    data_parser.add_argument('--days', type=int, default=7, help='Number of days of data to retrieve')
    
    # Assets command
    assets_parser = subparsers.add_parser('assets', help='List available crypto assets')
    
    # Clock command
    clock_parser = subparsers.add_parser('clock', help='Show market clock')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Start interactive mode if specified or if no command provided
    if args.interactive or (not args.command and not len(sys.argv) > 1):
        interactive_menu()
        return
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Check credentials
    check_credentials()
    
    # Initialize clients
    trader = AlpacaCryptoTrader()
    portfolio_analyzer = CryptoPortfolioAnalyzer(trader)
    strategies = CryptoTradingStrategies(trader)
    
    if args.command == 'account':
        display_account_info(trader)
    
    elif args.command == 'positions':
        display_positions(trader)
    
    elif args.command == 'orders':
        display_orders(trader, args.status)
    
    elif args.command == 'portfolio':
        display_portfolio_summary(portfolio_analyzer)
        
        if args.metrics:
            display_portfolio_metrics(portfolio_analyzer)
    
    elif args.command == 'ma_strategy':
        symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"Using symbols: {symbols}")
        execute_ma_strategy(trader, strategies, symbols)
    
    elif args.command == 'rsi_strategy':
        symbols = [s.strip() for s in args.symbols.split(',')]
        print(f"Using symbols: {symbols}")
        execute_rsi_strategy(trader, strategies, symbols)
    
    elif args.command == 'order':
        # Ensure symbol format is consistent
        symbol = args.symbol.strip()
        print(f"Placing order for symbol: {symbol}")
        place_order(
            trader, 
            symbol=symbol, 
            side=args.side, 
            qty=args.qty, 
            notional=args.notional, 
            order_type=args.type, 
            limit_price=args.limit_price
        )
    
    elif args.command == 'data':
        # Ensure symbol format is consistent
        symbol = args.symbol.strip()
        
        # Handle alternative formats (e.g., if user entered BTCUSD instead of BTC/USD)
        if len(symbol) >= 6 and '/' not in symbol:
            # Try to detect cryptocurrency symbols without slash
            base_currency = symbol[0:3]
            quote_currency = symbol[3:]
            formatted_symbol = f"{base_currency}/{quote_currency}"
            print(f"Converting symbol format from {symbol} to {formatted_symbol}")
            symbol = formatted_symbol
        
        print(f"Fetching data for symbol: {symbol}")
        get_crypto_data(trader, symbol, args.days)
    
    elif args.command == 'assets':
        list_crypto_assets(trader)
    
    elif args.command == 'clock':
        clock = trader.get_clock()
        print_header("Market Clock")
        print(f"Current Time: {clock.timestamp}")
        is_open = "Yes" if clock.is_open else "No"
        if HAS_COLOR:
            is_open = f"{GREEN}Yes{RESET}" if clock.is_open else f"{RED}No{RESET}"
        print(f"Market Is Open: {is_open}")
        print(f"Next Open: {clock.next_open}")
        print(f"Next Close: {clock.next_close}")
    
    elif args.command == 'interactive':
        interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print_error(f"An error occurred: {str(e)}")
        sys.exit(1) 