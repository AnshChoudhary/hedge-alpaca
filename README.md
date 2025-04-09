# AI Hedge - Cryptocurrency Trading Agent

AI Hedge is an intelligent cryptocurrency trading system built using the OpenAI Agents SDK. It analyzes market data, identifies trading opportunities, and can execute trades through the Alpaca trading API.

## Features

- **Advanced AI Trading Analysis**: Leverages Anthropic Claude and other language models via LiteLLM to analyze market data and make trading decisions
- **Technical Indicators**: Utilizes a variety of technical indicators including RSI, MACD, Bollinger Bands, and more
- **Risk Management**: Implements position sizing and risk assessment based on volatility and market conditions
- **Portfolio Management**: Analyzes entire portfolio allocation and provides rebalancing recommendations
- **Trading Modes**: Supports both conservative and aggressive trading strategies
- **Customizable Settings**: Easily configure trading parameters, symbols, and time periods
- **Automated Execution**: Can execute trades automatically via the Alpaca API
- Uses Large Language Models (LLMs) via OpenAI Agents SDK for intelligent trading decisions
- Fetches market data from Yahoo Finance (yfinance) for reliable historical data
- Executes trades via Alpaca API
- Paper trading mode for testing strategies
- Support for Anthropic Claude and other models via LiteLLM
- Automated trading with configurable intervals
- Multi-timeframe analysis for better decision making
- Market sentiment analysis using Sonar-Pro to scan news, social media, and market reports
- Real-time market data from Yahoo Finance
- Comprehensive risk management
- Portfolio optimization and position sizing
- Detailed performance tracking and reporting

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ai-hedge.git
cd ai-hedge
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:

   **Option 1:** Using a `.env` file (recommended)
   
   Copy the example file and fill in your API keys:
   ```bash
   cp .env.example .env
   # Edit the .env file with your API keys
   ```

   **Option 2:** Setting environment variables directly
   ```bash
   export ALPACA_API_KEY_ID='your-alpaca-key'
   export ALPACA_API_SECRET_KEY='your-alpaca-secret'
   export LITELLM_API_KEY='your-litellm-key'
   export OPENAI_BASE_URL='https://litellm.deriv.ai/v1'
   ```

### LiteLLM Setup

The system is configured to use LiteLLM by default, which provides access to various models including Anthropic's Claude models. The configuration uses:

- API Base URL: `https://litellm.deriv.ai/v1` 
- Model: `claude-3-7-sonnet-latest`

To use this setup:
1. Obtain your LiteLLM API key
2. Set it as the `LITELLM_API_KEY` environment variable in your `.env` file or export it directly
3. The system will automatically route requests to Claude through LiteLLM

## Usage

### Basic Usage

To run the trading bot with default settings (conservative, paper trading mode):

```bash
python crypto_trading_loop.py --symbols BTC/USD --interval 5
```

### Additional Options

The trading bot supports several command-line options:

- `--symbols`: Comma-separated list of cryptocurrency symbols to trade (default: BTC/USD)
- `--interval`: Trading interval in minutes (default: 5)
- `--paper`: Use paper trading mode (default: True)
- `--live`: Use live trading (overrides --paper)
- `--aggressive`: Use aggressive position sizing (up to 50% per position)
- `--no-sentiment`: Disable sentiment analysis

### Market Sentiment Analysis

The bot includes a sentiment analysis agent that processes news, social media, and market reports to gauge market sentiment for cryptocurrencies. This feature uses the Sonar-Pro model via LiteLLM to provide real-time analysis of market sentiment.

To test the sentiment analysis separately:

```bash
python test_sentiment.py --symbol BTC/USD --detailed
```

Options for the sentiment test script:

- `--symbol`: Cryptocurrency symbol to analyze (default: BTC/USD)
- `--timeframe`: Timeframe for sentiment analysis (e.g., 24h, 7d) (default: 24h)
- `--model`: Model to use for sentiment analysis (default: sonar-pro)
- `--detailed`: Generate a detailed report across multiple timeframes
- `--verbose`: Enable verbose logging
- `--save`: Save results to JSON file

### Incorporating Sentiment in Trading Decisions

The trading bot automatically incorporates market sentiment when making trading decisions:

1. The sentiment agent analyzes recent news, social media, and market reports
2. It generates a sentiment score from -1.0 (extremely bearish) to +1.0 (extremely bullish)
3. The trading agent considers this sentiment alongside technical indicators
4. When sentiment and technical analysis align, confidence in the trade increases
5. The agent explains how sentiment influenced its decision in the trade reasoning

To disable sentiment analysis (e.g., if you have API quota limitations):

```bash
python crypto_trading_loop.py --symbols BTC/USD --interval 5 --no-sentiment
```

## Trading Modes

The system supports two trading modes:

### Conservative Mode (Default)
- Focuses on capital preservation with moderate growth
- Limits position sizes to 20% of portfolio
- Targets 5% profit per trade
- Avoids highly volatile assets
- More patient with entries and exits

### Aggressive Mode
- Manages the entire account portfolio for maximum returns
- Allocates up to 50% of the portfolio to high-conviction trades
- Targets 15% profit per trade
- Takes calculated risks for higher returns
- Considers volatile assets with high potential returns
- Actively rebalances the portfolio to optimize performance

To enable aggressive mode, use the `--aggressive` flag when starting the bot.

### Automated Trading Bot

The automated trading bot runs continuously, analyzing market data and executing trades at specified intervals:

```bash
python crypto_trading_loop.py --symbols BTC/USD,ETH/USD --interval 5
```

#### Automated Trading Options

- `--symbols` - Comma-separated list of cryptocurrency symbols to trade (default: BTC/USD)
- `--interval` - Trading interval in minutes (default: 5 minutes)
- `--paper` - Use paper trading mode (default: true)
- `--live` - Use live trading with real money (use with caution)

#### How It Works

The automated trading bot:

1. Initializes with your specified cryptocurrencies and interval
2. Every interval (default 5 minutes):
   - Fetches the latest market data from Yahoo Finance for each symbol in multiple timeframes:
     - Short-term (7 days with 1-hour intervals)
     - Medium-term (30 days with 1-day intervals)
     - Long-term (90 days with 1-day intervals)
   - Performs comprehensive technical analysis
   - Makes informed buy/sell/hold decisions using Claude AI
   - Executes trades automatically via Alpaca when favorable conditions are detected
   - Displays portfolio status and notifies about actions taken
3. Records all decisions and trades in a history file for later analysis

The bot implements proper risk management, using position sizing based on account equity, and setting appropriate stop losses and take profits.

### Crypto Agent CLI (Single Analysis)

The main cryptocurrency trading agent can also be used for single analyses:

```bash
python crypto_agent.py --symbol BTC/USD --period 30d --action analyze
```

#### Command-line Options

- `--symbol` - Cryptocurrency symbol (e.g., BTC/USD, ETH/USD)
- `--period` - Analysis period (e.g., 7d, 30d, 90d)
- `--action` - Action to perform: analyze or trade
- `--trade-amount` - Dollar amount to trade (required if action is 'trade')
- `--verbose` - Enable verbose logging

### Examples

1. Analyze Bitcoin for the last 30 days:
```bash
python crypto_agent.py --symbol BTC/USD --period 30d --action analyze
```

2. Analyze Ethereum for the last week:
```bash
python crypto_agent.py --symbol ETH/USD --period 7d --action analyze
```

3. Execute a trade with $100 if a favorable signal is found:
```bash
python crypto_agent.py --symbol BTC/USD --action trade --trade-amount 100
```

4. Run the automated trading bot with multiple symbols:
```bash
python crypto_trading_loop.py --symbols BTC/USD,ETH/USD,SOL/USD --interval 10
```

### Verifying LiteLLM Connection

You can verify your LiteLLM connection before running the main application:

```bash
python test_litellm.py
```

This will check:
1. If your API key is valid
2. If you can connect to the LiteLLM API
3. If Claude models are available
4. If you can successfully complete a test request

### Testing Yahoo Finance Data Collection

You can verify that Yahoo Finance data collection is working properly with the test script:

```bash
python test_yfinance.py --symbol BTC-USD --plot
```

This will:
1. Fetch historical data for the specified cryptocurrency
2. Display statistics about the data
3. Optionally generate a price chart
4. Save the chart as an image file

To test multiple timeframes at once:

```bash
python test_yfinance.py --symbol ETH-USD --test-all
```

### Troubleshooting

If you encounter API connection issues:

1. Verify your LiteLLM API key is valid and properly set as `LITELLM_API_KEY` (or `OPENAI_API_KEY`)
2. Ensure the base URL is correctly set to `https://litellm.deriv.ai/v1`
3. Check the logs for any specific error messages
4. Try running with verbose logging: `python crypto_agent.py --verbose ...`
5. Use the test script: `python test_litellm.py --verbose`

### Traditional Crypto CLI

You can also use the traditional command-line interface for trading:

```bash
python crypto-cli.py interactive
```

## Components

### Automated Trading Bot (crypto_trading_loop.py)

Continuous trading system that runs at specified intervals for automated trading.

### Performance Monitor (performance_monitor.py)

Analyzes trading performance and generates visual reports of trading activity and profitability.

```bash
# View performance metrics:
python performance_monitor.py

# Use a custom trade history file:
python performance_monitor.py --file my_trade_history.json

# Generate metrics without plots (headless environment):
python performance_monitor.py --no-plot
```

### Crypto Agent (crypto_agent.py)

The main AI agent that uses LLMs for cryptocurrency analysis and trading.

### Technical Indicators (crypto_indicators.py)

Library of technical analysis indicators and trading signal generators.

### Alpaca Crypto Integration (alpaca_crypto.py)

Integration with the Alpaca trading API for cryptocurrency trading.

### Command-line Interface (crypto-cli.py)

Traditional command-line interface for crypto trading operations.

## Risk Warning

This is an experimental trading system. Never invest money you cannot afford to lose. Always start with paper trading and small positions. The AI may make incorrect predictions and trading involves substantial risk.

## License

MIT License

## Credits

- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/agents/
- Alpaca: https://alpaca.markets/
- Yahoo Finance: https://finance.yahoo.com/
- LiteLLM: https://docs.litellm.ai/
- Anthropic Claude: https://www.anthropic.com/claude 