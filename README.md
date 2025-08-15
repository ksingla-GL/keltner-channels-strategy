# Keltner Channels Trading Strategy

A Python implementation of the Keltner Channels trading strategy, converted from PineScript.

## Features
- Keltner Channels indicator with customizable parameters
- Pattern Day Trader (PDT) protection logic
- Market close position management
- Configurable via JSON config files
- Backtesting with detailed metrics
- Support for multiple data sources (yfinance, CSV)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Configure your strategy parameters in `configs/config.json`
2. Run the backtest:
```bash
python src/main.py
```

## Configuration

Edit `configs/config.json` to customize:
- Symbol and timeframe
- Strategy parameters (length, multiplier, ATR length)
- Risk management settings
- PDT protection settings
- Data source and date range

## Strategy Logic

The strategy enters positions when price crosses the Keltner Channel bands:
- **Long Entry**: Price crosses above upper band
- **Short Entry**: Price crosses below lower band
- **Exit**: Opposite signal or market close (if enabled)

### Key Parameters
- `length`: Moving average period (default: 20)
- `multiplier`: Band width multiplier (default: 2.0)
- `atr_length`: ATR calculation period (default: 10)

## Results

Results are saved in the `results/` directory:
- Trade log CSV
- Performance metrics
- Equity curve visualization

## Project Structure
```
keltner-channels-strategy/
├── configs/          # Configuration files
├── data/            # Historical data cache
├── results/         # Backtest results
├── src/             # Source code
│   ├── strategy.py  # Main strategy logic
│   ├── indicators.py # Technical indicators
│   ├── backtest.py  # Backtesting engine
│   ├── pdt.py       # PDT protection logic
│   └── main.py      # Entry point
└── tests/           # Unit tests
```
