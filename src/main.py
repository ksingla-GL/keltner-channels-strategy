"""Main entry point for the Keltner Channels Strategy"""

import json
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from strategy import KeltnerChannelsStrategy
from backtest import Backtester

def load_config(config_path='configs/config.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def fetch_data(config):
    """Fetch historical data based on config"""
    data_config = config['data']
    
    print(f"Fetching data for {data_config['symbol']}...")
    print(f"Period: {data_config['start_date']} to {data_config['end_date']}")
    print(f"Timeframe: {data_config['timeframe']}")
    
    if data_config['data_source'] == 'yfinance':
        # Download data from Yahoo Finance
        df = yf.download(
            data_config['symbol'],
            start=data_config['start_date'],
            end=data_config['end_date'],
            interval=data_config['timeframe'],
            progress=False
        )
        
        # Handle MultiIndex columns (occurs with single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in df.columns]
        df = df[available_columns]
        
        # Drop any rows with NaN values
        df = df.dropna()
        
    elif data_config['data_source'] == 'csv':
        # Load from CSV file
        csv_path = f"data/{data_config['symbol']}.csv"
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.columns = df.columns.str.lower()
    
    else:
        raise ValueError(f"Unknown data source: {data_config['data_source']}")
    
    print(f"Loaded {len(df)} bars of data")
    return df

def print_metrics(metrics):
    """Pretty print performance metrics"""
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Trades:    {metrics['total_trades']}")
    print(f"Winning Trades:  {metrics.get('winning_trades', 0)}")
    print(f"Losing Trades:   {metrics.get('losing_trades', 0)}")
    print(f"Win Rate:        {metrics['win_rate']:.2f}%")
    print(f"Avg Win:         ${metrics.get('avg_win', 0):.2f}")
    print(f"Avg Loss:        ${metrics.get('avg_loss', 0):.2f}")
    print(f"Total PnL:       ${metrics.get('total_pnl', 0):.2f}")
    print(f"Total Return:    {metrics['total_return']:.2f}%")
    print(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print("="*50)

def main():
    """Main execution function"""
    print("\nKeltner Channels Strategy Backtest")
    print("="*40)
    
    # Load configuration
    config = load_config()
    
    # Fetch data
    df = fetch_data(config)
    
    if df.empty:
        print("Error: No data fetched. Check your configuration.")
        return 1
    
    # Initialize strategy
    strategy = KeltnerChannelsStrategy(config)
    
    # Initialize backtester
    backtester = Backtester(strategy, config)
    
    # Run backtest
    metrics = backtester.run(df)
    
    # Print results
    if config['output']['verbose']:
        print_metrics(metrics)
    
    # Save results
    backtester.save_results(df)
    
    print("\nBacktest complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())