"""Backtesting engine for the strategy"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class Backtester:
    """Bar-by-bar backtesting engine with pending stop orders"""
    
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.config = config
        self.initial_capital = config['strategy']['initial_capital']
        self.position_size = config['strategy']['position_size']
        self.commission = config['strategy']['commission']
        self.mintick = config.get('mintick', 0.01)
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.df_with_signals = None
        
    def run(self, df):
        """Run backtest bar-by-bar with pending orders"""
        print(f"Running backtest on {len(df)} bars...")
        
        # Prepare data with indicators
        df = self.strategy.prepare_data(df.copy())
        self.df_with_signals = df
        
        # Initialize state
        cash = self.initial_capital
        position = 0
        shares = 0
        entry_price = 0
        entry_time = None
        
        # Pending orders
        pending_buy = None
        pending_sell = None
        
        # Track results
        equity_history = []
        position_history = []
        
        for i in range(len(df)):
            if pd.isna(df.iloc[i]['kc_upper']):
                equity_history.append(cash)
                position_history.append(0)
                continue
                
            row = df.iloc[i]
            current_time = df.index[i]
            
            # Check for fills on pending orders FIRST
            if pending_buy and row['high'] >= pending_buy['price']:
                # Buy stop triggered
                fill_price = pending_buy['price']
                
                # Close short if exists
                if position < 0:
                    trade_pnl = shares * (entry_price - fill_price)
                    trade_pnl -= shares * entry_price * self.commission
                    trade_pnl -= shares * fill_price * self.commission
                    cash += trade_pnl
                    
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': fill_price,
                        'position': -1,
                        'shares': shares,
                        'pnl': trade_pnl,
                        'return': (trade_pnl / (shares * entry_price)) * 100
                    })
                    position = 0
                    shares = 0
                
                # Enter long
                if position == 0:
                    position_value = cash * self.position_size
                    shares = position_value / fill_price
                    entry_price = fill_price
                    entry_time = current_time
                    position = 1
                pending_buy = None
                
            elif pending_sell and row['low'] <= pending_sell['price']:
                # Sell stop triggered
                fill_price = pending_sell['price']
                
                # Close long if exists
                if position > 0:
                    trade_pnl = shares * (fill_price - entry_price)
                    trade_pnl -= shares * entry_price * self.commission
                    trade_pnl -= shares * fill_price * self.commission
                    cash += trade_pnl
                    
                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': fill_price,
                        'position': 1,
                        'shares': shares,
                        'pnl': trade_pnl,
                        'return': (trade_pnl / (shares * entry_price)) * 100
                    })
                    position = 0
                    shares = 0
                
                # Enter short
                if position == 0:
                    position_value = cash * self.position_size
                    shares = position_value / fill_price
                    entry_price = fill_price
                    entry_time = current_time
                    position = -1
                pending_sell = None
            
            # Get source value for crosses
            src = self.strategy.get_source_value(row)
            
            # Check for new crosses
            if i > 0:
                prev_row = df.iloc[i-1]
                prev_src = self.strategy.get_source_value(prev_row)
                
                # Upper cross - place buy stop
                if src > row['kc_upper'] and prev_src <= prev_row['kc_upper']:
                    pending_buy = {'price': row['high'] + self.mintick}
                
                # Lower cross - place sell stop
                if src < row['kc_lower'] and prev_src >= prev_row['kc_lower']:
                    pending_sell = {'price': row['low'] - self.mintick}
            
            # Cancel orders if conditions met
            # Pine cancels if src crosses back below/above MA
            # (The high >= bprice check in Pine is for "already filled", but we handle that separately)
            if pending_buy:
                if src < row['kc_middle']:
                    pending_buy = None
            
            if pending_sell:
                if src > row['kc_middle']:
                    pending_sell = None
            
            # Calculate current equity
            current_equity = cash
            if position != 0:
                if position > 0:
                    unrealized_pnl = shares * (row['close'] - entry_price)
                else:  # Short
                    unrealized_pnl = shares * (entry_price - row['close'])
                current_equity = cash + unrealized_pnl
            
            equity_history.append(current_equity)
            position_history.append(position)
        
        # Force close any open position at the end
        if position != 0:
            final_row = df.iloc[-1]
            exit_price = final_row['close']
            
            if position > 0:
                trade_pnl = shares * (exit_price - entry_price)
            else:
                trade_pnl = shares * (entry_price - exit_price)
            
            trade_pnl -= shares * entry_price * self.commission
            trade_pnl -= shares * exit_price * self.commission
            cash += trade_pnl
            
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'shares': shares,
                'pnl': trade_pnl,
                'return': (trade_pnl / (shares * entry_price)) * 100,
                'forced_exit': True  # Mark as forced
            })
            
            # Update final equity
            equity_history[-1] = cash
        
        # Store results
        self.equity_curve = equity_history
        self.positions = position_history
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {}
        
        # Debug: Check for open position
        final_equity = self.equity_curve[-1]
        total_return = ((final_equity / self.initial_capital) - 1) * 100
        
        # Calculate realized PnL from closed trades
        realized_pnl = sum(t['pnl'] for t in self.trades) if self.trades else 0
        expected_equity = self.initial_capital + realized_pnl
        
        print(f"\nDEBUG: Initial capital: ${self.initial_capital:.2f}")
        print(f"DEBUG: Realized PnL: ${realized_pnl:.2f}")
        print(f"DEBUG: Expected equity (if flat): ${expected_equity:.2f}")
        print(f"DEBUG: Actual final equity: ${final_equity:.2f}")
        print(f"DEBUG: Difference (unrealized PnL): ${final_equity - expected_equity:.2f}")
        
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_pnl': 0,
                'total_return': total_return,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        metrics = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'total_pnl': trades_df['pnl'].sum(),
            'total_return': total_return,
            'max_drawdown': self.calculate_max_drawdown(),
            'sharpe_ratio': self.calculate_sharpe_ratio()
        }
        
        return metrics
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown percentage"""
        equity_series = pd.Series(self.equity_curve)
        cummax = equity_series.expanding().max()
        drawdown = (equity_series - cummax) / cummax * 100
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, periods=252):
        """Calculate Sharpe ratio"""
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0
        return np.sqrt(periods) * returns.mean() / returns.std()
    
    def plot_results(self, df=None):
        """Plot backtest results"""
        if df is None:
            df = self.df_with_signals
        
        if df is None or 'kc_upper' not in df.columns:
            print("Warning: Cannot plot - no data available")
            return None
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Price and signals
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1)
        ax1.plot(df.index, df['kc_upper'], label='Upper Band', color='red', alpha=0.5)
        ax1.plot(df.index, df['kc_middle'], label='Middle', color='blue', alpha=0.5)
        ax1.plot(df.index, df['kc_lower'], label='Lower Band', color='green', alpha=0.5)
        
        # Mark trades
        for trade in self.trades:
            if trade['position'] > 0:
                ax1.scatter(trade['entry_time'], trade['entry_price'], 
                           color='green', marker='^', s=100, zorder=5)
                ax1.scatter(trade['exit_time'], trade['exit_price'], 
                           color='red', marker='v', s=100, zorder=5)
            else:
                ax1.scatter(trade['entry_time'], trade['entry_price'], 
                           color='red', marker='v', s=100, zorder=5)
                ax1.scatter(trade['exit_time'], trade['exit_price'], 
                           color='green', marker='^', s=100, zorder=5)
        
        ax1.set_title('Keltner Channels Strategy - Price Action')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Equity curve
        ax2 = axes[1]
        ax2.plot(df.index[:len(self.equity_curve)], self.equity_curve, 
                label='Equity', color='blue', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Equity ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Drawdown
        ax3 = axes[2]
        equity_series = pd.Series(self.equity_curve, index=df.index[:len(self.equity_curve)])
        cummax = equity_series.expanding().max()
        drawdown = (equity_series - cummax) / cummax * 100
        ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, df=None):
        """Save results to files"""
        output_dir = "results"
        
        # Save trades
        if self.trades and self.config['output']['save_trades']:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(f"{output_dir}/trades.csv", index=False)
            print(f"Trades saved to {output_dir}/trades.csv")
        
        # Save metrics
        if self.config['output']['save_metrics']:
            metrics = self.calculate_metrics()
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)
            print(f"Metrics saved to {output_dir}/metrics.csv")
            
            # Debug print
            print(f"\nDEBUG: Final equity = ${self.equity_curve[-1]:.2f}")
            print(f"DEBUG: Initial = ${self.initial_capital:.2f}")
            print(f"DEBUG: Actual return = {((self.equity_curve[-1] / self.initial_capital) - 1) * 100:.2f}%")
        
        # Save plot
        if self.config['output']['plot_equity']:
            fig = self.plot_results()
            if fig:
                fig.savefig(f"{output_dir}/backtest_results.png", dpi=100, bbox_inches='tight')
                print(f"Plot saved to {output_dir}/backtest_results.png")
                plt.show()