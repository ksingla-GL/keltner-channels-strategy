"""Backtesting engine for the strategy"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class Backtester:
    """Simple backtesting engine"""
    
    def __init__(self, strategy, config):
        self.strategy = strategy
        self.config = config
        self.initial_capital = config['strategy']['initial_capital']
        self.position_size = config['strategy']['position_size']
        self.commission = config['strategy']['commission']
        
        # Results storage
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.df_with_signals = None  # Store the enhanced dataframe
        
    def run(self, df):
        """Run backtest on historical data"""
        print(f"Running backtest on {len(df)} bars...")
        
        # Initialize
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        shares = 0
        
        # Generate signals
        equity_curve = [capital] * len(df)
        df_enhanced = self.strategy.generate_signals(df.copy(), equity_curve)
        self.df_with_signals = df_enhanced  # Store for later use
        
        # Track equity and positions
        equity = []
        position_history = []
        
        for i in range(len(df_enhanced)):
            row = df_enhanced.iloc[i]
            current_time = df_enhanced.index[i]
            signal = row['signal'] if 'signal' in row else 0
            
            # Exit position
            if position != 0 and signal == -position:
                exit_price = row['close']
                pnl = shares * (exit_price - entry_price) * position
                commission_cost = abs(shares * exit_price) * self.commission
                pnl -= commission_cost
                capital += pnl
                
                # Record trade
                self.trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'shares': shares,
                    'pnl': pnl,
                    'return': (pnl / (shares * entry_price)) * 100
                })
                
                # Update PDT manager if needed
                if self.strategy.pdt_manager:
                    self.strategy.pdt_manager.record_trade(
                        entry_time, current_time, current_time
                    )
                
                position = 0
                shares = 0
            
            # Enter new position
            if signal != 0 and position == 0:
                position = signal
                entry_price = row['stop_price'] if not pd.isna(row.get('stop_price')) else row['close']
                entry_time = current_time
                
                # Calculate position size
                position_value = capital * self.position_size
                shares = position_value / entry_price
                commission_cost = position_value * self.commission
                capital -= commission_cost
            
            # Update equity (mark-to-market)
            if position != 0:
                current_value = capital + (shares * row['close'] * position)
                equity.append(current_value)
            else:
                equity.append(capital)
            
            position_history.append(position)
            equity_curve[i] = equity[-1]
        
        # Store results
        self.equity_curve = equity
        self.positions = position_history
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_series = pd.Series(self.equity_curve)
        
        # Calculate metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        metrics = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) * 100,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'total_pnl': trades_df['pnl'].sum(),
            'total_return': (self.equity_curve[-1] / self.initial_capital - 1) * 100,
            'max_drawdown': self.calculate_max_drawdown(equity_series),
            'sharpe_ratio': self.calculate_sharpe_ratio(equity_series)
        }
        
        return metrics
    
    def calculate_max_drawdown(self, equity_series):
        """Calculate maximum drawdown percentage"""
        cummax = equity_series.expanding().max()
        drawdown = (equity_series - cummax) / cummax * 100
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, equity_series, periods=252):
        """Calculate Sharpe ratio"""
        returns = equity_series.pct_change().dropna()
        if len(returns) == 0:
            return 0
        return np.sqrt(periods) * returns.mean() / returns.std() if returns.std() != 0 else 0
    
    def plot_results(self, df=None):
        """Plot backtest results"""
        # Use the enhanced dataframe with indicators
        if df is None or 'kc_upper' not in df.columns:
            df = self.df_with_signals
        
        if df is None or 'kc_upper' not in df.columns:
            print("Warning: Cannot plot Keltner Channels - indicators not found in dataframe")
            return None
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Price and signals
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close', color='black', linewidth=1)
        
        # Plot Keltner Channels if they exist
        if 'kc_upper' in df.columns:
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
        
        # Save plot
        if self.config['output']['plot_equity']:
            fig = self.plot_results()  # Don't pass df, use stored enhanced dataframe
            if fig:
                fig.savefig(f"{output_dir}/backtest_results.png", dpi=100, bbox_inches='tight')
                print(f"Plot saved to {output_dir}/backtest_results.png")
                plt.show()