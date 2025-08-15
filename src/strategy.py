"""Keltner Channels Strategy Implementation"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from indicators import keltner_channels
from pdt import PDTManager

class KeltnerChannelsStrategy:
    """
    Keltner Channels breakout strategy
    
    Enters long when price crosses above upper band
    Enters short when price crosses below lower band
    
    NOTE: PDT is calculated but NOT used for trade gating (matches PineScript exactly)
    """
    
    def __init__(self, config):
        self.config = config
        self.params = config['strategy']['parameters']
        self.risk_config = config['risk_management']
        
        # Strategy state
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = None
        self.entry_time = None
        self.trades = []
        
        # Stop order state (mimics Pine's persistent behavior)
        self.buy_stop_price = None
        self.sell_stop_price = None
        self.buy_stop_active = False
        self.sell_stop_active = False
        
        # PDT Manager - created but not used for gating (matches Pine behavior)
        # In Pine: canTrade = tools.PDT() is computed but never referenced
        if self.risk_config.get('calculate_pdt', False):  # Optional flag
            self.pdt_manager = PDTManager(
                threshold=self.risk_config.get('pdt_threshold', 25000),
                max_day_trades=self.risk_config.get('max_day_trades', 3),
                rolling_days=self.risk_config.get('rolling_days', 5)
            )
        else:
            self.pdt_manager = None
    
    def get_source_value(self, row):
        """Get the source value based on configuration"""
        src = self.params['src']
        if src == 'close':
            return row['close']
        elif src == 'hl2':
            return (row['high'] + row['low']) / 2
        elif src == 'hlc3':
            return (row['high'] + row['low'] + row['close']) / 3
        elif src == 'ohlc4':
            return (row['open'] + row['high'] + row['low'] + row['close']) / 4
        else:
            return row['close']
    
    def prepare_data(self, df):
        """Add indicators to dataframe"""
        # Calculate Keltner Channels
        kc = keltner_channels(
            df,
            length=self.params['length'],
            mult=self.params['multiplier'],
            src=self.params['src'],
            use_ema=self.params['use_ema'],
            bands_style=self.params['bands_style'],
            atr_length=self.params['atr_length']
        )
        
        # Merge with original dataframe
        for col in kc.columns:
            df[col] = kc[col]
        
        return df
    
    def check_market_close(self, current_time):
        """Check if it's time to exit before market close"""
        if not self.risk_config['exit_before_close']:
            return False
            
        # For daily data, this won't trigger
        # Would need 5-minute data to properly implement MC() logic
        return False
    
    def generate_signals(self, df, equity_curve):
        """Generate trading signals"""
        # Prepare data with indicators
        df = self.prepare_data(df)
        
        # Initialize signal columns
        df['signal'] = 0
        df['stop_price'] = np.nan
        
        current_equity = self.config['strategy']['initial_capital']
        
        for i in range(len(df)):
            if pd.isna(df.iloc[i]['kc_upper']) or pd.isna(df.iloc[i]['kc_lower']):
                continue  # Skip rows where indicators aren't ready
                
            row = df.iloc[i]
            current_time = df.index[i]
            
            # Update equity
            if i > 0 and len(equity_curve) > i:
                current_equity = equity_curve[i-1]
            
            # Calculate PDT status (but don't use it for gating)
            # This matches Pine where canTrade = tools.PDT() is computed but unused
            can_trade_pdt = True
            if self.pdt_manager:
                can_trade_pdt = self.pdt_manager.can_trade(current_equity, current_time)
                # In Pine, this value is calculated but NEVER used in entry conditions
            
            # Update stop prices on crosses (persistent like Pine)
            if row['cross_upper']:
                self.buy_stop_price = row['high'] + 0.01  # mintick substitute
                self.buy_stop_active = True
            
            if row['cross_lower']:
                self.sell_stop_price = row['low'] - 0.01  # mintick substitute
                self.sell_stop_active = True
            
            # Check cancellation conditions
            if self.buy_stop_active and self.buy_stop_price is not None:
                cancel_buy = (row['close'] < row['kc_middle'] or 
                             row['high'] >= self.buy_stop_price)
                if cancel_buy:
                    self.buy_stop_active = False
            
            if self.sell_stop_active and self.sell_stop_price is not None:
                cancel_sell = (row['close'] > row['kc_middle'] or 
                              row['low'] <= self.sell_stop_price)
                if cancel_sell:
                    self.sell_stop_active = False
            
            # Generate signals - NO PDT GATING (matches Pine exactly)
            # In Pine: if (crossUpper) with NO canTrade check
            if row['cross_upper']:  # Fresh cross, NO PDT check
                df.loc[current_time, 'signal'] = 1
                df.loc[current_time, 'stop_price'] = self.buy_stop_price
                
            elif row['cross_lower']:  # Fresh cross, NO PDT check
                df.loc[current_time, 'signal'] = -1
                df.loc[current_time, 'stop_price'] = self.sell_stop_price
            
            # Market close exit (would need 5-min data to work properly)
            if self.check_market_close(current_time) and self.position != 0:
                df.loc[current_time, 'signal'] = 0  # Exit signal
        
        return df