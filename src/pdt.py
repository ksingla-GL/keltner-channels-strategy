"""Pattern Day Trader (PDT) protection logic"""

from datetime import datetime, timedelta
import pandas as pd

class PDTManager:
    """Manages PDT rules and restrictions"""
    
    def __init__(self, threshold=25000, max_day_trades=3, rolling_days=5):
        self.threshold = threshold
        self.max_day_trades = max_day_trades
        self.rolling_days = rolling_days
        self.day_trades = []
        self.is_blocked = False
        self.block_until = None
        self.traded_today = False
        self.session_locked = False
        
    def is_pdt_account(self, equity):
        """Check if account qualifies as PDT (>= $25k)"""
        return equity >= self.threshold
    
    def can_trade(self, equity, current_time):
        """Check if trading is allowed"""
        if self.is_pdt_account(equity):
            return True
            
        # Check if currently blocked
        if self.is_blocked and current_time < self.block_until:
            return False
        elif self.is_blocked and current_time >= self.block_until:
            self.is_blocked = False
            self.block_until = None
        
        # Check session lock or already traded today
        if self.session_locked or self.traded_today:
            return False
            
        return True
    
    def record_trade(self, entry_time, exit_time, current_time):
        """Record a potential day trade"""
        if entry_time.date() == exit_time.date():
            # It's a day trade
            self.day_trades.append(exit_time)
            self.traded_today = True
            self.session_locked = True
            
            # Clean old trades
            cutoff = current_time - timedelta(days=self.rolling_days)
            self.day_trades = [dt for dt in self.day_trades if dt > cutoff]
            
            # Check if we hit the limit
            if len(self.day_trades) >= self.max_day_trades:
                # Block for 1 trading day
                self.is_blocked = True
                self.block_until = current_time + timedelta(days=1)
                # In real implementation, should check for next business day
    
    def new_day(self):
        """Reset daily flags"""
        self.traded_today = False
        self.session_locked = False
    
    def get_remaining_trades(self, current_time):
        """Get number of remaining day trades allowed"""
        cutoff = current_time - timedelta(days=self.rolling_days)
        recent_trades = [dt for dt in self.day_trades if dt > cutoff]
        return max(0, self.max_day_trades - len(recent_trades))
