"""Basic tests for the strategy components"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from indicators import sma, ema, atr, keltner_channels

def test_indicators():
    """Test that indicators calculate without errors"""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100)
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Test SMA
    sma_result = sma(df['close'], 20)
    assert len(sma_result) == len(df)
    assert not sma_result[20:].isna().any()
    
    # Test EMA  
    ema_result = ema(df['close'], 20)
    assert len(ema_result) == len(df)
    
    # Test ATR
    atr_result = atr(df['high'], df['low'], df['close'], 14)
    assert len(atr_result) == len(df)
    
    # Test Keltner Channels
    kc_result = keltner_channels(df)
    assert 'kc_upper' in kc_result.columns
    assert 'kc_middle' in kc_result.columns
    assert 'kc_lower' in kc_result.columns
    
    print("All indicator tests passed!")

if __name__ == "__main__":
    test_indicators()
