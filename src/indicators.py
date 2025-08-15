"""Technical indicators for the strategy"""

import pandas as pd
import numpy as np

def sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def atr(high, low, close, period):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def true_range(high, low, close):
    """True Range for single period"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def keltner_channels(df, length=20, mult=2.0, src='close', use_ema=True, 
                     bands_style='ATR', atr_length=10):
    """
    Calculate Keltner Channels
    
    Parameters:
    -----------
    df : DataFrame with OHLC data
    length : MA period
    mult : Multiplier for bands
    src : Source price (close, hl2, hlc3, etc.)
    use_ema : Use EMA (True) or SMA (False)
    bands_style : 'ATR', 'TR', or 'Range'
    atr_length : Period for ATR calculation
    
    Returns:
    --------
    DataFrame with upper, middle, lower bands
    """
    
    # Select source price
    if src == 'close':
        source = df['close']
    elif src == 'hl2':
        source = (df['high'] + df['low']) / 2
    elif src == 'hlc3':
        source = (df['high'] + df['low'] + df['close']) / 3
    else:
        source = df['close']
    
    # Calculate middle band (MA)
    if use_ema:
        middle = ema(source, length)
    else:
        middle = sma(source, length)
    
    # Calculate range for bands
    if bands_style == 'ATR':
        range_ma = atr(df['high'], df['low'], df['close'], atr_length)
    elif bands_style == 'TR':
        range_ma = true_range(df['high'], df['low'], df['close'])
    else:  # Range
        range_ma = sma(df['high'] - df['low'], length)
    
    # Calculate bands
    upper = middle + (range_ma * mult)
    lower = middle - (range_ma * mult)
    
    return pd.DataFrame({
        'kc_upper': upper,
        'kc_middle': middle,
        'kc_lower': lower
    })
