"""
Simple Trend Following Strategy

A simplified version of the trend-following strategy that's more likely to generate signals.
Uses basic moving average crossovers with less strict conditions.
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy

class SimpleTrendStrategy(BaseStrategy):
    """
    Simple Trend Following Strategy.
    
    This strategy uses basic moving average crossovers to identify trends.
    It's designed to be more responsive and generate more signals.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Strategy parameters with defaults
        self.short_ma_period = params.get('short_ma_period', 10)
        self.long_ma_period = params.get('long_ma_period', 30)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.min_trend_strength = params.get('min_trend_strength', 0.01)  # 1%
        
        # Position tracking
        self._entry_price = None
        self._peak_price = None
        self._last_trade_idx = -1
        self._trade_cooldown = params.get('trade_cooldown', 3)
        
        super().__init__(data, **params)
        
    def _compute_indicators(self):
        """Compute technical indicators for trend following."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # Simple Moving Averages
        self.data['SMA_short'] = self.data['Close'].rolling(window=self.short_ma_period).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=self.long_ma_period).mean()
        
        # Trend strength (price change over period)
        self.data['Trend_Strength'] = (
            (self.data['Close'] - self.data['Close'].shift(5)) 
            / self.data['Close'].shift(5)
        )
        
    def _is_golden_cross(self, idx: int) -> bool:
        """Check for golden cross."""
        if idx < self.long_ma_period:
            return False
        
        current_short = self.data['SMA_short'].iloc[idx]
        current_long = self.data['SMA_long'].iloc[idx]
        prev_short = self.data['SMA_short'].iloc[idx - 1]
        prev_long = self.data['SMA_long'].iloc[idx - 1]
        
        return (prev_short <= prev_long and current_short > current_long)
    
    def _is_death_cross(self, idx: int) -> bool:
        """Check for death cross."""
        if idx < self.long_ma_period:
            return False
        
        current_short = self.data['SMA_short'].iloc[idx]
        current_long = self.data['SMA_long'].iloc[idx]
        prev_short = self.data['SMA_short'].iloc[idx - 1]
        prev_long = self.data['SMA_long'].iloc[idx - 1]
        
        return (prev_short >= prev_long and current_short < current_long)
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on simple trend following.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < self.long_ma_period:
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            current_price = self.data['Close'].iloc[idx]
            rsi = self.data['RSI'].iloc[idx]
            trend_strength = self.data['Trend_Strength'].iloc[idx]
            
            if not self.bought:
                # Buy signal: Golden cross + RSI not overbought + positive trend
                if (self._is_golden_cross(idx) and 
                    rsi < self.rsi_overbought and 
                    trend_strength > self.min_trend_strength):
                    
                    self.bought = True
                    self._entry_price = current_price
                    self._peak_price = current_price
                    self._last_trade_idx = idx
                    return "BUY"
            
            else:
                # Update peak price
                if self._peak_price is None or current_price > self._peak_price:
                    self._peak_price = current_price
                
                # Sell signal: Death cross OR RSI overbought OR trailing stop
                if (self._is_death_cross(idx) or 
                    rsi > self.rsi_overbought or
                    current_price < self._peak_price * 0.95):  # 5% trailing stop
                    
                    self.bought = False
                    self._entry_price = None
                    self._peak_price = None
                    self._last_trade_idx = idx
                    return "SELL"
                    
        except Exception as e:
            print(f"Error checking signals at idx {idx}: {e}")
        
        return None
