"""
Aggressive Strategy

A very simple strategy that generates many signals for testing purposes.
Uses basic price momentum with minimal conditions.
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy

class AggressiveStrategy(BaseStrategy):
    """
    Aggressive Strategy for testing.
    
    This strategy generates many buy/sell signals based on simple price momentum.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Strategy parameters
        self.lookback = params.get('lookback', 5)
        self.momentum_threshold = params.get('momentum_threshold', 0.01)  # 1%
        self._last_trade_idx = -1
        self._trade_cooldown = params.get('trade_cooldown', 2)
        
        super().__init__(data, **params)
        
    def _compute_indicators(self):
        """Compute technical indicators."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # Simple momentum
        self.data['Momentum'] = self.data['Close'].pct_change(periods=self.lookback)
        
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on simple momentum.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < self.lookback + 1:
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            momentum = self.data['Momentum'].iloc[idx]
            
            if not self.bought:
                # Buy signal: positive momentum
                if momentum > self.momentum_threshold:
                    self.bought = True
                    self._last_trade_idx = idx
                    return "BUY"
            
            else:
                # Sell signal: negative momentum
                if momentum < -self.momentum_threshold:
                    self.bought = False
                    self._last_trade_idx = idx
                    return "SELL"
                    
        except Exception as e:
            print(f"Error checking signals at idx {idx}: {e}")
        
        return None
