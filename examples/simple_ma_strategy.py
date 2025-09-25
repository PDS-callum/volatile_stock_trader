"""
Simple Moving Average Crossover Strategy Example

This is an example of a custom strategy that can be loaded by the stock bot.
It demonstrates how to create a simple moving average crossover strategy.

To use this strategy:
python -m src.stock_bot run-back-strategy AAPL --strategy examples/simple_ma_strategy.py --strategy-class SimpleMAStrategy
"""

import pandas as pd
import numpy as np
from typing import Optional

# Import the base strategy from the package
from src.stock_bot.strategies.base_strategy import BaseStrategy

class SimpleMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    This strategy generates buy signals when the short MA crosses above the long MA,
    and sell signals when the short MA crosses below the long MA.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(data, **params)
        self._last_signal = None
    
    def _compute_indicators(self):
        """Compute technical indicators."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # Simple moving averages
        self.data['SMA_short'] = self.data['Close'].rolling(window=self.params.get('sma_short', 10)).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=self.params.get('sma_long', 30)).mean()
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on moving average crossover.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < 30:  # Need enough data for moving averages
            return None
        
        try:
            # Get current and previous values
            sma_short_now = self.data['SMA_short'].iloc[idx]
            sma_long_now = self.data['SMA_long'].iloc[idx]
            sma_short_prev = self.data['SMA_short'].iloc[idx - 1]
            sma_long_prev = self.data['SMA_long'].iloc[idx - 1]
            
            # Check for crossover
            if not self.bought:
                # Buy signal: short MA crosses above long MA
                if (sma_short_prev <= sma_long_prev and sma_short_now > sma_long_now):
                    self.bought = True
                    return "BUY"
            else:
                # Sell signal: short MA crosses below long MA
                if (sma_short_prev >= sma_long_prev and sma_short_now < sma_long_now):
                    self.bought = False
                    return "SELL"
                    
        except Exception as e:
            print(f"Error checking signals at idx {idx}: {e}")
        
        return None
