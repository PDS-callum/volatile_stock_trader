"""
Simple Mean Reversion Strategy

A simplified mean reversion strategy that focuses on the core concepts:
- Bollinger Bands for extreme price detection
- RSI for momentum confirmation
- Simple moving average reversion
"""

import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy

class SimpleMeanReversionStrategy(BaseStrategy):
    """
    Simple Mean Reversion Strategy.
    
    This strategy identifies oversold/overbought conditions and bets on price
    reversion to the mean using Bollinger Bands and RSI.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Strategy parameters with defaults - more responsive
        self.bb_period = 15        # Reduced from 20 to 15
        self.bb_std = 1.5          # Reduced from 2.0 to 1.5
        self.rsi_period = 14
        self.rsi_oversold = 35     # Increased from 30 to 35
        self.rsi_overbought = 65   # Decreased from 70 to 65
        self.sma_period = 15       # Reduced from 20 to 15
        self.profit_target_pct = 0.01  # Reduced from 0.015 to 0.01
        self.stop_loss_pct = 0.015  # Reduced from 0.02 to 0.015
        self.max_holding_period = 10  # Reduced from 15 to 10
        
        # Position tracking
        self._entry_price = None
        self._entry_idx = None
        self._last_trade_idx = -1
        self._trade_cooldown = 1  # Reduced from 2 to 1
        
        super().__init__(data, **params)
        
    def _compute_indicators(self):
        """Compute technical indicators for mean reversion."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=self.bb_period).mean()
        bb_std_val = self.data['Close'].rolling(window=self.bb_period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std_val * self.bb_std)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std_val * self.bb_std)
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        
        # Simple Moving Average
        self.data['SMA'] = self.data['Close'].rolling(window=self.sma_period).mean()
        self.data['Price_vs_SMA'] = (self.data['Close'] - self.data['SMA']) / self.data['SMA']
        
        # Price momentum (for mean reversion confirmation)
        self.data['Price_Momentum'] = self.data['Close'].pct_change(periods=3)
        
    def _is_oversold(self, idx: int) -> bool:
        """Check if price is in oversold territory."""
        if idx < max(self.bb_period, self.rsi_period, self.sma_period):
            return False
        
        current_price = self.data['Close'].iloc[idx]
        bb_lower = self.data['BB_Lower'].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        price_vs_sma = self.data['Price_vs_SMA'].iloc[idx]
        
        # Multiple oversold conditions (any can trigger)
        conditions = [
            # Price below lower Bollinger Band
            current_price < bb_lower,
            # RSI oversold
            rsi < self.rsi_oversold,
            # Price significantly below SMA (2% or more)
            price_vs_sma < -0.02
        ]
        
        return any(conditions)
    
    def _is_overbought(self, idx: int) -> bool:
        """Check if price is in overbought territory."""
        if idx < max(self.bb_period, self.rsi_period, self.sma_period):
            return False
        
        current_price = self.data['Close'].iloc[idx]
        bb_upper = self.data['BB_Upper'].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        price_vs_sma = self.data['Price_vs_SMA'].iloc[idx]
        
        # Multiple overbought conditions (any can trigger)
        conditions = [
            # Price above upper Bollinger Band
            current_price > bb_upper,
            # RSI overbought
            rsi > self.rsi_overbought,
            # Price significantly above SMA (2% or more)
            price_vs_sma > 0.02
        ]
        
        return any(conditions)
    
    def _is_mean_reverting(self, idx: int) -> bool:
        """Check if price is showing signs of mean reversion."""
        if idx < 3:
            return False
        
        # Check if price is moving back toward the mean
        current_price = self.data['Close'].iloc[idx]
        sma = self.data['SMA'].iloc[idx]
        bb_middle = self.data['BB_Middle'].iloc[idx]
        
        # Price moving toward SMA
        price_moving_to_sma = abs(current_price - sma) < abs(self.data['Close'].iloc[idx-1] - self.data['SMA'].iloc[idx-1])
        
        # Price moving toward BB middle
        price_moving_to_bb = abs(current_price - bb_middle) < abs(self.data['Close'].iloc[idx-1] - self.data['BB_Middle'].iloc[idx-1])
        
        # RSI moving away from extremes
        rsi = self.data['RSI'].iloc[idx]
        rsi_prev = self.data['RSI'].iloc[idx-1]
        rsi_reverting = (rsi_prev < 30 and rsi > rsi_prev) or (rsi_prev > 70 and rsi < rsi_prev)
        
        return price_moving_to_sma or price_moving_to_bb or rsi_reverting
    
    def _should_exit_long(self, idx: int) -> bool:
        """Check if we should exit a long position."""
        if not self.bought or self._entry_price is None or self._entry_idx is None:
            return False
        
        current_price = self.data['Close'].iloc[idx]
        holding_period = idx - self._entry_idx
        
        # Exit conditions
        # 1. Profit target hit
        if current_price >= self._entry_price * (1 + self.profit_target_pct):
            return True
        
        # 2. Stop loss hit
        if current_price <= self._entry_price * (1 - self.stop_loss_pct):
            return True
        
        # 3. Maximum holding period reached
        if holding_period >= self.max_holding_period:
            return True
        
        # 4. Price returns to SMA (within 1%)
        sma = self.data['SMA'].iloc[idx]
        if current_price >= sma * 0.99:
            return True
        
        # 5. Price returns to BB middle (within 1%)
        bb_middle = self.data['BB_Middle'].iloc[idx]
        if current_price >= bb_middle * 0.99:
            return True
        
        # 6. RSI becomes overbought
        if self.data['RSI'].iloc[idx] > self.rsi_overbought:
            return True
        
        # 7. Price becomes overbought
        if self._is_overbought(idx):
            return True
        
        return False
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on mean reversion logic.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < max(self.bb_period, self.rsi_period, self.sma_period):
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            current_price = self.data['Close'].iloc[idx]
            
            if not self.bought:
                # Look for oversold conditions (buy signal)
                if (self._is_oversold(idx) and 
                    self._is_mean_reverting(idx)):
                    
                    self.bought = True
                    self._entry_price = current_price
                    self._entry_idx = idx
                    self._last_trade_idx = idx
                    return "BUY"
            
            else:
                # Check for exit conditions
                if self._should_exit_long(idx):
                    self.bought = False
                    self._entry_price = None
                    self._entry_idx = None
                    self._last_trade_idx = idx
                    return "SELL"
                    
        except Exception as e:
            print(f"Error checking signals at idx {idx}: {e}")
        
        return None
