import pandas as pd
import numpy as np
import logging
from typing import Optional
from .base_strategy import BaseStrategy

class DefaultStrategy(BaseStrategy):
    """
    Improved strategy with reduced trading frequency and better signal filtering.
    
    This is the default strategy that was previously in the main strategy.py file.
    It uses MACD, RSI, ROC, and ATR indicators with conservative trading rules.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(data, **params)
        self._last_trade_idx = -1  # Track last trade to implement cooldown
        self._trade_cooldown = params.get('trade_cooldown', 20)  # Minimum bars between trades
        
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals with conservative multi-condition approach.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        # More conservative strategy with reduced trade frequency
        if idx < 30:  # Increased minimum lookback for more reliable signals
            return None
            
        try:
            close_now = self.data['Close'].iloc[idx]
            roc_now = self.data['ROC'].iloc[idx]
            atr_now = self.data['ATR'].iloc[idx]
            rsi_now = self.data['RSI'].iloc[idx]
            macd_now = self.data['MACD'].iloc[idx]
            macd_signal_now = self.data['MACD_signal'].iloc[idx]
            
            # Enhanced thresholds for reduced trading
            roc_buy_thresh = 0.012  # Increased from 0.004 (3x more conservative)
            atr_vol_thresh = self.data['ATR'].rolling(window=30).mean().iloc[idx] if idx >= 30 else 0
            
            # Trade cooldown - prevent frequent entries/exits
            if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
                return None
            
            if not self.bought:
                # Multi-condition buy signal (all must be true)
                rsi_oversold_recovery = 25 < rsi_now < 70  # RSI in reasonable range, not overbought
                macd_bullish = macd_now > macd_signal_now  # MACD above signal line
                strong_momentum = roc_now > roc_buy_thresh  # Strong price momentum
                sufficient_volatility = atr_now > atr_vol_thresh * 1.2  # 20% above average volatility
                
                # Trend confirmation - price above recent average
                recent_avg = self.data['Close'].iloc[max(0, idx-10):idx].mean()
                uptrend = close_now > recent_avg * 1.005  # Price at least 0.5% above recent average
                
                if all([rsi_oversold_recovery, macd_bullish, strong_momentum, sufficient_volatility, uptrend]):
                    self._entry_price = close_now
                    self._peak_price = close_now
                    self._last_trade_idx = idx
                    return "BUY"
                    
            else:
                # Update peak price for trailing stop
                if not hasattr(self, '_peak_price') or close_now > self._peak_price:
                    self._peak_price = close_now
                
                # More conservative exit conditions
                trailing_stop_pct = 0.025  # Increased from 0.01 (2.5% instead of 1%)
                stop_loss_pct = 0.015      # Increased from 0.005 (1.5% instead of 0.5%)
                
                # Additional exit conditions
                rsi_overbought = rsi_now > 75  # Exit on overbought conditions
                macd_bearish = macd_now < macd_signal_now  # MACD crosses below signal
                
                # Profit taking at reasonable levels
                profit_target = hasattr(self, '_entry_price') and close_now > self._entry_price * 1.03  # Take profit at 3%
                
                # Exit conditions (any can trigger)
                trailing_stop_hit = close_now < self._peak_price * (1 - trailing_stop_pct)
                stop_loss_hit = hasattr(self, '_entry_price') and close_now < self._entry_price * (1 - stop_loss_pct)
                
                if trailing_stop_hit or stop_loss_hit or rsi_overbought or (macd_bearish and profit_target):
                    self._last_trade_idx = idx
                    return "SELL"
                    
        except Exception as e:
            logging.error(f"Error checking signals at idx {idx}: {e}")
        return None
