import pandas as pd
import numpy as np
import logging
from typing import Any, List, Optional

class BaseStrategy:
    """
    Base class for trading strategies. Subclass and implement check_signals.
    """
    def __init__(self, data: pd.DataFrame, **params):
        self.data = data.copy()
        self.params = params
        self.bought = False
        self.signals: List[Any] = []
        self._compute_indicators()

    def _compute_indicators(self):
        # Default indicators, can be overridden
        p = self.params
        try:
            # 1. Reduce indicator lag: use shorter periods
            self.data['EMA_fast'] = self.data['Close'].ewm(span=p.get('macd_fast', 6), adjust=False).mean()
            self.data['EMA_slow'] = self.data['Close'].ewm(span=p.get('macd_slow', 13), adjust=False).mean()
            self.data['MACD'] = self.data['EMA_fast'] - self.data['EMA_slow']
            self.data['MACD_signal'] = self.data['MACD'].ewm(span=p.get('macd_signal', 5), adjust=False).mean()
            self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
            
            # RSI
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=p.get('rsi_period', 7)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p.get('rsi_period', 7)).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # Rate of Change
            self.data['ROC'] = self.data['Close'].pct_change(periods=p.get('roc_period', 2))
            
            # EMA Momentum
            self.data['EMA_mom'] = self.data['Close'].ewm(span=p.get('ema_mom_period', 10), adjust=False).mean()
            
            # Bollinger Bands
            bb_period = p.get('bb_period', 20)
            bb_std = p.get('bb_std', 2)
            self.data['BB_middle'] = self.data['Close'].rolling(window=bb_period).mean()
            bb_std_val = self.data['Close'].rolling(window=bb_period).std()
            self.data['BB_upper'] = self.data['BB_middle'] + (bb_std_val * bb_std)
            self.data['BB_lower'] = self.data['BB_middle'] - (bb_std_val * bb_std)
            
            # 3. Add volatility filter (ATR)
            high = self.data['Close'].rolling(window=2).max()
            low = self.data['Close'].rolling(window=2).min()
            prev_close = self.data['Close'].shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            self.data['ATR'] = tr.rolling(window=7).mean()
        except Exception as e:
            logging.error(f"Error computing indicators: {e}")

    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals at a given index. Override in subclass.
        """
        return None

class DefaultStrategy(BaseStrategy):
    """
    Improved strategy with reduced trading frequency and better signal filtering.
    """
    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(data, **params)
        self._last_trade_idx = -1  # Track last trade to implement cooldown
        self._trade_cooldown = params.get('trade_cooldown', 20)  # Minimum bars between trades
        
    def check_signals(self, idx: int) -> Optional[str]:
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
