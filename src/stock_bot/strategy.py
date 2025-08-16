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
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=p.get('rsi_period', 7)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p.get('rsi_period', 7)).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            self.data['ROC'] = self.data['Close'].pct_change(periods=p.get('roc_period', 2))
            self.data['EMA_mom'] = self.data['Close'].ewm(span=p.get('ema_mom_period', 10), adjust=False).mean()
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
    Default strategy logic, same as original check_signals.
    """
    def check_signals(self, idx: int) -> Optional[str]:
        # Volatile stock strategy: frequent trades, quick exits on small loss, hold for peaks
        if idx < 10:
            return None
        try:
            close_now = self.data['Close'].iloc[idx]
            roc_now = self.data['ROC'].iloc[idx]
            atr_now = self.data['ATR'].iloc[idx]
            # Buy on strong upward move and high volatility
            roc_buy_thresh = 0.004  # Aggressive ROC threshold
            atr_vol_thresh = self.data['ATR'].rolling(window=30).mean().iloc[idx] if idx >= 30 else 0
            if not self.bought:
                if roc_now > roc_buy_thresh and atr_now > atr_vol_thresh:
                    self._entry_price = close_now
                    self._peak_price = close_now
                    return "BUY"
            else:
                # Update peak price for trailing stop
                if not hasattr(self, '_peak_price') or close_now > self._peak_price:
                    self._peak_price = close_now
                # Sell if price drops by more than 1% from peak (trailing stop)
                if close_now < self._peak_price * 0.99:
                    return "SELL"
                # Sell if loss exceeds 0.5% from entry
                if hasattr(self, '_entry_price') and close_now < self._entry_price * 0.995:
                    return "SELL"
        except Exception as e:
            logging.error(f"Error checking signals at idx {idx}: {e}")
        return None
