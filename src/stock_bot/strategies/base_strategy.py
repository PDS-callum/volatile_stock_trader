import pandas as pd
import logging
from typing import Any, List, Optional
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies. 
    
    All custom strategies must inherit from this class and implement the check_signals method.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        """
        Initialize the strategy with historical data and parameters.
        
        Args:
            data: Historical price data DataFrame with OHLCV columns
            **params: Strategy-specific parameters
        """
        self.data = data.copy()
        self.params = params
        self.bought = False
        self.signals: List[Any] = []
        self._compute_indicators()

    def _compute_indicators(self):
        """
        Compute technical indicators. Can be overridden by subclasses.
        Default implementation provides common indicators.
        """
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

    @abstractmethod
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals at a given index.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        pass

    def reset_position(self):
        """Reset the strategy's position state."""
        self.bought = False
        self.signals = []
