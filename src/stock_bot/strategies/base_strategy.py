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
        Default implementation provides common indicators with improved robustness.
        """
        try:
            # Validate data
            if self.data.empty or 'Close' not in self.data.columns:
                logging.error("Invalid data: empty or missing Close column")
                return
            
            # Ensure we have enough data
            min_required = 20  # Reduced from 50 to 20
            if len(self.data) < min_required:
                logging.warning(f"Insufficient data: {len(self.data)} rows, need at least {min_required}")
            
            # 1. Enhanced MACD with better parameters
            self.data['EMA_fast'] = self.data['Close'].ewm(span=12, adjust=False).mean()
            self.data['EMA_slow'] = self.data['Close'].ewm(span=26, adjust=False).mean()
            self.data['MACD'] = self.data['EMA_fast'] - self.data['EMA_slow']
            self.data['MACD_signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
            self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
            
            # 2. Improved RSI calculation with Wilder's smoothing
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Use Wilder's smoothing for more accurate RSI
            alpha = 1.0 / 14  # RSI period
            gain_ema = gain.ewm(alpha=alpha, adjust=False).mean()
            loss_ema = loss.ewm(alpha=alpha, adjust=False).mean()
            
            rs = gain_ema / loss_ema
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # 3. Multiple timeframe ROC for better momentum detection
            self.data['ROC_1'] = self.data['Close'].pct_change(periods=1)
            self.data['ROC_5'] = self.data['Close'].pct_change(periods=5)
            self.data['ROC_10'] = self.data['Close'].pct_change(periods=10)
            
            # 4. Enhanced Bollinger Bands with adaptive periods
            bb_period = 20
            bb_std = 2
            self.data['BB_middle'] = self.data['Close'].rolling(window=bb_period).mean()
            bb_std_val = self.data['Close'].rolling(window=bb_period).std()
            self.data['BB_upper'] = self.data['BB_middle'] + (bb_std_val * bb_std)
            self.data['BB_lower'] = self.data['BB_middle'] - (bb_std_val * bb_std)
            self.data['BB_width'] = (self.data['BB_upper'] - self.data['BB_lower']) / self.data['BB_middle']
            self.data['BB_position'] = (self.data['Close'] - self.data['BB_lower']) / (self.data['BB_upper'] - self.data['BB_lower'])
            
            # 5. Improved ATR calculation with proper high/low handling
            if 'High' in self.data.columns and 'Low' in self.data.columns:
                high = self.data['High']
                low = self.data['Low']
            else:
                # Use Close as proxy if High/Low not available
                high = self.data['Close']
                low = self.data['Close']
            
            prev_close = self.data['Close'].shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.data['ATR'] = tr.rolling(window=14).mean()
            
            # 6. Additional momentum indicators
            self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
            self.data['EMA_20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
            
            # 7. Volume indicators (if available)
            if 'Volume' in self.data.columns:
                self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
                self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
            else:
                self.data['Volume_Ratio'] = 1.0
            
            # 8. Volatility indicators
            self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
            self.data['Price_Range'] = (self.data['Close'].rolling(window=20).max() - self.data['Close'].rolling(window=20).min()) / self.data['Close'].rolling(window=20).mean()
            
            # 9. Trend indicators
            self.data['Trend_Strength'] = (self.data['Close'] - self.data['SMA_20']) / self.data['SMA_20']
            self.data['Momentum'] = self.data['Close'] / self.data['Close'].shift(10) - 1
            
            # 10. Market regime detection
            self.data['Market_Regime'] = 'Neutral'  # Default
            uptrend_mask = (self.data['Close'] > self.data['SMA_20']) & (self.data['SMA_20'] > self.data['SMA_50'])
            downtrend_mask = (self.data['Close'] < self.data['SMA_20']) & (self.data['SMA_20'] < self.data['SMA_50'])
            
            self.data.loc[uptrend_mask, 'Market_Regime'] = 'Uptrend'
            self.data.loc[downtrend_mask, 'Market_Regime'] = 'Downtrend'
            
        except Exception as e:
            logging.error(f"Error computing indicators: {e}")
            # Set default values to prevent crashes
            self.data['RSI'] = 50.0
            self.data['MACD'] = 0.0
            self.data['MACD_signal'] = 0.0
            self.data['ATR'] = 0.0

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
