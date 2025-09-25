"""
MACD Crossover Strategy

This strategy uses MACD line and signal line crossovers to generate buy and sell signals.
A buy signal occurs when the MACD line crosses above the signal line, indicating rising momentum.
A sell signal occurs when the MACD line crosses below the signal line, indicating falling momentum.

Key Features:
- Simple and easy to follow
- Works well in trending markets and higher timeframes
- Can produce false signals in choppy markets
- Uses additional filters to reduce whipsaws
"""

import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy

class MACDCrossoverStrategy(BaseStrategy):
    """
    MACD Crossover Strategy.
    
    This strategy generates signals based on MACD line and signal line crossovers,
    with additional filters to reduce false signals in choppy markets.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Strategy parameters with defaults - more responsive
        self.macd_fast = 8         # Reduced from 12 to 8
        self.macd_slow = 21        # Reduced from 26 to 21
        self.macd_signal = 5       # Reduced from 9 to 5
        self.rsi_period = 14
        self.rsi_oversold = 35     # Increased from 30 to 35
        self.rsi_overbought = 65   # Decreased from 70 to 65
        self.volume_ma_period = 20
        self.min_volume_ratio = 0.6  # Reduced from 0.8 to 0.6
        self.trend_filter_period = 30  # Reduced from 50 to 30
        self.min_macd_strength = 0.0005  # Reduced from 0.001 to 0.0005
        self.max_holding_period = 20  # Reduced from 30 to 20
        self.profit_target_pct = 0.02  # Reduced from 0.03 to 0.02
        self.stop_loss_pct = 0.015  # Reduced from 0.02 to 0.015
        
        # Position tracking
        self._entry_price = None
        self._entry_idx = None
        self._last_trade_idx = -1
        self._trade_cooldown = 1  # Reduced from 3 to 1
        
        super().__init__(data, **params)
        
    def _compute_indicators(self):
        """Compute technical indicators for MACD crossover strategy."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # MACD calculation (override parent with custom parameters)
        self.data['EMA_fast'] = self.data['Close'].ewm(span=self.macd_fast).mean()
        self.data['EMA_slow'] = self.data['Close'].ewm(span=self.macd_slow).mean()
        self.data['MACD'] = self.data['EMA_fast'] - self.data['EMA_slow']
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=self.macd_signal).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
        
        # Trend filter (longer-term moving average)
        self.data['Trend_Filter'] = self.data['Close'].rolling(window=self.trend_filter_period).mean()
        
        # MACD strength (absolute value of MACD)
        self.data['MACD_Strength'] = abs(self.data['MACD'])
        
        # Volume indicators
        if 'Volume' in self.data.columns:
            self.data['Volume_MA'] = self.data['Volume'].rolling(window=self.volume_ma_period).mean()
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        else:
            self.data['Volume'] = 1000000
            self.data['Volume_MA'] = 1000000
            self.data['Volume_Ratio'] = 1.0
        
        # Price momentum for additional confirmation
        self.data['Price_Momentum'] = self.data['Close'].pct_change(periods=5)
        
    def _is_bullish_crossover(self, idx: int) -> bool:
        """Check for bullish MACD crossover."""
        if idx < max(self.macd_slow, self.macd_signal):
            return False
        
        current_macd = self.data['MACD'].iloc[idx]
        current_signal = self.data['MACD_signal'].iloc[idx]
        prev_macd = self.data['MACD'].iloc[idx - 1]
        prev_signal = self.data['MACD_signal'].iloc[idx - 1]
        
        # MACD crosses above signal line
        return (prev_macd <= prev_signal and current_macd > current_signal)
    
    def _is_bearish_crossover(self, idx: int) -> bool:
        """Check for bearish MACD crossover."""
        if idx < max(self.macd_slow, self.macd_signal):
            return False
        
        current_macd = self.data['MACD'].iloc[idx]
        current_signal = self.data['MACD_signal'].iloc[idx]
        prev_macd = self.data['MACD'].iloc[idx - 1]
        prev_signal = self.data['MACD_signal'].iloc[idx - 1]
        
        # MACD crosses below signal line
        return (prev_macd >= prev_signal and current_macd < current_signal)
    
    def _has_trend_confirmation(self, idx: int) -> bool:
        """Check if trend confirms the signal."""
        if idx < self.trend_filter_period:
            return False
        
        current_price = self.data['Close'].iloc[idx]
        trend_filter = self.data['Trend_Filter'].iloc[idx]
        
        # For buy signals, price should be above trend filter
        # For sell signals, price should be below trend filter
        return True  # We'll check this in the signal logic
    
    def _has_volume_confirmation(self, idx: int) -> bool:
        """Check if volume confirms the signal."""
        if 'Volume_Ratio' not in self.data.columns:
            return True
        
        volume_ratio = self.data['Volume_Ratio'].iloc[idx]
        return volume_ratio >= self.min_volume_ratio
    
    def _has_momentum_confirmation(self, idx: int) -> bool:
        """Check if momentum confirms the signal."""
        if idx < 5:
            return False
        
        # MACD strength
        macd_strength = self.data['MACD_Strength'].iloc[idx]
        if macd_strength < self.min_macd_strength:
            return False
        
        # Price momentum
        price_momentum = self.data['Price_Momentum'].iloc[idx]
        
        # For buy signals, we want positive momentum
        # For sell signals, we want negative momentum
        return True  # We'll check this in the signal logic
    
    def _should_exit_long(self, idx: int) -> bool:
        """Check if we should exit a long position."""
        if not self.bought or self._entry_price is None or self._entry_idx is None:
            return False
        
        current_price = self.data['Close'].iloc[idx]
        holding_period = idx - self._entry_idx
        
        # Exit conditions
        # 1. Bearish MACD crossover
        if self._is_bearish_crossover(idx):
            return True
        
        # 2. Profit target hit
        if current_price >= self._entry_price * (1 + self.profit_target_pct):
            return True
        
        # 3. Stop loss hit
        if current_price <= self._entry_price * (1 - self.stop_loss_pct):
            return True
        
        # 4. Maximum holding period reached
        if holding_period >= self.max_holding_period:
            return True
        
        # 5. RSI becomes overbought
        if self.data['RSI'].iloc[idx] > self.rsi_overbought:
            return True
        
        # 6. Price falls below trend filter
        trend_filter = self.data['Trend_Filter'].iloc[idx]
        if current_price < trend_filter * 0.98:  # 2% below trend
            return True
        
        return False
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on MACD crossovers.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < max(self.macd_slow, self.macd_signal, self.trend_filter_period):
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            current_price = self.data['Close'].iloc[idx]
            rsi = self.data['RSI'].iloc[idx]
            trend_filter = self.data['Trend_Filter'].iloc[idx]
            price_momentum = self.data['Price_Momentum'].iloc[idx]
            
            if not self.bought:
                # Look for bullish crossover with confirmations
                if (self._is_bullish_crossover(idx) and
                    current_price > trend_filter and  # Price above trend
                    rsi < self.rsi_overbought and  # RSI not overbought
                    price_momentum > 0 and  # Positive momentum
                    self._has_volume_confirmation(idx) and
                    self._has_momentum_confirmation(idx)):
                    
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
