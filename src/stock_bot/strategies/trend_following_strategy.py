"""
Trend-Following Momentum Strategy

This strategy identifies and rides trends using multiple technical indicators:
- Moving Average Crossovers (Golden Cross/Death Cross)
- RSI momentum confirmation
- Volume confirmation
- Trend strength indicators

The strategy buys into upward trends and sells into downward trends,
aiming to capture large moves while avoiding false signals.
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-Following Momentum Strategy.
    
    This strategy uses multiple technical indicators to identify strong trends
    and ride them for maximum profit while minimizing false signals.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Strategy parameters with defaults - set before calling parent
        self.short_ma_period = params.get('short_ma_period', 20)
        self.long_ma_period = params.get('long_ma_period', 50)
        self.trend_ma_period = params.get('trend_ma_period', 200)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.volume_ma_period = params.get('volume_ma_period', 20)
        self.min_volume_ratio = params.get('min_volume_ratio', 1.2)
        self.trend_strength_period = params.get('trend_strength_period', 10)
        self.min_trend_strength = params.get('min_trend_strength', 0.02)  # 2%
        
        # Position tracking
        self._entry_price = None
        self._peak_price = None
        self._trough_price = None
        self._last_trade_idx = -1
        self._trade_cooldown = params.get('trade_cooldown', 5)  # Minimum bars between trades
        
        super().__init__(data, **params)
        
    def _compute_indicators(self):
        """Compute technical indicators for trend following."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # Moving Averages for trend identification
        self.data['SMA_short'] = self.data['Close'].rolling(window=self.short_ma_period).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=self.long_ma_period).mean()
        self.data['SMA_trend'] = self.data['Close'].rolling(window=self.trend_ma_period).mean()
        
        # Exponential Moving Averages for more responsive signals
        self.data['EMA_short'] = self.data['Close'].ewm(span=self.short_ma_period).mean()
        self.data['EMA_long'] = self.data['Close'].ewm(span=self.long_ma_period).mean()
        
        # Volume indicators
        if 'Volume' in self.data.columns:
            self.data['Volume_MA'] = self.data['Volume'].rolling(window=self.volume_ma_period).mean()
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        else:
            # If no volume data, create dummy volume
            self.data['Volume'] = 1000000  # Default volume
            self.data['Volume_MA'] = 1000000
            self.data['Volume_Ratio'] = 1.0
        
        # Trend strength indicator (price momentum over period)
        self.data['Trend_Strength'] = (
            (self.data['Close'] - self.data['Close'].shift(self.trend_strength_period)) 
            / self.data['Close'].shift(self.trend_strength_period)
        )
        
        # Price position relative to trend
        self.data['Price_vs_Trend'] = (self.data['Close'] - self.data['SMA_trend']) / self.data['SMA_trend']
        
        # MACD for momentum confirmation
        self.data['MACD_Line'] = self.data['EMA_short'] - self.data['EMA_long']
        self.data['MACD_Signal'] = self.data['MACD_Line'].ewm(span=9).mean()
        self.data['MACD_Histogram'] = self.data['MACD_Line'] - self.data['MACD_Signal']
        
        # Bollinger Bands for volatility
        bb_period = 20
        bb_std = 2
        self.data['BB_Middle'] = self.data['Close'].rolling(window=bb_period).mean()
        bb_std_val = self.data['Close'].rolling(window=bb_period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std_val * bb_std)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std_val * bb_std)
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        
        # Average True Range for volatility
        high = self.data['High'] if 'High' in self.data.columns else self.data['Close']
        low = self.data['Low'] if 'Low' in self.data.columns else self.data['Close']
        prev_close = self.data['Close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        self.data['ATR'] = tr.rolling(window=14).mean()
        
    def _is_golden_cross(self, idx: int) -> bool:
        """Check for golden cross (short MA crosses above long MA)."""
        if idx < self.long_ma_period:
            return False
        
        current_short = self.data['SMA_short'].iloc[idx]
        current_long = self.data['SMA_long'].iloc[idx]
        prev_short = self.data['SMA_short'].iloc[idx - 1]
        prev_long = self.data['SMA_long'].iloc[idx - 1]
        
        return (prev_short <= prev_long and current_short > current_long)
    
    def _is_death_cross(self, idx: int) -> bool:
        """Check for death cross (short MA crosses below long MA)."""
        if idx < self.long_ma_period:
            return False
        
        current_short = self.data['SMA_short'].iloc[idx]
        current_long = self.data['SMA_long'].iloc[idx]
        prev_short = self.data['SMA_short'].iloc[idx - 1]
        prev_long = self.data['SMA_long'].iloc[idx - 1]
        
        return (prev_short >= prev_long and current_short < current_long)
    
    def _is_uptrend(self, idx: int) -> bool:
        """Check if we're in an uptrend based on multiple factors."""
        if idx < self.trend_ma_period:
            return False
        
        # Price above trend MA
        price_above_trend = self.data['Close'].iloc[idx] > self.data['SMA_trend'].iloc[idx]
        
        # Short MA above long MA
        ma_alignment = self.data['SMA_short'].iloc[idx] > self.data['SMA_long'].iloc[idx]
        
        # Positive trend strength
        trend_strength = self.data['Trend_Strength'].iloc[idx] > self.min_trend_strength
        
        # RSI not overbought
        rsi = self.data['RSI'].iloc[idx]
        rsi_ok = rsi < self.rsi_overbought
        
        # MACD bullish
        macd_bullish = self.data['MACD_Line'].iloc[idx] > self.data['MACD_Signal'].iloc[idx]
        
        return all([price_above_trend, ma_alignment, trend_strength, rsi_ok, macd_bullish])
    
    def _is_downtrend(self, idx: int) -> bool:
        """Check if we're in a downtrend based on multiple factors."""
        if idx < self.trend_ma_period:
            return False
        
        # Price below trend MA
        price_below_trend = self.data['Close'].iloc[idx] < self.data['SMA_trend'].iloc[idx]
        
        # Short MA below long MA
        ma_alignment = self.data['SMA_short'].iloc[idx] < self.data['SMA_long'].iloc[idx]
        
        # Negative trend strength
        trend_strength = self.data['Trend_Strength'].iloc[idx] < -self.min_trend_strength
        
        # RSI not oversold
        rsi = self.data['RSI'].iloc[idx]
        rsi_ok = rsi > self.rsi_oversold
        
        # MACD bearish
        macd_bearish = self.data['MACD_Line'].iloc[idx] < self.data['MACD_Signal'].iloc[idx]
        
        return all([price_below_trend, ma_alignment, trend_strength, rsi_ok, macd_bearish])
    
    def _has_volume_confirmation(self, idx: int) -> bool:
        """Check if volume confirms the signal."""
        if 'Volume_Ratio' not in self.data.columns:
            return True  # If no volume data, assume confirmation
        
        return self.data['Volume_Ratio'].iloc[idx] >= self.min_volume_ratio
    
    def _should_exit_long(self, idx: int) -> bool:
        """Check if we should exit a long position."""
        if not self.bought or self._entry_price is None:
            return False
        
        current_price = self.data['Close'].iloc[idx]
        
        # Update peak price for trailing stop
        if self._peak_price is None or current_price > self._peak_price:
            self._peak_price = current_price
        
        # Exit conditions
        # 1. Death cross
        if self._is_death_cross(idx):
            return True
        
        # 2. RSI overbought
        if self.data['RSI'].iloc[idx] > self.rsi_overbought:
            return True
        
        # 3. MACD bearish crossover
        if (self.data['MACD_Line'].iloc[idx] < self.data['MACD_Signal'].iloc[idx] and
            self.data['MACD_Line'].iloc[idx - 1] >= self.data['MACD_Signal'].iloc[idx - 1]):
            return True
        
        # 4. Trailing stop (2% below peak)
        trailing_stop_pct = 0.02
        if current_price < self._peak_price * (1 - trailing_stop_pct):
            return True
        
        # 5. Stop loss (5% below entry)
        stop_loss_pct = 0.05
        if current_price < self._entry_price * (1 - stop_loss_pct):
            return True
        
        # 6. Trend reversal
        if not self._is_uptrend(idx):
            return True
        
        return False
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on trend following logic.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < self.trend_ma_period:
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            current_price = self.data['Close'].iloc[idx]
            
            if not self.bought:
                # Look for buy signals
                if (self._is_golden_cross(idx) and 
                    self._is_uptrend(idx) and 
                    self._has_volume_confirmation(idx)):
                    
                    self.bought = True
                    self._entry_price = current_price
                    self._peak_price = current_price
                    self._last_trade_idx = idx
                    return "BUY"
            
            else:
                # Check for sell signals
                if self._should_exit_long(idx):
                    self.bought = False
                    self._entry_price = None
                    self._peak_price = None
                    self._last_trade_idx = idx
                    return "SELL"
                    
        except Exception as e:
            print(f"Error checking signals at idx {idx}: {e}")
        
        return None
