"""
MACD Histogram Strategy

This strategy uses the MACD histogram to measure the distance between the MACD line and signal line.
The histogram provides information about momentum strength and direction:
- Positive histogram: MACD above signal line, bullish momentum
- Negative histogram: MACD below signal line, bearish momentum
- Rising histogram: momentum increasing
- Falling histogram: momentum decreasing

Key Features:
- Provides detailed momentum information
- More useful in volatile markets and shorter timeframes
- Can be noisy and erratic
- Uses histogram slope and strength for signals
"""

import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy

class MACDHistogramStrategy(BaseStrategy):
    """
    MACD Histogram Strategy.
    
    This strategy uses MACD histogram characteristics to generate trading signals
    based on momentum strength and direction changes.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Strategy parameters with defaults
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        self.histogram_smoothing = params.get('histogram_smoothing', 3)
        self.min_histogram_strength = params.get('min_histogram_strength', 0.0005)
        self.momentum_threshold = params.get('momentum_threshold', 0.001)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.volume_ma_period = params.get('volume_ma_period', 20)
        self.min_volume_ratio = params.get('min_volume_ratio', 0.8)
        self.max_holding_period = params.get('max_holding_period', 20)
        self.profit_target_pct = params.get('profit_target_pct', 0.025)  # 2.5%
        self.stop_loss_pct = params.get('stop_loss_pct', 0.015)  # 1.5%
        
        # Position tracking
        self._entry_price = None
        self._entry_idx = None
        self._last_trade_idx = -1
        self._trade_cooldown = params.get('trade_cooldown', 2)
        
        super().__init__(data, **params)
        
    def _compute_indicators(self):
        """Compute technical indicators for MACD histogram strategy."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # MACD calculation (override parent with custom parameters)
        self.data['EMA_fast'] = self.data['Close'].ewm(span=self.macd_fast).mean()
        self.data['EMA_slow'] = self.data['Close'].ewm(span=self.macd_slow).mean()
        self.data['MACD'] = self.data['EMA_fast'] - self.data['EMA_slow']
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=self.macd_signal).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
        
        # Smoothed histogram to reduce noise
        self.data['MACD_Histogram_Smooth'] = self.data['MACD_histogram'].rolling(window=self.histogram_smoothing).mean()
        
        # Histogram slope (rate of change)
        self.data['Histogram_Slope'] = self.data['MACD_Histogram_Smooth'].diff()
        
        # Histogram strength (absolute value)
        self.data['Histogram_Strength'] = abs(self.data['MACD_Histogram_Smooth'])
        
        # Histogram momentum (second derivative)
        self.data['Histogram_Momentum'] = self.data['Histogram_Slope'].diff()
        
        # Zero line crossovers
        self.data['Histogram_Above_Zero'] = self.data['MACD_Histogram_Smooth'] > 0
        self.data['Histogram_Zero_Cross'] = self.data['Histogram_Above_Zero'] != self.data['Histogram_Above_Zero'].shift(1)
        
        # Volume indicators
        if 'Volume' in self.data.columns:
            self.data['Volume_MA'] = self.data['Volume'].rolling(window=self.volume_ma_period).mean()
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        else:
            self.data['Volume'] = 1000000
            self.data['Volume_MA'] = 1000000
            self.data['Volume_Ratio'] = 1.0
        
        # Price momentum for additional confirmation
        self.data['Price_Momentum'] = self.data['Close'].pct_change(periods=3)
        
        # MACD line momentum
        self.data['MACD_Momentum'] = self.data['MACD'].diff()
        
    def _is_bullish_histogram_signal(self, idx: int) -> bool:
        """Check for bullish histogram signal."""
        if idx < max(self.macd_slow, self.macd_signal, self.histogram_smoothing + 2):
            return False
        
        current_hist = self.data['MACD_Histogram_Smooth'].iloc[idx]
        prev_hist = self.data['MACD_Histogram_Smooth'].iloc[idx - 1]
        current_slope = self.data['Histogram_Slope'].iloc[idx]
        current_strength = self.data['Histogram_Strength'].iloc[idx]
        current_momentum = self.data['Histogram_Momentum'].iloc[idx]
        is_above_zero = self.data['Histogram_Above_Zero'].iloc[idx]
        zero_cross = self.data['Histogram_Zero_Cross'].iloc[idx]
        
        # Multiple bullish conditions (any can trigger)
        conditions = [
            # Histogram crosses above zero line
            zero_cross and is_above_zero,
            # Histogram turns positive from negative
            prev_hist < 0 and current_hist > 0,
            # Histogram slope turns positive (momentum increasing)
            current_slope > 0 and self.data['Histogram_Slope'].iloc[idx - 1] <= 0,
            # Strong positive histogram with increasing momentum
            current_hist > self.min_histogram_strength and current_momentum > 0,
            # Histogram strength increasing while positive
            is_above_zero and current_strength > self.data['Histogram_Strength'].iloc[idx - 1]
        ]
        
        return any(conditions)
    
    def _is_bearish_histogram_signal(self, idx: int) -> bool:
        """Check for bearish histogram signal."""
        if idx < max(self.macd_slow, self.macd_signal, self.histogram_smoothing + 2):
            return False
        
        current_hist = self.data['MACD_Histogram_Smooth'].iloc[idx]
        prev_hist = self.data['MACD_Histogram_Smooth'].iloc[idx - 1]
        current_slope = self.data['Histogram_Slope'].iloc[idx]
        current_strength = self.data['Histogram_Strength'].iloc[idx]
        current_momentum = self.data['Histogram_Momentum'].iloc[idx]
        is_above_zero = self.data['Histogram_Above_Zero'].iloc[idx]
        zero_cross = self.data['Histogram_Zero_Cross'].iloc[idx]
        
        # Multiple bearish conditions (any can trigger)
        conditions = [
            # Histogram crosses below zero line
            zero_cross and not is_above_zero,
            # Histogram turns negative from positive
            prev_hist > 0 and current_hist < 0,
            # Histogram slope turns negative (momentum decreasing)
            current_slope < 0 and self.data['Histogram_Slope'].iloc[idx - 1] >= 0,
            # Strong negative histogram with decreasing momentum
            current_hist < -self.min_histogram_strength and current_momentum < 0,
            # Histogram strength increasing while negative
            not is_above_zero and current_strength > self.data['Histogram_Strength'].iloc[idx - 1]
        ]
        
        return any(conditions)
    
    def _has_volume_confirmation(self, idx: int) -> bool:
        """Check if volume confirms the histogram signal."""
        if 'Volume_Ratio' not in self.data.columns:
            return True
        
        volume_ratio = self.data['Volume_Ratio'].iloc[idx]
        return volume_ratio >= self.min_volume_ratio
    
    def _has_momentum_confirmation(self, idx: int) -> bool:
        """Check if momentum confirms the histogram signal."""
        if idx < 3:
            return False
        
        # Price momentum
        price_momentum = self.data['Price_Momentum'].iloc[idx]
        
        # MACD momentum
        macd_momentum = self.data['MACD_Momentum'].iloc[idx]
        
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
        # 1. Bearish histogram signal
        if self._is_bearish_histogram_signal(idx):
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
        
        # 6. Histogram becomes weak
        current_strength = self.data['Histogram_Strength'].iloc[idx]
        if current_strength < self.min_histogram_strength * 0.5:
            return True
        
        return False
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on MACD histogram characteristics.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < max(self.macd_slow, self.macd_signal, self.histogram_smoothing + 2):
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            current_price = self.data['Close'].iloc[idx]
            rsi = self.data['RSI'].iloc[idx]
            price_momentum = self.data['Price_Momentum'].iloc[idx]
            macd_momentum = self.data['MACD_Momentum'].iloc[idx]
            
            if not self.bought:
                # Look for bullish histogram signal with confirmations
                if (self._is_bullish_histogram_signal(idx) and
                    rsi < self.rsi_overbought and  # RSI not overbought
                    price_momentum > 0 and  # Positive price momentum
                    macd_momentum > 0 and  # Positive MACD momentum
                    self._has_volume_confirmation(idx)):
                    
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
