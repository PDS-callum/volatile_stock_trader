"""
MACD Divergence Strategy

This strategy uses MACD histogram to detect divergences between price movement and MACD movement.
A bullish divergence occurs when price makes a lower low but MACD makes a higher low.
A bearish divergence occurs when price makes a higher high but MACD makes a lower high.

Key Features:
- Anticipates trend changes before they occur
- More effective in overbought/oversold markets
- Can be difficult to spot and confirm
- Uses multiple confirmation signals
"""

import pandas as pd
from typing import Optional, List, Tuple
from .base_strategy import BaseStrategy

class MACDDivergenceStrategy(BaseStrategy):
    """
    MACD Divergence Strategy.
    
    This strategy identifies divergences between price and MACD to anticipate trend reversals.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Strategy parameters with defaults
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.divergence_lookback = 20
        self.min_divergence_strength = 0.02  # 2%
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.volume_ma_period = 20
        self.min_volume_ratio = 0.8
        self.max_holding_period = 25
        self.profit_target_pct = 0.04  # 4%
        self.stop_loss_pct = 0.025  # 2.5%
        
        # Position tracking
        self._entry_price = None
        self._entry_idx = None
        self._last_trade_idx = -1
        self._trade_cooldown = 5
        
        # Divergence tracking
        self._price_peaks = []
        self._price_troughs = []
        self._macd_peaks = []
        self._macd_troughs = []
        
        super().__init__(data, **params)
        
    def _compute_indicators(self):
        """Compute technical indicators for MACD divergence strategy."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # MACD calculation (override parent with custom parameters)
        self.data['EMA_fast'] = self.data['Close'].ewm(span=self.macd_fast).mean()
        self.data['EMA_slow'] = self.data['Close'].ewm(span=self.macd_slow).mean()
        self.data['MACD'] = self.data['EMA_fast'] - self.data['EMA_slow']
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=self.macd_signal).mean()
        self.data['MACD_histogram'] = self.data['MACD'] - self.data['MACD_signal']
        
        # Price peaks and troughs
        self.data['Price_Peak'] = self.data['Close'].rolling(window=5, center=True).max() == self.data['Close']
        self.data['Price_Trough'] = self.data['Close'].rolling(window=5, center=True).min() == self.data['Close']
        
        # MACD peaks and troughs
        self.data['MACD_Peak'] = self.data['MACD_histogram'].rolling(window=5, center=True).max() == self.data['MACD_histogram']
        self.data['MACD_Trough'] = self.data['MACD_histogram'].rolling(window=5, center=True).min() == self.data['MACD_histogram']
        
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
        
    def _find_peaks_and_troughs(self, idx: int) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Find recent price and MACD peaks and troughs."""
        start_idx = max(0, idx - self.divergence_lookback)
        
        price_peaks = []
        price_troughs = []
        macd_peaks = []
        macd_troughs = []
        
        for i in range(start_idx, idx + 1):
            if i < len(self.data) and self.data['Price_Peak'].iloc[i]:
                price_peaks.append(i)
            if i < len(self.data) and self.data['Price_Trough'].iloc[i]:
                price_troughs.append(i)
            if i < len(self.data) and self.data['MACD_Peak'].iloc[i]:
                macd_peaks.append(i)
            if i < len(self.data) and self.data['MACD_Trough'].iloc[i]:
                macd_troughs.append(i)
        
        return price_peaks, price_troughs, macd_peaks, macd_troughs
    
    def _detect_bullish_divergence(self, idx: int) -> bool:
        """Detect bullish divergence (price lower low, MACD higher low)."""
        price_peaks, price_troughs, macd_peaks, macd_troughs = self._find_peaks_and_troughs(idx)
        
        if len(price_troughs) < 2 or len(macd_troughs) < 2:
            return False
        
        # Get the last two price troughs
        last_price_trough = price_troughs[-1]
        prev_price_trough = price_troughs[-2]
        
        # Get the last two MACD troughs
        last_macd_trough = macd_troughs[-1]
        prev_macd_trough = macd_troughs[-2]
        
        # Check if we have recent troughs
        if (idx - last_price_trough > 10 or idx - last_macd_trough > 10):
            return False
        
        # Price makes lower low
        last_price = self.data['Close'].iloc[last_price_trough]
        prev_price = self.data['Close'].iloc[prev_price_trough]
        price_lower_low = last_price < prev_price
        
        # MACD makes higher low
        last_macd = self.data['MACD_histogram'].iloc[last_macd_trough]
        prev_macd = self.data['MACD_histogram'].iloc[prev_macd_trough]
        macd_higher_low = last_macd > prev_macd
        
        # Check divergence strength
        price_change = abs(last_price - prev_price) / prev_price
        macd_change = abs(last_macd - prev_macd) / abs(prev_macd) if prev_macd != 0 else 0
        
        return (price_lower_low and macd_higher_low and 
                price_change > self.min_divergence_strength and
                macd_change > 0.1)  # 10% MACD change
    
    def _detect_bearish_divergence(self, idx: int) -> bool:
        """Detect bearish divergence (price higher high, MACD lower high)."""
        price_peaks, price_troughs, macd_peaks, macd_troughs = self._find_peaks_and_troughs(idx)
        
        if len(price_peaks) < 2 or len(macd_peaks) < 2:
            return False
        
        # Get the last two price peaks
        last_price_peak = price_peaks[-1]
        prev_price_peak = price_peaks[-2]
        
        # Get the last two MACD peaks
        last_macd_peak = macd_peaks[-1]
        prev_macd_peak = macd_peaks[-2]
        
        # Check if we have recent peaks
        if (idx - last_price_peak > 10 or idx - last_macd_peak > 10):
            return False
        
        # Price makes higher high
        last_price = self.data['Close'].iloc[last_price_peak]
        prev_price = self.data['Close'].iloc[prev_price_peak]
        price_higher_high = last_price > prev_price
        
        # MACD makes lower high
        last_macd = self.data['MACD_histogram'].iloc[last_macd_peak]
        prev_macd = self.data['MACD_histogram'].iloc[prev_macd_peak]
        macd_lower_high = last_macd < prev_macd
        
        # Check divergence strength
        price_change = abs(last_price - prev_price) / prev_price
        macd_change = abs(last_macd - prev_macd) / abs(prev_macd) if prev_macd != 0 else 0
        
        return (price_higher_high and macd_lower_high and 
                price_change > self.min_divergence_strength and
                macd_change > 0.1)  # 10% MACD change
    
    def _has_volume_confirmation(self, idx: int) -> bool:
        """Check if volume confirms the divergence signal."""
        if 'Volume_Ratio' not in self.data.columns:
            return True
        
        volume_ratio = self.data['Volume_Ratio'].iloc[idx]
        return volume_ratio >= self.min_volume_ratio
    
    def _is_oversold_condition(self, idx: int) -> bool:
        """Check if market is in oversold condition for bullish divergence."""
        rsi = self.data['RSI'].iloc[idx]
        price_vs_sma = (self.data['Close'].iloc[idx] - self.data['SMA_20'].iloc[idx]) / self.data['SMA_20'].iloc[idx]
        
        return rsi < self.rsi_oversold or price_vs_sma < -0.05  # 5% below SMA
    
    def _is_overbought_condition(self, idx: int) -> bool:
        """Check if market is in overbought condition for bearish divergence."""
        rsi = self.data['RSI'].iloc[idx]
        price_vs_sma = (self.data['Close'].iloc[idx] - self.data['SMA_20'].iloc[idx]) / self.data['SMA_20'].iloc[idx]
        
        return rsi > self.rsi_overbought or price_vs_sma > 0.05  # 5% above SMA
    
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
        
        # 4. Bearish divergence detected
        if self._detect_bearish_divergence(idx):
            return True
        
        # 5. RSI becomes overbought
        if self.data['RSI'].iloc[idx] > self.rsi_overbought:
            return True
        
        return False
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on MACD divergences.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < max(self.macd_slow, self.macd_signal, self.divergence_lookback):
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            current_price = self.data['Close'].iloc[idx]
            
            if not self.bought:
                # Look for bullish divergence with confirmations
                if (self._detect_bullish_divergence(idx) and
                    self._is_oversold_condition(idx) and
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
