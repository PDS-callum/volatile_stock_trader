"""
Mean Reversion Strategy

This strategy implements mean-reversion trading based on the principle that asset prices
tend to revert to their long-term average after deviating significantly from it.

Key Features:
- Bollinger Bands for extreme price detection
- Z-score analysis for statistical significance
- RSI for momentum confirmation
- Volume analysis for signal validation
- Multiple timeframe mean reversion
- Risk management with position sizing
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.
    
    This strategy identifies overbought/oversold conditions and bets on price
    reversion to the mean using multiple technical indicators and statistical analysis.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Strategy parameters with defaults
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2.0)
        self.zscore_period = params.get('zscore_period', 20)
        self.zscore_threshold = params.get('zscore_threshold', 2.0)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.volume_ma_period = params.get('volume_ma_period', 20)
        self.min_volume_ratio = params.get('min_volume_ratio', 0.8)
        self.max_volume_ratio = params.get('max_volume_ratio', 2.0)
        self.mean_reversion_period = params.get('mean_reversion_period', 10)
        self.min_reversion_strength = params.get('min_reversion_strength', 0.01)
        self.max_holding_period = params.get('max_holding_period', 20)
        self.profit_target_pct = params.get('profit_target_pct', 0.02)  # 2%
        self.stop_loss_pct = params.get('stop_loss_pct', 0.03)  # 3%
        
        # Position tracking
        self._entry_price = None
        self._entry_idx = None
        self._last_trade_idx = -1
        self._trade_cooldown = params.get('trade_cooldown', 3)
        
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
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        
        # Z-Score (price deviation from mean)
        rolling_mean = self.data['Close'].rolling(window=self.zscore_period).mean()
        rolling_std = self.data['Close'].rolling(window=self.zscore_period).std()
        self.data['Z_Score'] = (self.data['Close'] - rolling_mean) / rolling_std
        
        # Price position relative to moving averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['Price_vs_SMA20'] = (self.data['Close'] - self.data['SMA_20']) / self.data['SMA_20']
        self.data['Price_vs_SMA50'] = (self.data['Close'] - self.data['SMA_50']) / self.data['SMA_50']
        
        # Mean reversion strength (how far price has moved from mean)
        self.data['Mean_Reversion_Strength'] = abs(self.data['Price_vs_SMA20'])
        
        # Volume indicators
        if 'Volume' in self.data.columns:
            self.data['Volume_MA'] = self.data['Volume'].rolling(window=self.volume_ma_period).mean()
            self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        else:
            self.data['Volume'] = 1000000
            self.data['Volume_MA'] = 1000000
            self.data['Volume_Ratio'] = 1.0
        
        # Williams %R for additional oversold/overbought confirmation
        high = self.data['High'] if 'High' in self.data.columns else self.data['Close']
        low = self.data['Low'] if 'Low' in self.data.columns else self.data['Close']
        self.data['Williams_R'] = ((high.rolling(window=14).max() - self.data['Close']) / 
                                  (high.rolling(window=14).max() - low.rolling(window=14).min())) * -100
        
        # Stochastic Oscillator
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        self.data['Stoch_K'] = ((self.data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(window=3).mean()
        
        # Commodity Channel Index (CCI)
        typical_price = (high + low + self.data['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        self.data['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
    def _is_oversold(self, idx: int) -> bool:
        """Check if price is in oversold territory."""
        if idx < max(self.bb_period, self.zscore_period, 20):
            return False
        
        current_price = self.data['Close'].iloc[idx]
        bb_lower = self.data['BB_Lower'].iloc[idx]
        bb_upper = self.data['BB_Upper'].iloc[idx]
        bb_middle = self.data['BB_Middle'].iloc[idx]
        z_score = self.data['Z_Score'].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        williams_r = self.data['Williams_R'].iloc[idx]
        stoch_k = self.data['Stoch_K'].iloc[idx]
        cci = self.data['CCI'].iloc[idx]
        price_vs_sma20 = self.data['Price_vs_SMA20'].iloc[idx]
        
        # Multiple oversold conditions (any can trigger)
        conditions = [
            # Price below lower Bollinger Band
            current_price < bb_lower,
            # Z-score below threshold (extreme deviation)
            z_score < -self.zscore_threshold,
            # RSI oversold
            rsi < self.rsi_oversold,
            # Williams %R oversold
            williams_r < -80,
            # Stochastic oversold
            stoch_k < 20,
            # CCI oversold
            cci < -100,
            # Price significantly below 20-day SMA
            price_vs_sma20 < -0.05  # 5% below SMA
        ]
        
        return any(conditions)
    
    def _is_overbought(self, idx: int) -> bool:
        """Check if price is in overbought territory."""
        if idx < max(self.bb_period, self.zscore_period, 20):
            return False
        
        current_price = self.data['Close'].iloc[idx]
        bb_upper = self.data['BB_Upper'].iloc[idx]
        z_score = self.data['Z_Score'].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        williams_r = self.data['Williams_R'].iloc[idx]
        stoch_k = self.data['Stoch_K'].iloc[idx]
        cci = self.data['CCI'].iloc[idx]
        price_vs_sma20 = self.data['Price_vs_SMA20'].iloc[idx]
        
        # Multiple overbought conditions (any can trigger)
        conditions = [
            # Price above upper Bollinger Band
            current_price > bb_upper,
            # Z-score above threshold (extreme deviation)
            z_score > self.zscore_threshold,
            # RSI overbought
            rsi > self.rsi_overbought,
            # Williams %R overbought
            williams_r > -20,
            # Stochastic overbought
            stoch_k > 80,
            # CCI overbought
            cci > 100,
            # Price significantly above 20-day SMA
            price_vs_sma20 > 0.05  # 5% above SMA
        ]
        
        return any(conditions)
    
    def _has_volume_confirmation(self, idx: int) -> bool:
        """Check if volume confirms the mean reversion signal."""
        if 'Volume_Ratio' not in self.data.columns:
            return True
        
        volume_ratio = self.data['Volume_Ratio'].iloc[idx]
        return self.min_volume_ratio <= volume_ratio <= self.max_volume_ratio
    
    def _is_mean_reverting(self, idx: int) -> bool:
        """Check if price is showing signs of mean reversion."""
        if idx < self.mean_reversion_period:
            return False
        
        # Check if price is moving back toward the mean
        current_price = self.data['Close'].iloc[idx]
        sma20 = self.data['SMA_20'].iloc[idx]
        bb_middle = self.data['BB_Middle'].iloc[idx]
        
        # Price moving toward SMA20
        price_moving_to_sma = abs(current_price - sma20) < abs(self.data['Close'].iloc[idx-1] - self.data['SMA_20'].iloc[idx-1])
        
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
        
        # 4. Price returns to mean (SMA20)
        sma20 = self.data['SMA_20'].iloc[idx]
        if current_price >= sma20 * 0.98:  # Within 2% of SMA20
            return True
        
        # 5. Price returns to BB middle
        bb_middle = self.data['BB_Middle'].iloc[idx]
        if current_price >= bb_middle * 0.98:  # Within 2% of BB middle
            return True
        
        # 6. RSI becomes overbought
        if self.data['RSI'].iloc[idx] > self.rsi_overbought:
            return True
        
        # 7. Price becomes overbought
        if self._is_overbought(idx):
            return True
        
        return False
    
    def _should_exit_short(self, idx: int) -> bool:
        """Check if we should exit a short position."""
        if not self.bought or self._entry_price is None or self._entry_idx is None:
            return False
        
        current_price = self.data['Close'].iloc[idx]
        holding_period = idx - self._entry_idx
        
        # Exit conditions (opposite of long)
        # 1. Profit target hit (price fell)
        if current_price <= self._entry_price * (1 - self.profit_target_pct):
            return True
        
        # 2. Stop loss hit (price rose)
        if current_price >= self._entry_price * (1 + self.stop_loss_pct):
            return True
        
        # 3. Maximum holding period reached
        if holding_period >= self.max_holding_period:
            return True
        
        # 4. Price returns to mean (SMA20)
        sma20 = self.data['SMA_20'].iloc[idx]
        if current_price <= sma20 * 1.02:  # Within 2% of SMA20
            return True
        
        # 5. Price returns to BB middle
        bb_middle = self.data['BB_Middle'].iloc[idx]
        if current_price <= bb_middle * 1.02:  # Within 2% of BB middle
            return True
        
        # 6. RSI becomes oversold
        if self.data['RSI'].iloc[idx] < self.rsi_oversold:
            return True
        
        # 7. Price becomes oversold
        if self._is_oversold(idx):
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
        if idx < max(self.bb_period, self.zscore_period, 20):
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            current_price = self.data['Close'].iloc[idx]
            
            if not self.bought:
                # Look for oversold conditions (buy signal)
                if (self._is_oversold(idx) and 
                    self._has_volume_confirmation(idx) and
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
