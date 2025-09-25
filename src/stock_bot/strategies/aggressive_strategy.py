"""
Aggressive Strategy

A very simple strategy that generates many signals for testing purposes.
Uses basic price momentum with minimal conditions.
"""

import pandas as pd
import numpy as np
from typing import Optional
from .base_strategy import BaseStrategy

class AggressiveStrategy(BaseStrategy):
    """
    Aggressive Strategy for testing.
    
    This strategy generates many buy/sell signals based on simple price momentum.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        # Enhanced strategy parameters with validation - more responsive defaults
        self.lookback = max(1, int(params.get('lookback', 3)))  # Reduced from 5 to 3
        self.momentum_threshold = max(0.001, params.get('momentum_threshold', 0.003))  # Reduced from 0.01 to 0.003
        self._last_trade_idx = -1
        self._trade_cooldown = max(1, int(params.get('trade_cooldown', 1)))  # Reduced from 2 to 1
        
        # Additional parameters for improved strategy
        self.volatility_threshold = params.get('volatility_threshold', 0.02)  # 2% volatility threshold
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.volume_threshold = params.get('volume_threshold', 0.8)  # Volume ratio threshold
        
        # Risk management
        self.max_position_size = params.get('max_position_size', 1.0)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.05)  # 5% stop loss
        self.take_profit_pct = params.get('take_profit_pct', 0.08)  # 8% take profit
        
        # Performance tracking
        self._entry_price = None
        self._peak_price = None
        self._total_trades = 0
        self._winning_trades = 0
        
        super().__init__(data, **params)
        
    def _compute_indicators(self):
        """Compute enhanced technical indicators for aggressive strategy."""
        # Call parent to get default indicators
        super()._compute_indicators()
        
        # Enhanced momentum indicators
        self.data['Momentum'] = self.data['Close'].pct_change(periods=self.lookback)
        self.data['Momentum_1'] = self.data['Close'].pct_change(periods=1)
        self.data['Momentum_3'] = self.data['Close'].pct_change(periods=3)
        self.data['Momentum_10'] = self.data['Close'].pct_change(periods=10)
        
        # Volatility-adjusted momentum
        volatility = self.data['Close'].rolling(window=20).std()
        self.data['Volatility_Adjusted_Momentum'] = self.data['Momentum'] / volatility
        
        # Price acceleration (second derivative)
        self.data['Price_Acceleration'] = self.data['Momentum'].diff()
        
        # Volume-weighted momentum
        if 'Volume' in self.data.columns:
            volume_weighted_price = (self.data['Close'] * self.data['Volume']).rolling(window=self.lookback).sum() / self.data['Volume'].rolling(window=self.lookback).sum()
            self.data['Volume_Weighted_Momentum'] = (self.data['Close'] - volume_weighted_price) / volume_weighted_price
        else:
            self.data['Volume_Weighted_Momentum'] = self.data['Momentum']
        
        # Trend strength
        self.data['Trend_Strength'] = (self.data['Close'] - self.data['SMA_20']) / self.data['SMA_20']
        
        # Market regime detection
        self.data['Market_Regime'] = 'Neutral'
        uptrend_mask = (self.data['Close'] > self.data['SMA_20']) & (self.data['SMA_20'] > self.data['SMA_50'])
        downtrend_mask = (self.data['Close'] < self.data['SMA_20']) & (self.data['SMA_20'] < self.data['SMA_50'])
        
        self.data.loc[uptrend_mask, 'Market_Regime'] = 'Uptrend'
        self.data.loc[downtrend_mask, 'Market_Regime'] = 'Downtrend'
        
    def _is_strong_momentum(self, idx: int) -> bool:
        """Check for strong momentum with multiple confirmations."""
        if idx < max(self.lookback + 1, 20):
            return False
        
        momentum = self.data['Momentum'].iloc[idx]
        momentum_1 = self.data['Momentum_1'].iloc[idx]
        momentum_3 = self.data['Momentum_3'].iloc[idx]
        volatility_adj_momentum = self.data['Volatility_Adjusted_Momentum'].iloc[idx]
        price_acceleration = self.data['Price_Acceleration'].iloc[idx]
        volume_weighted_momentum = self.data['Volume_Weighted_Momentum'].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        volume_ratio = self.data.get('Volume_Ratio', pd.Series([1])).iloc[idx]
        market_regime = self.data.get('Market_Regime', pd.Series(['Neutral'])).iloc[idx]
        
        # Multiple momentum confirmations - more lenient
        conditions = [
            momentum > self.momentum_threshold,
            momentum_1 > 0,  # Positive short-term momentum
            momentum_3 > self.momentum_threshold * 0.3,  # Reduced from 0.5 to 0.3
            volatility_adj_momentum > self.momentum_threshold * 0.3,  # Reduced from 0.5 to 0.3
            price_acceleration > 0,  # Accelerating price movement
            volume_weighted_momentum > self.momentum_threshold * 0.5,  # Reduced from 0.7 to 0.5
            rsi < self.rsi_overbought,  # Not overbought
            volume_ratio > self.volume_threshold * 0.7,  # Reduced volume requirement
            market_regime in ['Uptrend', 'Neutral']  # Favorable market regime
        ]
        
        return sum(conditions) >= 4  # Reduced from 6 to 4 out of 9 conditions
    
    def _is_weak_momentum(self, idx: int) -> bool:
        """Check for weak momentum or reversal signals."""
        if idx < max(self.lookback + 1, 20):
            return False
        
        momentum = self.data['Momentum'].iloc[idx]
        momentum_1 = self.data['Momentum_1'].iloc[idx]
        price_acceleration = self.data['Price_Acceleration'].iloc[idx]
        rsi = self.data['RSI'].iloc[idx]
        market_regime = self.data.get('Market_Regime', pd.Series(['Neutral'])).iloc[idx]
        
        # Weak momentum conditions
        conditions = [
            momentum < -self.momentum_threshold * 0.5,  # Negative momentum
            momentum_1 < 0,  # Negative short-term momentum
            price_acceleration < 0,  # Decelerating price movement
            rsi > self.rsi_overbought,  # Overbought
            market_regime == 'Downtrend'  # Unfavorable market regime
        ]
        
        return sum(conditions) >= 3  # At least 3 out of 5 conditions must be true
    
    def _should_exit_position(self, idx: int) -> bool:
        """Check if we should exit the current position."""
        if not self.bought or self._entry_price is None:
            return False
        
        current_price = self.data['Close'].iloc[idx]
        
        # Update peak price
        if self._peak_price is None or current_price > self._peak_price:
            self._peak_price = current_price
        
        # Exit conditions
        # 1. Take profit
        if current_price >= self._entry_price * (1 + self.take_profit_pct):
            return True
        
        # 2. Stop loss
        if current_price <= self._entry_price * (1 - self.stop_loss_pct):
            return True
        
        # 3. Weak momentum
        if self._is_weak_momentum(idx):
            return True
        
        # 4. Trailing stop (3% below peak)
        if current_price < self._peak_price * 0.97:
            return True
        
        return False
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals based on enhanced momentum analysis.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        if idx < self.lookback + 1:
            return None
        
        # Trade cooldown
        if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
            return None
        
        try:
            if not self.bought:
                # Buy signal: strong momentum with multiple confirmations
                if self._is_strong_momentum(idx):
                    self.bought = True
                    self._entry_price = self.data['Close'].iloc[idx]
                    self._peak_price = self._entry_price
                    self._last_trade_idx = idx
                    self._total_trades += 1
                    return "BUY"
            
            else:
                # Check for exit conditions
                if self._should_exit_position(idx):
                    self.bought = False
                    # Track performance
                    if self._entry_price:
                        trade_return = (self.data['Close'].iloc[idx] - self._entry_price) / self._entry_price
                        if trade_return > 0:
                            self._winning_trades += 1
                    
                    self._entry_price = None
                    self._peak_price = None
                    self._last_trade_idx = idx
                    return "SELL"
                    
        except Exception as e:
            print(f"Error checking signals at idx {idx}: {e}")
        
        return None
