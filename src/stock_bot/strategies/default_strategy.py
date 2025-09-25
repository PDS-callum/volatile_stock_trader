import pandas as pd
import numpy as np
import logging
from typing import Optional
from .base_strategy import BaseStrategy

class DefaultStrategy(BaseStrategy):
    """
    Improved strategy with reduced trading frequency and better signal filtering.
    
    This is the default strategy that was previously in the main strategy.py file.
    It uses MACD, RSI, ROC, and ATR indicators with conservative trading rules.
    """
    
    def __init__(self, data: pd.DataFrame, **params):
        super().__init__(data, **params)
        
        # Enhanced parameter support with validation - more responsive defaults
        self._last_trade_idx = -1
        self._trade_cooldown = max(1, int(params.get('trade_cooldown', 5)))  # Reduced from 20 to 5
        
        # Risk management parameters
        self._max_position_size = params.get('max_position_size', 1.0)
        self._risk_per_trade = params.get('risk_per_trade', 0.02)  # 2% risk per trade
        self._max_drawdown = params.get('max_drawdown', 0.10)  # 10% max drawdown
        
        # Performance tracking
        self._entry_price = None
        self._peak_price = None
        self._trough_price = None
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._max_equity = 10000  # Starting equity
        self._current_equity = 10000
        self._max_equity_peak = 10000
        
    def _update_performance_metrics(self, current_price: float, is_exit: bool = False):
        """Update performance tracking metrics."""
        if self._entry_price is not None:
            if is_exit:
                trade_return = (current_price - self._entry_price) / self._entry_price
                self._current_equity *= (1 + trade_return)
                self._total_trades += 1
                
                if trade_return > 0:
                    self._winning_trades += 1
                else:
                    self._losing_trades += 1
                
                # Update max equity peak
                if self._current_equity > self._max_equity_peak:
                    self._max_equity_peak = self._current_equity
            else:
                # Update peak/trough for current position
                if self._peak_price is None or current_price > self._peak_price:
                    self._peak_price = current_price
                if self._trough_price is None or current_price < self._trough_price:
                    self._trough_price = current_price
    
    def _check_risk_limits(self, idx: int) -> bool:
        """Check if we should stop trading due to risk limits."""
        # Check maximum drawdown
        current_drawdown = (self._max_equity_peak - self._current_equity) / self._max_equity_peak
        if current_drawdown > self._max_drawdown:
            return False
        
        # Check if we have enough data for reliable signals
        if idx < 20:  # Reduced from 50 to 20
            return False
            
        return True
    
    def _get_adaptive_thresholds(self, idx: int) -> dict:
        """Get adaptive thresholds based on market conditions."""
        # Get market volatility
        volatility = self.data['Volatility'].iloc[idx] if 'Volatility' in self.data.columns else 0.01
        market_regime = self.data['Market_Regime'].iloc[idx] if 'Market_Regime' in self.data.columns else 'Neutral'
        
        # Adaptive thresholds based on volatility and market regime - more sensitive
        base_roc_thresh = 0.003  # Reduced from 0.012 to 0.003
        base_rsi_oversold = 35   # Increased from 25 to 35
        base_rsi_overbought = 65 # Decreased from 75 to 65
        
        # Adjust for volatility
        volatility_multiplier = 1 + (volatility - 0.01) * 10  # Scale with volatility
        
        # Adjust for market regime
        if market_regime == 'Uptrend':
            roc_thresh = base_roc_thresh * 0.8  # More sensitive in uptrends
            rsi_oversold = base_rsi_oversold + 5
            rsi_overbought = base_rsi_overbought + 5
        elif market_regime == 'Downtrend':
            roc_thresh = base_roc_thresh * 1.5  # Less sensitive in downtrends
            rsi_oversold = base_rsi_oversold - 5
            rsi_overbought = base_rsi_overbought - 5
        else:
            roc_thresh = base_roc_thresh
            rsi_oversold = base_rsi_oversold
            rsi_overbought = base_rsi_overbought
        
        return {
            'roc_thresh': roc_thresh * volatility_multiplier,
            'rsi_oversold': max(10, min(40, rsi_oversold)),
            'rsi_overbought': max(60, min(90, rsi_overbought)),
            'volatility_multiplier': volatility_multiplier
        }
    
    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals with enhanced risk management and adaptive parameters.
        
        Args:
            idx: Index in the data to check for signals
            
        Returns:
            "BUY", "SELL", or None
        """
        # Enhanced validation and risk checks
        if not self._check_risk_limits(idx):
            return None
            
        try:
            close_now = self.data['Close'].iloc[idx]
            
            # Update performance metrics
            self._update_performance_metrics(close_now)
            
            # Get adaptive thresholds
            thresholds = self._get_adaptive_thresholds(idx)
            
            # Get indicators with error handling
            roc_now = self.data.get('ROC_5', self.data.get('ROC', pd.Series([0]))).iloc[idx]
            atr_now = self.data.get('ATR', pd.Series([0])).iloc[idx]
            rsi_now = self.data.get('RSI', pd.Series([50])).iloc[idx]
            macd_now = self.data.get('MACD', pd.Series([0])).iloc[idx]
            macd_signal_now = self.data.get('MACD_signal', pd.Series([0])).iloc[idx]
            volume_ratio = self.data.get('Volume_Ratio', pd.Series([1])).iloc[idx]
            
            # Enhanced volatility threshold
            atr_vol_thresh = self.data['ATR'].rolling(window=30).mean().iloc[idx] if idx >= 30 else atr_now
            
            # Trade cooldown - prevent frequent entries/exits
            if self._last_trade_idx > 0 and (idx - self._last_trade_idx) < self._trade_cooldown:
                return None
            
            if not self.bought:
                # Enhanced multi-condition buy signal
                rsi_oversold_recovery = thresholds['rsi_oversold'] < rsi_now < thresholds['rsi_overbought']
                macd_bullish = macd_now > macd_signal_now
                strong_momentum = roc_now > thresholds['roc_thresh']
                sufficient_volatility = atr_now > atr_vol_thresh * 1.2
                
                # Trend confirmation - price above recent average
                recent_avg = self.data['Close'].iloc[max(0, idx-10):idx].mean()
                uptrend = close_now > recent_avg * 1.005
                
                # Volume confirmation
                volume_confirmation = volume_ratio > 0.8
                
                # Market regime confirmation
                market_regime = self.data.get('Market_Regime', pd.Series(['Neutral'])).iloc[idx]
                regime_confirmation = market_regime in ['Uptrend', 'Neutral']
                
                # Position sizing based on risk
                position_size = min(self._max_position_size, self._risk_per_trade / (atr_now / close_now))
                
                if all([rsi_oversold_recovery, macd_bullish, strong_momentum, sufficient_volatility, 
                       uptrend, volume_confirmation, regime_confirmation]) and position_size > 0.1:
                    self._entry_price = close_now
                    self._peak_price = close_now
                    self._trough_price = close_now
                    self._last_trade_idx = idx
                    return "BUY"
                    
            else:
                # Update performance metrics
                self._update_performance_metrics(close_now)
                
                # Enhanced exit conditions with adaptive thresholds
                trailing_stop_pct = 0.025 * thresholds['volatility_multiplier']
                stop_loss_pct = 0.015 * thresholds['volatility_multiplier']
                
                # Additional exit conditions
                rsi_overbought = rsi_now > thresholds['rsi_overbought']
                macd_bearish = macd_now < macd_signal_now
                
                # Profit taking at reasonable levels
                profit_target = self._entry_price and close_now > self._entry_price * 1.03
                
                # Exit conditions (any can trigger)
                trailing_stop_hit = close_now < self._peak_price * (1 - trailing_stop_pct)
                stop_loss_hit = self._entry_price and close_now < self._entry_price * (1 - stop_loss_pct)
                
                # Time-based exit (prevent holding too long)
                holding_period = idx - self._last_trade_idx
                time_exit = holding_period > 50  # Exit after 50 bars
                
                if (trailing_stop_hit or stop_loss_hit or rsi_overbought or 
                    (macd_bearish and profit_target) or time_exit):
                    self._update_performance_metrics(close_now, is_exit=True)
                    self._last_trade_idx = idx
                    return "SELL"
                    
        except Exception as e:
            logging.error(f"Error checking signals at idx {idx}: {e}")
        return None
