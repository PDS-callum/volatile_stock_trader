# Trading Strategies

This directory contains various trading strategies for the stock bot. Each strategy inherits from `BaseStrategy` and implements the `check_signals` method.

## Available Strategies

### 1. Default Strategy (`default`)
The original strategy with MACD, RSI, ROC, and ATR indicators. Uses conservative trading rules with reduced frequency.

**Key Features:**
- MACD crossover signals
- RSI momentum confirmation
- Rate of Change (ROC) filtering
- ATR volatility filtering
- Trade cooldown periods
- Trailing stops and stop losses

### 2. Trend Following Strategy (`trend_following`)
A comprehensive trend-following strategy that identifies and rides trends using multiple technical indicators.

**Key Features:**
- Golden Cross/Death Cross detection (SMA crossovers)
- Multiple moving average alignment (10, 50, 200-day)
- RSI momentum confirmation
- Volume confirmation
- Trend strength analysis
- MACD momentum confirmation
- Bollinger Bands for volatility
- ATR for position sizing
- Trailing stops and stop losses

**Parameters:**
- `short_ma_period`: Short moving average period (default: 20)
- `long_ma_period`: Long moving average period (default: 50)
- `trend_ma_period`: Trend moving average period (default: 200)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)
- `volume_ma_period`: Volume moving average period (default: 20)
- `min_volume_ratio`: Minimum volume ratio for confirmation (default: 1.2)
- `trend_strength_period`: Period for trend strength calculation (default: 10)
- `min_trend_strength`: Minimum trend strength threshold (default: 0.02)
- `trade_cooldown`: Minimum bars between trades (default: 5)

### 3. Simple Trend Strategy (`simple_trend`)
A simplified version of the trend-following strategy with basic moving average crossovers.

**Key Features:**
- Simple SMA crossovers (10/30-day)
- RSI confirmation
- Trend strength analysis
- Trailing stops

**Parameters:**
- `short_ma_period`: Short moving average period (default: 10)
- `long_ma_period`: Long moving average period (default: 30)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)
- `min_trend_strength`: Minimum trend strength threshold (default: 0.01)
- `trade_cooldown`: Minimum bars between trades (default: 3)

### 4. Aggressive Strategy (`aggressive`)
A simple momentum-based strategy for testing purposes.

**Key Features:**
- Price momentum signals
- Minimal conditions
- High frequency trading

**Parameters:**
- `lookback`: Momentum calculation period (default: 5)
- `momentum_threshold`: Momentum threshold (default: 0.01)
- `trade_cooldown`: Minimum bars between trades (default: 2)

### 5. Mean Reversion Strategy (`mean_reversion`)
A comprehensive mean reversion strategy that identifies overbought/oversold conditions.

**Key Features:**
- Bollinger Bands for extreme price detection
- Z-score analysis for statistical significance
- Multiple momentum oscillators (RSI, Williams %R, Stochastic, CCI)
- Volume confirmation
- Multiple timeframe mean reversion analysis
- Advanced risk management

**Parameters:**
- `bb_period`: Bollinger Bands period (default: 20)
- `bb_std`: Bollinger Bands standard deviation (default: 2.0)
- `zscore_period`: Z-score calculation period (default: 20)
- `zscore_threshold`: Z-score threshold (default: 2.0)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)
- `volume_ma_period`: Volume moving average period (default: 20)
- `min_volume_ratio`: Minimum volume ratio (default: 0.8)
- `max_volume_ratio`: Maximum volume ratio (default: 2.0)
- `mean_reversion_period`: Mean reversion analysis period (default: 10)
- `min_reversion_strength`: Minimum reversion strength (default: 0.01)
- `max_holding_period`: Maximum holding period (default: 20)
- `profit_target_pct`: Profit target percentage (default: 0.02)
- `stop_loss_pct`: Stop loss percentage (default: 0.03)
- `trade_cooldown`: Minimum bars between trades (default: 3)

### 6. Simple Mean Reversion Strategy (`simple_mean_reversion`)
A simplified mean reversion strategy focusing on core concepts.

**Key Features:**
- Bollinger Bands for extreme price detection
- RSI for momentum confirmation
- Simple moving average reversion
- Basic risk management

**Parameters:**
- `bb_period`: Bollinger Bands period (default: 20)
- `bb_std`: Bollinger Bands standard deviation (default: 2.0)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)
- `sma_period`: Simple moving average period (default: 20)
- `profit_target_pct`: Profit target percentage (default: 0.015)
- `stop_loss_pct`: Stop loss percentage (default: 0.02)
- `max_holding_period`: Maximum holding period (default: 15)
- `trade_cooldown`: Minimum bars between trades (default: 2)

### 7. MACD Crossover Strategy (`macd_crossover`)
A strategy based on MACD line and signal line crossovers.

**Key Features:**
- MACD line crosses above signal line → BUY
- MACD line crosses below signal line → SELL
- Trend filter confirmation
- Volume and momentum confirmation
- Works well in trending markets and higher timeframes

**Parameters:**
- `macd_fast`: MACD fast period (default: 12)
- `macd_slow`: MACD slow period (default: 26)
- `macd_signal`: MACD signal period (default: 9)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)
- `trend_filter_period`: Trend filter period (default: 50)
- `min_macd_strength`: Minimum MACD strength (default: 0.001)
- `max_holding_period`: Maximum holding period (default: 30)
- `profit_target_pct`: Profit target percentage (default: 0.03)
- `stop_loss_pct`: Stop loss percentage (default: 0.02)
- `trade_cooldown`: Minimum bars between trades (default: 3)

### 8. MACD Divergence Strategy (`macd_divergence`)
A strategy that identifies divergences between price and MACD movement.

**Key Features:**
- Bullish divergence: Price lower low, MACD higher low → BUY
- Bearish divergence: Price higher high, MACD lower high → SELL
- Anticipates trend changes before they occur
- More effective in overbought/oversold markets
- Uses multiple confirmation signals

**Parameters:**
- `macd_fast`: MACD fast period (default: 12)
- `macd_slow`: MACD slow period (default: 26)
- `macd_signal`: MACD signal period (default: 9)
- `divergence_lookback`: Divergence analysis period (default: 20)
- `min_divergence_strength`: Minimum divergence strength (default: 0.02)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)
- `max_holding_period`: Maximum holding period (default: 25)
- `profit_target_pct`: Profit target percentage (default: 0.04)
- `stop_loss_pct`: Stop loss percentage (default: 0.025)
- `trade_cooldown`: Minimum bars between trades (default: 5)

### 9. MACD Histogram Strategy (`macd_histogram`)
A strategy based on MACD histogram characteristics and momentum.

**Key Features:**
- Positive histogram: MACD above signal line, bullish momentum
- Negative histogram: MACD below signal line, bearish momentum
- Rising histogram: momentum increasing
- Falling histogram: momentum decreasing
- More useful in volatile markets and shorter timeframes

**Parameters:**
- `macd_fast`: MACD fast period (default: 12)
- `macd_slow`: MACD slow period (default: 26)
- `macd_signal`: MACD signal period (default: 9)
- `histogram_smoothing`: Histogram smoothing period (default: 3)
- `min_histogram_strength`: Minimum histogram strength (default: 0.0005)
- `momentum_threshold`: Momentum threshold (default: 0.001)
- `rsi_period`: RSI calculation period (default: 14)
- `rsi_oversold`: RSI oversold threshold (default: 30)
- `rsi_overbought`: RSI overbought threshold (default: 70)
- `max_holding_period`: Maximum holding period (default: 20)
- `profit_target_pct`: Profit target percentage (default: 0.025)
- `stop_loss_pct`: Stop loss percentage (default: 0.015)
- `trade_cooldown`: Minimum bars between trades (default: 2)

## Usage Examples

### Using Built-in Strategies

```bash
# Use default strategy
python -m src.stock_bot run-back-strategy AAPL --strategy default

# Use trend following strategy
python -m src.stock_bot run-back-strategy AAPL --strategy trend_following

# Use simple trend strategy
python -m src.stock_bot run-back-strategy AAPL --strategy simple_trend

# Use aggressive strategy
python -m src.stock_bot run-back-strategy AAPL --strategy aggressive

# Use mean reversion strategy
python -m src.stock_bot run-back-strategy AAPL --strategy mean_reversion

# Use simple mean reversion strategy
python -m src.stock_bot run-back-strategy AAPL --strategy simple_mean_reversion

# Use MACD crossover strategy
python -m src.stock_bot run-back-strategy AAPL --strategy macd_crossover

# Use MACD divergence strategy
python -m src.stock_bot run-back-strategy AAPL --strategy macd_divergence

# Use MACD histogram strategy
python -m src.stock_bot run-back-strategy AAPL --strategy macd_histogram
```

### Customizing Strategy Parameters

```bash
# Customize trend following strategy parameters
python -m src.stock_bot run-back-strategy AAPL --strategy trend_following \
    --short-ma-period 15 \
    --long-ma-period 40 \
    --trend-ma-period 150 \
    --rsi-period 21 \
    --min-trend-strength 0.015
```

### Using Custom Strategy Files

```bash
# Use a custom strategy file
python -m src.stock_bot run-back-strategy AAPL \
    --strategy examples/simple_ma_strategy.py \
    --strategy-class SimpleMAStrategy
```

## Strategy Performance Notes

- **Default Strategy**: Conservative, low frequency, good for stable markets
- **Trend Following Strategy**: Comprehensive, good for trending markets, may be conservative
- **Simple Trend Strategy**: Balanced approach, good for most market conditions
- **Aggressive Strategy**: High frequency, good for testing and volatile markets
- **Mean Reversion Strategy**: Comprehensive, good for ranging/sideways markets, high win rate potential
- **Simple Mean Reversion Strategy**: Balanced mean reversion, good for most market conditions
- **MACD Crossover Strategy**: Simple and reliable, good for trending markets and higher timeframes
- **MACD Divergence Strategy**: Anticipates reversals, good for overbought/oversold markets
- **MACD Histogram Strategy**: Detailed momentum analysis, good for volatile markets and shorter timeframes

## Creating Custom Strategies

To create a custom strategy:

1. Create a new file in the strategies directory
2. Import `BaseStrategy` from `.base_strategy`
3. Create a class that inherits from `BaseStrategy`
4. Implement the `check_signals(self, idx: int) -> Optional[str]` method
5. Optionally override `_compute_indicators()` for custom indicators

Example:

```python
from .base_strategy import BaseStrategy
from typing import Optional

class MyCustomStrategy(BaseStrategy):
    def __init__(self, data, **params):
        super().__init__(data, **params)
        # Custom initialization
    
    def _compute_indicators(self):
        super()._compute_indicators()  # Get default indicators
        # Add custom indicators here
    
    def check_signals(self, idx: int) -> Optional[str]:
        # Your signal logic here
        # Return "BUY", "SELL", or None
        pass
```
