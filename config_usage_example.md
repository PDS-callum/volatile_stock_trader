# Configuration Usage Examples

## Overview
The CLI now supports passing configuration parameters directly to both backtester and live strategy commands using the `--config` option. This allows you to use optimized parameters from the optimization process.

## Usage

### 1. Using JSON String (Windows Command Prompt)
```bash
# Note: Use double quotes for JSON and escape inner quotes
python -m src.stock_bot run-back-strategy AAPL --strategy aggressive --config "{\"lookback\": 8, \"momentum_threshold\": 0.015, \"trade_cooldown\": 3}" --output-mode equity_only
```

### 2. Using JSON File (Recommended)
```bash
# Create a JSON file with your parameters
echo {"lookback": 8, "momentum_threshold": 0.015, "trade_cooldown": 3} > config.json

# Use the config file
python -m src.stock_bot run-back-strategy AAPL --strategy aggressive --config config.json --output-mode equity_only
```

### 3. Live Strategy with Config
```bash
python -m src.stock_bot run-live-strategy AAPL --strategy aggressive --config config.json
```

## Workflow: Optimization → Config → Backtest/Live

### Step 1: Optimize Strategy Parameters
```bash
python -m src.stock_bot optimize-strategy-parameters AggressiveStrategy AAPL --method brute --steps 3 --output-mode params_only
```

### Step 2: Save Optimized Parameters to JSON
```json
{
  "lookback": 8,
  "momentum_threshold": 0.015,
  "trade_cooldown": 3
}
```

### Step 3: Use Config for Backtesting
```bash
python -m src.stock_bot run-back-strategy AAPL --strategy aggressive --config optimized_params.json --output-mode full
```

### Step 4: Use Config for Live Trading
```bash
python -m src.stock_bot run-live-strategy AAPL --strategy aggressive --config optimized_params.json
```

## Supported Strategy Parameters

### DefaultStrategy
- `trade_cooldown`: Minimum bars between trades

### AggressiveStrategy  
- `lookback`: Number of bars to look back for momentum calculation
- `momentum_threshold`: Minimum momentum threshold for trades
- `trade_cooldown`: Minimum bars between trades

### TrendFollowingStrategy
- `short_ma_period`: Short moving average period
- `long_ma_period`: Long moving average period
- `trend_ma_period`: Trend moving average period
- `rsi_period`: RSI calculation period
- `rsi_oversold`: RSI oversold threshold
- `rsi_overbought`: RSI overbought threshold
- `volume_ma_period`: Volume moving average period
- `min_volume_ratio`: Minimum volume ratio
- `trend_strength_period`: Trend strength calculation period
- `min_trend_strength`: Minimum trend strength
- `trade_cooldown`: Minimum bars between trades

### MeanReversionStrategy
- `bb_period`: Bollinger Bands period
- `bb_std`: Bollinger Bands standard deviation
- `rsi_period`: RSI period
- `rsi_oversold`: RSI oversold threshold
- `rsi_overbought`: RSI overbought threshold
- `zscore_threshold`: Z-score threshold
- `trade_cooldown`: Minimum bars between trades

### MACD Strategies
- `macd_fast`: MACD fast period
- `macd_slow`: MACD slow period
- `macd_signal`: MACD signal period
- `trade_cooldown`: Minimum bars between trades

## Examples

### Example 1: Test Optimized AggressiveStrategy
```bash
# Optimize
python -m src.stock_bot optimize-strategy-parameters AggressiveStrategy AAPL --method brute --steps 3 --output-mode params_only

# Save results to config.json
# Then test
python -m src.stock_bot run-back-strategy AAPL --strategy aggressive --config config.json --output-mode full
```

### Example 2: Compare Different Configurations
```bash
# Test with default parameters
python -m src.stock_bot run-back-strategy AAPL --strategy aggressive --output-mode equity_only

# Test with optimized parameters
python -m src.stock_bot run-back-strategy AAPL --strategy aggressive --config optimized_config.json --output-mode equity_only
```

### Example 3: Live Trading with Optimized Parameters
```bash
python -m src.stock_bot run-live-strategy AAPL --strategy aggressive --config optimized_config.json
```

## Benefits

1. **Seamless Integration**: Use optimized parameters directly in backtesting and live trading
2. **Flexibility**: Support both JSON strings and files
3. **Strategy-Specific**: Each strategy can have its own parameter set
4. **Backward Compatibility**: Original CLI parameters still work when no config is provided
5. **Easy Testing**: Quickly test different parameter combinations
