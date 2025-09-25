# Configuration File System

The stock bot now supports configuration files for testing multiple symbols and strategies with different parameters. This makes it easy to run comprehensive tests without having to specify all parameters manually.

## Quick Start

```bash
# Test with a simple config
python -m src.stock_bot test-config config_simple.yaml

# Override symbol from config
python -m src.stock_bot test-config config_simple.yaml --symbol AAPL

# Override strategy from config
python -m src.stock_bot test-config config_simple.yaml --symbol AAPL --strategy default
```

## Configuration File Format

Configuration files use YAML format and support the following sections:

### Required Sections

#### `symbols`
List of stock symbols to test:
```yaml
symbols:
  - AAPL
  - MSFT
  - GOOGL
  - TSLA
  - NVDA
```

#### `data`
Data configuration:
```yaml
data:
  period: "1y"          # Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
  interval: "1d"        # Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
```

#### `strategies`
Strategy configuration:
```yaml
strategies:
  # Built-in strategies
  builtin:
    - default
    - trend_following
    - simple_trend
    - aggressive
    - mean_reversion
    - simple_mean_reversion
    - macd_crossover
    - macd_divergence
    - macd_histogram
  
  # Custom strategy files (optional)
  custom:
    - path: "examples/simple_ma_strategy.py"
      class_name: "SimpleMAStrategy"
```

### Optional Sections

#### `strategy_params`
Strategy parameters (applied to all strategies):
```yaml
strategy_params:
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  rsi_period: 14
  roc_period: 5
  ema_mom_period: 20
```

#### `output`
Output configuration:
```yaml
output:
  mode: "comparison"        # equity_only, equity_curve, comparison
  use_notebook_plots: false
  dashboard_port: 8052
  save_results: true
  results_file: "strategy_comparison_results.json"
```

#### `test`
Test configuration:
```yaml
test:
  max_symbols: 10          # Maximum number of symbols to test (0 = no limit)
  parallel: false          # Whether to test symbols in parallel (future feature)
  save_results: true       # Whether to save results to file
```

## Example Configuration Files

### Simple Config (`config_simple.yaml`)
```yaml
symbols:
  - AAPL
  - TSLA

data:
  period: "30d"
  interval: "15m"

strategies:
  builtin:
    - default
    - trend_following
    - macd_crossover

output:
  mode: "comparison"
```

### Multi-Symbol Config (`config_multi_symbol.yaml`)
```yaml
symbols:
  - AAPL
  - MSFT
  - GOOGL
  - TSLA
  - NVDA
  - AMZN
  - META
  - NFLX

data:
  period: "6mo"
  interval: "1d"

strategies:
  builtin:
    - default
    - trend_following
    - mean_reversion
    - macd_crossover

strategy_params:
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  rsi_period: 14

output:
  mode: "comparison"
  save_results: true
  results_file: "multi_symbol_results.json"
```

### Comprehensive Config (`config_example.yaml`)
```yaml
symbols:
  - AAPL
  - MSFT
  - GOOGL
  - TSLA
  - NVDA

data:
  period: "1y"
  interval: "1d"

strategies:
  builtin:
    - default
    - trend_following
    - simple_trend
    - aggressive
    - mean_reversion
    - simple_mean_reversion
    - macd_crossover
    - macd_divergence
    - macd_histogram
  
  custom:
    - path: "examples/simple_ma_strategy.py"
      class_name: "SimpleMAStrategy"

strategy_params:
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  rsi_period: 14
  roc_period: 5
  ema_mom_period: 20

output:
  mode: "comparison"
  use_notebook_plots: false
  dashboard_port: 8052
  save_results: true
  results_file: "strategy_comparison_results.json"

test:
  max_symbols: 10
  parallel: false
  save_results: true
```

## CLI Commands

### Basic Usage
```bash
# Test with config file
python -m src.stock_bot test-config config_file.yaml

# Override symbol from config
python -m src.stock_bot test-config config_file.yaml --symbol AAPL

# Override strategy from config
python -m src.stock_bot test-config config_file.yaml --symbol AAPL --strategy default

# Override both symbol and strategy
python -m src.stock_bot test-config config_file.yaml --symbol AAPL --strategy default
```

### Other Commands
```bash
# Compare strategies on single symbol
python -m src.stock_bot compare-strategies AAPL --strategy-path builtin

# Run single strategy
python -m src.stock_bot run-back-strategy AAPL --strategy default
```

## Output

The config system provides:

1. **Progress Reporting**: Shows which symbol and strategy is being tested
2. **Results Summary**: Table showing returns for each symbol/strategy combination
3. **Error Handling**: Gracefully handles failed strategies and symbols
4. **Results Saving**: Optionally saves results to JSON file
5. **Override Support**: Can override symbols and strategies from command line

### Sample Output
```
✅ Loaded configuration from config_simple.yaml
📊 Testing 2 symbols with 3 built-in strategies
📅 Period: 30d, Interval: 15m

============================================================
🧪 Testing symbol 1/2: AAPL
============================================================

🔍 Testing default on AAPL
   ✅ default: 0.00% return

🔍 Testing trend_following on AAPL
   ✅ trend_following: 0.00% return

🔍 Testing macd_crossover on AAPL
   ✅ macd_crossover: 0.00% return

============================================================
🧪 Testing symbol 2/2: TSLA
============================================================

🔍 Testing default on TSLA
   ✅ default: 5.74% return

🔍 Testing trend_following on TSLA
   ✅ trend_following: 0.00% return

🔍 Testing macd_crossover on TSLA
   ✅ macd_crossover: 0.00% return

================================================================================
📊 CONFIG TEST SUMMARY
================================================================================
Symbol    default        macd_crossover trend_following
-------------------------------------------------------
AAPL                0.00%          0.00%          0.00%
TSLA                5.74%          0.00%          0.00%
```

## Benefits

1. **Reproducible Tests**: Config files ensure consistent testing parameters
2. **Batch Testing**: Test multiple symbols and strategies in one command
3. **Parameter Management**: Centralized parameter configuration
4. **Flexibility**: Override any parameter from command line
5. **Documentation**: Config files serve as documentation of test parameters
6. **Version Control**: Track changes to test configurations over time

## Tips

1. **Start Simple**: Use `config_simple.yaml` for quick tests
2. **Use Overrides**: Test specific symbols/strategies with `--symbol` and `--strategy`
3. **Save Results**: Enable `save_results: true` to save results for analysis
4. **Limit Symbols**: Use `max_symbols` to limit testing for quick iterations
5. **Custom Strategies**: Add your own strategy files in the `custom` section
