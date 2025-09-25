# Custom Strategy Examples

This directory contains examples of custom trading strategies that can be used with the stock bot.

## How to Use Custom Strategies

### 1. Using Built-in Strategies

The stock bot comes with a default strategy that can be used without any additional files:

```bash
# Use the default strategy
python -m src.stock_bot run-back-strategy AAPL

# Use the default strategy explicitly
python -m src.stock_bot run-back-strategy AAPL --strategy default
```

### 2. Using Custom Strategy Files

You can create your own strategy by creating a Python file with a class that inherits from `BaseStrategy`:

```bash
# Use a custom strategy file
python -m src.stock_bot run-back-strategy AAPL --strategy examples/simple_ma_strategy.py --strategy-class SimpleMAStrategy
```

### 3. Strategy File Requirements

Your custom strategy file must:

1. Import `BaseStrategy` from `src.stock_bot.strategies.base_strategy`
2. Create a class that inherits from `BaseStrategy`
3. Implement the `check_signals(self, idx: int) -> Optional[str]` method
4. Optionally override `_compute_indicators()` to add custom indicators

### 4. Example Strategy Structure

```python
from src.stock_bot.strategies.base_strategy import BaseStrategy
from typing import Optional

class MyCustomStrategy(BaseStrategy):
    def __init__(self, data, **params):
        super().__init__(data, **params)
        # Add any custom initialization
    
    def _compute_indicators(self):
        # Override to add custom indicators
        super()._compute_indicators()  # Call parent to get default indicators
        # Add your custom indicators here
    
    def check_signals(self, idx: int) -> Optional[str]:
        # Your signal logic here
        # Return "BUY", "SELL", or None
        pass
```

## Available Examples

- `simple_ma_strategy.py` - A simple moving average crossover strategy

## Strategy Parameters

You can pass custom parameters to your strategy using the CLI options:

```bash
python -m src.stock_bot run-back-strategy AAPL \
    --strategy examples/simple_ma_strategy.py \
    --strategy-class SimpleMAStrategy \
    --sma-short 5 \
    --sma-long 20
```

The parameters will be passed to your strategy's `__init__` method as keyword arguments.
