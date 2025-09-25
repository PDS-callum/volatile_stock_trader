# Strategy Optimization System

The stock bot now includes a powerful strategy optimization system that uses genetic algorithms and other optimization methods to automatically find the best parameters for trading strategies.

## Features

- **Multiple Optimization Methods**: Genetic Algorithm, Differential Evolution, and Gradient-based optimization
- **Multi-Objective Optimization**: Balances profit maximization with consistency (Sharpe ratio, win rate)
- **Parameter Range Definition**: Pre-defined parameter ranges for all built-in strategies
- **Multi-Symbol Testing**: Optimizes across multiple symbols for robustness
- **Comprehensive Metrics**: Tracks fitness, generation progress, and parameter evolution

## Usage

### Basic Optimization

```bash
# Optimize default strategy with genetic algorithm
python -m src.stock_bot optimize-strategy configs/config_optimization.yaml --strategy default

# Optimize with custom parameters
python -m src.stock_bot optimize-strategy configs/config_optimization.yaml \
  --strategy trend_following \
  --generations 20 \
  --population-size 30 \
  --consistency-weight 0.4 \
  --profit-weight 0.6
```

### Advanced Optimization

```bash
# Use differential evolution for faster convergence
python -m src.stock_bot optimize-strategy configs/config_optimization.yaml \
  --strategy macd_crossover \
  --optimization-method differential_evolution \
  --max-iterations 50

# Use gradient-based optimization
python -m src.stock_bot optimize-strategy configs/config_optimization.yaml \
  --strategy mean_reversion \
  --optimization-method minimize \
  --max-iterations 30
```

## CLI Options

- `--strategy`: Strategy to optimize (default: first strategy in config)
- `--generations`: Number of genetic algorithm generations (default: 50)
- `--population-size`: Population size for genetic algorithm (default: 20)
- `--mutation-rate`: Mutation rate for genetic algorithm (default: 0.1)
- `--crossover-rate`: Crossover rate for genetic algorithm (default: 0.7)
- `--max-iterations`: Maximum iterations for other methods (default: 100)
- `--optimization-method`: Method to use (genetic, differential_evolution, minimize)
- `--consistency-weight`: Weight for consistency in fitness (0-1, default: 0.3)
- `--profit-weight`: Weight for profit in fitness (0-1, default: 0.7)
- `--dashboard`: Show optimization dashboard (coming soon)

## Optimization Methods

### 1. Genetic Algorithm
- **Best for**: Complex parameter spaces, multiple local optima
- **Parameters**: Generations, population size, mutation rate, crossover rate
- **Advantages**: Explores parameter space thoroughly, handles non-linear relationships
- **Use when**: You have time for thorough exploration

### 2. Differential Evolution
- **Best for**: Continuous parameter spaces, faster convergence
- **Parameters**: Max iterations
- **Advantages**: Fast, good for continuous parameters
- **Use when**: You need quick results with continuous parameters

### 3. Gradient-based (minimize)
- **Best for**: Smooth parameter spaces, local optimization
- **Parameters**: Max iterations
- **Advantages**: Very fast convergence near optimum
- **Use when**: You have a good starting point and smooth parameter space

## Fitness Function

The optimization system uses a multi-objective fitness function:

```
fitness = profit_weight × average_return + consistency_weight × consistency_score
```

Where:
- `average_return`: Mean return across all symbols
- `consistency_score`: Sharpe ratio × win rate
- `profit_weight`: Weight for profit (default: 0.7)
- `consistency_weight`: Weight for consistency (default: 0.3)

## Supported Strategies

All built-in strategies support optimization with pre-defined parameter ranges:

- **default**: MACD, RSI, ROC, EMA parameters
- **trend_following**: Moving averages, RSI, volume, trend strength
- **mean_reversion**: Bollinger Bands, RSI, Z-score, Williams %R, Stochastic, CCI
- **macd_crossover**: MACD parameters, RSI, trend filter, profit/loss targets
- **macd_divergence**: MACD parameters, divergence detection, profit/loss targets
- **macd_histogram**: MACD parameters, histogram analysis, momentum thresholds

## Example Configurations

### Simple Optimization
```yaml
symbols:
  - AAPL
  - MSFT

data:
  period: "3mo"
  interval: "1d"

strategies:
  builtin:
    - default
```

### Advanced Multi-Strategy Optimization
```yaml
symbols:
  - AAPL
  - MSFT
  - GOOGL
  - TSLA
  - AMZN

data:
  period: "6mo"
  interval: "1d"

strategies:
  builtin:
    - default
    - trend_following
    - mean_reversion
    - macd_crossover

output:
  save_results: true
  results_file: "optimization_results.json"
```

## Results Interpretation

The optimization system provides:

1. **Best Parameters**: Optimal parameter values found
2. **Best Fitness**: Highest fitness score achieved
3. **Optimization Progress**: Generation-by-generation improvement
4. **Method Details**: Information about the optimization method used

### Example Output
```
🎯 OPTIMIZATION RESULTS for default
📊 Method: genetic_algorithm
🏆 Best fitness: 4.2956

🔧 OPTIMAL PARAMETERS:
   macd_fast: 10
   macd_slow: 28
   macd_signal: 15
   rsi_period: 15
   roc_period: 10
   ema_mom_period: 29

📈 OPTIMIZATION PROGRESS:
   Total generations: 20
   Population size: 30
   Final average fitness: 4.2133
   Final best fitness: 4.2956
```

## Tips for Effective Optimization

1. **Use Multiple Symbols**: Test across different market conditions
2. **Longer Time Periods**: Use 6+ months of data for robust results
3. **Balance Weights**: Adjust consistency vs profit weights based on your risk tolerance
4. **Start Simple**: Begin with genetic algorithm, then try other methods
5. **Validate Results**: Test optimized parameters on out-of-sample data

## Future Enhancements

- Real-time optimization dashboard
- Walk-forward optimization
- Portfolio-level optimization
- Custom fitness functions
- Parameter sensitivity analysis
