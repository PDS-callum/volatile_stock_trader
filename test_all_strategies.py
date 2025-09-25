#!/usr/bin/env python3
"""
Script to test all strategies on multiple symbols.
This provides a quick way to test all built-in strategies.
"""

import subprocess
import sys
import os

def test_all_strategies():
    """Test all built-in strategies on multiple symbols."""
    
    # Symbols to test
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    # All built-in strategies
    strategies = [
        "default",
        "trend_following", 
        "simple_trend",
        "aggressive",
        "mean_reversion",
        "simple_mean_reversion",
        "macd_crossover",
        "macd_divergence",
        "macd_histogram"
    ]
    
    print("🚀 Testing all strategies on multiple symbols")
    print(f"📊 Symbols: {', '.join(symbols)}")
    print(f"🧪 Strategies: {len(strategies)} strategies")
    print("=" * 60)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n🔍 Testing {symbol}...")
        symbol_results = {}
        
        for strategy in strategies:
            try:
                # Run the strategy test
                cmd = [
                    sys.executable, "-m", "src.stock_bot", 
                    "run-back-strategy", symbol,
                    "--strategy", strategy,
                    "--period", "6mo",
                    "--interval", "1d",
                    "--output-mode", "equity_only"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    # Extract the return percentage from output
                    output = result.stdout.strip()
                    if "Final equity change" in output:
                        return_pct = float(output.split(": ")[1].replace("%", ""))
                        symbol_results[strategy] = return_pct
                        print(f"   ✅ {strategy}: {return_pct:.2f}%")
                    else:
                        symbol_results[strategy] = 0.0
                        print(f"   ⚠️  {strategy}: 0.00% (no data)")
                else:
                    symbol_results[strategy] = None
                    print(f"   ❌ {strategy}: Error")
                    
            except subprocess.TimeoutExpired:
                symbol_results[strategy] = None
                print(f"   ⏰ {strategy}: Timeout")
            except Exception as e:
                symbol_results[strategy] = None
                print(f"   ❌ {strategy}: {e}")
        
        results[symbol] = symbol_results
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY RESULTS")
    print(f"{'='*80}")
    
    # Create summary table
    print(f"{'Symbol':<10}", end="")
    for strategy in strategies:
        print(f"{strategy:<15}", end="")
    print()
    print("-" * (10 + 15 * len(strategies)))
    
    for symbol, symbol_results in results.items():
        print(f"{symbol:<10}", end="")
        for strategy in strategies:
            if strategy in symbol_results and symbol_results[strategy] is not None:
                print(f"{symbol_results[strategy]:>14.2f}%", end="")
            else:
                print(f"{'ERROR':<15}", end="")
        print()
    
    # Find best performing strategy
    print(f"\n🏆 BEST PERFORMING STRATEGIES:")
    strategy_totals = {}
    for strategy in strategies:
        total_return = 0
        count = 0
        for symbol_results in results.values():
            if strategy in symbol_results and symbol_results[strategy] is not None:
                total_return += symbol_results[strategy]
                count += 1
        if count > 0:
            strategy_totals[strategy] = total_return / count
    
    sorted_strategies = sorted(strategy_totals.items(), key=lambda x: x[1], reverse=True)
    for i, (strategy, avg_return) in enumerate(sorted_strategies[:5], 1):
        print(f"   {i}. {strategy}: {avg_return:.2f}% average return")

if __name__ == "__main__":
    test_all_strategies()
