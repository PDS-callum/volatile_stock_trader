#!/usr/bin/env python3
"""
Debug the backtester equity curve calculation
"""

import sys
sys.path.append('src')

from stock_bot.strategies.aggressive_strategy import AggressiveStrategy
from stock_bot.strategy_backtester import StrategyBacktester
import yfinance as yf

def main():
    # Get data
    ticker = yf.Ticker('AAPL')
    data = ticker.history(period='90d', interval='1d')
    print(f'Data points: {len(data)}')
    
    # Create strategy with very sensitive parameters
    params = {
        'lookback': 2,
        'momentum_threshold': 0.001,
        'trade_cooldown': 1,
        'rsi_oversold': 20,
        'rsi_overbought': 80
    }
    
    strategy = AggressiveStrategy(data, **params)
    backtester = StrategyBacktester(strategy)
    
    # Manually trace through the backtester logic with debug output
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    equity_curve = []
    entry_price = None
    cumulative_return = 0.0
    trade_returns = []
    trade_indices = []
    
    print(f'\nTracing backtester logic...')
    
    for i, close_price in enumerate(data['Close']):
        signal = backtester.strategy.check_signals(i)
        
        if signal == "BUY":
            buy_x.append(i)
            buy_y.append(close_price)
            entry_price = close_price
        elif signal == "SELL" and entry_price is not None:
            sell_x.append(i)
            sell_y.append(close_price)
            trade_return = (close_price - entry_price) / entry_price * 100
            cumulative_return += trade_return
            trade_returns.append(cumulative_return)
            trade_indices.append(i)
            entry_price = None
        
        # Update equity curve
        if entry_price is not None:
            current_return = (close_price - entry_price) / entry_price * 100
            equity_curve.append(cumulative_return + current_return)
        else:
            equity_curve.append(cumulative_return)
    
    print(f'Manual calculation:')
    print(f'  Final cumulative return: {cumulative_return:.2f}%')
    print(f'  Final equity curve value: {equity_curve[-1] if equity_curve else 0.0:.2f}%')
    print(f'  Equity curve length: {len(equity_curve)}')
    
    # Now test the actual backtester
    print(f'\nTesting actual backtester with final_equity_only=False...')
    final_equity = backtester.run(final_equity_only=False, plot=False)
    print(f'Actual backtester final equity: {final_equity:.2f}%')

if __name__ == "__main__":
    main()
