#!/usr/bin/env python3
"""
Debug the equity curve calculation
"""

import sys
sys.path.append('src')

from stock_bot.strategies.aggressive_strategy import AggressiveStrategy
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
    
    # Manually simulate the backtester logic
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    equity_curve = []
    entry_price = None
    cumulative_return = 0.0
    trade_returns = []
    trade_indices = []
    
    for i, close_price in enumerate(data['Close']):
        signal = strategy.check_signals(i)
        
        if signal == "BUY" and not strategy.bought:
            buy_x.append(i)
            buy_y.append(close_price)
            entry_price = close_price
            strategy.bought = True
            print(f"BUY at index {i}, price: {close_price:.2f}")
        elif signal == "SELL" and strategy.bought and entry_price is not None:
            sell_x.append(i)
            sell_y.append(close_price)
            trade_return = (close_price - entry_price) / entry_price * 100
            cumulative_return += trade_return
            trade_returns.append(cumulative_return)
            trade_indices.append(i)
            print(f"SELL at index {i}, price: {close_price:.2f}, trade return: {trade_return:.2f}%, cumulative: {cumulative_return:.2f}%")
            entry_price = None
            strategy.bought = False
        
        # Update equity curve
        if strategy.bought and entry_price is not None:
            current_return = (close_price - entry_price) / entry_price * 100
            equity_curve.append(cumulative_return + current_return)
        else:
            equity_curve.append(cumulative_return)
    
    print(f'\nEquity curve length: {len(equity_curve)}')
    print(f'Final cumulative return: {cumulative_return:.2f}%')
    print(f'Final equity curve value: {equity_curve[-1] if equity_curve else 0.0:.2f}%')
    print(f'First 10 equity curve values: {equity_curve[:10]}')
    print(f'Last 10 equity curve values: {equity_curve[-10:]}')

if __name__ == "__main__":
    main()
