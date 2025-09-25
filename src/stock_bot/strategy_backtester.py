
import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import Any, List, Optional, Dict, Tuple
from scipy.optimize import brute, differential_evolution
from .strategies.base_strategy import BaseStrategy
import yfinance as yf


class StrategyBacktester:
    """
    Backtests a supplied trading strategy object.
    """
    def __init__(self, strategy: BaseStrategy):
        if not isinstance(strategy, BaseStrategy):
            raise ValueError("strategy must be an instance of BaseStrategy.")
        self.strategy = strategy
        self.data = strategy.data
        self.signals: List[Any] = []

    def run(
        self,
        final_equity_only: bool = False,
        stop_loss_pct: float = 5,
        plot: bool = True,
        slider: bool = True
    ) -> float:
        """
        Run the backtest simulation.
        Args:
            final_equity_only (bool): If True, only return final equity.
            stop_loss_pct (float): Stop loss percentage.
            plot (bool): Whether to plot results.
            slider (bool): Whether to launch dashboard.
        Returns:
            float: Final equity percentage.
        """
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        equity_curve: List[float] = []
        entry_price: Optional[float] = None
        cumulative_return: float = 0.0
        trade_returns: List[float] = []
        trade_indices: List[int] = []
        for i, close_price in enumerate(self.data['Close']):
            signal = self.strategy.check_signals(i)
            # Stop loss
            if self.strategy.bought and entry_price is not None:
                stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                if close_price <= stop_loss_price:
                    sell_x.append(i)
                    sell_y.append(close_price)
                    trade_return = (close_price - entry_price) / entry_price * 100
                    cumulative_return += trade_return
                    trade_returns.append(cumulative_return)
                    trade_indices.append(i)
                    entry_price = None
                    self.strategy.bought = False
                    continue
            if signal == "BUY" and not self.strategy.bought:
                buy_x.append(i)
                buy_y.append(close_price)
                entry_price = close_price
                self.strategy.bought = True
            elif signal == "SELL" and self.strategy.bought and entry_price is not None:
                sell_x.append(i)
                sell_y.append(close_price)
                trade_return = (close_price - entry_price) / entry_price * 100
                cumulative_return += trade_return
                trade_returns.append(cumulative_return)
                trade_indices.append(i)
                entry_price = None
                self.strategy.bought = False
            # Update running equity curve
            if self.strategy.bought and entry_price is not None:
                current_return = (close_price - entry_price) / entry_price * 100
                equity_curve.append(cumulative_return + current_return)
            else:
                equity_curve.append(cumulative_return)

        # Dashboard plotting
        if plot and slider:
            try:
                import dash
                from dash import dcc, html
                import plotly.graph_objs as go
                import threading
                def launch_dashboard():
                    x_vals = list(range(len(self.data)))
                    buy_points = [(i, buy_y[j]) for j, i in enumerate(buy_x)]
                    sell_points = [(i, sell_y[j]) for j, i in enumerate(sell_x)]
                    from dash.dependencies import Input, Output, State
                    app = dash.Dash(__name__)
                    equity_curve_fig = {
                        'data': [
                            go.Scatter(x=x_vals, y=self.data['Close'], mode='lines', name='Close Price', line=dict(color='blue')),
                            go.Scatter(x=[x for x, _ in buy_points], y=[y for _, y in buy_points], mode='markers', name='BUY', marker=dict(color='green', symbol='circle', size=10)),
                            go.Scatter(x=[x for x, _ in sell_points], y=[y for _, y in sell_points], mode='markers', name='SELL', marker=dict(color='red', symbol='x', size=10)),
                            go.Scatter(x=x_vals, y=self.data['EMA_fast'], mode='lines', name='EMA Fast', line=dict(dash='dash', color='orange')),
                            go.Scatter(x=x_vals, y=self.data['EMA_slow'], mode='lines', name='EMA Slow', line=dict(dash='dash', color='purple')),
                        ],
                        'layout': go.Layout(title='Equity Curve & Signals', xaxis={'title': 'Index'}, yaxis={'title': 'Price'})
                    }
                    rsi_fig = {
                        'data': [
                            go.Scatter(x=x_vals, y=self.data['RSI'], mode='lines', name='RSI', line=dict(color='blue')),
                        ],
                        'layout': go.Layout(title='RSI', xaxis={'title': 'Index'}, yaxis={'title': 'RSI'})
                    }
                    equity_fig = {
                        'data': [
                            go.Scatter(x=x_vals, y=equity_curve, mode='lines', name='Cumulative % Gain/Loss', line=dict(color='black')),
                            go.Scatter(x=trade_indices, y=trade_returns, mode='markers', name='Trade Close', marker=dict(color='magenta', symbol='x', size=10)),
                        ],
                        'layout': go.Layout(title='Cumulative % Gain/Loss', xaxis={'title': 'Minute Index'}, yaxis={'title': 'Cumulative %'})
                    }
                    app.layout = html.Div([
                        html.H1('Backtest Analysis Dashboard'),
                        dcc.Graph(id='equity-curve', figure=equity_curve_fig),
                        dcc.Graph(id='rsi', figure=rsi_fig),
                        dcc.Graph(id='equity', figure=equity_fig)
                    ])
                    import copy
                    @app.callback(
                        [Output('rsi', 'figure'), Output('equity', 'figure')],
                        [Input('equity-curve', 'relayoutData')],
                        [State('rsi', 'figure'), State('equity', 'figure')]
                    )
                    def sync_xaxis(relayoutData, rsi_fig, equity_fig):
                        rsi_fig = copy.deepcopy(rsi_fig)
                        equity_fig = copy.deepcopy(equity_fig)
                        if relayoutData:
                            if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
                                x0 = relayoutData['xaxis.range[0]']
                                x1 = relayoutData['xaxis.range[1]']
                                rsi_fig['layout']['xaxis']['range'] = [x0, x1]
                                equity_fig['layout']['xaxis']['range'] = [x0, x1]
                            elif 'xaxis.autorange' in relayoutData and relayoutData['xaxis.autorange']:
                                rsi_fig['layout']['xaxis'].pop('range', None)
                                equity_fig['layout']['xaxis'].pop('range', None)
                        return rsi_fig, equity_fig
                    app.run(debug=False)
                threading.Thread(target=launch_dashboard, daemon=True).start()
                logging.info('Dash dashboard launched at http://127.0.0.1:8050/')
            except ImportError:
                logging.warning('Dash or Plotly is not installed. Please install with "pip install dash plotly".')
            except Exception as e:
                logging.error(f'Error launching dashboard: {e}')
        elif plot and final_equity_only:
            plt.figure(figsize=(12,5))
            plt.plot(equity_curve, label='Cumulative % Gain/Loss', color='black')
            plt.scatter(trade_indices, trade_returns, color='magenta', label='Trade Close', marker='x')
            plt.title('Cumulative % Gain/Loss (Full Simulation)')
            plt.xlabel('Minute Index')
            plt.ylabel('Cumulative %')
            plt.legend()
            plt.tight_layout()
            plt.pause(0.01)
            plt.close()
        self.signals = list(zip(buy_x, buy_y, ['BUY']*len(buy_x))) + list(zip(sell_x, sell_y, ['SELL']*len(sell_x)))
        return equity_curve[-1] if equity_curve else 0.0

    def optimize_parameters(
        self,
        symbol: str,
        param_ranges: Dict[str, Tuple[float, float, int]],
        method: str = "brute",
        objective: str = "final_equity",
        consistency_weight: float = 0.3,
        max_equity_weight: float = 0.7,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters for maximum equity and consistency.
        
        Args:
            symbol: Stock symbol for data fetching
            param_ranges: Dict mapping parameter names to (min, max, steps) tuples
                Example: {
                    'macd_fast': (5, 20, 4),
                    'macd_slow': (20, 50, 4),
                    'macd_signal': (5, 15, 3),
                    'rsi_period': (10, 20, 3),
                    'roc_period': (3, 10, 3),
                    'ema_mom_period': (15, 30, 4)
                }
            method: Optimization method ('brute' or 'differential_evolution')
            objective: Optimization objective ('final_equity', 'consistency', or 'combined')
            consistency_weight: Weight for consistency in combined objective
            max_equity_weight: Weight for max equity in combined objective
            verbose: Whether to print progress
            
        Returns:
            Dict containing:
                - best_params: Dict of optimized parameters
                - best_score: Best objective score achieved
                - final_equity: Final equity with best parameters
                - consistency_score: Consistency score with best parameters
                - all_results: List of all parameter combinations tried
        """
        # Fetch data for the symbol
        if verbose:
            print(f"🔍 Fetching data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        historical_data = ticker.history(period="60d", interval="15m")
        
        if historical_data.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        
        # Prepare parameter combinations
        param_names = list(param_ranges.keys())
        param_bounds = []
        
        for param_name in param_names:
            min_val, max_val, steps = param_ranges[param_name]
            param_bounds.append(slice(min_val, max_val, steps))
        
        # Store all results for analysis
        all_results = []
        best_score = float('-inf')
        best_params = {}
        best_equity = 0.0
        best_consistency = 0.0
        
        def objective_function(params):
            """Objective function to maximize."""
            nonlocal all_results, best_score, best_params, best_equity, best_consistency
            
            # Convert params to integers where needed
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = int(params[i])
            
            try:
                # Create strategy with these parameters
                strategy = DefaultStrategy(
                    historical_data,
                    **param_dict
                )
                
                # Create backtester and run simulation
                backtester = StrategyBacktester(strategy)
                final_equity = backtester.run(plot=False, slider=False)
                
                # Calculate consistency score (lower standard deviation of equity curve is better)
                buy_signals, sell_signals, equity_curve = self._calculate_backtest_data(strategy, strategy.data)
                
                if len(equity_curve) > 1:
                    # Calculate consistency as inverse of volatility (lower volatility = higher consistency)
                    equity_changes = np.diff(equity_curve)
                    if len(equity_changes) > 0:
                        volatility = np.std(equity_changes)
                        consistency_score = 1.0 / (1.0 + volatility)  # Higher is better
                    else:
                        consistency_score = 0.0
                else:
                    consistency_score = 0.0
                
                # Calculate combined objective based on method
                if objective == "final_equity":
                    score = final_equity
                elif objective == "consistency":
                    score = consistency_score * 100  # Scale to match equity range
                elif objective == "combined":
                    score = (max_equity_weight * final_equity + 
                            consistency_weight * consistency_score * 100)
                else:
                    raise ValueError(f"Unknown objective: {objective}")
                
                # Store result
                result = {
                    'params': param_dict.copy(),
                    'final_equity': final_equity,
                    'consistency_score': consistency_score,
                    'combined_score': score
                }
                all_results.append(result)
                
                # Update best if this is better
                if score > best_score:
                    best_score = score
                    best_params = param_dict.copy()
                    best_equity = final_equity
                    best_consistency = consistency_score
                
                if verbose and len(all_results) % 10 == 0:
                    print(f"📊 Tested {len(all_results)} combinations. Best score: {best_score:.2f}")
                
                # Return negative for minimization algorithms
                return -score
                
            except Exception as e:
                if verbose:
                    print(f"❌ Error with params {param_dict}: {e}")
                return float('inf')  # Bad score for failed combinations
        
        if verbose:
            print(f"🎯 Starting optimization with {method} method...")
            print(f"📈 Objective: {objective}")
            if objective == "combined":
                print(f"⚖️  Weights - Equity: {max_equity_weight}, Consistency: {consistency_weight}")
        
        # Run optimization
        if method == "brute":
            # Brute force search over all combinations
            result = brute(objective_function, param_bounds, full_output=True, finish=None)
            optimal_params = result[0]
        elif method == "differential_evolution":
            # More sophisticated optimization
            bounds = [(param_ranges[name][0], param_ranges[name][1]) for name in param_names]
            result = differential_evolution(objective_function, bounds, seed=42, maxiter=100)
            optimal_params = result.x
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Convert optimal params back to dict
        final_optimal_params = {}
        for i, param_name in enumerate(param_names):
            final_optimal_params[param_name] = int(optimal_params[i])
        
        if verbose:
            print(f"\n🏆 OPTIMIZATION COMPLETE!")
            print(f"✅ Best Parameters: {final_optimal_params}")
            print(f"📈 Final Equity: {best_equity:.2f}%")
            print(f"🎯 Consistency Score: {best_consistency:.4f}")
            print(f"⭐ Combined Score: {best_score:.2f}")
            print(f"🔬 Total Combinations Tested: {len(all_results)}")
        
        return {
            'best_params': final_optimal_params,
            'best_score': best_score,
            'final_equity': best_equity,
            'consistency_score': best_consistency,
            'all_results': all_results,
            'method': method,
            'objective': objective
        }
    
    def _calculate_backtest_data(self, strategy, data):
        """Helper function to calculate equity curve and signal points."""
        buy_signals = {'x': [], 'y': []}
        sell_signals = {'x': [], 'y': []}
        equity_curve = []
        entry_price = None
        cumulative_return = 0.0
        
        for i in range(len(data)):
            signal = strategy.check_signals(i)
            close_price = data['Close'].iloc[i]
            
            if signal == "BUY" and not strategy.bought:
                buy_signals['x'].append(i)
                buy_signals['y'].append(close_price)
                entry_price = close_price
                strategy.bought = True
            elif signal == "SELL" and strategy.bought and entry_price is not None:
                sell_signals['x'].append(i)
                sell_signals['y'].append(close_price)
                trade_return = (close_price - entry_price) / entry_price * 100
                cumulative_return += trade_return
                entry_price = None
                strategy.bought = False
            
            # Update equity curve
            if strategy.bought and entry_price is not None:
                current_return = (close_price - entry_price) / entry_price * 100
                equity_curve.append(cumulative_return + current_return)
            else:
                equity_curve.append(cumulative_return)
        
        return buy_signals, sell_signals, equity_curve

# Example usage in your notebook:
# import strategy_backtester
# sb = strategy_backtester.StrategyBacktester(historical_data)
# sb.run(plot_every=5)  # plot_every controls how often the plot updates (every N points)