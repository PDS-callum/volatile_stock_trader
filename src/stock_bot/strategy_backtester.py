
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Any, List, Optional
from .strategy import BaseStrategy


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
        plot_every: int = 1,
        final_equity_only: bool = False,
        stop_loss_pct: float = 5,
        plot: bool = True,
        slider: bool = True
    ) -> float:
        """
        Run the backtest simulation.
        Args:
            plot_every (int): How often to update plot.
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
            elif signal == "SELL" and self.strategy.bought:
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

# Example usage in your notebook:
# import strategy_backtester
# sb = strategy_backtester.StrategyBacktester(historical_data)
# sb.run(plot_every=5)  # plot_every controls how often the plot updates (every N points)