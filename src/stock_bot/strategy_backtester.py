import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from IPython.display import clear_output
import logging
from typing import Any, List, Optional

class StrategyBacktester:
    """
    Backtests a trading strategy using technical indicators and simulates trades.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        macd_fast: int = 3,
        macd_slow: int = 7,
        macd_signal: int = 3,
        rsi_period: int = 5,
        roc_period: int = 1,
        ema_mom_period: int = 5
    ) -> None:
        """
        Initialize the backtester with data and indicator parameters.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        required_cols = ['Close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        self.data = data.copy()
        self.signals: List[Any] = []
        self.bought: bool = False
        # Precompute indicators
        try:
            self.data['EMA_fast'] = self.data['Close'].ewm(span=macd_fast, adjust=False).mean()
            self.data['EMA_slow'] = self.data['Close'].ewm(span=macd_slow, adjust=False).mean()
            self.data['MACD'] = self.data['EMA_fast'] - self.data['EMA_slow']
            self.data['MACD_signal'] = self.data['MACD'].ewm(span=macd_signal, adjust=False).mean()
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            self.data['ROC'] = self.data['Close'].pct_change(periods=roc_period)
            self.data['EMA_mom'] = self.data['Close'].ewm(span=ema_mom_period, adjust=False).mean()
        except Exception as e:
            logging.error(f"Error computing indicators: {e}")

    def check_signals(self, idx: int) -> Optional[str]:
        """
        Check for buy/sell signals at a given index.
        Args:
            idx (int): Index in the DataFrame.
        Returns:
            Optional[str]: "BUY", "SELL", or None.
        """
        if idx < 7 or idx < 1:
            return None
        try:
            macd_prev = self.data['MACD'].iloc[idx-1]
            macd_signal_prev = self.data['MACD_signal'].iloc[idx-1]
            macd_now = self.data['MACD'].iloc[idx]
            macd_signal_now = self.data['MACD_signal'].iloc[idx]
            rsi_now = self.data['RSI'].iloc[idx]
            roc_now = self.data['ROC'].iloc[idx]
            # Enhanced polynomial trend analysis: block buys if macro trend is strongly down
            trend_window = 20  # Number of points to check for trend
            trending_down = False
            if idx >= trend_window:
                recent_closes = self.data['Close'].iloc[idx-trend_window:idx]
                x = np.arange(trend_window)
                coeffs = np.polyfit(x, recent_closes, 2)
                # Calculate fitted values
                fitted = np.polyval(coeffs, x)
                # Assess macro trend: compare mean of first half vs second half
                mean_first = np.mean(fitted[:trend_window//2])
                mean_second = np.mean(fitted[trend_window//2:])
                slope_end = 2*coeffs[0]*x[-1] + coeffs[1]
                # Block buys if both the macro trend and slope are negative and the drop is significant
                if (mean_second < mean_first) and (slope_end < -0.01 * mean_first):
                    trending_down = True
            # More aggressive for volatility
            if (macd_prev < macd_signal_prev and macd_now > macd_signal_now and rsi_now < 40 and roc_now > 0 and not trending_down):
                return "BUY"
            if macd_prev > macd_signal_prev and macd_now < macd_signal_now and rsi_now > 60:
                return "SELL"
        except Exception as e:
            logging.error(f"Error checking signals at idx {idx}: {e}")
        return None

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
        try:
            for i in range(len(self.data)):
                signal = self.check_signals(i)
                # Stop loss
                if self.bought and entry_price is not None:
                    stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                    if self.data['Close'].iloc[i] <= stop_loss_price:
                        sell_x.append(i)
                        sell_y.append(self.data['Close'].iloc[i])
                        trade_return = (self.data['Close'].iloc[i] - entry_price) / entry_price * 100
                        cumulative_return += trade_return
                        trade_returns.append(cumulative_return)
                        trade_indices.append(i)
                        entry_price = None
                        self.bought = False
                        continue
                if signal == "BUY" and not self.bought:
                    buy_x.append(i)
                    buy_y.append(self.data['Close'].iloc[i])
                    entry_price = self.data['Close'].iloc[i]
                    self.bought = True
                elif signal == "SELL" and self.bought:
                    sell_x.append(i)
                    sell_y.append(self.data['Close'].iloc[i])
                    trade_return = (self.data['Close'].iloc[i] - entry_price) / entry_price * 100
                    cumulative_return += trade_return
                    trade_returns.append(cumulative_return)
                    trade_indices.append(i)
                    entry_price = None
                    self.bought = False
                # Update running equity curve
                if self.bought and entry_price is not None:
                    current_return = (self.data['Close'].iloc[i] - entry_price) / entry_price * 100
                    equity_curve.append(cumulative_return + current_return)
                else:
                    equity_curve.append(cumulative_return)
        except Exception as e:
            logging.error(f"Error during backtest run: {e}")

        # Launch Dash dashboard after backtest if plot and slider are True
        if plot and slider:
            try:
                import dash
                from dash import dcc, html
                import plotly.graph_objs as go
                import threading
                def launch_dashboard():
                    # ...existing code...
                    x_vals = list(range(len(self.data)))
                    buy_points = [(i, buy_y[j]) for j, i in enumerate(buy_x)]
                    sell_points = [(i, sell_y[j]) for j, i in enumerate(sell_x)]
                    from dash.dependencies import Input, Output, State
                    app = dash.Dash(__name__)
                    equity_curve_fig = {
                        'data': [
                            go.Scatter(x=x_vals, y=list(self.data['Close']), mode='lines', name='Close Price', line=dict(color='blue')),
                            go.Scatter(x=[x for x, _ in buy_points], y=[y for _, y in buy_points], mode='markers', name='BUY', marker=dict(color='green', symbol='circle', size=10)),
                            go.Scatter(x=[x for x, _ in sell_points], y=[y for _, y in sell_points], mode='markers', name='SELL', marker=dict(color='red', symbol='x', size=10)),
                            go.Scatter(x=x_vals, y=list(self.data['EMA_fast']), mode='lines', name='EMA Fast', line=dict(dash='dash', color='orange')),
                            go.Scatter(x=x_vals, y=list(self.data['EMA_slow']), mode='lines', name='EMA Slow', line=dict(dash='dash', color='purple')),
                        ],
                        'layout': go.Layout(title='Equity Curve & Signals', xaxis={'title': 'Index'}, yaxis={'title': 'Price'})
                    }
                    rsi_fig = {
                        'data': [
                            go.Scatter(x=x_vals, y=list(self.data['RSI']), mode='lines', name='RSI', line=dict(color='blue')),
                        ],
                        'layout': go.Layout(title='RSI', xaxis={'title': 'Index'}, yaxis={'title': 'RSI'})
                    }
                    equity_fig = {
                        'data': [
                            go.Scatter(x=x_vals, y=equity_curve, mode='lines', name='Cumulative % Gain/Loss', line=dict(color='black')),
                            go.Scatter(x=[i for i in trade_indices], y=trade_returns, mode='markers', name='Trade Close', marker=dict(color='magenta', symbol='x', size=10)),
                        ],
                        'layout': go.Layout(title='Cumulative % Gain/Loss', xaxis={'title': 'Minute Index'}, yaxis={'title': 'Cumulative %'})
                    }
                    app.layout = html.Div([
                        html.H1('Backtest Analysis Dashboard'),
                        dcc.Graph(id='equity-curve', figure=equity_curve_fig),
                        dcc.Graph(id='rsi', figure=rsi_fig),
                        dcc.Graph(id='equity', figure=equity_fig)
                    ])

                    # Synchronize x-axis zoom/pan
                    import copy
                    @app.callback(
                        [Output('rsi', 'figure'), Output('equity', 'figure')],
                        [Input('equity-curve', 'relayoutData')],
                        [State('rsi', 'figure'), State('equity', 'figure')]
                    )
                    def sync_xaxis(relayoutData, rsi_fig, equity_fig):
                        rsi_fig = copy.deepcopy(rsi_fig)
                        equity_fig = copy.deepcopy(equity_fig)
                        # Handle zoom/pan events
                        if relayoutData:
                            # Check for range update
                            if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
                                x0 = relayoutData['xaxis.range[0]']
                                x1 = relayoutData['xaxis.range[1]']
                                rsi_fig['layout']['xaxis']['range'] = [x0, x1]
                                equity_fig['layout']['xaxis']['range'] = [x0, x1]
                            # Check for autorange reset
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

        elif plot and not final_equity_only:
            pass  # ...existing code for non-slider plotting...
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