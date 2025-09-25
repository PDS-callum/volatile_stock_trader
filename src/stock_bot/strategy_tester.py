import pandas as pd
import time
import logging
from typing import List, Optional
from .strategies.base_strategy import BaseStrategy

class StrategyTester:
    """
    Collects live data, applies a strategy, and launches a dashboard for real-time trading simulation.
    """
    def __init__(self, symbol: str, strategy_cls, period: str = "60d", interval: str = "15m", min_points: int = 40, strategy_params=None):
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.min_points = min_points
        self.strategy_cls = strategy_cls
        self.strategy_params = strategy_params or {}
        self.data = pd.DataFrame()
        self.strategy: Optional[BaseStrategy] = None
        self.buy_x, self.buy_y, self.sell_x, self.sell_y = [], [], [], []
        self.equity_curve: List[float] = []
        self.entry_price: Optional[float] = None
        self.cumulative_return: float = 0.0
        self.trade_returns: List[float] = []
        self.trade_indices: List[int] = []
        self.running = False

    def fetch_live_data(self):
        """
        Fetch live price data using yfinance. Appends the latest close price to self.data.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Please install yfinance: pip install yfinance")

        try:
            ticker = yf.Ticker(self.symbol)
            # Get the latest 1 minute data
            df = ticker.history(period=self.period, interval=self.interval)
            if not df.empty:
                latest_close = df['Close'].iloc[-1]
                self.data = pd.concat([self.data, pd.DataFrame({"Close": [latest_close]})], ignore_index=True)
        except Exception as e:
            logging.warning(f"Could not fetch live price for {self.symbol}: {e}")

    def run(self, updates=999999999, closed_market=False):
        """
        Run live tester and print updates about price, signals, and equity change.
        If closed_market=True, use historical 30d 15m data from yfinance and feed it as if live.
        """
        import time
        if closed_market:
            try:
                import yfinance as yf
                import matplotlib.pyplot as plt
                from IPython.display import display, clear_output
            except ImportError:
                raise ImportError("Please install yfinance and matplotlib: pip install yfinance matplotlib")
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(period=self.period, interval=self.interval)
            if hist.empty:
                print("No historical data available for closed market simulation.")
                return
            closes = hist['Close'].tolist()
            plt.ion()
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            for i, price in enumerate(closes[:updates]):
                self.data = pd.concat([self.data, pd.DataFrame({"Close": [price]})], ignore_index=True)
                clear_output(wait=True)
                if len(self.data) >= self.min_points:
                    self.apply_strategy()
                    x_vals = list(range(len(self.data)))
                    axs[0].cla()
                    axs[1].cla()
                    axs[0].plot(x_vals, self.data['Close'], label='Close Price', color='blue')
                    axs[0].scatter(self.buy_x, self.buy_y, color='green', label='BUY', marker='o', s=60)
                    axs[0].scatter(self.sell_x, self.sell_y, color='red', label='SELL', marker='x', s=60)
                    axs[0].set_title('Price & Signals')
                    axs[0].set_xlabel('Index')
                    axs[0].set_ylabel('Price')
                    axs[0].legend()
                    axs[1].plot(x_vals, self.equity_curve, label='Cumulative % Gain/Loss', color='black')
                    axs[1].scatter(self.trade_indices, self.trade_returns, color='magenta', label='Trade Close', marker='x', s=60)
                    axs[1].set_title('Cumulative % Gain/Loss')
                    axs[1].set_xlabel('Index')
                    axs[1].set_ylabel('Cumulative %')
                    axs[1].legend()
                    plt.tight_layout()
                    display(fig)
                    plt.pause(0.01)
                else:
                    axs[0].cla()
                    axs[0].plot(list(range(len(self.data))), self.data['Close'], label='Close Price', color='blue')
                    axs[0].set_title('Price (Waiting for more data)')
                    axs[0].set_xlabel('Index')
                    axs[0].set_ylabel('Price')
                    axs[0].legend()
                    axs[1].cla()
                    axs[1].set_title('Cumulative % Gain/Loss')
                    axs[1].set_xlabel('Index')
                    axs[1].set_ylabel('Cumulative %')
                    plt.tight_layout()
                    display(fig)
                    plt.pause(0.01)
                time.sleep(0.01)
            plt.ioff()
            plt.show()
        else:
            for i in range(updates):
                self.fetch_live_data()
                if len(self.data) >= self.min_points:
                    self.apply_strategy()
                    latest_idx = len(self.data) - 1
                    latest_price = self.data['Close'].iloc[-1]
                    last_signal = None
                    if self.buy_x and self.buy_x[-1] == latest_idx:
                        last_signal = 'BUY'
                    elif self.sell_x and self.sell_x[-1] == latest_idx:
                        last_signal = 'SELL'
                    print(f"Update {i+1}: Price={latest_price:.2f} | Signal={last_signal or 'NONE'} | Equity={self.equity_curve[-1]:.2f}")
                else:
                    print(f"Update {i+1}: Price={self.data['Close'].iloc[-1]:.2f} | Waiting for more data...")
                time.sleep(self._interval_seconds())

    def _interval_seconds(self):
        # Convert interval string to seconds (supports '15m', '1h', etc.)
        if self.interval.endswith('m'):
            return int(self.interval[:-1]) * 60
        elif self.interval.endswith('h'):
            return int(self.interval[:-1]) * 3600
        return 60

    def apply_strategy(self):
        self.strategy = self.strategy_cls(self.data, **self.strategy_params)
        self.buy_x, self.buy_y, self.sell_x, self.sell_y = [], [], [], []
        self.equity_curve = []
        self.entry_price = None
        self.cumulative_return = 0.0
        self.trade_returns = []
        self.trade_indices = []
        for i, close_price in enumerate(self.data['Close']):
            signal = self.strategy.check_signals(i)
            if self.strategy.bought and self.entry_price is not None:
                stop_loss_price = self.entry_price * 0.95
                if close_price <= stop_loss_price:
                    self.sell_x.append(i)
                    self.sell_y.append(close_price)
                    trade_return = (close_price - self.entry_price) / self.entry_price * 100
                    self.cumulative_return += trade_return
                    self.trade_returns.append(self.cumulative_return)
                    self.trade_indices.append(i)
                    self.entry_price = None
                    self.strategy.bought = False
                    continue
            if signal == "BUY" and not self.strategy.bought:
                self.buy_x.append(i)
                self.buy_y.append(close_price)
                self.entry_price = close_price
                self.strategy.bought = True
            elif signal == "SELL" and self.strategy.bought:
                self.sell_x.append(i)
                self.sell_y.append(close_price)
                trade_return = (close_price - self.entry_price) / self.entry_price * 100
                self.cumulative_return += trade_return
                self.trade_returns.append(self.cumulative_return)
                self.trade_indices.append(i)
                self.entry_price = None
                self.strategy.bought = False
            if self.strategy.bought and self.entry_price is not None:
                current_return = (close_price - self.entry_price) / self.entry_price * 100
                self.equity_curve.append(self.cumulative_return + current_return)
            else:
                self.equity_curve.append(self.cumulative_return)

    # Removed launch_dashboard; plotting is now handled in run()
