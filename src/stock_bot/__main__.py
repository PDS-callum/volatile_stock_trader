import click
import yfinance as yf
import json
import time
from scipy.optimize import brute
from .strategy_backtester import StrategyBacktester
from .strategy import DefaultStrategy
from .strategy_tester import StrategyTester
import logging
from typing import Any, Tuple

logging.basicConfig(level=logging.INFO)

@click.group()
def cli():
    """Main CLI group."""
    pass
def run_live_strategy_core(
    symbol,
    macd_fast,
    macd_slow,
    macd_signal,
    rsi_period,
    roc_period,
    ema_mom_period,
    interval,
    min_points=40,
    plot=True
):
    """Run the live strategy tester with the given parameters."""
    strategy_params = dict(
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
        rsi_period=int(rsi_period),
        roc_period=int(roc_period),
        ema_mom_period=int(ema_mom_period)
    )
    tester = StrategyTester(
        symbol,
        strategy_cls=DefaultStrategy,
        interval=interval,
        min_points=min_points,
        strategy_params=strategy_params
    )
    tester.run()
    print("Live dashboard running at http://127.0.0.1:8050/. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting live dashboard...")
    return None
@cli.command()
@click.argument("symbol")
@click.option("--macd-fast", default=12, help="MACD fast period")
@click.option("--macd-slow", default=26, help="MACD slow period")
@click.option("--macd-signal", default=9, help="MACD signal period")
@click.option("--rsi-period", default=14, help="RSI period")
@click.option("--roc-period", default=5, help="ROC period")
@click.option("--ema-mom-period", default=20, help="EMA momentum period")
@click.option("--interval", default="15m", help="Data interval")
@click.option("--min-points", default=40, help="Minimum data points before strategy runs")
@click.option("--plot/--no-plot", default=True, help="Show plots")
def run_live_strategy(
    symbol,
    macd_fast,
    macd_slow,
    macd_signal,
    rsi_period,
    roc_period,
    ema_mom_period,
    interval,
    min_points,
    plot
):
    """CLI wrapper for run_live_strategy_core."""
    return run_live_strategy_core(
        symbol,
        macd_fast,
        macd_slow,
        macd_signal,
        rsi_period,
        roc_period,
        ema_mom_period,
        interval,
        min_points,
        plot
    )

def run_back_strategy_core(
    symbol,
    macd_fast,
    macd_slow,
    macd_signal,
    rsi_period,
    roc_period,
    ema_mom_period,
    period,
    interval,
    final_equity_only,
    plot,
    notebook=False
):
    """Run the strategy with the given parameters. If notebook=True, return a Plotly figure for inline display."""
    historical_data = get_data(
        symbol,
        period=period,
        interval=interval
    )
    strategy = DefaultStrategy(
        historical_data,
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
        rsi_period=int(rsi_period),
        roc_period=int(roc_period),
        ema_mom_period=int(ema_mom_period)
    )
    sb = StrategyBacktester(strategy)
    if notebook:
        import matplotlib.pyplot as plt
        x_vals = list(range(len(sb.data)))
        equity_curve = []
        entry_price = None
        cumulative_return = 0.0
        for i in range(len(sb.data)):
            signal = strategy.check_signals(i)
            if signal == "BUY" and not strategy.bought:
                entry_price = sb.data['Close'].iloc[i]
                strategy.bought = True
            elif signal == "SELL" and strategy.bought and entry_price is not None:
                trade_return = (sb.data['Close'].iloc[i] - entry_price) / entry_price * 100
                cumulative_return += trade_return
                entry_price = None
                strategy.bought = False
            # Update running equity curve
            if strategy.bought and entry_price is not None:
                current_return = (sb.data['Close'].iloc[i] - entry_price) / entry_price * 100
                equity_curve.append(cumulative_return + current_return)
            else:
                equity_curve.append(cumulative_return)
        plt.figure(figsize=(12, 5))
        plt.plot(x_vals, equity_curve, label='Equity Curve', color='blue')
        plt.title('Equity Change Over Time')
        plt.xlabel('Index')
        plt.ylabel('Cumulative % Gain/Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return None
    else:
        eq = sb.run(plot=True, slider=True, final_equity_only=final_equity_only)
        historical_data.to_csv("cli_historical_data.csv")
        with open("cli_signals.json", "w") as f:
            json.dump(sb.signals, f)
        with open("cli_equity_curve.json", "w") as f:
            json.dump(list(eq if isinstance(eq, (list, tuple)) else [eq]), f)
        print("Dashboard running at http://127.0.0.1:8050/. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting dashboard...")
        return eq

@cli.command()
@click.argument("symbol")
@click.option("--macd-fast", default=12, help="MACD fast period")
@click.option("--macd-slow", default=26, help="MACD slow period")
@click.option("--macd-signal", default=9, help="MACD signal period")
@click.option("--rsi-period", default=14, help="RSI period")
@click.option("--roc-period", default=5, help="ROC period")
@click.option("--ema-mom-period", default=20, help="EMA momentum period")
@click.option("--period", default="60d", help="Data period")
@click.option("--interval", default="15m", help="Data interval")
@click.option("--final-equity-only", is_flag=True, help="Return only final equity")
@click.option("--plot/--no-plot", default=True, help="Show plots")
def run_back_strategy(
    symbol,
    macd_fast,
    macd_slow,
    macd_signal,
    rsi_period,
    roc_period,
    ema_mom_period,
    period,
    interval,
    final_equity_only,
    plot
):
    """CLI wrapper for run_back_strategy_core."""
    return run_back_strategy_core(
        symbol,
        macd_fast,
        macd_slow,
        macd_signal,
        rsi_period,
        roc_period,
        ema_mom_period,
        period,
        interval,
        final_equity_only,
        plot,
        notebook=False
    )

def get_data(symbol: str, period: str = "60d", interval: str = "15m") -> Any:
    """
    Fetch historical data for a given symbol.
    Args:
        symbol (str): Stock symbol.
        period (str): Data period.
        interval (str): Data interval.
    Returns:
        Any: Historical data DataFrame.
    """
    try:
        ticker = yf.Ticker(symbol)
        historical_data = ticker.history(
            period=period,
            interval=interval
        )
        if historical_data.empty:
            logging.warning(f"No data found for symbol {symbol}.")
        return historical_data
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

def optimise(historical_data: Any) -> Tuple:
    """
    Optimise strategy parameters using brute force.
    Args:
        historical_data (Any): Historical data DataFrame.
    Returns:
        Tuple: Best parameters found.
    """
    def objective(params):
        macd_fast, macd_slow, macd_signal, rsi_period, roc_period, ema_mom_period = params
        sb = StrategyBacktester(
            historical_data,
            macd_fast=int(macd_fast),
            macd_slow=int(macd_slow),
            macd_signal=int(macd_signal),
            rsi_period=int(rsi_period),
            roc_period=int(roc_period),
            ema_mom_period=int(ema_mom_period)
        )
        return -sb.run(plot=False)  # Negative for maximization

    ranges = (
        slice(8, 20, 2),    # macd_fast
        slice(20, 40, 2),   # macd_slow
        slice(6, 15, 1),    # macd_signal
        slice(10, 20, 2),   # rsi_period
        slice(2, 10, 2),    # roc_period
        slice(10, 40, 5)    # ema_mom_period
    )

    try:
        best_params = brute(objective, ranges, finish=None)
        return best_params
    except Exception as e:
        logging.error(f"Error in optimisation: {e}")
        return tuple()


# Entry point for CLI execution
if __name__ == "__main__":
    cli()