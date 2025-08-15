import click
import yfinance as yf
import json
import time
from scipy.optimize import brute
from .strategy_backtester import StrategyBacktester
import logging
from typing import Any, Tuple

logging.basicConfig(level=logging.INFO)

@click.group()
def cli():
    """Main CLI group."""
    pass
import click
import yfinance as yf
from scipy.optimize import brute
from .strategy_backtester import StrategyBacktester
import logging
from typing import Any, Tuple

logging.basicConfig(level=logging.INFO)

def run_strategy_core(
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
    sb = StrategyBacktester(
        historical_data,
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
        rsi_period=int(rsi_period),
        roc_period=int(roc_period),
        ema_mom_period=int(ema_mom_period)
    )
    if notebook:
        import plotly.graph_objs as go
        x_vals = list(range(len(sb.data)))
        equity_curve = []
        entry_price = None
        cumulative_return = 0.0
        for i in range(len(sb.data)):
            signal = sb.check_signals(i)
            if signal == "BUY" and not sb.bought:
                entry_price = sb.data['Close'].iloc[i]
                sb.bought = True
            elif signal == "SELL" and sb.bought and entry_price is not None:
                trade_return = (sb.data['Close'].iloc[i] - entry_price) / entry_price * 100
                cumulative_return += trade_return
                entry_price = None
                sb.bought = False
            # Update running equity curve
            if sb.bought and entry_price is not None:
                current_return = (sb.data['Close'].iloc[i] - entry_price) / entry_price * 100
                equity_curve.append(cumulative_return + current_return)
            else:
                equity_curve.append(cumulative_return)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=equity_curve, mode='lines', name='Equity Curve'))
        fig.update_layout(
            title='Equity Change Over Time',
            xaxis_title='Index',
            yaxis_title='Cumulative % Gain/Loss'
        )
        return fig
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
def run_strategy(
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
    """CLI wrapper for run_strategy_core."""
    return run_strategy_core(
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