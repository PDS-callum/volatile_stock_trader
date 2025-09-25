import click
import yfinance as yf
import json
import time
import os
import glob
import threading
import webbrowser
import yaml
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize
from .strategy_backtester import StrategyBacktester
from .strategy_loader import load_strategy
from .strategy_tester import StrategyTester
import logging
from typing import Any, Tuple, List, Dict, Optional

logging.basicConfig(level=logging.INFO)

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If config file is missing required fields
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['symbols', 'data', 'strategies']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Config file missing required field: {field}")
    
    # Set defaults for optional fields
    config.setdefault('strategy_params', {})
    config.setdefault('output', {})
    config.setdefault('test', {})
    
    # Set default values
    config['output'].setdefault('mode', 'comparison')
    config['output'].setdefault('use_notebook_plots', False)
    config['output'].setdefault('dashboard_port', 8052)
    config['output'].setdefault('save_results', False)
    config['output'].setdefault('results_file', 'strategy_comparison_results.json')
    
    config['test'].setdefault('max_symbols', 0)
    config['test'].setdefault('parallel', False)
    config['test'].setdefault('save_results', False)
    
    return config

@click.group()
def cli():
    """Main CLI group."""
    pass
def run_live_strategy_core(
    symbol,
    strategy,
    strategy_class,
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
    try:
        # Load the strategy
        strategy_cls = load_strategy(strategy, strategy_class)
    except Exception as e:
        print(f"Error loading strategy: {e}")
        return None
        
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
        strategy_cls=strategy_cls,
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
@click.option("--strategy", default="default", help="Strategy name or path to strategy file")
@click.option("--strategy-class", help="Strategy class name (when using custom strategy file)")
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
    strategy,
    strategy_class,
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
        strategy,
        strategy_class,
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
    strategy,
    strategy_class,
    macd_fast,
    macd_slow,
    macd_signal,
    rsi_period,
    roc_period,
    ema_mom_period,
    period,
    interval,
    final_equity_only=False,
    plot=True,
    notebook=False,
    output_mode="full",
    use_notebook_plots=False
):
    """
    Run the strategy with the given parameters.
    
    Args:
        symbol: Stock symbol (str) or list of symbols to analyze
        macd_fast, macd_slow, macd_signal: MACD parameters
        rsi_period: RSI period
        roc_period: Rate of Change period
        ema_mom_period: EMA momentum period
        period: Data period (e.g., "60d")
        interval: Data interval (e.g., "15m")
        final_equity_only: Legacy parameter for backward compatibility
        plot: Whether to show plots
        notebook: Legacy parameter for backward compatibility
        output_mode: Controls what is displayed/returned
            - "equity_only": Return only the final equity change (no graphs)
            - "equity_curve": Return/show only the equity change over time
            - "full": Show equity curve, stock price with signals, bollinger bands, MACD and ROC
        use_notebook_plots: If True, use matplotlib plots in notebook. If False (default), use dashboard
    
    Returns:
        float or dict: Final equity percentage change (single symbol) or dict of results (multiple symbols)
    """
    # Handle multiple symbols
    if isinstance(symbol, (list, tuple)):
        return _run_multi_symbol_analysis(
            symbol, macd_fast, macd_slow, macd_signal, rsi_period, roc_period, 
            ema_mom_period, period, interval, final_equity_only, plot, notebook, 
            output_mode, use_notebook_plots
        )
    
    # Single symbol analysis (existing logic)
    return _run_single_symbol_analysis(
        symbol, strategy, strategy_class, macd_fast, macd_slow, macd_signal, rsi_period, roc_period,
        ema_mom_period, period, interval, final_equity_only, plot, notebook,
        output_mode, use_notebook_plots
    )

def _run_multi_symbol_analysis(
    symbols, macd_fast, macd_slow, macd_signal, rsi_period, roc_period,
    ema_mom_period, period, interval, final_equity_only, plot, notebook,
    output_mode, use_notebook_plots
):
    """Handle multiple symbol analysis with comparison."""
    results = {}
    all_equity_curves = {}
    
    print(f"🔍 Analyzing {len(symbols)} symbols: {', '.join(symbols)}")
    print("=" * 60)
    
    # Analyze each symbol
    for i, sym in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Analyzing {sym}...")
        
        try:
            # Run single symbol analysis
            result = _run_single_symbol_analysis(
                sym, strategy, strategy_class, macd_fast, macd_slow, macd_signal, rsi_period, roc_period,
                ema_mom_period, period, interval, final_equity_only, plot, notebook,
                output_mode, use_notebook_plots
            )
            results[sym] = result
            
        except Exception as e:
            print(f"❌ Error analyzing {sym}: {e}")
            results[sym] = 0.0
    
    # Show comparison
    _show_multi_symbol_comparison(results, symbols, output_mode, use_notebook_plots)
    
    return results

def _run_single_symbol_analysis(
    symbol, strategy, strategy_class, macd_fast, macd_slow, macd_signal, rsi_period, roc_period,
    ema_mom_period, period, interval, final_equity_only, plot, notebook,
    output_mode, use_notebook_plots
):
    """Single symbol analysis (original logic moved here)."""
    historical_data = get_data(
        symbol,
        period=period,
        interval=interval
    )
    
    if historical_data is None or historical_data.empty:
        print(f"No data available for {symbol}")
        return 0.0
    
    try:
        # Load the strategy
        strategy_cls = load_strategy(strategy, strategy_class)
    except Exception as e:
        print(f"Error loading strategy: {e}")
        return 0.0
    
    strategy = strategy_cls(
        historical_data,
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
        rsi_period=int(rsi_period),
        roc_period=int(roc_period),
        ema_mom_period=int(ema_mom_period)
    )
    sb = StrategyBacktester(strategy)
    
    # Calculate equity curve and signals
    buy_signals, sell_signals, equity_curve = _calculate_backtest_data(strategy, sb.data)
    final_equity = equity_curve[-1] if equity_curve else 0.0
    
    if output_mode == "equity_only":
        if plot:  # Only print if plot is True (avoid spam in multi-symbol mode)
            print(f"Final equity change for {symbol}: {final_equity:.2f}%")
        return final_equity
    
    # Handle legacy notebook parameter - if notebook=True but use_notebook_plots not specified,
    # default to notebook plots for backward compatibility
    if notebook and use_notebook_plots is False:
        use_notebook_plots = True
    
    if use_notebook_plots:
        # Use matplotlib plots in notebook
        _show_notebook_plots(symbol, sb.data, buy_signals, sell_signals, equity_curve, output_mode)
    else:
        # Use dashboard (default behavior)
        if output_mode == "equity_curve":
            # Create a custom dashboard for equity curve only
            _show_equity_dashboard(symbol, sb.data, buy_signals, sell_signals, equity_curve)
        else:  # "full" mode
            # Pass our calculated equity curve to the StrategyBacktester for consistent data
            _show_full_dashboard(symbol, sb.data, buy_signals, sell_signals, equity_curve)
            
        # Save data files for dashboard usage
        historical_data.to_csv("cli_historical_data.csv")
        with open("cli_signals.json", "w") as f:
            json.dump([(x, y, "BUY") for x, y in zip(buy_signals['x'], buy_signals['y'])] + 
                     [(x, y, "SELL") for x, y in zip(sell_signals['x'], sell_signals['y'])], f)
        with open("cli_equity_curve.json", "w") as f:
            json.dump(equity_curve, f)
    
    return final_equity

def _show_multi_symbol_comparison(results, symbols, output_mode, use_notebook_plots):
    """Show comparison chart and summary for multiple symbols."""
    print("\n" + "=" * 60)
    print("📊 MULTI-SYMBOL COMPARISON RESULTS")
    print("=" * 60)
    
    # Sort results by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    # Print summary table
    print(f"{'Rank':<6} {'Symbol':<8} {'Equity Change':<15} {'Performance'}")
    print("-" * 50)
    
    for i, (symbol, equity_change) in enumerate(sorted_results, 1):
        performance = "🟢 Profit" if equity_change > 0 else "🔴 Loss" if equity_change < 0 else "⚪ Flat"
        print(f"{i:<6} {symbol:<8} {equity_change:>+8.2f}%        {performance}")
    
    # Calculate summary stats
    equity_values = list(results.values())
    avg_return = sum(equity_values) / len(equity_values)
    profitable_count = sum(1 for eq in equity_values if eq > 0)
    
    print("-" * 50)
    print(f"📈 Average Return: {avg_return:+.2f}%")
    print(f"🎯 Profitable Stocks: {profitable_count}/{len(symbols)} ({profitable_count/len(symbols)*100:.1f}%)")
    print(f"🏆 Best Performer: {sorted_results[0][0]} ({sorted_results[0][1]:+.2f}%)")
    print(f"📉 Worst Performer: {sorted_results[-1][0]} ({sorted_results[-1][1]:+.2f}%)")
    
    # Show comparison chart if requested
    if output_mode != "equity_only":
        _show_comparison_chart(results, symbols, use_notebook_plots)

def _show_comparison_chart(results, symbols, use_notebook_plots):
    """Show comparison bar chart of results."""
    if use_notebook_plots:
        # Matplotlib version
        import matplotlib.pyplot as plt
        
        symbols_list = list(results.keys())
        equity_changes = list(results.values())
        
        plt.figure(figsize=(12, 6))
        
        # Color bars based on performance
        colors = ['green' if eq > 0 else 'red' if eq < 0 else 'gray' for eq in equity_changes]
        
        bars = plt.bar(symbols_list, equity_changes, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, equity_changes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                    f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Multi-Symbol Strategy Performance Comparison', fontsize=14)
        plt.xlabel('Stock Symbol', fontsize=12)
        plt.ylabel('Equity Change (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    else:
        # Dashboard version - just print message
        print("\n📊 For interactive comparison chart, use use_notebook_plots=True")
        print("   or view individual symbol dashboards above.")

def _calculate_backtest_data(strategy, data):
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
        
        # Update running equity curve
        if strategy.bought and entry_price is not None:
            current_return = (close_price - entry_price) / entry_price * 100
            equity_curve.append(cumulative_return + current_return)
        else:
            equity_curve.append(cumulative_return)
    
    return buy_signals, sell_signals, equity_curve

def _show_full_dashboard(symbol, data, buy_signals, sell_signals, equity_curve):
    """Helper function to show full analysis dashboard with consistent equity curve."""
    try:
        import dash
        from dash import dcc, html
        import plotly.graph_objs as go
        import threading
        
        def launch_full_dashboard():
            x_vals = list(range(len(data)))
            buy_points = [(i, buy_signals['y'][j]) for j, i in enumerate(buy_signals['x'])]
            sell_points = [(i, sell_signals['y'][j]) for j, i in enumerate(sell_signals['x'])]
            
            from dash.dependencies import Input, Output, State
            app = dash.Dash(__name__)
            
            # Stock price chart with signals and Bollinger Bands
            price_traces = [
                go.Scatter(x=x_vals, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue'))
            ]
            
            # Add Bollinger Bands if available
            if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
                price_traces.extend([
                    go.Scatter(x=x_vals, y=data['BB_upper'], mode='lines', name='BB Upper', 
                              line=dict(color='gray', dash='dash'), opacity=0.7),
                    go.Scatter(x=x_vals, y=data['BB_lower'], mode='lines', name='BB Lower', 
                              line=dict(color='gray', dash='dash'), opacity=0.7),
                ])
            
            # Add EMA lines if available
            if 'EMA_fast' in data.columns:
                price_traces.append(go.Scatter(x=x_vals, y=data['EMA_fast'], mode='lines', 
                                             name='EMA Fast', line=dict(dash='dash', color='orange')))
            if 'EMA_slow' in data.columns:
                price_traces.append(go.Scatter(x=x_vals, y=data['EMA_slow'], mode='lines', 
                                             name='EMA Slow', line=dict(dash='dash', color='purple')))
            
            # Add buy/sell signals
            if buy_points:
                price_traces.append(go.Scatter(x=[x for x, _ in buy_points], y=[y for _, y in buy_points], 
                                             mode='markers', name='BUY', 
                                             marker=dict(color='green', symbol='circle', size=10)))
            if sell_points:
                price_traces.append(go.Scatter(x=[x for x, _ in sell_points], y=[y for _, y in sell_points], 
                                             mode='markers', name='SELL', 
                                             marker=dict(color='red', symbol='x', size=10)))
            
            equity_curve_fig = {
                'data': price_traces,
                'layout': go.Layout(title=f'{symbol} - Price with Signals & Indicators', 
                                   xaxis={'title': 'Index'}, yaxis={'title': 'Price'})
            }
            
            # RSI chart
            rsi_fig = {
                'data': [
                    go.Scatter(x=x_vals, y=data['RSI'], mode='lines', name='RSI', line=dict(color='blue')),
                ],
                'layout': go.Layout(title='RSI', xaxis={'title': 'Index'}, yaxis={'title': 'RSI'})
            }
            
            # Equity curve chart - using our calculated equity curve
            equity_fig = {
                'data': [
                    go.Scatter(x=x_vals, y=equity_curve, mode='lines', 
                              name='Cumulative % Gain/Loss', line=dict(color='purple', width=3)),
                ],
                'layout': go.Layout(title=f'{symbol} - Equity Change Over Time', 
                                   xaxis={'title': 'Time Index'}, yaxis={'title': 'Cumulative %'})
            }
            
            # MACD chart
            macd_traces = []
            if 'MACD' in data.columns:
                macd_traces.append(go.Scatter(x=x_vals, y=data['MACD'], mode='lines', 
                                            name='MACD', line=dict(color='blue')))
            if 'MACD_signal' in data.columns:
                macd_traces.append(go.Scatter(x=x_vals, y=data['MACD_signal'], mode='lines', 
                                            name='MACD Signal', line=dict(color='red')))
            if 'MACD_histogram' in data.columns:
                macd_traces.append(go.Bar(x=x_vals, y=data['MACD_histogram'], 
                                        name='MACD Histogram', opacity=0.6))
            
            macd_fig = {
                'data': macd_traces,
                'layout': go.Layout(title='MACD Indicator', xaxis={'title': 'Index'}, yaxis={'title': 'MACD'})
            }
            
            app.layout = html.Div([
                html.H1(f'{symbol} - Comprehensive Trading Analysis Dashboard'),
                html.P(f"Final Equity Change: {equity_curve[-1]:.2f}%" if equity_curve else "No data"),
                dcc.Graph(id='price-chart', figure=equity_curve_fig),
                dcc.Graph(id='equity', figure=equity_fig),
                dcc.Graph(id='rsi', figure=rsi_fig),
                dcc.Graph(id='macd', figure=macd_fig)
            ])
            
            # Synchronized zooming
            import copy
            @app.callback(
                [Output('equity', 'figure'), Output('rsi', 'figure'), Output('macd', 'figure')],
                [Input('price-chart', 'relayoutData')],
                [State('equity', 'figure'), State('rsi', 'figure'), State('macd', 'figure')]
            )
            def sync_xaxis(relayoutData, equity_fig, rsi_fig, macd_fig):
                equity_fig = copy.deepcopy(equity_fig)
                rsi_fig = copy.deepcopy(rsi_fig)
                macd_fig = copy.deepcopy(macd_fig)
                if relayoutData:
                    if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
                        x0 = relayoutData['xaxis.range[0]']
                        x1 = relayoutData['xaxis.range[1]']
                        equity_fig['layout']['xaxis']['range'] = [x0, x1]
                        rsi_fig['layout']['xaxis']['range'] = [x0, x1]
                        macd_fig['layout']['xaxis']['range'] = [x0, x1]
                    elif 'xaxis.autorange' in relayoutData and relayoutData['xaxis.autorange']:
                        equity_fig['layout']['xaxis'].pop('range', None)
                        rsi_fig['layout']['xaxis'].pop('range', None)
                        macd_fig['layout']['xaxis'].pop('range', None)
                return equity_fig, rsi_fig, macd_fig
            
            app.run(debug=False)
        
        threading.Thread(target=launch_full_dashboard, daemon=True).start()
        print("Full analysis dashboard running at http://127.0.0.1:8050/. Press Ctrl+C to exit.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting full dashboard...")
            
    except ImportError:
        print('Dash or Plotly is not installed. Please install with "pip install dash plotly".')
    except Exception as e:
        print(f'Error launching full dashboard: {e}')

def _show_equity_dashboard(symbol, data, buy_signals, sell_signals, equity_curve):
    """Helper function to show equity-only dashboard."""
    try:
        import dash
        from dash import dcc, html
        import plotly.graph_objs as go
        import threading
        
        def launch_equity_dashboard():
            x_vals = list(range(len(data)))
            
            app = dash.Dash(__name__)
            
            equity_fig = {
                'data': [
                    go.Scatter(x=x_vals, y=equity_curve, mode='lines', 
                              name='Equity Curve', line=dict(color='purple', width=3)),
                ],
                'layout': go.Layout(
                    title=f'{symbol} - Equity Change Over Time',
                    xaxis={'title': 'Time Index'},
                    yaxis={'title': 'Cumulative % Gain/Loss'},
                    height=600
                )
            }
            
            app.layout = html.Div([
                html.H1(f'{symbol} Strategy Analysis - Equity Curve'),
                html.P(f"Final Equity Change: {equity_curve[-1]:.2f}%" if equity_curve else "No data"),
                dcc.Graph(id='equity-curve', figure=equity_fig)
            ])
            
            app.run(debug=False)
        
        threading.Thread(target=launch_equity_dashboard, daemon=True).start()
        print("Equity dashboard running at http://127.0.0.1:8050/. Press Ctrl+C to exit.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting equity dashboard...")
            
    except ImportError:
        print('Dash or Plotly is not installed. Please install with "pip install dash plotly".')
    except Exception as e:
        print(f'Error launching equity dashboard: {e}')

def _show_notebook_plots(symbol, data, buy_signals, sell_signals, equity_curve, output_mode):
    """Helper function to show plots in notebook environment."""
    import matplotlib.pyplot as plt
    
    x_vals = list(range(len(data)))
    
    if output_mode == "equity_curve":
        # Show only equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, equity_curve, label='Equity Curve', color='blue', linewidth=2)
        plt.title(f'{symbol} - Equity Change Over Time', fontsize=14)
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('Cumulative % Gain/Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    elif output_mode == "full":
        # Show comprehensive analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Stock price with signals and bollinger bands
        ax1.plot(x_vals, data['Close'], label='Close Price', color='blue', linewidth=1.5)
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
            ax1.plot(x_vals, data['BB_upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.7)
            ax1.plot(x_vals, data['BB_lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.7)
            ax1.fill_between(x_vals, data['BB_upper'], data['BB_lower'], alpha=0.1, color='gray')
        ax1.scatter(buy_signals['x'], buy_signals['y'], color='green', marker='^', s=100, label='BUY', zorder=5)
        ax1.scatter(sell_signals['x'], sell_signals['y'], color='red', marker='v', s=100, label='SELL', zorder=5)
        ax1.set_title(f'{symbol} - Price with Signals & Bollinger Bands', fontsize=12)
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Equity curve
        ax2.plot(x_vals, equity_curve, label='Equity Curve', color='purple', linewidth=2)
        ax2.set_title(f'{symbol} - Equity Change Over Time', fontsize=12)
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Cumulative % Gain/Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. MACD
        if 'MACD' in data.columns:
            ax3.plot(x_vals, data['MACD'], label='MACD', color='blue', linewidth=1.5)
            if 'MACD_signal' in data.columns:
                ax3.plot(x_vals, data['MACD_signal'], label='MACD Signal', color='red', linewidth=1.5)
            if 'MACD_histogram' in data.columns:
                ax3.bar(x_vals, data['MACD_histogram'], label='MACD Histogram', alpha=0.6, color='gray')
        ax3.set_title('MACD Indicator', fontsize=12)
        ax3.set_xlabel('Time Index')
        ax3.set_ylabel('MACD Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ROC
        if 'ROC' in data.columns:
            ax4.plot(x_vals, data['ROC'], label='Rate of Change', color='orange', linewidth=1.5)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('Rate of Change (ROC)', fontsize=12)
        ax4.set_xlabel('Time Index')
        ax4.set_ylabel('ROC Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

@cli.command()
@click.argument("symbol")  # Can be a single symbol or comma-separated list
@click.option("--strategy", default="default", help="Strategy name or path to strategy file")
@click.option("--strategy-class", help="Strategy class name (when using custom strategy file)")
@click.option("--macd-fast", default=12, help="MACD fast period")
@click.option("--macd-slow", default=26, help="MACD slow period")
@click.option("--macd-signal", default=9, help="MACD signal period")
@click.option("--rsi-period", default=14, help="RSI period")
@click.option("--roc-period", default=5, help="ROC period")
@click.option("--ema-mom-period", default=20, help="EMA momentum period")
@click.option("--period", default="60d", help="Data period")
@click.option("--interval", default="15m", help="Data interval")
@click.option("--final-equity-only", is_flag=True, help="Return only final equity (legacy parameter)")
@click.option("--plot/--no-plot", default=True, help="Show plots")
@click.option("--output-mode", 
              type=click.Choice(['equity_only', 'equity_curve', 'full']), 
              default='full',
              help="Output mode: equity_only (just final %), equity_curve (equity over time), full (all indicators)")
@click.option("--use-notebook-plots", is_flag=True, help="Use matplotlib plots instead of dashboard")
def run_back_strategy(
    symbol,
    strategy,
    strategy_class,
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
    output_mode,
    use_notebook_plots
):
    """CLI wrapper for run_back_strategy_core. Symbol can be a single symbol or comma-separated list (e.g., 'AAPL,MSFT,GOOGL')."""
    # Handle legacy parameter
    if final_equity_only:
        output_mode = "equity_only"
    
    # Parse symbol - could be single symbol or comma-separated list
    if ',' in symbol:
        symbols = [s.strip().upper() for s in symbol.split(',')]
    else:
        symbols = symbol.strip().upper()
    
    return run_back_strategy_core(
        symbols,  # Now supports both single symbol and list
        strategy,
        strategy_class,
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
        notebook=False,
        output_mode=output_mode,
        use_notebook_plots=use_notebook_plots
    )

@cli.command()
@click.argument("symbol")
@click.option("--macd-fast-min", default=8, help="MACD fast period minimum")
@click.option("--macd-fast-max", default=16, help="MACD fast period maximum")
@click.option("--macd-slow-min", default=20, help="MACD slow period minimum")
@click.option("--macd-slow-max", default=30, help="MACD slow period maximum")
@click.option("--macd-signal-min", default=7, help="MACD signal period minimum")
@click.option("--macd-signal-max", default=11, help="MACD signal period maximum")
@click.option("--rsi-period-min", default=12, help="RSI period minimum")
@click.option("--rsi-period-max", default=16, help="RSI period maximum")
@click.option("--roc-period-min", default=3, help="ROC period minimum")
@click.option("--roc-period-max", default=7, help="ROC period maximum")
@click.option("--ema-mom-period-min", default=15, help="EMA momentum period minimum")
@click.option("--ema-mom-period-max", default=25, help="EMA momentum period maximum")
@click.option("--method", 
              type=click.Choice(['brute', 'differential_evolution']), 
              default='brute',
              help="Optimization method")
@click.option("--objective",
              type=click.Choice(['final_equity', 'consistency', 'combined']),
              default='combined',
              help="Optimization objective")
@click.option("--equity-weight", default=0.6, help="Weight for equity in combined objective")
@click.option("--consistency-weight", default=0.4, help="Weight for consistency in combined objective")
@click.option("--output-mode", 
              type=click.Choice(['params_only', 'equity_only', 'full']), 
              default='full',
              help="Output mode: params_only (just parameters), equity_only (parameters + equity), full (complete analysis)")
@click.option("--use-notebook-plots", is_flag=True, help="Use matplotlib plots instead of dashboard")
@click.option("--steps", default=3, help="Number of steps to test in each parameter range")
@click.option("--period", default="60d", help="Data period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')")
@click.option("--interval", default="15m", help="Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')")
def optimize_parameters(
    symbol,
    macd_fast_min,
    macd_fast_max,
    macd_slow_min,
    macd_slow_max,
    macd_signal_min,
    macd_signal_max,
    rsi_period_min,
    rsi_period_max,
    roc_period_min,
    roc_period_max,
    ema_mom_period_min,
    ema_mom_period_max,
    method,
    objective,
    equity_weight,
    consistency_weight,
    output_mode,
    use_notebook_plots,
    steps,
    period,
    interval
):
    """Optimize strategy parameters for a given symbol. Example: python -m src.stock_bot optimize-parameters AAPL --steps 3 --output-mode full"""
    
    return run_optimization_core(
        symbol=symbol,
        macd_fast_range=(macd_fast_min, macd_fast_max),
        macd_slow_range=(macd_slow_min, macd_slow_max),
        macd_signal_range=(macd_signal_min, macd_signal_max),
        rsi_period_range=(rsi_period_min, rsi_period_max),
        roc_period_range=(roc_period_min, roc_period_max),
        ema_mom_period_range=(ema_mom_period_min, ema_mom_period_max),
        method=method,
        objective=objective,
        max_equity_weight=equity_weight,
        consistency_weight=consistency_weight,
        output_mode=output_mode,
        use_notebook_plots=use_notebook_plots,
        steps=steps,
        period=period,
        interval=interval
    )

def run_optimization_core(
    symbol,
    macd_fast_range,
    macd_slow_range, 
    macd_signal_range,
    rsi_period_range,
    roc_period_range,
    ema_mom_period_range,
    method="brute",
    objective="combined",
    max_equity_weight=0.6,
    consistency_weight=0.4,
    output_mode="full",
    use_notebook_plots=False,
    steps=3,
    period="60d",
    interval="15m"
):
    """
    Run parameter optimization with the given ranges and display results.
    
    Args:
        symbol: Stock symbol to optimize
        *_range: Parameter ranges as (min, max) tuples
        method: Optimization method ('brute' or 'differential_evolution')
        objective: Optimization objective ('final_equity', 'consistency', or 'combined')
        max_equity_weight: Weight for equity in combined objective
        consistency_weight: Weight for consistency in combined objective
        output_mode: Controls what is displayed/returned
            - "equity_only": Return only the optimized parameters and final equity
            - "params_only": Return only the optimized parameters
            - "full": Show comprehensive optimization analysis
        use_notebook_plots: If True, use matplotlib plots in notebook. If False (default), use dashboard
        steps: Number of steps to test in each parameter range
        period: Data period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
    Returns:
        dict: Optimization results including best parameters and performance metrics
    """
    
    # Fetch data for the symbol
    ticker = yf.Ticker(symbol)
    historical_data = ticker.history(period=period, interval=interval)
    
    if historical_data.empty:
        print(f"❌ No data available for symbol {symbol}")
        return {}
    
    # Create strategy and backtester
    try:
        # Load the strategy
        strategy_cls = load_strategy("default")  # Use default for optimization
    except Exception as e:
        print(f"Error loading strategy: {e}")
        return {}
    
    strategy = strategy_cls(historical_data)
    backtester = StrategyBacktester(strategy)
    
    # Prepare parameter ranges
    param_ranges = {
        'macd_fast': (macd_fast_range[0], macd_fast_range[1], steps),
        'macd_slow': (macd_slow_range[0], macd_slow_range[1], steps),
        'macd_signal': (macd_signal_range[0], macd_signal_range[1], steps),
        'rsi_period': (rsi_period_range[0], rsi_period_range[1], steps),
        'roc_period': (roc_period_range[0], roc_period_range[1], steps),
        'ema_mom_period': (ema_mom_period_range[0], ema_mom_period_range[1], steps)
    }
    
    total_combinations = steps ** 6
    print(f"🔍 Optimizing {symbol} with {total_combinations} parameter combinations...")
    
    # Run optimization
    results = backtester.optimize_parameters(
        symbol=symbol,
        param_ranges=param_ranges,
        method=method,
        objective=objective,
        max_equity_weight=max_equity_weight,
        consistency_weight=consistency_weight,
        verbose=(output_mode == "full")
    )
    
    # Handle different output modes
    if output_mode == "params_only":
        print(f"✅ Optimized Parameters for {symbol}: {results['best_params']}")
        return results['best_params']
    
    elif output_mode == "equity_only":
        print(f"✅ {symbol} Optimization Results:")
        print(f"   Best Parameters: {results['best_params']}")
        print(f"   Optimized Final Equity: {results['final_equity']:.2f}%")
        return {
            'symbol': symbol,
            'best_params': results['best_params'],
            'final_equity': results['final_equity']
        }
    
    else:  # "full" mode
        _show_optimization_analysis(symbol, results, param_ranges, use_notebook_plots)
        return results

def _show_optimization_analysis(symbol, results, param_ranges, use_notebook_plots):
    """Show comprehensive optimization analysis."""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print(f"\n🎯 OPTIMIZATION ANALYSIS FOR {symbol}")
    print("=" * 60)
    print(f"✅ Best Parameters: {results['best_params']}")
    print(f"📈 Final Equity: {results['final_equity']:.2f}%")
    print(f"🎯 Consistency Score: {results['consistency_score']:.4f}")
    print(f"⭐ Combined Score: {results['best_score']:.2f}")
    print(f"🔬 Method: {results['method'].title()}")
    print(f"🎚️  Objective: {results['objective'].title()}")
    
    if not results['all_results']:
        print("❌ No detailed results available for visualization")
        return
    
    # Create DataFrame for analysis
    results_df = pd.DataFrame([
        {
            **result['params'],
            'final_equity': result['final_equity'],
            'consistency_score': result['consistency_score'],
            'combined_score': result['combined_score']
        }
        for result in results['all_results']
    ])
    
    print(f"\n📊 OPTIMIZATION STATISTICS:")
    print(f"   Total Combinations Tested: {len(results_df)}")
    print(f"   Best Final Equity: {results_df['final_equity'].max():.2f}%")
    print(f"   Worst Final Equity: {results_df['final_equity'].min():.2f}%")
    print(f"   Mean Final Equity: {results_df['final_equity'].mean():.2f}%")
    print(f"   Equity Std Dev: {results_df['final_equity'].std():.2f}%")
    
    # Show top performers
    print(f"\n🏆 TOP 5 PARAMETER COMBINATIONS:")
    top_5 = results_df.nlargest(5, 'final_equity')[['macd_fast', 'macd_slow', 'rsi_period', 'final_equity', 'consistency_score']]
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. MACD({row['macd_fast']},{row['macd_slow']}) RSI({row['rsi_period']}) → {row['final_equity']:.2f}% (consistency: {row['consistency_score']:.3f})")
    
    if use_notebook_plots:
        # Create parameter sensitivity plots
        _show_optimization_plots(symbol, results_df, param_ranges)
    else:
        # Create dashboard
        _show_optimization_dashboard(symbol, results_df, param_ranges, results)

def _show_optimization_plots(symbol, results_df, param_ranges):
    """Show parameter sensitivity plots in notebook."""
    import matplotlib.pyplot as plt
    
    print(f"\n📈 Parameter Sensitivity Analysis")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    param_names = list(param_ranges.keys())
    
    for i, param in enumerate(param_names):
        # Group by parameter value and calculate statistics
        param_impact = results_df.groupby(param)['final_equity'].agg(['mean', 'std', 'count']).reset_index()
        
        axes[i].bar(param_impact[param], param_impact['mean'], 
                   yerr=param_impact['std'], capsize=5, alpha=0.7, color='steelblue')
        axes[i].set_title(f'{param.upper()} Impact on Final Equity', fontweight='bold')
        axes[i].set_xlabel(param.replace('_', ' ').title())
        axes[i].set_ylabel('Mean Final Equity (%)')
        axes[i].grid(True, alpha=0.3)
        
        # Highlight the best value
        best_value = results_df.loc[results_df['final_equity'].idxmax(), param]
        best_idx = param_impact[param_impact[param] == best_value].index[0]
        axes[i].bar(param_impact.iloc[best_idx][param], param_impact.iloc[best_idx]['mean'], 
                   color='gold', alpha=0.8, label='Optimal')
        axes[i].legend()
    
    plt.suptitle(f'{symbol} - Parameter Optimization Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def _show_optimization_dashboard(symbol, results_df, param_ranges, results):
    """Show optimization results in interactive dashboard."""
    try:
        import dash
        from dash import dcc, html, Input, Output, dash_table
        import plotly.graph_objs as go
        import plotly.express as px
        import threading
        import webbrowser
        import time
        import pandas as pd
        
        def launch_optimization_dashboard():
            app = dash.Dash(__name__)
            
            # Prepare data for visualization
            param_names = list(param_ranges.keys())
            
            # Parameter sensitivity data
            sensitivity_data = []
            for param in param_names:
                param_impact = results_df.groupby(param)['final_equity'].agg(['mean', 'std', 'count']).reset_index()
                for _, row in param_impact.iterrows():
                    sensitivity_data.append({
                        'parameter': param,
                        'value': row[param],
                        'mean_equity': row['mean'],
                        'std_equity': row['std'],
                        'count': row['count']
                    })
            
            sensitivity_df = pd.DataFrame(sensitivity_data)
            
            # Create parameter sensitivity plots
            sensitivity_figs = []
            for param in param_names:
                param_data = sensitivity_df[sensitivity_df['parameter'] == param]
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=param_data['value'],
                    y=param_data['mean_equity'],
                    error_y=dict(type='data', array=param_data['std_equity']),
                    name=f'{param} Impact',
                    marker_color='steelblue'
                ))
                
                # Highlight optimal value
                best_value = results['best_params'][param]
                best_data = param_data[param_data['value'] == best_value]
                if not best_data.empty:
                    fig.add_trace(go.Bar(
                        x=[best_value],
                        y=best_data['mean_equity'].iloc[0],
                        name='Optimal',
                        marker_color='gold'
                    ))
                
                fig.update_layout(
                    title=f'{param.upper()} Impact on Final Equity',
                    xaxis_title=param.replace('_', ' ').title(),
                    yaxis_title='Mean Final Equity (%)',
                    showlegend=True
                )
                
                sensitivity_figs.append(dcc.Graph(figure=fig))
            
            # Performance distribution
            performance_fig = px.histogram(
                results_df, 
                x='final_equity', 
                nbins=20,
                title='Performance Distribution',
                labels={'final_equity': 'Final Equity (%)', 'count': 'Number of Combinations'}
            )
            performance_fig.add_vline(
                x=results['final_equity'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Optimal: {results['final_equity']:.2f}%"
            )
            
            # Top combinations table
            top_10 = results_df.nlargest(10, 'final_equity')[
                ['macd_fast', 'macd_slow', 'macd_signal', 'rsi_period', 'roc_period', 'ema_mom_period', 'final_equity', 'consistency_score']
            ].round(4)
            
            app.layout = html.Div([
                html.H1(f'{symbol} - Parameter Optimization Dashboard', 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
                
                # Summary cards
                html.Div([
                    html.Div([
                        html.H3(f"{results['final_equity']:.2f}%", style={'color': '#27ae60', 'margin': 0}),
                        html.P("Optimized Final Equity", style={'margin': 0})
                    ], className='summary-card', style={'background': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        html.H3(f"{results['consistency_score']:.4f}", style={'color': '#3498db', 'margin': 0}),
                        html.P("Consistency Score", style={'margin': 0})
                    ], className='summary-card', style={'background': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        html.H3(f"{len(results_df)}", style={'color': '#e74c3c', 'margin': 0}),
                        html.P("Combinations Tested", style={'margin': 0})
                    ], className='summary-card', style={'background': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                    
                    html.Div([
                        html.H3(f"{results['method'].title()}", style={'color': '#9b59b6', 'margin': 0}),
                        html.P("Optimization Method", style={'margin': 0})
                    ], className='summary-card', style={'background': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '23%', 'display': 'inline-block', 'margin': '1%'}),
                ], style={'marginBottom': 30}),
                
                # Performance distribution
                html.Div([
                    dcc.Graph(figure=performance_fig)
                ], style={'marginBottom': 30}),
                
                # Parameter sensitivity plots
                html.H2('Parameter Sensitivity Analysis', style={'color': '#2c3e50', 'marginBottom': 20}),
                html.Div(sensitivity_figs[:3], style={'display': 'flex', 'flexWrap': 'wrap'}),
                html.Div(sensitivity_figs[3:], style={'display': 'flex', 'flexWrap': 'wrap'}),
                
                # Top combinations table
                html.H2('Top 10 Parameter Combinations', style={'color': '#2c3e50', 'marginTop': 30, 'marginBottom': 20}),
                dash_table.DataTable(
                    data=top_10.to_dict('records'),
                    columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in top_10.columns],
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 0},
                            'backgroundColor': '#f1c40f',
                            'color': 'black',
                        }
                    ]
                ),
                
                # Optimal parameters display
                html.H2('Optimal Parameters', style={'color': '#2c3e50', 'marginTop': 30, 'marginBottom': 20}),
                html.Div([
                    html.P(f"{param.replace('_', ' ').title()}: {value}", 
                          style={'fontSize': '18px', 'margin': '10px 0'})
                    for param, value in results['best_params'].items()
                ], style={'background': '#e8f6f3', 'padding': '20px', 'borderRadius': '10px', 'border': '2px solid #27ae60'})
            ])
            
            try:
                app.run(debug=False, port=8051)  # Use different port to avoid conflicts
            except Exception as e:
                print(f"❌ Dashboard error: {e}")
        
        # Launch dashboard in background thread
        dashboard_thread = threading.Thread(target=launch_optimization_dashboard, daemon=True)
        dashboard_thread.start()
        
        print(f"\n🌐 Optimization Dashboard launched at http://127.0.0.1:8051/")
        print("📊 Dashboard includes:")
        print("   • Performance summary cards")
        print("   • Parameter sensitivity analysis")
        print("   • Performance distribution histogram")
        print("   • Top 10 parameter combinations table")
        print("   • Optimal parameters display")
        
        # Open browser automatically
        time.sleep(2)
        try:
            webbrowser.open('http://127.0.0.1:8051/')
        except:
            pass
            
    except ImportError:
        print("❌ Dashboard dependencies not available. Install with: pip install dash plotly")
        # Fallback to simple text output
        print(f"\n📊 PARAMETER SENSITIVITY SUMMARY:")
        param_names = list(param_ranges.keys())
        for param in param_names:
            param_impact = results_df.groupby(param)['final_equity'].agg(['mean', 'std']).reset_index()
            best_value = results['best_params'][param]
            best_performance = param_impact[param_impact[param] == best_value]['mean'].iloc[0]
            print(f"   {param}: Optimal value = {best_value} (Performance: {best_performance:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        print("Falling back to summary display...")

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

def run_strategy_comparison_core(
    symbol,
    strategy_path,
    strategy_class,
    macd_fast,
    macd_slow,
    macd_signal,
    rsi_period,
    roc_period,
    ema_mom_period,
    period,
    interval,
    output_mode,
    use_notebook_plots
):
    """
    Compare multiple strategies on the same symbol with dashboard comparison.
    
    Args:
        symbol: Stock symbol to test
        strategy_path: Path to single strategy file or directory containing strategy files
        strategy_class: Strategy class name (when using single strategy file)
        macd_fast, macd_slow, macd_signal: MACD parameters
        rsi_period: RSI period
        roc_period: Rate of Change period
        ema_mom_period: EMA momentum period
        period: Data period (e.g., "60d")
        interval: Data interval (e.g., "15m")
        output_mode: Controls what is displayed/returned
        use_notebook_plots: If True, use matplotlib plots in notebook. If False (default), use dashboard
    
    Returns:
        dict: Results for all strategies tested
    """
    if not strategy_path:
        print("❌ Error: --strategy-path is required")
        return None
    
    # Determine if strategy_path is a file, directory, or built-in strategy names
    if strategy_path == "builtin" or strategy_path == "all":
        # Use all built-in strategies
        from .strategy_loader import get_available_strategies
        available_strategies = get_available_strategies()
        strategy_files = []
        strategy_names = []
        for name, strategy_cls in available_strategies.items():
            strategy_files.append(name)  # Use name as identifier
            strategy_names.append(name)
    elif os.path.isfile(strategy_path):
        strategy_files = [strategy_path]
        strategy_names = [os.path.splitext(os.path.basename(strategy_path))[0]]
    elif os.path.isdir(strategy_path):
        # Find all Python files in the directory
        strategy_files = glob.glob(os.path.join(strategy_path, "*.py"))
        strategy_files = [f for f in strategy_files if not f.endswith("__init__.py")]
        strategy_names = [os.path.splitext(os.path.basename(f))[0] for f in strategy_files]
    else:
        print(f"❌ Error: {strategy_path} is not a valid file, directory, or 'builtin'")
        return None
    
    if not strategy_files:
        print(f"❌ Error: No Python strategy files found in {strategy_path}")
        return None
    
    print(f"🔍 Found {len(strategy_files)} strategy files to test")
    
    # Fetch data once for all strategies
    print(f"📊 Fetching data for {symbol}...")
    data = get_data(symbol, period, interval)
    if data is None or data.empty:
        print(f"❌ Error: Could not fetch data for {symbol}")
        return None
    
    print(f"✅ Data fetched: {len(data)} data points")
    
    # Test each strategy
    results = {}
    strategy_classes = {}
    
    for i, (strategy_file, strategy_name) in enumerate(zip(strategy_files, strategy_names)):
        print(f"\n🧪 Testing strategy {i+1}/{len(strategy_files)}: {strategy_name}")
        
        try:
            # Load strategy class
            if strategy_file in ['default', 'trend_following', 'simple_trend', 'aggressive', 'mean_reversion', 'simple_mean_reversion', 'macd_crossover', 'macd_divergence', 'macd_histogram']:
                # Built-in strategy
                strategy_cls = load_strategy(strategy_file)
            elif strategy_class and i == 0:
                # Use provided class name for single file
                strategy_cls = load_strategy(strategy_file, strategy_class)
            else:
                # Try to auto-detect class name
                strategy_cls = load_strategy(strategy_file)
            
            strategy_classes[strategy_name] = strategy_cls
            
            # Create strategy instance
            strategy_params = {
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'macd_signal': macd_signal,
                'rsi_period': rsi_period,
                'roc_period': roc_period,
                'ema_mom_period': ema_mom_period
            }
            
            strategy = strategy_cls(data, **strategy_params)
            
            # Run backtest
            backtester = StrategyBacktester(strategy)
            final_equity = backtester.run(plot=False, slider=False)
            
            # Store results
            results[strategy_name] = {
                'final_equity': final_equity,
                'strategy_file': strategy_file,
                'strategy_class': strategy_cls.__name__,
                'data_points': len(data),
                'period': period,
                'interval': interval
            }
            
            print(f"   ✅ {strategy_name}: {final_equity:.2f}% return")
            
        except Exception as e:
            print(f"   ❌ {strategy_name}: Error - {e}")
            results[strategy_name] = {
                'final_equity': None,
                'error': str(e),
                'strategy_file': strategy_file
            }
    
    # Display results
    print(f"\n📊 STRATEGY COMPARISON RESULTS for {symbol}")
    print("=" * 60)
    
    # Sort by performance
    sorted_results = sorted(
        [(name, data) for name, data in results.items() if data.get('final_equity') is not None],
        key=lambda x: x[1]['final_equity'],
        reverse=True
    )
    
    for i, (name, data) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {name:20s}: {data['final_equity']:8.2f}%")
    
    # Show failed strategies
    failed_strategies = [(name, data) for name, data in results.items() if data.get('final_equity') is None]
    if failed_strategies:
        print(f"\n❌ Failed strategies:")
        for name, data in failed_strategies:
            print(f"   {name}: {data.get('error', 'Unknown error')}")
    
    # Create dashboard if not using notebook plots
    if not use_notebook_plots and output_mode == "comparison":
        try:
            _create_strategy_comparison_dashboard(symbol, results, data, strategy_classes)
        except ImportError:
            print("❌ Dashboard dependencies not available. Install with: pip install dash plotly")
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
    
    return results

def _create_strategy_comparison_dashboard(symbol, results, data, strategy_classes):
    """Create a dashboard for strategy comparison."""
    try:
        import dash
        from dash import dcc, html, Input, Output, callback
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import pandas as pd
        
        # Prepare data for dashboard
        results_df = pd.DataFrame([
            {
                'Strategy': name,
                'Return (%)': data.get('final_equity', 0),
                'Status': 'Success' if data.get('final_equity') is not None else 'Failed',
                'Error': data.get('error', '')
            }
            for name, data in results.items()
        ])
        
        # Create dashboard
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1(f"Strategy Comparison Dashboard - {symbol}", style={'textAlign': 'center'}),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H3(f"{len(results)}", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Strategies Tested", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
                html.Div([
                    html.H3(f"{len([r for r in results.values() if r.get('final_equity') is not None])}", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Successful", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
                html.Div([
                    html.H3(f"{max([r.get('final_equity', 0) for r in results.values() if r.get('final_equity') is not None]):.2f}%", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Best Return", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0'}),
            
            # Tabs
            dcc.Tabs(id="tabs", value="performance", children=[
                dcc.Tab(label="Performance Comparison", value="performance"),
                dcc.Tab(label="Strategy Details", value="details"),
                dcc.Tab(label="Price Chart", value="chart"),
            ]),
            
            html.Div(id="tab-content")
        ])
        
        @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
        def render_tab_content(active_tab):
            if active_tab == "performance":
                # Performance bar chart
                successful_results = results_df[results_df['Status'] == 'Success'].sort_values('Return (%)', ascending=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=successful_results['Strategy'],
                        x=successful_results['Return (%)'],
                        orientation='h',
                        marker_color=['#2E8B57' if x > 0 else '#DC143C' for x in successful_results['Return (%)']]
                    )
                ])
                
                fig.update_layout(
                    title=f"Strategy Performance Comparison - {symbol}",
                    xaxis_title="Return (%)",
                    yaxis_title="Strategy",
                    height=400 + len(successful_results) * 30
                )
                
                return html.Div([
                    dcc.Graph(figure=fig),
                    html.H4("Performance Summary"),
                    html.Table([
                        html.Tr([html.Th("Strategy"), html.Th("Return (%)"), html.Th("Status")])
                    ] + [
                        html.Tr([
                            html.Td(row['Strategy']),
                            html.Td(f"{row['Return (%)']:.2f}%" if row['Status'] == 'Success' else "N/A"),
                            html.Td(row['Status'], style={'color': '#2E8B57' if row['Status'] == 'Success' else '#DC143C'})
                        ]) for _, row in results_df.iterrows()
                    ], style={'width': '100%', 'border': '1px solid #ddd'})
                ])
            
            elif active_tab == "details":
                # Strategy details table
                details_data = []
                for name, data in results.items():
                    details_data.append({
                        'Strategy': name,
                        'File': os.path.basename(data.get('strategy_file', '')),
                        'Class': data.get('strategy_class', ''),
                        'Return (%)': f"{data.get('final_equity', 0):.2f}%" if data.get('final_equity') is not None else "Failed",
                        'Data Points': data.get('data_points', 0),
                        'Period': data.get('period', ''),
                        'Interval': data.get('interval', ''),
                        'Error': data.get('error', '')
                    })
                
                details_df = pd.DataFrame(details_data)
                
                return html.Div([
                    html.H4("Strategy Details"),
                    html.Table([
                        html.Tr([html.Th(col) for col in details_df.columns])
                    ] + [
                        html.Tr([html.Td(str(cell)) for cell in row]) for _, row in details_df.iterrows()
                    ], style={'width': '100%', 'border': '1px solid #ddd', 'fontSize': 12})
                ])
            
            elif active_tab == "chart":
                # Price chart with strategy signals
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=list(range(len(data))),
                    y=data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=1)
                ))
                
                # Add moving averages
                if 'SMA_20' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(data))),
                        y=data['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange', width=1, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"Price Chart - {symbol}",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=500
                )
                
                return dcc.Graph(figure=fig)
        
        # Add CSS
        app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .summary-card {
                        background: #f8f9fa;
                        border: 1px solid #dee2e6;
                        border-radius: 8px;
                        padding: 20px;
                        text-align: center;
                        min-width: 150px;
                    }
                    table {
                        border-collapse: collapse;
                        width: 100%;
                    }
                    th, td {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # Launch dashboard
        def launch_dashboard():
            try:
                app.run(debug=False, port=8052)
            except Exception as e:
                print(f"❌ Dashboard error: {e}")
        
        print(f"\n🌐 Strategy Comparison Dashboard launching at http://127.0.0.1:8052/")
        print("📊 Dashboard includes:")
        print("   • Performance comparison chart")
        print("   • Strategy details table")
        print("   • Price chart with technical indicators")
        print("   • Summary statistics")
        print("\n💡 Dashboard will stay open. Press Ctrl+C to stop.")
        
        # Open browser automatically
        time.sleep(2)
        try:
            webbrowser.open('http://127.0.0.1:8052/')
        except:
            pass
        
        # Launch dashboard in foreground (blocking)
        try:
            launch_dashboard()
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
            
    except ImportError:
        print("❌ Dashboard dependencies not available. Install with: pip install dash plotly")
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")

def run_config_test_core(config_file: str, symbol_override: Optional[str] = None, strategy_override: Optional[str] = None):
    """
    Run strategy testing using a configuration file.
    
    Args:
        config_file: Path to the YAML configuration file
        symbol_override: Override symbol from config (for single symbol testing)
        strategy_override: Override strategy from config (for single strategy testing)
    
    Returns:
        dict: Results for all symbols and strategies tested
    """
    try:
        # Load configuration
        config = load_config_file(config_file)
        print(f"✅ Loaded configuration from {config_file}")
        
        # Override symbols if specified
        symbols = config['symbols']
        if symbol_override:
            symbols = [symbol_override]
            print(f"🔧 Overriding symbols with: {symbol_override}")
        
        # Override strategies if specified
        strategies = config['strategies']
        if strategy_override:
            strategies = {'builtin': [strategy_override]}
            print(f"🔧 Overriding strategies with: {strategy_override}")
        
        # Limit symbols if specified
        max_symbols = config['test']['max_symbols']
        if max_symbols > 0 and len(symbols) > max_symbols:
            symbols = symbols[:max_symbols]
            print(f"🔧 Limited to first {max_symbols} symbols")
        
        # Get data configuration
        data_config = config['data']
        period = data_config['period']
        interval = data_config['interval']
        
        # Get strategy parameters
        strategy_params = config['strategy_params']
        
        # Get output configuration
        output_config = config['output']
        output_mode = output_config['mode']
        use_notebook_plots = output_config['use_notebook_plots']
        
        print(f"📊 Testing {len(symbols)} symbols with {len(strategies.get('builtin', []))} built-in strategies")
        print(f"📅 Period: {period}, Interval: {interval}")
        
        # Test each symbol
        all_results = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n{'='*60}")
            print(f"🧪 Testing symbol {i}/{len(symbols)}: {symbol}")
            print(f"{'='*60}")
            
            try:
                # Test built-in strategies
                if 'builtin' in strategies:
                    for strategy_name in strategies['builtin']:
                        print(f"\n🔍 Testing {strategy_name} on {symbol}")
                        
                        try:
                            # Run single strategy test
                            result = run_back_strategy_core(
                                symbol=symbol,
                                strategy=strategy_name,
                                strategy_class=None,
                                macd_fast=strategy_params.get('macd_fast', 12),
                                macd_slow=strategy_params.get('macd_slow', 26),
                                macd_signal=strategy_params.get('macd_signal', 9),
                                rsi_period=strategy_params.get('rsi_period', 14),
                                roc_period=strategy_params.get('roc_period', 5),
                                ema_mom_period=strategy_params.get('ema_mom_period', 20),
                                period=period,
                                interval=interval,
                                final_equity_only=False,
                                plot=False,
                                notebook=False,
                                output_mode="equity_only",
                                use_notebook_plots=False
                            )
                            
                            # Store result
                            if symbol not in all_results:
                                all_results[symbol] = {}
                            all_results[symbol][strategy_name] = {
                                'return': result,
                                'status': 'success'
                            }
                            
                            print(f"   ✅ {strategy_name}: {result:.2f}% return")
                            
                        except Exception as e:
                            print(f"   ❌ {strategy_name}: Error - {e}")
                            if symbol not in all_results:
                                all_results[symbol] = {}
                            all_results[symbol][strategy_name] = {
                                'return': None,
                                'status': 'error',
                                'error': str(e)
                            }
                
                # Test custom strategies
                if 'custom' in strategies:
                    for custom_strategy in strategies['custom']:
                        strategy_path = custom_strategy['path']
                        strategy_class = custom_strategy.get('class_name')
                        strategy_name = os.path.splitext(os.path.basename(strategy_path))[0]
                        
                        print(f"\n🔍 Testing custom strategy {strategy_name} on {symbol}")
                        
                        try:
                            # Run custom strategy test
                            result = run_back_strategy_core(
                                symbol=symbol,
                                strategy=strategy_path,
                                strategy_class=strategy_class,
                                macd_fast=strategy_params.get('macd_fast', 12),
                                macd_slow=strategy_params.get('macd_slow', 26),
                                macd_signal=strategy_params.get('macd_signal', 9),
                                rsi_period=strategy_params.get('rsi_period', 14),
                                roc_period=strategy_params.get('roc_period', 5),
                                ema_mom_period=strategy_params.get('ema_mom_period', 20),
                                period=period,
                                interval=interval,
                                final_equity_only=False,
                                plot=False,
                                notebook=False,
                                output_mode="equity_only",
                                use_notebook_plots=False
                            )
                            
                            # Store result
                            if symbol not in all_results:
                                all_results[symbol] = {}
                            all_results[symbol][strategy_name] = {
                                'return': result,
                                'status': 'success'
                            }
                            
                            print(f"   ✅ {strategy_name}: {result:.2f}% return")
                            
                        except Exception as e:
                            print(f"   ❌ {strategy_name}: Error - {e}")
                            if symbol not in all_results:
                                all_results[symbol] = {}
                            all_results[symbol][strategy_name] = {
                                'return': None,
                                'status': 'error',
                                'error': str(e)
                            }
            
            except Exception as e:
                print(f"❌ Error testing symbol {symbol}: {e}")
                all_results[symbol] = {'error': str(e)}
        
        # Display summary results
        print(f"\n{'='*80}")
        print(f"📊 CONFIG TEST SUMMARY")
        print(f"{'='*80}")
        
        # Create summary table
        all_strategies = set()
        for symbol_results in all_results.values():
            if isinstance(symbol_results, dict):
                all_strategies.update(symbol_results.keys())
        
        all_strategies = sorted(list(all_strategies))
        
        if all_strategies:
            # Print header
            print(f"{'Symbol':<10}", end="")
            for strategy in all_strategies:
                print(f"{strategy:<15}", end="")
            print()
            print("-" * (10 + 15 * len(all_strategies)))
            
            # Print results for each symbol
            for symbol, symbol_results in all_results.items():
                if isinstance(symbol_results, dict) and 'error' not in symbol_results:
                    print(f"{symbol:<10}", end="")
                    for strategy in all_strategies:
                        if strategy in symbol_results:
                            result = symbol_results[strategy]
                            if result['status'] == 'success' and result['return'] is not None:
                                print(f"{result['return']:>14.2f}%", end="")
                            else:
                                print(f"{'ERROR':<15}", end="")
                        else:
                            print(f"{'N/A':<15}", end="")
                    print()
                else:
                    print(f"{symbol:<10} {'ERROR':<15}")
        
        # Create dashboard if requested
        if output_config.get('mode') == 'comparison' and not output_config.get('use_notebook_plots', False):
            try:
                _create_multi_symbol_dashboard(symbols, all_results, output_config)
            except ImportError:
                print("❌ Dashboard dependencies not available. Install with: pip install dash plotly")
            except Exception as e:
                print(f"❌ Error creating dashboard: {e}")
        
        # Save results if requested
        if output_config.get('save_results', False) or config['test'].get('save_results', False):
            results_file = output_config.get('results_file', 'strategy_comparison_results.json')
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n💾 Results saved to {results_file}")
        
        return all_results
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML config file: {e}")
        return None
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def _create_multi_symbol_dashboard(symbols, results, output_config):
    """Create a comprehensive dashboard for multi-symbol, multi-strategy results."""
    try:
        import dash
        from dash import dcc, html, Input, Output, callback
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        import pandas as pd
        
        # Prepare data for dashboard
        dashboard_data = []
        for symbol, symbol_results in results.items():
            if isinstance(symbol_results, dict) and 'error' not in symbol_results:
                for strategy, strategy_data in symbol_results.items():
                    if strategy_data['status'] == 'success' and strategy_data['return'] is not None:
                        dashboard_data.append({
                            'Symbol': symbol,
                            'Strategy': strategy,
                            'Return (%)': strategy_data['return'],
                            'Status': 'Success'
                        })
                    else:
                        dashboard_data.append({
                            'Symbol': symbol,
                            'Strategy': strategy,
                            'Return (%)': 0,
                            'Status': 'Error'
                        })
        
        if not dashboard_data:
            print("❌ No valid results to display in dashboard")
            return
        
        results_df = pd.DataFrame(dashboard_data)
        
        # Create dashboard
        app = dash.Dash(__name__)
        
        # Calculate summary statistics
        total_tests = len(results_df)
        successful_tests = len(results_df[results_df['Status'] == 'Success'])
        avg_return = results_df[results_df['Status'] == 'Success']['Return (%)'].mean() if successful_tests > 0 else 0
        best_return = results_df[results_df['Status'] == 'Success']['Return (%)'].max() if successful_tests > 0 else 0
        worst_return = results_df[results_df['Status'] == 'Success']['Return (%)'].min() if successful_tests > 0 else 0
        
        app.layout = html.Div([
            html.H1("Multi-Symbol Strategy Analysis Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H3(f"{len(symbols)}", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Symbols Tested", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
                html.Div([
                    html.H3(f"{total_tests}", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Total Tests", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
                html.Div([
                    html.H3(f"{successful_tests}", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Successful", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
                html.Div([
                    html.H3(f"{avg_return:.2f}%", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Avg Return", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
                html.Div([
                    html.H3(f"{best_return:.2f}%", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Best Return", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
                html.Div([
                    html.H3(f"{worst_return:.2f}%", style={'margin': 0, 'color': '#2E8B57'}),
                    html.P("Worst Return", style={'margin': 0, 'fontSize': 14})
                ], className="summary-card"),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0', 'flexWrap': 'wrap'}),
            
            # Tabs
            dcc.Tabs(id="tabs", value="heatmap", children=[
                dcc.Tab(label="Performance Heatmap", value="heatmap"),
                dcc.Tab(label="Strategy Comparison", value="strategy_comparison"),
                dcc.Tab(label="Symbol Analysis", value="symbol_analysis"),
                dcc.Tab(label="Detailed Results", value="detailed_results"),
                dcc.Tab(label="Performance Distribution", value="distribution"),
            ]),
            
            html.Div(id="tab-content")
        ])
        
        @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
        def render_tab_content(active_tab):
            if active_tab == "heatmap":
                # Performance heatmap
                pivot_df = results_df[results_df['Status'] == 'Success'].pivot(
                    index='Symbol', columns='Strategy', values='Return (%)'
                ).fillna(0)
                
                fig = px.imshow(
                    pivot_df.values,
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    color_continuous_scale='RdYlGn',
                    aspect="auto",
                    title="Strategy Performance Heatmap"
                )
                fig.update_layout(
                    xaxis_title="Strategy",
                    yaxis_title="Symbol",
                    height=400 + len(pivot_df) * 30
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == "strategy_comparison":
                # Strategy performance comparison
                strategy_avg = results_df[results_df['Status'] == 'Success'].groupby('Strategy')['Return (%)'].agg(['mean', 'std', 'count']).reset_index()
                strategy_avg = strategy_avg.sort_values('mean', ascending=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=strategy_avg['Strategy'],
                        x=strategy_avg['mean'],
                        error_x=dict(type='data', array=strategy_avg['std']),
                        orientation='h',
                        marker_color=['#2E8B57' if x > 0 else '#DC143C' for x in strategy_avg['mean']]
                    )
                ])
                
                fig.update_layout(
                    title="Average Strategy Performance",
                    xaxis_title="Average Return (%)",
                    yaxis_title="Strategy",
                    height=400 + len(strategy_avg) * 30
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == "symbol_analysis":
                # Symbol performance analysis
                symbol_avg = results_df[results_df['Status'] == 'Success'].groupby('Symbol')['Return (%)'].agg(['mean', 'std', 'count']).reset_index()
                symbol_avg = symbol_avg.sort_values('mean', ascending=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        y=symbol_avg['Symbol'],
                        x=symbol_avg['mean'],
                        error_x=dict(type='data', array=symbol_avg['std']),
                        orientation='h',
                        marker_color=['#2E8B57' if x > 0 else '#DC143C' for x in symbol_avg['mean']]
                    )
                ])
                
                fig.update_layout(
                    title="Average Symbol Performance",
                    xaxis_title="Average Return (%)",
                    yaxis_title="Symbol",
                    height=400 + len(symbol_avg) * 30
                )
                
                return dcc.Graph(figure=fig)
            
            elif active_tab == "detailed_results":
                # Detailed results table
                return html.Div([
                    html.H4("Detailed Results Table"),
                    html.Table([
                        html.Tr([html.Th(col) for col in results_df.columns])
                    ] + [
                        html.Tr([
                            html.Td(str(cell)) for cell in row
                        ]) for _, row in results_df.iterrows()
                    ], style={'width': '100%', 'border': '1px solid #ddd', 'fontSize': 12})
                ])
            
            elif active_tab == "distribution":
                # Performance distribution
                successful_returns = results_df[results_df['Status'] == 'Success']['Return (%)']
                
                fig = go.Figure(data=[
                    go.Histogram(
                        x=successful_returns,
                        nbinsx=20,
                        marker_color='#2E8B57',
                        opacity=0.7
                    )
                ])
                
                fig.update_layout(
                    title="Return Distribution",
                    xaxis_title="Return (%)",
                    yaxis_title="Frequency",
                    height=400
                )
                
                return dcc.Graph(figure=fig)
        
        # Add CSS
        app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>Multi-Symbol Strategy Analysis</title>
                {%favicon%}
                {%css%}
                <style>
                    .summary-card {
                        background: #f8f9fa;
                        border: 1px solid #dee2e6;
                        border-radius: 8px;
                        padding: 20px;
                        text-align: center;
                        min-width: 120px;
                        margin: 5px;
                    }
                    table {
                        border-collapse: collapse;
                        width: 100%;
                    }
                    th, td {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # Launch dashboard
        def launch_dashboard():
            try:
                port = output_config.get('dashboard_port', 8052)
                app.run(debug=False, port=port)
            except Exception as e:
                print(f"❌ Dashboard error: {e}")
        
        port = output_config.get('dashboard_port', 8052)
        print(f"\n🌐 Multi-Symbol Strategy Dashboard launching at http://127.0.0.1:{port}/")
        print("📊 Dashboard includes:")
        print("   • Performance heatmap (symbols vs strategies)")
        print("   • Strategy comparison charts")
        print("   • Symbol analysis")
        print("   • Detailed results table")
        print("   • Performance distribution")
        print("   • Summary statistics")
        print("\n💡 Dashboard will stay open. Press Ctrl+C to stop.")
        
        # Open browser automatically
        time.sleep(2)
        try:
            webbrowser.open(f'http://127.0.0.1:{port}/')
        except:
            pass
        
        # Launch dashboard in foreground (blocking)
        try:
            launch_dashboard()
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
            
    except ImportError:
        print("❌ Dashboard dependencies not available. Install with: pip install dash plotly")
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")

def run_strategy_optimization_core(
    config_file: str,
    strategy_name: Optional[str],
    generations: int,
    population_size: int,
    mutation_rate: float,
    crossover_rate: float,
    max_iterations: int,
    optimization_method: str,
    consistency_weight: float,
    profit_weight: float,
    dashboard: bool
):
    """
    Optimize strategy parameters using genetic algorithms or other optimization methods.
    
    Args:
        config_file: Path to the YAML configuration file
        strategy_name: Strategy to optimize (default: first strategy in config)
        generations: Number of optimization generations
        population_size: Population size for genetic algorithm
        mutation_rate: Mutation rate for genetic algorithm
        crossover_rate: Crossover rate for genetic algorithm
        max_iterations: Maximum iterations for optimization
        optimization_method: Optimization method to use
        consistency_weight: Weight for consistency in fitness function (0-1)
        profit_weight: Weight for profit in fitness function (0-1)
        dashboard: Whether to show optimization dashboard
    
    Returns:
        dict: Optimization results
    """
    try:
        # Load configuration
        config = load_config_file(config_file)
        print(f"✅ Loaded configuration from {config_file}")
        
        # Get symbols and data config
        symbols = config['symbols']
        data_config = config['data']
        period = data_config['period']
        interval = data_config['interval']
        
        # Select strategy to optimize
        if not strategy_name:
            if 'builtin' in config['strategies'] and config['strategies']['builtin']:
                strategy_name = config['strategies']['builtin'][0]
            else:
                print("❌ Error: No strategy specified and no builtin strategies found in config")
                return None
        
        print(f"🎯 Optimizing strategy: {strategy_name}")
        print(f"📊 Symbols: {', '.join(symbols)}")
        print(f"📅 Period: {period}, Interval: {interval}")
        
        # Define parameter ranges for optimization
        param_ranges = _get_strategy_parameter_ranges(strategy_name)
        
        if not param_ranges:
            print(f"❌ Error: No parameter ranges defined for strategy {strategy_name}")
            return None
        
        print(f"🔧 Optimizing {len(param_ranges)} parameters: {list(param_ranges.keys())}")
        
        # Run optimization
        if optimization_method == "genetic":
            results = _run_genetic_optimization(
                strategy_name, symbols, period, interval, param_ranges,
                generations, population_size, mutation_rate, crossover_rate,
                consistency_weight, profit_weight, dashboard
            )
        elif optimization_method == "differential_evolution":
            results = _run_differential_evolution_optimization(
                strategy_name, symbols, period, interval, param_ranges,
                max_iterations, consistency_weight, profit_weight, dashboard
            )
        else:  # minimize
            results = _run_minimize_optimization(
                strategy_name, symbols, period, interval, param_ranges,
                max_iterations, consistency_weight, profit_weight, dashboard
            )
        
        # Display results
        _display_optimization_results(results, strategy_name, symbols)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return None

def _get_strategy_parameter_ranges(strategy_name: str) -> Dict[str, Tuple[float, float]]:
    """Get parameter ranges for different strategies."""
    ranges = {
        'default': {
            'macd_fast': (8, 20),
            'macd_slow': (20, 35),
            'macd_signal': (7, 15),
            'rsi_period': (10, 20),
            'roc_period': (3, 10),
            'ema_mom_period': (15, 30)
        },
        'trend_following': {
            'short_ma_period': (10, 30),
            'long_ma_period': (30, 70),
            'trend_ma_period': (100, 300),
            'rsi_period': (10, 20),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'volume_ma_period': (10, 30),
            'min_volume_ratio': (0.8, 2.0),
            'trend_strength_period': (5, 20),
            'min_trend_strength': (0.01, 0.05)
        },
        'mean_reversion': {
            'bb_period': (15, 25),
            'bb_std': (1.5, 2.5),
            'rsi_period': (10, 20),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'zscore_period': (15, 25),
            'zscore_threshold': (1.5, 3.0),
            'williams_period': (10, 20),
            'stoch_k': (10, 20),
            'stoch_d': (3, 7),
            'cci_period': (10, 20),
            'volume_ma_period': (10, 30),
            'min_volume_ratio': (0.8, 2.0)
        },
        'macd_crossover': {
            'macd_fast': (8, 20),
            'macd_slow': (20, 35),
            'macd_signal': (7, 15),
            'rsi_period': (10, 20),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'trend_filter_period': (30, 70),
            'min_macd_strength': (0.0005, 0.002),
            'max_holding_period': (15, 50),
            'profit_target_pct': (0.01, 0.05),
            'stop_loss_pct': (0.01, 0.04)
        },
        'macd_divergence': {
            'macd_fast': (8, 20),
            'macd_slow': (20, 35),
            'macd_signal': (7, 15),
            'divergence_lookback': (10, 30),
            'min_divergence_strength': (0.01, 0.05),
            'rsi_period': (10, 20),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'max_holding_period': (15, 40),
            'profit_target_pct': (0.02, 0.06),
            'stop_loss_pct': (0.015, 0.04)
        },
        'macd_histogram': {
            'macd_fast': (8, 20),
            'macd_slow': (20, 35),
            'macd_signal': (7, 15),
            'histogram_smoothing': (2, 5),
            'min_histogram_strength': (0.0002, 0.001),
            'momentum_threshold': (0.0005, 0.002),
            'rsi_period': (10, 20),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'max_holding_period': (10, 30),
            'profit_target_pct': (0.015, 0.04),
            'stop_loss_pct': (0.01, 0.03)
        }
    }
    
    return ranges.get(strategy_name, {})

def _run_genetic_optimization(
    strategy_name: str, symbols: List[str], period: str, interval: str,
    param_ranges: Dict[str, Tuple[float, float]], generations: int,
    population_size: int, mutation_rate: float, crossover_rate: float,
    consistency_weight: float, profit_weight: float, dashboard: bool
) -> Dict[str, Any]:
    """Run genetic algorithm optimization."""
    print(f"🧬 Starting genetic algorithm optimization...")
    print(f"   Generations: {generations}")
    print(f"   Population size: {population_size}")
    print(f"   Mutation rate: {mutation_rate}")
    print(f"   Crossover rate: {crossover_rate}")
    
    # Initialize population
    param_names = list(param_ranges.keys())
    population = _initialize_population(param_ranges, population_size)
    
    best_individual = None
    best_fitness = float('-inf')
    generation_results = []
    
    for generation in range(generations):
        print(f"\n🔄 Generation {generation + 1}/{generations}")
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for i, individual in enumerate(population):
            params = dict(zip(param_names, individual))
            fitness = _evaluate_fitness(
                strategy_name, symbols, period, interval, params,
                consistency_weight, profit_weight
            )
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        
        # Record generation results
        avg_fitness = np.mean(fitness_scores)
        max_fitness = np.max(fitness_scores)
        generation_results.append({
            'generation': generation + 1,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'best_params': dict(zip(param_names, best_individual)) if best_individual is not None else {}
        })
        
        print(f"   Average fitness: {avg_fitness:.4f}")
        print(f"   Best fitness: {max_fitness:.4f}")
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1 = _tournament_selection(population, fitness_scores)
            parent2 = _tournament_selection(population, fitness_scores)
            
            if random.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2, param_ranges)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            if random.random() < mutation_rate:
                child1 = _mutate(child1, param_ranges)
            if random.random() < mutation_rate:
                child2 = _mutate(child2, param_ranges)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        population = new_population[:population_size]
    
    # Return results
    best_params = dict(zip(param_names, best_individual)) if best_individual is not None else {}
    
    return {
        'method': 'genetic_algorithm',
        'best_params': best_params,
        'best_fitness': best_fitness,
        'generation_results': generation_results,
        'total_generations': generations,
        'population_size': population_size
    }

def _run_differential_evolution_optimization(
    strategy_name: str, symbols: List[str], period: str, interval: str,
    param_ranges: Dict[str, Tuple[float, float]], max_iterations: int,
    consistency_weight: float, profit_weight: float, dashboard: bool
) -> Dict[str, Any]:
    """Run differential evolution optimization."""
    print(f"🔬 Starting differential evolution optimization...")
    print(f"   Max iterations: {max_iterations}")
    
    param_names = list(param_ranges.keys())
    bounds = [param_ranges[name] for name in param_names]
    
    def objective_function(params):
        param_dict = dict(zip(param_names, params))
        fitness = _evaluate_fitness(
            strategy_name, symbols, period, interval, param_dict,
            consistency_weight, profit_weight
        )
        return -fitness  # Minimize negative fitness
    
    result = differential_evolution(
        objective_function, bounds, maxiter=max_iterations, seed=42
    )
    
    best_params = dict(zip(param_names, result.x))
    
    return {
        'method': 'differential_evolution',
        'best_params': best_params,
        'best_fitness': -result.fun,
        'iterations': result.nit,
        'success': result.success
    }

def _run_minimize_optimization(
    strategy_name: str, symbols: List[str], period: str, interval: str,
    param_ranges: Dict[str, Tuple[float, float]], max_iterations: int,
    consistency_weight: float, profit_weight: float, dashboard: bool
) -> Dict[str, Any]:
    """Run minimize optimization."""
    print(f"🎯 Starting minimize optimization...")
    print(f"   Max iterations: {max_iterations}")
    
    param_names = list(param_ranges.keys())
    bounds = [param_ranges[name] for name in param_names]
    x0 = [(param_ranges[name][0] + param_ranges[name][1]) / 2 for name in param_names]
    
    def objective_function(params):
        param_dict = dict(zip(param_names, params))
        fitness = _evaluate_fitness(
            strategy_name, symbols, period, interval, param_dict,
            consistency_weight, profit_weight
        )
        return -fitness  # Minimize negative fitness
    
    result = minimize(
        objective_function, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iterations}
    )
    
    best_params = dict(zip(param_names, result.x))
    
    return {
        'method': 'minimize',
        'best_params': best_params,
        'best_fitness': -result.fun,
        'iterations': result.nit,
        'success': result.success
    }

def _initialize_population(param_ranges: Dict[str, Tuple[float, float]], size: int) -> List[List[float]]:
    """Initialize random population."""
    population = []
    for _ in range(size):
        individual = []
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                individual.append(random.randint(min_val, max_val))
            else:
                individual.append(random.uniform(min_val, max_val))
        population.append(individual)
    return population

def _evaluate_fitness(
    strategy_name: str, symbols: List[str], period: str, interval: str,
    params: Dict[str, Any], consistency_weight: float, profit_weight: float
) -> float:
    """Evaluate fitness of a parameter set."""
    try:
        # Convert float parameters to integers where needed
        processed_params = {}
        for key, value in params.items():
            if key in ['macd_fast', 'macd_slow', 'macd_signal', 'rsi_period', 'roc_period', 'ema_mom_period']:
                processed_params[key] = int(round(value))
            else:
                processed_params[key] = value
        
        results = []
        
        for symbol in symbols:
            # Get data
            data = get_data(symbol, period, interval)
            if data is None or data.empty:
                continue
            
            # Create strategy
            strategy_cls = load_strategy(strategy_name)
            strategy = strategy_cls(data, **processed_params)
            
            # Run backtest
            backtester = StrategyBacktester(strategy)
            result = backtester.run(plot=False, slider=False)
            results.append(result)
        
        if not results:
            return float('-inf')
        
        # Calculate metrics
        avg_return = np.mean(results)
        std_return = np.std(results)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        win_rate = len([r for r in results if r > 0]) / len(results)
        
        # Calculate consistency score
        consistency_score = sharpe_ratio * win_rate
        
        # Calculate fitness
        fitness = profit_weight * avg_return + consistency_weight * consistency_score
        
        return fitness
        
    except Exception as e:
        return float('-inf')

def _tournament_selection(population: List[List[float]], fitness_scores: List[float], tournament_size: int = 3) -> List[float]:
    """Tournament selection for genetic algorithm."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index].copy()

def _crossover(parent1: List[float], parent2: List[float], param_ranges: Dict[str, Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    """Crossover operation for genetic algorithm."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Single point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1[crossover_point:] = parent2[crossover_point:]
    child2[crossover_point:] = parent1[crossover_point:]
    
    return child1, child2

def _mutate(individual: List[float], param_ranges: Dict[str, Tuple[float, float]]) -> List[float]:
    """Mutation operation for genetic algorithm."""
    mutated = individual.copy()
    param_names = list(param_ranges.keys())
    
    for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
        if random.random() < 0.1:  # 10% chance to mutate each parameter
            if isinstance(min_val, int) and isinstance(max_val, int):
                mutated[i] = random.randint(min_val, max_val)
            else:
                mutated[i] = random.uniform(min_val, max_val)
    
    return mutated

def _display_optimization_results(results: Dict[str, Any], strategy_name: str, symbols: List[str]):
    """Display optimization results."""
    print(f"\n{'='*80}")
    print(f"🎯 OPTIMIZATION RESULTS for {strategy_name}")
    print(f"{'='*80}")
    
    print(f"📊 Method: {results['method']}")
    print(f"🏆 Best fitness: {results['best_fitness']:.4f}")
    
    print(f"\n🔧 OPTIMAL PARAMETERS:")
    for param, value in results['best_params'].items():
        print(f"   {param}: {value}")
    
    if 'generation_results' in results:
        print(f"\n📈 OPTIMIZATION PROGRESS:")
        print(f"   Total generations: {results['total_generations']}")
        print(f"   Population size: {results['population_size']}")
        
        # Show final generation results
        final_gen = results['generation_results'][-1]
        print(f"   Final average fitness: {final_gen['avg_fitness']:.4f}")
        print(f"   Final best fitness: {final_gen['max_fitness']:.4f}")

@cli.command()
@click.argument("symbol")
@click.option("--strategy-path", help="Path to single strategy file or directory containing strategy files")
@click.option("--strategy-class", help="Strategy class name (when using single strategy file)")
@click.option("--macd-fast", default=12, help="MACD fast period")
@click.option("--macd-slow", default=26, help="MACD slow period")
@click.option("--macd-signal", default=9, help="MACD signal period")
@click.option("--rsi-period", default=14, help="RSI period")
@click.option("--roc-period", default=5, help="ROC period")
@click.option("--ema-mom-period", default=20, help="EMA momentum period")
@click.option("--period", default="60d", help="Data period")
@click.option("--interval", default="15m", help="Data interval")
@click.option("--output-mode", default="comparison", type=click.Choice(["equity_only", "equity_curve", "comparison"]), help="Output mode")
@click.option("--use-notebook-plots", is_flag=True, help="Use matplotlib plots instead of dashboard")
def compare_strategies(
    symbol,
    strategy_path,
    strategy_class,
    macd_fast,
    macd_slow,
    macd_signal,
    rsi_period,
    roc_period,
    ema_mom_period,
    period,
    interval,
    output_mode,
    use_notebook_plots
):
    """Compare multiple strategies on the same symbol with dashboard comparison."""
    return run_strategy_comparison_core(
        symbol, strategy_path, strategy_class, macd_fast, macd_slow, macd_signal, rsi_period, roc_period,
        ema_mom_period, period, interval, output_mode, use_notebook_plots
    )

@cli.command()
@click.argument("config_file")
@click.option("--symbol", help="Override symbol from config file (for single symbol testing)")
@click.option("--strategy", help="Override strategy from config file (for single strategy testing)")
def test_config(
    config_file,
    symbol,
    strategy
):
    """Test strategies using a configuration file with multiple symbols and strategies."""
    return run_config_test_core(config_file, symbol, strategy)

@cli.command()
@click.argument("config_file")
@click.option("--strategy", help="Strategy to optimize (default: first strategy in config)")
@click.option("--generations", default=50, help="Number of optimization generations")
@click.option("--population-size", default=20, help="Population size for genetic algorithm")
@click.option("--mutation-rate", default=0.1, help="Mutation rate for genetic algorithm")
@click.option("--crossover-rate", default=0.7, help="Crossover rate for genetic algorithm")
@click.option("--max-iterations", default=100, help="Maximum iterations for optimization")
@click.option("--optimization-method", default="genetic", type=click.Choice(["genetic", "differential_evolution", "minimize"]), help="Optimization method")
@click.option("--consistency-weight", default=0.3, help="Weight for consistency in fitness function (0-1)")
@click.option("--profit-weight", default=0.7, help="Weight for profit in fitness function (0-1)")
@click.option("--dashboard", is_flag=True, help="Show optimization dashboard")
def optimize_strategy(
    config_file,
    strategy,
    generations,
    population_size,
    mutation_rate,
    crossover_rate,
    max_iterations,
    optimization_method,
    consistency_weight,
    profit_weight,
    dashboard
):
    """Optimize strategy parameters using genetic algorithms to maximize profit and consistency."""
    return run_strategy_optimization_core(
        config_file, strategy, generations, population_size, mutation_rate, 
        crossover_rate, max_iterations, optimization_method, consistency_weight, 
        profit_weight, dashboard
    )

# Entry point for CLI execution
if __name__ == "__main__":
    cli()