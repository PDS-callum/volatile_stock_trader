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
import yfinance as yf
from .strategy_backtester import StrategyBacktester
from .strategy import DefaultStrategy

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
        symbol, macd_fast, macd_slow, macd_signal, rsi_period, roc_period,
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
                sym, macd_fast, macd_slow, macd_signal, rsi_period, roc_period,
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
    symbol, macd_fast, macd_slow, macd_signal, rsi_period, roc_period,
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
    strategy = DefaultStrategy(historical_data)
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


# Entry point for CLI execution
if __name__ == "__main__":
    cli()