"""
Strategy modules for the stock trading bot.

This package contains various trading strategies that can be used with the bot.
Each strategy should inherit from BaseStrategy and implement the check_signals method.
"""

from .base_strategy import BaseStrategy
from .default_strategy import DefaultStrategy
from .trend_following_strategy import TrendFollowingStrategy
from .simple_trend_strategy import SimpleTrendStrategy
from .aggressive_strategy import AggressiveStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .simple_mean_reversion_strategy import SimpleMeanReversionStrategy
from .macd_crossover_strategy import MACDCrossoverStrategy
from .macd_divergence_strategy import MACDDivergenceStrategy
from .macd_histogram_strategy import MACDHistogramStrategy

__all__ = ['BaseStrategy', 'DefaultStrategy', 'TrendFollowingStrategy', 'SimpleTrendStrategy', 'AggressiveStrategy', 'MeanReversionStrategy', 'SimpleMeanReversionStrategy', 'MACDCrossoverStrategy', 'MACDDivergenceStrategy', 'MACDHistogramStrategy']
