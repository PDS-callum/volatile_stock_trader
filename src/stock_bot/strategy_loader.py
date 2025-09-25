"""
Strategy loader utility for loading custom trading strategies from Python files.
"""

import os
import sys
import importlib.util
import logging
from typing import Type, Optional
from .strategies.base_strategy import BaseStrategy

def load_strategy_from_file(file_path: str, class_name: str = None) -> Type[BaseStrategy]:
    """
    Load a strategy class from a Python file.
    
    Args:
        file_path: Path to the Python file containing the strategy
        class_name: Name of the strategy class. If None, will try to find a class that inherits from BaseStrategy
        
    Returns:
        The strategy class
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ImportError: If the file can't be imported or doesn't contain a valid strategy class
        ValueError: If no valid strategy class is found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Strategy file not found: {file_path}")
    
    # Get absolute path
    file_path = os.path.abspath(file_path)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("custom_strategy", file_path)
    if spec is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_strategy"] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing strategy file {file_path}: {e}")
    
    # Find the strategy class
    strategy_class = None
    
    if class_name:
        # Look for specific class name
        if hasattr(module, class_name):
            strategy_class = getattr(module, class_name)
        else:
            raise ValueError(f"Class '{class_name}' not found in {file_path}")
    else:
        # Look for any class that inherits from BaseStrategy
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseStrategy) and 
                attr != BaseStrategy):
                strategy_class = attr
                break
    
    if strategy_class is None:
        raise ValueError(f"No strategy class found in {file_path}. "
                        "Make sure the file contains a class that inherits from BaseStrategy.")
    
    # Validate that it's a proper strategy class
    if not issubclass(strategy_class, BaseStrategy):
        raise ValueError(f"Class {strategy_class.__name__} does not inherit from BaseStrategy")
    
    logging.info(f"Successfully loaded strategy class: {strategy_class.__name__} from {file_path}")
    return strategy_class

def get_available_strategies() -> dict:
    """
    Get a dictionary of available built-in strategies.
    
    Returns:
        Dictionary mapping strategy names to their classes
    """
    from .strategies import DefaultStrategy, TrendFollowingStrategy, SimpleTrendStrategy, AggressiveStrategy, MeanReversionStrategy, SimpleMeanReversionStrategy, MACDCrossoverStrategy, MACDDivergenceStrategy, MACDHistogramStrategy
    
    return {
        'default': DefaultStrategy,
        'trend_following': TrendFollowingStrategy,
        'simple_trend': SimpleTrendStrategy,
        'aggressive': AggressiveStrategy,
        'mean_reversion': MeanReversionStrategy,
        'simple_mean_reversion': SimpleMeanReversionStrategy,
        'macd_crossover': MACDCrossoverStrategy,
        'macd_divergence': MACDDivergenceStrategy,
        'macd_histogram': MACDHistogramStrategy,
    }

def load_strategy(strategy_name_or_path: str, class_name: str = None) -> Type[BaseStrategy]:
    """
    Load a strategy either by name (for built-in strategies) or by file path.
    
    Args:
        strategy_name_or_path: Either a built-in strategy name or path to a Python file
        class_name: Name of the strategy class (only used when loading from file)
        
    Returns:
        The strategy class
    """
    # Check if it's a built-in strategy
    available_strategies = get_available_strategies()
    if strategy_name_or_path in available_strategies:
        return available_strategies[strategy_name_or_path]
    
    # Try to load from file
    return load_strategy_from_file(strategy_name_or_path, class_name)
