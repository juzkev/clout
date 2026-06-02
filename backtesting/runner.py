"""Backtesting engine."""

from typing import Any

import pandas as pd

from backtesting.metrics import BacktestResult
from strategies.base_strategy import BaseStrategy


class Backtester:
    """Drives a strategy over historical price data and collects trade records."""

    def __init__(self, initial_capital: float = 100_000.0, commission_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

    def run(self, strategy: BaseStrategy, price_data: dict[str, pd.DataFrame]) -> BacktestResult:
        """Backtest `strategy` over `price_data`.

        Args:
            strategy:   An initialised BaseStrategy subclass.
            price_data: Dict mapping ticker -> OHLCV DataFrame (indexed by date).

        Returns:
            BacktestResult with equity curve and trade log.
        """
        raise NotImplementedError
