"""Backtesting engine."""

from typing import Any

import pandas as pd

from backtesting.metrics import BacktestResult
from config import settings
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

        When implementing, enforce max_holding_days per instrument from
        settings.INSTRUMENT_META (see get_max_hold) so backtests respect the
        same hold constraints as the live pipeline.
        """
        raise NotImplementedError

    @staticmethod
    def validate_universe(tickers: list) -> bool:
        """Return True only if every ticker has an entry in settings.INSTRUMENT_META."""
        return all(t in settings.INSTRUMENT_META for t in tickers)

    @staticmethod
    def get_max_hold(ticker: str) -> int:
        """Return max_holding_days for `ticker` from settings.INSTRUMENT_META (0 if unset)."""
        return settings.INSTRUMENT_META.get(ticker, {}).get("max_holding_days", 0)
