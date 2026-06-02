"""Backtest performance metrics."""

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series  # indexed by date
    trades: list[dict]
    strategy_name: str


def sharpe(returns: Sequence[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from a sequence of period returns."""
    raise NotImplementedError


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    raise NotImplementedError


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    raise NotImplementedError


def win_rate(trades: list[dict]) -> float:
    """Fraction of closed trades that were profitable."""
    raise NotImplementedError


def summary(result: BacktestResult) -> dict[str, float]:
    """Return dict of all key metrics for a BacktestResult."""
    raise NotImplementedError
