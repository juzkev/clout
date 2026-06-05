"""Backtest performance metrics."""

import math
import statistics
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
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free_rate / periods_per_year for r in returns]
    mean = statistics.mean(excess)
    std = statistics.stdev(excess)
    if std == 0:
        return 0.0
    return round(mean / std * math.sqrt(periods_per_year), 4)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return round(float(dd.min()), 4)


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    if start <= 0:
        return 0.0
    n_years = len(equity_curve) / periods_per_year
    return round((end / start) ** (1 / n_years) - 1, 4)


def win_rate(trades: list[dict]) -> float:
    """Fraction of closed trades that were profitable."""
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if (t.get("pnl_pct") or 0.0) > 0)
    return round(winners / len(trades), 4)


def summary(result: BacktestResult) -> dict[str, float]:
    """Return dict of all key metrics for a BacktestResult."""
    returns = result.equity_curve.pct_change().dropna().tolist()
    return {
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(result.equity_curve),
        "cagr": cagr(result.equity_curve),
        "win_rate": win_rate(result.trades),
        "total_trades": len(result.trades),
    }
