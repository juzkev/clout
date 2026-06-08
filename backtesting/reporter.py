"""Backtest report generation."""

from backtesting.metrics import BacktestResult


def generate_report(result: BacktestResult) -> str:
    """Format a BacktestResult as a human-readable text report.

    Intended output: strategy name, date range, CAGR, Sharpe, max drawdown,
    win rate, number of trades, and an ASCII equity curve.
    """
    raise NotImplementedError
