"""Trade performance attribution — analyzes closed trades from logs/trades/.

Usage:
    from backtesting.trade_analyzer import load_and_analyze
    report = load_and_analyze(days_back=90)

All inputs are real closed-trade records (not simulated), so metrics reflect
actual pipeline performance without look-ahead bias.
"""

import statistics
from dataclasses import dataclass
from typing import Any

from execution import trade_logger


@dataclass
class GroupStats:
    name: str
    count: int
    win_rate: float
    avg_pnl_pct: float
    median_pnl_pct: float
    profit_factor: float  # gross_wins / gross_losses; float("inf") if no losses
    sharpe_ratio: float   # mean/std of per-trade pnl_pct; 0.0 if <2 trades
    best_pnl_pct: float
    worst_pnl_pct: float


@dataclass
class AnalysisReport:
    period_days: int
    total_trades: int
    overall: GroupStats
    by_signal_type: dict[str, GroupStats]
    by_conviction: dict[str, GroupStats]      # keys like "conviction_4"
    by_ticker: dict[str, GroupStats]
    by_primary_rule: dict[str, GroupStats]
    by_signal_source: dict[str, GroupStats]
    conviction_calibrated: bool  # True if conviction ≥4 outperforms conviction ≤3


def _group_stats(name: str, trades: list[dict[str, Any]]) -> GroupStats:
    if not trades:
        return GroupStats(
            name=name, count=0, win_rate=0.0, avg_pnl_pct=0.0,
            median_pnl_pct=0.0, profit_factor=0.0, sharpe_ratio=0.0,
            best_pnl_pct=0.0, worst_pnl_pct=0.0,
        )

    pnls = [(t.get("pnl_pct") or 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    if gross_losses > 0:
        pf: float = round(gross_wins / gross_losses, 2)
    else:
        pf = float("inf")

    avg = sum(pnls) / len(pnls)

    sharpe_ratio = 0.0
    if len(pnls) >= 2:
        std = statistics.stdev(pnls)
        if std > 0:
            sharpe_ratio = round(avg / std, 3)

    return GroupStats(
        name=name,
        count=len(trades),
        win_rate=round(len(wins) / len(trades), 3),
        avg_pnl_pct=round(avg, 2),
        median_pnl_pct=round(statistics.median(pnls), 2),
        profit_factor=pf,
        sharpe_ratio=sharpe_ratio,
        best_pnl_pct=round(max(pnls), 2),
        worst_pnl_pct=round(min(pnls), 2),
    )


def _check_conviction_calibration(trades: list[dict[str, Any]]) -> bool:
    """Return True if conviction ≥4 trades outperform conviction ≤3 trades on avg PnL."""
    high = [t for t in trades if (t.get("conviction") or 0) >= 4]
    low = [t for t in trades if (t.get("conviction") or 0) <= 3]
    if not high or not low:
        return False
    avg_high = sum((t.get("pnl_pct") or 0.0) for t in high) / len(high)
    avg_low = sum((t.get("pnl_pct") or 0.0) for t in low) / len(low)
    return avg_high > avg_low


def analyze(trades: list[dict[str, Any]], period_days: int = 90) -> AnalysisReport:
    """Produce a full attribution report from a list of closed trade records."""
    overall = _group_stats("overall", trades)

    by_signal_type = {
        st: _group_stats(st, [t for t in trades if t.get("signal_type") == st])
        for st in ("rule_based", "situational", "hybrid")
    }

    by_conviction: dict[str, GroupStats] = {}
    for c in range(1, 6):
        bucket = [t for t in trades if t.get("conviction") == c]
        if bucket:
            by_conviction[f"conviction_{c}"] = _group_stats(f"conviction_{c}", bucket)

    tickers = sorted({t.get("ticker") for t in trades if t.get("ticker")})
    by_ticker = {
        tk: _group_stats(tk, [t for t in trades if t.get("ticker") == tk])
        for tk in tickers
    }

    rules = sorted({t.get("primary_rule") for t in trades if t.get("primary_rule")})
    by_primary_rule = {
        rule: _group_stats(rule, [t for t in trades if t.get("primary_rule") == rule])
        for rule in rules
    }

    source_map: dict[str, list] = {}
    for t in trades:
        for src in (t.get("signal_sources") or []):
            source_map.setdefault(src, []).append(t)
    by_signal_source = {src: _group_stats(src, ts) for src, ts in sorted(source_map.items())}

    return AnalysisReport(
        period_days=period_days,
        total_trades=len(trades),
        overall=overall,
        by_signal_type=by_signal_type,
        by_conviction=by_conviction,
        by_ticker=by_ticker,
        by_primary_rule=by_primary_rule,
        by_signal_source=by_signal_source,
        conviction_calibrated=_check_conviction_calibration(trades),
    )


def load_and_analyze(days_back: int = 90) -> AnalysisReport:
    """Load closed trades from trade_logger and run attribution analysis."""
    trades = trade_logger.get_closed_trades(days_back=days_back)
    return analyze(trades, period_days=days_back)
