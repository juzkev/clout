"""Tests for trade_analyzer and metrics modules."""

import pandas as pd
import pytest

from backtesting import metrics
from backtesting.trade_analyzer import (
    GroupStats,
    AnalysisReport,
    _group_stats,
    _check_conviction_calibration,
    analyze,
)


_SAMPLE_TRADES = [
    {
        "ticker": "IBIT", "direction": "long", "pnl_pct": 4.5, "conviction": 4,
        "signal_type": "rule_based",
        "primary_rule": "if btc_funding < 0 then long IBIT",
        "signal_sources": ["funding_rate", "momentum"],
    },
    {
        "ticker": "IBIT", "direction": "long", "pnl_pct": -2.1, "conviction": 3,
        "signal_type": "rule_based",
        "primary_rule": "if btc_funding < 0 then long IBIT",
        "signal_sources": ["funding_rate"],
    },
    {
        "ticker": "GLD", "direction": "long", "pnl_pct": 1.8, "conviction": 4,
        "signal_type": "situational",
        "signal_sources": ["cot", "trends"],
    },
    {
        "ticker": "QQQ", "direction": "short", "pnl_pct": -3.2, "conviction": 2,
        "signal_type": "situational",
        "signal_sources": ["rsi"],
    },
    {
        "ticker": "GLD", "direction": "long", "pnl_pct": 2.9, "conviction": 5,
        "signal_type": "rule_based",
        "primary_rule": "if gold_cot_net > 0 then long GLD",
        "signal_sources": ["cot"],
    },
]


# ── GroupStats ────────────────────────────────────────────────────────────────

def test_group_stats_basic():
    stats = _group_stats("test", _SAMPLE_TRADES)
    assert stats.count == 5
    assert 0.0 <= stats.win_rate <= 1.0
    assert stats.best_pnl_pct == 4.5
    assert stats.worst_pnl_pct == -3.2


def test_group_stats_empty():
    stats = _group_stats("empty", [])
    assert stats.count == 0
    assert stats.win_rate == 0.0
    assert stats.avg_pnl_pct == 0.0


def test_group_stats_all_wins():
    trades = [{"pnl_pct": 2.0}, {"pnl_pct": 5.0}]
    stats = _group_stats("all_wins", trades)
    assert stats.win_rate == 1.0
    assert stats.profit_factor == float("inf")


def test_group_stats_profit_factor():
    trades = [{"pnl_pct": 10.0}, {"pnl_pct": 5.0}, {"pnl_pct": -3.0}]
    stats = _group_stats("t", trades)
    assert stats.profit_factor == pytest.approx(5.0)  # 15 / 3


def test_group_stats_sharpe_positive():
    trades = [{"pnl_pct": p} for p in [1.0, 2.0, 1.5, 1.8, 2.2]]
    stats = _group_stats("t", trades)
    assert stats.sharpe_ratio > 0


def test_group_stats_sharpe_zero_variance():
    trades = [{"pnl_pct": 2.0}, {"pnl_pct": 2.0}]
    stats = _group_stats("t", trades)
    assert stats.sharpe_ratio == 0.0


# ── AnalysisReport ────────────────────────────────────────────────────────────

def test_analyze_totals():
    report = analyze(_SAMPLE_TRADES, period_days=30)
    assert report.total_trades == 5
    assert report.period_days == 30


def test_analyze_by_signal_type():
    report = analyze(_SAMPLE_TRADES)
    assert report.by_signal_type["rule_based"].count == 3
    assert report.by_signal_type["situational"].count == 2
    assert report.by_signal_type["hybrid"].count == 0


def test_analyze_by_primary_rule():
    report = analyze(_SAMPLE_TRADES)
    assert "if btc_funding < 0 then long IBIT" in report.by_primary_rule
    rule_stats = report.by_primary_rule["if btc_funding < 0 then long IBIT"]
    assert rule_stats.count == 2


def test_analyze_by_ticker():
    report = analyze(_SAMPLE_TRADES)
    assert report.by_ticker["IBIT"].count == 2
    assert report.by_ticker["GLD"].count == 2
    assert report.by_ticker["QQQ"].count == 1


def test_analyze_by_signal_source():
    report = analyze(_SAMPLE_TRADES)
    assert "funding_rate" in report.by_signal_source
    assert report.by_signal_source["funding_rate"].count == 2
    assert "cot" in report.by_signal_source
    assert report.by_signal_source["cot"].count == 2  # GLD situational + GLD rule_based


def test_analyze_by_conviction():
    report = analyze(_SAMPLE_TRADES)
    assert "conviction_4" in report.by_conviction
    assert report.by_conviction["conviction_4"].count == 2
    assert "conviction_1" not in report.by_conviction  # no such trades


def test_analyze_no_trades():
    report = analyze([], period_days=30)
    assert report.total_trades == 0
    assert report.overall.count == 0
    assert report.by_primary_rule == {}
    assert report.by_ticker == {}


# ── Conviction calibration ────────────────────────────────────────────────────

def test_conviction_calibration_positive():
    trades = [
        {"conviction": 4, "pnl_pct": 5.0},
        {"conviction": 5, "pnl_pct": 4.0},
        {"conviction": 2, "pnl_pct": 1.0},
        {"conviction": 3, "pnl_pct": -1.0},
    ]
    assert _check_conviction_calibration(trades) is True


def test_conviction_calibration_negative():
    trades = [
        {"conviction": 4, "pnl_pct": -1.0},
        {"conviction": 5, "pnl_pct": -2.0},
        {"conviction": 2, "pnl_pct": 3.0},
        {"conviction": 3, "pnl_pct": 4.0},
    ]
    assert _check_conviction_calibration(trades) is False


def test_conviction_calibration_insufficient_data():
    trades = [{"conviction": 5, "pnl_pct": 3.0}]
    assert _check_conviction_calibration(trades) is False


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_win_rate_two_thirds():
    trades = [{"pnl_pct": 1.0}, {"pnl_pct": -1.0}, {"pnl_pct": 2.0}]
    assert metrics.win_rate(trades) == pytest.approx(2 / 3, abs=0.001)


def test_metrics_win_rate_empty():
    assert metrics.win_rate([]) == 0.0


def test_metrics_sharpe_positive():
    returns = [0.01, 0.02, -0.005, 0.015, 0.01]
    result = metrics.sharpe(returns)
    assert result > 0


def test_metrics_sharpe_too_few_returns():
    assert metrics.sharpe([0.05]) == 0.0


def test_metrics_max_drawdown():
    curve = pd.Series([100.0, 110.0, 90.0, 95.0, 80.0, 120.0])
    dd = metrics.max_drawdown(curve)
    assert dd < 0
    # Peak 110, trough 80: (80 - 110) / 110 ≈ -0.2727
    assert dd == pytest.approx(-110 / 110 + 80 / 110, abs=0.001)


def test_metrics_max_drawdown_empty():
    assert metrics.max_drawdown(pd.Series([], dtype=float)) == 0.0


def test_metrics_cagr_doubles():
    # 252 periods: 100 → 200 = 100% CAGR
    curve = pd.Series([100.0] * 252 + [200.0])
    # Approximate: final/initial over 1+ year
    result = metrics.cagr(curve)
    assert result > 0


def test_metrics_cagr_empty():
    assert metrics.cagr(pd.Series([], dtype=float)) == 0.0
