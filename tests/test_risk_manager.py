"""Tests for the minimum-viable risk system and thesis-field tracking."""

from datetime import datetime

import pytest

from research import risk_manager
from research.llm_client import merge_final_signals


def test_position_size_blocked():
    with pytest.raises(ValueError):
        risk_manager.check_position_size("SPY", 0.25, 10000)


def test_vixy_size_blocked():
    # 10% exceeds the dedicated 5% VIXY cap
    with pytest.raises(ValueError):
        risk_manager.check_position_size("VIXY", 0.10, 10000)


def test_drawdown_ok():
    assert risk_manager.check_drawdown(9600, 10000) == "ok"


def test_drawdown_pause():
    assert risk_manager.check_drawdown(9000, 10000) == "pause"


def test_drawdown_shutdown():
    assert risk_manager.check_drawdown(8400, 10000) == "shutdown"


def test_correlation_guard_applied():
    trades = [
        {"ticker": "GLD", "direction": "long", "size_multiplier": 1.0},
        {"ticker": "SLV", "direction": "long", "size_multiplier": 1.0},
    ]
    result = risk_manager.apply_correlation_guard(trades)
    slv = next(t for t in result if t["ticker"] == "SLV")
    assert slv["size_multiplier"] == 0.5


def test_correlation_guard_not_applied_when_different_directions():
    trades = [
        {"ticker": "GLD", "direction": "long", "size_multiplier": 1.0},
        {"ticker": "SLV", "direction": "short", "size_multiplier": 1.0},
    ]
    result = risk_manager.apply_correlation_guard(trades)
    slv = next(t for t in result if t["ticker"] == "SLV")
    assert slv["size_multiplier"] == 1.0


def test_friday_rule(monkeypatch):
    class _FakeDateTime:
        @staticmethod
        def now(tz=None):
            return datetime(2024, 1, 5, tzinfo=tz)  # 2024-01-05 is a Friday

    monkeypatch.setattr(risk_manager, "datetime", _FakeDateTime)
    assert risk_manager.check_friday_rule() is True


def test_validate_all_shutdown():
    trades = [{"ticker": "IBIT", "direction": "long", "conviction": 4}]
    result = risk_manager.validate_all(
        trades, current_value=8400, peak_value=10000, daily_pnl_pct=0.0
    )
    assert result["approved_trades"] == []
    assert result["risk_status"] == "shutdown"


def test_thesis_fields_present():
    regime = {"regime": "risk_on", "confidence": 4}
    trade_ideas = {
        "trade_ideas": [
            {
                "ticker": "IBIT",
                "direction": "long",
                "conviction": 4,
                "reasoning": "Momentum + funding supportive.",
                "invalidation": "BTC breaks 20d SMA.",
                "holding_days": 5,
            }
        ]
    }
    stress_test = {
        "reviewed_ideas": [
            {
                "ticker": "IBIT",
                "final_recommendation": "proceed",
                "size_adjustment": "full",
                "adjusted_conviction": 4,
            }
        ],
        "portfolio_level_risks": [],
        "overall_assessment": "",
    }
    result = merge_final_signals(regime, trade_ideas, stress_test, "2024-01-15")
    assert result["final_trades"], "expected at least one final trade"
    for trade in result["final_trades"]:
        for key in (
            "thesis",
            "invalidation",
            "thesis_outcome",
            "thesis_outcome_notes",
            "opened_at",
            "closed_at",
        ):
            assert key in trade, f"missing {key}"
        assert trade["thesis"] == "Momentum + funding supportive."
        assert trade["invalidation"] == "BTC breaks 20d SMA."
