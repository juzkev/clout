"""Tests for the minimum-viable risk system and thesis-field tracking."""

from datetime import datetime

import pytest
from loguru import logger

from execution import risk_guard
from research import risk_manager
from research.llm_client import merge_final_signals


def _capture_warnings():
    """Return (messages_list, sink_id). Caller must logger.remove(sink_id)."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    return messages, sink_id


def test_position_size_blocked():
    with pytest.raises(ValueError):
        risk_manager.check_position_size("SPY", 0.25, 10000)


def test_vixy_size_blocked(monkeypatch):
    """VIXY over the cap is clamped (not dropped), flagged, and warned — not raised."""
    monkeypatch.setattr(risk_manager, "check_friday_rule", lambda: False)
    messages, sink_id = _capture_warnings()
    trades = [{"ticker": "VIXY", "direction": "long", "conviction": 5}]
    result = risk_manager.validate_all(trades, current_value=10000, peak_value=10000, daily_pnl_pct=0.0)
    logger.remove(sink_id)

    approved = result["approved_trades"]
    assert len(approved) == 1                       # not removed
    assert approved[0]["size_multiplier"] == 0.05    # clamped to the 5% cap
    assert approved[0]["vixy_clamped"] is True
    assert any("clamped" in m for m in messages)     # a warning was logged


def test_vixy_clamped_not_dropped(monkeypatch):
    """A conviction-5 VIXY trade (would size to 18%) survives validate_all at 5%."""
    monkeypatch.setattr(risk_manager, "check_friday_rule", lambda: False)
    trades = [{"ticker": "VIXY", "direction": "long", "conviction": 5}]
    result = risk_manager.validate_all(trades, current_value=10000, peak_value=10000, daily_pnl_pct=0.0)
    tickers = [t["ticker"] for t in result["approved_trades"]]
    assert "VIXY" in tickers
    assert result["approved_trades"][0]["size_multiplier"] == 0.05


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


def test_pass3_compounds_with_conviction(monkeypatch):
    """conviction 5 (→18%) × Pass3 0.5 = 9% final size, via the execution gate."""
    monkeypatch.setattr(risk_manager, "check_friday_rule", lambda: False)
    trades = [{"ticker": "IBIT", "direction": "long", "conviction": 5, "pass3_size_multiplier": 0.5}]
    result = risk_guard.check_pre_trade(
        trades, {"current_value": 10000, "peak_value": 10000, "daily_pnl_pct": 0.0}
    )
    approved = result["approved_trades"]
    assert len(approved) == 1
    assert approved[0]["size_multiplier"] == pytest.approx(0.09)


def test_pass3_skip_survives_to_execution(monkeypatch):
    """A skip (pass3 multiplier 0.0) is removed in merge and never reaches execution."""
    monkeypatch.setattr(risk_manager, "check_friday_rule", lambda: False)
    regime = {"regime": "risk_on", "confidence": 4}
    trade_ideas = {
        "trade_ideas": [
            {"ticker": "IBIT", "direction": "long", "conviction": 4, "reasoning": "x", "invalidation": "y"},
            {"ticker": "GLD", "direction": "long", "conviction": 3, "reasoning": "x", "invalidation": "y"},
        ]
    }
    stress_test = {
        "reviewed_ideas": [
            {"ticker": "IBIT", "final_recommendation": "proceed", "size_adjustment": "full", "adjusted_conviction": 4},
            {"ticker": "GLD", "final_recommendation": "skip", "size_adjustment": "skip", "adjusted_conviction": 2},
        ],
        "portfolio_level_risks": [],
        "overall_assessment": "",
    }
    merged = merge_final_signals(regime, trade_ideas, stress_test, "2024-01-15")
    assert "GLD" not in [t["ticker"] for t in merged["final_trades"]]

    result = risk_guard.check_pre_trade(
        merged["final_trades"], {"current_value": 10000, "peak_value": 10000, "daily_pnl_pct": 0.0}
    )
    assert "GLD" not in [t["ticker"] for t in result["approved_trades"]]
