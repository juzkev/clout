"""Offline tests for backtesting.replay — uses injected OHLCV, no network."""

import pandas as pd
import pytest

from backtesting.replay import (
    simulate_trade,
    replay_trade,
    replay_signals,
    _build_record,
)


def _bars(rows: list[tuple]) -> pd.DataFrame:
    """Build an OHLCV frame from (date, open, high, low, close) tuples."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[4] for r in rows],
        },
        index=idx,
    )


# ── simulate_trade: exit conditions ───────────────────────────────────────────

def test_long_target_hit():
    trade = {"ticker": "SPY", "direction": "long", "stop_loss_pct": 3, "target_pct": 5, "holding_days": 5}
    # Entry 100 (open of bar 1). Target = 105. Bar 2 highs to 106 → target hit.
    bars = _bars([
        ("2024-01-02", 100, 101, 99, 100.5),
        ("2024-01-03", 101, 106, 100, 105.5),
    ])
    out = simulate_trade(trade, bars)
    assert out["outcome"] == "target_hit"
    assert out["exit_price"] == pytest.approx(105.0)
    assert out["pnl_pct"] == pytest.approx(5.0)
    assert out["bars_held"] == 1


def test_long_stop_hit():
    trade = {"ticker": "SPY", "direction": "long", "stop_loss_pct": 3, "target_pct": 5, "holding_days": 5}
    # Entry 100. Stop = 97. Bar 2 low 96 → stopped out.
    bars = _bars([
        ("2024-01-02", 100, 101, 99, 100.0),
        ("2024-01-03", 99, 100, 96, 96.5),
    ])
    out = simulate_trade(trade, bars)
    assert out["outcome"] == "stopped_out"
    assert out["exit_price"] == pytest.approx(97.0)
    assert out["pnl_pct"] == pytest.approx(-3.0)


def test_long_time_exit():
    trade = {"ticker": "SPY", "direction": "long", "stop_loss_pct": 10, "target_pct": 10, "holding_days": 2}
    # Neither stop nor target hit → exit at last bar close.
    bars = _bars([
        ("2024-01-02", 100, 102, 99, 101),
        ("2024-01-03", 101, 103, 100, 102),
        ("2024-01-04", 102, 104, 101, 103),
    ])
    out = simulate_trade(trade, bars)
    assert out["outcome"] == "manual_close"
    assert out["exit_price"] == pytest.approx(103.0)  # close of bar index 2 (holding_days+1)


def test_stop_priority_when_both_hit_same_bar():
    """If a bar touches both stop and target, the stop is taken (conservative)."""
    trade = {"ticker": "SPY", "direction": "long", "stop_loss_pct": 3, "target_pct": 5, "holding_days": 5}
    # Bar 2: low 96 (stop 97 hit) AND high 106 (target 105 hit) → stop wins.
    bars = _bars([
        ("2024-01-02", 100, 101, 99, 100),
        ("2024-01-03", 100, 106, 96, 101),
    ])
    out = simulate_trade(trade, bars)
    assert out["outcome"] == "stopped_out"


def test_short_target_hit():
    trade = {"ticker": "TLT", "direction": "short", "stop_loss_pct": 3, "target_pct": 5, "holding_days": 5}
    # Entry 100 short. Target = 95 (price falls). Bar 2 low 94 → target hit, +5%.
    bars = _bars([
        ("2024-01-02", 100, 101, 99, 100),
        ("2024-01-03", 99, 100, 94, 95),
    ])
    out = simulate_trade(trade, bars)
    assert out["outcome"] == "target_hit"
    assert out["pnl_pct"] == pytest.approx(5.0)


def test_short_stop_hit():
    trade = {"ticker": "TLT", "direction": "short", "stop_loss_pct": 3, "target_pct": 5, "holding_days": 5}
    # Entry 100 short. Stop = 103 (price rises). Bar 2 high 104 → stopped, -3%.
    bars = _bars([
        ("2024-01-02", 100, 101, 99, 100),
        ("2024-01-03", 101, 104, 100, 103),
    ])
    out = simulate_trade(trade, bars)
    assert out["outcome"] == "stopped_out"
    assert out["pnl_pct"] == pytest.approx(-3.0)


def test_simulate_empty_raises():
    with pytest.raises(ValueError):
        simulate_trade({"ticker": "SPY", "direction": "long"}, pd.DataFrame())


# ── replay_trade / replay_signals with injected fetcher ───────────────────────

def test_replay_trade_builds_record():
    trade = {
        "ticker": "SPY", "direction": "long", "stop_loss_pct": 3, "target_pct": 5,
        "holding_days": 5, "conviction": 4, "signal_type": "rule_based",
        "primary_rule": "if x then long SPY", "thesis": "momentum",
    }
    bars = _bars([
        ("2024-01-02", 100, 101, 99, 100),
        ("2024-01-03", 101, 106, 100, 105),
    ])
    rec = replay_trade(trade, "2024-01-01", lambda t, s, n: bars)
    assert rec["ticker"] == "SPY"
    assert rec["thesis_outcome"] == "target_hit"
    assert rec["pnl_pct"] == pytest.approx(5.0)
    assert rec["source"] == "replay"
    assert rec["signal_type"] == "rule_based"
    assert rec["entry_date"] == "2024-01-01"
    assert rec["closed_at"].startswith("2024-01-03")  # target hit on second bar


def test_replay_trade_no_data_returns_none():
    trade = {"ticker": "SPY", "direction": "long", "holding_days": 5}
    rec = replay_trade(trade, "2024-01-01", lambda t, s, n: None)
    assert rec is None


def test_replay_signals_empty_final_trades():
    signals = {"date": "2026-06-04", "final_trades": [], "trades_filtered_out": 2}
    records = replay_signals(signals, lambda t, s, n: None)
    assert records == []


def test_replay_signals_multiple_trades():
    signals = {
        "date": "2024-01-01",
        "final_trades": [
            {"ticker": "SPY", "direction": "long", "stop_loss_pct": 3, "target_pct": 5, "holding_days": 5},
            {"ticker": "TLT", "direction": "short", "stop_loss_pct": 3, "target_pct": 5, "holding_days": 5},
        ],
    }
    win_bars = _bars([("2024-01-02", 100, 101, 99, 100), ("2024-01-03", 101, 106, 100, 105)])
    records = replay_signals(signals, lambda t, s, n: win_bars)
    assert len(records) == 2
    spy = next(r for r in records if r["ticker"] == "SPY")
    assert spy["thesis_outcome"] == "target_hit"


def test_build_record_schema_matches_logger():
    """Replay records carry the fields backtest_report / trade_logger expect."""
    trade = {"ticker": "GLD", "direction": "long", "conviction": 3, "signal_sources": ["cot"]}
    sim = {"entry_price": 100.0, "exit_price": 103.0, "exit_date": "2024-01-05",
           "pnl_pct": 3.0, "outcome": "target_hit", "bars_held": 2}
    rec = _build_record(trade, "2024-01-01", sim)
    for key in ("ticker", "direction", "pnl_pct", "thesis_outcome", "signal_type",
                "conviction", "signal_sources", "closed_at", "source"):
        assert key in rec
    assert rec["source"] == "replay"
