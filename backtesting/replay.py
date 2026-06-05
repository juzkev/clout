"""Retrospective replay: simulate a signals file against real historical prices.

Given a saved signals JSON (with non-empty final_trades) and historical OHLCV
bars, walk each trade forward bar-by-bar and apply whichever exit triggers
first — stop-loss, target, or max holding period. Outcomes are real (no
look-ahead) because the price path is actual history.

The price fetcher is injectable so the core logic is unit-testable offline.
The CLI (scripts/replay_signals.py) wires in a yfinance fetcher.
"""

import math
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

# A fetcher takes (ticker, start_date_str, num_bars) and returns an OHLCV
# DataFrame indexed by date with columns Open/High/Low/Close (capitalised, as
# yfinance returns). Returns None / empty on failure.
PriceFetcher = Callable[[str, str, int], Optional[pd.DataFrame]]


def simulate_trade(trade: dict[str, Any], ohlcv: pd.DataFrame) -> dict[str, Any]:
    """Simulate one trade over `ohlcv` and return the realised outcome.

    Entry is the OPEN of the first bar. Each bar is then checked for a stop or
    target hit using its high/low; if both are touched in the same bar, the
    stop is assumed first (conservative). If neither triggers within
    holding_days bars, the trade exits at the close of the final bar.

    Args:
        trade: a final_trade dict with direction, stop_loss_pct, target_pct,
               holding_days (and ticker/conviction/etc. carried through).
        ohlcv: date-indexed DataFrame, Open/High/Low/Close, starting at entry.

    Returns:
        Dict with entry_price, exit_price, exit_date, pnl_pct, outcome,
        bars_held. Outcome ∈ {stopped_out, target_hit, manual_close}.
    """
    if ohlcv is None or ohlcv.empty:
        raise ValueError(f"No price data to simulate {trade.get('ticker')}")

    direction = trade.get("direction", "long")
    is_long = direction != "short"
    stop_pct = trade.get("stop_loss_pct")
    target_pct = trade.get("target_pct")
    holding_days = int(trade.get("holding_days") or len(ohlcv))

    bars = ohlcv.iloc[: max(1, holding_days + 1)]
    entry_price = float(bars.iloc[0]["Open"])

    # Stop / target absolute price levels
    stop_price = target_price = None
    if stop_pct:
        stop_price = (
            entry_price * (1 - abs(float(stop_pct)) / 100) if is_long
            else entry_price * (1 + abs(float(stop_pct)) / 100)
        )
    if target_pct:
        target_price = (
            entry_price * (1 + abs(float(target_pct)) / 100) if is_long
            else entry_price * (1 - abs(float(target_pct)) / 100)
        )

    exit_price = float(bars.iloc[-1]["Close"])
    exit_date = _bar_date(bars.index[-1])
    outcome = "manual_close"
    bars_held = len(bars) - 1

    for i in range(len(bars)):
        bar = bars.iloc[i]
        high, low = float(bar["High"]), float(bar["Low"])

        stop_hit = (
            stop_price is not None
            and (low <= stop_price if is_long else high >= stop_price)
        )
        target_hit = (
            target_price is not None
            and (high >= target_price if is_long else low <= target_price)
        )

        if stop_hit:  # conservative: stop takes priority over target in same bar
            exit_price, outcome, bars_held = stop_price, "stopped_out", i
            exit_date = _bar_date(bars.index[i])
            break
        if target_hit:
            exit_price, outcome, bars_held = target_price, "target_hit", i
            exit_date = _bar_date(bars.index[i])
            break

    raw = (exit_price - entry_price) / entry_price * 100
    pnl_pct = round(raw if is_long else -raw, 2)

    return {
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "exit_date": exit_date,
        "pnl_pct": pnl_pct,
        "outcome": outcome,
        "bars_held": bars_held,
    }


def replay_trade(
    trade: dict[str, Any],
    signal_date: str,
    fetch: PriceFetcher,
) -> Optional[dict[str, Any]]:
    """Fetch prices for one trade and simulate it. Returns a closed record or None."""
    ticker = trade.get("ticker")
    if not ticker:
        return None

    holding_days = int(trade.get("holding_days") or 7)
    # Pull a few extra bars beyond the hold window for safety
    bars = fetch(ticker, signal_date, holding_days + 5)
    if bars is None or len(bars) == 0:
        logger.warning("No price data for {} from {} — skipping", ticker, signal_date)
        return None

    sim = simulate_trade(trade, bars)
    return _build_record(trade, signal_date, sim)


def replay_signals(
    signals: dict[str, Any],
    fetch: PriceFetcher,
) -> list[dict[str, Any]]:
    """Replay every final_trade in a signals dict. Returns closed records."""
    signal_date = signals.get("date") or signals.get("generated_at", "")[:10]
    final_trades = signals.get("final_trades", [])

    if not final_trades:
        logger.info("Signals for {} have no final_trades — nothing to replay", signal_date)
        return []

    records: list[dict[str, Any]] = []
    for trade in final_trades:
        rec = replay_trade(trade, signal_date, fetch)
        if rec:
            records.append(rec)
            logger.info(
                "Replayed {} {} → {} pnl={:+.2f}% ({} bars)",
                rec["ticker"], rec["direction"], rec["thesis_outcome"],
                rec["pnl_pct"], rec.get("bars_held", "?"),
            )
    return records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar_date(idx_value: Any) -> str:
    """Normalise a DataFrame index value to a YYYY-MM-DD string."""
    if isinstance(idx_value, (pd.Timestamp, datetime)):
        return idx_value.strftime("%Y-%m-%d")
    return str(idx_value)[:10]


def _build_record(trade: dict[str, Any], signal_date: str, sim: dict[str, Any]) -> dict[str, Any]:
    """Construct a closed-trade record matching trade_logger's schema.

    Tagged with source='replay' so it can be distinguished from live paper
    trades while still being picked up by backtest_report attribution.
    """
    return {
        "ticker": trade.get("ticker"),
        "direction": trade.get("direction"),
        "entry_price": sim["entry_price"],
        "size_usd": None,
        "entry_date": signal_date,
        "thesis": trade.get("thesis", trade.get("reasoning", "")),
        "invalidation": trade.get("invalidation", ""),
        "stop_loss_pct": trade.get("stop_loss_pct"),
        "target_pct": trade.get("target_pct"),
        "holding_days": trade.get("holding_days"),
        "conviction": trade.get("conviction"),
        "size_multiplier": trade.get("size_multiplier"),
        "thesis_outcome": sim["outcome"],
        "thesis_outcome_notes": f"Replay: {sim['outcome']} @ {sim['exit_price']} on {sim['exit_date']}",
        "opened_at": signal_date,
        "closed_at": sim["exit_date"] + "T00:00:00+00:00",
        "exit_price": sim["exit_price"],
        "pnl_pct": sim["pnl_pct"],
        "bars_held": sim["bars_held"],
        "signal_sources": trade.get("signal_sources", []),
        "signal_type": trade.get("signal_type"),
        "primary_rule": trade.get("primary_rule"),
        "rule_backtest_status": trade.get("rule_backtest_status", "not_tested"),
        "validation_method": trade.get("validation_method"),
        "contributing_signals": trade.get(
            "contributing_signals", {"rule_based": [], "situational": []}
        ),
        "source": "replay",
    }
