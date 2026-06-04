"""Trade lifecycle logging and weekly performance/thesis review.

Open trades live at logs/trades/{ticker}_{date}_open.json. On close they are
rewritten to {ticker}_{date}_closed.json and the _open.json file is removed.
"""

import glob
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from config.settings import settings

_TRADES_DIR = settings.BASE_DIR / "logs" / "trades"

_VALID_OUTCOMES = {
    "played_out",
    "invalidated",
    "stopped_out",
    "target_hit",
    "manual_close",
}


def _ensure_dir() -> None:
    os.makedirs(_TRADES_DIR, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def open_trade(signal: dict[str, Any], entry_price: float, size_usd: float) -> dict[str, Any]:
    """Record a newly opened trade from a signal and persist it to logs/trades/."""
    _ensure_dir()
    entry_date = _utc_today()
    record: dict[str, Any] = {
        "ticker": signal.get("ticker"),
        "direction": signal.get("direction"),
        "entry_price": entry_price,
        "size_usd": size_usd,
        "entry_date": entry_date,
        "thesis": signal.get("thesis", signal.get("reasoning", "")),
        "invalidation": signal.get("invalidation", ""),
        "stop_loss_pct": signal.get("stop_loss_pct"),
        "target_pct": signal.get("target_pct"),
        "holding_days": signal.get("holding_days"),
        "conviction": signal.get("conviction"),
        "size_multiplier": signal.get("size_multiplier"),
        "thesis_outcome": None,
        "thesis_outcome_notes": None,
        "opened_at": _utc_now_iso(),
        "closed_at": None,
        "exit_price": None,
        "pnl_pct": None,
        "signal_sources": signal.get("signal_sources", []),
    }
    path = _TRADES_DIR / f"{record['ticker']}_{entry_date}_open.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Opened trade {} {} @ {} (size ${:,.0f}) → {}",
                record["ticker"], record["direction"], entry_price, size_usd, path.name)
    return record


def close_trade(
    ticker: str,
    open_date: str,
    exit_price: float,
    outcome: str,
    notes: str,
) -> dict[str, Any]:
    """Close an open trade: compute PnL, record outcome, and move to _closed.json."""
    if outcome not in _VALID_OUTCOMES:
        raise ValueError(f"Invalid outcome {outcome!r}. Must be one of {sorted(_VALID_OUTCOMES)}")

    _ensure_dir()
    open_path = _TRADES_DIR / f"{ticker}_{open_date}_open.json"
    if not open_path.exists():
        raise FileNotFoundError(f"No open trade found at {open_path}")

    record: dict[str, Any] = json.loads(open_path.read_text(encoding="utf-8"))

    entry_price = record.get("entry_price")
    pnl_pct = None
    if entry_price:
        raw = (exit_price - entry_price) / entry_price * 100
        # Short positions profit when price falls
        pnl_pct = round(raw if record.get("direction") != "short" else -raw, 2)

    record.update({
        "closed_at": _utc_now_iso(),
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
        "thesis_outcome": outcome,
        "thesis_outcome_notes": notes,
    })

    closed_path = _TRADES_DIR / f"{ticker}_{open_date}_closed.json"
    closed_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    open_path.unlink()
    logger.info("Closed trade {} ({}) @ {} | PnL {} | outcome={}",
                ticker, open_date, exit_price, pnl_pct, outcome)
    return record


def get_open_trades() -> list[dict[str, Any]]:
    """Return all currently open trade records."""
    _ensure_dir()
    trades: list[dict[str, Any]] = []
    for path in glob.glob(str(_TRADES_DIR / "*_open.json")):
        try:
            trades.append(json.loads(open(path, encoding="utf-8").read()))
        except Exception as exc:
            logger.warning("Failed to read open trade {}: {}", path, exc)
    return trades


def get_closed_trades(days_back: int = 30) -> list[dict[str, Any]]:
    """Return closed trades within the last `days_back` days, newest first."""
    _ensure_dir()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    trades: list[dict[str, Any]] = []
    for path in glob.glob(str(_TRADES_DIR / "*_closed.json")):
        try:
            rec = json.loads(open(path, encoding="utf-8").read())
        except Exception as exc:
            logger.warning("Failed to read closed trade {}: {}", path, exc)
            continue
        closed_at = rec.get("closed_at")
        if not closed_at:
            continue
        try:
            closed_dt = datetime.fromisoformat(closed_at)
        except ValueError:
            continue
        if closed_dt >= cutoff:
            trades.append(rec)
    trades.sort(key=lambda r: r.get("closed_at", ""), reverse=True)
    return trades


def summarise_week() -> dict[str, Any]:
    """Summarise the last 7 days of closed trades for the weekly review."""
    closed = get_closed_trades(days_back=7)
    total = len(closed)
    result: dict[str, Any] = {
        "total_trades": total,
        "winners": 0,
        "losers": 0,
        "win_rate": 0.0,
        "avg_pnl_pct": 0.0,
        "best_trade": None,
        "worst_trade": None,
        "thesis_accuracy": 0.0,
        "signal_source_hits": {},
    }
    if total == 0:
        return result

    pnls = [t.get("pnl_pct") or 0.0 for t in closed]
    winners = [t for t in closed if (t.get("pnl_pct") or 0.0) > 0]
    losers = [t for t in closed if (t.get("pnl_pct") or 0.0) <= 0]

    result["winners"] = len(winners)
    result["losers"] = len(losers)
    result["win_rate"] = round(len(winners) / total, 3)
    result["avg_pnl_pct"] = round(sum(pnls) / total, 2)
    result["best_trade"] = max(closed, key=lambda t: t.get("pnl_pct") or 0.0)
    result["worst_trade"] = min(closed, key=lambda t: t.get("pnl_pct") or 0.0)

    played_out = sum(1 for t in closed if t.get("thesis_outcome") == "played_out")
    result["thesis_accuracy"] = round(played_out / total, 3)

    source_hits: dict[str, int] = {}
    for t in winners:
        for src in t.get("signal_sources", []) or []:
            source_hits[src] = source_hits.get(src, 0) + 1
    result["signal_source_hits"] = source_hits

    return result
