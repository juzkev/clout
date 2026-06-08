"""Checks open trade records for stop-loss, target, and holding-period exits.

Called by OrderManager.process_exits() each time the paper loop runs.
Prices come from the IBKR client (live snapshot or last close).
"""

import math
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from execution import trade_logger
from execution.ibkr_client import IBKRClient


def check_exits(client: IBKRClient) -> list[dict[str, Any]]:
    """Return exit signals for every open trade that has hit stop / target / max hold.

    Each returned dict contains:
        ticker, open_date, direction, current_price, pnl_pct, exit_reason
    """
    open_trades = trade_logger.get_open_trades()
    if not open_trades:
        return []

    # Fetch all prices in one round-trip
    tickers = list({t["ticker"] for t in open_trades if t.get("ticker")})
    try:
        prices = client.get_prices(tickers)
    except Exception as exc:
        logger.error("Failed to fetch prices for exit check: {}", exc)
        return []

    exits: list[dict[str, Any]] = []
    for trade in open_trades:
        ticker = trade.get("ticker")
        entry_price = trade.get("entry_price")
        direction = trade.get("direction", "long")

        if not ticker or not entry_price:
            continue

        current_price = prices.get(ticker)
        if current_price is None or math.isnan(current_price) or current_price <= 0:
            logger.warning("No valid price for {} — skipping exit check", ticker)
            continue

        pnl_pct = (current_price - entry_price) / entry_price * 100
        if direction == "short":
            pnl_pct = -pnl_pct

        stop_pct = trade.get("stop_loss_pct")
        target_pct = trade.get("target_pct")
        holding_days = trade.get("holding_days")
        opened_at = trade.get("opened_at")

        exit_reason: str | None = None

        if stop_pct and pnl_pct <= -abs(float(stop_pct)):
            exit_reason = "stopped_out"
        elif target_pct and pnl_pct >= abs(float(target_pct)):
            exit_reason = "target_hit"
        elif holding_days and opened_at:
            try:
                days_held = (datetime.now(timezone.utc) - datetime.fromisoformat(opened_at)).days
                if days_held >= int(holding_days):
                    exit_reason = "manual_close"
            except (ValueError, TypeError):
                pass

        if exit_reason:
            logger.info(
                "Exit signal {} {} pnl={:+.2f}% reason={}",
                ticker, direction, pnl_pct, exit_reason,
            )
            exits.append({
                "ticker": ticker,
                "open_date": trade.get("entry_date"),
                "direction": direction,
                "current_price": current_price,
                "pnl_pct": round(pnl_pct, 2),
                "exit_reason": exit_reason,
            })

    return exits
