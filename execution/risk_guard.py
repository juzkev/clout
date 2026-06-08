"""Pre-trade risk gate for the execution layer.

Thin wrapper around risk_manager.validate_all. This is the SINGLE entry point
the order manager must call before placing any order — it must never be bypassed.
"""

from typing import Any

from loguru import logger

from research import risk_manager


def check_pre_trade(trades: list[dict[str, Any]], portfolio_state: dict[str, Any]) -> dict[str, Any]:
    """Validate `trades` against current portfolio state before any order is placed.

    Args:
        trades: candidate trades (each a dict with ticker/direction/conviction/...).
        portfolio_state: must contain current_value, peak_value, daily_pnl_pct.

    Returns:
        The validate_all result dict (approved_trades / blocked_trades /
        block_reasons / risk_status / warnings).
    """
    result = risk_manager.validate_all(
        trades=trades,
        current_value=portfolio_state["current_value"],
        peak_value=portfolio_state["peak_value"],
        daily_pnl_pct=portfolio_state["daily_pnl_pct"],
    )

    logger.info(
        "Pre-trade risk check: {} approved, {} blocked | status={}",
        len(result["approved_trades"]),
        len(result["blocked_trades"]),
        result["risk_status"],
    )
    for reason in result["block_reasons"]:
        logger.warning("Block reason: {}", reason)
    for warning in result["warnings"]:
        logger.warning("Risk warning: {}", warning)
    for t in result["approved_trades"]:
        logger.info(
            "Approved: {} {} @ {:.2%} size",
            t.get("ticker", "?"), t.get("direction", "?"), t.get("size_multiplier", 0.0),
        )

    return result
