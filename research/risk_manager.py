"""Stateless risk-management utilities.

Pure functions over trade lists and portfolio state. Every threshold is read
from settings.RISK_LIMITS — no values are hard-coded here.

These functions are used in two places:
  - merge_final_signals / run_research apply a subset (correlation guard,
    position sizing) to produce preliminary signals.
  - The execution layer (risk_guard.check_pre_trade) calls validate_all as the
    single authoritative gate before any order is placed.
"""

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from config.settings import settings


def check_position_size(ticker: str, size_pct: float, portfolio_value: float) -> None:
    """Raise ValueError if a position exceeds its size cap.

    VIXY has a stricter dedicated cap (vixy_max_portfolio_pct); everything else
    is bounded by max_position_size_pct. size_pct is a fraction of the portfolio.
    """
    rl = settings.RISK_LIMITS
    if ticker == "VIXY":
        limit = float(rl["vixy_max_portfolio_pct"])
        if size_pct > limit:
            raise ValueError(
                f"VIXY size {size_pct:.2%} exceeds VIXY cap {limit:.2%} "
                f"(portfolio ${portfolio_value:,.0f})"
            )
        return
    limit = float(rl["max_position_size_pct"])
    if size_pct > limit:
        raise ValueError(
            f"{ticker} size {size_pct:.2%} exceeds max position cap {limit:.2%} "
            f"(portfolio ${portfolio_value:,.0f})"
        )


def check_drawdown(current_value: float, peak_value: float) -> str:
    """Return 'ok' | 'pause' | 'shutdown' based on drawdown from peak."""
    if peak_value <= 0:
        return "ok"
    drawdown = (peak_value - current_value) / peak_value
    rl = settings.RISK_LIMITS
    if drawdown >= float(rl["drawdown_shutdown_threshold"]):
        logger.error(
            "Drawdown {:.2%} >= shutdown threshold {:.2%} — TRADING HALTED",
            drawdown, float(rl["drawdown_shutdown_threshold"]),
        )
        return "shutdown"
    if drawdown >= float(rl["drawdown_pause_threshold"]):
        logger.warning(
            "Drawdown {:.2%} >= pause threshold {:.2%} — new trades paused",
            drawdown, float(rl["drawdown_pause_threshold"]),
        )
        return "pause"
    return "ok"


def check_daily_loss(daily_pnl_pct: float) -> bool:
    """Return True if today's loss breaches the daily limit (stop trading).

    daily_pnl_pct is a signed fraction (e.g. -0.03 == down 3%).
    """
    limit = float(settings.RISK_LIMITS["max_daily_loss_pct"])
    if daily_pnl_pct <= -limit:
        logger.warning(
            "Daily PnL {:.2%} breached daily loss limit -{:.2%} — stop for the day",
            daily_pnl_pct, limit,
        )
        return True
    return False


def apply_correlation_guard(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Halve SLV size when GLD and SLV are both held long (correlated exposure)."""
    gld_long = any(t.get("ticker") == "GLD" and t.get("direction") == "long" for t in trades)
    slv_long = any(t.get("ticker") == "SLV" and t.get("direction") == "long" for t in trades)
    if gld_long and slv_long:
        mult = float(settings.RISK_LIMITS["gld_slv_both_long_slv_multiplier"])
        for t in trades:
            if t.get("ticker") == "SLV" and t.get("direction") == "long":
                old = t.get("size_multiplier", 1.0)
                t["size_multiplier"] = old * mult
                logger.warning(
                    "Correlation guard: GLD & SLV both long → SLV size {} × {} = {}",
                    old, mult, t["size_multiplier"],
                )
    return trades


def check_friday_rule() -> bool:
    """Return True if new trades should be blocked because it's Friday."""
    if not settings.RISK_LIMITS["no_new_trades_on_friday"]:
        return False
    is_friday = datetime.now(timezone.utc).weekday() == 4  # Mon=0 ... Fri=4
    if is_friday:
        logger.warning("Friday rule active — blocking new trades")
    return is_friday


def apply_position_sizing(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set each trade's size_multiplier from its conviction (overrides existing)."""
    for t in trades:
        conviction = t.get("conviction")
        size = settings.get_position_size_pct(conviction)
        t["size_multiplier"] = size
        logger.info(
            "Sizing {} (conviction {}) → {:.2%} of portfolio",
            t.get("ticker", "?"), conviction, size,
        )
    return trades


def validate_all(
    trades: list[dict[str, Any]],
    current_value: float,
    peak_value: float,
    daily_pnl_pct: float,
) -> dict[str, Any]:
    """Run the full ordered risk pipeline and return the approval result."""
    warnings: list[str] = []
    risk_status = check_drawdown(current_value, peak_value)

    # 1. Drawdown shutdown — block everything immediately
    if risk_status == "shutdown":
        reason = "Drawdown shutdown threshold breached — all trading halted"
        return _blocked_result(trades, [reason], "shutdown", warnings)

    # 2. Daily loss limit
    if check_daily_loss(daily_pnl_pct):
        reason = "Daily loss limit breached — no new trades today"
        return _blocked_result(trades, [reason], risk_status, warnings)

    # 3. Friday rule
    if check_friday_rule():
        reason = "Friday rule active — no new trades"
        return _blocked_result(trades, [reason], risk_status, warnings)

    # 4. Position sizing (conviction-based)
    trades = apply_position_sizing(trades)

    # 5. Correlation guard
    trades = apply_correlation_guard(trades)

    # 6. Per-trade size cap — drop violators
    approved: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for t in trades:
        try:
            check_position_size(t.get("ticker", "?"), t.get("size_multiplier", 0.0), current_value)
            approved.append(t)
        except ValueError as exc:
            logger.warning("Trade blocked by size check: {}", exc)
            warnings.append(str(exc))
            blocked.append(t)

    return {
        "approved_trades": approved,
        "blocked_trades": blocked,
        "block_reasons": [],
        "risk_status": risk_status,
        "warnings": warnings,
    }


def _blocked_result(
    trades: list[dict[str, Any]],
    reasons: list[str],
    risk_status: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Helper: build a validate_all result where all trades are blocked."""
    for reason in reasons:
        logger.warning("All trades blocked: {}", reason)
    return {
        "approved_trades": [],
        "blocked_trades": list(trades),
        "block_reasons": reasons,
        "risk_status": risk_status,
        "warnings": warnings,
    }
