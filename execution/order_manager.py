"""Order manager: bridges research signals → IBKR execution + trade logging."""

import math
from typing import Any

from loguru import logger

from execution import risk_guard, trade_logger
from execution.exit_checker import check_exits as _check_exits
from execution.ibkr_client import IBKRClient


class OrderManager:
    """Submit signals and process exits through the IBKR client."""

    def __init__(self, client: IBKRClient) -> None:
        self._client = client

    # ── Signal submission ─────────────────────────────────────────────────────

    def submit_signal(
        self,
        signal: dict[str, Any],
        portfolio_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply risk gate → place order → log open trade.

        Args:
            signal:          A final_trade dict from merge_final_signals.
            portfolio_state: Dict with current_value, peak_value, daily_pnl_pct.

        Returns:
            Status dict with "status": "submitted" | "blocked" | "skipped" | "error".
        """
        result = risk_guard.check_pre_trade([signal], portfolio_state)
        approved = result["approved_trades"]

        if not approved:
            logger.info("Signal {} blocked: {}", signal.get("ticker"), result["block_reasons"])
            return {"status": "blocked", "reasons": result["block_reasons"]}

        trade = approved[0]
        ticker = trade["ticker"]
        direction = trade["direction"]
        size_multiplier = trade.get("size_multiplier", 0.0)
        size_usd = portfolio_state["current_value"] * size_multiplier

        try:
            price = self._client.get_price(ticker)
        except Exception as exc:
            logger.error("Cannot get price for {}: {}", ticker, exc)
            return {"status": "error", "reason": str(exc)}

        if math.isnan(price) or price <= 0:
            return {"status": "error", "reason": f"invalid price {price} for {ticker}"}

        quantity = size_usd / price
        if quantity < 1:
            logger.warning("{}: ${:.0f} too small for 1 share @ {:.2f} — skipped", ticker, size_usd, price)
            return {"status": "skipped", "reason": "position too small for 1 share"}

        action = "BUY" if direction == "long" else "SELL"
        order_result = self._client.place_order(ticker, action, quantity)
        trade_logger.open_trade(trade, entry_price=price, size_usd=size_usd)

        logger.info(
            "Submitted {} {} {:.0f} shares @ {:.2f} (${:,.0f}, {:.1%} of portfolio)",
            action, ticker, math.floor(quantity), price, size_usd, size_multiplier,
        )
        return {
            "status": "submitted",
            "order": order_result,
            "entry_price": price,
            "size_usd": size_usd,
        }

    def submit_all(
        self,
        signals: list[dict[str, Any]],
        portfolio_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Submit each signal in `signals` and return the results list."""
        return [self.submit_signal(s, portfolio_state) for s in signals]

    # ── Exit processing ───────────────────────────────────────────────────────

    def check_exits(self) -> list[dict[str, Any]]:
        """Return raw exit signals for open trades (stop/target/hold expiry)."""
        return _check_exits(self._client)

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Return currently open IBKR orders as plain dicts."""
        return [
            {
                "order_id": t.order.orderId,
                "ticker": t.contract.symbol,
                "action": t.order.action,
                "quantity": t.order.totalQuantity,
                "status": t.orderStatus.status,
            }
            for t in self._client._ib.openTrades()
        ]

    def process_exits(self, portfolio_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Close trades that hit stop/target/max-hold and log outcomes."""
        exits = _check_exits(self._client)
        closed: list[dict[str, Any]] = []
        positions = {p["ticker"]: p for p in self._client.get_positions()}

        for ex in exits:
            ticker = ex["ticker"]
            open_date = ex["open_date"]
            direction = ex["direction"]
            current_price = ex["current_price"]
            exit_reason = ex["exit_reason"]

            close_action = "SELL" if direction == "long" else "BUY"
            pos = positions.get(ticker)
            if not pos:
                logger.warning("Exit signal for {} but no IBKR position — logging only", ticker)
            else:
                qty = abs(pos["quantity"])
                if qty > 0:
                    try:
                        self._client.place_order(ticker, close_action, qty)
                    except Exception as exc:
                        logger.error("Close order failed for {}: {}", ticker, exc)
                        continue

            try:
                record = trade_logger.close_trade(
                    ticker=ticker,
                    open_date=open_date,
                    exit_price=current_price,
                    outcome=exit_reason,
                    notes=f"Automated exit: {exit_reason} @ {current_price:.4f}",
                )
                closed.append(record)
                logger.info("Closed {} pnl={:+.2f}% outcome={}", ticker, record.get("pnl_pct", 0), exit_reason)
            except Exception as exc:
                logger.error("Failed to log close for {}: {}", ticker, exc)

        return closed
