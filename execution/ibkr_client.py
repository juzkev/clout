"""Interactive Brokers client via ib_insync.

Paper trading: connect to TWS paper (port 7497) or IB Gateway paper (port 4002).
Live trading:  connect to TWS live  (port 7496) or IB Gateway live  (port 4001).

TWS/IB Gateway must be running and API connections enabled (File → Global Config →
API → Enable ActiveX and Socket Clients).

Install: pip install ib_insync
"""

import math
from typing import Any

from loguru import logger


class IBKRClient:
    """Manages the ib_insync connection to Interactive Brokers."""

    def __init__(self) -> None:
        try:
            from ib_insync import IB
            self._ib: Any = IB()
        except ImportError:
            self._ib = None
            logger.warning("ib_insync not installed — IBKRClient unavailable. pip install ib_insync")

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> None:
        """Connect to IB Gateway / TWS. Default port 7497 = TWS paper trading."""
        self._require_ib()
        self._ib.connect(host, port, clientId=client_id, readonly=False)
        logger.info("Connected to IBKR at {}:{} (clientId={})", host, port, client_id)

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        return bool(self._ib and self._ib.isConnected())

    # ── Market data ───────────────────────────────────────────────────────────

    def get_price(self, ticker: str) -> float:
        """Return the current market price for a US ETF/stock ticker.

        Tries a snapshot quote first; falls back to the last daily close from
        historical data if the snapshot returns NaN (outside market hours).
        """
        self._require_connected()
        from ib_insync import Stock

        contract = Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        [snap] = self._ib.reqTickers(contract)
        price = snap.marketPrice()

        if math.isnan(price) or price <= 0:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
            )
            price = float(bars[-1].close) if bars else float("nan")
            logger.debug("Snapshot NaN for {} — using last close {:.2f}", ticker, price)

        return price

    def get_prices(self, tickers: list[str]) -> dict[str, float]:
        """Return current prices for multiple tickers in one round-trip."""
        self._require_connected()
        from ib_insync import Stock

        contracts = [Stock(t, "SMART", "USD") for t in tickers]
        self._ib.qualifyContracts(*contracts)
        snaps = self._ib.reqTickers(*contracts)
        result: dict[str, float] = {}
        for contract, snap in zip(contracts, snaps):
            p = snap.marketPrice()
            if math.isnan(p) or p <= 0:
                bars = self._ib.reqHistoricalData(
                    contract, endDateTime="", durationStr="2 D",
                    barSizeSetting="1 day", whatToShow="TRADES", useRTH=True,
                )
                p = float(bars[-1].close) if bars else float("nan")
            result[contract.symbol] = p
        return result

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict[str, float]:
        """Return key account values: NetLiquidation, AvailableFunds, UnrealizedPnL."""
        self._require_connected()
        tags = {"NetLiquidation", "AvailableFunds", "UnrealizedPnL", "RealizedPnL"}
        summary = {}
        for v in self._ib.accountSummary():
            if v.tag in tags:
                try:
                    summary[v.tag] = float(v.value)
                except ValueError:
                    pass
        return summary

    def get_positions(self) -> list[dict[str, Any]]:
        """Return current portfolio positions as plain dicts."""
        self._require_connected()
        result = []
        for p in self._ib.positions():
            result.append({
                "ticker": p.contract.symbol,
                "quantity": p.position,
                "avg_cost": p.avgCost,
            })
        return result

    # ── Orders ────────────────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        action: str,        # "BUY" | "SELL"
        quantity: float,
        order_type: str = "MKT",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Place an order and return a status dict."""
        self._require_connected()
        from ib_insync import Stock, MarketOrder, LimitOrder

        contract = Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        qty = max(1, round(quantity))
        if order_type == "LMT" and limit_price is not None:
            order = LimitOrder(action, qty, limit_price)
        else:
            order = MarketOrder(action, qty)

        trade = self._ib.placeOrder(contract, order)
        self._ib.sleep(1)  # let TWS acknowledge
        logger.info("Placed {} {} {} qty={} status={}", order_type, action, ticker, qty, trade.orderStatus.status)
        return {
            "order_id": trade.order.orderId,
            "status": trade.orderStatus.status,
            "ticker": ticker,
            "action": action,
            "quantity": qty,
        }

    def cancel_order(self, order_id: int) -> None:
        self._require_connected()
        for trade in self._ib.openTrades():
            if trade.order.orderId == order_id:
                self._ib.cancelOrder(trade.order)
                logger.info("Cancelled order {}", order_id)
                return
        raise ValueError(f"No open order with id {order_id}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require_ib(self) -> None:
        if self._ib is None:
            raise RuntimeError("ib_insync not installed. Run: pip install ib_insync")

    def _require_connected(self) -> None:
        self._require_ib()
        if not self._ib.isConnected():
            raise RuntimeError("Not connected to IBKR. Call connect() first.")
