"""Interactive Brokers client (stub).

Wraps ib_insync for live order routing. Requires IB Gateway or TWS running.
Install: pip install ib_insync  (commented out in requirements.txt).
"""

from typing import Any


class IBKRClient:
    """Manages the connection to Interactive Brokers."""

    def connect(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> None:
        """Connect to IB Gateway / TWS."""
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def get_positions(self) -> list[dict[str, Any]]:
        """Return current portfolio positions."""
        raise NotImplementedError

    def place_order(
        self,
        ticker: str,
        action: str,  # "BUY" | "SELL"
        quantity: float,
        order_type: str = "MKT",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Place an order and return order status dict."""
        raise NotImplementedError

    def cancel_order(self, order_id: int) -> None:
        raise NotImplementedError
