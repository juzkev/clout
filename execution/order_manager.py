"""Order lifecycle management."""

from typing import Any

from execution.ibkr_client import IBKRClient
from signals.signal_types import Signal


class OrderManager:
    """Translates Signals into orders and tracks their lifecycle."""

    def __init__(self, client: IBKRClient):
        self.client = client

    def submit_signal(self, signal: Signal, quantity: float) -> dict[str, Any]:
        """Convert a Signal to an order and submit via IBKRClient.

        Handles entry type (market_open vs limit_at_X), attaches
        stop-loss bracket, and records order metadata.
        """
        raise NotImplementedError

    def check_exits(self) -> list[dict[str, Any]]:
        """Check open positions against stop/target levels; return exit orders filled."""
        raise NotImplementedError

    def get_open_orders(self) -> list[dict[str, Any]]:
        raise NotImplementedError
