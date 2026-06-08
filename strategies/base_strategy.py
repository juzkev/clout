"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Any

from signals.signal_types import Signal


class BaseStrategy(ABC):
    """All strategies must implement generate_signals()."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @abstractmethod
    def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """Generate trade signals from collected market data.

        Args:
            data: dict containing price, fred, sentiment, crypto, news data
                  as returned by the respective collectors.

        Returns:
            List of Signal objects. Empty list = no trade.
        """
        raise NotImplementedError
