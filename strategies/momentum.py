"""Cross-sectional momentum strategy.

Intended logic: rank universe by 20-day return (from price_collector),
go long the top-N tickers above their 50-day SMA when market regime is
risk-on (VIX < 20, yield curve not deeply inverted), short or cash
otherwise. Size positions by inverse volatility.
"""

from typing import Any

from signals.signal_types import Signal
from strategies.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "cross_sectional_momentum"

    def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """Rank universe by 20d return; long top-N above SMA50 in risk-on regimes."""
        raise NotImplementedError(
            "MomentumStrategy.generate_signals not yet implemented. "
            "Use price_collector data['price'] ranked by momentum_rank_20d "
            "and fred_collector data for regime filter."
        )
