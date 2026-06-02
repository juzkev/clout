"""BTC funding-rate mean-reversion strategy.

Intended logic: when BTC perpetual funding rate is extremely positive
(>0.05% per 8h, "very_elevated_longs_strongly_bearish") take a short
position on IBIT or go to cash. When funding is extremely negative
(<-0.05%), go long IBIT. Scale position by deviation from historical
funding mean. Exit when funding reverts to neutral band.
"""

from typing import Any

from signals.signal_types import Signal
from strategies.base_strategy import BaseStrategy


class BTCFundingRateStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "btc_funding_rate_reversion"

    def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """Generate contrarian signal based on BTC perpetual funding rate extremes."""
        raise NotImplementedError(
            "BTCFundingRateStrategy.generate_signals not yet implemented. "
            "Key input: data['crypto']['btc_funding_rate'] and "
            "data['crypto']['funding_interpretation']."
        )
