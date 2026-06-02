"""RSI mean-reversion strategy.

Intended logic: within the universe, identify tickers where RSI(14) < 30
(oversold) and price is above the 50-day SMA (uptrend intact). Go long
with a stop below the recent swing low. Exit when RSI > 50 or stop is
hit. Filter by VIX < 25 to avoid reversion traps in high-volatility
environments.
"""

from typing import Any

from signals.signal_types import Signal
from strategies.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """Identify RSI-oversold tickers in uptrend for mean-reversion longs."""
        raise NotImplementedError(
            "MeanReversionStrategy.generate_signals not yet implemented. "
            "Key inputs: data['price'][ticker]['rsi_14'], "
            "data['price'][ticker]['above_sma50'], "
            "data['fred']['VIXCLS']['latest']."
        )
