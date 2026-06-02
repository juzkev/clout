"""Pre-trade risk checks and position sizing."""

from signals.signal_types import Signal


class RiskManager:
    """Enforces position limits and sizes trades by risk budget."""

    def __init__(
        self,
        max_position_pct: float = 0.20,  # max 20% of portfolio per position
        max_portfolio_risk_pct: float = 0.02,  # risk at most 2% of portfolio per trade
        max_open_positions: int = 5,
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_open_positions = max_open_positions

    def position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
    ) -> float:
        """Return number of shares/units to trade for this signal.

        Sizes by fixed-fractional risk: (portfolio_value * max_portfolio_risk_pct)
        / (current_price * stop_loss_pct / 100).
        Also caps at max_position_pct of portfolio.
        """
        raise NotImplementedError

    def approve(self, signal: Signal, open_positions: list[dict]) -> tuple[bool, str]:
        """Return (approved, reason). Checks max open positions and concentration."""
        raise NotImplementedError
