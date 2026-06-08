"""Signal data types used across strategies and execution."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class Signal:
    ticker: str
    direction: Literal["long", "short", "cash"]
    conviction: int  # 1–5
    catalyst: str
    entry: str  # e.g. "market_open" or "limit_at_182.50"
    stop_loss_pct: float
    target_pct: float
    holding_days: int
    reasoning: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "conviction": self.conviction,
            "catalyst": self.catalyst,
            "entry": self.entry,
            "stop_loss_pct": self.stop_loss_pct,
            "target_pct": self.target_pct,
            "holding_days": self.holding_days,
            "reasoning": self.reasoning,
            "generated_at": self.generated_at,
        }


def signals_from_llm_output(llm_result: dict[str, Any]) -> list[Signal]:
    """Convert llm_client.get_trade_ideas() output into Signal objects."""
    signals = []
    for idea in llm_result.get("trade_ideas", []):
        try:
            signals.append(
                Signal(
                    ticker=idea["ticker"],
                    direction=idea["direction"],
                    conviction=int(idea["conviction"]),
                    catalyst=idea["catalyst"],
                    entry=idea["entry"],
                    stop_loss_pct=float(idea["stop_loss_pct"]),
                    target_pct=float(idea["target_pct"]),
                    holding_days=int(idea["holding_days"]),
                    reasoning=idea["reasoning"],
                    generated_at=llm_result.get("generated_at", ""),
                )
            )
        except (KeyError, ValueError) as exc:
            import logging
            logging.getLogger(__name__).warning("Skipping malformed trade idea: %s", exc)
    return signals
