"""Weekly trade review: summarise the last 7 days and build a Claude prompt.

Run:  python -m scripts.weekly_review

Produces console stats and a prompt file at data/prompts/weekly_review_{date}.txt
to paste into Claude.ai for a qualitative post-mortem.
"""

import json
import os
from datetime import datetime, timezone

from loguru import logger

from config.settings import settings
from execution import trade_logger


def _build_prompt(summary: dict, closed_trades: list[dict]) -> str:
    too_few = summary["total_trades"] < 10
    lines: list[str] = [
        "You are a trading performance and risk reviewer.",
        "Review the past week of closed swing trades below and produce a candid post-mortem.",
        "",
        "=== WEEKLY SUMMARY STATS ===",
        json.dumps(summary, indent=2, default=str),
        "",
        "=== CLOSED TRADES (with thesis and outcome) ===",
    ]
    for t in closed_trades:
        lines.append(json.dumps({
            "ticker": t.get("ticker"),
            "direction": t.get("direction"),
            "conviction": t.get("conviction"),
            "pnl_pct": t.get("pnl_pct"),
            "thesis": t.get("thesis"),
            "invalidation": t.get("invalidation"),
            "thesis_outcome": t.get("thesis_outcome"),
            "thesis_outcome_notes": t.get("thesis_outcome_notes"),
            "signal_sources": t.get("signal_sources"),
        }, indent=2, default=str))
        lines.append("")

    lines += [
        "=== PERFORMANCE BY SIGNAL TYPE ===",
        json.dumps(summary.get("by_signal_type", {}), indent=2, default=str),
        "",
        "=== YOUR TASK ===",
        "1. Identify which signal sources predicted correctly vs incorrectly.",
        "2. For each loss, distinguish a BAD SIGNAL from BAD RISK MANAGEMENT as the cause.",
        "3. Flag any pattern of persistent bullish or bearish bias.",
        "4. Recommend whether to adjust conviction thresholds for any signal source.",
        "5. Compare performance across signal types (rule_based vs situational vs "
        "hybrid): which is carrying the results, and is any type a net drag?",
        "6. Flag any rule_based signal (with its primary_rule) that has accumulated "
        "enough trades to justify a formal backtest, and state the rule explicitly.",
    ]
    if too_few:
        lines.append(
            f"7. NOTE: only {summary['total_trades']} trade(s) this week (<10) — "
            "explicitly caution that this is too small a sample to draw firm conclusions."
        )
    return "\n".join(lines)


def main() -> None:
    settings.configure_logging()
    summary = trade_logger.summarise_week()
    closed = trade_logger.get_closed_trades(days_back=7)

    prompt = _build_prompt(summary, closed)

    os.makedirs(settings.PROMPTS_DIR, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = settings.PROMPTS_DIR / f"weekly_review_{date_str}.txt"
    out_path.write_text(
        prompt + "\n\n(Paste the above into Claude.ai for your weekly review.)\n",
        encoding="utf-8",
    )

    print("\n=== WEEKLY REVIEW ===")
    print(f"Trades (7d):     {summary['total_trades']}")
    print(f"Winners/Losers:  {summary['winners']}/{summary['losers']}")
    print(f"Win rate:        {summary['win_rate']:.0%}")
    print(f"Avg PnL:         {summary['avg_pnl_pct']:+.2f}%")
    print(f"Thesis accuracy: {summary['thesis_accuracy']:.0%}")
    print(f"Signal hits:     {summary['signal_source_hits']}")
    if summary["total_trades"] < 10:
        print("\n⚠ Fewer than 10 trades — sample too small for firm conclusions.")
    print(f"\nPrompt saved to: {out_path}")
    logger.info("Weekly review prompt written to {}", out_path)


if __name__ == "__main__":
    main()
