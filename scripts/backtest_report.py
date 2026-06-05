"""Trade attribution backtest report.

Reads all closed trades from logs/trades/ and produces a structured
performance analysis grouped by signal type, conviction, ticker, primary
rule, and signal source.

Run:  python -m scripts.backtest_report [--days N]

Outputs:
  data/backtest/backtest_{date}.json        machine-readable report
  data/prompts/backtest_review_{date}.txt   LLM review prompt
"""

import argparse
import dataclasses
import json
import os
from datetime import datetime, timezone

from loguru import logger

from backtesting import trade_analyzer
from config.settings import settings


def _stats_to_dict(stats: trade_analyzer.GroupStats) -> dict:
    d = dataclasses.asdict(stats)
    if d.get("profit_factor") == float("inf"):
        d["profit_factor"] = "inf"
    return d


def _build_prompt(report: trade_analyzer.AnalysisReport) -> str:
    lines = [
        "You are a trading performance and risk analyst.",
        "Review the attribution report below from a multi-pass LLM swing trading pipeline.",
        "All trades are real outcomes logged by the live paper-trading system — no simulation.",
        f"Analysis window: {report.period_days} days  |  Total trades: {report.total_trades}",
        "",
        "=== OVERALL ===",
        json.dumps(_stats_to_dict(report.overall), indent=2),
        "",
        "=== BY SIGNAL TYPE ===",
        json.dumps({k: _stats_to_dict(v) for k, v in report.by_signal_type.items()}, indent=2),
        "",
        "=== BY CONVICTION LEVEL ===",
        json.dumps({k: _stats_to_dict(v) for k, v in report.by_conviction.items()}, indent=2),
        f"Conviction calibrated (≥4 outperforms ≤3): {report.conviction_calibrated}",
        "",
        "=== BY PRIMARY RULE (rule_based trades only) ===",
        json.dumps({k: _stats_to_dict(v) for k, v in report.by_primary_rule.items()}, indent=2),
        "",
        "=== BY TICKER ===",
        json.dumps({k: _stats_to_dict(v) for k, v in report.by_ticker.items()}, indent=2),
        "",
        "=== BY SIGNAL SOURCE ===",
        json.dumps({k: _stats_to_dict(v) for k, v in report.by_signal_source.items()}, indent=2),
        "",
        "=== YOUR TASK ===",
        "1. Which signal type (rule_based / situational / hybrid) is carrying results?",
        "2. Is conviction well-calibrated? What threshold adjustment would improve expected PnL?",
        "3. Which tickers have the best and worst risk-adjusted performance (Sharpe, PF)?",
        "4. For rule_based signals with ≥5 trades, is there evidence of a persistent edge?",
        "5. Which signal sources have the highest predictive contribution?",
        "6. Flag any rules showing consistent losses that should be retired.",
        "7. Recommend specific pipeline adjustments (conviction threshold, position sizing, "
        "max hold, signal source weights) based purely on this data.",
    ]
    return "\n".join(lines)


def main(days_back: int = 90) -> None:
    settings.configure_logging()
    report = trade_analyzer.load_and_analyze(days_back=days_back)

    if report.total_trades == 0:
        print(f"No closed trades in the last {days_back} days. Nothing to report.")
        return

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Save JSON report
    backtest_dir = settings.BASE_DIR / "data" / "backtest"
    os.makedirs(backtest_dir, exist_ok=True)
    report_path = backtest_dir / f"backtest_{date_str}.json"
    report_dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_days": report.period_days,
        "total_trades": report.total_trades,
        "conviction_calibrated": report.conviction_calibrated,
        "overall": _stats_to_dict(report.overall),
        "by_signal_type": {k: _stats_to_dict(v) for k, v in report.by_signal_type.items()},
        "by_conviction": {k: _stats_to_dict(v) for k, v in report.by_conviction.items()},
        "by_ticker": {k: _stats_to_dict(v) for k, v in report.by_ticker.items()},
        "by_primary_rule": {k: _stats_to_dict(v) for k, v in report.by_primary_rule.items()},
        "by_signal_source": {k: _stats_to_dict(v) for k, v in report.by_signal_source.items()},
    }
    report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    # Save LLM review prompt
    os.makedirs(settings.PROMPTS_DIR, exist_ok=True)
    prompt_path = settings.PROMPTS_DIR / f"backtest_review_{date_str}.txt"
    prompt_path.write_text(
        _build_prompt(report) + "\n\n(Paste the above into Claude.ai for a post-mortem review.)\n",
        encoding="utf-8",
    )

    # Console summary
    o = report.overall
    print(f"\n=== BACKTEST REPORT ({days_back}d, {report.total_trades} trades) ===")
    print(f"Win rate:          {o.win_rate:.0%}")
    print(f"Avg PnL:           {o.avg_pnl_pct:+.2f}%")
    print(f"Median PnL:        {o.median_pnl_pct:+.2f}%")
    print(f"Profit factor:     {o.profit_factor}")
    print(f"Trade Sharpe:      {o.sharpe_ratio:.3f}")
    print(f"Best / Worst:      {o.best_pnl_pct:+.2f}% / {o.worst_pnl_pct:+.2f}%")
    print(f"Conviction calib.: {'YES' if report.conviction_calibrated else 'NO'}")
    print()

    print("By signal type:")
    for st, s in report.by_signal_type.items():
        if s.count > 0:
            print(f"  {st:<15}  n={s.count:<4}  win={s.win_rate:.0%}  avg={s.avg_pnl_pct:+.2f}%  PF={s.profit_factor}")

    if report.by_conviction:
        print("\nBy conviction:")
        for key in sorted(report.by_conviction):
            s = report.by_conviction[key]
            print(f"  {key}  n={s.count:<4}  win={s.win_rate:.0%}  avg={s.avg_pnl_pct:+.2f}%")

    if report.by_primary_rule:
        print("\nBy primary rule:")
        for rule, s in report.by_primary_rule.items():
            print(f"  [{s.count}tx] {rule[:64]}  win={s.win_rate:.0%}  avg={s.avg_pnl_pct:+.2f}%")

    print(f"\nReport: {report_path}")
    print(f"Prompt: {prompt_path}")
    logger.info("Backtest report written: {}", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade attribution backtest report")
    parser.add_argument("--days", type=int, default=90, help="Look-back window in days (default 90)")
    args = parser.parse_args()
    main(days_back=args.days)
