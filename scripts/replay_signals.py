"""Retrospectively paper-test saved signals against real historical prices.

Loads a signals file (signals/{date}_signals.json), replays trades through
actual yfinance OHLCV, and writes the outcomes to logs/trades/ as closed
records so backtest_report.py can analyze them.

By default replays only final_trades (Pass 3-approved). With --include-rejected
it also replays the raw Pass 2 ideas that were filtered out, tagged
source='replay_rejected' — useful for checking whether Pass 3's vetoes
actually saved money.

Requires network access to yfinance — run on a machine that can reach it.

Run:
  python -m scripts.replay_signals signals/2026-06-04_signals.json
  python -m scripts.replay_signals signals/2026-06-04_signals.json --include-rejected
  python -m scripts.replay_signals --all [--include-rejected]
  python -m scripts.replay_signals <file> --dry-run
"""

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from backtesting import replay
from config.settings import settings

_SIGNALS_DIR = settings.BASE_DIR / "signals"
_TRADES_DIR = settings.BASE_DIR / "logs" / "trades"


def _yfinance_fetch(ticker: str, start_date: str, num_bars: int) -> pd.DataFrame | None:
    """Fetch daily OHLCV starting the trading day AFTER `start_date`."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. pip install yfinance")
        return None

    yf_ticker = "BTC-USD" if ticker in ("IBIT", "BTC") else ticker
    start = (pd.Timestamp(start_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(start_date) + pd.Timedelta(days=num_bars * 2 + 5)).strftime("%Y-%m-%d")

    try:
        df = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=False)
    except Exception as exc:
        logger.error("yfinance download failed for {}: {}", yf_ticker, exc)
        return None

    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.head(num_bars)


def _write_record(record: dict, suffix: str = "replay") -> Path:
    os.makedirs(_TRADES_DIR, exist_ok=True)
    fname = f"{record['ticker']}_{record['entry_date']}_{suffix}_closed.json"
    path = _TRADES_DIR / fname
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def _rejected_ideas(signals: dict) -> list[dict]:
    """Return Pass 2 ideas that did NOT make it into final_trades.

    Works with both old signals files (no pass2_trade_ideas key) and new ones.
    For old files, falls back to reconstructing from final_trades tickers alone,
    which means the rejected set will be empty — that's correct behaviour for
    pre-patch files.
    """
    all_p2 = signals.get("pass2_trade_ideas", [])
    if not all_p2:
        return []
    approved_tickers = {t["ticker"] for t in signals.get("final_trades", [])}
    return [t for t in all_p2 if t.get("ticker") not in approved_tickers]


def _print_record(rec: dict, dry_run: bool, label: str = "") -> None:
    tag = label or ("" if not dry_run else "")
    logged = "" if dry_run else " [logged]"
    print(
        f"    {rec['ticker']:<6} {rec['direction'].upper():<5} "
        f"entry={rec['entry_price']:<10.4f} exit={rec['exit_price']:<10.4f} "
        f"pnl={rec['pnl_pct']:+6.2f}%  {rec['thesis_outcome']}{tag}{logged}"
    )


def replay_file(path: Path, dry_run: bool, include_rejected: bool) -> int:
    if not path.exists():
        logger.error("Signals file not found: {}", path)
        return 0

    signals = json.loads(path.read_text(encoding="utf-8"))
    date = signals.get("date", path.stem.replace("_signals", ""))
    final_trades = signals.get("final_trades", [])
    rejected = _rejected_ideas(signals) if include_rejected else []

    if not final_trades and not rejected:
        print(
            f"  {path.name}: nothing to replay "
            f"(0 approved, 0 rejected ideas persisted — "
            f"{signals.get('trades_filtered_out', 0)} filtered at runtime)"
        )
        return 0

    print(f"\n  {path.name} ({date}):")
    count = 0

    # ── Approved trades ───────────────────────────────────────────────────────
    if final_trades:
        approved_records = replay.replay_signals(signals, _yfinance_fetch)
        if approved_records:
            print(f"    [APPROVED — {len(approved_records)} trade(s)]")
            for rec in approved_records:
                _print_record(rec, dry_run)
                if not dry_run:
                    _write_record(rec, suffix="replay")
                count += 1
        else:
            print(f"    [APPROVED] {len(final_trades)} trade(s) but no price data fetched")
    else:
        print("    [APPROVED] none")

    # ── Rejected ideas (Pass 2 only, filtered by Pass 3) ─────────────────────
    if include_rejected:
        if rejected:
            print(f"    [REJECTED by Pass 3 — {len(rejected)} idea(s)]")
            for idea in rejected:
                rec = replay.replay_trade(idea, date, _yfinance_fetch)
                if rec:
                    rec["source"] = "replay_rejected"
                    _print_record(rec, dry_run, label=" [rejected]")
                    if not dry_run:
                        _write_record(rec, suffix="replay_rejected")
                    count += 1
                else:
                    print(f"    [REJECTED] {idea.get('ticker')} — no price data")
        else:
            if signals.get("pass2_trade_ideas") is None:
                print("    [REJECTED] pass2_trade_ideas not in this file — re-run research to capture them")
            else:
                print("    [REJECTED] none (all Pass 2 ideas were approved)")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospective signal replay / paper test")
    parser.add_argument("file", nargs="?", help="Path to a signals JSON file")
    parser.add_argument("--all", action="store_true", help="Replay every signals/*_signals.json")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Also replay Pass 2 ideas that were filtered out by Pass 3",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing to logs/trades/")
    args = parser.parse_args()

    settings.configure_logging()

    if args.all:
        files = sorted(Path(p) for p in glob.glob(str(_SIGNALS_DIR / "*_signals.json")))
        if not files:
            print(f"No signals files found in {_SIGNALS_DIR}")
            return
    elif args.file:
        files = [Path(args.file)]
    else:
        parser.error("Provide a signals file path or --all")

    label = "[DRY RUN] " if args.dry_run else ""
    rejected_label = "+ rejected ideas " if args.include_rejected else ""
    print(f"=== Replaying {len(files)} signals file(s) {label}{rejected_label}===")

    total = sum(replay_file(f, args.dry_run, args.include_rejected) for f in files)
    print(f"\nTotal trades replayed: {total}")
    if not args.dry_run and total:
        print("Run `python -m scripts.backtest_report` to analyze the replayed trades.")


if __name__ == "__main__":
    main()
