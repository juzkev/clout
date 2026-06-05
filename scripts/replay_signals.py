"""Retrospectively paper-test saved signals against real historical prices.

Loads a signals file (signals/{date}_signals.json), replays each final_trade
through actual yfinance OHLCV, and writes the outcomes to logs/trades/ as
closed records (tagged source='replay') so backtest_report.py can analyze them.

Requires network access to yfinance — run on a machine that can reach it
(this won't work in a locked-down CI/sandbox environment).

Run:
  python -m scripts.replay_signals signals/2026-06-04_signals.json
  python -m scripts.replay_signals --all          # replay every signals file
  python -m scripts.replay_signals <file> --dry-run   # print, don't write
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
    if isinstance(df.columns, pd.MultiIndex):  # flatten yfinance multiindex
        df.columns = df.columns.get_level_values(0)
    return df.head(num_bars)


def _write_record(record: dict) -> Path:
    os.makedirs(_TRADES_DIR, exist_ok=True)
    fname = f"{record['ticker']}_{record['entry_date']}_replay_closed.json"
    path = _TRADES_DIR / fname
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def replay_file(path: Path, dry_run: bool) -> int:
    if not path.exists():
        logger.error("Signals file not found: {}", path)
        return 0

    signals = json.loads(path.read_text(encoding="utf-8"))
    final_trades = signals.get("final_trades", [])
    date = signals.get("date", path.stem.replace("_signals", ""))

    if not final_trades:
        print(f"  {path.name}: 0 final_trades — nothing to replay "
              f"({signals.get('trades_filtered_out', 0)} filtered by stress test)")
        return 0

    records = replay.replay_signals(signals, _yfinance_fetch)
    if not records:
        print(f"  {path.name}: {len(final_trades)} trade(s) but no price data fetched")
        return 0

    print(f"\n  {path.name} ({date}):")
    for rec in records:
        tag = "" if dry_run else " [logged]"
        print(f"    {rec['ticker']:<6} {rec['direction'].upper():<5} "
              f"entry={rec['entry_price']:<10.4f} exit={rec['exit_price']:<10.4f} "
              f"pnl={rec['pnl_pct']:+6.2f}%  {rec['thesis_outcome']}{tag}")
        if not dry_run:
            _write_record(rec)

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospective signal replay / paper test")
    parser.add_argument("file", nargs="?", help="Path to a signals JSON file")
    parser.add_argument("--all", action="store_true", help="Replay every signals/*_signals.json")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to logs/trades/")
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

    print(f"=== Replaying {len(files)} signals file(s) {'[DRY RUN]' if args.dry_run else ''} ===")
    total = sum(replay_file(f, args.dry_run) for f in files)
    print(f"\nTotal trades replayed: {total}")
    if not args.dry_run and total:
        print("Run `python -m scripts.backtest_report` to analyze the replayed trades.")


if __name__ == "__main__":
    main()
