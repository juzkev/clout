"""Daily paper-trading loop.

Connects to IBKR (TWS or IB Gateway in paper mode), checks open trades for
exits, then submits today's research signals through the risk gate.

Prerequisites:
  1. pip install ib_insync
  2. TWS or IB Gateway running, paper account logged in, API enabled
     (File → Global Configuration → API → Enable ActiveX and Socket Clients)
  3. Run today's research: python -m research.run_research
     (produces data/signals_{date}.json)

Run:
  python -m scripts.paper_loop [--host 127.0.0.1] [--port 7497] [--dry-run]

  --dry-run  Checks exits and prints signals but places no orders.
  --port 7497  TWS paper. Use 4002 for IB Gateway paper.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from config.settings import settings
from execution.ibkr_client import IBKRClient
from execution.order_manager import OrderManager


def _load_signals(date_str: str) -> list[dict]:
    """Load final_trades from today's signals file."""
    path = settings.BASE_DIR / "data" / f"signals_{date_str}.json"
    if not path.exists():
        logger.warning("No signals file found at {} — skipping entry orders", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    trades = data.get("final_trades", [])
    logger.info("Loaded {} signal(s) from {}", len(trades), path.name)
    return trades


def _build_portfolio_state(client: IBKRClient) -> dict:
    summary = client.get_account_summary()
    net_liq = summary.get("NetLiquidation", 0.0)
    return {
        "current_value": net_liq,
        "peak_value": net_liq,   # simplified: no persistent peak tracking yet
        "daily_pnl_pct": 0.0,    # not fetched; drawdown guard uses current_value
    }


def run(host: str, port: int, dry_run: bool) -> None:
    settings.configure_logging()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"\n=== Paper Loop: {date_str} {'[DRY RUN]' if dry_run else ''} ===")

    client = IBKRClient()
    try:
        client.connect(host=host, port=port)
    except Exception as exc:
        logger.error("Cannot connect to IBKR at {}:{} — {}", host, port, exc)
        logger.error("Make sure TWS/IB Gateway is running with API enabled.")
        sys.exit(1)

    try:
        om = OrderManager(client)
        portfolio_state = _build_portfolio_state(client)
        net_liq = portfolio_state["current_value"]
        print(f"Portfolio value:  ${net_liq:,.2f}")

        # ── 1. Process exits ─────────────────────────────────────────────────
        print("\n--- Checking exits ---")
        if dry_run:
            from execution.exit_checker import check_exits
            exits = check_exits(client)
            if exits:
                for ex in exits:
                    print(f"  [DRY] Would close {ex['ticker']} {ex['exit_reason']} "
                          f"pnl={ex['pnl_pct']:+.2f}%")
            else:
                print("  No exit conditions triggered.")
        else:
            closed = om.process_exits(portfolio_state)
            if closed:
                for r in closed:
                    print(f"  Closed {r['ticker']} pnl={r.get('pnl_pct', '?'):+}%  outcome={r['thesis_outcome']}")
            else:
                print("  No positions closed.")

        # ── 2. Submit new signals ─────────────────────────────────────────────
        print("\n--- Submitting new signals ---")
        signals = _load_signals(date_str)

        if not signals:
            print("  No signals to submit.")
        elif dry_run:
            for sig in signals:
                print(f"  [DRY] Would submit {sig.get('ticker')} {sig.get('direction')} "
                      f"conviction={sig.get('conviction')}")
        else:
            results = om.submit_all(signals, portfolio_state)
            for sig, res in zip(signals, results):
                status = res["status"]
                ticker = sig.get("ticker")
                if status == "submitted":
                    print(f"  Submitted {ticker}  entry={res['entry_price']:.2f}  "
                          f"size=${res['size_usd']:,.0f}")
                elif status == "blocked":
                    print(f"  Blocked  {ticker}  {res.get('reasons', '')}")
                else:
                    print(f"  {status.upper()}  {ticker}  {res.get('reason', '')}")

        print("\nDone.")

    finally:
        client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="IBKR paper-trading loop")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497,
                        help="7497=TWS paper, 4002=IB Gateway paper")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check exits and print signals without placing orders")
    args = parser.parse_args()
    run(host=args.host, port=args.port, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
