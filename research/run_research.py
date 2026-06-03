"""Orchestrator for the multi-pass trading research pipeline.

Flow:
  Collectors (8, parallel) → Pass 1 (Regime) → Pass 2 (Trade Ideas)
                           → Pass 3 (Stress Test) → Merge → signals/

CLI:
  python -m research.run_research [--provider manual|deepseek|claude]
                                  [--dry-run] [--pass1-only] [--no-pass3]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from research.collectors import (
    calendar_collector,
    cot_collector,
    crypto_collector,
    fred_collector,
    news_collector,
    price_collector,
    sentiment_collector,
    trends_collector,
)
from research.llm_client import (
    merge_final_signals,
    run_pass1_regime,
    run_pass2_ideas,
    run_pass3_stress_test,
)

_SIGNALS_DIR = settings.BASE_DIR / "signals"


def _run_safe(name: str, fn) -> tuple[str, dict]:
    """Run a collector, returning (name, {}) on any exception."""
    try:
        result = fn()
        logger.info("✓ {} collected", name)
        return name, result
    except Exception as exc:
        logger.warning("⚠ {} failed: {}", name, exc)
        return name, {}


def _print_data_summary(collected: dict[str, dict]) -> None:
    print("\n=== DRY RUN — DATA SUMMARY ===")
    for name, data in sorted(collected.items()):
        if data:
            sample_keys = list(data.keys())[:5]
            print(f"  ✓ {name}: {len(data)} top-level keys — {sample_keys}")
        else:
            print(f"  ⚠ {name}: empty (failed or no API key)")
    print()


def _print_summary(signals: dict) -> None:
    regime = signals.get("market_regime", {})
    regime_name = regime.get("regime", "unknown").upper()
    confidence = regime.get("confidence", "?")
    reasoning = regime.get("regime_reasoning", "")
    final_trades = signals.get("final_trades", [])
    filtered = signals.get("trades_filtered_out", 0)
    assessment = signals.get("overall_assessment", "")

    print(f"\n{'='*60}")
    print(f"=== REGIME: {regime_name} (confidence {confidence}/5) ===")
    if reasoning:
        print(reasoning)
    print()

    if final_trades:
        print(f"FINAL TRADE IDEAS ({len(final_trades)} passed):")
        for t in final_trades:
            size_pct = int(t.get("size_multiplier", 1.0) * 100)
            print(
                f"  {t['ticker']:<6} {t['direction'].upper():<5} | "
                f"conviction {t['conviction']}/5 | "
                f"{size_pct}% size | "
                f"{t.get('catalyst', '')}"
            )
    else:
        print("FINAL TRADE IDEAS: None — no high-conviction setups passed stress test")

    if filtered:
        print(f"\n({filtered} idea(s) filtered by stress test)")

    if assessment:
        print(f"\nASSESSMENT: {assessment}")
    print("=" * 60)


def run(
    provider: str | None = None,
    dry_run: bool = False,
    pass1_only: bool = False,
    no_pass3: bool = False,
) -> dict | None:
    settings.ensure_dirs()
    os.makedirs(_SIGNALS_DIR, exist_ok=True)

    provider = provider or settings.LLM_PROVIDER
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"\n=== Trading Research Run: {date_str} | Provider: {provider} ===\n")

    # ── Step 2: Run all 8 collectors in parallel ──────────────────────────────
    collector_fns = {
        "fred": fred_collector.collect,
        "sentiment": sentiment_collector.collect,
        "crypto": crypto_collector.collect,
        "news": news_collector.collect,
        "price": price_collector.collect,
        "trends": trends_collector.collect,
        "cot": cot_collector.collect,
        "calendar": calendar_collector.collect,
    }

    collected: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_run_safe, name, fn): name for name, fn in collector_fns.items()}
        for future in as_completed(futures):
            name, data = future.result()
            collected[name] = data

    if dry_run:
        _print_data_summary(collected)
        return None

    # Manual mode needs an interactive terminal
    if provider == "manual" and not sys.stdin.isatty():
        logger.warning("Manual mode requires an interactive terminal — skipping LLM calls")
        print(f"Collectors finished. Run interactively with --provider manual to get trade ideas.")
        return None

    # ── Step 3: Pass 1 — Regime classification ────────────────────────────────
    try:
        regime = run_pass1_regime(
            macro_data=collected.get("fred", {}),
            sentiment_data=collected.get("sentiment", {}),
            price_data=collected.get("price", {}),
            calendar_data=collected.get("calendar", {}),
            provider=provider,
            date_str=date_str,
        )
    except RuntimeError as exc:
        logger.error("Pass 1 failed: {}", exc)
        return None

    if pass1_only:
        print(f"\nRegime: {regime.get('regime', 'unknown').upper()} "
              f"(confidence {regime.get('confidence', '?')}/5)")
        print(f"Reasoning: {regime.get('regime_reasoning', '')}")
        return {"market_regime": regime}

    # ── Step 4: Pass 2 — Trade ideas ─────────────────────────────────────────
    try:
        trade_ideas = run_pass2_ideas(
            regime=regime,
            price_data=collected.get("price", {}),
            crypto_data=collected.get("crypto", {}),
            cot_data=collected.get("cot", {}),
            trends_data=collected.get("trends", {}),
            news_data=collected.get("news", {}),
            provider=provider,
            date_str=date_str,
        )
    except RuntimeError as exc:
        logger.error("Pass 2 failed: {}", exc)
        return None

    # ── Step 5: Early exit if no trade ideas ─────────────────────────────────
    if not trade_ideas.get("trade_ideas"):
        reason = trade_ideas.get("no_trade_reason", "No trade ideas generated")
        logger.info("No trade ideas: {}", reason)
        minimal: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "date": date_str,
            "market_regime": regime,
            "final_trades": [],
            "trades_filtered_out": 0,
            "portfolio_risks": [],
            "overall_assessment": reason,
            "no_trade_reason": reason,
        }
        out_path = _SIGNALS_DIR / f"{date_str}_signals.json"
        out_path.write_text(json.dumps(minimal, indent=2), encoding="utf-8")
        logger.info("Minimal signals saved to {}", out_path)
        _print_summary(minimal)
        return minimal

    # ── Step 6: Pass 3 — Stress test ─────────────────────────────────────────
    if no_pass3:
        logger.warning("Pass 3 (stress test) skipped via --no-pass3")
        stress_test: dict[str, Any] = {
            "reviewed_ideas": [],
            "portfolio_level_risks": [],
            "overall_assessment": "Stress test skipped (--no-pass3).",
        }
    else:
        try:
            stress_test = run_pass3_stress_test(
                regime=regime,
                trade_ideas=trade_ideas,
                macro_data=collected.get("fred", {}),
                provider=provider,
                date_str=date_str,
            )
        except RuntimeError as exc:
            logger.error("Pass 3 failed: {} — continuing without stress test", exc)
            stress_test = {
                "reviewed_ideas": [],
                "portfolio_level_risks": [],
                "overall_assessment": f"Stress test failed: {exc}",
            }

    # ── Step 7: Merge and save ────────────────────────────────────────────────
    signals = merge_final_signals(
        regime=regime,
        trade_ideas=trade_ideas,
        stress_test=stress_test,
        date_str=date_str,
    )
    out_path = _SIGNALS_DIR / f"{date_str}_signals.json"
    out_path.write_text(json.dumps(signals, indent=2), encoding="utf-8")
    logger.info("Signals saved to {}", out_path)

    # ── Step 8: Summary ───────────────────────────────────────────────────────
    _print_summary(signals)
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-pass LLM trading research pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["manual", "deepseek", "claude"],
        default=None,
        help="LLM provider (overrides LLM_PROVIDER from .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all collectors, print data summary, skip LLM calls",
    )
    parser.add_argument(
        "--pass1-only",
        action="store_true",
        help="Run Pass 1 (regime) only and stop",
    )
    parser.add_argument(
        "--no-pass3",
        action="store_true",
        help="Skip Pass 3 stress test (faster, less safe)",
    )
    args = parser.parse_args()
    run(
        provider=args.provider,
        dry_run=args.dry_run,
        pass1_only=args.pass1_only,
        no_pass3=args.no_pass3,
    )


if __name__ == "__main__":
    main()
