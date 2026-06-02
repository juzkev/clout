"""Entry point for the trading research pipeline.

Run with:
    python -m research.run_research

Steps:
  1. Run all 5 collectors in parallel (ThreadPoolExecutor)
  2. Build and save the LLM prompt
  3. Call the configured LLM provider (skipped if non-interactive and manual mode)
  4. Save results JSON and print a human summary
"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

from config import settings
from research.collectors import (
    fred_collector,
    sentiment_collector,
    crypto_collector,
    news_collector,
    price_collector,
)
from research.prompt_builder import build_prompt, save_prompt

logger = logging.getLogger(__name__)


def _run_collector(name: str, fn) -> tuple[str, Any]:
    """Run a collector function, catching all exceptions."""
    try:
        result = fn()
        logger.info("Collector '%s' completed successfully", name)
        return name, result
    except Exception as exc:
        logger.error("Collector '%s' failed: %s", name, exc)
        return name, {}


def run() -> dict[str, Any] | None:
    settings.configure_logging()
    settings.ensure_dirs()

    now = datetime.now(timezone.utc)
    print(f"\n=== Trading Research Run: {now.isoformat()} ===\n")

    collectors = {
        "fred": fred_collector.collect,
        "sentiment": sentiment_collector.collect,
        "crypto": crypto_collector.collect,
        "news": news_collector.collect,
        "price": price_collector.collect,
    }

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_run_collector, name, fn): name for name, fn in collectors.items()}
        for future in as_completed(futures):
            name, data = future.result()
            results[name] = data

    fred = results.get("fred", {})
    sentiment = results.get("sentiment", {})
    crypto = results.get("crypto", {})
    news = results.get("news", {})
    price = results.get("price", {})

    print("Building prompt...")
    prompt = build_prompt(fred, sentiment, crypto, price, news)
    prompt_path = save_prompt(prompt)
    print(f"Prompt saved to: {prompt_path}\n")

    # Skip blocking LLM call in non-interactive environments
    is_interactive = sys.stdin.isatty()
    if settings.LLM_PROVIDER == "manual" and not is_interactive:
        print("Non-interactive mode: skipping LLM call.")
        print(f"Paste the prompt at {prompt_path} into Claude.ai to get trade ideas.")
        return None

    from research.llm_client import get_trade_ideas
    try:
        print(f"Calling LLM provider: {settings.LLM_PROVIDER}")
        ideas = get_trade_ideas(prompt)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        print(f"\nLLM call failed: {exc}")
        return None

    # Save result
    today = now.strftime("%Y-%m-%d")
    out_path = settings.DATA_DIR / f"research_{today}.json"
    out_path.write_text(json.dumps(ideas, indent=2), encoding="utf-8")
    print(f"Results saved to: {out_path}\n")

    # Human-readable summary
    regime = ideas.get("market_regime", "unknown")
    reasoning = ideas.get("regime_reasoning", "")
    print(f"Market Regime: {regime.upper()}")
    print(f"Reasoning:     {reasoning}\n")
    print("Trade Ideas:")
    for idea in ideas.get("trade_ideas", []):
        ticker = idea.get("ticker", "?")
        direction = idea.get("direction", "?")
        conviction = idea.get("conviction", "?")
        catalyst = idea.get("catalyst", "")
        print(f"  [{conviction}/5] {ticker} {direction.upper()} — {catalyst}")

    risks = ideas.get("risks", [])
    if risks:
        print("\nKey Risks:")
        for r in risks:
            print(f"  • {r}")

    print()
    return ideas


if __name__ == "__main__":
    run()
