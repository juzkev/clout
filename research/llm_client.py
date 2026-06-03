"""Multi-pass LLM pipeline for trading research.

Three-pass architecture:
  Pass 1 — Regime classification  (macro + sentiment + price + calendar)
  Pass 2 — Trade idea generation  (regime + price + crypto + COT + trends + news)
  Pass 3 — Adversarial stress test (separate model where possible)

Each pass output is validated JSON before the next pass starts.
All prompts saved to data/prompts/ for full auditability.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from config import settings


# ── System prompts ────────────────────────────────────────────────────────────

_PASS1_SYSTEM = (
    "You are a quantitative macro analyst.\n"
    "Analyse the provided market data and classify the current market regime.\n"
    "Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."
)

_PASS2_SYSTEM = (
    "You are a systematic swing trading analyst.\n"
    "Given a market regime assessment and asset-specific signals, generate specific "
    "trade ideas with clear entry logic and risk parameters.\n"
    "Respond ONLY with valid JSON. No markdown, no explanation outside the JSON.\n"
    "Be selective — 0 trade ideas is a valid output if conditions are not right."
)

_PASS3_SYSTEM = (
    "You are a risk manager and devil's advocate.\n"
    "Your job is to CHALLENGE trade ideas — find flaws, hidden risks, and reasons "
    "they could fail.\n"
    "You are NOT trying to be helpful to the trader.\n"
    "You are trying to protect capital.\n"
    "Respond ONLY with valid JSON."
)


# ── Core utilities ────────────────────────────────────────────────────────────

def _parse_json_response(
    raw: str,
    retry_fn: Callable[[str], str] | None = None,
) -> dict:
    """Strip markdown fences and parse JSON. Retry once with fix prompt on failure."""
    def _strip(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            end = len(lines) - 1 if lines and lines[-1].strip() == "```" else len(lines)
            return "\n".join(lines[1:end]).strip()
        return text

    try:
        return json.loads(_strip(raw))
    except json.JSONDecodeError:
        if retry_fn is not None:
            logger.warning("JSON parse failed — retrying with fix prompt")
            try:
                fixed = retry_fn("Fix this invalid JSON and return only valid JSON:\n" + raw)
                return json.loads(_strip(fixed))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"JSON parse failed. Response (first 500 chars): {raw[:500]}")


def _save_prompt(prompt: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt)


def _call_deepseek(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _call_claude(system_prompt: str, user_prompt: str) -> str:
    import anthropic
    if not settings.CLAUDE_API_KEY:
        raise RuntimeError("CLAUDE_API_KEY not set")
    client = anthropic.Anthropic(api_key=settings.CLAUDE_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def _call_manual(pass_name: str, prompt: str, output_path: str) -> str:
    """Save prompt to file, print instructions, collect multiline input until 'END'."""
    _save_prompt(prompt, output_path)
    print(f"\n{'='*70}")
    print(f"PASS: {pass_name}")
    print(f"Prompt saved to: {output_path}")
    print("Paste the prompt into Claude.ai, then paste the JSON response here.")
    print("Type END on a new line when done.")
    print("=" * 70 + "\n")
    if not sys.stdin.isatty():
        raise RuntimeError(f"Manual mode requires an interactive terminal (pass: {pass_name})")
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def _route_llm(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    pass_name: str,
    output_path: str,
) -> str:
    """Dispatch to the configured provider and save the full prompt for auditability."""
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    if provider == "manual":
        return _call_manual(pass_name, full_prompt, output_path)
    _save_prompt(full_prompt, output_path)
    if provider == "deepseek":
        return _call_deepseek(system_prompt, user_prompt)
    if provider == "claude":
        return _call_claude(system_prompt, user_prompt)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'manual', 'deepseek', or 'claude'.")


# ── Price data helpers ────────────────────────────────────────────────────────

def _price_summary_for_pass1(price_data: dict) -> dict:
    """Extract returns_20d / volatility / vs_sma50 per ticker for regime pass."""
    r, v, s = {}, {}, {}
    for ticker, d in price_data.items():
        if "error" not in d:
            r[ticker] = d.get("return_20d_pct")
            v[ticker] = d.get("vol_20d_ann_pct")
            s[ticker] = d.get("above_sma50")
    return {"returns_20d": r, "volatility": v, "vs_sma50": s}


def _price_structure_for_pass2(price_data: dict) -> dict:
    """Build momentum_ranking list and technicals dict for trade ideas pass."""
    valid = {t: d for t, d in price_data.items() if "error" not in d}
    ranked = sorted(valid.items(), key=lambda x: x[1].get("momentum_rank_20d", 999))
    return {
        "momentum_ranking": [
            {
                "rank": d.get("momentum_rank_20d"),
                "ticker": t,
                "return_20d_pct": d.get("return_20d_pct"),
                "return_5d_pct": d.get("return_5d_pct"),
                "return_1d_pct": d.get("return_1d_pct"),
            }
            for t, d in ranked
        ],
        "technicals": {
            t: {
                "rsi_14": d.get("rsi_14"),
                "above_sma50": d.get("above_sma50"),
                "current_price": d.get("current_price"),
            }
            for t, d in valid.items()
        },
    }


# ── Pass 1: Regime Classification ────────────────────────────────────────────

def run_pass1_regime(
    macro_data: dict,
    sentiment_data: dict,
    price_data: dict,
    calendar_data: dict,
    provider: str,
    date_str: str,
) -> dict:
    """Classify current market regime from macro + sentiment + price + calendar."""
    price_summary = _price_summary_for_pass1(price_data)
    user_prompt = "\n\n".join([
        f"MACRO DATA (FRED):\n{json.dumps(macro_data, indent=2)}",
        f"SENTIMENT INDICATORS:\n{json.dumps(sentiment_data, indent=2)}",
        f"CROSS-ASSET PRICE SUMMARY:\n{json.dumps(price_summary, indent=2)}",
        f"UPCOMING CATALYSTS:\n{json.dumps(calendar_data, indent=2)}",
        "Classify the market regime and return the required JSON schema.",
    ])
    output_path = str(settings.PROMPTS_DIR / f"{date_str}_pass1_regime.txt")

    try:
        raw = _route_llm(provider, _PASS1_SYSTEM, user_prompt, "Pass 1 — Regime", output_path)
        retry_fn: Callable | None = None
        if provider == "deepseek":
            retry_fn = lambda fp: _call_deepseek(_PASS1_SYSTEM, fp)  # noqa: E731
        elif provider == "claude":
            retry_fn = lambda fp: _call_claude(_PASS1_SYSTEM, fp)  # noqa: E731
        result = _parse_json_response(raw, retry_fn=retry_fn)
    except Exception as exc:
        raise RuntimeError(f"Pass 1 (Regime) failed: {exc}") from exc

    logger.info(
        "Pass 1 complete — regime: {} (confidence: {}/5)",
        result.get("regime", "?"),
        result.get("confidence", "?"),
    )
    return result


# ── Pass 2: Trade Idea Generation ────────────────────────────────────────────

def run_pass2_ideas(
    regime: dict,
    price_data: dict,
    crypto_data: dict,
    cot_data: dict,
    trends_data: dict,
    news_data: dict,
    provider: str,
    date_str: str,
) -> dict:
    """Generate trade ideas given regime + asset-specific signals."""
    price_structured = _price_structure_for_pass2(price_data)
    user_prompt = "\n\n".join([
        f"REGIME ASSESSMENT (from Pass 1):\n{json.dumps(regime, indent=2)}",
        f"ASSET MOMENTUM RANKING:\n{json.dumps(price_structured['momentum_ranking'], indent=2)}",
        f"TECHNICAL SIGNALS:\n{json.dumps(price_structured['technicals'], indent=2)}",
        f"CRYPTO-SPECIFIC SIGNALS:\n{json.dumps(crypto_data, indent=2)}",
        f"POSITIONING EXTREMES (COT):\n{json.dumps(cot_data, indent=2)}",
        f"SEARCH TREND SIGNALS:\n{json.dumps(trends_data, indent=2)}",
        f"RECENT NEWS THEMES:\n{json.dumps(news_data, indent=2)}",
        (
            "TRADEABLE UNIVERSE: IBIT, GLD, SPY, QQQ, TLT, USO, HYG\n"
            "STRATEGY: Swing trades, 2-10 day holds, no leverage.\n"
            "Only generate high-conviction ideas (4-5/5).\n"
            "It is better to have no trade than a bad trade."
        ),
    ])
    output_path = str(settings.PROMPTS_DIR / f"{date_str}_pass2_ideas.txt")

    try:
        raw = _route_llm(provider, _PASS2_SYSTEM, user_prompt, "Pass 2 — Trade Ideas", output_path)
        retry_fn: Callable | None = None
        if provider == "deepseek":
            retry_fn = lambda fp: _call_deepseek(_PASS2_SYSTEM, fp)  # noqa: E731
        elif provider == "claude":
            retry_fn = lambda fp: _call_claude(_PASS2_SYSTEM, fp)  # noqa: E731
        result = _parse_json_response(raw, retry_fn=retry_fn)
    except Exception as exc:
        raise RuntimeError(f"Pass 2 (Trade Ideas) failed: {exc}") from exc

    n = len(result.get("trade_ideas", []))
    logger.info("Pass 2 complete — {} trade idea(s) generated", n)
    if result.get("no_trade_reason"):
        logger.info("No-trade reason: {}", result["no_trade_reason"])
    return result


# ── Pass 3: Adversarial Stress Test ──────────────────────────────────────────

def run_pass3_stress_test(
    regime: dict,
    trade_ideas: dict,
    macro_data: dict,
    provider: str,
    date_str: str,
) -> dict:
    """Stress-test trade ideas adversarially. Always tries Claude first for cross-model review."""
    if not trade_ideas.get("trade_ideas"):
        logger.info("Pass 3 skipped — no trade ideas to review")
        return {"reviewed_ideas": [], "overall_assessment": "No trades to review."}

    user_prompt = "\n\n".join([
        f"MARKET REGIME:\n{json.dumps(regime, indent=2)}",
        f"PROPOSED TRADE IDEAS (generated by a separate model):\n"
        f"{json.dumps(trade_ideas['trade_ideas'], indent=2)}",
        f"MACRO CONTEXT:\n{json.dumps(macro_data, indent=2)}",
        (
            "Your job: find every reason these trades could fail.\n"
            "Be adversarial. Flag any bullish bias in the original analysis.\n"
            "A trade that survives scrutiny is worth taking.\n"
            "A trade that does not should be skipped or sized down."
        ),
    ])
    output_path = str(settings.PROMPTS_DIR / f"{date_str}_pass3_stress.txt")

    if provider == "manual":
        full = f"{_PASS3_SYSTEM}\n\n{user_prompt}"
        try:
            raw = _call_manual(
                "Pass 3 — Stress Test (paste into Claude.ai for cross-model perspective)",
                full,
                output_path,
            )
        except Exception as exc:
            raise RuntimeError(f"Pass 3 (Stress Test) failed: {exc}") from exc
    else:
        # Always try Claude first for genuine cross-model adversarial perspective
        _save_prompt(f"{_PASS3_SYSTEM}\n\n{user_prompt}", output_path)
        try:
            raw = _call_claude(_PASS3_SYSTEM, user_prompt)
            logger.info("Pass 3 using Claude for cross-model stress test")
        except Exception as claude_exc:
            logger.warning("Pass 3: Claude unavailable ({}), falling back to DeepSeek", claude_exc)
            try:
                raw = _call_deepseek(_PASS3_SYSTEM, user_prompt)
            except Exception as ds_exc:
                raise RuntimeError(
                    f"Pass 3 (Stress Test) failed — Claude: {claude_exc}; DeepSeek: {ds_exc}"
                ) from ds_exc

    def _retry(fix_prompt: str) -> str:
        try:
            return _call_claude(_PASS3_SYSTEM, fix_prompt)
        except Exception:
            return _call_deepseek(_PASS3_SYSTEM, fix_prompt)

    try:
        result = _parse_json_response(raw, retry_fn=None if provider == "manual" else _retry)
    except Exception as exc:
        raise RuntimeError(f"Pass 3 JSON parse failed: {exc}") from exc

    for reviewed in result.get("reviewed_ideas", []):
        logger.info(
            "  Stress test {} → {} | adjusted conviction: {}/5",
            reviewed.get("ticker", "?"),
            reviewed.get("final_recommendation", "?"),
            reviewed.get("adjusted_conviction", "?"),
        )
    return result


# ── Merge ─────────────────────────────────────────────────────────────────────

_SIZE_MAP: dict[str, float] = {"full": 1.0, "half": 0.5, "quarter": 0.25, "skip": 0.0}


def merge_final_signals(
    regime: dict,
    trade_ideas: dict,
    stress_test: dict,
    date_str: str,
) -> dict:
    """Merge Pass 2 ideas with Pass 3 reviews into final actionable signals."""
    reviews = {r["ticker"]: r for r in stress_test.get("reviewed_ideas", [])}
    final_trades: list[dict[str, Any]] = []
    filtered_count = 0

    for idea in trade_ideas.get("trade_ideas", []):
        ticker = idea.get("ticker", "?")
        review = reviews.get(ticker, {})
        recommendation = review.get("final_recommendation", "proceed")
        size_adj = review.get("size_adjustment", "full")
        size_multiplier = _SIZE_MAP.get(size_adj, 1.0)

        if recommendation == "skip" or size_multiplier == 0.0:
            logger.info("⛔ {} filtered out by stress test ({})", ticker, recommendation)
            filtered_count += 1
            continue

        final_trade: dict[str, Any] = {
            **idea,
            "conviction": review.get("adjusted_conviction", idea.get("conviction")),
            "size_multiplier": size_multiplier,
            "bear_case": review.get("bear_case", ""),
            "hidden_risks": review.get("hidden_risks", []),
            "passed_stress_test": True,
        }
        final_trades.append(final_trade)
        logger.info(
            "✓ {} passed stress test | conviction: {}/5 | size: {}",
            ticker,
            final_trade["conviction"],
            size_adj,
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date_str,
        "market_regime": regime,
        "final_trades": final_trades,
        "trades_filtered_out": filtered_count,
        "portfolio_risks": stress_test.get("portfolio_level_risks", []),
        "overall_assessment": stress_test.get("overall_assessment", ""),
    }
