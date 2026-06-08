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
from research import risk_manager


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


# ── Required output schemas (pinned so models don't invent field names) ───────

_PASS1_SCHEMA = """\
Return ONLY this exact JSON object (no markdown, no commentary):
{
  "regime": "risk_on | risk_off | neutral | stagflation",
  "confidence": <integer 1-5>,
  "macro_bias": "bullish | bearish | neutral",
  "volatility_regime": "low | normal | elevated | high",
  "key_signals": ["...", "..."],
  "regime_reasoning": "2-3 sentence explanation",
  "cross_asset_message": "what bonds/gold/crypto/equities jointly imply",
  "upcoming_risks": ["...", "..."],
  "rate_regime": "easing | tightening | on_hold | transitioning",
  "rate_regime_implication": {
    "GLD": "bullish | bearish | neutral",
    "TLT": "bullish | bearish | neutral",
    "QQQ": "bullish | bearish | neutral",
    "IBIT": "bullish | bearish | neutral"
  }
}
IMPORTANT: "confidence" is an INTEGER from 1 to 5 (not a probability). Use the key
"regime_reasoning" (not "narrative").
Set "rate_regime" from the FORWARD RATE PRICING, REAL YIELDS, RATE MOMENTUM, and
CREDIT CONDITIONS sections. "rate_regime_implication" states, for each
rate-sensitive instrument, whether the rate path is bullish/bearish/neutral
(e.g. falling real yields → bullish GLD; easing → bullish TLT)."""

_PASS2_SCHEMA = """\
Return ONLY this exact JSON object (no markdown, no commentary):
{
  "no_trade_reason": null,
  "trade_ideas": [
    {
      "ticker": "<one ticker from the tradeable universe above>",
      "direction": "long | short",
      "conviction": <integer 1-5>,
      "catalyst": "the specific trigger for this trade",
      "signal_sources": ["...", "..."],
      "entry": "market_open | limit_at_<price>",
      "stop_loss_pct": <number>,
      "target_pct": <number>,
      "holding_days": <integer, must be <= max_holding_days>,
      "max_holding_days": <integer from the universe constraints above>,
      "invalidation": "what would prove this idea wrong",
      "reasoning": "...",
      "signal_type": "rule_based | situational | hybrid",
      "signal_type_reasoning": "one sentence justifying the signal_type",
      "primary_rule": null,
      "rule_backtest_status": "not_tested",
      "contributing_signals": {"rule_based": [], "situational": []},
      "validation_method": "backtest | paper_trade | both"
    }
  ]
}
Output 0-3 ideas (empty list + a "no_trade_reason" string is valid).
Use the key "conviction" (integer 1-5) — NOT "confidence".
For "primary_rule": give an explicit if/then rule string for rule_based/hybrid
ideas, otherwise null. "rule_backtest_status" is always "not_tested" here.
Set "validation_method" from "signal_type": rule_based→"both",
situational→"paper_trade", hybrid→"both"."""

_PASS2_CLASSIFICATION = """\
SIGNAL TYPE CLASSIFICATION (set signal_type per idea):
- rule_based: fires because ONE quantitative threshold was crossed; definable
  as an explicit if/then rule; testable historically.
- situational: a unique confluence of conditions, news, or qualitative factors
  unlikely to repeat identically.
- hybrid: a rule-based anchor with situational factors meaningfully affecting
  conviction.
Put the concrete signals that drove the idea into contributing_signals, split
into the "rule_based" and "situational" buckets."""

_PASS3_SCHEMA = """\
Return ONLY this exact JSON object (no markdown, no commentary):
{
  "reviewed_ideas": [
    {
      "ticker": "...",
      "original_conviction": <integer 1-5>,
      "adjusted_conviction": <integer 1-5>,
      "bull_case": "...",
      "bear_case": "...",
      "hidden_risks": ["...", "..."],
      "prompt_bias_check": "any bullish bias detected in the original analysis",
      "regime_fit": "does this fit the stated regime?",
      "final_recommendation": "proceed | reduce_size | skip",
      "size_adjustment": "full | half | quarter | skip"
    }
  ],
  "portfolio_level_risks": ["...", "..."],
  "overall_assessment": "..."
}
Use the key "reviewed_ideas" (NOT "trade_review"), "final_recommendation"
(NOT "recommendation"), and "size_adjustment" = one of full/half/quarter/skip.
If you would not take a trade, set final_recommendation = "skip"."""


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
    """Extract returns_20d / volatility / vs_sma50 per ticker for regime pass.

    Includes both tradeable and signal-only readings (SPY/HYG are valuable for
    regime classification even though they are never traded).
    """
    r, v, s = {}, {}, {}

    def _add(ticker: str, d: dict) -> None:
        if isinstance(d, dict) and "error" not in d:
            r[ticker] = d.get("return_20d_pct")
            v[ticker] = d.get("vol_20d_ann_pct")
            s[ticker] = d.get("above_sma50")

    for ticker, d in price_data.items():
        if ticker == "signal_readings":
            continue
        _add(ticker, d)
    for ticker, d in price_data.get("signal_readings", {}).items():
        _add(ticker, d)

    return {"returns_20d": r, "volatility": v, "vs_sma50": s}


def _price_structure_for_pass2(price_data: dict) -> dict:
    """Build momentum_ranking list and technicals dict for trade ideas pass.

    Only tradeable instruments are ranked/surfaced; signal-only readings are
    excluded here (they inform the regime pass, not trade selection).
    """
    valid = {
        t: d
        for t, d in price_data.items()
        if t != "signal_readings" and isinstance(d, dict) and "error" not in d
    }
    ranked = sorted(valid.items(), key=lambda x: x[1].get("momentum_rank_20d", 999))

    def _technicals(d: dict) -> dict:
        tech = {
            "rsi_14": d.get("rsi_14"),
            "above_sma50": d.get("above_sma50"),
            "current_price": d.get("current_price"),
        }
        # Surface VIXY-specific flags when present
        if "vixy_hold_warning" in d:
            tech["vixy_hold_warning"] = d["vixy_hold_warning"]
        if "vix_backwardation_proxy" in d:
            tech["vix_backwardation_proxy"] = d["vix_backwardation_proxy"]
        return tech

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
        "technicals": {t: _technicals(d) for t, d in valid.items()},
    }


# ── Universe context + holding-period enforcement ─────────────────────────────

def _build_universe_block() -> str:
    """Build the tradeable-universe + signal-only context from INSTRUMENT_META."""
    lines = ["TRADEABLE UNIVERSE AND CONSTRAINTS:"]
    for ticker in settings.get_tradeable_universe():
        m = settings.get_instrument_meta(ticker)
        sigs = ", ".join(m.get("signal_sources", []))
        lines.append(
            f"- {ticker} ({m.get('asset_class', '?')}): "
            f"max {m.get('max_holding_days', '?')} day hold. "
            f"Primary signals: {sigs}. Note: {m.get('notes', '')}"
        )
    lines.append("")
    lines.append("SIGNAL-ONLY (do not trade, use as context):")
    for ticker in settings.SIGNAL_ONLY:
        m = settings.get_instrument_meta(ticker)
        lines.append(f"- {ticker}: {m.get('notes', '')}")
    return "\n".join(lines)


def _normalize_pass1(result: Any) -> dict:
    """Map common Pass 1 field aliases and coerce confidence to an integer 1-5."""
    if not isinstance(result, dict):
        return {"regime": "unknown", "confidence": None, "regime_reasoning": ""}
    if "regime_reasoning" not in result:
        for key in ("narrative", "reasoning", "rationale", "explanation"):
            if key in result:
                result["regime_reasoning"] = result[key]
                break
    c = result.get("confidence")
    if isinstance(c, bool):  # guard: bool is a subclass of int
        pass
    elif isinstance(c, float) and 0.0 <= c <= 1.0:
        # model returned a probability instead of a 1-5 score
        result["confidence"] = max(1, min(5, round(c * 5)))
    elif isinstance(c, (int, float)):
        result["confidence"] = max(1, min(5, int(round(c))))
    return result


_IDEA_FIELD_ALIASES = {
    "conviction": ("confidence", "conviction_score"),
    "catalyst": ("entry_logic", "trigger"),
    "stop_loss_pct": ("stop_loss_percent",),
    "target_pct": ("take_profit_pct", "target_percent"),
}


_VALIDATION_METHOD = {"rule_based": "both", "situational": "paper_trade", "hybrid": "both"}


def _fill_signal_type_defaults(idea: dict) -> None:
    """Ensure every idea carries the signal-type / thesis classification fields."""
    signal_type = idea.get("signal_type")
    if signal_type not in _VALIDATION_METHOD:
        signal_type = "situational"  # conservative default → paper_trade only
        idea["signal_type"] = signal_type
    idea.setdefault("signal_type_reasoning", "")
    idea.setdefault("primary_rule", None)
    idea["rule_backtest_status"] = "not_tested"  # always at generation time
    cs = idea.get("contributing_signals")
    if not isinstance(cs, dict):
        cs = {}
    cs.setdefault("rule_based", [])
    cs.setdefault("situational", [])
    idea["contributing_signals"] = cs
    # validation_method is derived from signal_type
    idea["validation_method"] = _VALIDATION_METHOD[signal_type]


def _normalize_idea(idea: dict) -> dict:
    """Map per-idea field aliases (e.g. confidence → conviction) in place."""
    if not isinstance(idea, dict):
        return idea
    for canonical, aliases in _IDEA_FIELD_ALIASES.items():
        if canonical not in idea:
            for alias in aliases:
                if alias in idea:
                    idea[canonical] = idea[alias]
                    break
    _fill_signal_type_defaults(idea)
    return idea


def _normalize_pass2(result: Any) -> dict:
    """Coerce Pass 2 output into {'trade_ideas': [...], ...}.

    LLMs sometimes return a bare JSON array of ideas, a single idea object, or
    nest the list under an alternate key. Normalise all of these so downstream
    code can rely on a dict with a 'trade_ideas' list, and map per-idea field
    aliases (e.g. confidence → conviction).
    """
    if isinstance(result, list):
        result = {"trade_ideas": result}
    elif not isinstance(result, dict):
        logger.warning("Pass 2 returned unexpected type {} — treating as no trades", type(result).__name__)
        return {"trade_ideas": [], "no_trade_reason": "Unparseable Pass 2 output"}
    elif "trade_ideas" not in result or not isinstance(result.get("trade_ideas"), list):
        # Look for the list under a differently-named key
        moved = False
        for key in ("ideas", "trades", "trade_idea", "recommendations"):
            if isinstance(result.get(key), list):
                result["trade_ideas"] = result.pop(key)
                moved = True
                break
        if not moved:
            if "ticker" in result:  # a single trade-idea object returned directly
                result = {"trade_ideas": [result]}
            else:
                result.setdefault("trade_ideas", [])

    for idea in result.get("trade_ideas", []):
        _normalize_idea(idea)
    return result


def _cap_holding_days(result: dict) -> dict:
    """Enforce per-instrument max_holding_days on Pass 2 ideas (cap, never reject)."""
    for idea in result.get("trade_ideas", []):
        if not isinstance(idea, dict):
            continue
        ticker = idea.get("ticker")
        meta = settings.get_instrument_meta(ticker)
        max_hold = meta.get("max_holding_days")
        if max_hold is None:
            continue
        idea["max_holding_days"] = max_hold
        proposed = idea.get("holding_days")
        if isinstance(proposed, (int, float)) and proposed > max_hold:
            logger.warning("{}: holding_days capped from {} to {}", ticker, proposed, max_hold)
            idea["holding_days"] = max_hold
    return result


# ── Pass 3 normalisation (so a 'skip' can never be silently dropped) ──────────

_SKIP_WORDS = {"skip", "reject", "avoid", "no", "do_not_trade", "drop", "pass", "decline"}
_REDUCE_WORDS = {"reduce", "reduce_size", "size_down", "trim", "downsize", "scale_back"}
_PROCEED_WORDS = {"proceed", "approve", "approved", "take", "yes", "ok", "accept", "go"}


def _coerce_size_adjustment(value: Any, recommendation: str) -> str:
    """Map a free-text/aliased size hint to one of full/half/quarter/skip."""
    if recommendation == "skip":
        return "skip"
    if isinstance(value, str):
        v = value.lower()
        if "quarter" in v or "25%" in v or "1/4" in v:
            return "quarter"
        if "half" in v or "50%" in v or "1/2" in v:
            return "half"
        if "skip" in v or "avoid" in v or "zero" in v or "0%" in v or "no position" in v:
            return "skip"
        if "full" in v or "100%" in v or "standard" in v:
            return "full"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # treat as a multiplier or percentage
        m = value / 100.0 if value > 1 else value
        if m <= 0.0:
            return "skip"
        if m <= 0.3:
            return "quarter"
        if m <= 0.6:
            return "half"
        return "full"
    if recommendation == "reduce_size":
        return "half"
    return "full"


def _normalize_pass3(result: Any) -> dict:
    """Coerce Pass 3 output into the canonical reviewed-ideas schema.

    Critically maps recommendation/sizing aliases (and approved=false) so an
    adversarial 'skip' is never silently lost to a default 'proceed'.
    """
    if isinstance(result, list):
        result = {"reviewed_ideas": result}
    elif not isinstance(result, dict):
        return {"reviewed_ideas": [], "portfolio_level_risks": [], "overall_assessment": ""}

    if "reviewed_ideas" not in result or not isinstance(result.get("reviewed_ideas"), list):
        for key in ("trade_review", "trade_reviews", "reviews", "reviewed", "ideas", "trades"):
            if isinstance(result.get(key), list):
                result["reviewed_ideas"] = result.pop(key)
                break
        result.setdefault("reviewed_ideas", [])

    for r in result["reviewed_ideas"]:
        if not isinstance(r, dict):
            continue
        # final_recommendation aliases
        if "final_recommendation" not in r:
            rec = r.get("recommendation") or r.get("decision") or r.get("verdict") or r.get("action")
            if rec is None and "approved" in r:
                rec = "proceed" if r.get("approved") else "skip"
            if rec is not None:
                r["final_recommendation"] = rec
        rec_norm = str(r.get("final_recommendation", "proceed")).lower().strip().replace(" ", "_")
        if rec_norm in _SKIP_WORDS:
            rec_norm = "skip"
        elif rec_norm in _REDUCE_WORDS or rec_norm in ("half", "quarter"):
            rec_norm = "reduce_size"
        elif rec_norm in _PROCEED_WORDS:
            rec_norm = "proceed"
        # an explicit approved=false always means skip
        if r.get("approved") is False:
            rec_norm = "skip"
        r["final_recommendation"] = rec_norm

        # size_adjustment aliases
        if "size_adjustment" not in r:
            for key in ("sizing_suggestion", "size", "sizing", "position_size", "size_multiplier"):
                if key in r:
                    r["size_adjustment"] = r[key]
                    break
        r["size_adjustment"] = _coerce_size_adjustment(r.get("size_adjustment"), rec_norm)

        # conviction alias
        if "adjusted_conviction" not in r:
            for key in ("adjusted_confidence", "new_conviction", "conviction"):
                if key in r:
                    r["adjusted_conviction"] = r[key]
                    break

    result.setdefault("portfolio_level_risks", result.get("portfolio_risks", []))
    result.setdefault("overall_assessment", result.get("assessment", ""))
    return result


# ── Pass 1: Regime Classification ────────────────────────────────────────────

def _pick(data: dict, *keys: str) -> dict:
    """Return only the requested keys that are present in `data`."""
    return {k: data[k] for k in keys if k in data}


def _build_rates_block(macro_data: dict, fed_futures_data: dict) -> str:
    """Format the forward-rate / real-yield / credit sections for Pass 1."""
    forward = fed_futures_data or {"data_source": "unavailable"}
    real_yields = _pick(
        macro_data,
        "real_yield_10y", "real_yield_10y_interpretation",
        "real_yield_10y_20d_change", "real_yield_10y_20d_change_interpretation",
        "breakeven_inflation_10y", "breakeven_inflation_10y_interpretation",
        "forward_inflation_5y5y", "forward_inflation_5y5y_interpretation",
    )
    momentum = _pick(
        macro_data,
        "yield_curve_momentum_20d", "yield_curve_momentum_label",
        "yield_curve_momentum_interpretation",
        "dgs10_vs_200sma", "dgs10_200sma", "dgs10_vs_200sma_interpretation",
    )
    credit = _pick(
        macro_data,
        "credit_spread_oas", "credit_spread_oas_interpretation",
        "credit_spread_20d_change", "credit_spread_20d_change_interpretation",
    )
    return "\n\n".join([
        f"FORWARD RATE PRICING:\n{json.dumps(forward, indent=2)}",
        f"REAL YIELDS AND INFLATION EXPECTATIONS:\n{json.dumps(real_yields, indent=2)}",
        f"RATE MOMENTUM:\n{json.dumps(momentum, indent=2)}",
        f"CREDIT CONDITIONS:\n{json.dumps(credit, indent=2)}",
    ])


def _build_labor_block(macro_data: dict) -> str:
    """Format the Tier 2 labor-market section for Pass 1."""
    labor = _pick(
        macro_data,
        "nfp_level_k", "nfp_change_mom_k", "nfp_change_interpretation",
        "jolts_openings_k", "jolts_openings_change", "jolts_openings_change_interpretation",
        "jolts_openings_per_unemployed", "labor_tightness_interpretation",
        "initial_claims", "initial_claims_interpretation",
        "initial_claims_20d_change", "initial_claims_change_interpretation",
        "wage_growth_yoy", "wage_growth_interpretation",
    )
    return (
        "LABOR MARKET (TIER 2 — payrolls, openings, claims, wages):\n"
        f"{json.dumps(labor, indent=2)}\n"
        "Strong payrolls/openings/wages and low claims argue for a tighter "
        "(hawkish) rate path; weakening labor argues for easing — weigh this in "
        "both the regime call and rate_regime."
    )


def run_pass1_regime(
    macro_data: dict,
    sentiment_data: dict,
    price_data: dict,
    calendar_data: dict,
    provider: str,
    date_str: str,
    fed_futures_data: dict | None = None,
) -> dict:
    """Classify current market regime from macro + sentiment + price + calendar."""
    price_summary = _price_summary_for_pass1(price_data)
    user_prompt = "\n\n".join([
        f"MACRO DATA (FRED):\n{json.dumps(macro_data, indent=2)}",
        f"SENTIMENT INDICATORS:\n{json.dumps(sentiment_data, indent=2)}",
        f"CROSS-ASSET PRICE SUMMARY:\n{json.dumps(price_summary, indent=2)}",
        f"UPCOMING CATALYSTS:\n{json.dumps(calendar_data, indent=2)}",
        _build_rates_block(macro_data, fed_futures_data or {}),
        _build_labor_block(macro_data),
        _PASS1_SCHEMA,
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

    result = _normalize_pass1(result)
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
            _build_universe_block()
            + "\n\nSTRATEGY: Swing trades, no leverage. Respect each instrument's max hold.\n"
            "Each trade idea MUST include a 'max_holding_days' field equal to the "
            "instrument's max hold listed above, and its 'holding_days' MUST NOT "
            "exceed that value.\n"
            "For rate-sensitive instruments (GLD, TLT, QQQ, IBIT, SLV), consult the "
            "regime's 'rate_regime' and 'rate_regime_implication': do not propose a "
            "direction that contradicts the rate-path implication for that ticker "
            "without an explicit overriding catalyst.\n"
            "Only generate high-conviction ideas (4-5/5).\n"
            "It is better to have no trade than a bad trade."
        ),
        _PASS2_CLASSIFICATION,
        _PASS2_SCHEMA,
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

    # Normalise shape (model may return a bare list / single object), then
    # enforce per-instrument holding-period limits (cap, never reject)
    result = _normalize_pass2(result)
    result = _cap_holding_days(result)

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
            "SPECIAL INSTRUMENT RULES TO ENFORCE:\n"
            "- VIXY: reject any idea with holding_days > 3. Flag if proposed during "
            "a high-VIX FOMC week.\n"
            "- SLV: flag if no corresponding GLD signal exists to justify the trade.\n"
            "- TLT: flag if the macro_bias from Pass 1 contradicts the direction "
            "(e.g. long TLT in a risk_on regime).\n"
            "- XLE: verify a news or COT catalyst exists — do not approve "
            "momentum-only XLE trades."
        ),
        (
            "Your job: find every reason these trades could fail.\n"
            "Be adversarial. Flag any bullish bias in the original analysis.\n"
            "A trade that survives scrutiny is worth taking.\n"
            "A trade that does not should be skipped or sized down."
        ),
        _PASS3_SCHEMA,
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

    # Map aliased field names so a 'skip' / approved=false is never silently lost
    result = _normalize_pass3(result)

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
        pass3_size_multiplier = _SIZE_MAP.get(size_adj, 1.0)

        if recommendation == "skip" or pass3_size_multiplier == 0.0:
            logger.info("⛔ {} filtered out by stress test ({})", ticker, recommendation)
            filtered_count += 1
            continue

        final_trade: dict[str, Any] = {
            **idea,
            "conviction": review.get("adjusted_conviction", idea.get("conviction")),
            # Pass 3 "trust" multiplier — conviction-based sizing is compounded with
            # this in the execution layer (risk_manager.validate_all), not here.
            "pass3_size_multiplier": pass3_size_multiplier,
            "bear_case": review.get("bear_case", ""),
            "hidden_risks": review.get("hidden_risks", []),
            "passed_stress_test": True,
            # Thesis tracking (lifecycle fields populated by the execution layer)
            "thesis": idea.get("reasoning", ""),
            "invalidation": idea.get("invalidation", ""),
            "thesis_outcome": None,
            "thesis_outcome_notes": None,
            "opened_at": None,
            "closed_at": None,
        }
        final_trades.append(final_trade)
        logger.info(
            "✓ {} passed stress test | conviction: {}/5 | size: {}",
            ticker,
            final_trade["conviction"],
            size_adj,
        )

    # Apply the GLD/SLV correlation guard before the signals are saved
    final_trades = risk_manager.apply_correlation_guard(final_trades)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date_str,
        "market_regime": regime,
        "final_trades": final_trades,
        "trades_filtered_out": filtered_count,
        "portfolio_risks": stress_test.get("portfolio_level_risks", []),
        "overall_assessment": stress_test.get("overall_assessment", ""),
    }
