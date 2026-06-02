"""Pluggable LLM client for the trading research system.

Dispatches on config.settings.LLM_PROVIDER:
  "manual"   — saves prompt, asks user to paste JSON response
  "deepseek" — OpenAI-compatible API at api.deepseek.com
  "claude"   — Anthropic API

All modes validate the returned JSON and retry once on failure.
Heavy SDK imports are lazy so manual mode needs no extra packages.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a quantitative trading analyst. "
    "Respond only with valid JSON. No markdown, no explanation."
)

REQUIRED_TOP_KEYS = {"market_regime", "regime_reasoning", "trade_ideas", "risks", "generated_at"}
REQUIRED_IDEA_KEYS = {"ticker", "direction", "conviction", "catalyst", "entry",
                      "stop_loss_pct", "target_pct", "holding_days", "reasoning"}


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _validate_schema(data: dict) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []
    missing_top = REQUIRED_TOP_KEYS - data.keys()
    if missing_top:
        errors.append(f"Missing top-level keys: {missing_top}")

    ideas = data.get("trade_ideas", [])
    if not isinstance(ideas, list):
        errors.append("trade_ideas must be a list")
    else:
        for i, idea in enumerate(ideas):
            missing = REQUIRED_IDEA_KEYS - idea.keys()
            if missing:
                errors.append(f"trade_ideas[{i}] missing keys: {missing}")

    if not isinstance(data.get("risks", []), list):
        errors.append("risks must be a list")

    return errors


def _parse_and_validate(text: str) -> dict:
    data = _parse_json(text)
    errors = _validate_schema(data)
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")
    return data


# ── Provider implementations ──────────────────────────────────────────────────

def _call_manual(prompt: str) -> dict:
    from research.prompt_builder import save_prompt
    path = save_prompt(prompt)
    print("\n" + "=" * 70)
    print(prompt)
    print("=" * 70)
    print(f"\nPrompt saved to: {path}")
    print("\nPaste the above into Claude.ai and paste the JSON response back here.")
    print("Enter JSON (paste multi-line, then press Enter twice to finish):\n")

    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
    except EOFError:
        pass  # non-interactive environment

    raw = "\n".join(lines).strip()
    if not raw:
        raise ValueError("No JSON response received")
    return _parse_and_validate(raw)


def _call_deepseek(prompt: str) -> str:
    from openai import OpenAI
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def _call_claude(prompt: str) -> str:
    import anthropic
    api_key = settings.CLAUDE_API_KEY
    if not api_key:
        raise RuntimeError("CLAUDE_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _retry_fix(raw: str, provider: str) -> dict:
    fix_prompt = "Fix the JSON and return only valid JSON: " + raw
    logger.info("Retrying with fix prompt...")
    if provider == "deepseek":
        raw2 = _call_deepseek(fix_prompt)
    elif provider == "claude":
        raw2 = _call_claude(fix_prompt)
    else:
        raise ValueError("Cannot auto-retry in manual mode — please provide valid JSON")
    return _parse_and_validate(raw2)


# ── Public interface ──────────────────────────────────────────────────────────

def get_trade_ideas(prompt: str) -> dict[str, Any]:
    """Call the configured LLM provider and return validated trade ideas JSON."""
    provider = settings.LLM_PROVIDER

    if provider == "manual":
        return _call_manual(prompt)

    if provider == "deepseek":
        raw = _call_deepseek(prompt)
    elif provider == "claude":
        raw = _call_claude(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'manual', 'deepseek', or 'claude'.")

    try:
        return _parse_and_validate(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Initial JSON parse/validate failed (%s) — retrying with fix prompt", exc)
        return _retry_fix(raw, provider)
