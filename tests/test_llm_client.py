"""Tests for research/llm_client.py (offline — no real API calls)."""

import json
import pytest


VALID_RESPONSE = {
    "market_regime": "risk_on",
    "regime_reasoning": "Strong momentum and low VIX.",
    "trade_ideas": [
        {
            "ticker": "IBIT",
            "direction": "long",
            "conviction": 4,
            "catalyst": "BTC funding rate neutral, price above SMA50",
            "entry": "market_open",
            "stop_loss_pct": 3.0,
            "target_pct": 8.0,
            "holding_days": 5,
            "reasoning": "Momentum setup with supportive macro.",
        }
    ],
    "risks": ["Unexpected Fed hawkishness", "BTC flash crash"],
    "generated_at": "2024-01-15T10:00:00Z",
}


def test_parse_json_plain():
    from research.llm_client import _parse_json
    raw = json.dumps(VALID_RESPONSE)
    result = _parse_json(raw)
    assert result["market_regime"] == "risk_on"


def test_parse_json_strips_markdown_fences():
    from research.llm_client import _parse_json
    raw = "```json\n" + json.dumps(VALID_RESPONSE) + "\n```"
    result = _parse_json(raw)
    assert result["market_regime"] == "risk_on"


def test_validate_schema_valid():
    from research.llm_client import _validate_schema
    errors = _validate_schema(VALID_RESPONSE)
    assert errors == []


def test_validate_schema_missing_top_key():
    from research.llm_client import _validate_schema
    bad = dict(VALID_RESPONSE)
    del bad["market_regime"]
    errors = _validate_schema(bad)
    assert any("market_regime" in e for e in errors)


def test_validate_schema_missing_idea_key():
    from research.llm_client import _validate_schema
    bad = dict(VALID_RESPONSE)
    bad["trade_ideas"] = [{"ticker": "SPY"}]  # missing required idea keys
    errors = _validate_schema(bad)
    assert len(errors) > 0


def test_validate_schema_risks_not_list():
    from research.llm_client import _validate_schema
    bad = dict(VALID_RESPONSE)
    bad["risks"] = "some string"
    errors = _validate_schema(bad)
    assert any("risks" in e for e in errors)


def test_unknown_provider_raises(monkeypatch):
    import config.settings as s
    from research.llm_client import get_trade_ideas
    monkeypatch.setattr(s, "LLM_PROVIDER", "unknown_provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_trade_ideas("test prompt")
