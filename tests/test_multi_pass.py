"""Tests for the multi-pass LLM pipeline (llm_client.py)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from config import settings
from research.llm_client import (
    _cap_holding_days,
    _coerce_size_adjustment,
    _normalize_pass1,
    _normalize_pass2,
    _normalize_pass3,
    _parse_json_response,
    merge_final_signals,
    run_pass3_stress_test,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

_REGIME = {
    "regime": "risk_on",
    "confidence": 4,
    "macro_bias": "bullish",
    "volatility_regime": "normal",
    "key_signals": ["yield curve steepening", "VIX below 20"],
    "regime_reasoning": "Macro indicators support risk-on positioning.",
    "cross_asset_message": "Bonds stable, gold flat, BTC momentum positive.",
    "upcoming_risks": ["FOMC meeting", "CPI release"],
}

_IDEAS_TWO = {
    "no_trade_reason": None,
    "trade_ideas": [
        {
            "ticker": "IBIT",
            "direction": "long",
            "conviction": 4,
            "catalyst": "BTC funding neutral, price above SMA50",
            "signal_sources": ["funding_rate", "momentum"],
            "entry": "market_open",
            "stop_loss_pct": 3.0,
            "target_pct": 8.0,
            "holding_days": 5,
            "invalidation": "BTC drops below 20d SMA",
            "reasoning": "Setup aligns with risk-on regime.",
        },
        {
            "ticker": "GLD",
            "direction": "long",
            "conviction": 3,
            "catalyst": "Gold search trends rising",
            "signal_sources": ["trends", "cot"],
            "entry": "market_open",
            "stop_loss_pct": 2.5,
            "target_pct": 5.0,
            "holding_days": 7,
            "invalidation": "Real yields spike",
            "reasoning": "Safe-haven demand building.",
        },
    ],
    "position_sizing_note": "Risk 1% per trade.",
    "correlation_warning": "IBIT and GLD may diverge.",
}

_STRESS_PROCEED_SKIP = {
    "reviewed_ideas": [
        {
            "ticker": "IBIT",
            "original_conviction": 4,
            "adjusted_conviction": 4,
            "bull_case": "BTC momentum strong",
            "bear_case": "Regulatory risk could trigger sudden drop",
            "hidden_risks": ["ETF flow reversal"],
            "prompt_bias_check": "Acceptable",
            "regime_fit": "Fits risk-on regime",
            "final_recommendation": "proceed",
            "size_adjustment": "full",
        },
        {
            "ticker": "GLD",
            "original_conviction": 3,
            "adjusted_conviction": 2,
            "bull_case": "Safe haven demand building",
            "bear_case": "Risk-on regime contradicts gold long",
            "hidden_risks": ["USD strength"],
            "prompt_bias_check": "Contradicts stated regime",
            "regime_fit": "Poor fit for risk-on regime",
            "final_recommendation": "skip",
            "size_adjustment": "skip",
        },
    ],
    "portfolio_level_risks": ["Correlation risk", "FOMC surprise"],
    "overall_assessment": "IBIT trade is valid; GLD contradicts regime.",
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_parse_json_clean():
    valid = {"market_regime": "risk_on", "confidence": 4}
    result = _parse_json_response(json.dumps(valid))
    assert result == valid


def test_parse_json_with_fences():
    valid = {"regime": "neutral", "confidence": 3}
    raw = f"```json\n{json.dumps(valid)}\n```"
    result = _parse_json_response(raw)
    assert result == valid


def test_merge_skips_filtered():
    result = merge_final_signals(
        regime=_REGIME,
        trade_ideas=_IDEAS_TWO,
        stress_test=_STRESS_PROCEED_SKIP,
        date_str="2024-01-15",
    )
    tickers = [t["ticker"] for t in result["final_trades"]]
    assert "IBIT" in tickers
    assert "GLD" not in tickers
    assert result["trades_filtered_out"] == 1


def test_merge_applies_size():
    stress_half = {
        "reviewed_ideas": [
            {
                "ticker": "IBIT",
                "adjusted_conviction": 3,
                "final_recommendation": "reduce_size",
                "size_adjustment": "half",
                "bear_case": "Some risk present",
                "hidden_risks": ["liquidity thin"],
            }
        ],
        "portfolio_level_risks": [],
        "overall_assessment": "Size down IBIT.",
    }
    ideas_one = {
        "no_trade_reason": None,
        "trade_ideas": [_IDEAS_TWO["trade_ideas"][0]],
    }
    result = merge_final_signals(
        regime=_REGIME,
        trade_ideas=ideas_one,
        stress_test=stress_half,
        date_str="2024-01-15",
    )
    assert len(result["final_trades"]) == 1
    # merge stores the Pass 3 trust multiplier; conviction sizing is compounded
    # later in the execution layer, not here.
    assert result["final_trades"][0]["pass3_size_multiplier"] == 0.5


def test_early_exit_no_ideas():
    empty_ideas = {
        "no_trade_reason": "No high-conviction setups today",
        "trade_ideas": [],
    }
    result = merge_final_signals(
        regime=_REGIME,
        trade_ideas=empty_ideas,
        stress_test={"reviewed_ideas": [], "portfolio_level_risks": [], "overall_assessment": ""},
        date_str="2024-01-15",
    )
    assert result["final_trades"] == []
    assert result["trades_filtered_out"] == 0


def test_pass3_skipped_when_empty():
    """Pass 3 returns the early-exit dict without making any LLM call."""
    empty_ideas = {"no_trade_reason": "Nothing looks good", "trade_ideas": []}

    with patch("research.llm_client._call_claude") as mock_claude, \
         patch("research.llm_client._call_deepseek") as mock_deepseek, \
         patch("research.llm_client._call_manual") as mock_manual:
        result = run_pass3_stress_test(
            regime=_REGIME,
            trade_ideas=empty_ideas,
            macro_data={},
            provider="claude",
            date_str="2024-01-15",
        )

    mock_claude.assert_not_called()
    mock_deepseek.assert_not_called()
    mock_manual.assert_not_called()
    assert result["reviewed_ideas"] == []
    assert result["overall_assessment"] == "No trades to review."


# ── Universe / instrument-metadata tests ──────────────────────────────────────

def test_universe_no_signal_only_traded():
    """No signal-only instrument may appear in the tradeable universe."""
    tradeable = settings.get_tradeable_universe()
    for ticker in settings.SIGNAL_ONLY:
        assert ticker not in tradeable
    assert tradeable == settings.UNIVERSE


def test_vixy_max_hold_enforced():
    """A VIXY idea proposing a 7-day hold is capped to the 3-day max."""
    pass2_output = {
        "trade_ideas": [
            {"ticker": "VIXY", "direction": "long", "holding_days": 7, "conviction": 4},
        ],
    }
    capped = _cap_holding_days(pass2_output)
    idea = capped["trade_ideas"][0]
    assert idea["holding_days"] == 3
    assert idea["max_holding_days"] == 3


def test_instrument_meta_complete():
    """Every tradeable + signal-only ticker has metadata with required keys."""
    required = ("name", "asset_class", "notes")
    for ticker in settings.UNIVERSE + settings.SIGNAL_ONLY:
        meta = settings.get_instrument_meta(ticker)
        assert meta, f"{ticker} missing from INSTRUMENT_META"
        for key in required:
            assert key in meta, f"{ticker} missing '{key}'"


# ── Pass 2 output normalisation ───────────────────────────────────────────────

def test_normalize_pass2_bare_list():
    """A bare JSON array of ideas is wrapped into {'trade_ideas': [...]}."""
    ideas = [{"ticker": "IBIT", "holding_days": 5}, {"ticker": "GLD", "holding_days": 3}]
    result = _normalize_pass2(ideas)
    assert result["trade_ideas"] == ideas


def test_normalize_pass2_single_object():
    """A single trade-idea object is wrapped into a one-element list."""
    idea = {"ticker": "QQQ", "direction": "long", "holding_days": 4}
    result = _normalize_pass2(idea)
    assert result["trade_ideas"] == [idea]


def test_normalize_pass2_alternate_key():
    """A list nested under an alternate key is moved to 'trade_ideas'."""
    raw = {"ideas": [{"ticker": "SLV"}], "no_trade_reason": None}
    result = _normalize_pass2(raw)
    assert "ideas" not in result
    assert len(result["trade_ideas"]) == 1
    assert result["trade_ideas"][0]["ticker"] == "SLV"
    # normalisation also fills the signal-type classification defaults
    assert result["trade_ideas"][0]["signal_type"] == "situational"


def test_normalize_then_cap_bare_list():
    """End-to-end: a bare list flows through normalise + cap without error."""
    capped = _cap_holding_days(_normalize_pass2([{"ticker": "VIXY", "holding_days": 9}]))
    assert capped["trade_ideas"][0]["holding_days"] == 3


def test_normalize_pass2_confidence_alias():
    """Per-idea 'confidence' is mapped to 'conviction'."""
    result = _normalize_pass2({"trade_ideas": [{"ticker": "XLE", "confidence": 4}]})
    assert result["trade_ideas"][0]["conviction"] == 4


# ── Pass 1 normalisation ──────────────────────────────────────────────────────

def test_normalize_pass1_probability_confidence():
    """A 0-1 probability confidence is rescaled to an integer 1-5."""
    result = _normalize_pass1({"regime": "stagflation", "confidence": 0.72})
    assert result["confidence"] == 4


def test_normalize_pass1_narrative_alias():
    """'narrative' is mapped to 'regime_reasoning'."""
    result = _normalize_pass1({"regime": "risk_on", "confidence": 3, "narrative": "because"})
    assert result["regime_reasoning"] == "because"


# ── Pass 3 normalisation (the bug from the real run) ──────────────────────────

def test_normalize_pass3_trade_review_skip():
    """A model that uses trade_review/recommendation/approved=false must yield a skip."""
    raw = {
        "trade_review": [
            {
                "ticker": "XLE",
                "approved": False,
                "recommendation": "skip",
                "sizing_suggestion": "reduce to no more than 25% of standard risk",
            }
        ]
    }
    norm = _normalize_pass3(raw)
    review = norm["reviewed_ideas"][0]
    assert review["final_recommendation"] == "skip"
    assert review["size_adjustment"] == "skip"


def test_normalize_pass3_then_merge_filters_skip():
    """End-to-end: the real-run payload must filter the XLE trade out, not approve it."""
    ideas = {"trade_ideas": [{"ticker": "XLE", "direction": "long", "conviction": 4}]}
    raw_stress = {
        "trade_review": [
            {"ticker": "XLE", "approved": False, "recommendation": "skip"}
        ]
    }
    merged = merge_final_signals(
        regime={"regime": "stagflation"},
        trade_ideas=ideas,
        stress_test=_normalize_pass3(raw_stress),
        date_str="2026-06-04",
    )
    assert merged["final_trades"] == []
    assert merged["trades_filtered_out"] == 1


def test_coerce_size_adjustment_variants():
    assert _coerce_size_adjustment("reduce to 25% of risk", "reduce_size") == "quarter"
    assert _coerce_size_adjustment("half size", "reduce_size") == "half"
    assert _coerce_size_adjustment(None, "skip") == "skip"
    assert _coerce_size_adjustment(0.5, "reduce_size") == "half"
    assert _coerce_size_adjustment(None, "proceed") == "full"
