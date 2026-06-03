"""Tests for the multi-pass LLM pipeline (llm_client.py)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from research.llm_client import (
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
    assert result["final_trades"][0]["size_multiplier"] == 0.5


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
