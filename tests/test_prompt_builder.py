"""Tests for research/prompt_builder.py."""

import pytest


SAMPLE_FRED = {
    "T10Y2Y": {"label": "10Y-2Y Yield Curve Spread", "latest": -0.42, "latest_date": "2024-01-15", "change_20d": 0.12},
    "FEDFUNDS": {"label": "Fed Funds Rate", "latest": 5.33, "latest_date": "2024-01-15", "change_20d": 0.0},
    "CPIAUCSL": {"label": "CPI", "latest": 312.0, "latest_date": "2024-01-15", "change_20d": 0.5, "yoy_pct": 3.4},
}

SAMPLE_SENTIMENT = {
    "cnn_fear_greed": {"score": 62.0, "rating": "Greed", "week_change": 5.0},
    "aaii": {"bullish_pct": 45.2, "neutral_pct": 29.1, "bearish_pct": 25.7},
    "crypto_fear_greed": {"value": 71, "classification": "Greed"},
}

SAMPLE_CRYPTO = {
    "source": "coinglass",
    "btc_funding_rate": 0.025,
    "funding_interpretation": "elevated_longs_bearish",
    "btc_oi_24h_change_pct": 4.5,
    "oi_interpretation": "oi_growing_leverage_building",
    "btc_long_short_ratio": 1.15,
    "ls_interpretation": "balanced",
}

SAMPLE_PRICE = {
    "SPY": {
        "ticker": "SPY",
        "current_price": 485.20,
        "return_1d_pct": 0.45,
        "return_5d_pct": 1.2,
        "return_20d_pct": 3.8,
        "vol_20d_ann_pct": 12.5,
        "rsi_14": 58.3,
        "above_sma50": True,
        "sma50": 475.0,
        "momentum_rank_20d": 1,
    },
    "TLT": {
        "ticker": "TLT",
        "current_price": 95.10,
        "return_1d_pct": -0.2,
        "return_5d_pct": -1.5,
        "return_20d_pct": -4.2,
        "vol_20d_ann_pct": 15.1,
        "rsi_14": 38.0,
        "above_sma50": False,
        "sma50": 97.0,
        "momentum_rank_20d": 2,
    },
}

SAMPLE_NEWS = {
    "macro": [{"title": "Fed holds rates", "source": "Reuters", "publishedAt": "2024-01-15T10:00:00Z", "url": "https://example.com"}],
    "crypto": [{"title": "BTC rallies", "source": "CoinDesk", "publishedAt": "2024-01-15T09:00:00Z", "url": "https://example.com"}],
    "commodity": [],
}


def test_build_prompt_contains_all_sections():
    from research.prompt_builder import build_prompt
    prompt = build_prompt(SAMPLE_FRED, SAMPLE_SENTIMENT, SAMPLE_CRYPTO, SAMPLE_PRICE, SAMPLE_NEWS)
    assert "MACRO ENVIRONMENT" in prompt
    assert "MARKET SENTIMENT" in prompt
    assert "CRYPTO SIGNALS" in prompt
    assert "PRICE MOMENTUM" in prompt
    assert "RECENT NEWS THEMES" in prompt


def test_build_prompt_contains_output_format():
    from research.prompt_builder import build_prompt
    prompt = build_prompt(SAMPLE_FRED, SAMPLE_SENTIMENT, SAMPLE_CRYPTO, SAMPLE_PRICE, SAMPLE_NEWS)
    assert "OUTPUT FORMAT" in prompt
    assert "market_regime" in prompt
    assert "trade_ideas" in prompt


def test_build_prompt_with_empty_inputs():
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, {}, {}, {}, {})
    assert "MACRO ENVIRONMENT" in prompt
    assert "data unavailable" in prompt


def test_build_prompt_fred_values_present():
    from research.prompt_builder import build_prompt
    prompt = build_prompt(SAMPLE_FRED, {}, {}, {}, {})
    assert "T10Y2Y" in prompt
    assert "FEDFUNDS" in prompt


def test_build_prompt_sentiment_values_present():
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, SAMPLE_SENTIMENT, {}, {}, {})
    assert "62" in prompt  # CNN score
    assert "Greed" in prompt
    assert "45.2" in prompt  # AAII bullish


def test_build_prompt_price_ranked():
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, {}, {}, SAMPLE_PRICE, {})
    spy_pos = prompt.find("SPY")
    tlt_pos = prompt.find("TLT")
    assert spy_pos < tlt_pos  # SPY rank 1 should appear before TLT rank 2


def test_save_prompt_writes_file(tmp_data_dirs):
    from research.prompt_builder import build_prompt, save_prompt
    prompt = build_prompt(SAMPLE_FRED, SAMPLE_SENTIMENT, SAMPLE_CRYPTO, SAMPLE_PRICE, SAMPLE_NEWS)
    path = save_prompt(prompt)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "MACRO ENVIRONMENT" in content
    assert len(content) > 100


def test_save_prompt_filename_format(tmp_data_dirs):
    from datetime import date
    from research.prompt_builder import build_prompt, save_prompt
    prompt = build_prompt({}, {}, {}, {}, {})
    path = save_prompt(prompt)
    today = date.today().strftime("%Y-%m-%d")
    assert path.name == f"{today}_prompt.txt"


# ── New sections ──────────────────────────────────────────────────────────────

SAMPLE_TRENDS = {
    "Bitcoin": {"current_score": 72, "score_4w_ago": 55, "change_4w": 17.0, "interpretation": "retail_btc_interest_rising_bullish_momentum"},
    "recession": {"current_score": 38, "score_4w_ago": 30, "change_4w": 8.0, "interpretation": "rising_recession_fear_risk_off_signal"},
}

SAMPLE_COT = {
    "gold": {
        "label": "Gold (GC)", "latest_date": "240112", "weeks_of_history": 26,
        "spec_longs": 180000, "spec_shorts": 60000, "net_spec_position": 120000,
        "net_spec_percentile": 82.0, "interpretation": "extreme_longs_contrarian_bearish",
        "affected_tickers": ["GLD"],
    }
}

SAMPLE_CALENDAR = {
    "economic": [
        {"date": "2024-01-15", "time": "8:30am", "event": "CPI m/m", "impact": "high",
         "forecast": "0.2%", "previous": "0.1%", "potential_affected_tickers": ["TLT", "GLD"]},
    ],
    "earnings": [],
    "all_events": [
        {"date": "2024-01-15", "time": "8:30am", "event": "CPI m/m", "impact": "high",
         "forecast": "0.2%", "previous": "0.1%", "potential_affected_tickers": ["TLT", "GLD"]},
    ],
    "days_ahead": 7,
}


def test_build_prompt_contains_new_sections():
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, {}, {}, {}, {}, trends=SAMPLE_TRENDS, cot=SAMPLE_COT, calendar=SAMPLE_CALENDAR)
    assert "SEARCH TREND SIGNALS" in prompt
    assert "POSITIONING EXTREMES (COT)" in prompt
    assert "UPCOMING CATALYSTS" in prompt


def test_build_prompt_trends_values():
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, {}, {}, {}, {}, trends=SAMPLE_TRENDS)
    assert "Bitcoin" in prompt
    assert "recession" in prompt
    assert "risk_off" in prompt


def test_build_prompt_cot_values():
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, {}, {}, {}, {}, cot=SAMPLE_COT)
    assert "Gold (GC)" in prompt
    assert "82.0%" in prompt
    assert "contrarian_bearish" in prompt


def test_build_prompt_calendar_values():
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, {}, {}, {}, {}, calendar=SAMPLE_CALENDAR)
    assert "CPI m/m" in prompt
    assert "TLT" in prompt


def test_build_prompt_new_sections_empty_graceful():
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, {}, {}, {}, {}, trends={}, cot={}, calendar={})
    assert "SEARCH TREND SIGNALS" in prompt
    assert "data unavailable" in prompt


def test_build_prompt_backward_compatible():
    """Old 5-arg call still works (new params default to None)."""
    from research.prompt_builder import build_prompt
    prompt = build_prompt({}, {}, {}, {}, {})
    assert "SEARCH TREND SIGNALS" in prompt  # section present but shows unavailable
