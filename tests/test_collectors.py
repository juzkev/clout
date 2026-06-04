"""Offline smoke tests for all collectors.

These tests verify:
  - No-key / no-network paths return empty dicts/lists and don't raise
  - Interpretation helper functions map values correctly
  - Price technical indicator math is correct (unit tests without network)
"""

import numpy as np
import pandas as pd
import pytest


# ── FRED collector ────────────────────────────────────────────────────────────

def test_fred_no_key_returns_empty():
    from research.collectors import fred_collector
    result = fred_collector.collect()
    assert result == {}


def test_fred_prior_observation_monthly():
    """Monthly series: a ~20-day lookback resolves to the PREVIOUS month, not 20 months."""
    from research.collectors.fred_collector import _prior_observation

    # Descending monthly observations (FEDFUNDS-style, month-start dated)
    monthly = [
        {"date": "2026-05-01", "value": "4.25"},
        {"date": "2026-04-01", "value": "4.25"},
        {"date": "2026-03-01", "value": "4.50"},
        {"date": "2026-02-01", "value": "4.50"},
        {"date": "2026-01-01", "value": "4.75"},
    ]
    val, prior_date = _prior_observation(monthly, "2026-05-01", 20)
    assert prior_date == "2026-04-01"  # previous month, NOT 20 months back
    assert val == 4.25


def test_fred_prior_observation_daily():
    """Daily series: ~20-day lookback lands ~20 calendar days back."""
    from datetime import date, timedelta
    from research.collectors.fred_collector import _prior_observation

    latest = date(2026, 5, 1)
    daily = [
        {"date": (latest - timedelta(days=i)).isoformat(), "value": str(100 - i)}
        for i in range(0, 40)
    ]
    val, prior_date = _prior_observation(daily, latest.isoformat(), 20)
    assert prior_date == (latest - timedelta(days=20)).isoformat()
    assert val == 80.0


# ── Sentiment collector ───────────────────────────────────────────────────────

def test_sentiment_returns_three_keys(monkeypatch):
    """With network calls mocked to raise, all sources return None gracefully."""
    import requests
    from research.collectors import sentiment_collector

    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    result = sentiment_collector.collect()
    assert "cnn_fear_greed" in result
    assert "aaii" in result
    assert "crypto_fear_greed" in result
    # All should be None (failed gracefully)
    for v in result.values():
        assert v is None


def test_aaii_regex_parses_percentages():
    """Layout-agnostic AAII fallback extracts the three readings from arbitrary HTML."""
    from research.collectors.sentiment_collector import _parse_aaii_regex

    html = """
        <div class="row"><span>Bullish</span><span>35.2%</span></div>
        <div class="row"><span>Neutral</span><span>30.8%</span></div>
        <div class="row"><span>Bearish</span><span>34.0%</span></div>
    """
    result = _parse_aaii_regex(html)
    assert result["bullish_pct"] == 35.2
    assert result["neutral_pct"] == 30.8
    assert result["bearish_pct"] == 34.0


def test_aaii_regex_no_match_returns_empty():
    from research.collectors.sentiment_collector import _parse_aaii_regex
    assert _parse_aaii_regex("<html>login required</html>") == {}


# ── Crypto collector ──────────────────────────────────────────────────────────

def test_crypto_no_network_returns_dict(monkeypatch):
    """All exchange endpoints blocked → collect() returns a dict without raising."""
    import requests
    from research.collectors import crypto_collector

    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    result = crypto_collector.collect()
    assert isinstance(result, dict)
    # sources key always present
    assert "sources" in result


def test_funding_interpretation_values():
    from research.collectors.crypto_collector import _funding_interpretation
    assert _funding_interpretation(0.1) == "very_elevated_longs_strongly_bearish"
    assert _funding_interpretation(0.03) == "elevated_longs_bearish"
    assert _funding_interpretation(0.0) == "balanced_neutral"
    assert _funding_interpretation(-0.03) == "elevated_shorts_bullish"
    assert _funding_interpretation(-0.1) == "very_elevated_shorts_strongly_bullish"


def test_oi_interpretation_values():
    from research.collectors.crypto_collector import _oi_interpretation
    assert _oi_interpretation(15.0) == "strong_oi_growth_watch_for_squeeze"
    assert _oi_interpretation(5.0) == "oi_growing_leverage_building"
    assert _oi_interpretation(0.0) == "oi_stable"
    assert _oi_interpretation(-5.0) == "oi_declining_deleveraging"
    assert _oi_interpretation(-15.0) == "sharp_oi_drop_forced_liquidations_likely"


def test_ls_interpretation_values():
    from research.collectors.crypto_collector import _ls_interpretation
    assert _ls_interpretation(1.5) == "longs_dominant_crowded"
    assert _ls_interpretation(1.0) == "balanced"
    assert _ls_interpretation(0.7) == "shorts_dominant_crowded"


# ── News collector ────────────────────────────────────────────────────────────

def test_news_no_network_returns_empty_lists(monkeypatch):
    """All RSS feeds blocked → collect() returns empty lists per bucket without raising."""
    import requests
    from research.collectors import news_collector

    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    result = news_collector.collect()
    assert isinstance(result, dict)
    for bucket in ("macro", "crypto", "commodity"):
        assert bucket in result
        assert result[bucket] == []


# ── Price collector — unit tests (no network) ─────────────────────────────────

def _make_close(n: int = 100, start: float = 100.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, n)
    prices = start * np.cumprod(1 + returns)
    return pd.Series(prices)


def test_rsi_range():
    from research.collectors.price_collector import _rsi
    close = _make_close(100)
    rsi = _rsi(close)
    assert 0 <= rsi <= 100


def test_rsi_overbought_when_all_gains():
    from research.collectors.price_collector import _rsi
    close = pd.Series([float(i) for i in range(1, 51)])
    rsi = _rsi(close)
    assert rsi > 70


def test_rsi_oversold_when_all_losses():
    from research.collectors.price_collector import _rsi
    close = pd.Series([float(50 - i) for i in range(50)])
    rsi = _rsi(close)
    assert rsi < 30


def test_momentum_rank_ordering():
    from research.collectors.price_collector import _add_momentum_rank
    results = {
        "A": {"return_20d_pct": 10.0},
        "B": {"return_20d_pct": -5.0},
        "C": {"return_20d_pct": 2.0},
    }
    _add_momentum_rank(results)
    assert results["A"]["momentum_rank_20d"] == 1
    assert results["C"]["momentum_rank_20d"] == 2
    assert results["B"]["momentum_rank_20d"] == 3


def test_price_no_yfinance(monkeypatch):
    """If yfinance not available, collect() returns empty dict without raising."""
    from research.collectors import price_collector
    monkeypatch.setattr(price_collector, "_YF_AVAILABLE", False)
    result = price_collector.collect()
    assert result == {}


# ── Trends collector ──────────────────────────────────────────────────────────

def test_trends_no_network_returns_empty(monkeypatch):
    """pytrends network failure → returns empty dict without raising."""
    from research.collectors import trends_collector

    def _fail(*a, **kw):
        raise ConnectionError("no network")

    monkeypatch.setattr("pytrends.request.TrendReq.build_payload", _fail)
    result = trends_collector.collect()
    assert isinstance(result, dict)


def test_trends_interpretation_rising():
    from research.collectors.trends_collector import _interpretation
    assert _interpretation("recession", 10.0) == "rising_recession_fear_risk_off_signal"


def test_trends_interpretation_falling():
    from research.collectors.trends_collector import _interpretation
    assert _interpretation("Bitcoin", -10.0) == "declining_btc_interest_bearish"


def test_trends_interpretation_neutral():
    from research.collectors.trends_collector import _interpretation
    assert _interpretation("inflation", 2.0) == "inflation_concern_stable"


# ── COT collector ─────────────────────────────────────────────────────────────

def test_cot_no_network_returns_empty(monkeypatch):
    """Network failure → returns empty dict without raising."""
    import requests
    from research.collectors import cot_collector

    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    result = cot_collector.collect()
    assert isinstance(result, dict)


def test_cot_percentile_rank():
    from research.collectors.cot_collector import _percentile_rank
    import pandas as pd
    series = pd.Series([0.0, 10.0, 20.0, 30.0, 40.0])
    assert _percentile_rank(series, 40.0) == 80.0  # 4 of 5 values are < 40
    assert _percentile_rank(series, 0.0) == 0.0
    assert _percentile_rank(series, 50.0) == 100.0


def test_cot_interpretation_extremes():
    from research.collectors.cot_collector import _cot_interpretation
    assert _cot_interpretation(85.0) == "extreme_longs_contrarian_bearish"
    assert _cot_interpretation(15.0) == "extreme_shorts_contrarian_bullish"
    assert _cot_interpretation(50.0) == "positioning_neutral"


def test_cot_interpretation_mid():
    from research.collectors.cot_collector import _cot_interpretation
    assert _cot_interpretation(65.0) == "elevated_longs_mild_bearish"
    assert _cot_interpretation(35.0) == "depressed_longs_mild_bullish"


# ── Calendar collector ────────────────────────────────────────────────────────

def test_calendar_no_network_returns_structure(monkeypatch):
    """Network failure → returns dict with empty lists, no exception."""
    import requests
    from research.collectors import calendar_collector

    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("no network")))
    result = calendar_collector.collect()
    assert isinstance(result, dict)
    assert "economic" in result
    assert "earnings" in result
    assert "all_events" in result
    assert result["economic"] == []


def test_calendar_affected_tickers_fomc():
    from research.collectors.calendar_collector import _affected_tickers
    tickers = _affected_tickers("FOMC Statement")
    assert "TLT" in tickers
    assert "SPY" in tickers


def test_calendar_affected_tickers_cpi():
    from research.collectors.calendar_collector import _affected_tickers
    tickers = _affected_tickers("CPI m/m")
    assert "TLT" in tickers
    assert "GLD" in tickers


def test_calendar_ff_date_parsing():
    from research.collectors.calendar_collector import _parse_ff_date
    dt = _parse_ff_date("Jan 15 2025")
    assert dt is not None
    assert dt.month == 1
    assert dt.day == 15
    assert dt.year == 2025
