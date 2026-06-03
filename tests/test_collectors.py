"""Offline smoke tests for all five collectors.

These tests verify:
  - No-key paths return empty dicts/lists and don't raise
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
