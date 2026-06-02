"""Crypto derivatives and market data collector.

Primary source: Coinglass v2 API (requires COINGLASS_API_KEY).
Fallback:       CoinGecko free API (no key required).

Every numeric field is paired with a human-readable interpretation hint,
e.g. {"btc_funding_rate": 0.03, "funding_interpretation": "elevated_longs_bearish"}.
"""

import logging
from typing import Any

import requests

from config import settings

logger = logging.getLogger(__name__)

_COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

FUNDING_EXCHANGES = ["Binance", "Bybit", "OKX"]


# ── Interpretation helpers ─────────────────────────────────────────────────────

def _funding_interpretation(rate: float) -> str:
    if rate > 0.05:
        return "very_elevated_longs_strongly_bearish"
    if rate > 0.02:
        return "elevated_longs_bearish"
    if rate > 0.005:
        return "mild_longs_neutral"
    if rate > -0.005:
        return "balanced_neutral"
    if rate > -0.02:
        return "mild_shorts_neutral"
    if rate > -0.05:
        return "elevated_shorts_bullish"
    return "very_elevated_shorts_strongly_bullish"


def _oi_interpretation(change_pct: float) -> str:
    if change_pct > 10:
        return "strong_oi_growth_watch_for_squeeze"
    if change_pct > 3:
        return "oi_growing_leverage_building"
    if change_pct > -3:
        return "oi_stable"
    if change_pct > -10:
        return "oi_declining_deleveraging"
    return "sharp_oi_drop_forced_liquidations_likely"


def _ls_interpretation(ratio: float) -> str:
    if ratio > 1.2:
        return "longs_dominant_crowded"
    if ratio > 0.9:
        return "balanced"
    return "shorts_dominant_crowded"


# ── Coinglass ─────────────────────────────────────────────────────────────────

def _coinglass_headers() -> dict[str, str]:
    return {"coinglassSecret": settings.COINGLASS_API_KEY}


def _fetch_funding_rate() -> dict[str, Any] | None:
    """Average BTC funding rate across major exchanges."""
    try:
        url = f"{_COINGLASS_BASE}/funding_rate_history"
        resp = requests.get(
            url,
            headers=_coinglass_headers(),
            params={"symbol": "BTC", "interval": "8h", "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None

        # data is a list of exchange entries
        rates = []
        for entry in data:
            exch = entry.get("exchangeName", "")
            if exch in FUNDING_EXCHANGES:
                rate = entry.get("fundingRate")
                if rate is not None:
                    rates.append(float(rate))

        if not rates:
            return None

        avg = round(sum(rates) / len(rates), 6)
        return {
            "btc_funding_rate": avg,
            "funding_interpretation": _funding_interpretation(avg),
            "exchanges_sampled": FUNDING_EXCHANGES,
        }
    except Exception as exc:
        logger.warning("Coinglass funding rate fetch failed: %s", exc)
        return None


def _fetch_open_interest() -> dict[str, Any] | None:
    try:
        url = f"{_COINGLASS_BASE}/open_interest"
        resp = requests.get(
            url,
            headers=_coinglass_headers(),
            params={"symbol": "BTC"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        change_pct = data.get("openInterestChangePercent24h")
        if change_pct is None:
            return None
        change_pct = float(change_pct)
        return {
            "btc_oi_24h_change_pct": round(change_pct, 2),
            "oi_interpretation": _oi_interpretation(change_pct),
        }
    except Exception as exc:
        logger.warning("Coinglass open interest fetch failed: %s", exc)
        return None


def _fetch_long_short_ratio() -> dict[str, Any] | None:
    try:
        url = f"{_COINGLASS_BASE}/futures/longShortChart"
        resp = requests.get(
            url,
            headers=_coinglass_headers(),
            params={"symbol": "BTC", "interval": "1h", "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        ratio = float(data[-1].get("longShortRatio", 1.0))
        return {
            "btc_long_short_ratio": round(ratio, 3),
            "ls_interpretation": _ls_interpretation(ratio),
        }
    except Exception as exc:
        logger.warning("Coinglass long/short ratio fetch failed: %s", exc)
        return None


def _fetch_liquidations() -> dict[str, Any] | None:
    try:
        url = f"{_COINGLASS_BASE}/liquidation_history"
        resp = requests.get(
            url,
            headers=_coinglass_headers(),
            params={"symbol": "BTC", "interval": "24h", "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return None
        latest = data[-1]
        return {
            "btc_long_liquidations_24h_usd": latest.get("longLiquidationUsd"),
            "btc_short_liquidations_24h_usd": latest.get("shortLiquidationUsd"),
        }
    except Exception as exc:
        logger.warning("Coinglass liquidations fetch failed: %s", exc)
        return None


def _collect_coinglass() -> dict[str, Any] | None:
    """Attempt to pull all Coinglass data; return None if all sub-calls fail."""
    result: dict[str, Any] = {"source": "coinglass"}
    success = False

    for fetcher in (_fetch_funding_rate, _fetch_open_interest, _fetch_long_short_ratio, _fetch_liquidations):
        sub = fetcher()
        if sub:
            result.update(sub)
            success = True

    return result if success else None


# ── CoinGecko fallback ────────────────────────────────────────────────────────

def _collect_coingecko() -> dict[str, Any]:
    """CoinGecko free API — no key required."""
    try:
        url = f"{_COINGECKO_BASE}/coins/markets"
        resp = requests.get(
            url,
            params={
                "vs_currency": "usd",
                "ids": "bitcoin",
                "sparkline": "false",
                "price_change_percentage": "24h,7d",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {}
        btc = data[0]
        price_24h_chg = btc.get("price_change_percentage_24h", 0.0) or 0.0
        price_7d_chg = btc.get("price_change_percentage_7d_in_currency", 0.0) or 0.0
        return {
            "source": "coingecko_fallback",
            "btc_price_usd": btc.get("current_price"),
            "btc_24h_change_pct": round(price_24h_chg, 2),
            "btc_7d_change_pct": round(price_7d_chg, 2),
            "btc_market_cap_usd": btc.get("market_cap"),
        }
    except Exception as exc:
        logger.warning("CoinGecko fallback fetch failed: %s", exc)
        return {}


# ── Public interface ──────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """Collect crypto market data. Falls back to CoinGecko if Coinglass unavailable."""
    if settings.COINGLASS_API_KEY:
        coinglass_data = _collect_coinglass()
        if coinglass_data:
            return coinglass_data
        logger.warning("Coinglass returned no data — falling back to CoinGecko")
    else:
        logger.warning("COINGLASS_API_KEY not set — using CoinGecko fallback")

    return _collect_coingecko()


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
