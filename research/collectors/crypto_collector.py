"""Crypto derivatives and market data collector.

Pulls directly from exchange public APIs — no API key required.

Primary sources:
  Binance Futures  — funding rate, open interest, long/short ratio
  Bybit            — funding rate, open interest, long/short ratio (averaged with Binance)

Fallback:
  CoinGecko free API — spot price and 24h/7d change

Every numeric field is paired with a human-readable interpretation hint,
e.g. {"btc_funding_rate": 0.03, "funding_interpretation": "elevated_longs_bearish"}.
"""

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BINANCE_BASE = "https://fapi.binance.com"
_BYBIT_BASE = "https://api.bybit.com/v5"
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

_TIMEOUT = 15


# ── Interpretation helpers ────────────────────────────────────────────────────

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


# ── Binance Futures ───────────────────────────────────────────────────────────

def _binance_funding_rate() -> float | None:
    """Current BTC/USDT perpetual funding rate from Binance."""
    try:
        resp = requests.get(
            f"{_BINANCE_BASE}/fapi/v1/premiumIndex",
            params={"symbol": "BTCUSDT"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return float(resp.json()["lastFundingRate"])
    except Exception as exc:
        logger.warning("Binance funding rate failed: %s", exc)
        return None


def _binance_open_interest() -> dict[str, Any] | None:
    """BTC open interest + 24h change from Binance."""
    try:
        # Current OI
        oi_resp = requests.get(
            f"{_BINANCE_BASE}/fapi/v1/openInterest",
            params={"symbol": "BTCUSDT"},
            timeout=_TIMEOUT,
        )
        oi_resp.raise_for_status()
        current_oi = float(oi_resp.json()["openInterest"])

        # 24h ago OI for change calculation
        hist_resp = requests.get(
            f"{_BINANCE_BASE}/futures/data/openInterestHist",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 25},
            timeout=_TIMEOUT,
        )
        hist_resp.raise_for_status()
        hist = hist_resp.json()
        prior_oi = float(hist[0]["sumOpenInterest"]) if hist else None

        change_pct = None
        if prior_oi and prior_oi > 0:
            change_pct = round((current_oi / prior_oi - 1) * 100, 2)

        return {
            "btc_oi_contracts": round(current_oi, 2),
            "btc_oi_24h_change_pct": change_pct,
            "oi_interpretation": _oi_interpretation(change_pct) if change_pct is not None else "unknown",
        }
    except Exception as exc:
        logger.warning("Binance open interest failed: %s", exc)
        return None


def _binance_long_short_ratio() -> dict[str, Any] | None:
    """Global BTC long/short account ratio from Binance."""
    try:
        resp = requests.get(
            f"{_BINANCE_BASE}/futures/data/globalLongShortAccountRatio",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        ratio = float(data[0]["longShortRatio"])
        return {
            "btc_long_short_ratio": round(ratio, 3),
            "ls_interpretation": _ls_interpretation(ratio),
        }
    except Exception as exc:
        logger.warning("Binance long/short ratio failed: %s", exc)
        return None


# ── Bybit ─────────────────────────────────────────────────────────────────────

def _bybit_funding_rate() -> float | None:
    """Current BTC/USDT perpetual funding rate from Bybit."""
    try:
        resp = requests.get(
            f"{_BYBIT_BASE}/market/tickers",
            params={"category": "linear", "symbol": "BTCUSDT"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("result", {}).get("list", [])
        if not items:
            return None
        return float(items[0]["fundingRate"])
    except Exception as exc:
        logger.warning("Bybit funding rate failed: %s", exc)
        return None


# ── CoinGecko fallback ────────────────────────────────────────────────────────

def _coingecko_spot() -> dict[str, Any]:
    """BTC spot price and returns from CoinGecko (no key required)."""
    try:
        resp = requests.get(
            f"{_COINGECKO_BASE}/coins/markets",
            params={
                "vs_currency": "usd",
                "ids": "bitcoin",
                "sparkline": "false",
                "price_change_percentage": "24h,7d",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {}
        btc = data[0]
        return {
            "btc_price_usd": btc.get("current_price"),
            "btc_24h_change_pct": round(btc.get("price_change_percentage_24h") or 0.0, 2),
            "btc_7d_change_pct": round(btc.get("price_change_percentage_7d_in_currency") or 0.0, 2),
            "btc_market_cap_usd": btc.get("market_cap"),
        }
    except Exception as exc:
        logger.warning("CoinGecko spot price failed: %s", exc)
        return {}


# ── Public interface ──────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """Collect BTC derivatives data from Binance + Bybit public APIs."""
    result: dict[str, Any] = {"sources": ["binance_futures", "bybit"]}

    # Funding rate: average Binance and Bybit
    binance_rate = _binance_funding_rate()
    bybit_rate = _bybit_funding_rate()
    rates = [r for r in (binance_rate, bybit_rate) if r is not None]
    if rates:
        avg_rate = round(sum(rates) / len(rates), 6)
        result["btc_funding_rate"] = avg_rate
        result["btc_funding_rate_binance"] = binance_rate
        result["btc_funding_rate_bybit"] = bybit_rate
        result["funding_interpretation"] = _funding_interpretation(avg_rate)
    else:
        logger.warning("No funding rate data available from Binance or Bybit")

    # Open interest (Binance)
    oi = _binance_open_interest()
    if oi:
        result.update(oi)

    # Long/short ratio (Binance)
    ls = _binance_long_short_ratio()
    if ls:
        result.update(ls)

    # Spot price (CoinGecko, always attempt)
    result.update(_coingecko_spot())

    return result


if __name__ == "__main__":
    import json
    from config import settings
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
