"""Google Trends data collector via pytrends.

Returns current interest score (0-100) and 4-week change per keyword.
Results are cached for 24 hours to avoid rate limits.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

KEYWORDS = ["Bitcoin", "recession", "gold price", "stock market crash", "inflation"]

_CACHE_PATH: Path  # set lazily so tests can redirect settings paths

_TREND_HINTS: dict[str, dict[str, str]] = {
    "Bitcoin": {
        "rising": "retail_btc_interest_rising_bullish_momentum",
        "falling": "declining_btc_interest_bearish",
        "neutral": "btc_interest_stable",
    },
    "recession": {
        "rising": "rising_recession_fear_risk_off_signal",
        "falling": "declining_recession_fear_risk_on",
        "neutral": "recession_concern_stable",
    },
    "gold price": {
        "rising": "rising_safe_haven_demand_risk_off",
        "falling": "declining_safe_haven_demand",
        "neutral": "gold_demand_stable",
    },
    "stock market crash": {
        "rising": "rising_crash_fear_strongly_risk_off",
        "falling": "declining_crash_fear_risk_on",
        "neutral": "crash_fear_stable",
    },
    "inflation": {
        "rising": "rising_inflation_concern_bearish_growth_assets",
        "falling": "declining_inflation_concern_neutral",
        "neutral": "inflation_concern_stable",
    },
}


def _interpretation(keyword: str, change_4w: float) -> str:
    hints = _TREND_HINTS.get(keyword, {})
    if change_4w >= 5:
        return hints.get("rising", "rising")
    if change_4w <= -5:
        return hints.get("falling", "falling")
    return hints.get("neutral", "stable")


def _cache_path() -> Path:
    return settings.SENTIMENT_DIR / "trends_latest.json"


def _cache_valid() -> bool:
    path = _cache_path()
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < 24


def _load_cache() -> dict | None:
    try:
        return json.loads(_cache_path().read_text())
    except Exception:
        return None


def _save_cache(data: dict) -> None:
    try:
        settings.ensure_dirs()
        _cache_path().write_text(json.dumps(data))
    except Exception as exc:
        logger.warning("Failed to save trends cache: %s", exc)


def _fetch() -> dict[str, Any]:
    from pytrends.request import TrendReq

    pytrends = TrendReq(
        hl="en-US",
        tz=0,
        timeout=(10, 30),
        retries=2,
        backoff_factor=0.5,
    )
    pytrends.build_payload(KEYWORDS, cat=0, timeframe="today 3-m", geo="")
    df: pd.DataFrame = pytrends.interest_over_time()

    if df.empty:
        logger.warning("Google Trends returned empty data")
        return {}

    df.index = pd.to_datetime(df.index)
    now = pd.Timestamp.now(tz=df.index.tz)

    results: dict[str, Any] = {}
    for keyword in KEYWORDS:
        if keyword not in df.columns:
            continue

        series = df[keyword].dropna()
        if series.empty:
            continue

        current_score = float(series.iloc[-1])

        # Value closest to 28 days ago
        cutoff = now - pd.Timedelta(days=28)
        prior = series[series.index <= cutoff]
        score_4w_ago = float(prior.iloc[-1]) if not prior.empty else float(series.iloc[0])
        change_4w = round(current_score - score_4w_ago, 1)

        results[keyword] = {
            "current_score": current_score,
            "score_4w_ago": score_4w_ago,
            "change_4w": change_4w,
            "interpretation": _interpretation(keyword, change_4w),
        }

    return results


def collect() -> dict[str, Any]:
    """Collect Google Trends interest scores. Returns cached data if <24h old."""
    if _cache_valid():
        cached = _load_cache()
        if cached:
            logger.debug("Returning cached Google Trends data")
            return cached

    try:
        data = _fetch()
        if data:
            _save_cache(data)
        return data
    except Exception as exc:
        logger.warning("Google Trends fetch failed: %s", exc)
        cached = _load_cache()
        if cached:
            logger.info("Returning stale trends cache due to fetch failure")
            return cached
        return {}


if __name__ == "__main__":
    import json as _json
    settings.configure_logging()
    print(_json.dumps(collect(), indent=2))
