"""FRED macroeconomic data collector.

Pulls the latest values and 20-day change for key series from the
Federal Reserve Economic Data API (https://fred.stlouisfed.org/docs/api/fred/).
Returns an empty dict and logs a warning when FRED_API_KEY is missing.
"""

import logging
from datetime import date, timedelta
from typing import Any

import requests

from config import settings

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"

SERIES: dict[str, str] = {
    "T10Y2Y": "10Y-2Y Yield Curve Spread",
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "UNRATE": "Unemployment Rate",
    "VIXCLS": "VIX (CBOE Volatility Index)",
    "DGS10": "10-Year Treasury Yield",
}


def _fetch_observations(series_id: str, api_key: str, limit: int = 30) -> list[dict]:
    """Return the most recent `limit` observations for a FRED series."""
    url = f"{FRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("observations", [])


def _valid_obs(obs: list[dict]) -> list[dict]:
    """Filter out missing-value markers ('.')."""
    return [o for o in obs if o.get("value", ".") != "."]


def _parse_series(series_id: str, api_key: str) -> dict[str, Any]:
    """Fetch a series and return latest value, date, and 20-day change."""
    obs = _fetch_observations(series_id, api_key, limit=40)
    valid = _valid_obs(obs)
    if not valid:
        return {}

    latest_val = float(valid[0]["value"])
    latest_date = valid[0]["date"]

    # ~20 observations back (obs are in descending order)
    prior_val = float(valid[min(20, len(valid) - 1)]["value"])
    change_20d = round(latest_val - prior_val, 4)

    result: dict[str, Any] = {
        "latest": latest_val,
        "latest_date": latest_date,
        "change_20d": change_20d,
    }

    # For CPI, add year-over-year change using obs from ~12 months ago
    if series_id == "CPIAUCSL":
        obs_long = _fetch_observations(series_id, api_key, limit=15)
        valid_long = _valid_obs(obs_long)
        if len(valid_long) >= 13:
            yoy_prior = float(valid_long[12]["value"])
            result["yoy_pct"] = round((latest_val / yoy_prior - 1) * 100, 2)

    return result


def collect() -> dict[str, Any]:
    """Collect FRED macro data. Returns {} with a warning if no API key."""
    if not settings.FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — skipping FRED data collection")
        return {}

    results: dict[str, Any] = {}
    for series_id, label in SERIES.items():
        try:
            data = _parse_series(series_id, settings.FRED_API_KEY)
            if data:
                results[series_id] = {"label": label, **data}
            else:
                logger.warning("No valid data returned for FRED series %s", series_id)
        except Exception as exc:
            logger.warning("Failed to fetch FRED series %s: %s", series_id, exc)

    return results


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
