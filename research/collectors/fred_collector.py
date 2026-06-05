"""FRED macroeconomic data collector.

Pulls the latest value and a recent (~20 calendar-day) change for key series
from the Federal Reserve Economic Data API
(https://fred.stlouisfed.org/docs/api/fred/).

The recent change uses a DATE-BASED lookback so it is consistent across series
frequencies: for daily series (yields, VIX) it is roughly a 20-day change; for
monthly series (Fed Funds, CPI, unemployment) it resolves to the prior month's
print (a true month-over-month change). The matched comparison date is returned
as `prior_date` for transparency.

Returns an empty dict and logs a warning when FRED_API_KEY is missing.
"""

import logging
import time
from datetime import date, timedelta
from typing import Any

import requests

from config import settings

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"

# Transient statuses worth retrying (gateway / rate-limit / overload)
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3


def _scrub(text: str) -> str:
    """Remove the API key from any string before it reaches a log."""
    key = settings.FRED_API_KEY
    return text.replace(key, "***") if key else text

SERIES: dict[str, str] = {
    "T10Y2Y": "10Y-2Y Yield Curve Spread",
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "UNRATE": "Unemployment Rate",
    "VIXCLS": "VIX (CBOE Volatility Index)",
    "DGS10": "10-Year Treasury Yield",
    "T10YIE": "10Y Breakeven Inflation Rate",
    "T5YIFR": "5Y5Y Forward Inflation Expectation Rate",
    "DFII10": "10Y Real Yield (TIPS)",
    "DFII5": "5Y Real Yield (TIPS)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS Credit Spread",
}

_LOOKBACK_DAYS = 20
_SMA_WINDOW = 200


def _fetch_observations(series_id: str, api_key: str, limit: int = 30) -> list[dict]:
    """Return the most recent `limit` observations, retrying transient errors."""
    url = f"{FRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            status = resp.status_code
            if status in _RETRY_STATUS:
                last_exc = requests.HTTPError(f"{status} from FRED for {series_id}")
                if attempt < _MAX_RETRIES - 1:
                    backoff = 2 ** attempt  # 1s, 2s, 4s
                    logger.info(
                        "FRED %s returned %d — retrying in %ds (attempt %d/%d)",
                        series_id, status, backoff, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(backoff)
                    continue
                raise last_exc
            resp.raise_for_status()  # non-retryable 4xx (e.g. bad key) → raise now
            return resp.json().get("observations", [])
        except requests.exceptions.RequestException as exc:
            # Connection/timeout errors with no response are also transient
            if exc.response is not None and exc.response.status_code not in _RETRY_STATUS:
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                backoff = 2 ** attempt
                logger.info(
                    "FRED %s request failed — retrying in %ds (attempt %d/%d)",
                    series_id, backoff, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(backoff)

    raise last_exc if last_exc else RuntimeError(f"FRED fetch failed for {series_id}")


def _valid_obs(obs: list[dict]) -> list[dict]:
    """Filter out missing-value markers ('.')."""
    return [o for o in obs if o.get("value", ".") != "."]


def _prior_observation(
    valid: list[dict], latest_date: str, lookback_days: int = _LOOKBACK_DAYS
) -> tuple[float, str]:
    """Find the observation on/just before (latest_date - lookback_days).

    `valid` is in descending date order. This is frequency-aware: for monthly
    series ~20 days before a month-start print lands in the prior month, so the
    change becomes a true month-over-month change rather than a 20-period change.
    """
    target = date.fromisoformat(latest_date) - timedelta(days=lookback_days)
    for o in valid:
        if date.fromisoformat(o["date"]) <= target:
            return float(o["value"]), o["date"]
    # fallback: oldest observation we have
    return float(valid[-1]["value"]), valid[-1]["date"]


def _parse_series(series_id: str, api_key: str) -> dict[str, Any]:
    """Fetch a series and return latest value, date, and recent change."""
    obs = _fetch_observations(series_id, api_key, limit=60)
    valid = _valid_obs(obs)
    if not valid:
        return {}

    latest_val = float(valid[0]["value"])
    latest_date = valid[0]["date"]

    prior_val, prior_date = _prior_observation(valid, latest_date, _LOOKBACK_DAYS)
    change_20d = round(latest_val - prior_val, 4)

    result: dict[str, Any] = {
        "latest": latest_val,
        "latest_date": latest_date,
        "change_20d": change_20d,
        "prior_date": prior_date,
    }

    # For CPI, add year-over-year change using the observation ~12 months back
    if series_id == "CPIAUCSL":
        yoy_target = date.fromisoformat(latest_date) - timedelta(days=365)
        for o in valid:
            if date.fromisoformat(o["date"]) <= yoy_target:
                result["yoy_pct"] = round((latest_val / float(o["value"]) - 1) * 100, 2)
                break

    return result


# ── Interpretation helpers (same pattern as crypto_collector.py) ──────────────

def _real_yield_interpretation(ry: float) -> str:
    if ry > 2.5:
        return "very_restrictive_bearish_gold_and_risk"
    if ry > 1.5:
        return "restrictive_headwind_for_gold"
    if ry > 0.5:
        return "moderately_positive_neutral"
    if ry > 0.0:
        return "low_positive_supportive_risk"
    return "negative_real_yield_supportive_gold"


def _real_yield_change_interpretation(change: float) -> str:
    if change > 0.15:
        return "real_yields_rising_headwind_gold_risk"
    if change < -0.15:
        return "real_yields_falling_tailwind_gold_risk"
    return "real_yields_stable"


def _inflation_expectation_interpretation(val: float) -> str:
    if val > 2.8:
        return "elevated_inflation_expectations_unanchored"
    if val > 2.2:
        return "anchored_near_target"
    if val > 1.8:
        return "moderate_disinflation"
    return "low_inflation_expectations_deflation_risk"


def _curve_momentum_label(change_bps: float) -> str:
    return "steepening" if change_bps >= 0 else "flattening"


def _curve_momentum_interpretation(change_bps: float) -> str:
    if change_bps > 10:
        return "rapid_steepening_growth_or_easing_repricing"
    if change_bps > 2:
        return "steepening_reflation_bias"
    if change_bps > -2:
        return "curve_stable"
    if change_bps > -10:
        return "flattening_slowdown_bias"
    return "rapid_flattening_recession_signal"


def _dgs10_sma_interpretation(position: str) -> str:
    return (
        "rates_uptrend_bearish_bonds" if position == "above"
        else "rates_downtrend_bullish_bonds"
    )


def _credit_spread_interpretation(oas: float) -> str:
    if oas > 8.0:
        return "wide_credit_stress_risk_off"
    if oas > 5.0:
        return "elevated_caution"
    if oas > 3.5:
        return "normal"
    return "tight_complacent_risk_on"


def _credit_spread_change_interpretation(change: float) -> str:
    if change > 0.3:
        return "widening_risk_off_warning"
    if change < -0.3:
        return "tightening_risk_on"
    return "credit_spreads_stable"


# ── Derived rate / inflation fields ───────────────────────────────────────────

def _latest(results: dict, series_id: str) -> float | None:
    entry = results.get(series_id)
    return entry.get("latest") if isinstance(entry, dict) else None


def _change_20d(results: dict, series_id: str) -> float | None:
    entry = results.get(series_id)
    return entry.get("change_20d") if isinstance(entry, dict) else None


def _dgs10_200sma(api_key: str) -> float | None:
    """Compute the 200-observation SMA of the 10Y yield, or None on failure."""
    try:
        obs = _fetch_observations("DGS10", api_key, limit=_SMA_WINDOW + 20)
        values = [float(o["value"]) for o in _valid_obs(obs)][:_SMA_WINDOW]
        if len(values) < 20:  # too little history to be meaningful
            return None
        return round(sum(values) / len(values), 4)
    except Exception as exc:
        logger.warning("Failed to compute DGS10 200d SMA: %s", _scrub(str(exc)))
        return None


def _compute_derived(results: dict, api_key: str) -> dict[str, Any]:
    """Build forward-rate / inflation-expectation derived fields from raw series."""
    derived: dict[str, Any] = {}

    dgs10 = _latest(results, "DGS10")
    t10yie = _latest(results, "T10YIE")
    t5yifr = _latest(results, "T5YIFR")

    # Real yield (10Y nominal minus 10Y breakeven inflation)
    if dgs10 is not None and t10yie is not None:
        real_yield = round(dgs10 - t10yie, 4)
        derived["real_yield_10y"] = real_yield
        derived["real_yield_10y_interpretation"] = _real_yield_interpretation(real_yield)

        dgs10_chg = _change_20d(results, "DGS10")
        t10yie_chg = _change_20d(results, "T10YIE")
        if dgs10_chg is not None and t10yie_chg is not None:
            ry_chg = round(dgs10_chg - t10yie_chg, 4)
            derived["real_yield_10y_20d_change"] = ry_chg
            derived["real_yield_10y_20d_change_interpretation"] = _real_yield_change_interpretation(ry_chg)

    # Inflation expectations
    if t10yie is not None:
        derived["breakeven_inflation_10y"] = t10yie
        derived["breakeven_inflation_10y_interpretation"] = _inflation_expectation_interpretation(t10yie)
    if t5yifr is not None:
        derived["forward_inflation_5y5y"] = t5yifr
        derived["forward_inflation_5y5y_interpretation"] = _inflation_expectation_interpretation(t5yifr)

    # Yield-curve momentum (20d change of the 10Y-2Y spread, in basis points)
    curve_chg = _change_20d(results, "T10Y2Y")
    if curve_chg is not None:
        curve_bps = round(curve_chg * 100, 1)
        derived["yield_curve_momentum_20d"] = curve_bps
        derived["yield_curve_momentum_label"] = _curve_momentum_label(curve_bps)
        derived["yield_curve_momentum_interpretation"] = _curve_momentum_interpretation(curve_bps)

    # 10Y yield vs its 200d SMA
    if dgs10 is not None:
        sma = _dgs10_200sma(api_key)
        if sma is not None:
            position = "above" if dgs10 >= sma else "below"
            derived["dgs10_vs_200sma"] = position
            derived["dgs10_200sma"] = sma
            derived["dgs10_vs_200sma_interpretation"] = _dgs10_sma_interpretation(position)

    # Credit spreads (HY OAS)
    oas = _latest(results, "BAMLH0A0HYM2")
    if oas is not None:
        derived["credit_spread_oas"] = oas
        derived["credit_spread_oas_interpretation"] = _credit_spread_interpretation(oas)
        oas_chg = _change_20d(results, "BAMLH0A0HYM2")
        if oas_chg is not None:
            derived["credit_spread_20d_change"] = oas_chg
            derived["credit_spread_20d_change_interpretation"] = _credit_spread_change_interpretation(oas_chg)

    return derived


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
            logger.warning("Failed to fetch FRED series %s: %s", series_id, _scrub(str(exc)))

    # Derived forward-rate / inflation-expectation fields (best-effort)
    try:
        results.update(_compute_derived(results, settings.FRED_API_KEY))
    except Exception as exc:
        logger.warning("Failed to compute derived FRED fields: %s", _scrub(str(exc)))

    return results


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
