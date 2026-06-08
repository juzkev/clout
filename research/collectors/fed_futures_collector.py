"""CME FedWatch / Fed funds futures collector.

Pulls market-implied probabilities for the next FOMC decision and the amount of
easing priced over the next ~12 months.

Primary source:
  CME FedWatch implied probabilities (public CmeWS endpoint).

Fallback:
  FRED series FEDTARMD (FOMC median projection for the federal funds rate),
  compared against the current effective rate (FEDFUNDS) as a directional proxy.

This data is useful but NOT critical — every failure path returns a fully-formed
dict with null values and data_source="unavailable" rather than raising, so the
research pipeline always runs.
"""

import logging
import time
from datetime import date, timedelta
from typing import Any

import requests

from config import settings

logger = logging.getLogger(__name__)

_CME_URL = (
    "https://www.cmegroup.com/CmeWS/mvc/FedWatch/rateProbability.getCurrent.do"
)
# Loaded first to obtain Akamai bot-protection cookies before calling the API.
_CME_TOOL_PAGE = (
    "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"
)
_FRED_BASE = "https://api.stlouisfed.org/fred"
_TIMEOUT = 15

# Realistic browser headers. CME sits behind Akamai, which 403s requests that
# don't look like a browser (default python-requests UA, missing Referer, etc.).
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _CME_TOOL_PAGE,
    "Origin": "https://www.cmegroup.com",
    "sec-ch-ua": '"Chromium";v="124", "Not(A:Brand";v="24", "Google Chrome";v="124"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


# ── Interpretation ─────────────────────────────────────────────────────────────

def _interpretation(cut_p: float | None, hold_p: float | None, hike_p: float | None) -> str:
    """Classify the next-meeting stance from implied probabilities."""
    cut_p = cut_p or 0.0
    hike_p = hike_p or 0.0
    if cut_p >= 0.5 or (cut_p - hike_p) > 0.25:
        return "dovish"
    if hike_p >= 0.5 or (hike_p - cut_p) > 0.25:
        return "hawkish"
    return "neutral"


def _unavailable(note: str) -> dict[str, Any]:
    return {
        "next_meeting": {
            "date": "unknown",
            "cut_probability": None,
            "hold_probability": None,
            "hike_probability": None,
        },
        "cuts_priced_12m": None,
        "interpretation": "neutral",
        "data_source": "unavailable",
        "note": note,
    }


# ── Primary: CME FedWatch ──────────────────────────────────────────────────────

def _fetch_cme_payload() -> Any:
    """Prime Akamai cookies via the tool page, then call the FedWatch API.

    Raises on any network/HTTP error so the caller can fall back.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    # Step 1: load the tool page so Akamai sets its bot-detection cookies.
    try:
        session.get(_CME_TOOL_PAGE, timeout=_TIMEOUT)
    except Exception as exc:
        logger.debug("CME tool-page priming request failed (continuing): %s", exc)
    # Step 2: call the JSON API with a cache-busting param, carrying any cookies.
    resp = session.get(
        _CME_URL,
        params={"_": int(time.time() * 1000)},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _try_cme() -> dict[str, Any] | None:
    """Attempt the CME FedWatch endpoint. Return a populated dict or None."""
    try:
        payload = _fetch_cme_payload()
    except Exception as exc:
        logger.warning(
            "CME FedWatch fetch failed (%s) — falling back. CME is Akamai-protected "
            "and may 403 server-side requests; the FRED proxy is the reliable source.",
            exc,
        )
        return None

    try:
        # The CME payload nests meeting rows; field names have changed over time,
        # so probe defensively and bail (→ fallback) if the shape is unfamiliar.
        meetings = (
            payload.get("fomcMeetings")
            or payload.get("meetings")
            or payload.get("data")
        )
        if not isinstance(meetings, list) or not meetings:
            return None

        nxt = meetings[0]
        meeting_date = nxt.get("meetingDate") or nxt.get("date") or "unknown"

        cut_p = _coerce_prob(nxt.get("cutProbability", nxt.get("easeProbability")))
        hike_p = _coerce_prob(nxt.get("hikeProbability"))
        hold_p = _coerce_prob(nxt.get("noChangeProbability", nxt.get("holdProbability")))
        if hold_p is None and cut_p is not None and hike_p is not None:
            hold_p = round(max(0.0, 1.0 - cut_p - hike_p), 4)

        if cut_p is None and hold_p is None and hike_p is None:
            return None  # nothing usable → fall back

        cuts_12m = _coerce_prob(nxt.get("cutsPriced12m"))  # may be absent

        return {
            "next_meeting": {
                "date": meeting_date,
                "cut_probability": cut_p,
                "hold_probability": hold_p,
                "hike_probability": hike_p,
            },
            "cuts_priced_12m": cuts_12m,
            "interpretation": _interpretation(cut_p, hold_p, hike_p),
            "data_source": "cme_fedwatch",
            "note": "",
        }
    except Exception as exc:
        logger.warning("CME FedWatch parse failed: %s", exc)
        return None


def _coerce_prob(value: Any) -> float | None:
    """Coerce a probability that may arrive as a 0-1 float or a 0-100 percentage."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v > 1.0:  # percentage form
        v = v / 100.0
    return round(v, 4)


# ── Fallback: FRED FEDTARMD proxy ──────────────────────────────────────────────

def _fred_latest(series_id: str, api_key: str) -> float | None:
    url = f"{_FRED_BASE}/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 12,
    }
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    for o in resp.json().get("observations", []):
        if o.get("value", ".") != ".":
            return float(o["value"])
    return None


def _try_fred_proxy() -> dict[str, Any] | None:
    """Use FEDTARMD (median projection) vs FEDFUNDS as a directional proxy."""
    api_key = settings.FRED_API_KEY
    if not api_key:
        return None

    try:
        projected = _fred_latest("FEDTARMD", api_key)
        current = _fred_latest("FEDFUNDS", api_key)
    except Exception as exc:
        logger.warning("FRED FedWatch proxy fetch failed: %s", exc)
        return None

    if projected is None or current is None:
        return None

    # Number of 25bp moves implied between current and the median projection.
    delta = current - projected  # positive ⇒ cuts expected
    cuts_priced = round(delta / 0.25, 1)
    if delta > 0.125:
        interpretation = "dovish"
    elif delta < -0.125:
        interpretation = "hawkish"
    else:
        interpretation = "neutral"

    return {
        "next_meeting": {
            "date": "unknown",
            "cut_probability": None,
            "hold_probability": None,
            "hike_probability": None,
        },
        "cuts_priced_12m": cuts_priced,
        "interpretation": interpretation,
        "data_source": "fred_proxy",
        "note": (
            f"Proxy from FEDTARMD median projection {projected} vs current "
            f"effective rate {current}. CME FedWatch unavailable."
        ),
    }


# ── Public interface ───────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """Collect Fed funds futures pricing. Never raises — degrades to 'unavailable'."""
    result = _try_cme()
    if result:
        logger.info("Fed futures: CME FedWatch data collected")
        return result

    result = _try_fred_proxy()
    if result:
        logger.info("Fed futures: using FRED FEDTARMD proxy")
        return result

    logger.warning("Fed futures data unavailable (CME + FRED proxy both failed)")
    return _unavailable(
        "CME FedWatch endpoint unreachable and no FRED proxy available "
        "(missing FRED_API_KEY or FEDTARMD/FEDFUNDS data)."
    )


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
