"""Market sentiment data collector.

Sources:
  1. CNN Fear & Greed Index   — public JSON endpoint
  2. AAII Sentiment Survey    — scraped HTML (best-effort)
  3. Crypto Fear & Greed      — alternative.me free API

Each source is wrapped independently; a failure returns None for that
source without affecting the others.
"""

import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ── CNN Fear & Greed ──────────────────────────────────────────────────────────

def _collect_cnn_fear_greed() -> dict[str, Any] | None:
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        fg = data.get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating")

        # 1-week change: compare current score to score ~7 data points back
        history = data.get("fear_and_greed_historical", {}).get("data", [])
        week_change = None
        if history and len(history) >= 7 and score is not None:
            prior_score = history[-7].get("y")
            if prior_score is not None:
                week_change = round(score - prior_score, 1)

        return {
            "score": round(score, 1) if score is not None else None,
            "rating": rating,
            "week_change": week_change,
        }
    except Exception as exc:
        logger.warning("CNN Fear & Greed fetch failed: %s", exc)
        return None


# ── AAII Sentiment Survey ─────────────────────────────────────────────────────

# AAII's detailed results page is member-gated and JS-rendered; the public landing
# page often carries the latest headline readings. Try both, best-effort.
_AAII_URLS = [
    "https://www.aaii.com/sentimentsurvey",
    "https://www.aaii.com/sentimentsurvey/sent_results",
]


def _parse_aaii_table(html: str) -> dict[str, Any]:
    """Strategy 1: walk HTML table rows for bullish/neutral/bearish labels."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {}
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, Any] = {}
    for row in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        label = cells[0].lower()
        value = _parse_pct(cells[1])
        if value is None:
            continue
        if "bullish" in label:
            result.setdefault("bullish_pct", value)
        elif "neutral" in label:
            result.setdefault("neutral_pct", value)
        elif "bearish" in label:
            result.setdefault("bearish_pct", value)
    return result


def _parse_aaii_regex(html: str) -> dict[str, Any]:
    """Strategy 2: layout-agnostic regex for 'Bullish ... NN.N%' across the page text."""
    result: dict[str, Any] = {}
    for label, key in (("bullish", "bullish_pct"), ("neutral", "neutral_pct"), ("bearish", "bearish_pct")):
        # label followed (within a short window, ignoring tags/whitespace) by a percentage
        m = re.search(rf"{label}[^0-9%]{{0,40}}?([0-9]{{1,3}}(?:\.[0-9]+)?)\s*%", html, re.IGNORECASE)
        if m:
            result[key] = _parse_pct(m.group(1))
    return result


def _collect_aaii_sentiment() -> dict[str, Any] | None:
    last_exc: Exception | None = None
    for url in _AAII_URLS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
            html = resp.text

            result = _parse_aaii_table(html)
            if not all(k in result for k in ("bullish_pct", "neutral_pct", "bearish_pct")):
                # fill any gaps with the regex strategy
                for k, v in _parse_aaii_regex(html).items():
                    result.setdefault(k, v)

            # require all three to consider it a valid reading
            if all(result.get(k) is not None for k in ("bullish_pct", "neutral_pct", "bearish_pct")):
                return {k: result[k] for k in ("bullish_pct", "neutral_pct", "bearish_pct")}
        except Exception as exc:
            last_exc = exc
            logger.debug("AAII fetch failed (%s): %s", url, exc)

    if last_exc is not None:
        logger.info("AAII sentiment unavailable: %s", last_exc)
    else:
        logger.info(
            "AAII sentiment unavailable — results page is member-gated/JS-rendered; "
            "skipping (best-effort source)"
        )
    return None


def _parse_pct(text: str) -> float | None:
    try:
        return float(text.replace("%", "").strip())
    except ValueError:
        return None


# ── Crypto Fear & Greed ───────────────────────────────────────────────────────

def _collect_crypto_fear_greed() -> dict[str, Any] | None:
    url = "https://api.alternative.me/fng/"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [{}])[0]
        return {
            "value": int(data.get("value", 0)),
            "classification": data.get("value_classification", ""),
        }
    except Exception as exc:
        logger.warning("Crypto Fear & Greed fetch failed: %s", exc)
        return None


# ── Public interface ──────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """Collect sentiment data from all three sources."""
    return {
        "cnn_fear_greed": _collect_cnn_fear_greed(),
        "aaii": _collect_aaii_sentiment(),
        "crypto_fear_greed": _collect_crypto_fear_greed(),
    }


if __name__ == "__main__":
    import json
    from config import settings
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
