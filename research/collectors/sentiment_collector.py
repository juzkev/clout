"""Market sentiment data collector.

Sources:
  1. CNN Fear & Greed Index   — public JSON endpoint
  2. AAII Sentiment Survey    — scraped HTML (best-effort)
  3. Crypto Fear & Greed      — alternative.me free API

Each source is wrapped independently; a failure returns None for that
source without affecting the others.
"""

import logging
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

def _collect_aaii_sentiment() -> dict[str, Any] | None:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed — skipping AAII scrape")
        return None

    url = "https://www.aaii.com/sentimentsurvey/sent_results"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # AAII page structure: look for a table with bullish/neutral/bearish rows
        result: dict[str, Any] = {}
        for row in soup.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if not cells:
                continue
            label = cells[0].lower()
            if "bullish" in label and len(cells) >= 2:
                result["bullish_pct"] = _parse_pct(cells[1])
            elif "neutral" in label and len(cells) >= 2:
                result["neutral_pct"] = _parse_pct(cells[1])
            elif "bearish" in label and len(cells) >= 2:
                result["bearish_pct"] = _parse_pct(cells[1])

        if not result:
            logger.warning("AAII scrape: could not find sentiment rows (page layout may have changed)")
            return None

        return result
    except Exception as exc:
        logger.warning("AAII sentiment scrape failed: %s", exc)
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
