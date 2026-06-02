"""News headline collector using NewsAPI.

Queries three topic buckets over the last 24 hours, top 5 English results each.
Returns empty lists with a warning when NEWSAPI_KEY is missing.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from config import settings

logger = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2/everything"

QUERIES: dict[str, str] = {
    "macro": "Federal Reserve OR inflation OR recession OR interest rates",
    "crypto": "Bitcoin OR cryptocurrency OR crypto",
    "commodity": "gold OR commodities OR oil OR energy",
}


def _fetch_headlines(query: str, api_key: str, top_n: int = 5) -> list[dict[str, str]]:
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q": query,
        "from": since,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": top_n,
        "apiKey": api_key,
    }
    resp = requests.get(NEWSAPI_BASE, params=params, timeout=15)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", ""),
            "publishedAt": a.get("publishedAt", ""),
            "url": a.get("url", ""),
        }
        for a in articles[:top_n]
    ]


def collect() -> dict[str, list[dict[str, str]]]:
    """Collect news headlines. Returns empty lists with warning if no API key."""
    if not settings.NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set — skipping news collection")
        return {key: [] for key in QUERIES}

    results: dict[str, list[dict[str, str]]] = {}
    for bucket, query in QUERIES.items():
        try:
            results[bucket] = _fetch_headlines(query, settings.NEWSAPI_KEY)
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for bucket '%s': %s", bucket, exc)
            results[bucket] = []

    return results


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
