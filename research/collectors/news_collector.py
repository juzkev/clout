"""News headline collector using free RSS feeds — no API key required.

Sources:
  Google News RSS   — macro and commodity queries (real-time, no key, no cap)
  CoinDesk RSS      — crypto news
  CoinTelegraph RSS — crypto news
"""

import logging
from typing import Any

import feedparser
import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 15
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q={query}"

SOURCES: dict[str, list[dict[str, str]]] = {
    "macro": [
        {
            "url": _GOOGLE_NEWS_RSS.format(
                query="Federal+Reserve+OR+inflation+OR+recession+OR+interest+rates"
            ),
            "name": "Google News",
        }
    ],
    "crypto": [
        {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "name": "CoinDesk"},
        {"url": "https://cointelegraph.com/rss", "name": "CoinTelegraph"},
    ],
    "commodity": [
        {
            "url": _GOOGLE_NEWS_RSS.format(
                query="gold+OR+oil+OR+commodities+OR+energy+OR+natural+gas"
            ),
            "name": "Google News",
        }
    ],
}


def _fetch_feed(url: str, source_name: str, top_n: int = 5) -> list[dict[str, str]]:
    """Fetch an RSS feed and return up to top_n items."""
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    items = []
    for entry in feed.entries[:top_n]:
        # Google News embeds the real source in entry.source.title
        source = getattr(getattr(entry, "source", None), "title", None) or source_name
        items.append({
            "title": entry.get("title", ""),
            "source": source,
            "publishedAt": entry.get("published", ""),
            "url": entry.get("link", ""),
        })
    return items


def _collect_bucket(bucket: str, top_n: int = 5) -> list[dict[str, str]]:
    """Collect headlines for one bucket, merging multiple sources if configured."""
    seen_titles: set[str] = set()
    results: list[dict[str, str]] = []

    for source_cfg in SOURCES[bucket]:
        try:
            items = _fetch_feed(source_cfg["url"], source_cfg["name"], top_n=top_n)
            for item in items:
                title_key = item["title"].lower()[:60]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    results.append(item)
                    if len(results) >= top_n:
                        return results
        except Exception as exc:
            logger.warning("RSS fetch failed for %s (%s): %s", bucket, source_cfg["name"], exc)

    return results


def collect() -> dict[str, list[dict[str, str]]]:
    """Collect news headlines from RSS feeds for all three buckets."""
    return {bucket: _collect_bucket(bucket) for bucket in SOURCES}


if __name__ == "__main__":
    import json
    from config import settings
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
