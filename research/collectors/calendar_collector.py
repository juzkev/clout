"""Upcoming market events collector (next 7 days).

Sources:
  Economic events: Forex Factory JSON (free, no key, real-time)
  Earnings:        yfinance .calendar for major S&P 500 names
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 15
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Forex Factory unofficial JSON endpoints
_FF_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]

# High-impact event title keywords and the universe tickers they affect
_EVENT_TICKERS: dict[str, list[str]] = {
    "fomc": ["SPY", "QQQ", "TLT", "IBIT", "GLD"],
    "federal funds rate": ["SPY", "QQQ", "TLT", "IBIT", "GLD"],
    "interest rate": ["SPY", "QQQ", "TLT", "GLD"],
    "cpi": ["TLT", "GLD", "HYG", "SPY"],
    "consumer price": ["TLT", "GLD", "HYG", "SPY"],
    "pce": ["TLT", "GLD", "SPY", "QQQ"],
    "personal consumption": ["TLT", "GLD", "SPY"],
    "non-farm": ["SPY", "QQQ", "TLT"],
    "nonfarm": ["SPY", "QQQ", "TLT"],
    "unemployment": ["SPY", "QQQ", "TLT"],
    "jobless": ["SPY", "QQQ"],
    "gdp": ["SPY", "QQQ", "TLT"],
    "ppi": ["TLT", "GLD", "SPY"],
    "producer price": ["TLT", "GLD", "SPY"],
    "retail sales": ["SPY", "QQQ"],
    "ism": ["SPY", "QQQ"],
}

# Major S&P 500 components to watch for earnings
_EARNINGS_WATCHLIST = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA",
    "JPM", "V", "MA", "XOM", "UNH", "JNJ", "AVGO", "HD",
    "BAC", "WMT", "PG", "LLY", "MRK",
]


def _affected_tickers(event_title: str) -> list[str]:
    title_lower = event_title.lower()
    for keyword, tickers in _EVENT_TICKERS.items():
        if keyword in title_lower:
            return tickers
    return []


def _parse_ff_date(date_str: str) -> datetime | None:
    """Parse Forex Factory date string (e.g. 'Jan 15 2025')."""
    for fmt in ("%b %d %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    try:
        from dateutil import parser as dateutil_parser
        return dateutil_parser.parse(date_str)
    except Exception:
        return None


def _collect_economic_events(days_ahead: int = 7) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now + timedelta(days=days_ahead)
    events: list[dict[str, Any]] = []
    seen: set[str] = set()

    for url in _FF_URLS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            items = resp.json()
        except Exception as exc:
            logger.warning("Forex Factory fetch failed (%s): %s", url, exc)
            continue

        for item in items:
            if item.get("country", "").upper() != "USD":
                continue
            if item.get("impact", "").lower() != "high":
                continue

            event_date = _parse_ff_date(item.get("date", ""))
            if event_date is None:
                continue
            if not (now.date() <= event_date.date() <= cutoff.date()):
                continue

            title = item.get("title", "")
            dedup_key = f"{event_date.date()}:{title}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            events.append({
                "date": event_date.strftime("%Y-%m-%d"),
                "time": item.get("time", ""),
                "event": title,
                "impact": "high",
                "forecast": item.get("forecast", ""),
                "previous": item.get("previous", ""),
                "potential_affected_tickers": _affected_tickers(title),
            })

    return sorted(events, key=lambda e: e["date"])


def _collect_earnings(days_ahead: int = 7) -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping earnings calendar")
        return []

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now + timedelta(days=days_ahead)
    earnings: list[dict[str, Any]] = []

    for ticker in _EARNINGS_WATCHLIST:
        try:
            cal = yf.Ticker(ticker).calendar
            if not cal:
                continue

            # yfinance returns either a dict or DataFrame depending on version
            if hasattr(cal, "to_dict"):
                cal = cal.to_dict()

            dates = cal.get("Earnings Date") or cal.get("earningsDate", [])
            if not isinstance(dates, list):
                dates = [dates]

            for d in dates:
                if d is None:
                    continue
                # normalise to naive datetime
                try:
                    import pandas as pd
                    dt = pd.Timestamp(d).to_pydatetime().replace(tzinfo=None)
                except Exception:
                    continue
                if now.date() <= dt.date() <= cutoff.date():
                    earnings.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "event": f"{ticker} Earnings",
                        "impact": "medium",
                        "potential_affected_tickers": [ticker],
                    })
                    break  # one entry per company
        except Exception as exc:
            logger.debug("Earnings fetch failed for %s: %s", ticker, exc)

    return sorted(earnings, key=lambda e: e["date"])


def collect() -> dict[str, Any]:
    """Collect upcoming economic events and earnings for the next 7 days."""
    economic = _collect_economic_events()
    earnings = _collect_earnings()
    all_events = sorted(economic + earnings, key=lambda e: e["date"])
    return {
        "economic": economic,
        "earnings": earnings,
        "all_events": all_events,
        "days_ahead": 7,
    }


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
