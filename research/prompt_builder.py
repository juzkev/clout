"""Prompt builder for the trading research system.

Assembles data from all collectors into a structured LLM prompt and
saves it to data/prompts/{YYYY-MM-DD}_prompt.txt.
Robust to None/empty inputs — missing sections print "data unavailable".
"""

from datetime import date
from pathlib import Path
from typing import Any

from config import settings

_SYSTEM_HEADER = """\
You are a quantitative trading analyst. Analyse the following market data \
and generate 1-3 specific trade ideas for a swing trading strategy \
(2-10 day holds) on US ETFs and Bitcoin.\
"""

_OUTPUT_FORMAT = """\
OUTPUT FORMAT (respond ONLY with valid JSON):
{
  "market_regime": "risk_on | risk_off | neutral",
  "regime_reasoning": "...",
  "trade_ideas": [
    {
      "ticker": "IBIT",
      "direction": "long | short | cash",
      "conviction": 1,
      "catalyst": "...",
      "entry": "market_open | limit_at_X",
      "stop_loss_pct": 3.0,
      "target_pct": 7.0,
      "holding_days": 5,
      "reasoning": "..."
    }
  ],
  "risks": ["...", "..."],
  "generated_at": "ISO timestamp"
}\
"""


# ── Section formatters ────────────────────────────────────────────────────────

def _fmt_fred(fred: dict) -> str:
    if not fred:
        return "  data unavailable (FRED_API_KEY not set or fetch failed)"
    lines = []
    for series_id, vals in fred.items():
        label = vals.get("label", series_id)
        latest = vals.get("latest", "N/A")
        chg = vals.get("change_20d", "N/A")
        yoy = vals.get("yoy_pct")
        line = f"  {label} ({series_id}): {latest}"
        if yoy is not None:
            line += f"  [YoY: {yoy:+.2f}%]"
        line += f"  [20d chg: {chg:+.4f}]" if isinstance(chg, float) else f"  [20d chg: {chg}]"
        lines.append(line)
    return "\n".join(lines)


def _fmt_sentiment(sentiment: dict) -> str:
    if not sentiment:
        return "  data unavailable"
    parts = []

    cnn = sentiment.get("cnn_fear_greed")
    if cnn:
        score = cnn.get("score", "N/A")
        rating = cnn.get("rating", "N/A")
        wchg = cnn.get("week_change")
        wchg_str = f" ({wchg:+.1f} vs last week)" if wchg is not None else ""
        parts.append(f"  CNN Fear & Greed: {score}/100 — {rating}{wchg_str}")
    else:
        parts.append("  CNN Fear & Greed: unavailable")

    aaii = sentiment.get("aaii")
    if aaii:
        bull = aaii.get("bullish_pct", "N/A")
        bear = aaii.get("bearish_pct", "N/A")
        neut = aaii.get("neutral_pct", "N/A")
        parts.append(f"  AAII Survey: Bullish {bull}% | Neutral {neut}% | Bearish {bear}%")
    else:
        parts.append("  AAII Survey: unavailable")

    cfg = sentiment.get("crypto_fear_greed")
    if cfg:
        val = cfg.get("value", "N/A")
        cls = cfg.get("classification", "N/A")
        parts.append(f"  Crypto Fear & Greed: {val}/100 — {cls}")
    else:
        parts.append("  Crypto Fear & Greed: unavailable")

    return "\n".join(parts)


def _fmt_crypto(crypto: dict) -> str:
    if not crypto:
        return "  data unavailable"
    lines = []
    sources = crypto.get("sources", [crypto.get("source", "unknown")])
    lines.append(f"  Sources: {', '.join(sources) if isinstance(sources, list) else sources}")

    if "btc_funding_rate" in crypto:
        rate = crypto["btc_funding_rate"]
        interp = crypto.get("funding_interpretation", "")
        binance = crypto.get("btc_funding_rate_binance")
        bybit = crypto.get("btc_funding_rate_bybit")
        detail = ""
        if binance is not None and bybit is not None:
            detail = f" (Binance: {binance:.4%}, Bybit: {bybit:.4%})"
        lines.append(f"  BTC Funding Rate (avg): {rate:.4%}{detail}  [{interp}]")
    if "btc_oi_24h_change_pct" in crypto:
        oi_chg = crypto["btc_oi_24h_change_pct"]
        interp = crypto.get("oi_interpretation", "")
        lines.append(f"  BTC Open Interest 24h Change: {oi_chg:+.2f}%  [{interp}]")
    if "btc_long_short_ratio" in crypto:
        ratio = crypto["btc_long_short_ratio"]
        interp = crypto.get("ls_interpretation", "")
        lines.append(f"  BTC Long/Short Ratio: {ratio}  [{interp}]")
    if "btc_price_usd" in crypto:
        lines.append(f"  BTC Price: ${crypto['btc_price_usd']:,.0f}")
    if "btc_24h_change_pct" in crypto:
        lines.append(f"  BTC 24h Change: {crypto['btc_24h_change_pct']:+.2f}%")
    if "btc_7d_change_pct" in crypto:
        lines.append(f"  BTC 7d Change: {crypto['btc_7d_change_pct']:+.2f}%")
    return "\n".join(lines)


def _fmt_price(price: dict) -> str:
    if not price:
        return "  data unavailable (yfinance not installed or download failed)"

    signal_readings = price.get("signal_readings", {})
    tradeable = {t: d for t, d in price.items() if t != "signal_readings"}

    def sort_key(item):
        return item[1].get("momentum_rank_20d", 999)

    sorted_tickers = sorted(tradeable.items(), key=sort_key)

    header = f"  {'Ticker':<10} {'Price':>10} {'1d%':>7} {'5d%':>7} {'20d%':>8} {'Vol20d':>8} {'RSI14':>7} {'SMA50':>7} {'Rank':>5}"
    rows = [header, "  " + "-" * 76]
    for ticker, d in sorted_tickers:
        if "error" in d:
            rows.append(f"  {ticker:<10} ERROR: {d['error']}")
            continue
        sma_flag = "▲" if d.get("above_sma50") else "▼"
        rows.append(
            f"  {ticker:<10}"
            f" {d.get('current_price', 0):>10.2f}"
            f" {d.get('return_1d_pct', 0):>+7.2f}"
            f" {d.get('return_5d_pct', 0):>+7.2f}"
            f" {d.get('return_20d_pct', 0):>+8.2f}"
            f" {d.get('vol_20d_ann_pct', 0):>7.1f}%"
            f" {d.get('rsi_14', 0):>7.1f}"
            f" {sma_flag:>7}"
            f" {d.get('momentum_rank_20d', '-'):>5}"
        )

    if signal_readings:
        rows.append("")
        rows.append("  Signal-only instruments (not traded):")
        for ticker, d in signal_readings.items():
            if "error" in d:
                rows.append(f"    {ticker:<10} ERROR: {d['error']}")
                continue
            rows.append(
                f"    {ticker:<10}"
                f" price {d.get('current_price', 0):>10.2f}"
                f"  20d {d.get('return_20d_pct', 0):>+7.2f}%"
                f"  RSI {d.get('rsi_14', 0):>5.1f}"
            )

    return "\n".join(rows)


def _fmt_universe_context() -> str:
    """Render the tradeable universe and signal-only instruments from INSTRUMENT_META."""
    lines = []
    for ticker in settings.get_tradeable_universe():
        m = settings.get_instrument_meta(ticker)
        sigs = ", ".join(m.get("signal_sources", []))
        lines.append(
            f"  {ticker} | {m.get('asset_class', '?')} | "
            f"max hold: {m.get('max_holding_days', '?')}d | "
            f"signals: {sigs} | {m.get('notes', '')}"
        )
    lines.append("")
    lines.append("  SIGNAL INSTRUMENTS (context only, do not trade):")
    for ticker in settings.SIGNAL_ONLY:
        m = settings.get_instrument_meta(ticker)
        lines.append(f"    {ticker}: {m.get('notes', '')}")
    return "\n".join(lines)


def _fmt_news(news: dict) -> str:
    if not news:
        return "  data unavailable"
    lines = []
    labels = {"macro": "Macro / Fed / Rates", "crypto": "Crypto", "commodity": "Commodities / Gold / Oil"}
    for bucket, label in labels.items():
        headlines = news.get(bucket, [])
        lines.append(f"  [{label}]")
        if not headlines:
            lines.append("    No headlines available")
        else:
            for h in headlines:
                pub = h.get("publishedAt", "")[:10]
                lines.append(f"    • {h['title']} ({h['source']}, {pub})")
    return "\n".join(lines)


def _fmt_trends(trends: dict) -> str:
    if not trends:
        return "  data unavailable"
    lines = []
    for keyword, data in trends.items():
        score = data.get("current_score", "N/A")
        chg = data.get("change_4w")
        interp = data.get("interpretation", "")
        chg_str = f" ({chg:+.0f} vs 4w ago)" if chg is not None else ""
        lines.append(f"  {keyword:<22}: {score:>3}/100{chg_str}  [{interp}]")
    return "\n".join(lines)


def _fmt_cot(cot: dict) -> str:
    if not cot:
        return "  data unavailable"
    lines = []
    for market_key, data in cot.items():
        label = data.get("label", market_key)
        net = data.get("net_spec_position", "N/A")
        pct = data.get("net_spec_percentile", "N/A")
        interp = data.get("interpretation", "")
        weeks = data.get("weeks_of_history", "?")
        tickers = ", ".join(data.get("affected_tickers", []))
        net_str = f"{net:+,}" if isinstance(net, int) else str(net)
        lines.append(
            f"  {label:<20} Net spec: {net_str:<10} "
            f"Percentile: {pct}% ({weeks}w history)  [{interp}]"
            + (f"  → {tickers}" if tickers else "")
        )
    return "\n".join(lines)


def _fmt_calendar(calendar: dict) -> str:
    if not calendar:
        return "  data unavailable"

    all_events = calendar.get("all_events", [])
    if not all_events:
        return "  No high-impact events in the next 7 days"

    lines = []
    for event in all_events:
        dt = event.get("date", "")
        time_str = event.get("time", "")
        name = event.get("event", "")
        impact = event.get("impact", "").upper()
        tickers = ", ".join(event.get("potential_affected_tickers", []))
        forecast = event.get("forecast", "")
        prev = event.get("previous", "")

        line = f"  {dt}"
        if time_str:
            line += f" {time_str}"
        line += f"  [{impact}] {name}"
        if forecast:
            line += f"  (forecast: {forecast}, prev: {prev})"
        if tickers:
            line += f"  → affects: {tickers}"
        lines.append(line)

    return "\n".join(lines)


# ── Public interface ──────────────────────────────────────────────────────────

def build_prompt(
    fred: dict,
    sentiment: dict,
    crypto: dict,
    price: dict,
    news: dict,
    trends: dict | None = None,
    cot: dict | None = None,
    calendar: dict | None = None,
) -> str:
    sections = [
        _SYSTEM_HEADER,
        "",
        "UNIVERSE CONTEXT:",
        _fmt_universe_context(),
        "",
        "MACRO ENVIRONMENT:",
        _fmt_fred(fred),
        "",
        "MARKET SENTIMENT:",
        _fmt_sentiment(sentiment),
        "",
        "CRYPTO SIGNALS:",
        _fmt_crypto(crypto),
        "",
        "PRICE MOMENTUM (Universe Ranking):",
        _fmt_price(price),
        "",
        "RECENT NEWS THEMES:",
        _fmt_news(news),
        "",
        "SEARCH TREND SIGNALS:",
        _fmt_trends(trends or {}),
        "",
        "POSITIONING EXTREMES (COT):",
        _fmt_cot(cot or {}),
        "",
        "UPCOMING CATALYSTS (next 7 days):",
        _fmt_calendar(calendar or {}),
        "",
        _OUTPUT_FORMAT,
    ]
    return "\n".join(sections)


def save_prompt(text: str) -> Path:
    settings.ensure_dirs()
    today = date.today().strftime("%Y-%m-%d")
    path = settings.PROMPTS_DIR / f"{today}_prompt.txt"
    path.write_text(text, encoding="utf-8")
    return path
