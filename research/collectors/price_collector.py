"""Price and technical indicator collector.

Downloads 60-day OHLCV via yfinance for settings.PRICE_DOWNLOAD_LIST
(tradeable universe + signal-only instruments).
Raw OHLCV is cached to data/price/ as parquet; cache is reused if <6 hours old.

Per-ticker output:
  current_price, return_1d, return_5d, return_20d,
  vol_20d_ann, rsi_14, above_sma50, momentum_rank_20d, is_signal_only

Tradeable instruments are top-level keys (ranked); signal-only instruments are
grouped under a "signal_readings" key and are not ranked.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed — price collection unavailable")

try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except ImportError:
    _PARQUET_AVAILABLE = False


ALL_TICKERS = settings.PRICE_DOWNLOAD_LIST


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("-", "_")
    return settings.PRICE_DIR / f"{safe}_{date.today()}.parquet"


def _cache_fresh(path: Path) -> bool:
    if not path.exists() or not _PARQUET_AVAILABLE:
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < settings.PRICE_CACHE_MAX_AGE_HOURS


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if _cache_fresh(path):
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Failed to read cache for %s: %s", ticker, exc)
    return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    if not _PARQUET_AVAILABLE:
        return
    try:
        settings.ensure_dirs()
        path = _cache_path(ticker)
        df.to_parquet(path)
    except Exception as exc:
        logger.warning("Failed to save cache for %s: %s", ticker, exc)


# ── Download ──────────────────────────────────────────────────────────────────

def _download(ticker: str) -> pd.DataFrame | None:
    cached = _load_cache(ticker)
    if cached is not None:
        logger.debug("Using cached data for %s", ticker)
        return cached

    end = datetime.now()
    start = end - timedelta(days=settings.DATA_LOOKBACK_DAYS + 30)  # extra buffer for SMA/RSI
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            logger.warning("yfinance returned empty data for %s", ticker)
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        _save_cache(ticker, df)
        return df
    except Exception as exc:
        logger.warning("yfinance download failed for %s: %s", ticker, exc)
        return None


# ── Technical indicators ──────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI."""
    delta = close.diff().dropna()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # When avg_loss == 0 all gains → RSI = 100; when avg_gain == 0 → RSI = 0
    rsi = pd.Series(index=avg_gain.index, dtype=float)
    zero_loss = avg_loss == 0
    rsi[zero_loss] = 100.0
    rsi[~zero_loss] = 100 - (100 / (1 + avg_gain[~zero_loss] / avg_loss[~zero_loss]))
    return round(float(rsi.iloc[-1]), 2)


def _analyse(ticker: str, df: pd.DataFrame) -> dict[str, Any]:
    close = df["Close"].dropna()
    if len(close) < 22:
        return {"error": "insufficient data"}

    daily_ret = close.pct_change().dropna()

    current_price = round(float(close.iloc[-1]), 4)
    ret_1d = round(float(daily_ret.iloc[-1]) * 100, 2) if len(daily_ret) >= 1 else None
    ret_5d = round(float((close.iloc[-1] / close.iloc[-6] - 1) * 100), 2) if len(close) >= 6 else None
    ret_20d = round(float((close.iloc[-1] / close.iloc[-21] - 1) * 100), 2) if len(close) >= 21 else None
    vol_20d = round(float(daily_ret.iloc[-20:].std() * np.sqrt(252) * 100), 2) if len(daily_ret) >= 20 else None

    rsi_val = _rsi(close)

    sma50 = float(close.iloc[-50:].mean()) if len(close) >= 50 else float(close.mean())
    above_sma50 = bool(current_price > sma50)

    out = {
        "ticker": ticker,
        "current_price": current_price,
        "return_1d_pct": ret_1d,
        "return_5d_pct": ret_5d,
        "return_20d_pct": ret_20d,
        "vol_20d_ann_pct": vol_20d,
        "rsi_14": rsi_val,
        "above_sma50": above_sma50,
        "sma50": round(sma50, 4),
        "is_signal_only": ticker in settings.SIGNAL_ONLY,
    }

    # VIXY-specific handling: roll-decay hold warning + a backwardation proxy.
    # True VIX term structure (spot vs 1-month future) isn't available here, so we
    # approximate: a sharp recent rise in VIXY typically coincides with VIX-futures
    # backwardation (acute stress).
    if ticker == "VIXY":
        out["vixy_hold_warning"] = "Max 3 day hold. Roll decay accelerates after day 3."
        out["vix_backwardation_proxy"] = bool(ret_5d is not None and ret_5d > 5.0)

    return out


def _add_momentum_rank(results: dict[str, dict]) -> None:
    """Add momentum_rank_20d (1=best) ranked by 20d return within universe."""
    ranked = [
        (t, d.get("return_20d_pct"))
        for t, d in results.items()
        if d.get("return_20d_pct") is not None
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)
    for rank, (ticker, _) in enumerate(ranked, start=1):
        results[ticker]["momentum_rank_20d"] = rank


# ── Public interface ──────────────────────────────────────────────────────────

def collect() -> dict[str, Any]:
    """Download and analyse price data for the full download list.

    Tradeable instruments are returned as top-level keys (with momentum ranks).
    Signal-only instruments (SPY, HYG, BTC-USD) are grouped under a separate
    "signal_readings" key and are NOT ranked.
    """
    if not _YF_AVAILABLE:
        logger.warning("yfinance not installed — returning empty price data")
        return {}

    settings.ensure_dirs()
    tradeable: dict[str, Any] = {}
    signal_readings: dict[str, Any] = {}

    for ticker in ALL_TICKERS:
        df = _download(ticker)
        if df is not None:
            analysis = _analyse(ticker, df)
        else:
            analysis = {
                "ticker": ticker,
                "error": "download_failed",
                "is_signal_only": ticker in settings.SIGNAL_ONLY,
            }
        if ticker in settings.SIGNAL_ONLY:
            signal_readings[ticker] = analysis
        else:
            tradeable[ticker] = analysis

    # Rank momentum within the tradeable universe only
    _add_momentum_rank(tradeable)

    results: dict[str, Any] = dict(tradeable)
    results["signal_readings"] = signal_readings
    return results


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
