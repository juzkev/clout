"""CFTC Commitments of Traders (COT) collector.

Downloads the CFTC Disaggregated Futures COT report (free, no API key).
Extracts net Managed Money positioning for Gold, Crude Oil, and Bitcoin.
Calculates positioning percentile vs available history (up to 52 weeks).

COT data is released weekly (Friday evenings). Cache is valid for 7 days.
"""

import io
import logging
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 60  # large file (~30MB)

# CFTC has reorganised its download paths over the years; try all known patterns
_COT_URL_TEMPLATES = [
    "https://www.cftc.gov/dta/cos/current/fut_disagg_txt_{year}.zip",
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
    "https://www.cftc.gov/dea/newcot/fut_disagg_txt_{year}.zip",
]
_CACHE_MAX_AGE_DAYS = 7

# Managed Money column names in the CFTC disaggregated CSV
_MM_LONG = "M_Money_Positions_Long_All"
_MM_SHORT = "M_Money_Positions_Short_All"
_DATE_COL = "As_of_Date_In_Form_YYMMDD"
_NAME_COL = "Market_and_Exchange_Names"

# Substring patterns to identify each market (case-insensitive)
MARKETS: dict[str, dict[str, Any]] = {
    "gold": {
        "pattern": "GOLD",
        "exclude": "MINI",  # exclude e-mini gold contracts
        "label": "Gold (GC)",
        "affected_tickers": ["GLD"],
    },
    "crude_oil": {
        "pattern": "CRUDE OIL, LIGHT SWEET",
        "exclude": None,
        "label": "Crude Oil (CL)",
        "affected_tickers": ["USO"],
    },
    "bitcoin": {
        "pattern": "BITCOIN",
        "exclude": None,
        "label": "Bitcoin (BTC)",
        "affected_tickers": ["IBIT"],
    },
}


def _cache_path() -> Path:
    return settings.MACRO_DIR / "cot_latest.parquet"


def _cache_valid() -> bool:
    path = _cache_path()
    if not path.exists():
        return False
    age_days = (datetime.now().timestamp() - path.stat().st_mtime) / 86400
    return age_days < _CACHE_MAX_AGE_DAYS


def _download_cot(year: int) -> pd.DataFrame | None:
    for template in _COT_URL_TEMPLATES:
        url = template.format(year=year)
        try:
            logger.info("Downloading COT data from %s", url)
            resp = requests.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                txt_files = [f for f in zf.namelist() if f.lower().endswith((".txt", ".csv"))]
                if not txt_files:
                    logger.warning("No text files found in COT zip")
                    continue
                with zf.open(txt_files[0]) as f:
                    df = pd.read_csv(f, low_memory=False)
            return df
        except Exception as exc:
            logger.debug("COT URL failed (%s): %s", url, exc)
    logger.warning("COT download failed for year %d (tried %d URL patterns)", year, len(_COT_URL_TEMPLATES))
    return None


def _load_raw() -> pd.DataFrame | None:
    """Load COT data, using cache if fresh or downloading otherwise."""
    if _cache_valid():
        try:
            return pd.read_parquet(_cache_path())
        except Exception as exc:
            logger.warning("COT cache read failed: %s", exc)

    current_year = datetime.now().year
    df = None
    for year in range(current_year, current_year - 3, -1):
        df = _download_cot(year)
        if df is not None:
            break
        logger.info("COT data unavailable for %d, trying previous year", year)

    if df is not None:
        try:
            settings.ensure_dirs()
            df.to_parquet(_cache_path())
        except Exception as exc:
            logger.warning("COT cache write failed: %s", exc)

    return df


def _find_market(df: pd.DataFrame, pattern: str, exclude: str | None) -> pd.DataFrame:
    mask = df[_NAME_COL].str.upper().str.contains(pattern.upper(), na=False)
    if exclude:
        mask &= ~df[_NAME_COL].str.upper().str.contains(exclude.upper(), na=False)
    return df[mask]


def _percentile_rank(series: pd.Series, current: float) -> float:
    """Percentile of current value within the series (0–100)."""
    if len(series) < 2:
        return 50.0
    return round(float((series < current).sum() / len(series) * 100), 1)


def _cot_interpretation(percentile: float) -> str:
    if percentile >= 80:
        return "extreme_longs_contrarian_bearish"
    if percentile >= 60:
        return "elevated_longs_mild_bearish"
    if percentile <= 20:
        return "extreme_shorts_contrarian_bullish"
    if percentile <= 40:
        return "depressed_longs_mild_bullish"
    return "positioning_neutral"


def _analyse_market(df: pd.DataFrame, market_key: str, cfg: dict) -> dict[str, Any] | None:
    market_df = _find_market(df, cfg["pattern"], cfg.get("exclude"))
    if market_df.empty:
        logger.warning("No COT rows found for %s", cfg["label"])
        return None

    required = [_DATE_COL, _MM_LONG, _MM_SHORT]
    missing = [c for c in required if c not in market_df.columns]
    if missing:
        logger.warning("COT missing columns %s for %s", missing, cfg["label"])
        return None

    market_df = market_df.copy()
    market_df[_MM_LONG] = pd.to_numeric(market_df[_MM_LONG], errors="coerce")
    market_df[_MM_SHORT] = pd.to_numeric(market_df[_MM_SHORT], errors="coerce")
    market_df["net_spec"] = market_df[_MM_LONG] - market_df[_MM_SHORT]
    market_df = market_df.dropna(subset=["net_spec"]).sort_values(_DATE_COL)

    if market_df.empty:
        return None

    # Limit to last 52 weeks
    market_df = market_df.tail(52)
    current_net = float(market_df["net_spec"].iloc[-1])
    current_long = float(market_df[_MM_LONG].iloc[-1])
    current_short = float(market_df[_MM_SHORT].iloc[-1])
    latest_date = str(market_df[_DATE_COL].iloc[-1])
    weeks_of_history = len(market_df)

    net_percentile = _percentile_rank(market_df["net_spec"], current_net)

    return {
        "label": cfg["label"],
        "latest_date": latest_date,
        "weeks_of_history": weeks_of_history,
        "mm_longs": int(current_long),
        "mm_shorts": int(current_short),
        "net_spec_position": int(current_net),
        "net_spec_percentile": net_percentile,
        "interpretation": _cot_interpretation(net_percentile),
        "affected_tickers": cfg["affected_tickers"],
    }


def collect() -> dict[str, Any]:
    """Collect CFTC COT positioning data. Returns {} on failure."""
    df = _load_raw()
    if df is None or df.empty:
        logger.warning("COT data unavailable")
        return {}

    if _NAME_COL not in df.columns:
        logger.warning("COT CSV missing expected column '%s'", _NAME_COL)
        return {}

    results: dict[str, Any] = {}
    for market_key, cfg in MARKETS.items():
        result = _analyse_market(df, market_key, cfg)
        if result:
            results[market_key] = result

    return results


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
