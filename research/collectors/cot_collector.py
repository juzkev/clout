"""CFTC Commitments of Traders (COT) collector.

Downloads two free CFTC COT reports (no API key):
  - Disaggregated Futures (fut_disagg) — physical commodities: Gold, Crude Oil.
    Speculator category = Managed Money.
  - Traders in Financial Futures (fut_fin) — financial futures: Bitcoin.
    Speculator category = Leveraged Funds (Bitcoin is not in the disaggregated report).

Extracts net speculative positioning per market and computes its percentile
vs available history (up to 52 weeks).

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

_CACHE_MAX_AGE_DAYS = 7
_DATE_COL = "As_of_Date_In_Form_YYMMDD"
_NAME_COL = "Market_and_Exchange_Names"

# Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
# Each report has a different file stem and speculator column set.
# Primary path is /files/dea/history/; others kept as fallback in case CFTC reorganises again.
_REPORTS: dict[str, dict[str, Any]] = {
    "disaggregated": {
        "stem": "fut_disagg_txt",
        "long_col": "M_Money_Positions_Long_All",   # Managed Money
        "short_col": "M_Money_Positions_Short_All",
        "cache": "cot_disagg_latest.parquet",
    },
    "financial": {
        "stem": "fut_fin_txt",
        "long_col": "Lev_Money_Positions_Long_All",  # Leveraged Funds
        "short_col": "Lev_Money_Positions_Short_All",
        "cache": "cot_fin_latest.parquet",
    },
}
_URL_TEMPLATES = [
    "https://www.cftc.gov/files/dea/history/{stem}_{year}.zip",
    "https://www.cftc.gov/dea/newcot/{stem}_{year}.zip",
    "https://www.cftc.gov/dta/cos/current/{stem}_{year}.zip",
]

# Substring patterns to identify each market (case-insensitive), and which report holds it
MARKETS: dict[str, dict[str, Any]] = {
    "gold": {
        "report": "disaggregated",
        "pattern": "GOLD",
        "exclude": "MINI",  # exclude e-mini gold contracts
        "label": "Gold (GC)",
        "affected_tickers": ["GLD"],
    },
    "crude_oil": {
        "report": "disaggregated",
        "pattern": "CRUDE OIL, LIGHT SWEET",
        "exclude": None,
        "label": "Crude Oil (CL)",
        "affected_tickers": ["USO"],
    },
    "bitcoin": {
        "report": "financial",
        "pattern": "BITCOIN",
        "exclude": "MICRO",  # exclude Micro Bitcoin contracts
        "label": "Bitcoin (BTC)",
        "affected_tickers": ["IBIT"],
    },
}


def _cache_path(cache_name: str) -> Path:
    return settings.MACRO_DIR / cache_name


def _cache_valid(cache_name: str) -> bool:
    path = _cache_path(cache_name)
    if not path.exists():
        return False
    age_days = (datetime.now().timestamp() - path.stat().st_mtime) / 86400
    return age_days < _CACHE_MAX_AGE_DAYS


def _download_report(stem: str, year: int) -> pd.DataFrame | None:
    for template in _URL_TEMPLATES:
        url = template.format(stem=stem, year=year)
        try:
            logger.info("Downloading COT data from %s", url)
            resp = requests.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                txt_files = [f for f in zf.namelist() if f.lower().endswith((".txt", ".csv"))]
                if not txt_files:
                    logger.warning("No text files found in COT zip (%s)", url)
                    continue
                with zf.open(txt_files[0]) as f:
                    df = pd.read_csv(f, low_memory=False)
            return df
        except Exception as exc:
            logger.debug("COT URL failed (%s): %s", url, exc)
    return None


def _load_report(report_cfg: dict) -> pd.DataFrame | None:
    """Load one COT report, using cache if fresh or downloading otherwise."""
    cache_name = report_cfg["cache"]
    if _cache_valid(cache_name):
        try:
            return pd.read_parquet(_cache_path(cache_name))
        except Exception as exc:
            logger.warning("COT cache read failed: %s", exc)

    stem = report_cfg["stem"]
    current_year = datetime.now().year
    df = None
    for year in range(current_year, current_year - 3, -1):
        df = _download_report(stem, year)
        if df is not None:
            break
        logger.info("COT %s data unavailable for %d, trying previous year", stem, year)

    if df is None:
        logger.warning("COT download failed for %s (tried 3 years x %d URL patterns)",
                       stem, len(_URL_TEMPLATES))
        return None

    try:
        settings.ensure_dirs()
        df.to_parquet(_cache_path(cache_name))
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


def _analyse_market(
    df: pd.DataFrame, cfg: dict, long_col: str, short_col: str
) -> dict[str, Any] | None:
    market_df = _find_market(df, cfg["pattern"], cfg.get("exclude"))
    if market_df.empty:
        logger.warning("No COT rows found for %s", cfg["label"])
        return None

    required = [_DATE_COL, long_col, short_col]
    missing = [c for c in required if c not in market_df.columns]
    if missing:
        logger.warning("COT missing columns %s for %s", missing, cfg["label"])
        return None

    market_df = market_df.copy()
    market_df[long_col] = pd.to_numeric(market_df[long_col], errors="coerce")
    market_df[short_col] = pd.to_numeric(market_df[short_col], errors="coerce")
    market_df["net_spec"] = market_df[long_col] - market_df[short_col]
    market_df = market_df.dropna(subset=["net_spec"]).sort_values(_DATE_COL)

    if market_df.empty:
        return None

    # Limit to last 52 weeks
    market_df = market_df.tail(52)
    current_net = float(market_df["net_spec"].iloc[-1])
    current_long = float(market_df[long_col].iloc[-1])
    current_short = float(market_df[short_col].iloc[-1])
    latest_date = str(market_df[_DATE_COL].iloc[-1])
    weeks_of_history = len(market_df)

    net_percentile = _percentile_rank(market_df["net_spec"], current_net)

    return {
        "label": cfg["label"],
        "latest_date": latest_date,
        "weeks_of_history": weeks_of_history,
        "spec_longs": int(current_long),
        "spec_shorts": int(current_short),
        "net_spec_position": int(current_net),
        "net_spec_percentile": net_percentile,
        "interpretation": _cot_interpretation(net_percentile),
        "affected_tickers": cfg["affected_tickers"],
    }


def collect() -> dict[str, Any]:
    """Collect CFTC COT positioning data across reports. Returns {} on total failure."""
    # Only load reports that at least one market actually needs
    needed_reports = {cfg["report"] for cfg in MARKETS.values()}
    loaded: dict[str, pd.DataFrame] = {}
    for report_key in needed_reports:
        df = _load_report(_REPORTS[report_key])
        if df is None or df.empty:
            logger.warning("COT %s report unavailable", report_key)
            continue
        if _NAME_COL not in df.columns:
            logger.warning("COT %s report missing column '%s'", report_key, _NAME_COL)
            continue
        loaded[report_key] = df

    if not loaded:
        logger.warning("COT data unavailable")
        return {}

    results: dict[str, Any] = {}
    for market_key, cfg in MARKETS.items():
        report_key = cfg["report"]
        df = loaded.get(report_key)
        if df is None:
            continue
        report_cfg = _REPORTS[report_key]
        result = _analyse_market(df, cfg, report_cfg["long_col"], report_cfg["short_col"])
        if result:
            results[market_key] = result

    return results


if __name__ == "__main__":
    import json
    settings.configure_logging()
    print(json.dumps(collect(), indent=2))
