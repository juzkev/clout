import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRICE_DIR = DATA_DIR / "price"
PROMPTS_DIR = DATA_DIR / "prompts"
MACRO_DIR = DATA_DIR / "macro"
SENTIMENT_DIR = DATA_DIR / "sentiment"
CRYPTO_DIR = DATA_DIR / "crypto"
NEWS_DIR = DATA_DIR / "news"


def ensure_dirs() -> None:
    """Create all data directories if they don't exist."""
    for d in (PRICE_DIR, PROMPTS_DIR, MACRO_DIR, SENTIMENT_DIR, CRYPTO_DIR, NEWS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── Universe ─────────────────────────────────────────────────────────────────

# Instruments actually traded
UNIVERSE: list[str] = ["IBIT", "GLD", "SLV", "QQQ", "TLT", "XLE", "VIXY"]

# Signal-only instruments (used as inputs, never traded)
SIGNAL_ONLY: list[str] = ["SPY", "HYG", "BTC-USD"]

# Combined download list for price_collector (deduped, order-preserving)
PRICE_DOWNLOAD_LIST: list[str] = list(dict.fromkeys(UNIVERSE + SIGNAL_ONLY))

CRYPTO_SYMBOLS: list[str] = ["BTC", "ETH"]

# ── Per-instrument metadata ───────────────────────────────────────────────────

INSTRUMENT_META: dict[str, dict] = {
    "IBIT": {
        "name": "iShares Bitcoin Trust",
        "asset_class": "crypto",
        "rate_sensitive": True,
        "max_holding_days": 7,
        "signal_sources": ["coinglass", "crypto_fear_greed", "trends"],
        "notes": "BTC proxy. Use funding rate as primary signal.",
    },
    "GLD": {
        "name": "SPDR Gold Shares",
        "asset_class": "commodity",
        "rate_sensitive": True,
        "max_holding_days": 10,
        "signal_sources": ["cot", "fred", "trends"],
        "notes": "Core gold position. COT positioning is primary signal. "
                 "Avoid doubling up with IAU.",
    },
    "SLV": {
        "name": "iShares Silver Trust",
        "asset_class": "commodity",
        "rate_sensitive": True,
        "max_holding_days": 7,
        "signal_sources": ["cot", "trends"],
        "notes": "Higher beta gold play. 2-3x gold moves. "
                 "Only trade when GLD signal is strong.",
    },
    "QQQ": {
        "name": "Invesco QQQ Trust",
        "asset_class": "equity",
        "rate_sensitive": True,
        "max_holding_days": 10,
        "signal_sources": ["fred", "sentiment", "calendar"],
        "notes": "Primary equity instrument. Higher beta than SPY. "
                 "Earnings calendar critical — check XLK earnings.",
    },
    "TLT": {
        "name": "iShares 20+ Year Treasury Bond ETF",
        "asset_class": "bonds",
        "rate_sensitive": True,
        "max_holding_days": 10,
        "signal_sources": ["fred", "calendar"],
        "notes": "Duration risk instrument since 2022, not pure safe haven. "
                 "Treat as macro rates bet. FOMC dates are critical catalysts.",
    },
    "XLE": {
        "name": "Energy Select Sector SPDR",
        "asset_class": "equity",
        "rate_sensitive": False,
        "max_holding_days": 7,
        "signal_sources": ["cot", "news", "trends"],
        "notes": "Cleaner oil exposure than USO — no roll yield drag. "
                 "Energy equities not pure crude proxy.",
    },
    "VIXY": {
        "name": "ProShares VIX Short-Term Futures ETF",
        "asset_class": "volatility",
        "rate_sensitive": False,
        "max_holding_days": 3,
        "signal_sources": ["fred", "sentiment"],
        "notes": "CRITICAL: max 3 day hold due to VIX futures roll decay. "
                 "Only long when CNN Fear and Greed below 20 AND VIX in "
                 "backwardation. Never hold overnight through FOMC.",
    },
    "SPY": {
        "name": "SPDR S&P 500 ETF",
        "asset_class": "equity",
        "signal_only": True,
        "notes": "Regime and benchmark signal only. Not traded.",
    },
    "HYG": {
        "name": "iShares iBoxx High Yield Corporate Bond ETF",
        "asset_class": "bonds",
        "signal_only": True,
        "notes": "Credit stress indicator. HYG falling while SPY holds = "
                 "early warning. Not traded directly.",
    },
    "BTC-USD": {
        "name": "Bitcoin spot price",
        "asset_class": "crypto",
        "signal_only": True,
        "notes": "Raw BTC price signal for IBIT trades. "
                 "Not traded directly on IBKR US.",
    },
}


def get_tradeable_universe() -> list[str]:
    """Returns only instruments that can be traded."""
    return [t for t, m in INSTRUMENT_META.items() if not m.get("signal_only", False)]


def get_instrument_meta(ticker: str) -> dict:
    """Returns metadata for a ticker, empty dict if not found."""
    return INSTRUMENT_META.get(ticker, {})


def get_vixy_max_hold() -> int:
    """Special accessor — VIXY has a hard 3-day max hold."""
    return INSTRUMENT_META["VIXY"]["max_holding_days"]

# ── Research config ───────────────────────────────────────────────────────────

DATA_LOOKBACK_DAYS: int = int(os.getenv("DATA_LOOKBACK_DAYS", "60"))
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "manual")  # "manual" | "deepseek" | "claude"

# ── API keys (empty string = key missing → graceful degradation) ──────────────

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Price cache ───────────────────────────────────────────────────────────────

PRICE_CACHE_MAX_AGE_HOURS: int = 6

# ── Risk limits ────────────────────────────────────────────────────────────────

# All risk thresholds live here — never hard-code these values elsewhere.
RISK_LIMITS: dict[str, float | int | bool] = {
    # Hard stops
    "max_position_size_pct": 0.20,
    "drawdown_pause_threshold": 0.10,
    "drawdown_shutdown_threshold": 0.15,
    "max_daily_loss_pct": 0.03,
    "vixy_max_portfolio_pct": 0.05,
    # Position sizing (fraction of portfolio)
    "default_position_size_pct": 0.12,   # conviction 3-4
    "high_conviction_size_pct": 0.18,    # conviction 5
    "low_conviction_size_pct": 0.08,     # conviction 1-2
    # Correlation guard
    "gld_slv_both_long_slv_multiplier": 0.5,
    # No-trade conditions
    "min_regime_confidence_to_trade": 3,
    "no_new_trades_on_friday": True,
}


def get_position_size_pct(conviction: int) -> float:
    """Map a conviction score (1-5) to a portfolio position-size fraction."""
    rl = RISK_LIMITS
    try:
        c = int(conviction)
    except (TypeError, ValueError):
        c = 3  # unknown conviction → default sizing
    if c >= 5:
        return float(rl["high_conviction_size_pct"])
    if c >= 3:
        return float(rl["default_position_size_pct"])
    return float(rl["low_conviction_size_pct"])


# ── Logging ───────────────────────────────────────────────────────────────────

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


# Self-reference so callers can use either `from config import settings`
# (module) or `from config.settings import settings` (this alias → same module).
settings = sys.modules[__name__]
