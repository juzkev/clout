import logging
import os
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

UNIVERSE: list[str] = ["IBIT", "GLD", "SPY", "QQQ", "TLT", "USO", "HYG"]
CRYPTO_SYMBOLS: list[str] = ["BTC", "ETH"]

# ── Research config ───────────────────────────────────────────────────────────

DATA_LOOKBACK_DAYS: int = int(os.getenv("DATA_LOOKBACK_DAYS", "60"))
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "manual")  # "manual" | "deepseek" | "claude"

# ── API keys (empty string = key missing → graceful degradation) ──────────────

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Price cache ───────────────────────────────────────────────────────────────

PRICE_CACHE_MAX_AGE_HOURS: int = 6

# ── Logging ───────────────────────────────────────────────────────────────────

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
