# Trading Research System

A daily market research pipeline that collects macro, sentiment, crypto, and price data, assembles it into a structured LLM prompt, and produces JSON swing-trade ideas for a universe of US ETFs and Bitcoin.

## Architecture

```
data/                   # Runtime outputs (parquet cache, prompts, results)
config/settings.py      # All configuration; loads from .env
research/
  collectors/           # Five independent data collectors (fully built)
    fred_collector      # FRED macro data (T10Y2Y, VIX, CPI, FEDFUNDS, …)
    sentiment_collector # CNN F&G, AAII Survey, Crypto F&G
    crypto_collector    # Binance Futures + Bybit public APIs (no key) → CoinGecko spot
    news_collector      # NewsAPI — macro/crypto/commodity headlines
    price_collector     # yfinance OHLCV + RSI/vol/SMA/momentum rank
  prompt_builder.py     # Assembles collector outputs into LLM prompt
  llm_client.py         # Pluggable: manual | deepseek | claude
  run_research.py       # Entry point — runs all collectors in parallel
signals/                # Signal dataclass + helpers (built)
strategies/             # BaseStrategy ABC + 3 stubs (momentum, mean reversion, BTC funding)
backtesting/            # Backtester, metrics, reporter (stubs)
execution/              # IBKR client, order manager, risk manager (stubs)
notifications/          # Telegram bot (stub)
```

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv + install all dependencies
uv sync

cp config/.env.example .env
# Edit .env — all keys are optional; system degrades gracefully without them
```

## Running

```bash
uv run python -m research.run_research
```

With `LLM_PROVIDER=manual` (default):
1. All collectors run in parallel
2. Formatted prompt is saved to `data/prompts/YYYY-MM-DD_prompt.txt`
3. You are asked to paste the prompt into Claude.ai and paste the JSON response back
4. Results are saved to `data/research_YYYY-MM-DD.json`

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `manual` | `manual` \| `deepseek` \| `claude` |
| `DATA_LOOKBACK_DAYS` | `60` | Days of price history to download |
| `FRED_API_KEY` | — | [fred.stlouisfed.org](https://fred.stlouisfed.org) (free) |
| `NEWSAPI_KEY` | — | [newsapi.org](https://newsapi.org) (free tier available) |
| `COINGLASS_API_KEY` | — | [coinglass.com](https://coinglass.com) (fallback to CoinGecko) |
| `DEEPSEEK_API_KEY` | — | Required only if `LLM_PROVIDER=deepseek` |
| `CLAUDE_API_KEY` | — | Required only if `LLM_PROVIDER=claude` |

## Graceful Degradation

Every component handles missing keys or network failures without crashing:
- Missing API key → logs a warning, returns empty dict/list
- Network failure per collector → logged, other collectors continue
- Full run with zero keys → generates a prompt with available data sections

## Running Tests

```bash
uv run pytest -q
```

All tests run offline with no API keys required.

## Build-Out Roadmap

1. **Strategies** — implement `generate_signals()` in `momentum.py`, `mean_reversion.py`, `btc_funding_rate.py`
2. **Backtesting** — fill in `Backtester.run()`, metrics functions, `generate_report()`
3. **Execution** — implement `IBKRClient`, `OrderManager`, `RiskManager.position_size()`
4. **Notifications** — implement `TelegramBot.send_message()` for daily research alerts
5. **Scheduler** — add a cron job or APScheduler wrapper around `run_research.run()`
