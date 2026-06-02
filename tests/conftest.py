"""Test fixtures: zero out API keys and redirect data paths to tmp."""

import os
import pytest


@pytest.fixture(autouse=True)
def clear_api_keys(monkeypatch):
    """Ensure no real API keys leak into tests."""
    for key in ("FRED_API_KEY", "NEWSAPI_KEY", "COINGLASS_API_KEY",
                 "DEEPSEEK_API_KEY", "CLAUDE_API_KEY",
                 "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.setenv(key, "")


@pytest.fixture
def tmp_data_dirs(tmp_path, monkeypatch):
    """Redirect all data paths to a temporary directory."""
    import config.settings as s

    price_dir = tmp_path / "price"
    prompts_dir = tmp_path / "prompts"
    price_dir.mkdir()
    prompts_dir.mkdir()

    monkeypatch.setattr(s, "DATA_DIR", tmp_path)
    monkeypatch.setattr(s, "PRICE_DIR", price_dir)
    monkeypatch.setattr(s, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(s, "MACRO_DIR", tmp_path / "macro")
    monkeypatch.setattr(s, "SENTIMENT_DIR", tmp_path / "sentiment")
    monkeypatch.setattr(s, "CRYPTO_DIR", tmp_path / "crypto")
    monkeypatch.setattr(s, "NEWS_DIR", tmp_path / "news")
    return tmp_path
