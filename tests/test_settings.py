"""Smoke tests for config/settings.py."""

import config.settings as settings


def test_universe_non_empty():
    assert len(settings.UNIVERSE) > 0
    assert "IBIT" in settings.UNIVERSE
    # SPY is a signal-only instrument now, not part of the tradeable universe
    assert "SPY" in settings.SIGNAL_ONLY
    assert "SPY" not in settings.UNIVERSE


def test_crypto_symbols():
    assert "BTC" in settings.CRYPTO_SYMBOLS


def test_llm_provider_default():
    # The default when env var is absent; clear_api_keys fixture doesn't set LLM_PROVIDER
    assert settings.LLM_PROVIDER in ("manual", "deepseek", "claude")


def test_data_lookback_positive():
    assert settings.DATA_LOOKBACK_DAYS > 0


def test_paths_resolve(tmp_data_dirs):
    assert settings.DATA_DIR == tmp_data_dirs
    assert settings.PRICE_DIR == tmp_data_dirs / "price"
    assert settings.PROMPTS_DIR == tmp_data_dirs / "prompts"


def test_ensure_dirs_creates_paths(tmp_path, monkeypatch):
    import config.settings as s
    for attr in ("PRICE_DIR", "PROMPTS_DIR", "MACRO_DIR", "SENTIMENT_DIR", "CRYPTO_DIR", "NEWS_DIR"):
        monkeypatch.setattr(s, attr, tmp_path / attr.lower())
    s.ensure_dirs()
    for attr in ("PRICE_DIR", "PROMPTS_DIR", "MACRO_DIR", "SENTIMENT_DIR", "CRYPTO_DIR", "NEWS_DIR"):
        assert getattr(s, attr).exists()
