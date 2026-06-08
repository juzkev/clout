"""Tests for research/llm_client.py — covers the _route_llm provider dispatch."""

import pytest


def test_unknown_provider_raises():
    from research.llm_client import _route_llm
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        _route_llm(
            provider="unknown_provider",
            system_prompt="sys",
            user_prompt="user",
            pass_name="test",
            output_path="/tmp/test_prompt.txt",
        )
