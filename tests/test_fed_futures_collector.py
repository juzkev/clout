"""Tests for forward-rate pricing: fed_futures_collector, fred derived fields,
and the rate_regime addition to Pass 1 output."""

import json
from datetime import date, timedelta

import pytest
import requests

from research.collectors import fed_futures_collector, fred_collector
from research import llm_client


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mk_obs(latest: float, older: float, n: int = 210, switch: int = 15) -> list[dict]:
    """Build FRED-style descending observations.

    Index 0 is the most recent (value=`latest`); from `switch` onward the value
    is `older`. `switch` is < 20 so the ~20-day lookback lands on `older`,
    giving a non-zero change_20d.
    """
    base = date(2024, 6, 3)
    obs = []
    for i in range(n):
        d = (base - timedelta(days=i)).isoformat()
        value = latest if i < switch else older
        obs.append({"date": d, "value": str(value)})
    return obs


def _mk_obs_yoy(latest: float, year_ago: float, n: int = 400) -> list[dict]:
    """Build observations long enough (>365d) for a YoY computation."""
    base = date(2024, 6, 3)
    obs = []
    for i in range(n):
        d = (base - timedelta(days=i)).isoformat()
        obs.append({"date": d, "value": str(latest if i < 365 else year_ago)})
    return obs


_SAMPLE_FRED = {
    "DGS10": _mk_obs(4.50, 4.00),          # latest above its ~4.0 SMA → "above"
    "T10YIE": _mk_obs(2.30, 2.30),         # flat breakeven
    "T5YIFR": _mk_obs(2.40, 2.40),
    "T10Y2Y": _mk_obs(0.50, 0.30),         # +20bps over 20d → steepening
    "BAMLH0A0HYM2": _mk_obs(3.20, 3.10),   # HY OAS
    "FEDFUNDS": _mk_obs(5.25, 5.25),
    "DFII10": _mk_obs(2.10, 2.05),
    "DFII5": _mk_obs(2.00, 1.95),
    "CPIAUCSL": _mk_obs(310.0, 305.0),
    "UNRATE": _mk_obs(4.3, 4.2),
    "VIXCLS": _mk_obs(16.0, 18.0),
    # Tier 2 labor
    "PAYEMS": _mk_obs(159180, 159000),     # +180k MoM → solid job growth
    "JTSJOL": _mk_obs(8000, 8100),         # openings
    "UNEMPLOY": _mk_obs(6500, 6400),       # → openings/unemployed ≈ 1.23 (tight)
    "ICSA": _mk_obs(220000, 215000),       # low claims, tight labor
    "AHETPI": _mk_obs_yoy(30.00, 28.85),   # wage growth ≈ 3.99% YoY
}


# ── 1. fed_futures_collector graceful failure ─────────────────────────────────

def test_returns_dict_on_failure(monkeypatch):
    """A ConnectionError on every request → unavailable dict, no exception."""
    def _boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network down")

    # Cover both the session-based CME fetch and the module-level FRED proxy call.
    monkeypatch.setattr(fed_futures_collector.requests.Session, "get", _boom)
    monkeypatch.setattr(fed_futures_collector.requests, "get", _boom)
    monkeypatch.setattr(fed_futures_collector.settings, "FRED_API_KEY", "")

    result = fed_futures_collector.collect()

    assert isinstance(result, dict)
    assert result["data_source"] == "unavailable"
    assert result["next_meeting"]["cut_probability"] is None
    assert result["cuts_priced_12m"] is None
    assert result["interpretation"] in ("dovish", "neutral", "hawkish")
    assert result["note"]


def test_cme_primes_cookies_then_parses(monkeypatch):
    """_try_cme loads the tool page first (Akamai cookies), then parses the API."""
    calls: list[str] = []

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "fomcMeetings": [
                    {
                        "meetingDate": "2026-06-17",
                        "cutProbability": 65.0,   # percentage form
                        "hikeProbability": 0.0,
                        "noChangeProbability": 35.0,
                    }
                ]
            }

    class _FakeSession:
        headers: dict = {}

        def get(self, url, **kwargs):
            calls.append(url)
            return _FakeResp()

    monkeypatch.setattr(fed_futures_collector.requests, "Session", _FakeSession)

    result = fed_futures_collector.collect()
    # Tool page primed before the API call
    assert calls[0] == fed_futures_collector._CME_TOOL_PAGE
    assert any("rateProbability" in c for c in calls)
    assert result["data_source"] == "cme_fedwatch"
    assert result["next_meeting"]["cut_probability"] == pytest.approx(0.65)  # 65% → 0.65
    assert result["interpretation"] == "dovish"


def test_fred_proxy_used_when_cme_fails(monkeypatch):
    """When CME fails but FRED has FEDTARMD + FEDFUNDS, fall back to the proxy."""
    monkeypatch.setattr(fed_futures_collector, "_try_cme", lambda: None)
    monkeypatch.setattr(fed_futures_collector.settings, "FRED_API_KEY", "testkey")

    def _fake_latest(series_id, api_key):
        return {"FEDTARMD": 4.50, "FEDFUNDS": 5.25}[series_id]

    monkeypatch.setattr(fed_futures_collector, "_fred_latest", _fake_latest)

    result = fed_futures_collector.collect()
    assert result["data_source"] == "fred_proxy"
    # current 5.25 vs projected 4.50 → 3 cuts of 25bp priced, dovish
    assert result["cuts_priced_12m"] == pytest.approx(3.0)
    assert result["interpretation"] == "dovish"


# ── 2. fred_collector derived fields ──────────────────────────────────────────

def test_derived_fields_present(monkeypatch):
    """With mocked FRED data the derived forward-rate fields are computed."""
    monkeypatch.setattr(fred_collector.settings, "FRED_API_KEY", "testkey")

    def _fake_fetch(series_id, api_key, limit=30):
        return _SAMPLE_FRED.get(series_id, _mk_obs(1.0, 1.0))

    monkeypatch.setattr(fred_collector, "_fetch_observations", _fake_fetch)

    out = fred_collector.collect()

    assert "real_yield_10y" in out
    assert out["real_yield_10y"] == pytest.approx(2.20)  # 4.50 - 2.30
    assert "yield_curve_momentum_20d" in out
    assert out["yield_curve_momentum_20d"] == pytest.approx(20.0)  # +0.20pp = 20bps
    assert out["yield_curve_momentum_label"] == "steepening"
    assert "dgs10_vs_200sma" in out
    assert out["dgs10_vs_200sma"] == "above"
    assert "credit_spread_oas" in out
    assert out["credit_spread_oas"] == pytest.approx(3.20)

    # interpretation hints accompany each derived field
    assert "real_yield_10y_interpretation" in out
    assert "credit_spread_oas_interpretation" in out
    assert "breakeven_inflation_10y" in out
    assert out["breakeven_inflation_10y"] == pytest.approx(2.30)
    assert "forward_inflation_5y5y" in out


# ── 2b. Tier 2 labor-market fields ────────────────────────────────────────────

def test_labor_fields_present(monkeypatch):
    """NFP, JOLTS, claims, and wage-growth derived fields are computed."""
    monkeypatch.setattr(fred_collector.settings, "FRED_API_KEY", "testkey")
    monkeypatch.setattr(
        fred_collector, "_fetch_observations",
        lambda series_id, api_key, limit=30: _SAMPLE_FRED.get(series_id, _mk_obs(1.0, 1.0)),
    )

    out = fred_collector.collect()

    # NFP: +180k MoM → solid
    assert out["nfp_change_mom_k"] == pytest.approx(180.0)
    assert out["nfp_change_interpretation"] == "solid_job_growth"

    # JOLTS: openings present + vacancy/unemployment ratio ≈ 1.23 (tight)
    assert out["jolts_openings_k"] == pytest.approx(8000)
    assert out["jolts_openings_per_unemployed"] == pytest.approx(8000 / 6500, abs=0.01)
    assert out["labor_tightness_interpretation"] == "tight_labor_market"

    # Initial claims: 220k → low/tight
    assert out["initial_claims"] == pytest.approx(220000)
    assert out["initial_claims_interpretation"] == "low_claims_tight_labor"

    # Wage growth YoY ≈ 3.99%
    assert out["wage_growth_yoy"] == pytest.approx(3.99, abs=0.05)
    assert out["wage_growth_interpretation"] == "elevated_wage_growth"


def test_labor_block_in_pass1(monkeypatch):
    """The Pass 1 labor block surfaces the labor fields under a LABOR MARKET header."""
    block = llm_client._build_labor_block({
        "nfp_change_mom_k": 180.0,
        "nfp_change_interpretation": "solid_job_growth",
        "jolts_openings_per_unemployed": 1.23,
        "initial_claims": 220000,
        "wage_growth_yoy": 3.99,
    })
    assert "LABOR MARKET" in block
    assert "solid_job_growth" in block
    assert "rate_regime" in block  # instructs the model to weigh labor into the rate path


# ── 3. rate_regime flows through Pass 1 ───────────────────────────────────────

def test_rate_regime_in_pass1_output(monkeypatch):
    """A model response with rate_regime survives normalisation into Pass 1 output."""
    llm_payload = {
        "regime": "risk_on",
        "confidence": 4,
        "macro_bias": "bullish",
        "volatility_regime": "low",
        "key_signals": ["falling real yields"],
        "regime_reasoning": "Easing bias supports risk.",
        "cross_asset_message": "Bonds and gold bid.",
        "upcoming_risks": ["CPI"],
        "rate_regime": "easing",
        "rate_regime_implication": {
            "GLD": "bullish",
            "TLT": "bullish",
            "QQQ": "bullish",
            "IBIT": "neutral",
        },
    }
    monkeypatch.setattr(llm_client, "_route_llm", lambda *a, **k: json.dumps(llm_payload))

    result = llm_client.run_pass1_regime(
        macro_data={
            "real_yield_10y": 2.2,
            "real_yield_10y_interpretation": "restrictive_headwind_for_gold",
            "yield_curve_momentum_20d": 20.0,
            "dgs10_vs_200sma": "above",
            "credit_spread_oas": 3.2,
        },
        sentiment_data={},
        price_data={},
        calendar_data={},
        fed_futures_data={"interpretation": "dovish", "data_source": "fred_proxy"},
        provider="claude",
        date_str="2026-06-05",
    )

    assert result["rate_regime"] == "easing"
    assert "rate_regime_implication" in result
    assert result["rate_regime_implication"]["GLD"] == "bullish"
    assert result["rate_regime_implication"]["TLT"] == "bullish"


def test_pass1_rates_block_formats_fed_futures():
    """The Pass 1 rates block includes the forward-rate / credit sections."""
    block = llm_client._build_rates_block(
        macro_data={
            "real_yield_10y": 2.2,
            "breakeven_inflation_10y": 2.3,
            "yield_curve_momentum_20d": 20.0,
            "dgs10_vs_200sma": "above",
            "credit_spread_oas": 3.2,
        },
        fed_futures_data={"interpretation": "dovish", "cuts_priced_12m": 3.0},
    )
    assert "FORWARD RATE PRICING:" in block
    assert "REAL YIELDS AND INFLATION EXPECTATIONS:" in block
    assert "RATE MOMENTUM:" in block
    assert "CREDIT CONDITIONS:" in block
    assert "dovish" in block
