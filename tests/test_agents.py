from __future__ import annotations

import json

import pytest

from v2 import agents
from v2 import event_gate


class _DummyContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _DummyOutput:
    def __init__(self, text: str) -> None:
        self.content = [_DummyContent(text)]


class _DummyUsage:
    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens


class _DummyResponse:
    def __init__(self, payload: dict, tokens: int = 42) -> None:
        self.output = [_DummyOutput(json.dumps(payload))]
        self.usage = _DummyUsage(tokens)


class _DummyResponses:
    def __init__(self, payload: dict, called: dict) -> None:
        self._payload = payload
        self._called = called

    def create(self, **kwargs):
        self._called["kwargs"] = kwargs
        return _DummyResponse(self._payload)


class _DummyClient:
    def __init__(self, payload: dict, called: dict) -> None:
        self.responses = _DummyResponses(payload, called)


@pytest.fixture(autouse=True)
def _clear_openai_cache(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(agents, "OpenAI", object)


def test_score_events_invokes_openai_with_reasoning(monkeypatch):
    called: dict = {}
    payload = {
        "symbol": "AAPL",
        "direction": 1,
        "magnitude": "high",
        "confidence": 0.5,
        "half_life_days": 3,
        "rationale": ["Earnings beat"],
        "risks": ["Guidance uncertainty"],
    }

    dummy_client = _DummyClient(payload, called)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(agents, "_client", lambda: dummy_client)
    monkeypatch.setattr(agents, "record_assessment", lambda: None)

    news = {"AAPL": [{"headline": "AAPL beats", "summary": "Strong growth"}]}
    bps_map = {"low": 3, "med": 7, "high": 12}

    scores, meta = agents.score_events_for_symbols(news, "gpt-x", "medium", bps_map, max_abs_bps=20)

    assert called["kwargs"]["reasoning"] == {"effort": "medium"}
    assert called["kwargs"]["model"] == meta["model"]
    assert scores["AAPL"] == pytest.approx(6.0)
    assert meta["called"] is True
    assert meta["tokens"] == 42
    assert "AAPL" in meta["details"]
    detail = meta["details"]["AAPL"]
    assert detail["bps"] == pytest.approx(6.0)
    assert "Buy tilt" in detail["summary"]
    assert "Earnings beat" in detail["summary"]
    assert isinstance(meta["summaries"], list) and meta["summaries"], "summary list populated"


def test_successful_assessment_records_timestamp(tmp_path, monkeypatch):
    last_path = tmp_path / "last.json"

    called: dict = {}
    payload = {
        "symbol": "MSFT",
        "direction": -1,
        "magnitude": "med",
        "confidence": 1.0,
        "half_life_days": 1,
        "rationale": [],
        "risks": [],
    }

    dummy_client = _DummyClient(payload, called)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(agents, "_client", lambda: dummy_client)
    monkeypatch.setattr(event_gate, "LAST_FILE", last_path)

    news = {"MSFT": [{"headline": "MSFT guidance", "summary": "Mixed"}]}
    bps_map = {"low": 3, "med": 7, "high": 12}

    scores, meta = agents.score_events_for_symbols(news, "gpt-x", "medium", bps_map, max_abs_bps=20)

    assert last_path.exists(), "event timestamp should be recorded"
    saved = json.loads(last_path.read_text())
    assert "ts_iso" in saved and isinstance(saved["ts_iso"], str)
    assert meta["called"] is True
    assert scores["MSFT"] == pytest.approx(-7.0)
    msft_detail = meta["details"].get("MSFT")
    assert msft_detail and "Sell tilt" in msft_detail["summary"]
    assert msft_detail["bps"] == pytest.approx(-7.0)
