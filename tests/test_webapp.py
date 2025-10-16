def test_run_now_triggers_manual_analysis(monkeypatch):
    import importlib

    webapp = importlib.import_module("webapp")
    agents_module = importlib.import_module("v2.agents")

    sample_episode = {
        "as_of": "2025-01-01T09:30:00-05:00",
        "gate": {
            "expected_alpha_bps": 15.0,
            "cost_bps": 5.0,
            "net_bps": 10.0,
            "turnover_pct": 8.0,
            "order_count": 2,
            "reasons": [],
            "proceed": True,
        },
        "orders_submitted": ["order-1", "order-2"],
        "planned_orders_count": 2,
        "diag": {},
    }

    call_counter = {"count": 0}

    def fake_run_once():
        call_counter["count"] += 1
        return sample_episode

    monkeypatch.setattr(webapp, "run_once", fake_run_once)
    monkeypatch.setattr(
        webapp,
        "_load_episodes",
        lambda path, limit=200, offset=0: ([sample_episode], 1),
    )

    client = webapp.app.test_client()

    try:
        response = client.post("/run-now")

        assert response.status_code == 302
        assert call_counter["count"] == 1

        location = response.headers["Location"]
        assert "/dashboard" in location
        assert "run_status=success" in location

        follow_up = client.get(location)
        assert b"Manual analysis complete" in follow_up.data
    finally:
        importlib.reload(agents_module)


def test_run_now_handles_tuple_result(monkeypatch):
    import importlib

    webapp = importlib.import_module("webapp")
    agents_module = importlib.import_module("v2.agents")

    sample_episode = {
        "as_of": "2025-01-01T09:30:00-05:00",
        "gate": {"expected_alpha_bps": 12.0, "cost_bps": 3.0, "net_bps": 9.0, "turnover_pct": 5.0, "order_count": 1, "reasons": []},
        "orders_submitted": ["order-1"],
        "planned_orders_count": 1,
    }
    sample_summary = {"summary": "sample summary", "plain_english": "sample summary"}

    def fake_run_once():
        return sample_episode, sample_summary

    monkeypatch.setattr(webapp, "run_once", fake_run_once)
    monkeypatch.setattr(
        webapp,
        "_load_episodes",
        lambda path, limit=200, offset=0: ([sample_episode], 1),
    )

    client = webapp.app.test_client()

    try:
        response = client.post("/run-now")

        assert response.status_code == 302

        location = response.headers["Location"]
        assert "run_status=success" in location

        follow_up = client.get(location)
        assert b"Manual analysis complete" in follow_up.data
        assert sample_summary["summary"].encode() in follow_up.data
    finally:
        importlib.reload(agents_module)
