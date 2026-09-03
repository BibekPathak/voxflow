from __future__ import annotations

from fastapi.testclient import TestClient

from app.evaluation.reports import compare_reports, to_markdown
from app.evaluation.runner import run_all
from app.main import create_app
from tests.conftest import make_settings


def _fake_report(metric_value: int) -> dict:
    return {
        "run_id": "x",
        "duration_ms": 10,
        "summary": {"total": 2, "passed": 2, "failed": 0},
        "scenarios": [
            {"name": "a", "passed": True, "duration_ms": 5, "latencies_ms": {"ttfa": metric_value}},
            {"name": "b", "passed": True, "duration_ms": 5, "latencies_ms": {"ttfa": metric_value}},
        ],
    }


async def test_all_scenarios_pass() -> None:
    report = await run_all()
    summary = report["summary"]
    assert summary["total"] == 7
    assert summary["failed"] == 0
    for scenario in report["scenarios"]:
        assert scenario["passed"], scenario
    assert report["run_id"]


def test_markdown_report_render() -> None:
    rendered = to_markdown(_fake_report(250))
    assert "# VoxFlow Evaluation Report" in rendered
    assert "2/2 passed" in rendered
    assert "ttfa" in rendered


def test_compare_reports_delta() -> None:
    rendered = compare_reports(_fake_report(300), _fake_report(150))
    assert "300" in rendered
    assert "150" in rendered
    assert "-150" in rendered


def test_evaluation_api_endpoints() -> None:
    with TestClient(create_app(make_settings())) as client:
        created = client.post("/evaluations/run", json={"scenario": "simple_question"})
        assert created.status_code == 201
        body = created.json()
        assert body["summary"]["total"] == 1
        assert body["summary"]["passed"] == 1
        run_id = body["run_id"]

        assert client.get("/evaluations/scenarios").json()["scenarios"]
        listing = client.get("/evaluations").json()
        assert any(run["run_id"] == run_id for run in listing)

        markdown = client.get(f"/evaluations/{run_id}?format=markdown")
        assert markdown.status_code == 200
        assert "VoxFlow Evaluation Report" in markdown.text

        assert client.get("/evaluations/does-not-exist").status_code == 404
