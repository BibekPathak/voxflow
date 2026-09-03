from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.evaluation.reports import to_markdown
from app.evaluation.runner import list_scenario_names, run_all, run_scenario

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


class RunRequest(BaseModel):
    scenario: str | None = Field(default=None, description="Run one scenario by name; omit to run all.")


def _store(request: Request) -> list[dict]:
    return request.app.state.evaluation_runs


@router.get("/scenarios")
async def available_scenarios() -> dict:
    return {"scenarios": list_scenario_names()}


@router.post("/run", status_code=201)
async def run_evaluation(request: Request, body: RunRequest | None = None) -> dict:
    if body is None or body.scenario is None:
        report = await run_all()
    else:
        scenario = await run_scenario(body.scenario)
        passed = scenario["passed"]
        report = {
            "run_id": f"single_{scenario['name']}_{int(time.time())}",
            "started_at": time.time(),
            "duration_ms": scenario["duration_ms"],
            "summary": {"total": 1, "passed": 1 if passed else 0, "failed": 0 if passed else 1},
            "scenarios": [scenario],
        }
    _store(request).append(report)
    return report


@router.get("")
async def list_evaluations(request: Request) -> list[dict]:
    return [
        {
            "run_id": run["run_id"],
            "duration_ms": run["duration_ms"],
            "summary": run["summary"],
        }
        for run in _store(request)
    ]


@router.get("/{run_id}")
async def get_evaluation(request: Request, run_id: str, format: str = "json") -> Response:
    for run in reversed(_store(request)):
        if run["run_id"] == run_id:
            if format == "markdown":
                return PlainTextResponse(to_markdown(run), media_type="text/markdown")
            return JSONResponse(run)
    return JSONResponse(status_code=404, content={"code": "EVAL_NOT_FOUND", "run_id": run_id})
