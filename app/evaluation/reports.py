from __future__ import annotations

from typing import Any


def _metric_rows(report: dict[str, Any]) -> dict[str, str]:
    """Aggregate median latency metrics across all scenarios of a report."""
    rows: dict[str, list[float]] = {}
    for scenario in report["scenarios"]:
        for metric, value in (scenario.get("latencies_ms") or {}).items():
            if isinstance(value, (int, float)):
                rows.setdefault(metric, []).append(float(value))
    return {metric: f"{sum(values) / len(values):.0f}" for metric, values in rows.items()}


def to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# VoxFlow Evaluation Report",
        "",
        f"- Run: `{report['run_id']}`",
        f"- Duration: {report['duration_ms']} ms",
        f"- Scenarios: {summary['passed']}/{summary['total']} passed",
        "",
        "## Per-scenario results",
        "",
        "| Scenario | Passed | Duration (ms) |",
        "| --- | --- | --- |",
    ]
    for scenario in report["scenarios"]:
        lines.append(f"| {scenario['name']} | {'yes' if scenario['passed'] else 'no'} | {scenario['duration_ms']} |")

    lines += ["", "## Voice metrics (mean across scenarios)", "", "| Metric | Mean (ms) |", "| --- | --- |"]
    for metric, value in _metric_rows(report).items():
        lines.append(f"| {metric} | {value} |")

    for scenario in report["scenarios"]:
        if scenario["passed"]:
            continue
        lines += ["", f"### {scenario['name']} failed", "", "```json", f"{scenario.get('checks')}", "```"]
    return "\n".join(lines) + "\n"


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    base_metrics = _metric_rows(baseline)
    candidate_metrics = _metric_rows(candidate)
    metrics = sorted(set(base_metrics) | set(candidate_metrics))
    lines = [
        "## Latency comparison",
        "",
        "| Metric | baseline | candidate | delta |",
        "| --- | --- | --- | --- |",
    ]
    for metric in metrics:
        baseline_value = base_metrics.get(metric)
        candidate_value = candidate_metrics.get(metric)
        delta = ""
        if baseline_value is not None and candidate_value is not None:
            delta = f"{int(candidate_value) - int(baseline_value):+d}"
        lines.append(f"| {metric} | {baseline_value or '-'} | {candidate_value or '-'} | {delta or '-'} |")
    lines += [
        "",
        f"- Task success: {baseline['summary']['passed']}/{baseline['summary']['total']}"
        f" -> {candidate['summary']['passed']}/{candidate['summary']['total']}",
    ]
    return "\n".join(lines) + "\n"
