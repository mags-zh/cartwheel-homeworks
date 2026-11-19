"""Monitor one failure mode on Langfuse traces: sample, judge, correct, score.

Run from the repository root:

    uv run --env-file .env python -m monitoring.run --period before --dry-run
    uv run --env-file .env python -m monitoring.run --period after
    uv run --env-file .env python -m monitoring.run --last-hours 24

``--period`` reads one period from ``monitoring/config.json``. Every monitoring
scenario must be present with exactly its planned number of traces, so a
period never combines separate retries. ``--last-hours`` groups the window's
traces by session instead. Every conversation must come from the configured
Cartwheel model. ``--dry-run`` stops after showing the sample and the number
of judge calls.

Verdicts use 1 when the failure is present and 0 when it is absent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from monitoring.sample import DEFAULT_RISK_GROUPS, select_traces

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "monitoring" / "config.json"
SCENARIOS_PATH = REPO_ROOT / "scenarios" / "monitoring_scenarios.jsonl"
HISTORY_PATH = REPO_ROOT / "monitoring" / "history.jsonl"
CHART_PATH = REPO_ROOT / "monitoring" / "prevalence.svg"
OUTPUT_DIR = REPO_ROOT / "monitoring" / "output"


class PeriodRejected(ValueError):
    """The traces in a window do not form a valid monitoring period."""


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(config["risk_groups"]) - set(DEFAULT_RISK_GROUPS)
    if unknown:
        raise ValueError(f"unknown risk groups in config: {sorted(unknown)}")
    return config


def planned_turns(path: Path = SCENARIOS_PATH) -> dict[str, int]:
    """Scenario id -> number of user turns, and so of Langfuse traces."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["id"]: 1 + len(row.get("followups") or []) for row in rows}


def fetch_window(start: datetime, end: datetime, client: Any | None = None) -> list[dict[str, Any]]:
    """Fetch and normalize every Langfuse trace with a timestamp in [start, end]."""
    from langfuse import Langfuse

    from analysis.helpers.normalization import normalize_trace

    lf = client or Langfuse()
    summaries: list[Any] = []
    page = 1
    while True:
        response = lf.api.trace.list(from_timestamp=start, to_timestamp=end, page=page, limit=100)
        batch = list(response.data or [])
        summaries.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    traces: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        if summary.id in traces:
            continue
        try:
            traces[summary.id] = normalize_trace(lf.api.trace.get(summary.id))
        except ValueError:
            # A trace with no messages (for example a health check) is not a conversation turn.
            continue
    return list(traces.values())


def same_model(observed: str, configured: str) -> bool:
    """Langfuse records the provider snapshot, e.g. gpt-4o-mini-2024-07-18."""
    return observed == configured or bool(
        re.fullmatch(re.escape(configured) + r"-\d{4}-\d{2}-\d{2}", observed)
    )


def build_conversations(traces: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Group normalized traces by ``meta[key]`` into one record per conversation.

    Each Langfuse trace holds one user turn, so the turns are joined in time
    order. The record id is the final trace id, which receives the scores.
    Records are sorted by the grouping value so the random sample is
    reproducible.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        value = trace["meta"].get(key)
        if value:
            groups[str(value)].append(trace)

    conversations = []
    for value in sorted(groups):
        group = sorted(groups[value], key=lambda t: (t.get("timestamp") or "", t["trace_id"]))
        messages = [message for trace in group for message in trace["trace"]]
        conversations.append(
            {
                "id": group[-1]["trace_id"],
                "scenario_id": group[0]["meta"].get("scenario_id"),
                "session_id": group[0]["meta"].get("session_id"),
                "trace_ids": [trace["trace_id"] for trace in group],
                "models": list(dict.fromkeys(m for trace in group for m in trace["models"])),
                "tools": list(
                    dict.fromkeys(
                        m["name"] for m in messages if m.get("role") == "tool_call" and m.get("name")
                    )
                ),
                "turn_count": sum(m.get("role") == "user" for m in messages),
                "metadata": group[0]["metadata"],
                "messages": messages,
            }
        )
    return conversations


def validate_period(conversations: list[dict[str, Any]], planned: dict[str, int]) -> None:
    """Reject a period that misses a scenario or combines separate retries."""
    by_scenario = {c["scenario_id"]: c for c in conversations}
    missing = sorted(set(planned) - set(by_scenario))
    if missing:
        raise PeriodRejected(f"{len(missing)} monitoring scenario ids missing, e.g. {missing[:5]}")
    wrong = {
        sid: f"{len(by_scenario[sid]['trace_ids'])} traces, {turns} planned"
        for sid, turns in planned.items()
        if len(by_scenario[sid]["trace_ids"]) != turns
    }
    if wrong:
        raise PeriodRejected(f"trace counts differ from planned turns (a retry or an incomplete run): {wrong}")


def check_models(conversations: list[dict[str, Any]], model: str) -> None:
    wrong = {
        c["id"]: c["models"]
        for c in conversations
        if not c["models"] or not all(same_model(m, model) for m in c["models"])
    }
    if wrong:
        raise PeriodRejected(f"{len(wrong)} conversations not from model {model!r}: {dict(list(wrong.items())[:3])}")


def judge_text(conversation: dict[str, Any], conn: Any) -> str:
    """The Homework 5 judge input: session context, turns, named tool calls and results."""
    from analysis.helpers.normalization import normalize_trace
    from analysis.run_judges import _judge_message, _session_context

    messages = [_session_context(conn, conversation["metadata"])]
    messages += [_judge_message(message) for message in conversation["messages"]]
    return normalize_trace({"trace_id": conversation["id"], "trace": messages})["text"]


def attach_period_trace(
    scores: list[dict[str, Any]], record: dict[str, Any], estimate: dict[str, Any], client: Any | None = None
) -> list[dict[str, Any]]:
    """Give the period score a trace to attach to.

    Langfuse 3 rejects a score without a trace, session or dataset run, so each
    monitoring run writes one small trace for its period. The trace id comes
    from the mode and the period label, so a rerun updates the same trace and
    score.
    """
    from langfuse import get_client

    lf = client or get_client()
    mode = record["judge_mode"]
    trace_id = lf.create_trace_id(seed=f"monitoring|{mode}|{record['label']}")
    summary = {key: record[key] for key in ("label", "from", "to", "judge_id", "model",
                                            "trace_count", "conversation_count")}
    span = lf.start_span(trace_context={"trace_id": trace_id}, name=f"monitoring.{mode}",
                         input=summary, output=estimate)
    span.update_trace(name=f"monitoring.{mode}", input=summary, output=estimate,
                      metadata={"monitoring.period": record["label"]}, tags=["monitoring"])
    span.end()
    lf.flush()
    return [score if score["trace_id"] is not None else {**score, "trace_id": trace_id} for score in scores]


def _upsert_history(record: dict[str, Any], path: Path = HISTORY_PATH) -> None:
    """Keep one line per label: a rerun replaces its line in place."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    labels = [line["label"] for line in lines]
    if record["label"] in labels:
        lines[labels.index(record["label"])] = record
    else:
        lines.append(record)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _write_chart(config: dict[str, Any]) -> None:
    from monitoring.chart import prevalence_chart

    history = [json.loads(line) for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    points = [h for h in history if h.get("corrected") is not None]
    if points:
        CHART_PATH.write_text(
            prevalence_chart(points, threshold=config["threshold"], mode=config["judge_mode"]) + "\n",
            encoding="utf-8",
        )


def monitor(
    conversations: list[dict[str, Any]],
    config: dict[str, Any],
    label: str,
    window: tuple[datetime, datetime],
    *,
    dry_run: bool = False,
    judge: Callable[[str, list[dict[str, Any]]], dict[str, int]] | None = None,
    post: Callable[[list[dict[str, Any]]], int] | None = None,
) -> dict[str, Any] | None:
    """Sample, judge, correct and score one window. Returns the history record."""
    record: dict[str, Any] = {
        "label": label,
        "from": _iso(window[0]),
        "to": _iso(window[1]),
        "judge_id": config["judge_id"],
        "judge_mode": config["judge_mode"],
        "model": config["model"],
        "trace_count": sum(len(c["trace_ids"]) for c in conversations),
        "conversation_count": len(conversations),
    }
    print(f"{label}: {record['trace_count']} Langfuse traces -> {len(conversations)} conversations")
    if not conversations:
        print("no eligible conversations; recording a zero count without calling the judge")
        if not dry_run:
            _upsert_history({**record, "random_count": 0, "risk_count": 0, "judge_calls": 0,
                             "raw": None, "corrected": None, "run_at": _iso(datetime.now(timezone.utc))})
        return record

    groups = {name: DEFAULT_RISK_GROUPS[name] for name in config["risk_groups"]}
    plan = select_traces(conversations, config["random_rate"], groups)
    random_ids = [c["id"] for c in plan["random"]]
    risk_ids = list(dict.fromkeys(c["id"] for members in plan["risk_groups"].values() for c in members))
    print(f"random sample: {len(random_ids)} ({config['random_rate']:.0%} of {len(conversations)})")
    for name, members in plan["risk_groups"].items():
        print(f"risk group {name}: {len(members)}")
    both = len(set(random_ids) & set(risk_ids))
    print(f"judge calls: {len(plan['to_judge'])} (union; {both} in both) "
          f"with {config['judge_id']}")
    if dry_run:
        return record

    from agent import db
    from monitoring.correct import corrected_mode_prevalence
    from monitoring.run_judges import judge_sample, judge_test_data
    from monitoring.write_scores import build_score_records, post_scores

    with db.connection() as conn:
        to_judge = [{"id": c["id"], "text": judge_text(c, conn)} for c in plan["to_judge"]]
    verdicts = (judge or judge_sample)(config["judge_id"], to_judge)
    random_verdicts = {tid: verdicts[tid] for tid in random_ids}
    risk_verdicts = {tid: verdicts[tid] for tid in risk_ids}

    estimate = corrected_mode_prevalence(list(random_verdicts.values()), *judge_test_data(config["judge_id"]))
    scores = build_score_records(config["judge_mode"], random_verdicts, risk_verdicts, estimate, label)
    if post is None:
        scores = attach_period_trace(scores, record, estimate)
    posted = (post or post_scores)(scores)

    by_id = {c["id"]: c for c in conversations}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{label}.json").write_text(
        json.dumps(
            {
                **record,
                "random": [
                    {"trace_id": tid, "scenario_id": by_id[tid]["scenario_id"], "verdict": v}
                    for tid, v in random_verdicts.items()
                ],
                "risk_groups": {
                    name: [
                        {"trace_id": c["id"], "scenario_id": c["scenario_id"], "verdict": verdicts[c["id"]]}
                        for c in members
                    ]
                    for name, members in plan["risk_groups"].items()
                },
                "estimate": estimate,
                "score_ids": [s["score_id"] for s in scores],
                "conversations": [
                    {key: c[key] for key in ("id", "scenario_id", "session_id", "trace_ids", "tools", "turn_count")}
                    for c in conversations
                ],
                "judge_inputs": to_judge,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    record.update(
        {
            "random_count": len(random_ids),
            "risk_count": len(risk_ids),
            "risk_group_counts": {name: len(m) for name, m in plan["risk_groups"].items()},
            "judge_calls": len(to_judge),
            "random_flagged": sum(random_verdicts.values()),
            "risk_flagged": sum(risk_verdicts.values()),
            **{key: estimate[key] for key in ("raw", "corrected", "ci_low", "ci_high", "confidence",
                                              "failure_sensitivity", "pass_specificity")},
            "threshold": config["threshold"],
            "above_threshold": estimate["corrected"] > config["threshold"],
            "scores_sent": posted,
            "run_at": _iso(datetime.now(timezone.utc)),
        }
    )
    _upsert_history(record)
    _write_chart(config)
    print(f"raw {estimate['raw']}  corrected {estimate['corrected']}  "
          f"95% CI {estimate['ci_low']}-{estimate['ci_high']}  threshold {config['threshold']}")
    print(f"risk verdicts: {sum(risk_verdicts.values())} of {len(risk_verdicts)} flagged; "
          f"{posted} scores sent to Langfuse")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--period", help="period label in monitoring/config.json")
    window.add_argument("--last-hours", type=float, help="monitor the traces of the last N hours")
    parser.add_argument("--dry-run", action="store_true", help="stop before calling the judge")
    args = parser.parse_args(argv)
    config = load_config()

    try:
        if args.period:
            period = next((p for p in config["periods"] if p["label"] == args.period), None)
            if period is None:
                parser.error(f"no period {args.period!r} in {CONFIG_PATH.name}")
            start, end, label = _utc(period["from"]), _utc(period["to"]), period["label"]
            planned = planned_turns()
            traces = fetch_window(start, end)
            conversations = [c for c in build_conversations(traces, "scenario_id") if c["scenario_id"] in planned]
            validate_period(conversations, planned)
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=args.last_hours)
            label = f"last-{args.last_hours:g}h-{end:%Y-%m-%dT%H%MZ}"
            traces = fetch_window(start, end)
            conversations = build_conversations(traces, "session_id")
        check_models(conversations, config["model"])
    except PeriodRejected as error:
        print(f"period rejected: {error}", file=sys.stderr)
        return 2

    monitor(conversations, config, label, (start, end), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
