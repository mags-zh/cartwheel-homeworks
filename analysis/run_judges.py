"""HW5 judge pipeline for ``missing_policy_lookup``.

Run from the repository root:

    uv run python -m analysis.run_judges prepare
    uv run python -m analysis.run_judges split
    uv run --env-file .env python -m analysis.run_judges dev analysis/prompts/missing_policy_lookup-v0.txt
    uv run --env-file .env python -m analysis.run_judges dev --judge-id missing_policy_lookup-v0   # resume
    uv run --env-file .env python -m analysis.run_judges test --judge-id missing_policy_lookup-v1
    uv run python -m analysis.run_judges metrics --judge-id missing_policy_lookup-v1 --split test
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from agent import db
from analysis.helpers import freeze_judge, judge_alignment, register_judge, run_judge, split_labels
from analysis.helpers.selection import load_traces
from analysis.helpers.tools import _load_labels

MODE = "missing_policy_lookup"
REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_EXPORT = REPO_ROOT / "traces" / "support_traces.json"
INPUTS_PATH = REPO_ROOT / "analysis" / "state" / "hw5_trace_inputs.json"
REPORT_DIR = REPO_ROOT / "analysis" / "report"
JUDGE_MODEL = "gpt-4o-mini"

# One conversation is kept from each group of duplicate runs or close scenario
# variants. The excluded ones keep their labels but get no judge input, so
# split_labels never sees them. Scenario id -> reason.
EXCLUDED: dict[str, str] = {
    "support-0242": "close variant of support-0241 (product 3, missing title, product search)",
    "support-0244": "close variant of support-0241 (product 3, missing title, product search)",
    "support-0248": "close variant of support-0246 (product 4, invalid price, product search)",
    "support-0249": "close variant of support-0246 (product 4, invalid price, product search)",
}

INPUT_ROLES = {"system", "user", "assistant", "tool_call", "tool_result"}


def _session_context(conn: Any, metadata: dict[str, Any]) -> dict[str, str]:
    """The session fields the agent's system prompt shows (agent.py, "Session context")."""
    user_id = metadata.get("cartwheel.user_id")
    user = db.get_user(conn, int(user_id)) if user_id is not None else None
    store_id = user.store_id if user else None
    return {
        "role": "system",
        "text": f"Session context: User role: {metadata.get('cartwheel.user_role')}; "
        f"User id: {user_id}; Store id: {store_id}",
    }


def _judge_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep one message's role and its text or tool data.

    Tool calls and results carry the tool name in the flattened text the
    judge reads; the normalizer would otherwise show only the arguments.
    """
    role = message["role"]
    if role == "tool_call":
        arguments = json.dumps(message.get("arguments"), ensure_ascii=False, sort_keys=True)
        return {"role": role, "name": message["name"], "arguments": f"{message['name']}({arguments})"}
    if role == "tool_result":
        content = json.dumps(message.get("content"), ensure_ascii=False, sort_keys=True)
        return {"role": role, "name": message["name"], "content": f"{message['name']} returned {content}"}
    if role in {"user", "assistant"}:
        return {"role": role, "text": message["text"]}
    raise ValueError(f"unexpected message role in trace: {role}")


def prepare_inputs(mode: str = MODE) -> list[dict[str, Any]]:
    """Write one judge input per eligible labelled conversation.

    Each record holds only ``trace_id`` and ``trace``: the session context,
    every user turn, every assistant message including interim text, and every
    tool call and result. Labels, evidence, notes and scenario metadata stay out.
    """
    traces = {trace["trace_id"]: trace for trace in load_traces(TRACE_EXPORT)}
    labels = _load_labels(mode)
    records = []
    with db.connection() as conn:
        for label in sorted(labels, key=lambda row: traces[row["trace_id"]]["meta"]["scenario_id"]):
            trace = traces[label["trace_id"]]
            if trace["meta"]["scenario_id"] in EXCLUDED:
                continue
            messages = [_session_context(conn, trace["metadata"])]
            messages += [_judge_message(message) for message in trace["trace"]]
            records.append({"trace_id": trace["trace_id"], "trace": messages})
    INPUTS_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return records


def split_data(mode: str = MODE) -> dict[str, list[str]]:
    """Split the eligible labels 20/40/40 into train, dev and test. Run once.

    Saved to ``analysis/state/splits.json`` under ``mode``; keep it unchanged
    during development.
    """
    records = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    return split_labels(
        mode,
        fractions=(0.20, 0.40, 0.40),
        seed=7,
        min_per_class=10,
        eligible_trace_ids=[record["trace_id"] for record in records],
    )


def _use_saved_inputs() -> None:
    """Make the judge read the saved inputs, never live Langfuse traces."""
    source = os.environ.setdefault("CARTWHEEL_JUDGE_TRACE_SOURCE", str(INPUTS_PATH))
    if Path(source).resolve() != INPUTS_PATH.resolve():
        raise RuntimeError(f"CARTWHEEL_JUDGE_TRACE_SOURCE points to {source}, not {INPUTS_PATH}")


def _save_report(name: str, judge_id: str, metrics: dict[str, Any], prompt_path: str | None) -> Path:
    judge = json.loads((REPO_ROOT / "analysis" / "state" / "judges" / f"{judge_id}.json").read_text())
    report = {
        **metrics,
        "model": judge["model"],
        "prompt_hash": judge["prompt_hash"],
        "prompt_path": prompt_path,
        "class_counts": {"pass": metrics["tp"] + metrics["fn"], "fail": metrics["tn"] + metrics["fp"]},
    }
    path = REPORT_DIR / f"{name}-{judge_id}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def run_development(mode: str, prompt_path: str, judge_id: str | None = None) -> dict[str, Any]:
    """Register a prompt version, judge the dev split, and save the dev metrics.

    Pass ``judge_id`` to resume an interrupted run without registering again;
    completed batches are cached.
    """
    _use_saved_inputs()
    if judge_id is None:
        judge_id = register_judge(
            mode=mode, prompt_text=Path(prompt_path).read_text(encoding="utf-8"), judge_model=JUDGE_MODEL
        )["judge_id"]
    run_judge(judge_id, split="dev", batch_size=10)
    metrics = judge_alignment(judge_id, split="dev")
    _save_report("dev", judge_id, metrics, prompt_path)
    return metrics


def run_test(judge_id: str) -> dict[str, Any]:
    """Freeze the chosen judge, judge the test split once, and save the test metrics.

    Rerunning after an interruption skips the freeze (it is one-way) and only
    classifies the test traces that have no cached prediction yet.
    """
    _use_saved_inputs()
    judge = json.loads((REPO_ROOT / "analysis" / "state" / "judges" / f"{judge_id}.json").read_text())
    if judge.get("status") != "frozen":
        freeze_judge(judge_id)
    run_judge(judge_id, split="test", batch_size=10)
    metrics = judge_alignment(judge_id, split="test")
    _save_report("test", judge_id, metrics, f"analysis/prompts/{judge_id}.txt")
    return metrics


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def recalculate_metrics(judge_id: str, split: str) -> dict[str, Any]:
    """Recompute the confusion counts, TPR, TNR and Wilson intervals from saved files only.

    Reads the judge's cached predictions, the human labels and splits.json; no
    model call. Pass (1) is the positive class, Fail (0) the negative class.
    """
    judge = json.loads((REPO_ROOT / "analysis" / "state" / "judges" / f"{judge_id}.json").read_text())
    if split == "test" and judge.get("status") != "frozen":
        raise RuntimeError(f"{judge_id} is not frozen; test metrics are only for a frozen judge")
    predictions = judge["predictions"][judge["prompt_hash"]]  # 1 = judge says Pass
    labels = {row["trace_id"]: 1 - row["label"] for row in _load_labels(judge["mode"])}  # 1 = human Pass
    ids = json.loads((REPO_ROOT / "analysis" / "state" / "splits.json").read_text())[judge["mode"]][split]
    tp = sum(labels[t] == 1 and predictions[t] == 1 for t in ids)
    fn = sum(labels[t] == 1 and predictions[t] == 0 for t in ids)
    tn = sum(labels[t] == 0 and predictions[t] == 0 for t in ids)
    fp = sum(labels[t] == 0 and predictions[t] == 1 for t in ids)
    return {"judge_id": judge_id, "split": split, "n": len(ids), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "tpr": tp / (tp + fn), "tnr": tn / (tn + fp),
            "tpr_interval": _wilson(tp, tp + fn), "tnr_interval": _wilson(tn, tn + fp)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "split", "dev", "test", "metrics"])
    parser.add_argument("prompt_path", nargs="?")
    parser.add_argument("--judge-id")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    if args.command == "metrics":
        m = recalculate_metrics(args.judge_id, args.split)
        print(f"{m['judge_id']}, {m['split']} split, {m['n']} conversations (Pass = positive class)\n")
        print(f"                 human Pass   human Fail")
        print(f"  judge Pass     TP {m['tp']:>3}       FP {m['fp']:>3}   (FP = missed failure)")
        print(f"  judge Fail     FN {m['fn']:>3}       TN {m['tn']:>3}\n")
        print(f"  TPR = TP / (TP + FN) = {m['tp']} / {m['tp'] + m['fn']} = {m['tpr']:.3f}   "
              f"95% Wilson {m['tpr_interval'][0]:.3f} to {m['tpr_interval'][1]:.3f}")
        print(f"  TNR = TN / (TN + FP) = {m['tn']} / {m['tn'] + m['fp']} = {m['tnr']:.3f}   "
              f"95% Wilson {m['tnr_interval'][0]:.3f} to {m['tnr_interval'][1]:.3f}")
        return
    if args.command in ("dev", "test"):
        if args.command == "dev":
            metrics = run_development(MODE, args.prompt_path, judge_id=args.judge_id)
        else:
            metrics = run_test(args.judge_id)
        print(f"{metrics['judge_id']} {args.command}: TP {metrics['tp']} FN {metrics['fn']} TN {metrics['tn']} FP {metrics['fp']}")
        print(f"TPR {metrics['tpr']:.3f} {metrics['tpr_interval']}  TNR {metrics['tnr']:.3f} {metrics['tnr_interval']}")
    elif args.command == "prepare":
        records = prepare_inputs()
        print(f"wrote {len(records)} records to {INPUTS_PATH.relative_to(REPO_ROOT)}")
    elif args.command == "split":
        labels = {row["trace_id"]: row["label"] for row in _load_labels(MODE)}
        for name, ids in split_data().items():
            fails = sum(labels[tid] == 1 for tid in ids)
            print(f"{name:5} {len(ids):3} conversations: Fail {fails}, Pass {len(ids) - fails}")


if __name__ == "__main__":
    main()
