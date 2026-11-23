"""Run the cascade's proxy judge on the HW5 labelled traces (HW9 Part D).

The proxy is a less expensive model given the oracle's judge prompt
(``missing_policy_lookup-v1``) and trace text exactly as
``analysis/helpers/scale.py`` builds them, plus one instruction to state its
confidence. Each verdict is appended to ``cascade/results/proxy-<split>.jsonl``
with the token counts the LLM API returned, so thresholds are evaluated offline
and an interrupted run resumes without paying again for finished traces.

The test split is run once, after the threshold is chosen on development
labels: ``--split test`` requires ``--threshold``, which is saved to
``cascade/results/chosen_threshold.json`` before any call and cannot change.

Usage (from the repository root, live model):

    uv run --env-file .env python -m cascade.run_proxy --split dev [--limit N]
    uv run --env-file .env python -m cascade.run_proxy --split test --threshold T
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGE_PATH = REPO_ROOT / "analysis" / "state" / "judges" / "missing_policy_lookup-v1.json"
SPLITS_PATH = REPO_ROOT / "analysis" / "state" / "splits.json"
INPUTS_PATH = REPO_ROOT / "analysis" / "state" / "hw5_trace_inputs.json"
CONFIG_PATH = REPO_ROOT / "optimize" / "config.json"
RESULTS = REPO_ROOT / "cascade" / "results"
CHOSEN_THRESHOLD = RESULTS / "chosen_threshold.json"

CONFIDENCE_INSTRUCTION = (
    "Also return confidence: a number from 0 to 1 stating how confident you are "
    "that your result is correct."
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "critique": {"type": "string"},
        "result": {"type": "string", "enum": ["Pass", "Fail"]},
        "confidence": {"type": "number"},
    },
    "required": ["critique", "result", "confidence"],
    "additionalProperties": False,
}


def judge_prompt(prompt_text: str, trace_text: str) -> str:
    """The oracle's DocETL prompt (scale.py) with the confidence instruction added."""
    return (
        f"{prompt_text}\n\n"
        "--- Trace to evaluate ---\n"
        f"{trace_text}\n\n"
        "First write a critique of the trace against the criterion. "
        "Use specific evidence from the provided trace. Then return result "
        "as exactly Pass when the named failure is absent, or Fail when present. "
        f"{CONFIDENCE_INSTRUCTION}"
    )


def trace_texts() -> dict[str, str]:
    os.environ["CARTWHEEL_JUDGE_TRACE_SOURCE"] = str(INPUTS_PATH)
    from analysis.helpers.scale import load_store_traces

    return {trace["trace_id"]: trace["text"] for trace in load_store_traces()}


def call_cost(prices: dict, usage: dict) -> float:
    cached = usage["cached_input_tokens"]
    return (
        (usage["input_tokens"] - cached) * prices["input"]
        + cached * prices["cached_input"]
        + usage["output_tokens"] * prices["output"]
    ) / 1_000_000


def call_proxy(model: str, effort: str, prompt: str) -> tuple[dict[str, Any], dict[str, int]]:
    import litellm

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=effort,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "judge_verdict", "strict": True, "schema": OUTPUT_SCHEMA},
        },
    )
    answer = json.loads(response.choices[0].message.content)
    if answer["result"] not in ("Pass", "Fail"):
        raise ValueError(f"proxy result must be Pass or Fail, got {answer['result']!r}")
    if not 0 <= float(answer["confidence"]) <= 1:
        raise ValueError(f"proxy confidence must be in [0, 1], got {answer['confidence']!r}")
    u = response.usage
    prompt_details = getattr(u, "prompt_tokens_details", None)
    completion_details = getattr(u, "completion_tokens_details", None)
    usage = {
        "input_tokens": int(u.prompt_tokens),
        "cached_input_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "output_tokens": int(u.completion_tokens),
        "reasoning_tokens": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
    }
    return answer, usage


def fix_threshold(threshold: float) -> None:
    if CHOSEN_THRESHOLD.exists():
        saved = json.loads(CHOSEN_THRESHOLD.read_text())["threshold"]
        if saved != threshold:
            raise SystemExit(f"threshold already fixed at {saved} in {CHOSEN_THRESHOLD}; it cannot change")
        return
    CHOSEN_THRESHOLD.write_text(json.dumps(
        {"threshold": threshold, "fixed_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning-effort", default="minimal")
    parser.add_argument("--threshold", type=float, help="required for --split test")
    parser.add_argument("--limit", type=int, help="judge only the first N unfinished traces")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.split == "test":
        if args.threshold is None:
            raise SystemExit("--split test needs the threshold chosen on development labels (--threshold)")
        fix_threshold(args.threshold)

    judge = json.loads(JUDGE_PATH.read_text())
    prices = json.loads(CONFIG_PATH.read_text())["prices_per_million_tokens_usd"][args.model]
    ids = json.loads(SPLITS_PATH.read_text())["missing_policy_lookup"][args.split]
    texts = trace_texts()
    out_path = RESULTS / f"proxy-{args.split}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            if (row["model"], row["reasoning_effort"]) != (args.model, args.reasoning_effort):
                raise SystemExit(f"{out_path} holds another proxy setup; move it before switching")
            done.add(row["trace_id"])
    todo = [tid for tid in ids if tid not in done][: args.limit]
    print(f"{args.split}: {len(ids)} traces, {len(done)} already judged, {len(todo)} to judge "
          f"with {args.model} (reasoning {args.reasoning_effort})")

    lock = threading.Lock()

    def judge_one(tid: str) -> None:
        answer, usage = call_proxy(args.model, args.reasoning_effort, judge_prompt(judge["prompt_text"], texts[tid]))
        row = {
            "trace_id": tid,
            "split": args.split,
            "verdict": 1 if answer["result"] == "Pass" else 0,
            "confidence": float(answer["confidence"]),
            "critique": answer["critique"],
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "oracle_judge_id": judge["judge_id"],
            "oracle_prompt_hash": judge["prompt_hash"],
            "usage": usage,
            "cost_usd": round(call_cost(prices, usage), 8),
        }
        with lock:
            with out_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"  {tid[:12]} {answer['result']:<4} conf {row['confidence']:.2f}  in {usage['input_tokens']} "
                  f"(cached {usage['cached_input_tokens']}) out {usage['output_tokens']} "
                  f"(reasoning {usage['reasoning_tokens']})  ${row['cost_usd']:.6f}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(judge_one, todo))


if __name__ == "__main__":
    main()
