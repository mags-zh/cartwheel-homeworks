"""Build profile/results/costs.csv: model cost per call site (HW9 Part A).

Token counts are the ones the LLM API returned:

- HW3 agent conversations: the ``gen_ai.usage.*`` attributes of each generation
  span in the Langfuse export. Langfuse's own ``usageDetails["input"]`` leaves
  out cached tokens, so it is not used.
- HW8 agent development runs: ``input_tokens``, ``cached_input_tokens`` and
  ``output_tokens`` in the ``optimize.runner`` result files.
- Judge runs (HW5, HW7, HW8): DocETL's "Token Usage" summary in each run's log.

Costs use the prices in ``optimize/config.json``; cached input tokens are billed
at ``cached_input`` when the model lists one, as ``optimize.runner`` does.

Usage (from the repository root; ``profile`` is also a standard-library module,
so run the file by path rather than with ``-m``):

    uv run python profile/scripts/costs.py \
        --hw5-logs LOG [LOG ...] --hw7-logs LOG [LOG ...] --hw8-judge-logs LOG [LOG ...]

Pass each judge run's log once; a log that repeats another run's output (e.g. a
CI job log next to the run's own monitor log) would count that run twice.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

FIELDS = [
    "cost_category",
    "call_site",
    "purpose",
    "model",
    "calls",
    "input_tokens",
    "output_tokens",
    "cost_usd",
]

DOCETL_USAGE = re.compile(r"([\w.\-]+): ([\d,]+) input(?: \(([\d,]+) cached\))?, ([\d,]+) output")
DOCETL_BATCH = re.compile(r"on all documents: +0%\|[^|]*\| 0/(\d+)")
DATED_MODEL = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def number(text: str | None) -> int:
    return int(text.replace(",", "")) if text else 0


def cost_usd(prices: dict, model: str, input_tokens: int, cached: int, output_tokens: int) -> float:
    p = prices[model]
    cached_price = p.get("cached_input")
    billed_cached = cached if cached_price is not None else 0
    return (
        (input_tokens - billed_cached) * p["input"]
        + billed_cached * (cached_price or 0.0)
        + output_tokens * p["output"]
    ) / 1_000_000


def trace_usage(path: Path) -> dict:
    export = json.loads(path.read_text())
    totals = {"models": set(), "calls": 0, "input": 0, "cached": 0, "output": 0}
    for trace in export["traces"]:
        for obs in trace["observations"]:
            if obs["type"] != "GENERATION":
                continue
            attrs = obs["metadata"]["attributes"]
            totals["models"].add(DATED_MODEL.sub("", obs["model"]))
            totals["calls"] += 1
            totals["input"] += int(attrs["gen_ai.usage.input_tokens"])
            totals["cached"] += int(attrs.get("gen_ai.usage.cache_read.input_tokens", 0))
            totals["output"] += int(attrs["gen_ai.usage.output_tokens"])
    totals["traces"] = len(export["traces"])
    totals["scenarios"] = export["scenario_count"]
    return totals


def runner_usage(paths: list[Path]) -> dict:
    totals = {"models": set(), "runs": len(paths), "calls": 0, "input": 0, "cached": 0, "output": 0}
    for path in paths:
        result = json.loads(path.read_text())
        totals["models"].add(result["model"])
        totals["calls"] += int(result["evaluated_case_runs"])
        totals["input"] += int(result["input_tokens"])
        totals["cached"] += int(result.get("cached_input_tokens", 0))
        totals["output"] += int(result["output_tokens"])
    return totals


def docetl_usage(paths: list[Path]) -> dict:
    """Sum DocETL's per-pipeline token summaries.

    A pipeline answered from DocETL's cache prints no token summary and makes
    no API call, so its documents are not counted as calls.
    """
    totals = {"models": set(), "calls": 0, "input": 0, "cached": 0, "output": 0}
    for path in paths:
        documents = 0
        for match in re.finditer(f"{DOCETL_BATCH.pattern}|{DOCETL_USAGE.pattern}", path.read_text(errors="replace")):
            if match.group(1):
                documents = int(match.group(1))
                continue
            model, input_tokens, cached, output_tokens = match.group(2, 3, 4, 5)
            totals["models"].add(model)
            totals["calls"] += documents
            totals["input"] += number(input_tokens)
            totals["cached"] += number(cached)
            totals["output"] += number(output_tokens)
            documents = 0
    return totals


def row(category: str, call_site: str, purpose: str, usage: dict, prices: dict) -> dict:
    if len(usage["models"]) != 1:
        raise ValueError(f"{call_site}: expected one model, found {sorted(usage['models'])}")
    model = next(iter(usage["models"]))
    cached_note = f"; {usage['cached']:,} of the input tokens were cached"
    return {
        "cost_category": category,
        "call_site": call_site,
        "purpose": purpose + cached_note,
        "model": model,
        "calls": usage["calls"],
        "input_tokens": usage["input"],
        "output_tokens": usage["output"],
        "cost_usd": round(cost_usd(prices, model, usage["input"], usage["cached"], usage["output"]), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--traces", type=Path, default=Path("traces/support_traces.json"))
    parser.add_argument("--hw8-results", type=Path, nargs="+",
                        default=sorted(Path("optimize/results").glob("development-*.json")))
    parser.add_argument("--hw5-logs", type=Path, nargs="+", required=True)
    parser.add_argument("--hw7-logs", type=Path, nargs="+", required=True)
    parser.add_argument("--hw8-judge-logs", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=Path("optimize/config.json"))
    parser.add_argument("--out", type=Path, default=Path("profile/results/costs.csv"))
    args = parser.parse_args()

    prices = json.loads(args.config.read_text())["prices_per_million_tokens_usd"]
    traces = trace_usage(args.traces)
    hw8_agent = runner_usage(args.hw8_results)
    rows = [
        row(
            "customer_conversation",
            "server/app.py:204",
            f"HW3 trace dataset: {traces['scenarios']} scenarios ({traces['traces']} traces) "
            "answered by the agent through the HTTP server",
            traces,
            prices,
        ),
        row(
            "customer_conversation",
            "replay/rollout.py:221",
            f"HW8 development runs of optimize.runner: {hw8_agent['runs']} runs x 8 cases, "
            "agent answering the case's user messages",
            hw8_agent,
            prices,
        ),
        row(
            "judge",
            "analysis/helpers/scale.py:259",
            "HW5 missing_policy_lookup judge: development runs of prompts v0, v1, v2 and the test run of v1",
            docetl_usage(args.hw5_logs),
            prices,
        ),
        row(
            "judge",
            "monitoring/run_judges.py:127",
            "HW7 monitor with missing_policy_lookup-v1: before and after periods and one manual CI run",
            docetl_usage(args.hw7_logs),
            prices,
        ),
        row(
            "judge",
            "replay/judge.py:61",
            "HW8 development scoring with missing_policy_lookup-v1 in optimize.runner; "
            "logged calls only: the 3 runs before the scorer fix (9 calls) saved no token counts",
            docetl_usage(args.hw8_judge_logs),
            prices,
        ),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['cost_category']:<22} {r['call_site']:<30} calls {r['calls']:>4}  "
              f"in {r['input_tokens']:>8,}  out {r['output_tokens']:>6,}  ${r['cost_usd']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
