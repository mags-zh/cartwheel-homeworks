"""Build profile/results/frontier.csv from saved development results (HW9 Part E).

One row per ``optimize.runner`` result file: the Homework 8 final run and every
Homework 9 run. A configuration is ``dominated`` when another one has an equal
or higher development score and an equal or lower cost per 100 conversations,
and is strictly better on at least one of the two; every other configuration
is ``frontier``. ``write_pass_5`` and latency are reported but do not change
the status. An empty ``write_pass_5`` means the split has no write cases.

Usage (from the repository root, offline; ``profile`` is also a standard-library
module, so run the file by path):

    uv run python profile/scripts/frontier.py [RESULT.json ...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "candidate",
    "model",
    "development_score",
    "write_pass_5",
    "cost_per_100_conversations_usd",
    "median_latency_seconds",
    "frontier_status",
]
DEFAULT_RESULTS = [
    "optimize/results/development-final-gpt-4o-mini-20261120-211400.json",
    "optimize/results/development-fewer-tokens-gpt-4o-mini-20261123-111424.json",
    "optimize/results/development-cache-before-gpt-4o-mini-20261123-112452.json",
    "optimize/results/development-cache-after-gpt-4o-mini-20261123-112652.json",
    "optimize/results/development-cache-after-run2-gpt-4o-mini-20261123-113526.json",
    "optimize/results/development-cache-after-run3-gpt-4o-mini-20261123-113616.json",
]


def dominates(a: dict, b: dict) -> bool:
    score_a, score_b = a["development_score"], b["development_score"]
    cost_a, cost_b = a["cost_per_100_conversations_usd"], b["cost_per_100_conversations_usd"]
    return score_a >= score_b and cost_a <= cost_b and (score_a > score_b or cost_a < cost_b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, nargs="*", default=[Path(p) for p in DEFAULT_RESULTS])
    parser.add_argument("--out", type=Path, default=Path("profile/results/frontier.csv"))
    args = parser.parse_args()

    rows = []
    for path in args.results:
        result = json.loads(path.read_text())
        if result["split"] != "development":
            raise SystemExit(f"{path} is a {result['split']} result; HW9 compares development runs only")
        rows.append({
            "candidate": result["candidate"],
            "model": result["model"],
            "development_score": result["score"],
            "write_pass_5": result["write_pass_5"],
            "cost_per_100_conversations_usd": result["cost_per_100_conversations_usd"],
            "median_latency_seconds": result["median_latency_seconds"],
        })
    for row in rows:
        others = [other for other in rows if other is not row]
        row["frontier_status"] = "dominated" if any(dominates(o, row) for o in others) else "frontier"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in sorted(rows, key=lambda r: r["cost_per_100_conversations_usd"]):
        print(f"{row['candidate']:<18} score {row['development_score']:<6} "
              f"${row['cost_per_100_conversations_usd']:.6f} / 100  {row['median_latency_seconds']:.2f} s  "
              f"{row['frontier_status']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
