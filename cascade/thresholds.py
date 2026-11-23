"""Evaluate cascade thresholds offline from saved proxy and oracle verdicts (HW9 Part D).

The sweep tries a 0.05 grid plus every confidence value the proxy gave, so the
lowest threshold that meets a requirement is not skipped between grid points.

For a threshold T the cascade keeps the proxy's verdict when its confidence is
at or above T and uses the oracle's saved verdict otherwise. The proxy runs on
every example, so cost per verdict = mean proxy cost + (1 - proxy share) x
oracle cost per verdict. Verdicts use the HW5 convention (1 = Pass, 0 = Fail).

Usage (from the repository root, offline):

    uv run python -m cascade.thresholds sweep --oracle-cost-per-verdict USD
    uv run python -m cascade.thresholds check --oracle-cost-per-verdict USD --requirement 0.95
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "cascade" / "results"
JUDGE_PATH = REPO_ROOT / "analysis" / "state" / "judges" / "missing_policy_lookup-v1.json"
LABELS_PATH = REPO_ROOT / "analysis" / "state" / "hw5_labels" / "missing_policy_lookup.jsonl"
SPLITS_PATH = REPO_ROOT / "analysis" / "state" / "splits.json"
SWEEP = [round(0.05 * i, 2) for i in range(21)] + [1.01]


def load(split: str) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    judge = json.loads(JUDGE_PATH.read_text())
    oracle = judge["predictions"][judge["prompt_hash"]]
    labels = {}
    for line in LABELS_PATH.read_text().splitlines():
        row = json.loads(line)
        labels[row["trace_id"]] = int(row["label"])
    ids = json.loads(SPLITS_PATH.read_text())["missing_policy_lookup"][split]
    proxy = {}
    for line in (RESULTS / f"proxy-{split}.jsonl").read_text().splitlines():
        row = json.loads(line)
        proxy[row["trace_id"]] = row
    missing = [tid for tid in ids if tid not in proxy]
    if missing:
        raise SystemExit(f"{len(missing)} {split} traces have no proxy verdict yet")
    return [proxy[tid] for tid in ids], oracle, labels


def evaluate(rows: list[dict], oracle: dict, labels: dict, threshold: float, oracle_cost: float) -> dict:
    kept = [r["confidence"] >= threshold for r in rows]
    verdicts = [r["verdict"] if k else oracle[r["trace_id"]] for r, k in zip(rows, kept)]
    n = len(rows)
    oracle_agree = sum(v == oracle[r["trace_id"]] for r, v in zip(rows, verdicts))
    human_agree = sum(v == labels[r["trace_id"]] for r, v in zip(rows, verdicts))
    share = sum(kept) / n
    proxy_cost = sum(r["cost_usd"] for r in rows) / n
    return {
        "threshold": threshold,
        "agreement_with_oracle": round(oracle_agree / n, 4),
        "agreement_with_human_labels": round(human_agree / n, 4),
        "proxy_share": round(share, 4),
        "cost_per_1000_verdicts_usd": round(1000 * (proxy_cost + (1 - share) * oracle_cost), 6),
        "disagreements_with_oracle": n - oracle_agree,
        "n": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["sweep", "check"])
    parser.add_argument("--oracle-cost-per-verdict", type=float, required=True)
    parser.add_argument("--requirement", type=float, help="agreement requirement, for check")
    args = parser.parse_args()

    if args.command == "sweep":
        rows, oracle, labels = load("dev")
        thresholds = sorted(set(SWEEP) | {r["confidence"] for r in rows})
        results = [evaluate(rows, oracle, labels, t, args.oracle_cost_per_verdict) for t in thresholds]
        fields = ["threshold", "agreement_with_oracle", "agreement_with_human_labels",
                  "proxy_share", "cost_per_1000_verdicts_usd"]
        with (RESULTS / "thresholds.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"oracle alone: ${1000 * args.oracle_cost_per_verdict:.4f} per 1,000 verdicts")
        for r in results:
            print(f"  T {r['threshold']:<5} oracle {r['agreement_with_oracle']:.3f} "
                  f"({r['disagreements_with_oracle']} disagree)  human {r['agreement_with_human_labels']:.3f}  "
                  f"proxy share {r['proxy_share']:.3f}  ${r['cost_per_1000_verdicts_usd']:.4f} / 1,000")
        return

    if args.requirement is None:
        raise SystemExit("check needs --requirement")
    threshold = json.loads((RESULTS / "chosen_threshold.json").read_text())["threshold"]
    rows, oracle, labels = load("test")
    result = evaluate(rows, oracle, labels, threshold, args.oracle_cost_per_verdict)
    result["agreement_requirement"] = args.requirement
    result["meets_requirement"] = result["agreement_with_oracle"] >= args.requirement
    result["oracle_cost_per_1000_verdicts_usd"] = round(1000 * args.oracle_cost_per_verdict, 6)
    (RESULTS / "check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
