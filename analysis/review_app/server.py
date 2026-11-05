"""Cartwheel conversation review app.

Serves the HW4 interface and joins the selected review sample with the
scenario answer keys and the raw Langfuse traces. Traces are loaded live from
Langfuse at startup; the HW3 export is used only when Langfuse is not
configured or cannot be reached (or with ``--trace-source export``). Run from
the repository root with:

    uv run --env-file .env python analysis/review_app/server.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agent import db
from analysis.helpers import langfuse_io
from analysis.helpers.normalization import normalize_trace
from scenarios.export_langfuse import export_scenario_traces


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
REPO_ROOT = ANALYSIS_DIR.parent
DEFAULT_STATE_DIR = ANALYSIS_DIR / "state"
DEFAULT_TRACE_EXPORT = REPO_ROOT / "traces" / "support_traces.json"
DEFAULT_SCENARIOS = REPO_ROOT / "scenarios" / "support_scenarios.jsonl"

STATE_DEFAULTS: dict[str, Any] = {
    "samples.json": [],
    "annotations.json": {"annotations": []},
    "patterns.json": {"modes": []},
    "suggestions.json": [],
    "sample_manifest.json": {},
}

MODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _annotation_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("annotations", [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _pattern_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    modes = value.get("modes")
    if isinstance(modes, list):
        return [mode for mode in modes if isinstance(mode, dict) and mode.get("name")]
    return [
        {"name": name, **details}
        for name, details in value.items()
        if isinstance(details, dict)
    ]


class Catalog:
    """In-memory conversation catalog built from live Langfuse traces.

    Only traces whose ``cartwheel.scenario_id`` is a final ``support-*``
    scenario are kept, so pilot, smoke and ``tmp-*`` side-test runs stay out.
    """

    def __init__(
        self, trace_export: Path, scenarios_path: Path, trace_source: str = "langfuse"
    ) -> None:
        self.trace_export = trace_export
        self.scenarios_path = scenarios_path
        self.requested_source = trace_source
        self.source = trace_source
        self.source_note = ""
        self.by_canonical_id: dict[str, dict[str, Any]] = {}
        self.by_scenario_id: dict[str, dict[str, Any]] = {}
        self._load()
        self.loaded_at = _utcnow()

    def summary(self) -> dict[str, Any]:
        return {
            "trace_source": self.source,
            "trace_source_note": self.source_note,
            "traces_loaded_at": self.loaded_at,
            "catalog_conversations": len(self.by_canonical_id),
        }

    def _raw_traces(self, scenario_ids: set[str]) -> list[Any]:
        if self.requested_source == "langfuse":
            if not langfuse_io.is_configured():
                self.source_note = "Langfuse is not configured (start with --env-file .env)"
            else:
                try:
                    from langfuse import Langfuse

                    return export_scenario_traces(scenario_ids, Langfuse())
                except Exception as exc:  # live Langfuse unreachable or failing
                    self.source_note = f"Langfuse fetch failed: {exc}"
            self.source = "export"
        export = _read_json(self.trace_export, {})
        raw_traces = export.get("traces", []) if isinstance(export, dict) else export
        if not isinstance(raw_traces, list):
            raise ValueError("trace export must contain a traces list")
        return raw_traces

    def _load(self) -> None:
        scenario_rows = [
            json.loads(line)
            for line in self.scenarios_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        scenarios = {
            str(row["id"]): row
            for row in scenario_rows
            if str(row.get("id", "")).startswith("support-")
        }
        scenario_store_ids: dict[str, int | None] = {}
        with db.connection() as conn:
            for scenario_id, scenario in scenarios.items():
                scenario_tuple = scenario.get("tuple", {})
                store_id = scenario_tuple.get("store_id")
                order_id = scenario_tuple.get("order_id")
                product_id = scenario_tuple.get("product_id")
                user_id = scenario_tuple.get("user_id")
                if store_id is None and order_id is not None:
                    order = db.get_order(conn, int(order_id))
                    store_id = order.store_id if order else None
                if store_id is None and product_id is not None:
                    product = conn.execute(
                        "SELECT store_id FROM products WHERE id = ?",
                        (int(product_id),),
                    ).fetchone()
                    store_id = product["store_id"] if product else None
                if (
                    store_id is None
                    and scenario_tuple.get("role") == "merchant"
                    and user_id is not None
                ):
                    user = db.get_user(conn, int(user_id))
                    store_id = user.store_id if user else None
                scenario_store_ids[scenario_id] = (
                    int(store_id) if store_id is not None else None
                )

        raw_traces = self._raw_traces(set(scenarios))

        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw in raw_traces:
            if not isinstance(raw, dict):
                continue
            normalized = normalize_trace(raw)
            html_path = raw.get("htmlPath")
            if html_path and not normalized.get("permalink"):
                normalized["permalink"] = (
                    str(html_path)
                    if str(html_path).startswith(("http://", "https://"))
                    else f"http://localhost:3000{html_path}"
                )
            scenario_id = normalized.get("meta", {}).get("scenario_id")
            if scenario_id not in scenarios:
                continue
            session_id = (
                normalized.get("meta", {}).get("session_id")
                or raw.get("sessionId")
                or normalized["trace_id"]
            )
            grouped.setdefault(str(session_id), []).append(normalized)

        for session_id, turns in grouped.items():
            turns.sort(key=lambda turn: turn.get("timestamp") or "")
            first = turns[0]
            scenario_id = str(first["meta"]["scenario_id"])
            scenario = scenarios[scenario_id]
            conversation = {
                "trace_id": first["trace_id"],
                "canonical_trace_id": first["trace_id"],
                "scenario_id": scenario_id,
                "session_id": session_id,
                "timestamp": first.get("timestamp"),
                "permalink": first.get("permalink"),
                "meta": first.get("meta", {}),
                "scenario_group": scenario.get("scenario_group"),
                "data_quality_case_id": scenario.get("data_quality_case_id"),
                "tuple": scenario.get("tuple", {}),
                "store_id": scenario_store_ids.get(scenario_id),
                "opening_message": scenario.get("opening_message"),
                "followups": scenario.get("followups", []),
                "expected": scenario.get("expected", {}),
                "models": list(
                    dict.fromkeys(
                        model
                        for turn in turns
                        for model in turn.get("models", [])
                    )
                ),
                "turns": [
                    {
                        "number": number,
                        "trace_id": turn["trace_id"],
                        "timestamp": turn.get("timestamp"),
                        "permalink": turn.get("permalink"),
                        "messages": turn.get("trace", []),
                        "features": turn.get("features", {}),
                        "observations": turn.get("observations", []),
                    }
                    for number, turn in enumerate(turns, start=1)
                ],
            }
            self.by_canonical_id[first["trace_id"]] = conversation
            self.by_scenario_id[scenario_id] = conversation

    def selected(self, samples: Any) -> list[dict[str, Any]]:
        if not isinstance(samples, list):
            return []
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for position, sample in enumerate(samples):
            if isinstance(sample, str):
                sample = {"trace_id": sample}
            if not isinstance(sample, dict):
                continue
            trace_id = str(sample.get("trace_id") or sample.get("id") or "")
            scenario_id = str(
                sample.get("scenario_id")
                or sample.get("meta", {}).get("scenario_id")
                or ""
            )
            conversation = (
                self.by_canonical_id.get(trace_id)
                or self.by_scenario_id.get(scenario_id)
            )
            if not conversation or conversation["trace_id"] in seen:
                continue
            seen.add(conversation["trace_id"])
            output.append(
                {
                    **conversation,
                    "reason": sample.get("reason"),
                    "batch": sample.get("batch") or sample.get("batch_id"),
                    "sample_position": position,
                }
            )
        return output


class ReviewApplication:
    def __init__(self, state_dir: Path, catalog: Catalog) -> None:
        self.state_dir = state_dir
        self.catalog = catalog

    def state_path(self, name: str) -> Path:
        return self.state_dir / name

    def get_state(self, name: str) -> Any:
        return _read_json(self.state_path(name), STATE_DEFAULTS[name])

    def put_state(self, name: str, value: Any) -> None:
        _write_json(self.state_path(name), value)

    def samples(self) -> list[dict[str, Any]]:
        return self.catalog.selected(self.get_state("samples.json"))

    def reload_traces(self) -> dict[str, Any]:
        # Build the new catalog fully before swapping, so requests served
        # during the fetch keep reading the old one.
        old = self.catalog
        self.catalog = Catalog(old.trace_export, old.scenarios_path, old.requested_source)
        return {**self.catalog.summary(), "selected_conversations": len(self.samples())}

    def labels(self) -> list[dict[str, Any]]:
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        labels_dir = self.state_dir / "labels"
        if not labels_dir.exists():
            return []
        for path in sorted(labels_dir.glob("*.jsonl")):
            mode = path.stem
            for row in _read_jsonl(path):
                trace_id = str(row.get("trace_id", ""))
                if trace_id and row.get("label") in (0, 1):
                    latest[(trace_id, mode)] = {**row, "mode": mode}
        return sorted(latest.values(), key=lambda row: (row["trace_id"], row["mode"]))

    def save_label(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        trace_id = str(value.get("trace_id", ""))
        mode = str(value.get("mode", ""))
        label = value.get("label")
        if trace_id not in self.catalog.by_canonical_id:
            raise ValueError("label trace_id must be a canonical conversation trace id")
        if not MODE_RE.fullmatch(mode):
            raise ValueError("mode must be a snake_case identifier")
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")

        previous = next(
            (
                row
                for row in reversed(self.labels())
                if row["trace_id"] == trace_id and row["mode"] == mode
            ),
            None,
        )
        record = {
            "id": f"label-{int(datetime.now().timestamp() * 1000)}",
            "trace_id": trace_id,
            "mode": mode,
            "label": int(label),
            "evidence": value.get("evidence", ""),
            "note": value.get("note", ""),
            "source": "human",
            "created_at": _utcnow(),
        }
        if previous:
            record["replaces"] = previous.get("id")

        if langfuse_io.is_configured():
            langfuse_io.write_label_score(
                trace_id=trace_id,
                mode=mode,
                label=int(label),
                comment=str(value.get("note") or value.get("evidence") or ""),
            )
        _append_jsonl(self.state_dir / "labels" / f"{mode}.jsonl", record)
        return record


APP: ReviewApplication


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return None
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            self._file(HERE / "index.html", "text/html; charset=utf-8")
        elif path == "/api/samples":
            self._json(APP.samples())
        elif path == "/api/annotations":
            self._json(APP.get_state("annotations.json"))
        elif path == "/api/patterns":
            self._json(APP.get_state("patterns.json"))
        elif path == "/api/suggestions":
            self._json(APP.get_state("suggestions.json"))
        elif path == "/api/manifest":
            self._json(APP.get_state("sample_manifest.json"))
        elif path == "/api/labels":
            self._json(APP.labels())
        elif path == "/api/health":
            self._json(
                {
                    "ok": True,
                    **APP.catalog.summary(),
                    "selected_conversations": len(APP.samples()),
                    "langfuse_configured": langfuse_io.is_configured(),
                }
            )
        else:
            self._json({"error": f"unknown path: {path}"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            value = self._body()
            if path == "/api/annotations":
                APP.put_state("annotations.json", {"annotations": _annotation_list(value)})
                self._json({"ok": True, "count": len(_annotation_list(value))})
            elif path == "/api/patterns":
                APP.put_state("patterns.json", value)
                self._json({"ok": True, "count": len(_pattern_list(value))})
            elif path == "/api/suggestions":
                if not isinstance(value, list):
                    raise ValueError("suggestions must be a list")
                APP.put_state("suggestions.json", value)
                self._json({"ok": True, "count": len(value)})
            elif path == "/api/samples":
                if not isinstance(value, list):
                    raise ValueError("samples must be a list")
                APP.put_state("samples.json", value)
                self._json({"ok": True, "count": len(APP.samples())})
            elif path == "/api/labels":
                self._json({"ok": True, "label": APP.save_label(value)})
            elif path == "/api/reload":
                self._json({"ok": True, **APP.reload_traces()})
            else:
                self._json({"error": f"cannot POST to {path}"}, 404)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # pragma: no cover - live Langfuse failures
            self._json({"error": str(exc)}, 502)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--trace-source",
        choices=("langfuse", "export"),
        default="langfuse",
        help="where trace content comes from; langfuse falls back to the export",
    )
    parser.add_argument("--trace-export", type=Path, default=DEFAULT_TRACE_EXPORT)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    args = parser.parse_args()

    os.environ.setdefault("CARTWHEEL_ANALYSIS_STATE", str(args.state_dir))
    global APP
    APP = ReviewApplication(
        args.state_dir.resolve(),
        Catalog(
            args.trace_export.resolve(), args.scenarios.resolve(), args.trace_source
        ),
    )
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"review app on http://{args.host}:{args.port}")
    source = APP.catalog.source
    if source == "export":
        source += f" ({APP.catalog.trace_export})"
    if APP.catalog.source_note:
        source += f"; {APP.catalog.source_note}"
    print(f"Trace source: {source}")
    print(
        f"{len(APP.catalog.by_canonical_id)} conversations catalogued; "
        f"{len(APP.samples())} selected"
    )
    print(
        "Langfuse score writes: "
        + ("enabled" if langfuse_io.is_configured() else "disabled (start with --env-file .env)")
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
