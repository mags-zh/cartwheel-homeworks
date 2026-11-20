"""Run a frozen Homework 5 judge the way it was validated.

The judge receives the HW5 trace format (``judge_trace_text``: session
context, user turns, tool calls and results, and assistant messages in order)
and is called through one DocETL map operation with a ``critique`` and
``result`` output schema, as in ``analysis/helpers/scale.py`` (HW5) and the
Harbor verifier written by ``harbor_adapter/export.py`` (HW6).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from replay.rollout import judge_trace_text


def judge_content(judge: dict[str, Any], content: str) -> tuple[str, str]:
    """Judge one trace text. Returns ("pass" or "fail", critique with result).

    A result other than Pass or Fail counts as fail, as in the Harbor verifier.
    """
    from docetl.api import Dataset, MapOp, Pipeline, PipelineOutput, PipelineStep

    from agent.agent import LITELLM_COURSE_MODELS

    with tempfile.TemporaryDirectory(prefix="cartwheel-judge-") as directory:
        root = Path(directory)
        input_path = root / "input.json"
        output_path = root / "output.json"
        input_path.write_text(json.dumps([{"trace_id": "current", "content": content}]))
        operation = MapOp(
            name="classify_failure_mode",
            type="map",
            model=LITELLM_COURSE_MODELS.get(judge["model"], judge["model"]),
            prompt=(
                f"{judge['prompt_text']}\n\n"
                "--- Trace to evaluate ---\n"
                "{{ input.content }}\n\n"
                "First write a critique of the trace against the criterion. "
                "Use specific evidence from the provided trace. Then return result "
                "as exactly Pass when the named failure is absent, or Fail when present."
            ),
            output={"schema": {"critique": "string", "result": "string"}},
        )
        pipeline = Pipeline(
            name="cartwheel_judge",
            datasets={"traces": Dataset(type="file", path=str(input_path))},
            operations=[operation],
            steps=[
                PipelineStep(
                    name="classify", input="traces", operations=["classify_failure_mode"]
                )
            ],
            output=PipelineOutput(
                type="file", path=str(output_path), intermediate_dir=str(root)
            ),
        )
        pipeline.run()
        rows = json.loads(output_path.read_text())
    if len(rows) != 1:
        raise ValueError("judge returned an unexpected number of results")
    raw = str(rows[0].get("result", "")).strip()
    verdict = raw.lower().rstrip(".")
    if verdict not in ("pass", "fail"):
        verdict = "fail"
    return verdict, f"{rows[0].get('critique', '')}\n\nResult: {raw}"


def judge_transcript(
    judge: dict[str, Any], transcript: dict[str, Any], case_input: dict[str, Any]
) -> tuple[str, str]:
    """Judge one runtime transcript in the HW5 trace format."""
    return judge_content(judge, judge_trace_text(transcript, case_input))
