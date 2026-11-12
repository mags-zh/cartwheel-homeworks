from __future__ import annotations

import pytest

from replay.harness import ReplayInfraError, replay_case, summarize_rollouts
from replay.rollout import _one_check, judge_trace_text


def test_replay_resets_before_each_run_and_retries_only_infrastructure() -> None:
    resets = 0
    calls = 0

    def reset() -> None:
        nonlocal resets
        resets += 1

    def runner() -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ReplayInfraError("temporary service error")
        return {"passed": calls % 2 == 0}

    records = replay_case(runner, reset, n=2, max_infra_retries=1)

    assert resets == 3
    assert calls == 3
    assert [record["rollout"] for record in records] == [0, 1]
    assert [record["passed"] for record in records] == [True, False]


def test_replay_does_not_retry_a_completed_failure() -> None:
    calls = 0

    def runner() -> dict:
        nonlocal calls
        calls += 1
        return {"passed": False}

    records = replay_case(runner, lambda: None, n=2)

    assert calls == 2
    assert len(records) == 2


def test_rollout_summary_is_deterministic() -> None:
    records = [
        {"passed": True, "steps": 2},
        {"passed": False, "failure_modes": ["judge:a"], "steps": 4},
        {
            "passed": False,
            "failure_modes": ["judge:a", "check:b"],
            "steps": 6,
        },
    ]

    first = summarize_rollouts(records, bootstrap_iterations=100, seed=7)
    second = summarize_rollouts(records, bootstrap_iterations=100, seed=7)

    assert first == second
    assert first["failures"] == 2
    assert first["failure_rate"] == pytest.approx(2 / 3)
    assert first["mode_counts"] == {"judge:a": 2, "check:b": 1}
    assert first["steps"] == {"min": 2, "median": 4, "max": 6}


def test_judge_trace_text_matches_the_hw5_judge_input() -> None:
    transcript = {
        "turns": [
            {
                "user": "Where is my order?",
                "reply": "Let me check.\nIt shipped today.",
                "tool_calls": [
                    {
                        "name": "get_order",
                        "args": {"order_id": 42},
                        "result": {"status": "shipped", "ok": True},
                    }
                ],
                "events": [
                    {"role": "assistant", "text": "Let me check."},
                    {"role": "tool_call", "name": "get_order", "args": {"order_id": 42}},
                    {
                        "role": "tool_result",
                        "name": "get_order",
                        "result": {"status": "shipped", "ok": True},
                    },
                    {"role": "assistant", "text": "It shipped today."},
                ],
            }
        ]
    }
    case_input = {"role": "shopper", "user_id": 63, "message": "Where is my order?"}

    assert judge_trace_text(transcript, case_input).splitlines() == [
        "system: Session context: User role: shopper; User id: 63; Store id: None",
        "user: Where is my order?",
        "assistant: Let me check.",
        'tool_call: get_order({"order_id": 42})',
        'tool_result: get_order returned {"ok": true, "status": "shipped"}',
        "assistant: It shipped today.",
    ]


def test_judge_trace_text_names_tools_for_transcripts_without_events() -> None:
    transcript = {
        "turns": [
            {
                "user": "Where is my order?",
                "reply": "It shipped today.",
                "tool_calls": [
                    {"name": "get_order", "args": {"order_id": 42}, "result": {"ok": True}}
                ],
            }
        ]
    }

    assert judge_trace_text(transcript).splitlines() == [
        "user: Where is my order?",
        'tool_call: get_order({"order_id": 42})',
        'tool_result: get_order returned {"ok": true}',
        "assistant: It shipped today.",
    ]


def test_tool_result_not_error_fails_only_on_the_named_error() -> None:
    def transcript(*results: dict) -> dict:
        calls = [
            {"name": "search_products", "args": {"store": "1"}, "result": result}
            for result in results
        ]
        return {"turns": [{"reply": "", "tool_calls": calls}], "final_reply": ""}

    check = {"check": "tool_result_not_error", "name": "search_products", "error": "not_found"}
    not_found = {"ok": False, "error": "not_found", "reason": "no store named '1'"}
    invalid = {"ok": False, "error": "invalid_argument", "reason": "empty query"}

    assert _one_check(check, transcript({"ok": True, "products": []}), None)[0] is True
    assert _one_check(check, transcript(invalid), None)[0] is True
    assert _one_check(check, transcript({"ok": True}, not_found), None)[0] is False
    assert _one_check(check, transcript(), None)[0] is True
