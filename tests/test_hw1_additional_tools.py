"""Tests for the HW1 additional tools in agent/tools.py. Offline, no API keys."""

from __future__ import annotations

from agent import db, tools
from agent.agent import TOOLS_BY_ROLE
from agent.auth import AuthContext

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
MERCHANT_STORE_1 = AuthContext(user_id=9001, role="merchant", store_id=1)
SUPPORT = AuthContext(user_id=9501, role="support")


def _owners(order_ids: list[int]) -> set[int]:
    with db.connection() as conn:
        return {db.get_order(conn, order_id).user_id for order_id in order_ids}


def test_search_customer_orders_finds_older_order_for_one_customer(world: dict) -> None:
    # find_order as support shows the 5 newest vase orders across all
    # customers, which excludes customer 1's May order 3980.
    result = tools.search_customer_orders(SUPPORT, 1, "Heavy-Duty Vase")
    assert result["ok"] is True
    ids = [o["order_id"] for o in result["orders"]]
    assert ids == [4455, 4127, 3980]
    assert _owners(ids) == {1}
    assert all(o["product_title"] == "Heavy-Duty Vase" for o in result["orders"])
    assert result["total_matches"] == 3
    assert result["truncated"] is False
    assert "note" not in result


def test_search_customer_orders_reports_truncation(world: dict) -> None:
    with db.connection() as conn:
        total = len(db.list_order_search_candidates(conn, user_id=1))
    assert total > tools.CUSTOMER_ORDER_LIMIT

    result = tools.search_customer_orders(SUPPORT, 1)
    assert result["count"] == tools.CUSTOMER_ORDER_LIMIT
    assert result["total_matches"] == total
    assert result["truncated"] is True
    assert "order number" in result["note"]


def test_search_customer_orders_is_support_only(world: dict) -> None:
    for ctx in (SHOPPER_1, MERCHANT_STORE_1):
        result = tools.search_customer_orders(ctx, 1)
        assert result["ok"] is False
        assert result["error"] == "permission_denied"
        assert "orders" not in result

    support_tools = [t.name for t in TOOLS_BY_ROLE["support"]]
    assert "search_customer_orders" in support_tools
    for role in ("shopper", "merchant"):
        assert "search_customer_orders" not in [t.name for t in TOOLS_BY_ROLE[role]]


def test_search_customer_orders_errors(world: dict) -> None:
    assert tools.search_customer_orders(SUPPORT, 999_999)["error"] == "not_found"
    # Staff accounts are not customers.
    assert tools.search_customer_orders(SUPPORT, 9501)["error"] == "not_found"

    empty = tools.search_customer_orders(SUPPORT, 1, "zzzznonexistent9999")
    assert empty["ok"] is True
    assert empty["orders"] == []
    assert empty["total_matches"] == 0


def test_check_return_eligibility_store_override(world: dict) -> None:
    # Juniper Home Goods order delivered 19 days before the world date:
    # inside the platform 30 days, outside the store's 14-day override.
    result = tools.check_return_eligibility(AuthContext(user_id=38, role="shopper"), 3255)
    assert result["ok"] is True
    assert result["eligible"] is False
    assert result["days_since_delivery"] == 19
    assert result["return_window_days"] == 14
    assert result["window_source"] == "store_override"
    assert result["policy_id"] == "store-juniper-home-goods-policy"
    assert result["window_closes_on"] == "2026-06-26"
    assert "14 days" in result["reason"]
    assert "store-juniper-home-goods-policy" in result["reason"]


def test_check_return_eligibility_platform_window(world: dict) -> None:
    closed = tools.check_return_eligibility(SHOPPER_1, 3980)
    assert closed["eligible"] is False
    assert closed["days_since_delivery"] == 45
    assert closed["return_window_days"] == 30
    assert closed["window_source"] == "platform"
    assert closed["policy_id"] == "cw-returns"
    assert "cw-returns" in closed["reason"]

    open_ = tools.check_return_eligibility(SHOPPER_1, 4127)
    assert open_["eligible"] is True
    assert open_["days_since_delivery"] == 12
    assert open_["window_closes_on"] == "2026-07-19"


def test_check_return_eligibility_matches_seeded_flags(world: dict) -> None:
    with db.connection() as conn:
        orders = db.list_order_search_candidates(conn, all_orders=True)
    results = {o.id: tools.check_return_eligibility(SUPPORT, o.id) for o in orders}
    flagged = {oid for oid, r in results.items() if "data_issue" in r}
    assert flagged == {8001, 8002, 8003}
    mismatches = [
        o.id for o in orders if o.id not in flagged and results[o.id]["eligible"] != o.refund_eligible
    ]
    assert mismatches == []


def test_check_return_eligibility_flags_seeded_data_defects(world: dict) -> None:
    expected = {
        8001: "reversed_chronology",
        8002: "missing_delivery_date",
        8003: "store_mismatch",
    }
    for order_id, code in expected.items():
        result = tools.check_return_eligibility(SUPPORT, order_id)
        assert result["ok"] is True
        assert result["eligible"] is None
        assert result["data_issue"] == code
        assert "escalate" in result["reason"]
        assert "window_closes_on" not in result


def test_check_return_eligibility_after_refund(world_copy) -> None:
    from agent.agent import issue_refund_logic

    refund = issue_refund_logic(SHOPPER_1, 4127, 84.0, "test")
    assert refund["status"] == "auto_approved"
    result = tools.check_return_eligibility(SHOPPER_1, 4127)
    assert result["status"] == "refunded"
    assert result["eligible"] is False
    assert "already been refunded" in result["reason"]


def test_check_return_eligibility_scope(world: dict) -> None:
    other_store = AuthContext(user_id=9002, role="merchant", store_id=2)
    denied = tools.check_return_eligibility(other_store, 4127)
    assert denied["error"] == "permission_denied"
    assert set(denied) == {"ok", "error", "reason"}
    assert tools.check_return_eligibility(SHOPPER_1, 999_999)["error"] == "not_found"

    for role in ("shopper", "merchant", "support"):
        assert "check_return_eligibility" in [t.name for t in TOOLS_BY_ROLE[role]]
