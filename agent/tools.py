"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from rapidfuzz import fuzz

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.config import load_facts
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch
from seed.eligibility import effective_return_window_days, is_refund_eligible

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20
FIND_ORDER_LIMIT = 5
FIND_ORDER_MIN_SCORE = 80
CUSTOMER_ORDER_LIMIT = 10
_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id. Risk tier: read.

    Every role may read every policy doc (the corpus is public help-center
    content), so this tool needs no permission check.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.

    Implementation notes:
        agent.helpcenter.load_policy_docs() returns every parsed doc.
    """
    for doc in load_policy_docs():
        if doc.policy_id == policy_id:
            return {
                "ok": True,
                "policy_id": doc.policy_id,
                "title": doc.title,
                "audience": doc.audience,
                "body": doc.body,
            }
    return {
        "ok": False,
        "error": "not_found",
        "reason": f"no policy doc with id {policy_id!r}; search_help_center lists valid ids",
    }


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog. Risk tier: read.

    Every role may search products. Matching is deterministic keyword
    matching, not semantic search: a product matches when every whitespace
    token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.

    Implementation notes:
        agent.db.list_products(conn, store_id) gives the candidate set.
        Use `with db.connection() as conn:` to close the database automatically.
    """
    terms = query.lower().split()
    if not terms:
        return {"ok": False, "error": "invalid_argument", "reason": "query must not be empty"}
    if max_price_usd is not None and max_price_usd <= 0:
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": f"max_price_usd must be positive, got {max_price_usd}",
        }
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    with db.connection() as conn:
        store_id = None
        if store:
            found = db.get_store_by_name(conn, store)
            if found is None:
                return {"ok": False, "error": "not_found", "reason": f"no store named {store!r}"}
            store_id = found.id
        products = db.list_products(conn, store_id)

    matches = [
        p
        for p in products
        if all(term in f"{p.title} {p.description}".lower() for term in terms)
        and (max_price_usd is None or p.price_usd <= max_price_usd)
    ]
    matches.sort(key=lambda p: (p.price_cents, p.id))
    results = [
        {"product_id": p.id, "store_id": p.store_id, "title": p.title, "price_usd": p.price_usd}
        for p in matches[:limit]
    ]
    return {"ok": True, "products": results, "count": len(results)}


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders in the caller's own scope. Risk tier: read.

    Role behavior, straight from the access matrix in SPEC.md:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders (ctx.store_id).
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead, so return {"ok": False,
          "error": "invalid_argument", "reason": ...} saying exactly that.

    Returns:
        For shopper and merchant: {"ok": True, "orders": [...],
        "count": <len(orders)>} where each order is
        agent.db.Order.to_public_dict() and the list holds at most
        DEFAULT_ORDER_LIMIT orders, newest first (agent.db.list_orders_for_user
        and list_orders_for_store already sort and limit this way).

    Implementation notes:
        No permission check is needed beyond the role dispatch, because the
        scope is baked into which query you run. That is the point of the
        tool: the model cannot ask for someone else's orders through it.
    """
    if ctx.role not in ("shopper", "merchant"):
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": (
                f"role '{ctx.role}' has no orders of its own; "
                "look up a specific order with get_order instead"
            ),
        }
    with db.connection() as conn:
        if ctx.role == "shopper":
            orders = db.list_orders_for_user(conn, ctx.user_id, DEFAULT_ORDER_LIMIT)
        else:
            orders = db.list_orders_for_store(conn, ctx.store_id, DEFAULT_ORDER_LIMIT)
    public = [order.to_public_dict() for order in orders]
    return {"ok": True, "orders": public, "count": len(public)}


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order. Risk tier: write.

    This is the homework's write tool, and it must enforce two independent
    rules in this order:

    1. The access matrix (scope): use agent.auth.can_cancel_order. Shoppers
       may cancel only their own orders, merchants only their own store's
       orders, support any order. On failure return
       agent.auth.permission_denied(...) with a reason naming the role and
       the order id. Scope is checked before the status rule so that an
       out-of-scope caller learns nothing about the order's state.
    2. The pre-shipment rule (facts.yaml `cancel_cutoff`): only orders whose
       status is exactly "placed" can be cancelled, for every role. If the
       order is in scope but its status is not "placed", return
       {"ok": False, "error": "not_eligible", "reason": ...} that names the
       current status and states that orders can be cancelled only before
       shipment.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.

    Implementation notes:
        Fetch with agent.db.get_order. Note the argument order of
        can_cancel_order(ctx, order_user_id, order_store_id).

    The Module 4 kill switch is checked first (before the scope and
    status rules and before your code), so that a paused write tool touches
    nothing. It is provided; the default ("off") returns None and falls
    through to your implementation.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not cancel order #{order_id}"
            )
        if order.status != "placed":
            return {
                "ok": False,
                "error": "not_eligible",
                "reason": (
                    f"order #{order_id} has status '{order.status}'; orders can be "
                    "cancelled only before shipment, while the status is 'placed'"
                ),
            }
        db.set_order_status(conn, order_id, "cancelled")
    return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search the caller's orders by product name. Risk tier: read.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and searches the authenticated user's orders for products whose name
    matches. Use fuzzy string matching (e.g., thefuzz.fuzz.partial_ratio
    or case-insensitive substring matching) to find orders whose product name is close to the
    query.

    Access rules: a shopper searches only the shopper's own orders, a
    merchant searches orders from the merchant's store, and support staff
    can search any orders. Use agent.db.list_order_search_candidates with
    user_id=ctx.user_id for shoppers, store_id=ctx.store_id for merchants,
    or all_orders=True only for support. Derive the scope from ctx, never
    from the query; reject unsupported roles or missing required identity.
    Use agent.db.list_products to map product IDs to product titles.

    The helper returns the complete authorised scope, newest first with
    order ID descending as the tie-breaker. Match product names first,
    preserve that order, then return at most five matches. Do not search
    only the 20 most recent orders. Convert matches with to_public_dict().

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), each as the dict returned by agent.db. If no orders
        match, return {"ok": True, "orders": []}.
    """
    if ctx.role == "shopper":
        scope: dict[str, Any] = {"user_id": ctx.user_id}
    elif ctx.role == "merchant" and ctx.store_id is not None:
        scope = {"store_id": ctx.store_id}
    elif ctx.role == "support":
        scope = {"all_orders": True}
    else:
        return permission_denied(
            f"role '{ctx.role}' (user {ctx.user_id}) has no order search scope"
        )

    query_words = _words(query)
    if not query_words:
        return {"ok": True, "orders": []}
    with db.connection() as conn:
        titles = {p.id: p.title for p in db.list_products(conn)}
        candidates = db.list_order_search_candidates(conn, **scope)

    matching_products = _matching_product_ids(titles, query_words)
    matches = [o.to_public_dict() for o in candidates if o.product_id in matching_products]
    return {"ok": True, "orders": matches[:FIND_ORDER_LIMIT]}


def _matching_product_ids(titles: dict[int, str], query_words: list[str]) -> set[int]:
    # Every query word must closely match some title word. Whole-string fuzzy
    # scores match on shared adjectives ("Heavy-Duty Vase" ~ "Heavy-Duty Webcam").
    return {
        product_id
        for product_id, title in titles.items()
        if all(
            any(fuzz.ratio(q, t) >= FIND_ORDER_MIN_SCORE for t in _words(title))
            for q in query_words
        )
    }


def search_customer_orders(
    ctx: AuthContext, customer_id: int, product_query: str | None = None
) -> dict[str, Any]:
    """Search one customer's orders on behalf of support staff. Risk tier: read.

    Support may view any customer's orders (SPEC access matrix), but get_order
    needs an exact id and find_order searches every customer at once and shows
    only the five newest matches. This tool scopes the search to one customer.

    Access rules: support only. Shoppers and merchants get permission_denied
    even if the tool is somehow exposed to them; their own scopes are covered
    by list_my_orders and find_order.

    Args:
        ctx: The caller's auth context.
        customer_id: The shopper whose orders to search.
        product_query: Optional product name, matched with the same per-word
            fuzzy rule as find_order. Blank means no product filter.

    There is deliberately no date filter: the model does not know the world
    date, so it would have to guess the year for requests like "in May". It
    reads the dates in the results instead.

    Returns:
        {"ok": True, "customer_id": int, "orders": [...], "count": int,
        "total_matches": int, "truncated": bool} with at most
        CUSTOMER_ORDER_LIMIT orders in find_order's newest-first order, each
        the public order dict plus "product_title". Matching is applied before
        truncation. When truncated, a "note" tells the model that older matches
        exist and to ask for the order number or a more specific product.
        Errors: permission_denied for a non-support caller; not_found when
        customer_id is not a shopper.
    """
    if ctx.role != "support":
        return permission_denied(
            f"role '{ctx.role}' (user {ctx.user_id}) may not search another customer's orders"
        )
    query_words = _words(product_query or "")

    with db.connection() as conn:
        customer = db.get_user(conn, customer_id)
        if customer is None or customer.role != "shopper":
            return {"ok": False, "error": "not_found", "reason": f"no customer with id {customer_id}"}
        titles = {p.id: p.title for p in db.list_products(conn)}
        candidates = db.list_order_search_candidates(conn, user_id=customer_id)

    matching_products = _matching_product_ids(titles, query_words) if query_words else None
    matches = [
        o for o in candidates if matching_products is None or o.product_id in matching_products
    ]
    shown = [
        {**o.to_public_dict(), "product_title": titles.get(o.product_id)}
        for o in matches[:CUSTOMER_ORDER_LIMIT]
    ]
    result: dict[str, Any] = {
        "ok": True,
        "customer_id": customer_id,
        "orders": shown,
        "count": len(shown),
        "total_matches": len(matches),
        "truncated": len(matches) > len(shown),
    }
    if result["truncated"]:
        result["note"] = (
            f"showing the {len(shown)} newest of {len(matches)} matching orders; older "
            "matches are not shown. Ask for the order number, or narrow the search "
            "with a more specific product_query."
        )
    return result


def _order_data_issue(
    order: db.Order, store_product_ids: set[int], as_of: date
) -> tuple[str, str] | None:
    """Return (code, detail) when the order record is too inconsistent to judge."""
    if order.status == "delivered" and order.delivered_at is None:
        return "missing_delivery_date", "status is delivered but no delivery date is recorded"
    if order.delivered_at is not None and order.delivered_at > as_of:
        return "future_delivery_date", f"delivery date {order.delivered_at} is after {as_of}"
    dates = [
        ("ordered", order.ordered_at),
        ("shipped", order.shipped_at),
        ("delivered", order.delivered_at),
    ]
    known = [(name, value) for name, value in dates if value is not None]
    for (earlier, a), (later, b) in zip(known, known[1:]):
        if a > b:
            return "reversed_chronology", f"{later} on {b} is before {earlier} on {a}"
    if order.product_id not in store_product_ids:
        return "store_mismatch", "the product does not belong to the order's store"
    return None


_NOT_DELIVERED_REASONS = {
    "placed": (
        "has not shipped yet, so it cannot be returned; it can still be cancelled "
        "(cw-cancellations)"
    ),
    "shipped": (
        "has not been delivered yet; the return window starts on the delivery date "
        "(cw-returns)"
    ),
    "cancelled": "was cancelled, so there is nothing to return",
    "refunded": "has already been refunded",
}


def check_return_eligibility(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Explain whether an order can be returned for a refund, and why. Risk tier: read.

    get_order reports only a refund_eligible flag. This tool computes the facts
    behind the decision so the model can explain it and cite the policy that
    applies: the return window (store override or platform default), the days
    since delivery, and the policy id.

    Eligibility is recomputed with seed.eligibility.is_refund_eligible on the
    world date, so it reflects the order's current status. The refund_eligible
    flag stamped at seed time (which issue_refund trusts) is not updated when
    the status changes, so the two can disagree after a refund.

    Access rules: the same as get_order (agent.auth.can_view_order). Scope is
    checked before anything about the order is returned.

    Returns:
        {"ok": True, "order_id", "store_name", "status", "delivered_at",
        "as_of", "days_since_delivery", "return_window_days", "window_source"
        ("store_override" or "platform"), "window_closes_on", "eligible",
        "policy_id", "reason"}. Dates are YYYY-MM-DD; delivery fields are None
        for undelivered orders.
        When the record is inconsistent (missing or future delivery date,
        dates out of order, product not sold by the order's store), no window
        is computed: {"ok": True, "order_id", "store_name", "status",
        "eligible": None, "data_issue": <code>, "reason"} with a reason telling
        the model to escalate (seed data_quality_cases expected handling).
        Errors: not_found for an unknown order; permission_denied for an order
        outside the caller's scope.
    """
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view order #{order_id}"
            )
        store = db.get_store(conn, order.store_id)
        as_of = db.world_asof(conn)
        store_product_ids = {p.id for p in db.list_products(conn, order.store_id)}

    data_issue = _order_data_issue(order, store_product_ids, as_of)
    if data_issue is not None:
        code, detail = data_issue
        return {
            "ok": True,
            "order_id": order_id,
            "store_name": store.name if store else None,
            "status": order.status,
            "eligible": None,
            "data_issue": code,
            "reason": (
                f"Order #{order_id} has an inconsistent record ({detail}), so return "
                "eligibility cannot be determined. Do not state a return deadline; "
                "escalate to a human."
            ),
        }

    platform_days = load_facts()["return_window_days"]
    override = store.return_window_days_override if store else None
    window = effective_return_window_days(platform_days, override)
    if override is None:
        window_source, policy_id = "platform", "cw-returns"
        rule = f"Cartwheel accepts returns within {window} days of delivery"
    else:
        window_source = "store_override"
        store_doc = f"store-{store.slug}-policy"
        known = {doc.policy_id for doc in load_policy_docs()}
        policy_id = store_doc if store_doc in known else "cw-store-overrides"
        rule = (
            f"{store.name} accepts returns within {window} days of delivery, "
            f"instead of the Cartwheel platform default of {platform_days} days"
        )

    delivered = order.delivered_at
    days = (as_of - delivered).days if delivered else None
    closes = delivered + timedelta(days=window) if delivered else None
    eligible = is_refund_eligible(
        status=order.status, delivered_at=delivered, as_of=as_of, return_window_days=window
    )

    if order.status != "delivered":
        why = _NOT_DELIVERED_REASONS.get(order.status, f"has status '{order.status}'")
        reason = f"Order #{order_id} {why}."
    elif eligible:
        reason = (
            f"Delivered {delivered} ({days} days ago). {rule} ({policy_id}), "
            f"so the return window is open until {closes}."
        )
    else:
        reason = (
            f"Delivered {delivered} ({days} days ago). {rule} ({policy_id}), "
            f"so the return window closed on {closes}."
        )

    return {
        "ok": True,
        "order_id": order_id,
        "store_name": store.name if store else None,
        "status": order.status,
        "delivered_at": delivered.isoformat() if delivered else None,
        "as_of": as_of.isoformat(),
        "days_since_delivery": days,
        "return_window_days": window,
        "window_source": window_source,
        "window_closes_on": closes.isoformat() if closes else None,
        "eligible": eligible,
        "policy_id": policy_id,
        "reason": reason,
    }
