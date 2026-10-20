import hashlib
import json

from conftest import SOURCE_DB, STATE_REF


def test_source_tool_reads_are_authorized_and_scoped(env):
    own = env.dispatch_tool("session-a", "get_order_details", {"order_id": "#W5056519"})
    assert own["ok"] and own["data"]["status"] == "pending"
    foreign = env.dispatch_tool("session-a", "get_order_details", {"order_id": "#W1679211"})
    assert foreign["code"] == "POLICY_DENIED"
    profile = env.dispatch_tool("session-a", "get_user_details", {"user_id": "yara_muller_8652"})
    assert profile["ok"] and profile["data"]["email"]
    assert env.dispatch_tool("session-a", "get_user_details", {"user_id": "yusuf_gonzalez_8900"})["code"] == "POLICY_DENIED"


def test_identity_must_be_located_and_cannot_switch_user(env):
    env.create_session("session-b", STATE_REF)
    assert env.dispatch_tool("session-b", "get_user_details", {"user_id": "yara_muller_8652"})["code"] == "AUTH_REQUIRED"
    assert env.dispatch_tool("session-a", "find_user_id_by_email", {"email": "yusuf.gonzalez2399@example.com"})["code"] == "POLICY_DENIED"
    assert env.dispatch_tool("session-a", "find_user_id_by_email", {"email": "missing@example.com"})["code"] == "NOT_FOUND"


def test_schema_and_tool_allowlist(env):
    assert env.dispatch_tool("session-a", "modify_pending_order_items", {})["code"] == "UNKNOWN_TOOL"
    assert env.dispatch_tool("session-a", "cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake", "approved": True})["code"] == "INVALID_ARGUMENT"
    assert env.dispatch_tool("session-a", "get_order_details", {"order_id": 5056519})["code"] == "INVALID_ARGUMENT"


def test_policy_denial_does_not_make_proposal(env):
    before = env.state_hash("session-a")
    bad_reason = env.dispatch_tool("session-a", "cancel_pending_order", {"order_id": "#W5056519", "reason": "changed mind"})
    assert bad_reason["code"] == "POLICY_DENIED"
    env.create_session("delivered-thread", STATE_REF)
    env.dispatch_tool("delivered-thread", "find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"})
    bad_status = env.dispatch_tool("delivered-thread", "cancel_pending_order", {"order_id": "#W2378156", "reason": "no longer needed"})
    assert bad_status["code"] == "POLICY_DENIED" and "Non-pending" in bad_status["message"]
    assert env.pending_proposal("delivered-thread") is None
    assert env.pending_proposal("session-a") is None
    assert env.state_hash("session-a") == before


def test_unknown_order_and_item_report_not_found(env):
    assert env.dispatch_tool("session-a", "cancel_pending_order", {"order_id": "#WDOESNOTEXIST", "reason": "ordered by mistake"})["code"] == "NOT_FOUND"
    env.create_session("return-thread", STATE_REF)
    env.dispatch_tool("return-thread", "find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Gonzalez", "zip": "91455"})
    assert env.dispatch_tool("return-thread", "return_delivered_order_items", {"order_id": "#W1679211", "item_ids": ["not-an-item"], "payment_method_id": "paypal_3022415"})["code"] == "NOT_FOUND"


def test_each_session_has_independent_state_and_original_db_is_read_only(env):
    source_hash = hashlib.sha256(SOURCE_DB.read_bytes()).hexdigest()
    env.create_session("session-b", STATE_REF)
    auth = env.dispatch_tool("session-b", "find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"})
    assert auth["ok"]
    first = env.dispatch_tool("session-a", "cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"})
    env.confirm_action("session-a", first["proposal_id"], accepted=True)
    assert env.dispatch_tool("session-a", "cancel_pending_order", first["proposal"]["args"])["ok"]
    assert env.get_order("session-a", "#W5056519")["status"] == "cancelled"
    assert env.get_order("session-b", "#W5056519")["status"] == "pending"
    assert hashlib.sha256(SOURCE_DB.read_bytes()).hexdigest() == source_hash


def test_return_calls_original_business_rules(env):
    env.create_session("return-thread", STATE_REF)
    assert env.dispatch_tool("return-thread", "find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Gonzalez", "zip": "91455"})["ok"]
    args = {"order_id": "#W1679211", "item_ids": ["6268080249"], "payment_method_id": "paypal_3022415"}
    before = env.state_hash("return-thread")
    requested = env.dispatch_tool("return-thread", "return_delivered_order_items", args)
    assert requested["code"] == "CONFIRMATION_REQUIRED"
    assert env.state_hash("return-thread") == before
    env.confirm_action("return-thread", requested["proposal_id"], accepted=True)
    result = env.dispatch_tool("return-thread", "return_delivered_order_items", args)
    assert result["ok"] and result["data"]["status"] == "return requested"
    assert env.get_order("return-thread", "#W1679211")["return_items"] == ["6268080249"]
    saved = env.state_hash("return-thread")
    assert env.dispatch_tool("return-thread", "return_delivered_order_items", args) == result
    assert env.state_hash("return-thread") == saved


def test_return_invalid_item_or_payment_is_policy_denied(env):
    env.create_session("return-thread", STATE_REF)
    env.dispatch_tool("return-thread", "find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Gonzalez", "zip": "91455"})
    base = {"order_id": "#W1679211", "item_ids": ["6268080249"], "payment_method_id": "paypal_3022415"}
    assert env.dispatch_tool("return-thread", "return_delivered_order_items", {**base, "item_ids": ["not-an-item"]})["code"] == "NOT_FOUND"
    assert env.dispatch_tool("return-thread", "return_delivered_order_items", {**base, "payment_method_id": "credit_card_7918119"})["code"] == "POLICY_DENIED"
    assert env.pending_proposal("return-thread") is None


def test_product_lookup_uses_upstream_product_schema(env):
    result = env.dispatch_tool("session-a", "get_product_details", {"product_id": "9523456873"})
    assert result["ok"] and result["data"]["name"] == "T-Shirt"
    assert result["data"]["variants"]
