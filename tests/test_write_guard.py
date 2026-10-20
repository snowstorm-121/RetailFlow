from conftest import STATE_REF


CANCEL = {"order_id": "#W5056519", "reason": "ordered by mistake"}


def test_write_requires_matching_confirmation(env):
    before = env.state_hash("session-a")
    result = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    assert result["code"] == "CONFIRMATION_REQUIRED"
    assert env.state_hash("session-a") == before
    proposal = env.pending_proposal("session-a")
    env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    result = env.dispatch_tool("session-a", "cancel_pending_order", proposal["args"])
    assert result["ok"] is True
    assert env.get_order("session-a", "#W5056519")["status"] == "cancelled"
    assert len(env.get_order("session-a", "#W5056519")["payment_history"]) == 2


def test_rejected_confirmation_has_no_side_effect(env):
    before = env.state_hash("session-a")
    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    env.confirm_action("session-a", proposal["proposal_id"], accepted=False)
    assert env.pending_proposal("session-a") is None
    assert env.state_hash("session-a") == before
    again = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    assert again["code"] == "CONFIRMATION_REQUIRED" and again["proposal_id"] != proposal["proposal_id"]


def test_wrong_or_cross_thread_confirmation_fails(env):
    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    assert env.confirm_action("session-a", "wrong", accepted=True)["code"] == "INVALID_CONFIRMATION"
    env.create_session("session-b", STATE_REF)
    assert env.confirm_action("session-b", proposal["proposal_id"], accepted=True)["code"] == "INVALID_CONFIRMATION"
    assert env.get_order("session-a", "#W5056519")["status"] == "pending"


def test_expired_proposal_cannot_be_confirmed(env, monkeypatch):
    import retailflow.environment as module

    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    start = env.pending_proposal("session-a")["created_at"]
    monkeypatch.setattr(module.time, "time", lambda: start + module.PROPOSAL_TTL_SECONDS + 1)
    result = env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    assert result["code"] == "INVALID_CONFIRMATION"
    assert env.pending_proposal("session-a") is None
    assert env.get_order("session-a", "#W5056519")["status"] == "pending"


def test_changed_parameters_invalidate_old_approval(env):
    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    changed = {**CANCEL, "reason": "no longer needed"}
    result = env.dispatch_tool("session-a", "cancel_pending_order", changed)
    assert result["code"] == "CONFIRMATION_REQUIRED"
    assert result["proposal_id"] != proposal["proposal_id"]
    assert env.get_order("session-a", "#W5056519")["status"] == "pending"
    assert env.confirm_action("session-a", proposal["proposal_id"], accepted=True)["code"] == "INVALID_CONFIRMATION"


def test_invalid_new_target_also_invalidates_old_approval(env):
    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    changed = {**CANCEL, "order_id": "#W1679211"}  # another user's order
    assert env.dispatch_tool("session-a", "cancel_pending_order", changed)["code"] == "POLICY_DENIED"
    assert env.confirm_action("session-a", proposal["proposal_id"], accepted=True)["code"] == "INVALID_CONFIRMATION"
    assert env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)["code"] == "CONFIRMATION_REQUIRED"
    assert env.get_order("session-a", "#W5056519")["status"] == "pending"


def test_reading_another_order_invalidates_old_approval(env):
    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    assert env.dispatch_tool("session-a", "get_order_details", {"order_id": "#W5995614"})["ok"]
    assert env.confirm_action("session-a", proposal["proposal_id"], accepted=True)["code"] == "INVALID_CONFIRMATION"
    assert env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)["code"] == "CONFIRMATION_REQUIRED"
    assert env.get_order("session-a", "#W5056519")["status"] == "pending"


def test_switching_user_invalidates_old_approval(env):
    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    other = {"first_name": "Yusuf", "last_name": "Gonzalez", "zip": "91455"}
    assert env.dispatch_tool("session-a", "find_user_id_by_name_zip", other)["code"] == "POLICY_DENIED"
    assert env.confirm_action("session-a", proposal["proposal_id"], accepted=True)["code"] == "INVALID_CONFIRMATION"
    assert env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)["code"] == "CONFIRMATION_REQUIRED"
    assert env.get_order("session-a", "#W5056519")["status"] == "pending"


def test_changed_return_item_requires_new_confirmation(env):
    env.create_session("return-thread", STATE_REF)
    env.dispatch_tool("return-thread", "find_user_id_by_name_zip", {"first_name": "Yusuf", "last_name": "Gonzalez", "zip": "91455"})
    first_args = {"order_id": "#W1679211", "item_ids": ["6268080249"], "payment_method_id": "paypal_3022415"}
    first = env.dispatch_tool("return-thread", "return_delivered_order_items", first_args)
    env.confirm_action("return-thread", first["proposal_id"], accepted=True)
    changed = {**first_args, "item_ids": ["9612497925"]}
    second = env.dispatch_tool("return-thread", "return_delivered_order_items", changed)
    assert second["code"] == "CONFIRMATION_REQUIRED" and second["proposal_id"] != first["proposal_id"]
    assert env.confirm_action("return-thread", first["proposal_id"], accepted=True)["code"] == "INVALID_CONFIRMATION"
    assert env.get_order("return-thread", "#W1679211")["status"] == "delivered"


def test_replay_returns_existing_receipt_after_reopen(env):
    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    first = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    from retailflow.environment import RetailEnvironment

    reopened = RetailEnvironment(env.store_path)
    again = reopened.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    assert again == first
    assert len(reopened.get_order("session-a", "#W5056519")["payment_history"]) == 2


def test_gift_card_refund_occurs_once_after_confirmation(env):
    env.create_session("gift-thread", STATE_REF)
    env.dispatch_tool("gift-thread", "find_user_id_by_name_zip", {"first_name": "Ethan", "last_name": "Lopez", "zip": "43275"})
    profile_before = env.dispatch_tool("gift-thread", "get_user_details", {"user_id": "ethan_lopez_6291"})["data"]
    old_balance = profile_before["payment_methods"]["gift_card_7219486"]["balance"]
    args = {"order_id": "#W6779827", "reason": "ordered by mistake"}
    proposed = env.dispatch_tool("gift-thread", "cancel_pending_order", args)
    assert env.dispatch_tool("gift-thread", "get_user_details", {"user_id": "ethan_lopez_6291"})["data"]["payment_methods"]["gift_card_7219486"]["balance"] == old_balance
    env.confirm_action("gift-thread", proposed["proposal_id"], accepted=True)
    first = env.dispatch_tool("gift-thread", "cancel_pending_order", args)
    assert first["ok"]
    assert env.dispatch_tool("gift-thread", "cancel_pending_order", args) == first
    profile_after = env.dispatch_tool("gift-thread", "get_user_details", {"user_id": "ethan_lopez_6291"})["data"]
    assert profile_after["payment_methods"]["gift_card_7219486"]["balance"] == round(old_balance + 4079.45, 2)
    assert len(env.get_order("gift-thread", "#W6779827")["payment_history"]) == 2


def test_mutate_then_raise_does_not_persist_partial_business_state(env, monkeypatch):
    from retailflow.upstream.tools import RetailTools

    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    before = env.state_hash("session-a")

    def broken(self, order_id, reason):
        self.db.orders[order_id].status = "cancelled"
        raise RuntimeError("injected failure after mutation")

    with monkeypatch.context() as patch:
        patch.setattr(RetailTools, "cancel_pending_order", broken)
        result = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    assert result["code"] == "TOOL_ERROR"
    assert env.state_hash("session-a") == before
    assert env.get_order("session-a", "#W5056519")["status"] == "pending"


def test_confirmed_write_calls_upstream_business_method_once(env, monkeypatch):
    from retailflow.upstream.tools import RetailTools

    original = RetailTools.cancel_pending_order
    calls = []

    def count(self, order_id, reason):
        calls.append((order_id, reason))
        return original(self, order_id, reason)

    monkeypatch.setattr(RetailTools, "cancel_pending_order", count)
    proposal = env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)
    assert len(calls) == 1  # disposable policy preflight
    env.confirm_action("session-a", proposal["proposal_id"], accepted=True)
    assert env.dispatch_tool("session-a", "cancel_pending_order", CANCEL)["ok"]
    assert len(calls) == 2  # one committed execution, no second preflight
