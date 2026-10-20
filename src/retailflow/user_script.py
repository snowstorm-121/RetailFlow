"""A deterministic user role that never receives expected states or scores."""

from copy import deepcopy


def user_view(task: dict) -> dict:
    return deepcopy({key: task[key] for key in ("initial_user_message", "user_private_facts", "script")})


class ScriptedUser:
    def __init__(self, spec: dict, consent: dict):
        self.spec = spec
        self.consent = consent
        self.asked = {}
        self.used_rules = set()

    def answer(self, request: dict) -> str:
        replies = []
        for field in request["fields"]:
            if field not in {"identity", "order", "cancel_reason", "return_items", "refund_method"}:
                raise ValueError(f"script_unhandled: unsupported field {field}")
            value = self.spec["script"]["field_responses"].get(field)
            if not value:
                raise ValueError(f"script_unhandled: no registered response for {field}")
            if isinstance(value, list):
                index = self.asked.get(field, 0)
                value = value[min(index, len(value) - 1)]
            self.asked[field] = self.asked.get(field, 0) + 1
            replies.append(value)
        if not replies:
            raise ValueError("script_unhandled: no requested fields")
        return " ".join(replies)

    def decide(self, proposal: dict) -> tuple[bool, str | None]:
        display = proposal.get("display")
        if not display:
            raise ValueError("script_unhandled: proposal has no user-visible item details")
        args = proposal["args"]
        items = display["items"]
        names = {item["name"].casefold() for item in items}
        for index, rule in enumerate(self.spec["script"]["proposal_rules"]):
            if index in self.used_rules or rule["tool"] != proposal["name"]:
                continue
            if rule.get("order_id") and rule["order_id"] != args.get("order_id"):
                continue
            if rule.get("unless_item_name", "").casefold() in names:
                continue
            self.used_rules.add(index)
            if rule["decision"] == "decline":
                self.consent = {}
            return False, rule["reply"]
        if proposal["name"] not in self.consent.get("tools", [self.consent.get("tool")]):
            return False, None
        include = {name.casefold() for name in self.consent.get("include", [])}
        exclude = {name.casefold() for name in self.consent.get("exclude", [])}
        allowed = bool(names) and not names & exclude
        if include:
            allowed &= bool(names & include) if proposal["name"] == "cancel_pending_order" else names <= include
        else:
            allowed &= bool(self.consent.get("all"))
        if proposal["name"] == "return_delivered_order_items":
            allowed &= args["payment_method_id"] == display["original_payment_method_id"]
            all_items = display.get("all_items", items)
            expected_ids = sorted(item["item_id"] for item in all_items
                                  if item["name"].casefold() not in exclude
                                  and (not include or item["name"].casefold() in include))
            allowed &= sorted(args["item_ids"]) == expected_ids
        reason = self.consent.get("cancel_reason")
        if reason and proposal["name"] == "cancel_pending_order":
            allowed &= args["reason"] == reason
        return bool(allowed), None
