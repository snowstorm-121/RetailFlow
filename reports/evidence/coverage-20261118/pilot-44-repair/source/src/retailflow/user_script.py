"""A deterministic user role that never receives expected states or scores."""

from copy import deepcopy


def user_view(task: dict, field_responses: dict | None = None) -> dict:
    replies = field_responses or {}
    if set(replies) - {"identity", "order", "cancel_reason", "return_items", "refund_method", "exchange_items", "exchange_options", "payment_method"}:
        raise ValueError("Unsupported user reply field")
    spec = deepcopy({key: task[key] for key in ("initial_user_message", "user_private_facts", "script")})
    spec["script"]["field_responses"].update(deepcopy(replies))
    return spec


class ScriptedUser:
    def __init__(self, spec: dict, consent: dict):
        self.spec = spec
        self.consent = consent
        self.asked = {}
        self.used_rules = set()

    def answer(self, request: dict) -> str:
        replies = []
        for field in request["fields"]:
            if field not in {"identity", "order", "cancel_reason", "return_items", "refund_method", "exchange_items", "exchange_options", "payment_method"}:
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
        consent = self.consent
        if "by_tool" in consent:
            consent = consent["by_tool"].get(proposal["name"], {})
            consent = {"tool": proposal["name"], **consent} if consent else {}
        if "by_order" in consent:
            consent = {**consent, **consent["by_order"].get(args["order_id"], {})}
            if args["order_id"] not in consent["by_order"]:
                return False, None
        if proposal["name"] not in consent.get("tools", [consent.get("tool")]):
            return False, None
        include = {name.casefold() for name in consent.get("include", [])}
        exclude = {name.casefold() for name in consent.get("exclude", [])}
        allowed = bool(names) and not names & exclude
        if include:
            allowed &= bool(names & include) if proposal["name"] == "cancel_pending_order" else names <= include
        else:
            allowed &= bool(consent.get("all"))
        if proposal["name"] == "return_delivered_order_items":
            allowed &= args["payment_method_id"] == display["original_payment_method_id"]
            all_items = display.get("all_items", items)
            expected_ids = sorted(item["item_id"] for item in all_items
                                  if item["name"].casefold() not in exclude
                                  and (not include or item["name"].casefold() in include)
                                  and all(item.get("options", {}).get(k) == v for k, v in
                                          consent.get("return_options", {}).get(item["name"], {}).items()))
            allowed &= sorted(args["item_ids"]) == expected_ids
        if proposal["name"] in {"exchange_delivered_order_items", "modify_pending_order_items"}:
            pairs = display.get("exchange_pairs", [])
            allowed &= bool(pairs) and len(pairs) == len(args["item_ids"]) == len(args["new_item_ids"])
            required = consent.get("exchange_required_names")
            if required:
                allowed &= names == {name.casefold() for name in required}
            for pair in pairs:
                expected = consent.get("exchange_options", {}).get(pair["name"], {})
                old_expected = consent.get("exchange_old_options", {}).get(pair["name"], {})
                allowed &= all(pair["new_options"].get(k) == v for k, v in expected.items())
                allowed &= all(pair["old_options"].get(k) == v for k, v in old_expected.items())
            requirements = consent.get("exchange_requirements")
            if requirements is not None:
                # Match a multiset, not a set of names: two kettles may need different variants.
                assigned = {}
                def match(index, seen):
                    requirement = requirements[index]
                    for slot, pair in enumerate(pairs):
                        if (slot in seen or pair["name"] != requirement["name"]
                                or any(pair["new_options"].get(k) != v for k, v in requirement.get("new_options", {}).items())
                                or any(pair["old_options"].get(k) != v for k, v in requirement.get("old_options", {}).items())):
                            continue
                        seen.add(slot)
                        if slot not in assigned or match(assigned[slot], seen):
                            assigned[slot] = index
                            return True
                    return False
                allowed &= len(requirements) == len(pairs) and all(match(i, set()) for i in range(len(requirements)))
            payment_source = consent.get("exchange_payment_source")
            if payment_source:
                allowed &= display.get("payment_source") == payment_source
            else:
                allowed &= args["payment_method_id"] == display["original_payment_method_id"]
        reason = consent.get("cancel_reason")
        if reason and proposal["name"] == "cancel_pending_order":
            allowed &= args["reason"] == reason
        return bool(allowed), None if allowed else self.consent.get("rejection_reply")
