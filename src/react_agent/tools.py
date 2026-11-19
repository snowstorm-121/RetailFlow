"""Small, local-only tools for the T1 model check."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from retailflow.tools import SCHEMAS


@tool
def add(a: int, b: int) -> int:
    """Add two integers. Use this tool for requested addition."""
    return a + b


@tool
def lookup_route_minutes(route_id: str) -> int:
    """Look up the travel time in minutes for route A1 or B2."""
    return {"A1": 7, "B2": 11}[route_id]


@tool
def minutes_to_seconds(minutes: int) -> int:
    """Convert a number of minutes to seconds. Use after looking up a route."""
    return minutes * 60


TOOLS = [add, lookup_route_minutes, minutes_to_seconds]


ASK_USER_FIELDS = ["identity", "order", "cancel_reason", "return_items", "refund_method",
                   "exchange_items", "exchange_options", "payment_method", "address"]


RETAIL_DESCRIPTIONS = {
    "modify_pending_order_address": "Propose a shipping address change for a pending order. Provide the full address, using an empty address2 if none. Explicit confirmation required.",
    "modify_user_address": "Propose changing the authenticated account default address. This does not change existing orders. Provide the full address; explicit confirmation required.",
    "find_user_id_by_email": "Find a retail user ID from their email address.",
    "find_user_id_by_name_zip": "Find a retail user ID from first name, last name, and ZIP code.",
    "get_user_details": "Read the authenticated user's profile and payment methods.",
    "get_order_details": "Read one order after locating its owner. Preserve the leading # in an order ID such as #W5056519.",
    "get_product_details": "Read product details by exact product ID.",
    "cancel_pending_order": "Stage a cancellation proposal for the authenticated user's pending order. The first call does NOT cancel; it returns a proposal for the application to confirm. Preserve the leading # in the order ID. Reason must be 'ordered by mistake' or 'no longer needed'.",
    "return_delivered_order_items": "Stage a return proposal for selected item IDs in the authenticated user's delivered order. The first call does NOT return items; it creates an application confirmation proposal. Preserve the leading # in the order ID.",
    "exchange_delivered_order_items": "Stage an exchange proposal for a delivered order. Pair each old item_id with a new_item_id of the SAME product, in matching list positions. Read available variants and choose an existing user payment method for the price difference. The application displays details and requires confirmation; no settlement occurs here.",
    "modify_pending_order_items": "Stage a pending order item modification. Pair old/new IDs of the same product; all changes must be collected before the single allowed modification. Application confirmation is required; signed price difference is recorded in simulated payments.",
    "calculate": "Calculate a monetary expression using numbers, parentheses and + - * / only. Use for totals and price comparisons; this does not change any business data.",
    "transfer_to_human_agents": "Mark a request for human handoff when it cannot be handled by these tools. No human is contacted here.",
}


def _schema(name, fields, description):
    properties = {}
    for key, kind in fields.items():
        properties[key] = {"type": "array", "items": {"type": "string"}} if kind is list else {"type": "string"}
        if name == "ask_user" and key == "fields":
            properties[key].update(uniqueItems=True, minItems=1, items={"type": "string", "enum": ASK_USER_FIELDS})
        if key == "order_id":
            properties[key]["pattern"] = "^#W[0-9]+$"
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(fields), "additionalProperties": False,
    }}}


RETAIL_TOOLS = [_schema(name, fields, RETAIL_DESCRIPTIONS[name]) for name, fields in SCHEMAS.items()]
RETAIL_TOOLS.append(_schema("ask_user", {"fields": list, "question": str}, "Request missing customer information. Use this tool instead of asking a question in normal text. Pause and wait for the customer. fields must use the listed categories."))


def native_dialogue_examples():
    """Protocol-only examples: no identity, order, business result or evaluation labels."""
    messages = []
    for index, text in enumerate(("I want to return something.", "Cancel my purchase please.")):
        args = {"fields": ["identity"], "question": "What is your email or full name and ZIP?"}
        ident = f"example-ask-{index}"
        marked = {"protocol_example": True}
        messages.extend([
            HumanMessage(content=text, additional_kwargs=marked),
            AIMessage(content="", tool_calls=[{"name": "ask_user", "args": args, "id": ident}], additional_kwargs=marked),
            ToolMessage(content=json.dumps({"ok": True, "code": "NEEDS_INPUT", "data": args}),
                        name="ask_user", tool_call_id=ident, additional_kwargs=marked),
        ])
    return messages


def retail_tools_for_session(session):
    """Offer tools for the last inspected order; dispatch still checks all permissions."""
    names = {"ask_user", "transfer_to_human_agents"}
    if not session.get("user_id"):
        names.update({"find_user_id_by_email", "find_user_id_by_name_zip"})
    else:
        names.update({"get_user_details", "get_order_details", "get_product_details", "calculate", "modify_user_address"})
        if session.get("target_order_status") in {"pending", "pending (item modified)"}:
            names.add("modify_pending_order_address")
        if session.get("target_order_status") == "pending":
            names.add("modify_pending_order_items")
        write = {"pending": "cancel_pending_order", "delivered": "return_delivered_order_items"}.get(
            session.get("target_order_status"))
        if write:
            names.add(write)
        if session.get("target_order_status") == "delivered":
            names.add("exchange_delivered_order_items")
    return [tool for tool in RETAIL_TOOLS if tool["function"]["name"] in names]
