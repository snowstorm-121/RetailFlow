"""Small, local-only tools for the T1 model check."""

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


RETAIL_DESCRIPTIONS = {
    "find_user_id_by_email": "Find a retail user ID from their email address.",
    "find_user_id_by_name_zip": "Find a retail user ID from first name, last name, and ZIP code.",
    "get_user_details": "Read the authenticated user's profile and payment methods.",
    "get_order_details": "Read one order after locating its owner. Preserve the leading # in an order ID such as #W5056519.",
    "get_product_details": "Read product details by exact product ID.",
    "cancel_pending_order": "Stage a cancellation proposal for the authenticated user's pending order. The first call does NOT cancel; it returns a proposal for the application to confirm. Preserve the leading # in the order ID. Reason must be 'ordered by mistake' or 'no longer needed'.",
    "return_delivered_order_items": "Stage a return proposal for selected item IDs in the authenticated user's delivered order. The first call does NOT return items; it creates an application confirmation proposal. Preserve the leading # in the order ID.",
    "transfer_to_human_agents": "Mark a request for human handoff when it cannot be handled by these tools. No human is contacted here.",
}


def _schema(name, fields, description):
    properties = {}
    for key, kind in fields.items():
        properties[key] = {"type": "array", "items": {"type": "string"}} if kind is list else {"type": "string"}
        if key == "order_id":
            properties[key]["pattern"] = "^#W[0-9]+$"
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(fields), "additionalProperties": False,
    }}}


RETAIL_TOOLS = [_schema(name, fields, RETAIL_DESCRIPTIONS[name]) for name, fields in SCHEMAS.items()]
RETAIL_TOOLS.append(_schema("ask_user", {"fields": list, "question": str}, "Ask the user for missing fields, then pause this turn."))
