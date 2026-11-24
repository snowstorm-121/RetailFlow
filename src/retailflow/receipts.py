"""Customer-facing summaries for confirmed return and item modification outcomes."""

import json
import re

from langchain_core.messages import HumanMessage, ToolMessage

from retailflow.address_followups import advisory_opening
from retailflow.preproposal import requested_variants


def mixed_item_receipt_reply(messages, user_id):
    """Render an active return and modification request with optional catalog alternatives."""
    start = next((index for index in range(len(messages) - 1, -1, -1)
                  if messages[index].additional_kwargs.get("request_start")), 0)
    active = messages[start:]
    opening = advisory_opening([message.content for message in active
                                if isinstance(message, HumanMessage) and isinstance(message.content, str)])
    clauses = [part.strip().rstrip(".") for part in re.split(r"(?<=[.!?])\s+", opening) if part.strip()]
    if clauses and re.fullmatch(r"My email is [\w.+-]+@[\w.-]+\.[A-Za-z]+", clauses[0], re.I):
        clauses.pop(0)
    if (len(clauses) not in {2, 3}
            or not re.match(r"(?:please )?return\s+", clauses[0], re.I)
            or not re.match(r"(?:please )?(?:modify|change|switch)\s+", clauses[1], re.I)
            or re.search(r"\?|\b(?:cancel|exchange|transfer|track|tell|show|explain|how|what|when|where|address)\b",
                         " ".join(clauses[:2]), re.I)):
        return None
    alternate = None
    if len(clauses) == 3:
        alternate = re.fullmatch(r"If the exact version is unavailable, list all available ([\w -]+?) options"
                                 r"(?:; I prefer [\w -]+)?", clauses[2], re.I)
        if not alternate:
            return None

    receipts = []
    proposal_ids = set()
    for message in active:
        if isinstance(message, ToolMessage) and message.name in {"return_delivered_order_items", "modify_pending_order_items"}:
            result = json.loads(message.content)
            if result.get("code") == "CONFIRMATION_REQUIRED":
                proposal_ids.add(result["proposal_id"])
        if message.additional_kwargs.get("committed_tool"):
            receipts.append(json.loads(message.content.removeprefix("Application outcome: ")))
    if (len(receipts) != 2
            or any(receipt.get("user_id") != user_id or receipt.get("proposal_id") not in proposal_ids
                   for receipt in receipts)):
        return None
    by_tool = {receipt["name"]: receipt for receipt in receipts}
    if set(by_tool) != {"return_delivered_order_items", "modify_pending_order_items"}:
        return None
    returned = by_tool["return_delivered_order_items"]
    modified = by_tool["modify_pending_order_items"]
    details = modified["modification_details"]
    by_item_id = {item["item_id"]: item for item in details["items"]}
    new_item_ids = details.get("new_item_ids", [])
    if not new_item_ids or any(item_id not in by_item_id for item_id in new_item_ids):
        return None
    replacements = [by_item_id[item_id] for item_id in new_item_ids]
    return_details = returned["return_details"]
    returned_names = ", ".join(f"{item['name']} ({item['item_id']})" for item in return_details["items"])
    summary = [
        f"Return requested for order {returned['order_id']}: {returned_names}.",
        f"Refund method: {return_details['payment_method_id']}. The refund is not settled.",
        return_details["instructions"],
        f"Item modification confirmed for order {modified['order_id']}. Status: {modified['status']}.",
    ]

    def format_item(item):
        options = ", ".join(f"{key}: {value}" for key, value in item["options"].items())
        return f"{item['item_id']} ({options})"

    summary.extend(f"Replacement {item['name']}: {format_item(item)}." for item in replacements)
    payment = details["simulated_payment"]
    direction = "no difference"
    if payment["amount"]:
        direction = "amount due" if payment["transaction_type"] == "payment" else "refund due"
    summary.append(f"Local price difference: ${payment['amount']:.2f} ({direction}); "
                   f"payment method: {payment['payment_method_id']}. No real payment was sent.")
    if alternate:
        catalogs = {}
        for message in active:
            if isinstance(message, ToolMessage) and message.name == "get_product_details" and message.status == "success":
                product = json.loads(message.content)["data"]
                catalogs[product["product_id"]] = product
        options = []
        for product_id in dict.fromkeys(item["product_id"] for item in replacements):
            product = catalogs.get(product_id)
            if product:
                variants = requested_variants(product["variants"], alternate[1].lower())
                options.extend(f"- {format_item(item)} — ${item['price']:.2f}" for item in variants or [])
        summary.append(f"Available {alternate[1]} options from the catalog read:" if options
                       else "The requested alternatives could not be verified from the catalog reads.")
        summary.extend(options)
    return "\n".join(summary)
