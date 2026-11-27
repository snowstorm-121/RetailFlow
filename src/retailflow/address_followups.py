"""Read-only hints for explicit order references in compound address requests."""

import re


def advisory_opening(customer_messages):
    """Abstain after unrecognized follow-ups, including revisions to the opening."""
    if not customer_messages:
        return ""
    lookup = r"I don't remember[;.] please look through my orders\.?"
    email = r"[\w.+-]+@[\w.-]+\.[A-Za-z]+"
    acknowledgement = re.compile(
        r"(?:I explicitly confirm the displayed proposal\.|"
        r"My email is [\w.+-]+@[\w.-]+\.[A-Za-z]+\.?|"
        r"My name is [\w' -]+, ZIP \d{5}\.?(?: Please look for the matching order in my account\.)?|"
        r"(?:Your|You) name is [\w' -]+ and your (?:email is [\w.+-]+@[\w.-]+\.[A-Za-z]+|"
        r"zip code is \d{5})\.?(?: I don't remember; please look through my orders\.)?|"
        rf"You are [\w-]+ and {email}(?: and {email})*\.?(?: {lookup})?|"
        rf"{lookup}|"
        r"You are [\w' -]+\. You live in [\w -]+, [A-Z]{2} \d{5}\.?|"
        r"You are [\w' -]+ in zip ?code \d{5}\.?|"
        r"You(?: are|'re) [\w' -]+, living in [\w -]+ \d{5}\.?|"
        r"I no longer need (?:it|the order)\.?|I ordered (?:it|the order) by mistake\.?|"
        r"\d{5}\.?|"
        r"ordered by mistake\.?|no longer needed\.?|"
        r"Use my (?:credit card|original payment method)\.)", re.I,
    )
    return customer_messages[0] if all(acknowledgement.fullmatch(text.strip()) or text == customer_messages[0]
                                       for text in customer_messages[1:]) else ""


def explicit_address_steps(text):
    if not re.search(r"\b(?:address|shipping)\b", text, re.I):
        return []
    steps = []
    for clause in re.split(r"(?<=[.!?])\s+|\n", text):
        if ("?" in clause or re.search(r"\b(?:if|unless|until|do not|don.t|never|except|or|whichever|after|before)\b",
                                       clause, re.I)):
            continue
        shipping = re.match(
            r"(?:please\s+)?copy\b[^.!?]*\baddress\b[^.!?]*from(?: order)?\s+#W\d+\s+to(?: order)?\s+(#W\d+)"
            r"|(?:please\s+)?(?:change|update)\s+(?:the\s+)?(?:shipping(?: address)?|address)\s+for(?: order)?\s+(#W\d+)",
            clause, re.I,
        )
        references = re.findall(r"#W\d+", clause, re.I)
        if shipping and len(references) == (2 if shipping[1] else 1):
            steps.append({"tool": "modify_pending_order_address", "order_id": (shipping[1] or shipping[2]).upper()})
        item = re.match(r"(?:please\s+)?in(?: order)?\s+(#W\d+)\s*,?\s+change\s+(.+?)\s+to\s+([^.!?]+)", clause, re.I)
        if item and len(references) == 1 and not re.search(r"\b(?:shipping|address)\b", item[2], re.I):
            steps.append({"tool": "modify_pending_order_items", "order_id": item[1].upper(),
                          "item_description": item[2], "preferences": item[3]})
        cancel = re.fullmatch(r"(?:(?:also|please)\s+)*cancel(?: order)?\s+(#W\d+)"
                              r"(?: because I (?:ordered it by mistake|no longer need it))?[.!]?",
                              clause.strip(), re.I)
        if cancel:
            steps.append({"tool": "cancel_pending_order", "order_id": cancel[1].upper()})
    return steps


def processing_guarantee_fact(text):
    if not re.search(r"\b(?:can't|cannot|can not) guarantee\b[^.!?]*\b(?:processing|processed|ship\w*)\b", text, re.I):
        return ""
    return ("The available tools cannot guarantee a processing or shipping deadline. If the customer explicitly "
            "conditions cancellation on being unable to guarantee it, that condition is met. Order list position "
            "is not evidence of age. If the target still depends on which order is older and no dates are available, "
            "ask which order to cancel. Never choose the target by guessing.")


def matching_variants(variants, preferences):
    if re.search(r"\b(?:not|no|except|or|rather|instead of)\b", preferences, re.I):
        return []
    requested = {}
    for variant in variants.values():
        for key, value in variant["options"].items():
            if re.search(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", preferences, re.I):
                requested.setdefault(key, set()).add(value)
    if not requested or any(len(values) != 1 for values in requested.values()):
        return []
    return [{"item_id": item_id, "options": variant["options"], "price": variant["price"],
             "matched_options": {key: next(iter(values)) for key, values in requested.items()}}
            for item_id, variant in variants.items()
            if variant["available"] is True and all(variant["options"].get(key) in values
                                                     for key, values in requested.items())][:8]


def verified_address_reply(opening, receipts, orders, user_id, copy=None, source_order_id=None):
    """Render supported address outcomes from active-request receipts and verified reads."""
    if not opening or re.search(r"\b(?:cancel|return|exchange|transfer)\b", opening, re.I):
        return None

    def format_address(address):
        return ", ".join(address[key] for key in ("address1", "address2", "city", "state", "zip", "country")
                         if address.get(key))

    account = next((receipt for receipt in reversed(receipts)
                    if receipt["name"] == "modify_user_address" and receipt.get("user_id") == user_id), None)
    if copy:
        if re.search(r"\b(?:tell|check|show|tracking|how|what|whether)\b", opening, re.I):
            return None
        destination = copy["destination_order_id"]
        lines = []
        for receipt in receipts:
            if receipt["name"] == "modify_pending_order_items":
                lines.append(f"Item change confirmed for order {receipt['order_id']}.")
                payment = receipt.get("modification_details", {}).get("simulated_payment")
                if payment:
                    lines.append(f"Local {payment['transaction_type']} recorded: ${payment['amount']:.2f}. "
                                 "No real payment was sent.")
        if (not lines and re.search(r"(?:^|[.!?]\s+)(?:please )?(?:change|modify|switch) ", opening, re.I)):
            lines.append("No item change was confirmed in this request.")
        shipping = next((receipt for receipt in reversed(receipts)
                         if receipt["name"] == "modify_pending_order_address"
                         and receipt["order_id"] == destination and receipt.get("updated_address") == copy["address"]), None)
        if shipping:
            lines.append(f"Shipping address updated for order {destination}: {format_address(shipping['updated_address'])}.")
        else:
            lines.append(f"Shipping address change for order {destination} has no confirmed update in this request.")
        if copy["update_default"]:
            if account and account.get("updated_address") == copy["address"]:
                lines.append(f"Account default address updated: {format_address(account['updated_address'])}.")
            else:
                lines.append("Account default address change has no confirmed update in this request.")
        return "\n".join(lines)

    if (not account or not source_order_id or not re.search(r"\baccount default\b", opening, re.I)
            or not re.search(r"\bshipping address\b", opening, re.I)
            or not re.search(r"\b(?:tracking|in transit|delivery status)\b", opening, re.I)):
        return None
    address = format_address(account["updated_address"])
    lines = [f"Account default address updated: {address}."]
    order = next((order for order in reversed(orders)
                  if order["order_id"] == source_order_id and order["user_id"] == user_id), None)
    if not order:
        lines.append("The requested order details could not be verified.")
        return "\n".join(lines)
    lines.extend([f"Order {source_order_id} shipping address: {format_address(order['address'])}.",
                  f"Order status: {order['status']}."])
    if order["address"] != account["updated_address"]:
        lines.append("The updated account default differs from this order's shipping address.")
    tracking = [tracking_id for shipment in order["fulfillments"] for tracking_id in shipment["tracking_id"]]
    if tracking:
        lines.append("Tracking number: " + ", ".join(tracking) + ".")
    for item in order["items"]:
        if re.search(r"\b" + re.escape(item["name"]) + r"\b", opening, re.I):
            options = {key: value for key, value in item["options"].items()
                       if re.search(r"\b" + re.escape(key) + r"\b", opening, re.I)}
            if options:
                lines.append(item["name"] + " " + ", ".join(f"{key}: {value}" for key, value in options.items()) + ".")
    return "\n".join(lines)
