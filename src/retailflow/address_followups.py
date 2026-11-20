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
