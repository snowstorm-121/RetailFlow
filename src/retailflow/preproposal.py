"""Read-only evidence and checks before a retail proposal is shown."""

import re

from retailflow.address_followups import advisory_opening


def inventory_order_request(customer_messages):
    """Keep product mentions across recognized identity and order-discovery replies."""
    if not customer_messages:
        return "", False
    lookup = (r"(?:I don't remember[;.] )?please look through my orders\.?|"
              r"please find the order containing those items\.?")
    email = r"[\w.+-]+@[\w.-]+\.[A-Za-z]+"
    identity = (rf"(?:You|Your) name is [\w' -]+ and your emails are {email}(?: and {email})+\.?|"
                r"You are [\w' -]+, residing in [\w -]+ \d{5}\.?")
    followups = []
    delegated = False
    for text in customer_messages[1:]:
        text = text.strip()
        requested_lookup = re.search(rf"(?:^|(?<=[.!?])\s+)(?:{lookup})$", text, re.I)
        if requested_lookup:
            delegated = True
            text = text[:requested_lookup.start()].strip()
        if text and not re.fullmatch(identity, text, re.I):
            followups.append(text)
    return advisory_opening([customer_messages[0], *followups]), delegated


def product_order_candidates(customer_text, business, user_id, order_ids):
    """List owned orders containing explicitly mentioned product names."""
    if not user_id or not customer_text or re.search(r"#w\d+", customer_text, re.I):
        return []
    candidates = []
    for order_id in order_ids:
        order = business["orders"].get(order_id)
        if not order or order["user_id"] != user_id:
            continue
        names = {item["name"] for item in order["items"]}
        matched = set()
        for name in sorted(names):
            words = re.findall(r"[a-z0-9]+", name.casefold())
            pattern = r"(?<!\w)" + r"[-\s]+".join(map(re.escape, words)) + r"(?:s|es)?(?!\w)"
            if words and re.search(pattern, customer_text, re.I):
                matched.add(name)
        if matched:
            items = [{"name": item["name"], "options": item["options"]} for item in order["items"] if item["name"] in matched]
            candidates.append({"order_id": order_id, "status": order["status"], "items": items})
    return candidates if len(candidates) <= 8 else []


def address_order_candidates(customer_text, business, user_id, order_ids):
    clauses = [clause.lower() for clause in re.split(r"(?<=[.!?])\s+|\n", customer_text)
               if re.search(r"\baddress\b", clause, re.I)]
    orders = [(order_id, order) for order_id in order_ids if (order := business["orders"].get(order_id))
              and order["user_id"] == user_id]
    source_ids = {order_id.upper() for clause in clauses
                  for order_id in re.findall(r"\baddress\b[^.;!?]{0,65}?\bfrom\s+(?:order\s+)?(#w\d+)\b", clause)}
    if source_ids:
        return [{"order_id": order_id, "city": order["address"]["city"], "matched_terms": ["source order"]}
                for order_id, order in orders if len(source_ids) == 1 and order_id in source_ids]

    source_names = {name.strip() for clause in clauses
                    for name in re.findall(r"\baddress\b[^.;!?]{0,65}?\bfrom\s+(?:my|the)\s+([a-z ]{1,50}?) order\b", clause)
                    if name.strip() != "other"}
    aliases = {"new york": ("nyc",), "los angeles": ("la",)}
    named_cities = set()
    address_cities = set()
    for city in {order["address"]["city"].lower() for _, order in orders} | aliases.keys():
        terms = "|".join(map(re.escape, (city, *aliases.get(city, ()))))
        if any(re.search(r"\b(?:" + terms + r") address\b", clause) for clause in clauses):
            address_cities.add(city)
        if any(re.search(r"\b(?:" + terms + r")\b", clause) for clause in clauses):
            named_cities.add(city)

    rows = []
    words = set(re.findall(r"[a-z]{5,}", customer_text.lower()))
    for order_id, order in orders:
        city = order["address"]["city"]
        city_hit = city.lower() in (address_cities or named_cities)
        names = [item["name"].lower() for item in order["items"]]
        source_hits = {source for source in source_names if any(re.search(r"\b" + re.escape(source) + r"\b", name)
                                                               for name in names)}
        if (source_names and not source_hits) or (address_cities and not city_hit):
            continue
        if city_hit or source_hits:
            matched = set(re.findall(r"[a-z]{5,}", " ".join(names))) & words
            rows.append((len(matched), {"order_id": order_id, "city": city,
                                       "matched_terms": sorted(matched | source_hits | ({city.lower()} if city_hit else set()))}))
    if not rows:
        return []
    best = max(rank for rank, _ in rows)
    return [row for rank, row in rows if rank == best][:5]


def address_copy_target(text, business, user_id):
    """Resolve a named source and the distinct pending order at the current account address."""
    copy = re.search(
        r"(?:^|[.!?]\s+)(?:please\s+)?copy (?:the )?(?:shipping )?address "
        r"from (?:my|the) ([a-z][a-z -]*?) order to (?:the|my) (?:pending )?order "
        r"(?:still )?(?:going|shipping) to my old (?:house|address)"
        r"( and set it as my account default)?(?=[.!?]|$)", text, re.I,
    )
    if not copy or not user_id:
        return None
    other_text = text[:copy.start()] + "\n" + text[copy.end():]
    if re.search(r"\b(?:copy|change|update|modify|switch|set|send|ship)\b[^.!?\n]*"
                 r"\b(?:address|account default)\b", other_text, re.I):
        return None
    user = business["users"][user_id]
    orders = [business["orders"][key] for key in user["orders"]]
    owned = [order for order in orders if order["user_id"] == user_id]
    sources = [order for order in owned if any(
        re.search(r"\b" + re.escape(copy[1]) + r"\b", item["name"], re.I) for item in order["items"]
    )]
    if len(sources) != 1 or sources[0]["address"] == user["address"]:
        return None
    source = sources[0]
    destinations = [order for order in owned if order["address"] == user["address"]
                    and order["status"] in {"pending", "pending (item modified)"}]
    if len(destinations) != 1:
        return None
    return {"source_order_id": source["order_id"], "destination_order_id": destinations[0]["order_id"],
            "address": source["address"], "update_default": bool(copy[2])}


def cross_product_cancel_target(text, business, user_id):
    """Resolve a uniquely named pending item in an explicit change, return, cancel chain."""
    request = re.fullmatch(
        r"\s*(?:please )?(?:change|replace|switch) my ([a-z -]+?) (?:to|with) an? ([a-z -]+?)\.\s*"
        r"if that (?:cannot be done|is impossible), return (?:that|the) item[.;]\s*"
        r"if that is also impossible, cancel the (?:whole|entire) order[.!]?", text, re.I,
    )
    if not request or not user_id:
        return None
    source, replacement = (value.lower() for value in request.groups())
    names = {product["name"].lower() for product in business["products"].values()}
    # Recognize explicit garment types even if the retail catalog has no entry for them.
    if replacement not in names | {"coat", "jacket", "shirt", "sweater", "dress"}:
        return None
    if replacement in source.split() or source == replacement:
        return None
    orders = business["orders"]
    matches = [(order_id, item) for order_id in business["users"][user_id]["orders"]
               if orders[order_id]["user_id"] == user_id
               for item in orders[order_id]["items"] if item["name"].lower() == source]
    if len(matches) != 1:
        return None
    order_id, item = matches[0]
    if orders[order_id]["status"] != "pending":
        return None
    variants = business["products"][item["product_id"]]["variants"]
    if any(replacement in {str(value).lower() for value in variant["options"].values()}
           for variant in variants.values()):
        return None
    return order_id


def item_cancel_fallbacks(text, business, user_id):
    """Resolve explicit item fallbacks only when every write clause can be verified."""
    if not user_id:
        return [], ""
    request = re.compile(
        r"(?:please |can you )?(?:(?:change|modify) my (?P<name>[a-z -]+?) to "
        r"(?P<options>[a-z0-9 ,.-]+?)(?:, keeping its (?P<keep>[a-z ,]+?))?"
        r"[.?;] +if (?:that option is )?unavailable|"
        r"remove (?:the|my) (?P<removed>[a-z -]+?)(?: i ordered by mistake)?"
        r"; +if removing one item is impossible), +cancel "
        r"(?:the|that) (?:whole |entire )?order"
        r"(?: as (?P<reason>ordered by mistake|no longer needed)| and i will buy again)?(?:[.!] +|[.!]?$)",
        re.I,
    )
    owned = [business["orders"][key] for key in business["users"][user_id]["orders"]
             if business["orders"][key]["user_id"] == user_id]
    remaining = text.strip().lower()
    steps = []
    while match := request.match(remaining):
        clause = match.groupdict()
        name = clause["name"] or clause["removed"]
        matches = [(order, item) for order in owned for item in order["items"] if item["name"].lower() == name]
        if len(matches) != 1 or matches[0][0]["status"] not in {"pending", "cancelled"}:
            return [], ""
        order, item = matches[0]
        explanation = "Removing just one item from the order was unavailable."
        if clause["options"]:
            variants = business["products"][item["product_id"]]["variants"]
            keep = re.split(r", *(?:and )?| and ", clause["keep"]) if clause["keep"] else []
            if any(key not in item["options"] for key in keep):
                return [], ""
            wanted = {key: item["options"][key] for key in keep}
            for option in map(str.strip, clause["options"].split(",")):
                option = re.sub(r"^(\d+) (?:foot|feet|ft)(?: tall)?$", r"\1 ft", option)
                hits = {(key, value) for variant in variants.values() for key, value in variant["options"].items()
                        if str(value).lower() == option}
                if len(hits) != 1:
                    return [], ""
                key, value = hits.pop()
                if key in wanted and wanted[key] != value:
                    return [], ""
                wanted[key] = value
            if any(v["available"] is True and all(v["options"].get(k) == value for k, value in wanted.items())
                   for v in variants.values()):
                return [], ""
            explanation = "The requested product variant was unavailable."
        if any(step["order_id"] == order["order_id"] for step in steps):
            return [], ""
        steps.append({"order_id": order["order_id"], "reason": clause["reason"], "explanation": explanation})
        remaining = remaining[match.end():]
    if remaining and not re.fullmatch(r"(?:also )?tell me the total paid for (?:all |the )?"
                                      r"[a-z -]+ in (?:my )?prior orders[.!?]?", remaining):
        return [], ""
    return (steps, remaining) if steps else ([], "")


def item_order_excluded(text, order_id):
    """Check an explicit order exclusion even when the alternative cannot be identified."""
    return (
        len(re.findall(r"\b(?:modify|change|switch|replace|update)\b", text, re.I)) == 1
        and order_id.lower() in re.findall(r"#w\d+", text.lower())
        and bool(re.search(r"(?:^|[.!?]\s+)(?:please )?(?:modify|change|switch) "
                           r"[^.!?]+ in (?:another|the other) order to ", text, re.I))
    )


def item_change_targets(text, business, user_id):
    """Resolve the old item only from an unambiguous product and original option description."""
    if not user_id or not re.search(r"\b(?:modify|change|switch)\b", text, re.I):
        return []
    orders = [business["orders"][key] for key in business["users"][user_id]["orders"]]
    targets = []
    order_references = set(re.findall(r"#w\d+", text.lower()))
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text.lower()):
        change = re.fullmatch(
            r"(?:please )?(?:modify|change|switch) (?:the |my )?(pending )?(.+?) to ([^.!?]+)[.]?", sentence.strip()
        )
        if not change or re.search(r"\b(?:not|never|except|instead|and|or)\b", change[2]):
            continue
        description = re.sub(r" in (?:another|the other) order$", "", change[2])
        excluded = order_references if description != change[2] else set()
        matches = []
        for order in orders:
            if order["user_id"] != user_id or order["status"] != "pending" or order["order_id"].lower() in excluded:
                continue
            for item in order["items"]:
                name = item["name"].lower()
                if description == name and change[1]:
                    matches.append((order, item))
                elif description.endswith(" " + name):
                    modifier = description[:-(len(name) + 1)]
                    if modifier in {str(value).lower() for value in item["options"].values()}:
                        matches.append((order, item))
        if len(matches) == 1:
            order, item = matches[0]
            targets.append({"order_id": order["order_id"], **item, "requested_variant": change[3]})
    return targets


def wrong_item_target(text, targets, order, args):
    if len(targets) == 1 and len(re.findall(r"\b(?:modify|change|switch|replace|update)\b", text, re.I)) == 1:
        target = targets[0]
        if (args["order_id"], args["item_ids"]) != (target["order_id"], [target["item_id"]]):
            return target
        return None
    allowed = {}
    for target in targets:
        allowed.setdefault(target["name"], set()).add((target["order_id"], target["item_id"]))
    for item in order["items"]:
        scope = allowed.get(item["name"])
        if item["item_id"] in args["item_ids"] and scope and (order["order_id"], item["item_id"]) not in scope:
            return next(target for target in targets if target["name"] == item["name"])
    return None


def requested_variants(variants, preference):
    """Return available matches for a literal option or lowest price, or None for a free-form request."""
    available = [{"item_id": item_id, "options": variant["options"], "price": variant["price"]}
                 for item_id, variant in variants.items() if variant["available"] is True]
    if preference == "the cheapest available version":
        lowest = min((variant["price"] for variant in available), default=None)
        return [variant for variant in available if variant["price"] == lowest]
    if any(preference in {str(value).lower() for value in variant["options"].values()}
           for variant in variants.values()):
        return [variant for variant in available
                if preference in {str(value).lower() for value in variant["options"].values()}]
    return None


def preferred_variant_matches(candidates, customer_text):
    preference = re.search(r"\bI prefer ([\w-]+) over ([\w-]+) and ([\w-]+) or ([\w-]+) "
                           r"over other colors[.!]?\s*$", customer_text, re.I)
    if not preference or preference[1].casefold() == preference[2].casefold():
        return []
    processor = preference[1].casefold()
    colors = {preference[3].casefold(), preference[4].casefold()}
    return [row for row in candidates if str(row["options"].get("processor", "")).casefold() == processor
            and str(row["options"].get("color", "")).casefold() in colors]


def preserved_variant_conflicts(customer_text, business, user_id, args):
    order = business["orders"].get(args["order_id"])
    if not order or order["user_id"] != user_id:
        return []
    preserved = {}
    for sentence in re.split(r"[.!?\n]", customer_text.lower()):
        references = set(re.findall(r"#w\d+", sentence))
        if references and references != {args["order_id"].lower()}:
            continue
        if re.search(r"\b(?:not|never|don't)\s+(?:change|modify|switch)\b", sentence):
            continue
        for change in re.split(r"\b(?:change|modify|switch)\s+", sentence)[1:]:
            keep = re.search(r"\bkeep(?:ing)?\s+(?:its|the same)\s+([a-z ,]+)", change)
            if not keep:
                continue
            description, separator, _ = change[:keep.start()].partition(" to ")
            if not separator or re.search(r"\b(?:not|never|don't|instead|except)\b", description):
                continue
            names = {item["name"].lower() for item in order["items"]
                     if re.search(r"(?<!\w)" + re.escape(item["name"].lower()) + r"(?!\w)", description)}
            if len(names) == 1:
                preserved.setdefault(names.pop(), set()).update(
                    key.strip() for key in re.split(r",|\band\b", keep[1]))

    conflicts = set()
    for old_id, new_id in zip(args["item_ids"], args["new_item_ids"]):
        old = next((item for item in order["items"] if item["item_id"] == old_id), None)
        if old and (keys := preserved.get(old["name"].lower())):
            variant = business["products"][old["product_id"]]["variants"].get(new_id)
            if variant:
                conflicts.update(key for key in old["options"].keys() & keys
                                 if old["options"][key] != variant["options"].get(key))
    return sorted(conflicts)


def false_cancellation_claim(text, business, user_id, committed_orders=(), *, unresolved_cancellation=False):
    start = r"^[\s>*`_#✅-]*(?:\d+\.\s*)?(?:[*`_]+\s*)?"
    order_ref = r"(?:(?:your\s+)?order\s+)?(#W\d+)[*`_]*(?:\s*\([^\n()]{1,80}\))?[*`_]*\s+(?:[—–-]\s+)?"
    cancel = r"(?:successfully\s+)?cancel(?:l)?ed\b"
    completion = r"(?:initiated|confirmed|completed|processed)\b"
    for sentence in re.split(r"(?<=[.!?])\s+|\n|;\s*|,\s+and\s+|\s+and\s+(?=" + cancel + ")",
                             text, flags=re.I):
        if "?" in sentence:
            continue
        operation = re.match(start + r"(?:(?:I\s+(?:have\s+)?(?:just\s+)?)?" + cancel
                             + r"\s+(?:order\s+)?[*`_]*(#W\d+)\b|Cancellation of\s+" + order_ref
                             + r"(?:was|has been|is now)\s+" + completion + ")", sentence, re.I)
        if operation and (operation[1] or operation[2]).upper() not in committed_orders:
            return True
        if unresolved_cancellation and re.match(
                start + r"(?:(?:your|the) orders? (?:was|were|has been|have been|is now)\s+" + cancel
                + r"|(?:your|the) cancellation (?:was|has been|is now)\s+" + completion + ")", sentence, re.I):
            return True
        status = re.match(start + order_ref + r"((?:was|is|has)\s+(?:now\s+)?(?:been\s+)?)?"
                          + cancel + r"(\s+for you\b)?", sentence, re.I)
        if status:
            order_id = status[1].upper()
            order = business["orders"].get(order_id)
            if not order or order["user_id"] != user_id or order["status"] != "cancelled":
                return True
            if order_id not in committed_orders and ("now " in (status[2] or "").lower() or status[3]):
                return True
    return False
