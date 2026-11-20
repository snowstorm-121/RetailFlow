"""Read-only evidence and checks before a retail proposal is shown."""

import re


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


def item_change_targets(text, business, user_id):
    """Resolve the old item only from an unambiguous product and original option description."""
    if not user_id or not re.search(r"\b(?:modify|change|switch)\b", text, re.I):
        return []
    orders = [business["orders"][key] for key in business["users"][user_id]["orders"]]
    targets = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text.lower()):
        change = re.fullmatch(
            r"(?:please )?(?:modify|change|switch) (?:the |my )?(pending )?(.+?) to ([^.!?]+)[.]?", sentence.strip()
        )
        if not change or re.search(r"\b(?:not|never|except|instead|and|or)\b", change[2]):
            continue
        description = re.sub(r" in (?:another|the other) order$", "", change[2])
        matches = []
        for order in orders:
            if order["user_id"] != user_id or order["status"] != "pending":
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


def wrong_item_target(targets, order, args):
    allowed = {}
    for target in targets:
        allowed.setdefault(target["name"], set()).add((target["order_id"], target["item_id"]))
    for item in order["items"]:
        scope = allowed.get(item["name"])
        if item["item_id"] in args["item_ids"] and scope and (order["order_id"], item["item_id"]) not in scope:
            return next(target for target in targets if target["name"] == item["name"])
    return None


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
    start = r"^[\s>*`_#-]*(?:\d+\.\s*)?(?:[*`_]+\s*)?"
    order_ref = r"(?:(?:your\s+)?order\s+)?(#W\d+)[*`_]*(?:\s*\([^\n()]{1,80}\))?[*`_]*\s+(?:[—–-]\s+)?"
    cancel = r"(?:successfully\s+)?cancel(?:l)?ed\b"
    completion = r"(?:initiated|confirmed|completed|processed)\b"
    for sentence in re.split(r"(?<=[.!?])\s+|\n|;\s*|,\s+and\s+", text, flags=re.I):
        if "?" in sentence:
            continue
        operation = re.match(start + r"(?:I\s+(?:have\s+)?(?:just\s+)?" + cancel
                             + r"\s+(?:order\s+)?(#W\d+)\b|Cancellation of\s+" + order_ref
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
