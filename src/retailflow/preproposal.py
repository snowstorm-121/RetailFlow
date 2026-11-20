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
    order_ref = r"(?:order\s+)?(#W\d+)[*`_]*(?:\s*\([^\n()]{1,80}\))?[*`_]*\s+(?:[—–-]\s+)?"
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
