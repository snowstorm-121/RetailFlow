"""Prepare address tasks from the fixed source, preserving prior account assignments."""

import json

from retailflow.dataset import ROOT, prepare_dataset
from retailflow.tools import WRITE_TOOLS
from retailflow.upstream.paths import RETAIL_DB_PATH

OPENINGS = {
    "17": "For order #W8665881, change only the suite to Suite 641. Keep the rest of its shipping address.",
    "33": "I no longer need the office items in my pending Seattle order and want to keep the hiking items. If partial cancellation is possible, tell me the refund total and use my original payment method. If it is impossible, leave that order alone and change my account default to my parents' Seattle address from the order.",
    "34": "I no longer need the office items in my pending Seattle order and want to keep the hiking items. If partial cancellation is possible, tell me the refund total and use my original payment method. Otherwise send the entire pending order to my NYC address instead. I'll handle any returns later.",
    "39": "I moved from Florida to Phoenix. Change my account default to the Phoenix address from my recent order. Tell me the cheapest available T-shirt price and whether I can order it through you.",
    "42": "Check my account and order addresses. I think I entered them incorrectly. Correct my default and every eligible order address to my real address. After that, if my jigsaw order is still pending, switch to the easiest puzzle with the fewest pieces. Use PayPal.",
    "43": "Tell me the shipping address, tracking number and whether the order for my daughter in Chicago is still in transit. Also check the tablet storage in that order. Set my account default to my daughter's shipping address from the order.",
    "59": "Compare pending orders #W2702727 and #W8268610. If you cannot guarantee the older one will be processed within five days, cancel it and tell me the refund total. Change shipping for #W2702727 to 1234 Elm St, Springfield, IL 62701, USA, no suite. Confirm the new address and total order price.",
    "78": "Copy the shipping address from order #W8277957 to #W5056519. In #W5056519 change the light professional Brand B makeup kit to a dark Brand A kit. Also cancel #W5995614 because I ordered it by mistake.",
    "86": "Change my fleece jacket to red with a half zipper. Update my account default address to my Washington DC address from an existing order.",
    "87": "Copy my Washington DC address from an existing order to all my pending orders and my account default.",
    "97": "Change the Bluetooth speaker in my LA order to the cheapest available green variant. Also switch the shipping address for that order to my NYC address from my other order.",
    "101": "Change the order containing two watches to my New York address from another order. Switch the silicone watch to a metal one, preferably white. In my other order with an air purifier and speaker, modify the purifier to large with night mode, keeping HEPA.",
    "102": "Change the order containing two watches to my New York address from another order. Switch the silicone watch to a metal one, preferably white. Exchange the delivered air purifier that came with sneakers for a large version with night mode and HEPA.",
    "103": "Cancel the bookshelf and jigsaw from the same order I received. Also return the backpack that came with my vacuum cleaner. Change the pending order to my default Chicago address and its item color to red. Tell me the tracking number of my cancelled order.",
    "104": "Cancel all the bookshelves and jigsaw puzzles I received in separate orders. Return the backpack I received with my vacuum cleaner. Change my pending order item to red and ship it to my Chicago address from my profile. Tell me the tracking number of my cancelled order.",
    "109": "I moved to another house on Elm Avenue in Houston. Copy the shipping address from my luggage order to the order still going to my old house and set it as my account default. Change my pending tablet to the cheapest available version.",
    "111": "Use the NYC address from one of my orders for my pending laptop order. Change the laptop to i5, 256GB SSD, space grey. Also switch my watch to a black dial while keeping the leather strap.",
    "112": "Use the NYC address from one of my orders for my pending laptop order, and change the laptop to item 9844888101. Also change my watch to a black dial, keeping its leather strap.",
}
EXCLUDED = {
    "22": "Requires a post-commit change of mind and verification that the default address was changed twice in sequence.",
    "41": "Same reference writes and business goals as task 42, with a different request order.",
    "71": "Needs an initial gift-card item proposal, then a PayPal revision which drops the desk lamp.",
    "72": "Same final writes and change-of-mind goal as task 71.",
    "96": "Same reference writes and business goals as task 97.",
    "110": "The requested default address is already set in the fixed database. The reference insists on a write with no state change.",
}


def build():
    config = json.loads((ROOT / "configs/coverage-evaluation.json").read_text())
    data = json.loads(RETAIL_DB_PATH.read_text())
    sources = {task["id"]: task for task in json.loads(RETAIL_DB_PATH.with_name("tasks.json").read_text())}
    splits = json.loads(RETAIL_DB_PATH.with_name("split_tasks.json").read_text())
    groups = config["group_assignments"]
    for sid, opening in OPENINGS.items():
        source = sources[sid]
        writes = [{"name": a["name"], "arguments": a["arguments"]}
                  for a in source["evaluation_criteria"]["actions"] if a["name"] in WRITE_TOOLS]
        first = writes[0]["arguments"]
        uid = first.get("user_id") or data["orders"][first["order_id"]]["user_id"]
        profile = data["users"][uid]
        known = source["user_scenario"]["instructions"]["known_info"]
        name = profile["name"]
        identity = (f"My email is {profile['email']}." if profile["email"] in known else
                    f"My name is {name['first_name']} {name['last_name']}, ZIP {profile['address']['zip']}.")
        replies = {
            "identity": identity,
            "address": "Please look up the address as I described.",
            "order": "Please look for the matching order in my account.",
            "exchange_items": opening,
            "exchange_options": opening,
            "return_items": opening,
        }
        if sid == "42":
            replies["address"] = "445 Maple Drive, Suite 394, Fort Worth, TX 76165, USA."
        if sid in {"59", "78"}:
            replies["cancel_reason"] = next(a["arguments"]["reason"] for a in writes if a["name"] == "cancel_pending_order")
        methods = {a["arguments"]["payment_method_id"] for a in writes if "payment_method_id" in a["arguments"]}
        if len(methods) == 1:
            method = profile["payment_methods"][next(iter(methods))]
            label = {"paypal": "PayPal", "gift_card": "gift card", "credit_card": "credit card"}[method["source"]]
            replies["payment_method"] = f"Use my {label}."
            replies["refund_method"] = "Use my original payment method."
        groups.setdefault(uid, "dev" if sid in splits["train"] else "test")
        config["selected_tasks"].append({"id": sid, "user_id": uid, "category": "legal_write",
                                         "opening": opening, "field_responses": replies})
        config["user_consent"][f"tau-retail-{sid}"] = {"approved_reference_actions": writes}
    root = "data/versions/t5-address-v1"
    config.update(dataset_root=root, output_root=root, dataset_version="t5-address-v1",
                  experiment_version="address-expansion-v1",
                  revision_note="Local address adaptations with source provenance and strict private proposal checks.")
    # These tasks ask for tracking information as well as a return receipt.
    config["preserve_return_answers"] = ["tau-retail-103", "tau-retail-104"]
    config["additional_tools"] += ["modify_pending_order_address", "modify_user_address"]
    manifest = prepare_dataset(config)
    (ROOT / "configs/address-evaluation.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "data/reviews/address-expansion.json").write_text(json.dumps({
        "included": list(OPENINGS), "deferred": EXCLUDED,
        "adaptation": ("Openings preserve the business requests. Presentation order and source persona are not scored. "
                       "Partial cancellation requests in 103/104 are voiced as cancellation while goods are delivered, "
                       "so the expected supported operation is return. Compare all requested communication manually. "
                       "Private consent uses exact reference parameters; it can reject other reasonable proposals."),
    }, indent=2) + "\n")
    print(manifest["counts"])


if __name__ == "__main__":
    build()
