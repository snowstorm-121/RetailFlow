"""Prepare a small, auditable local derivative of the fixed tau2 retail tasks."""

import hashlib
import json
from copy import deepcopy
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "LICENSE", "pyproject.toml", "CHANGELOG.md",
    "data/tau2/domains/retail/db.json",
    "data/tau2/domains/retail/policy.md",
    "data/tau2/domains/retail/tasks.json",
    "data/tau2/domains/retail/split_tasks.json",
    "src/tau2/domains/retail/tools.py",
    "src/tau2/domains/retail/data_model.py",
    "src/tau2/domains/retail/utils.py",
    "src/tau2/domains/retail/environment.py",
    "src/tau2/environment/toolkit.py",
)
ALLOWED_TOOLS = {
    "find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details",
    "get_order_details", "get_product_details", "cancel_pending_order",
    "return_delivered_order_items", "transfer_to_human_agents",
}
WRITE_TOOLS = {"cancel_pending_order", "return_delivered_order_items"}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _source(config):
    path = Path(config["source_root"])
    return path if path.is_absolute() else ROOT / path


def agent_initial_message(task: dict) -> dict:
    """The sole task field allowed into the initial production user message."""
    return {"role": "user", "content": task["initial_user_message"]}


def check_final_state(task: dict, before: dict, after: dict, events: list[dict]) -> dict:
    """Score recorded business state and required user-visible effects, independent of answer fluency."""
    failures = []
    checks = task["expected_checks"]
    targets = {check["order_id"]: check for check in checks["orders"]}
    if before["products"] != after["products"]:
        failures.append("product_side_effect")
    if before["orders"].keys() != after["orders"].keys():
        failures.append("order_set_changed")
    for order_id, original in before["orders"].items():
        actual = after["orders"].get(order_id)
        expected = targets.get(order_id)
        if not expected:
            if actual != original:
                failures.append("other_order_side_effect")
            continue
        if not actual or any(actual.get(key) != value for key, value in expected.items() if key != "order_id"):
            failures.append("target_state_mismatch")
            continue
        allowed = {"status", "cancel_reason", "payment_history"} if expected["status"] == "cancelled" else {"status", "return_items", "return_payment_method_id"}
        if any(actual.get(key) != value for key, value in original.items() if key not in allowed) or set(actual) - set(original) - allowed:
            failures.append("target_extra_change")
        if expected["status"] == "cancelled":
            refunds = [{"transaction_type": "refund", "amount": p["amount"], "payment_method_id": p["payment_method_id"]} for p in original["payment_history"]]
            if actual["payment_history"] != original["payment_history"] + refunds:
                failures.append("refund_mismatch")
        elif actual["payment_history"] != original["payment_history"]:
            failures.append("unexpected_payment_change")
    expected_users = deepcopy(before["users"])
    user_id = task["group_id"].removeprefix("source-user:")
    for expected in checks["orders"]:
        if expected["status"] != "cancelled":
            continue
        for payment in before["orders"][expected["order_id"]]["payment_history"]:
            method = expected_users[user_id]["payment_methods"][payment["payment_method_id"]]
            if method["source"] == "gift_card":
                method["balance"] = round(method["balance"] + payment["amount"], 2)
    if after["users"] != expected_users:
        failures.append("user_side_effect_or_refund_mismatch")
    confirmed = [e["order_id"] for e in events if e.get("type") == "confirmed_write"]
    if Counter(confirmed) != Counter(targets.keys()):
        failures.append("missing_confirmation")
    visible_text = " ".join(e.get("text", "") for e in events if e.get("type") == "assistant_message")
    if any(str(info).lower() not in visible_text.lower() for info in checks["communicate_info"]):
        failures.append("missing_communication")
    if checks["transfer_to_human"] != any(e.get("type") == "transfer_to_human_agents" for e in events):
        failures.append("transfer_mismatch")
    return {"passed": not failures, "failures": sorted(set(failures))}


def _make_task(source_task, chosen, split, db, db_hash):
    criteria = source_task["evaluation_criteria"]
    actions = criteria.get("actions") or []
    for action in actions:
        order_id = action["arguments"].get("order_id")
        if order_id and (order_id not in db["orders"] or db["orders"][order_id]["user_id"] != chosen["user_id"]):
            raise ValueError(f"task {source_task['id']}: referenced order belongs to another user or is missing")
    writes = []
    for action in actions:
        if action["name"] not in WRITE_TOOLS:
            continue
        args = action["arguments"]
        order = db["orders"][args["order_id"]]
        if order["user_id"] != chosen["user_id"]:
            raise ValueError(f"task {source_task['id']}: order belongs to another user")
        if action["name"] == "cancel_pending_order":
            if order["status"] != "pending" or args["reason"] not in {"no longer needed", "ordered by mistake"}:
                raise ValueError(f"task {source_task['id']}: invalid cancellation")
            writes.append({"order_id": args["order_id"], "status": "cancelled", "cancel_reason": args["reason"]})
        else:
            user = db["users"][chosen["user_id"]]
            ids = [item["item_id"] for item in order["items"]]
            method = args["payment_method_id"]
            original = order["payment_history"][0]["payment_method_id"]
            if order["status"] != "delivered" or not set(args["item_ids"]) <= set(ids):
                raise ValueError(f"task {source_task['id']}: invalid return items/status")
            if method not in user["payment_methods"] or not (method == original or method.startswith("gift_card_")):
                raise ValueError(f"task {source_task['id']}: invalid refund method")
            writes.append({"order_id": args["order_id"], "status": "return requested", "return_items": sorted(args["item_ids"]), "return_payment_method_id": method})
    facts = source_task["user_scenario"]["instructions"]
    responses = {"identity": facts["known_info"], **chosen.get("field_responses", {})}
    script = {
        "field_responses": responses,
        "proposal_rules": chosen.get("proposal_rules", []),
        "default_proposal_response": "accept_matching_intent",
        "max_user_turns": 8,
        "on_unknown": "script_unhandled",
        "terminal": "goal_complete_or_declined",
    }
    return {
        "task_id": f"tau-retail-{source_task['id']}",
        "group_id": f"source-user:{chosen['user_id']}",
        "origin": "adapted",
        "source_task_id": source_task["id"],
        "source_split": split,
        "split": "dev" if split == "train" else "test",
        "category": chosen["category"],
        "initial_state_ref": f"retail-db@{db_hash}",
        "initial_user_message": chosen["opening"],
        "user_private_facts": {
            "known_info": facts["known_info"],
            "unknown_info": facts["unknown_info"],
            "intent": facts["reason_for_call"],
        },
        "script": script,
        "expected_checks": {
            "orders": writes,
            "communicate_info": criteria.get("communicate_info") or [],
            "transfer_to_human": any(a["name"] == "transfer_to_human_agents" for a in actions),
        },
        "forbidden_effects": {
            "other_orders_unchanged": True,
            "no_unconfirmed_write": True,
            "no_extra_return_items": True,
        },
        "fault_plan": {},
    }


def prepare_dataset(config: dict) -> dict:
    """Build versioned task files without overwriting a changed frozen version."""
    source = _source(config)
    output = Path(config.get("output_root", ROOT))
    data = source / "data/tau2/domains/retail"
    tasks = json.loads((data / "tasks.json").read_text())
    split_ids = json.loads((data / "split_tasks.json").read_text())
    db = json.loads((data / "db.json").read_text())
    db_hash = _hash((data / "db.json").read_bytes())
    if set(split_ids["train"]) & set(split_ids["test"]) or set(split_ids["base"]) != set(split_ids["train"]) | set(split_ids["test"]):
        raise ValueError("upstream split is inconsistent")
    by_id = {task["id"]: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate upstream task IDs")
    chosen = config["selected_tasks"]
    if len({entry["id"] for entry in chosen}) != len(chosen):
        raise ValueError("duplicate selected source IDs")
    selected = []
    for entry in chosen:
        task = by_id[entry["id"]]
        if entry["user_id"] not in db["users"]:
            raise ValueError(f"task {entry['id']}: missing user")
        if not all(a["name"] in ALLOWED_TOOLS for a in task["evaluation_criteria"].get("actions") or []):
            raise ValueError(f"task {entry['id']}: unsupported reference action")
        upstream_split = "train" if entry["id"] in split_ids["train"] else "test"
        selected.append(_make_task(task, entry, upstream_split, db, db_hash))
    groups = {"dev": sorted({t["group_id"] for t in selected if t["split"] == "dev"}),
              "test": sorted({t["group_id"] for t in selected if t["split"] == "test"})}
    if set(groups["dev"]) & set(groups["test"]):
        raise ValueError("source user group crosses dev/test")
    excluded = Counter()
    selected_ids = {entry["id"] for entry in chosen}
    for task in tasks:
        if task["id"] in selected_ids:
            continue
        names = {a["name"] for a in task["evaluation_criteria"].get("actions") or []}
        excluded["unsupported_action" if names - ALLOWED_TOOLS else "not_selected_after_full_goal_review"] += 1
    source_manifest = {
        "repository": "https://github.com/sierra-research/tau2-bench",
        "commit": config["upstream_commit"],
        "license": "MIT (repository LICENSE; redistribution of raw task/db data requires separate review)",
        "source_root": str(source),
        "files": {name: {"bytes": (source / name).stat().st_size, "sha256": _hash((source / name).read_bytes())} for name in SOURCE_FILES},
    }
    files = {}
    for split in ("dev", "test"):
        rows = sorted((t for t in selected if t["split"] == split), key=lambda t: int(t["source_task_id"]))
        files[f"data/derived/{split}.jsonl"] = b"".join(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows
        )
    files["data/manifests/sources.json"] = _json_bytes(source_manifest)
    counts = {"dev": sum(t["split"] == "dev" for t in selected), "test": sum(t["split"] == "test" for t in selected), "total": len(selected)}
    manifest = {
        "version": config["dataset_version"], "seed": config["seed"],
        "source_commit": config["upstream_commit"],
        "source_db_sha256": db_hash,
        "selected_source_ids": sorted(selected_ids, key=int),
        "counts": counts, "categories": dict(Counter(t["category"] for t in selected)),
        "groups": groups, "excluded_counts": dict(excluded),
        "file_hashes": {name: _hash(content) for name, content in files.items()},
        "review_status": "assistant_source_and_static_checks_only; runtime_checker_pending_T3_T6",
    }
    files["data/manifests/dataset.json"] = _json_bytes(manifest)
    for name, content in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"frozen file differs: {path}; create a new dataset version")
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
    return manifest


def validate_dataset(manifest_path: str) -> dict:
    """Recompute hashes, split/group boundaries and source-state preconditions."""
    path = Path(manifest_path).resolve()
    root = path.parents[2]
    errors = []
    manifest = json.loads(path.read_text())
    sources = json.loads((root / "data/manifests/sources.json").read_text())
    for name, digest in manifest["file_hashes"].items():
        target = root / name
        if not target.exists() or _hash(target.read_bytes()) != digest:
            errors.append(f"hash mismatch: {name}")
    source_root = root / "vendor/tau_retail/source"
    if not source_root.exists():  # Frozen T2 manifests also validate in isolated output directories.
        source_root = Path(sources["source_root"])
    for name, info in sources["files"].items():
        target = source_root / name
        if not target.exists() or target.stat().st_size != info["bytes"] or _hash(target.read_bytes()) != info["sha256"]:
            errors.append(f"source hash mismatch: {name}")
    db = json.loads((source_root / "data/tau2/domains/retail/db.json").read_text())
    source_splits = json.loads((source_root / "data/tau2/domains/retail/split_tasks.json").read_text())
    source_tasks = {t["id"]: t for t in json.loads((source_root / "data/tau2/domains/retail/tasks.json").read_text())}
    tasks = []
    for split in ("dev", "test"):
        for line in (root / f"data/derived/{split}.jsonl").read_text().splitlines():
            task = json.loads(line)
            tasks.append(task)
            if task["split"] != split or task["source_split"] != ("train" if split == "dev" else "test"):
                errors.append(f"split mismatch: {task['task_id']}")
            if task["source_task_id"] not in source_splits["train" if split == "dev" else "test"]:
                errors.append(f"upstream split mismatch: {task['task_id']}")
            if task["source_task_id"] not in source_tasks:
                errors.append(f"missing source task: {task['task_id']}")
            if task["initial_state_ref"] != f"retail-db@{manifest['source_db_sha256']}":
                errors.append(f"initial state mismatch: {task['task_id']}")
            script = task["script"]
            allowed_fields = {"identity", "order", "cancel_reason", "return_items", "refund_method"}
            if (not task["initial_user_message"] or not script["field_responses"].get("identity")
                    or set(script["field_responses"]) - allowed_fields
                    or script["on_unknown"] != "script_unhandled" or script["max_user_turns"] < 1):
                errors.append(f"invalid user script: {task['task_id']}")
            if set(task) - {"task_id", "group_id", "origin", "source_task_id", "source_split", "split", "category", "initial_state_ref", "initial_user_message", "user_private_facts", "script", "expected_checks", "forbidden_effects", "fault_plan"}:
                errors.append(f"agent-visible or unknown task field: {task['task_id']}")
            if agent_initial_message(task).keys() != {"role", "content"}:
                errors.append(f"agent projection invalid: {task['task_id']}")
            user_id = task["group_id"].removeprefix("source-user:")
            if user_id not in db["users"]:
                errors.append(f"missing user: {task['task_id']}")
            for rule in script["proposal_rules"]:
                order = db["orders"].get(rule.get("order_id")) if rule.get("order_id") else None
                if (rule.get("tool") not in WRITE_TOOLS or rule.get("decision") not in {"decline", "revise", "accept"}
                        or (order and order["user_id"] != user_id)
                        or ("order_id" in rule and not order)
                        or ("unless_item_name" in rule and (not order or rule["unless_item_name"] not in {item["name"] for item in order["items"]}))):
                    errors.append(f"invalid proposal branch: {task['task_id']}")
            source_task = source_tasks.get(task["source_task_id"])
            if source_task:
                for action in source_task["evaluation_criteria"].get("actions") or []:
                    order_id = action["arguments"].get("order_id")
                    if order_id and db["orders"].get(order_id, {}).get("user_id") != user_id:
                        errors.append(f"source order/group mismatch: {task['task_id']}")
            for check in task["expected_checks"]["orders"]:
                order = db["orders"].get(check["order_id"])
                if not order or order["user_id"] != user_id:
                    errors.append(f"missing or foreign order: {task['task_id']}")
                elif check["status"] == "cancelled" and order["status"] != "pending":
                    errors.append(f"invalid cancellation precondition: {task['task_id']}")
                elif check["status"] == "return requested" and (order["status"] != "delivered" or not set(check["return_items"]) <= {i["item_id"] for i in order["items"]}):
                    errors.append(f"invalid return precondition: {task['task_id']}")
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        errors.append("duplicate task ID")
    if sorted((t["source_task_id"] for t in tasks), key=int) != manifest["selected_source_ids"]:
        errors.append("selected source IDs differ from manifest")
    groups = {split: {t["group_id"] for t in tasks if t["split"] == split} for split in ("dev", "test")}
    if groups["dev"] & groups["test"]:
        errors.append("dev/test group overlap")
    counts = {"dev": sum(t["split"] == "dev" for t in tasks), "test": sum(t["split"] == "test" for t in tasks), "total": len(tasks)}
    if counts != manifest["counts"] or any(sorted(groups[k]) != manifest["groups"][k] for k in groups):
        errors.append("manifest count/group mismatch")
    return {"valid": not errors, "counts": counts, "errors": errors}
