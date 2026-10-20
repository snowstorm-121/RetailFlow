import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from retailflow.dataset import agent_initial_message, check_final_state, prepare_dataset, validate_dataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation.json"


def test_prepare_and_validate_in_isolated_directory(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(tmp_path)
    manifest = prepare_dataset(config)
    result = validate_dataset(str(tmp_path / "data/manifests/dataset.json"))
    assert result["valid"], result["errors"]
    assert manifest["counts"] == result["counts"]
    assert manifest["counts"]["total"] < 80
    assert not set(manifest["groups"]["dev"]) & set(manifest["groups"]["test"])
    tasks = [json.loads(line) for split in ("dev", "test") for line in (tmp_path / f"data/derived/{split}.jsonl").read_text().splitlines()]
    assert len({task["task_id"] for task in tasks}) == len(tasks)
    assert all(task["initial_state_ref"] == f"retail-db@{manifest['source_db_sha256']}" for task in tasks)
    assert all(task["script"]["on_unknown"] == "script_unhandled" for task in tasks)
    assert all(agent_initial_message(task) == {"role": "user", "content": task["initial_user_message"]} for task in tasks)


def test_unsupported_upstream_goal_excluded(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(tmp_path)
    manifest = prepare_dataset(config)
    selected = set(manifest["selected_source_ids"])
    assert "0" not in selected  # exchange requires an unexposed write tool
    assert "5" not in selected  # conditional exchange cannot be silently shortened to return
    assert "19" not in selected  # return plus exchange is a single original goal
    assert "67" not in selected  # near-duplicate of upstream test task 68
    assert manifest["excluded_counts"]["unsupported_action"] > 0


def test_validation_detects_tampered_gold_and_group(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(tmp_path)
    prepare_dataset(config)
    path = tmp_path / "data/derived/test.jsonl"
    tasks = [json.loads(line) for line in path.read_text().splitlines()]
    tasks[0]["agent_visible_gold"] = tasks[0]["expected_checks"]
    path.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    result = validate_dataset(str(tmp_path / "data/manifests/dataset.json"))
    assert not result["valid"]
    assert any("hash" in error or "agent" in error for error in result["errors"])


def test_prepare_refuses_to_overwrite_changed_freeze(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(tmp_path)
    prepare_dataset(config)
    changed = copy.deepcopy(config)
    changed["selected_tasks"][0]["opening"] = "changed after freeze"
    with pytest.raises(FileExistsError):
        prepare_dataset(changed)


def test_near_duplicate_cannot_cross_upstream_splits(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(tmp_path)
    config["selected_tasks"].append({
        "id": "67", "category": "query", "user_id": "noah_ito_3850",
        "opening": "How much was my latest order?", "field_responses": {},
    })
    with pytest.raises(ValueError, match="group crosses"):
        prepare_dataset(config)


def test_validation_checks_source_state_even_if_hashes_are_updated(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(tmp_path)
    prepare_dataset(config)
    path = tmp_path / "data/derived/dev.jsonl"
    tasks = [json.loads(line) for line in path.read_text().splitlines()]
    tasks[0]["expected_checks"]["orders"][0]["order_id"] = "#WDOESNOTEXIST"
    path.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    manifest_path = tmp_path / "data/manifests/dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_hashes"]["data/derived/dev.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result = validate_dataset(str(manifest_path))
    assert not result["valid"]
    assert any("missing or foreign order" in error for error in result["errors"])


def test_frozen_dataset_validates_from_a_clone_path(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "vendor/tau_retail/source", tmp_path / "vendor/tau_retail/source")
    manifest = tmp_path / "data/manifests/dataset.json"
    assert validate_dataset(str(manifest))["valid"]
    source = tmp_path / "vendor/tau_retail/source/data/tau2/domains/retail/policy.md"
    source.write_text(source.read_text() + "\nchanged in clone\n")
    result = validate_dataset(str(manifest))
    assert not result["valid"]
    assert any("source hash mismatch" in error for error in result["errors"])


def test_final_state_checker_accepts_reference_and_rejects_side_effects(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(tmp_path)
    prepare_dataset(config)
    db = json.loads((ROOT / "vendor/tau_retail/source/data/tau2/domains/retail/db.json").read_text())
    tasks = [json.loads(line) for line in (tmp_path / "data/derived/dev.jsonl").read_text().splitlines()]
    for task in tasks:
        after = copy.deepcopy(db)
        for expected in task["expected_checks"]["orders"]:
            order_id = expected["order_id"]
            after["orders"][order_id].update(expected)
            if expected["status"] == "cancelled":
                for payment in db["orders"][order_id]["payment_history"]:
                    after["orders"][order_id]["payment_history"].append({
                        "transaction_type": "refund", "amount": payment["amount"],
                        "payment_method_id": payment["payment_method_id"],
                    })
                    method = after["users"][task["group_id"].removeprefix("source-user:")]["payment_methods"][payment["payment_method_id"]]
                    if method["source"] == "gift_card":
                        method["balance"] = round(method["balance"] + payment["amount"], 2)
        events = [{"type": "assistant_message", "text": " ".join(map(str, task["expected_checks"]["communicate_info"]))}]
        events += [{"type": "confirmed_write", "order_id": expected["order_id"]} for expected in task["expected_checks"]["orders"]]
        if task["expected_checks"]["transfer_to_human"]:
            events.append({"type": "transfer_to_human_agents"})
        assert check_final_state(task, db, after, events)["passed"], task["task_id"]
        victim = next(order_id for order_id in db["orders"] if order_id not in {e["order_id"] for e in task["expected_checks"]["orders"]})
        after["orders"][victim]["status"] = "cancelled"
        assert not check_final_state(task, db, after, events)["passed"]


def test_final_state_checker_requires_confirmation_and_communication(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(tmp_path)
    prepare_dataset(config)
    db = json.loads((ROOT / "vendor/tau_retail/source/data/tau2/domains/retail/db.json").read_text())
    task = next(json.loads(line) for line in (tmp_path / "data/derived/dev.jsonl").read_text().splitlines() if json.loads(line)["source_task_id"] == "2")
    after = copy.deepcopy(db)
    after["orders"][task["expected_checks"]["orders"][0]["order_id"]].update(task["expected_checks"]["orders"][0])
    result = check_final_state(task, db, after, [])
    assert not result["passed"]
    assert "missing_confirmation" in result["failures"]
    assert "missing_communication" in result["failures"]
