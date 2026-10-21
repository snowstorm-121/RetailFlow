"""Local derived retail evaluation: isolated episodes, role separation, and explicit denominators."""

import json
import hashlib
import importlib.metadata
import math
import resource
import re
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, ToolMessage

from retailflow.dataset import ROOT, check_final_state, validate_dataset
from retailflow.environment import RetailEnvironment
from retailflow.runtime import RETAIL_PROMPT, RetailRuntime
from retailflow.tools import IDENTITY_TOOLS, SCHEMAS, WRITE_TOOLS, valid_arguments
from retailflow.user_script import ScriptedUser, user_view


VARIANTS = {"B0": {"structured_context": False, "max_retry_per_call": 0},
            "B1": {"structured_context": False, "max_retry_per_call": 1},
            "B2": {"structured_context": True, "max_retry_per_call": 1}}
TERMINAL = {"answered", "refused", "handed_off"}


class FaultEnvironment(RetailEnvironment):
    """An evaluation fixture intercepting a registered read before its business method starts."""

    def __init__(self, path, plan):
        super().__init__(path)
        self.plan, self.seen, self.triggered = plan, 0, 0

    def dispatch_tool(self, thread_id, name, args):
        if name == self.plan.get("tool") and name in SCHEMAS and name not in WRITE_TOOLS and valid_arguments(name, args):
            located = self.load_session(thread_id).get("user_id")
            if located or name in IDENTITY_TOOLS:
                self.seen += 1
                if self.seen == self.plan.get("occurrence", 1):
                    self.triggered += 1
                    return {"ok": False, "code": "TEMPORARY_FAILURE", "retryable": True,
                            "message": "Registered temporary read failure before execution."}
        return super().dispatch_tool(thread_id, name, args)


def score_episode(task: dict, before: dict, after: dict, events: list[dict], status: str) -> dict:
    checked = check_final_state(task, before, after, events)
    state_failures = [failure for failure in checked["failures"] if failure != "missing_communication"]
    policy_failures = [event["type"] for event in events if event["type"] in {"unconfirmed_write", "unauthorized_read"}]
    state_passed = not state_failures and not policy_failures
    objective = state_passed and status in TERMINAL
    return {"task_id": task["task_id"], "category": task["category"], "status": status,
            "requires_write": bool(task["expected_checks"]["orders"]), "success": False,
            "objective_passed": objective, "state_passed": state_passed, "state_failures": state_failures,
            "policy_passed": not policy_failures, "policy_failures": policy_failures,
            "required_info_literal_match": "missing_communication" not in checked["failures"],
            "review_required": objective, "communication_review": None}


def apply_review(result: dict, review: dict) -> dict:
    if review.get("decision") not in {"pass", "fail"} or not review.get("reason"):
        raise ValueError("A communication review needs a pass/fail decision and evidence.")
    revised = deepcopy(result)
    revised["communication_review"] = review
    revised["review_required"] = False
    revised["success"] = bool(result["objective_passed"] and review["decision"] == "pass")
    if revised["success"]:
        revised["failure_reason"] = revised["failure_category"] = None
    if result["objective_passed"] and not revised["success"]:
        revised["failure_reason"] = "communication_failure"
        revised["failure_category"] = "communication"
    return revised


def _distribution(values):
    values = sorted(value for value in values if value is not None)
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "p95": values[math.ceil(len(values) * .95) - 1] if values else None}


def aggregate_results(results: list[dict]) -> dict:
    counts = Counter(result.get("category", "unclassified") for result in results)
    successes = Counter(result.get("category", "unclassified") for result in results if result["success"])
    writes = [result for result in results if result.get("requires_write")]
    triggered = [result for result in results if result.get("fault_triggered")]
    n = len(results)
    legal = sum(result.get("legal_tool_requests", 0) for result in results)
    tool_requests = sum(result.get("model_tool_requests", 0) for result in results)
    return {"n_assigned": n, "n_success": sum(result["success"] for result in results),
            "task_success_rate": sum(result["success"] for result in results) / n if n else None,
            "n_pending_reviews": sum(bool(result.get("review_required")) for result in results),
            "state_modification": {"numerator": sum(result.get("state_passed") is True for result in writes),
                                   "denominator": len(writes), "unknown": sum(result.get("state_passed") is None for result in writes)},
            "refusal_handoff": {"numerator": sum(r["success"] for r in results if r.get("category") == "refusal_handoff"),
                                "denominator": sum(r.get("category") == "refusal_handoff" for r in results)},
            "blocked_requests": sum(r.get("blocked_requests", 0) for r in results),
            "legal_tool_calls": {"numerator": legal, "denominator": tool_requests},
            "actual_violations": {"numerator": sum(result.get("policy_passed") is False for result in results),
                                  "denominator": n, "unknown": sum(result.get("policy_passed") is None for result in results)},
            "recovery": {"numerator": sum(result["success"] for result in triggered), "denominator": len(triggered),
                         "registered_but_untriggered": sum(bool(r.get("fault_registered")) and r.get("fault_triggered") == 0 for r in results),
                         "trigger_state_unknown": sum(bool(r.get("fault_registered")) and r.get("fault_triggered") is None for r in results)},
            "by_category": {category: {"n_assigned": count, "n_success": successes[category]} for category, count in counts.items()},
            "failure_reasons": dict(Counter(result.get("failure_reason") or "objective_or_review_failed" for result in results if not result["success"])),
            "failure_categories": dict(Counter(result.get("failure_category") or
                                                ("communication_review_pending" if result.get("review_required") else "unclassified")
                                                for result in results if not result["success"])),
            "efficiency": {label: {key: _distribution([r.get("elapsed_seconds") if key == "elapsed_seconds" else r.get("usage", {}).get(key) for r in subset])
                                    for key in ("elapsed_seconds", "model_calls", "tool_calls", "retries", "dialogue_repairs", "input_tokens", "output_tokens")}
                           for label, subset in {"all": results, "success": [r for r in results if r["success"]],
                                                 "failed": [r for r in results if not r["success"]]}.items()}}


def _memory():
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    service = None
    try:
        output = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True, timeout=3)
        if output.returncode == 0:
            service = sum(int(line.split(maxsplit=1)[0]) * 1024 for line in output.stdout.splitlines()
                          if "ollama" in line.lower())
    except (OSError, subprocess.TimeoutExpired):
        pass
    loaded = None
    try:
        loaded = sum(model["size"] for model in httpx.get("http://127.0.0.1:11434/api/ps", timeout=2).json()["models"])
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    return {"python_peak_rss_bytes": own if sys.platform == "darwin" else own * 1024,
            "ollama_observed_rss_bytes": service, "ollama_reported_loaded_model_bytes": loaded}


def _tool_stats(messages):
    total = legal = blocked = 0
    user_id = None
    exposures = []
    handoffs = []
    for message in messages:
        if isinstance(message, AIMessage):
            total += len(message.tool_calls) + len(message.invalid_tool_calls)
            for call in message.tool_calls:
                name, args = call["name"], call["args"]
                if name in SCHEMAS:
                    valid = valid_arguments(name, args)
                    if valid and "order_id" in args:
                        valid = bool(re.fullmatch(r"#W[0-9]+", args["order_id"]))
                    legal += int(valid)
                elif name == "ask_user":
                    legal += int(set(args) == {"fields", "question"} and isinstance(args["question"], str)
                                 and isinstance(args["fields"], list) and all(isinstance(f, str) for f in args["fields"])
                                 and bool(args["question"]) and bool(args["fields"])
                                 and all(f in {"identity", "order", "cancel_reason", "return_items", "refund_method"} for f in args["fields"])
                                 and len(set(args["fields"])) == len(args["fields"]))
        elif isinstance(message, ToolMessage):
            result = json.loads(message.content)
            blocked += result.get("code") in {"AUTH_REQUIRED", "POLICY_DENIED", "INVALID_ARGUMENT", "UNKNOWN_TOOL", "PROTOCOL_ERROR", "MISSING_USER_INFORMATION"}
            if result.get("ok") and message.name in IDENTITY_TOOLS:
                user_id = result["data"]
            if result.get("ok") and message.name in {"get_user_details", "get_order_details"}:
                if result["data"]["user_id"] != user_id:
                    exposures.append({"type": "unauthorized_read", "tool_call_id": message.tool_call_id})
            if result.get("ok") and message.name == "transfer_to_human_agents":
                handoffs.append({"type": "transfer_to_human_agents"})
    return {"model_tool_requests": total, "legal_tool_requests": legal, "blocked_requests": blocked}, exposures + handoffs


async def evaluate_episode(task: dict, variant: str, config: dict) -> dict:
    if variant not in VARIANTS:
        raise ValueError("Unknown evaluation variant")
    started = time.perf_counter()
    output = Path(config["output_dir"])
    thread = f"{variant}-{task['task_id']}-{uuid.uuid4().hex[:8]}"
    worker = config.get("runtime") or RetailRuntime(output / "business.sqlite", output / "traces")
    plan = config.get("fault_overrides", {}).get(task["task_id"], task.get("fault_plan", {}))
    if "runtime" not in config:
        worker.environment = FaultEnvironment(worker.environment.store_path, plan)
    execution = {**config.get("execution", {}), **VARIANTS[variant], "initial_state_ref": task["initial_state_ref"]}
    worker.environment.create_session(thread, task["initial_state_ref"])
    before = worker.environment.load_session(thread)["business"]
    before_hash = worker.environment.state_hash(thread)
    user = ScriptedUser(user_view(task, config.get("user_script_overrides", {}).get(task["task_id"])),
                        config.get("user_consent", {}).get(task["task_id"], {}))
    event = {"kind": "message", "text": task["initial_user_message"]}
    events, traces, accepted, followup, errors = [], [], {}, None, []
    usage = {key: 0 for key in ("model_calls", "tool_calls", "retries", "dialogue_repairs", "input_tokens", "output_tokens")}
    missing_tokens = False
    status, reason = "failed", None
    observed = _memory()
    for _ in range(min(task["script"].get("max_user_turns", 8), execution.get("max_user_turns_per_episode", 10))):
        events.append({"type": "user_event", **event})
        try:
            result = await worker.run_turn(thread, event, execution)
        except Exception as exc:
            status, reason = "failed", f"episode_exception:{type(exc).__name__}:{exc}"
            break
        traces.append(result["trace_path"])
        raw_trace = json.loads(Path(result["trace_path"]).read_text())
        errors.extend(attempt for attempt in raw_trace["usage"].get("tool_attempts", [])
                      if attempt["code"] not in {"OK", "CONFIRMATION_REQUIRED"})
        for key in usage:
            value = result["usage"].get(key)
            if value is None and key.endswith("tokens") and result["usage"]["model_calls"]:
                missing_tokens = True
            usage[key] += value or 0
        sampled = _memory()
        for key in observed:
            observed[key] = max(observed[key] or 0, sampled[key] or 0) or None
        status, reason = result["status"], result["failure_reason"]
        if status in TERMINAL:
            events.append({"type": "assistant_message", "text": result["text"]})
        if followup is not None and status == "refused":
            event, followup = {"kind": "message", "text": followup}, None
            continue
        try:
            if status == "needs_input":
                event = {"kind": "message", "text": user.answer(result["request"])}
            elif status == "needs_confirmation":
                proposal = result["request"]
                decision, followup = user.decide(proposal)
                if decision:
                    accepted[proposal["proposal_id"]] = deepcopy(proposal)
                event = {"kind": "confirmation", "proposal_id": proposal["proposal_id"], "accepted": decision}
            else:
                break
        except ValueError as exc:
            status, reason = "failed", "script_unhandled"
            events.append({"type": "script_error", "message": str(exc)})
            break
    else:
        status, reason = "budget_exceeded", "script_turn_budget"
    final = worker.load_session(thread)
    with worker.environment.storage.transaction() as db:
        receipts = [dict(row) for row in db.execute("SELECT * FROM receipts WHERE thread = ?", (thread,))]
    for receipt in receipts:
        response, args = json.loads(receipt["result_json"]), json.loads(receipt["args_json"])
        approved = accepted.get(response["proposal_id"])
        valid = approved and approved["name"] == receipt["name"] and approved["args"] == args
        events.append({"type": "confirmed_write" if valid else "unconfirmed_write", "order_id": args["order_id"],
                       "name": receipt["name"], "args": args, "proposal_id": response["proposal_id"]})
    stats, tool_events = _tool_stats(final["messages"])
    events.extend(tool_events)
    scored = score_episode(task, before, final["business"], events, status)
    if missing_tokens:
        usage["input_tokens"] = usage["output_tokens"] = None
    failure_category = ("script" if reason == "script_unhandled" else "budget_environment" if status in {"failed", "budget_exceeded"}
                        else "model_no_tool" if not stats["model_tool_requests"] else "model_decision_or_communication")
    if not scored["objective_passed"] and errors and not reason:
        failure_category = {"NOT_FOUND": "parameter_semantics", "INVALID_ARGUMENT": "parameter_semantics",
                            "AUTH_REQUIRED": "tool_selection", "POLICY_DENIED": "policy_understanding",
                            "TEMPORARY_FAILURE": "tool_exception", "TOOL_ERROR": "tool_exception"}.get(errors[0]["code"], "protocol")
    reason = reason or ("communication_review_pending" if scored["objective_passed"] else
                        scored["state_failures"][0] if scored["state_failures"] else "objective_failed")
    result = {**scored, **stats, "variant": variant, "thread_id": thread, "failure_reason": reason,
              "failure_category": failure_category if not scored["objective_passed"] else None,
              "fault_registered": bool(plan), "fault_triggered": getattr(worker.environment, "triggered", 0),
              "usage": usage, "elapsed_seconds": round(time.perf_counter() - started, 3), "memory": observed,
              "trace_paths": traces, "events": events,
              "first_observed_error": errors[0] if errors else None,
              "state_hash_before": before_hash,
              "state_hash_after": worker.environment.state_hash(thread),
              "expected_checks": task["expected_checks"]}
    return result


def build_manifest(config_path: Path, config: dict) -> dict:
    from react_agent.tools import RETAIL_TOOLS

    execution = config["execution"]
    if execution["base_url"] != "http://127.0.0.1:11434":
        raise ValueError("Evaluation requires the authorized local Ollama endpoint.")
    tags = httpx.get(execution["base_url"] + "/api/tags", timeout=5).json()["models"]
    model = next((model for model in tags if model["name"] == execution["model"]), None)
    if not model or model["digest"] != config["expected_model_digest"]:
        raise ValueError("The fixed local model is missing or its digest changed; no download was attempted.")
    runtime = httpx.get(execution["base_url"] + "/api/version", timeout=5).json()
    paths = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "uv.lock", config_path.resolve(),
             ROOT / "data/manifests/dataset.json", ROOT / "data/manifests/sources.json",
             ROOT / "data/derived/dev.jsonl", ROOT / "data/derived/test.jsonl"]
    manifest = {"version": "t6-v1", "model": model, "ollama_runtime": runtime, "execution": execution,
                "variants": VARIANTS, "config": config, "python": sys.version,
                "dependencies": {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()},
                "prompt_sha256": hashlib.sha256(RETAIL_PROMPT.encode()).hexdigest(),
                "tool_schema_sha256": hashlib.sha256(json.dumps(RETAIL_TOOLS, sort_keys=True).encode()).hexdigest(),
                "file_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return manifest


async def evaluate_batch(split: str, variant: str, config_path: Path, output: Path, limit: int | None = None) -> dict:
    checked = validate_dataset(ROOT / "data/manifests/dataset.json")
    if not checked["valid"]:
        raise ValueError(f"Frozen dataset failed validation: {checked['errors']}")
    config = json.loads(config_path.read_text())
    tasks = [json.loads(line) for line in (ROOT / f"data/derived/{split}.jsonl").read_text().splitlines()]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    if (output / "results.jsonl").exists():
        raise FileExistsError("Use a new output directory; existing evaluation results are immutable.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config_path, config)
    manifest["assigned_task_ids"] = [task["task_id"] for task in tasks]
    manifest["split"], manifest["variant"] = split, variant
    # The environment/code freeze hash is separate from the run selection and its manifest file hash.
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    results = []
    for task in tasks:
        try:
            result = await evaluate_episode(task, variant, {**config, "output_dir": str(output)})
        except Exception as exc:
            result = {"task_id": task["task_id"], "category": task["category"], "variant": variant, "success": False,
                      "requires_write": bool(task["expected_checks"]["orders"]),
                      "fault_registered": bool(config.get("fault_overrides", {}).get(task["task_id"], task.get("fault_plan", {}))),
                      "fault_triggered": None, "state_passed": None, "policy_passed": None,
                      "status": "failed", "failure_reason": f"episode_exception:{type(exc).__name__}:{exc}",
                      "failure_category": "budget_environment"}
        results.append(result)
        with (output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        report = {**aggregate_results(results), "n_planned": len(tasks), "run_complete": len(results) == len(tasks)}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"{variant} {task['task_id']}: {result['status']}; objective={result.get('objective_passed', False)}; "
              f"reason={result.get('failure_reason')}; seconds={result.get('elapsed_seconds')}", flush=True)
    return report
