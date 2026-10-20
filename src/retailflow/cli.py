"""CLI entry point for T1 local model checks."""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import messages_to_dict

from react_agent.context import Context
from react_agent.graph import graph
from retailflow.dataset import prepare_dataset, validate_dataset
from retailflow.runtime import RetailRuntime, run_turn
from retailflow.smoke import CASES, evaluate_case


SYSTEM_PROMPT = (
    "You are checking a local tool-calling agent. Use one tool call at a time. "
    "For route duration, first look up minutes, then call minutes_to_seconds with that result. "
    "If a required number is missing, ask the user for it instead of inventing it. "
    "For addition requests, call add. Give a short final answer. Current time: {system_time}."
)


async def smoke_model(config_path: Path) -> int:
    """Run ten fixed cases through the upstream LangGraph loop."""
    config = json.loads(config_path.read_text())
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    output = Path("artifacts/t1")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    passed = 0
    results = []
    for case in CASES:
        start = time.perf_counter()
        context = Context(**config, system_prompt=SYSTEM_PROMPT, trace_path=str(output / f"{stamp}-{case['id']}-http.jsonl"))
        try:
            state = await graph.ainvoke(
                {"messages": [{"role": "user", "content": case["prompt"]}]},
                context=context,
                config={"recursion_limit": 14},
            )
            messages = state["messages"]
            grade = evaluate_case(case, messages)
            entry = {"id": case["id"], **grade, "elapsed_seconds": round(time.perf_counter() - start, 3), "messages": messages_to_dict(messages)}
        except Exception as exc:
            entry = {"id": case["id"], "passed": False, "reasons": ["runtime_failure"], "failure": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(time.perf_counter() - start, 3)}
        passed += entry["passed"]
        results.append(entry)
        print(f"{case['id']}: {'PASS' if entry['passed'] else 'FAIL'} {entry['reasons']}", flush=True)
    trace_path = output / f"smoke-{stamp}.json"
    trace_path.write_text(json.dumps({"config": config, "cases": results}, ensure_ascii=False, indent=2))
    two_step_ok = all(r["passed"] for r in results if r["id"] in {"route-a1", "route-b2"})
    print(f"Result: {passed}/10; two-step both passed: {two_step_ok}; trace: {trace_path}")
    return 0 if passed >= 8 and two_step_ok else 1


async def chat(thread_id: str, config_path: Path, message: str | None = None) -> int:
    """Run one supplied message or an interactive session through the retail runtime."""
    config = json.loads(config_path.read_text())
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    next_text = message
    resume_event = None
    if message is None:
        session = RetailRuntime(Path(config.get("store_path", "runtime/business.sqlite")),
                                Path(config.get("trace_dir", "artifacts/t4"))).load_session(thread_id)
        pending = session.get("pending_action")
        if pending:
            print(f"Pending {pending['name']} with {json.dumps(pending['args'], ensure_ascii=False)}")
            try:
                choice = input("Confirm this exact pending action? Type yes or no: ").strip().lower()
            except EOFError:
                return 0
            if choice not in {"yes", "no"}:
                print("No confirmation submitted. The proposal remains pending.")
                return 0
            resume_event = {"kind": "confirmation", "proposal_id": pending["proposal_id"], "accepted": choice == "yes"}
    while True:
        if next_text is None and resume_event is None:
            try:
                next_text = input("You (or /quit): ").strip()
            except EOFError:
                return 0
            if next_text == "/quit":
                return 0
            if not next_text:
                next_text = None
                continue
        event = resume_event or {"kind": "message", "text": next_text}
        resume_event = None
        result = await run_turn(thread_id, event, config)
        next_text = None
        while True:
            print(f"[{result['status']}] {result['text']}")
            print(f"trace: {result['trace_path']}")
            if result["status"] == "needs_confirmation":
                try:
                    choice = input("Confirm this exact action? Type yes or no: ").strip().lower()
                except EOFError:
                    return 0
                if choice not in {"yes", "no"}:
                    print("No confirmation submitted. The proposal remains pending.")
                    return 0
                event = {"kind": "confirmation", "proposal_id": result["request"]["proposal_id"], "accepted": choice == "yes"}
                result = await run_turn(thread_id, event, config)
                continue
            if result["status"] == "failed":
                return 1
            break
        if message is not None and result["status"] != "needs_input":
            return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke-model")
    smoke.add_argument("--config", type=Path, default=Path("configs/local.json"))
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=Path("configs/evaluation.json"))
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, default=Path("data/manifests/dataset.json"))
    retail_chat = commands.add_parser("chat")
    retail_chat.add_argument("--thread", required=True)
    retail_chat.add_argument("--config", type=Path, default=Path("configs/local.json"))
    retail_chat.add_argument("--message")
    args = parser.parse_args()
    if args.command == "smoke-model":
        return asyncio.run(smoke_model(args.config))
    if args.command == "prepare":
        print(json.dumps(prepare_dataset(json.loads(args.config.read_text())), ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate":
        result = validate_dataset(str(args.manifest))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["valid"] else 1
    if args.command == "chat":
        return asyncio.run(chat(args.thread, args.config, args.message))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
