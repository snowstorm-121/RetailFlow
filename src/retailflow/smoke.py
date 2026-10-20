"""Fixed, read-only cases for the T1 tool-calling check."""

import re

from langchain_core.messages import AIMessage, ToolMessage


CASES = [
    {"id": "greeting", "prompt": "Say hello in one sentence. No calculation is needed.", "calls": [], "answer": "hello"},
    {"id": "ready", "prompt": "Reply with the word READY. No calculation is needed.", "calls": [], "answer": "READY"},
    {"id": "add-2-3", "prompt": "Use the addition tool to add 2 and 3. What is the result?", "calls": [("add", {"a": 2, "b": 3}, "5")], "answer": "5"},
    {"id": "add-10-7", "prompt": "Use the addition tool to add 10 and 7. What is the result?", "calls": [("add", {"a": 10, "b": 7}, "17")], "answer": "17"},
    {"id": "add-8-6", "prompt": "Use the addition tool to add 8 and 6. What is the result?", "calls": [("add", {"a": 8, "b": 6}, "14")], "answer": "14"},
    {"id": "add-19-23", "prompt": "Use the addition tool to add 19 and 23. What is the result?", "calls": [("add", {"a": 19, "b": 23}, "42")], "answer": "42"},
    {"id": "route-a1", "prompt": "Use tools to look up the travel time for route A1, then use the conversion tool to give the duration in seconds.", "calls": [("lookup_route_minutes", {"route_id": "A1"}, "7"), ("minutes_to_seconds", {"minutes": 7}, "420")], "answer": "420"},
    {"id": "route-b2", "prompt": "Use tools to look up the travel time for route B2, then use the conversion tool to give the duration in seconds.", "calls": [("lookup_route_minutes", {"route_id": "B2"}, "11"), ("minutes_to_seconds", {"minutes": 11}, "660")], "answer": "660"},
    {"id": "missing-addend", "prompt": "Please use the addition tool to add 4 to another number. I have not given the other number yet. Ask me for it.", "calls": [], "ask": True},
    {"id": "missing-minutes", "prompt": "Please convert my travel time from minutes to seconds. I have not told you how many minutes yet. Ask me for it.", "calls": [], "ask": True},
]


def evaluate_case(case: dict, messages: list) -> dict:
    """Check tool calls, paired results, and the final user-facing response."""
    reasons = []
    requests = [call for message in messages if isinstance(message, AIMessage) for call in message.tool_calls]
    invalid = [call for message in messages if isinstance(message, AIMessage) for call in message.invalid_tool_calls]
    results = [message for message in messages if isinstance(message, ToolMessage)]
    expected = case["calls"]
    if invalid:
        reasons.append("invalid_tool_call")
    if len(requests) != len(expected):
        reasons.append("wrong_tool_count")
    for request, (name, args, result) in zip(requests, expected):
        if request["name"] != name or request["args"] != args:
            reasons.append("wrong_tool_or_arguments")
        matching = [m for m in results if m.tool_call_id == request["id"]]
        if len(matching) != 1 or matching[0].name != name or matching[0].content != result or matching[0].status == "error":
            reasons.append("wrong_tool_result")
    if any(sum(m.tool_call_id == request["id"] for request in requests) != 1 for m in results):
        reasons.append("unpaired_tool_message")
    final = messages[-1] if messages else None
    if not isinstance(final, AIMessage) or final.tool_calls:
        reasons.append("missing_final_answer")
    elif case.get("ask"):
        if "?" not in final.text:
            reasons.append("missing_clarification")
    elif not re.search(r"(?<!\d)" + re.escape(case["answer"]) + r"(?!\d)", final.text, re.I):
        reasons.append("wrong_final_answer")
    return {"passed": not reasons, "reasons": reasons}
