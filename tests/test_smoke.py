from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_two_step_case_requires_both_tools_and_final_answer():
    from retailflow.smoke import CASES, evaluate_case

    case = next(c for c in CASES if c["id"] == "route-a1")
    messages = [
        HumanMessage(content=case["prompt"]),
        AIMessage(content="", tool_calls=[{
            "name": "lookup_route_minutes", "args": {"route_id": "A1"}, "id": "a"
        }]),
        ToolMessage(content="7", name="lookup_route_minutes", tool_call_id="a"),
        AIMessage(content="", tool_calls=[{
            "name": "minutes_to_seconds", "args": {"minutes": 7}, "id": "b"
        }]),
        ToolMessage(content="420", name="minutes_to_seconds", tool_call_id="b"),
        AIMessage(content="420 seconds"),
    ]
    assert evaluate_case(case, messages)["passed"] is True
    assert evaluate_case(case, messages[:-2] + [AIMessage(content="420 seconds")])["passed"] is False


def test_case_rejects_unpaired_tool_message():
    from retailflow.smoke import CASES, evaluate_case

    case = next(c for c in CASES if c["id"] == "add-2-3")
    messages = [
        HumanMessage(content=case["prompt"]),
        ToolMessage(content="5", name="add", tool_call_id="made-up"),
        AIMessage(content="5"),
    ]
    result = evaluate_case(case, messages)
    assert result["passed"] is False
    assert "unpaired_tool_message" in result["reasons"]


def test_missing_value_case_needs_question_without_tool():
    from retailflow.smoke import CASES, evaluate_case

    case = next(c for c in CASES if c["id"] == "missing-addend")
    ok = [HumanMessage(content=case["prompt"]), AIMessage(content="What is the other number?")]
    bad = [HumanMessage(content=case["prompt"]), AIMessage(content="The answer is 4.")]
    assert evaluate_case(case, ok)["passed"] is True
    assert evaluate_case(case, bad)["passed"] is False
