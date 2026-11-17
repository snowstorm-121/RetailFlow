"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
import json
import re
import time
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, messages_to_dict
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, RetailState, State
from react_agent.tools import ASK_USER_FIELDS, RETAIL_TOOLS, TOOLS, native_dialogue_examples, retail_tools_for_session
from retailflow.tools import IDENTITY_TOOLS, WRITE_TOOLS, valid_arguments
from react_agent.utils import load_chat_model

# Define the function that calls the model


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(runtime.context).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast( # type: ignore[redundant-cast]
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages],
            stream=False,
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent")


def _requests_information(text):
    if not isinstance(text, str):
        return False
    request = (r"(?:(?:could|can|would) you (?:please )?(?:provide|share|tell|confirm|give)|"
               r"please (?:provide|share|tell|confirm|give)|(?:i |i'll |i will )(?:need|require)|"
               r"may i (?:have|ask)|(?:until|before) you (?:provide|confirm)|what (?:is|are)|which)")
    field = r"email|(?:full|first|last) name|ZIP|postal|identity|order (?:number|id)|items?|payment|refund method|reason"
    return bool(re.search(request + r"[^.!?]{0,100}(?:" + field + r")", text, re.I))




def _is_text_tool_action(text):
    """Recognize the observed malformed action envelope; never execute its contents."""
    if not isinstance(text, str):
        return False
    candidate = text.strip()
    if candidate.startswith("```json\n") and candidate.endswith("```"):
        candidate = candidate[8:-3].strip()
    try:
        value = json.loads(candidate)
    except ValueError:
        return False
    return (isinstance(value, dict) and isinstance(value.get("action"), str)
            and value["action"] in {tool["function"]["name"] for tool in RETAIL_TOOLS})


def _identity_is_grounded(args, messages):
    supplied = "\n".join(m.content for m in messages if isinstance(m, HumanMessage) and isinstance(m.content, str)).casefold()
    return all(re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", supplied) for value in args.values())


def make_retail_graph(environment, thread_id, model_factory, metrics, max_model_calls, config=None,
                      persist_messages=None, before_model=None, goal_state=None, persist_goal=None):
    """Build the retail loop around the template's model → tool → model pattern."""

    from retailflow.context import ContextBudgetExceeded, build_model_context
    from retailflow.cancel_goals import GOAL_TOOL, GOAL_PROMPT, validate_goal, goal_progress
    from retailflow.cancel_goals import GOAL_REFERENCE_TOOL, GOAL_REFERENCE_PROMPT, goal_reference_input, resolve_goal_references

    config = config or {}
    metrics.setdefault("dialogue_repairs", 0)
    metrics.setdefault("receipt_reviews", 0)

    def persist(state, messages):
        if persist_messages:
            persist_messages(add_messages(state["messages"], messages))

    async def retail_model(state: RetailState, runtime: Runtime[Context]):
        if state["model_calls"] >= max_model_calls:
            return {"status": "budget_exceeded", "failure_reason": "model_call_budget"}
        prompt = runtime.context.system_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
        session = environment.load_session(thread_id)
        planning = goal_state is not None and goal_state.get("needs_refresh", True)
        progress = None
        if goal_state is not None and not planning:
            inventory = None
            for message in state["messages"]:
                if isinstance(message, ToolMessage) and message.name == "get_user_details":
                    value = json.loads(message.content)
                    if value.get("ok") and value["data"]["user_id"] == session.get("user_id"):
                        inventory = value["data"]["orders"]
            with environment.storage.transaction() as db:
                receipts = db.execute("SELECT args_json, result_json FROM receipts WHERE thread = ? AND name = ?",
                                      (thread_id, "cancel_pending_order")).fetchall()
            committed = {json.loads(row["args_json"])["order_id"] for row in receipts if json.loads(row["result_json"]).get("ok")}
            progress = goal_progress(goal_state, session.get("user_id"), session["business"], inventory, committed)
            persist_goal()
            prompt += "\nCancellation goal progress (not write authorization): " + json.dumps(progress)
        examples = native_dialogue_examples()
        example_tokens = len(json.dumps(messages_to_dict(examples))) // 4
        available = (int(runtime.context.num_ctx) - int(runtime.context.num_predict) -
                     len(json.dumps(RETAIL_TOOLS)) // 4 - len(prompt) // 4 - example_tokens - 128)
        budget = min(int(config.get("context_budget_tokens", available)), available)
        metrics["context_budget_tokens"] = budget
        try:
            visible = build_model_context({**session, "messages": state["messages"]},
                                          {"context_budget_tokens": budget, "structured_context": config.get("structured_context", True)})
        except ContextBudgetExceeded:
            return {"status": "budget_exceeded", "failure_reason": "context_budget"}
        latest = state["messages"][-1]
        repair = (isinstance(latest, SystemMessage) and latest.additional_kwargs.get("protocol_feedback"))
        missing_identity = (isinstance(latest, ToolMessage) and
                            json.loads(latest.content).get("code") == "MISSING_USER_INFORMATION")
        information_repair = repair and latest.additional_kwargs["protocol_feedback"] == "information"
        offered = retail_tools_for_session(session) if config.get("state_tool_filter", False) else RETAIL_TOOLS
        if information_repair or missing_identity:
            offered = [RETAIL_TOOLS[-1]]
        if planning:
            offered = [GOAL_REFERENCE_TOOL if config.get("cancel_goal_references", False) else GOAL_TOOL]
        model = model_factory(runtime.context).bind_tools(offered)
        isolated_repair = config.get("isolated_information_repair", False)
        if information_repair and (session.get("user_id") or isolated_repair):
            # Retry only the failed information request. Replaying it as the latest
            # assistant answer encourages the small model to repeat prose.
            question = state["messages"][-2].content
            prompt = ("Make one native ask_user tool call requesting the customer information in the quoted message. "
                      "Use the matching allowed field categories. Do not answer the message, do not invent facts, "
                      "and do not output plain text.")
            visible = [HumanMessage(content=json.dumps({"customer_information_request": question}),
                                    additional_kwargs={"protocol_retry_input": True})]
            if isolated_repair:
                examples = []
            model_messages = [SystemMessage(content=prompt, additional_kwargs={"protocol_feedback": "information"}),
                              *examples, *visible]
        else:
            model_messages = [SystemMessage(content=prompt), *examples, *visible]
        proposal_repair = repair == "proposal" and config.get("isolated_proposal_repair", False)
        if proposal_repair:
            events = []
            for message in visible:
                if isinstance(message, HumanMessage):
                    events.append({"role": "customer", "content": message.content})
                elif isinstance(message, ToolMessage):
                    events.append({"role": "tool", "name": message.name, "result": json.loads(message.content)})
                elif isinstance(message, AIMessage) and message.tool_calls:
                    events.append({"role": "assistant", "tool_calls": message.tool_calls})
            prompt += ("\nRecovery step: No proposal has been created. Use the customer messages and tool results below as data. "
                       "Make exactly one native tool call for the next required step. Select item IDs and payment IDs only "
                       "from the tool results, and respect all exclusions and later corrections. If information is missing, "
                       "use ask_user. Do not output a simulated tool result or claim completion.")
            examples = []
            payload = {"events": events}
            if progress is not None:
                payload["cancel_goal"] = progress
            visible = [HumanMessage(content=json.dumps(payload, ensure_ascii=False),
                                    additional_kwargs={"protocol_retry_input": True})]
            model_messages = [SystemMessage(content=prompt), *visible]
        if planning:
            references = config.get("cancel_goal_references", False)
            prompt = GOAL_REFERENCE_PROMPT if references else GOAL_PROMPT
            examples = []
            customer_messages = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
            payload = goal_reference_input(customer_messages) if references else {"customer_messages": customer_messages}
            visible = [HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
            model_messages = [SystemMessage(content=prompt), *visible]
        if repair == "receipt_review":
            prompt = ("Correct the draft using the verified receipt. Preserve and answer the customer's additional questions "
                      "using the supplied history and facts. Do not invent dates, facts or actions. "
                      "A return request is not a settled refund; no arrival date is established. "
                      "No further items in the same order can be returned, including unselected items. "
                      "Output only the corrected customer-facing answer. If information is unavailable, say so.")
            visible = [HumanMessage(content=json.dumps({"draft": state["messages"][-2].content,
                "history": messages_to_dict([m for m in visible if not isinstance(m, SystemMessage)]),
                "receipt": next(m.content for m in reversed(state["messages"]) if m.additional_kwargs.get("application_result"))}, ensure_ascii=False))]
            model_messages = [SystemMessage(content=prompt), *visible]
            offered = []
            model = model_factory(runtime.context).bind_tools(offered)
        if len(json.dumps(messages_to_dict(visible), ensure_ascii=False)) // 4 + 1 > budget:
            return {"status": "budget_exceeded", "failure_reason": "context_budget"}
        if before_model:
            reason = before_model()
            if reason:
                return {"status": "budget_exceeded", "failure_reason": reason}
        metrics["context_trimmed_messages"] = len(state["messages"]) - len(visible) + int(config.get("structured_context", True))
        metrics["context_estimated_tokens"] = (len(prompt) + len(json.dumps(RETAIL_TOOLS)) +
                                                len(json.dumps(messages_to_dict([*examples, *visible]), ensure_ascii=False))) // 4
        kept_ids = {message.id for message in visible if message.id}
        input_record = {"budget_tokens": budget, "estimate_method": "serialized_characters_divided_by_4",
                        "estimated_total_input_tokens": metrics["context_estimated_tokens"],
                        "actual_input_tokens": None,
                        "messages": messages_to_dict(model_messages),
                        "tool_names": [tool["function"]["name"] for tool in offered],
                        "dropped_messages": [{"index": index, "type": message.type, "id": message.id}
                                             for index, message in enumerate(state["messages"]) if message.id not in kept_ids]}
        metrics["context_inputs"].append(input_record)
        start = time.perf_counter()
        metrics["model_calls"] += 1
        try:
            response = await model.ainvoke(model_messages, stream=False)
        finally:
            metrics["model_seconds"] += time.perf_counter() - start
        persist(state, [response])
        usage = response.usage_metadata
        if usage:
            input_record["actual_input_tokens"] = usage.get("input_tokens")
            metrics["input_tokens"] = (metrics["input_tokens"] or 0) + usage.get("input_tokens", 0)
            metrics["output_tokens"] = (metrics["output_tokens"] or 0) + usage.get("output_tokens", 0)
            if usage.get("input_tokens", 0) > int(runtime.context.num_ctx) - int(runtime.context.num_predict):
                return {"messages": [response], "model_calls": state["model_calls"] + 1,
                        "status": "budget_exceeded", "failure_reason": "actual_context_limit"}
        if response.response_metadata.get("done_reason") == "length":
            return {"messages": [response], "model_calls": state["model_calls"] + 1,
                    "status": "failed", "failure_reason": "output_truncated"}
        if response.invalid_tool_calls:
            return {"messages": [response], "model_calls": state["model_calls"] + 1,
                    "status": "failed", "failure_reason": "invalid_tool_call"}
        if planning and (len(response.tool_calls) != 1 or response.tool_calls[0]["name"] != "update_cancel_goal"):
            return {"messages": [response], "model_calls": state["model_calls"] + 1,
                    "status": "failed", "failure_reason": "cancel_goal_not_registered"}
        if repair == "receipt_review":
            if response.tool_calls or not isinstance(response.content, str) or not response.content.strip() or _is_text_tool_action(response.content):
                return {"messages": [response], "model_calls": state["model_calls"] + 1,
                        "status": "failed", "failure_reason": "receipt_review_invalid_response"}
            return {"messages": [response], "model_calls": state["model_calls"] + 1, "status": "answered"}
        if not response.tool_calls:
            if (progress is not None and not progress["complete"]
                    and not _requests_information(response.content)):
                if metrics["dialogue_repairs"] >= 1:
                    return {"messages": [response], "model_calls": state["model_calls"] + 1,
                            "status": "failed", "failure_reason": "cancel_goal_incomplete"}
                metrics["dialogue_repairs"] += 1
                feedback = SystemMessage(content="Cancellation goal remains incomplete. Use a native tool for the next step; each proposal needs confirmation.",
                                         additional_kwargs={"protocol_feedback": "proposal"})
                persist(state, [response, feedback])
                return {"messages": [response, feedback], "model_calls": state["model_calls"] + 1, "status": "repairing"}
            if _is_text_tool_action(response.content):
                return {"messages": [response], "model_calls": state["model_calls"] + 1,
                        "status": "failed", "failure_reason": "non_native_tool_action"}
            if proposal_repair:
                return {"messages": [response], "model_calls": state["model_calls"] + 1,
                        "status": "failed", "failure_reason": "proposal_not_created"}
            if information_repair:
                return {"messages": [response], "model_calls": state["model_calls"] + 1,
                        "status": "failed", "failure_reason": "information_request_not_structured"}
            text = response.content
            proposal_missing = (isinstance(text, str) and session.get("user_id")
                                and session.get("target_order_status") in {None, "pending", "delivered"}
                                and re.search(r"please confirm|(?:could|can|would) you confirm|do you (?:confirm|want)|would you like me to (?:proceed|cancel|return)|\b(?:proposal|return request|cancellation request)(?: (?:has been |was |is now )?| for [^.!?\n]+? (?:has been |was |is now ))(?:successfully )?(?:created|submitted|staged)\b", text, re.I)
                                and re.search(r"cancel|return", text, re.I))
            explicit_confirmation = (isinstance(text, str) and re.search(
                r"please confirm (?:if|whether) you (?:would like|want|wish)|would you like me to (?:proceed|cancel|return)|do you want me to",
                text, re.I))
            kind = ("empty" if not text else "proposal" if proposal_missing and explicit_confirmation else
                    "information" if _requests_information(text) else "proposal" if proposal_missing else None)
            if kind:
                reason = {"empty": "empty_model_response", "proposal": "proposal_not_created",
                          "information": "information_request_not_structured"}[kind]
                if metrics["dialogue_repairs"] >= 1:
                    return {"messages": [response], "model_calls": state["model_calls"] + 1,
                            "status": "failed", "failure_reason": reason}
                metrics["dialogue_repairs"] += 1
                feedback_text = {
                    "information": "Your previous response requests information from the customer, so the task is not complete. "
                                   "Request that information by making one native ask_user call with the allowed field categories. "
                                   "Do not repeat a plain-text question and do not invent customer details.",
                    "proposal": "No operation proposal exists. Do not ask for confirmation in text. "
                                "Use the appropriate native business tool with verified parameters to stage the requested operation first. "
                                "The application obtains confirmation. Do not claim completion or substitute a different action.",
                    "empty": "The previous model response was empty. Continue with a native tool call if needed, "
                             "or give an accurate nonempty final answer. Do not invent customer details.",
                }[kind]
                feedback = SystemMessage(content=feedback_text, additional_kwargs={"protocol_feedback": kind})
                persist(state, [response, feedback])
                return {"messages": [response, feedback], "model_calls": state["model_calls"] + 1, "status": "repairing"}
        if (not response.tool_calls and config.get("receipt_answer_review", False)
                and not metrics["receipt_reviews"] and session.get("target_order_status") == "return requested"
                and any(m.additional_kwargs.get("application_result") for m in state["messages"])):
            metrics["receipt_reviews"] += 1
            feedback = SystemMessage(content="Review the draft against the committed return receipt.",
                                     additional_kwargs={"protocol_feedback": "receipt_review"})
            persist(state, [response, feedback])
            return {"messages": [response, feedback], "model_calls": state["model_calls"] + 1, "status": "repairing"}
        return {"messages": [response], "model_calls": state["model_calls"] + 1,
                "status": "running" if response.tool_calls else "answered"}

    def retail_tools(state: RetailState):
        calls = state["messages"][-1].tool_calls
        if len(calls) != 1:
            metrics["protocol_errors"] += 1
            messages = [ToolMessage(content=json.dumps({"code": "PROTOCOL_ERROR", "message": "Call exactly one tool at a time."}),
                                    name=call["name"], tool_call_id=call["id"], status="error") for call in calls]
            persist(state, messages)
            return {"messages": messages}
        call = calls[0]
        name, args = call["name"], call["args"]
        start = time.perf_counter()
        retries = 0
        if name not in metrics["context_inputs"][-1]["tool_names"]:
            result = {"ok": False, "code": "INVALID_ARGUMENT", "message": "This tool was not offered in the current model request."}
            status, request = "running", None
        elif name == "update_cancel_goal":
            customer_messages = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
            plan = (resolve_goal_references(args, customer_messages) if config.get("cancel_goal_references", False)
                    else args if validate_goal(args, customer_messages) else None)
            if plan is None:
                result = {"ok": False, "code": "INVALID_ARGUMENT", "message": "Invalid cancellation goal; no business action allowed."}
                message = ToolMessage(content=json.dumps(result), name=name, tool_call_id=call["id"], status="error")
                metrics["tool_calls"] += 1
                metrics["tool_seconds"] += time.perf_counter() - start
                persist(state, [message])
                return {"messages": [message], "status": "failed", "failure_reason": "invalid_cancel_goal"}
            targets = (goal_state.get("targets") if goal_state.get("scope") == plan["scope"]
                       and goal_state.get("order_ids") == plan["order_ids"] else None)
            goal_state.clear()
            goal_state.update(plan, targets=targets, needs_refresh=False)
            persist_goal()
            result = {"ok": True, "code": "GOAL_RECORDED", "data": plan}
            status, request = "running", None
        elif name == "ask_user":
            if (set(args) == {"fields", "question"} and isinstance(args["question"], str) and args["question"]
                    and isinstance(args["fields"], list) and args["fields"]
                    and all(isinstance(field, str) and field in ASK_USER_FIELDS for field in args["fields"])
                    and len(set(args["fields"])) == len(args["fields"])):
                result = {"ok": True, "code": "NEEDS_INPUT", "data": args}
                status, request = "needs_input", args
            else:
                result = {"ok": False, "code": "INVALID_ARGUMENT", "message": "ask_user needs fields and question."}
                status, request = "running", None
        else:
            if (name == "cancel_pending_order" and goal_state is not None
                    and (goal_state.get("needs_refresh") or goal_state.get("scope") == "none"
                         or args.get("order_id") not in (goal_state.get("targets") or []))):
                result = {"ok": False, "code": "POLICY_DENIED", "message": "Cancellation is outside the recorded goal. No proposal was created."}
            elif name in IDENTITY_TOOLS and valid_arguments(name, args) and not _identity_is_grounded(args, state["messages"]):
                result = {"ok": False, "code": "MISSING_USER_INFORMATION",
                          "message": "These identity values were not supplied by the customer. Call ask_user for identity; do not guess.",
                          "fields": ["identity"]}
            else:
                result = environment.dispatch_tool(thread_id, name, args)
            metrics["tool_attempts"].append({"tool_call_id": call["id"], "name": name, "args": args,
                                             "code": result["code"], "message": result.get("message"), "attempt": 1})
            while (result["code"] == "TEMPORARY_FAILURE" and name not in WRITE_TOOLS
                   and retries < min(1, max(0, int(config.get("max_retry_per_call", 1))))):
                retries += 1
                result = environment.dispatch_tool(thread_id, name, args)
                metrics["tool_attempts"].append({"tool_call_id": call["id"], "name": name, "args": args,
                                                 "code": result["code"], "message": result.get("message"), "attempt": retries + 1})
            metrics["retries"] = metrics.get("retries", 0) + retries
            status = {"CONFIRMATION_REQUIRED": "needs_confirmation", "POLICY_DENIED": "refused",
                      "OK": "handed_off" if name == "transfer_to_human_agents" else "running"}.get(result["code"], "running")
            request = result.get("proposal") if status == "needs_confirmation" else None
        if config.get("profile_order_summaries", False) and name == "get_user_details" and result["ok"]:
            session = environment.load_session(thread_id)
            result["data"]["order_summaries"] = [
                {"order_id": order_id, "status": order["status"],
                 "items": [{key: item[key] for key in ("name", "item_id", "product_id", "options")}
                           for item in order["items"]]}
                for order_id in result["data"]["orders"]
                if (order := session["business"]["orders"][order_id])["user_id"] == session["user_id"]]
        metrics["tool_seconds"] += time.perf_counter() - start
        metrics["tool_calls"] += 1 + retries
        message = ToolMessage(content=json.dumps(result, ensure_ascii=False), name=name, tool_call_id=call["id"],
                              status="success" if result["ok"] else "error")
        persist(state, [message])
        return {"messages": [message], "status": status, "request": request}

    def after_model(state: RetailState):
        if state["status"] == "repairing":
            return "model"
        return "tools" if state["status"] == "running" else "__end__"

    def after_tools(state: RetailState):
        return "model" if state["status"] == "running" else "__end__"

    retail = StateGraph(RetailState, context_schema=Context)
    retail.add_node("model", retail_model)
    retail.add_node("tools", retail_tools)
    retail.add_edge("__start__", "model")
    retail.add_conditional_edges("model", after_model)
    retail.add_conditional_edges("tools", after_tools)
    return retail.compile(name="RetailFlow Agent")
