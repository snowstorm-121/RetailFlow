"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
import json
import time
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, messages_to_dict
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, RetailState, State
from react_agent.tools import RETAIL_TOOLS, TOOLS
from retailflow.tools import WRITE_TOOLS
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


def make_retail_graph(environment, thread_id, model_factory, metrics, max_model_calls, config=None,
                      persist_messages=None, before_model=None):
    """Build the retail loop around the template's model → tool → model pattern."""

    from retailflow.context import ContextBudgetExceeded, build_model_context

    config = config or {}

    def persist(state, messages):
        if persist_messages:
            persist_messages(add_messages(state["messages"], messages))

    async def retail_model(state: RetailState, runtime: Runtime[Context]):
        if state["model_calls"] >= max_model_calls:
            return {"status": "budget_exceeded", "failure_reason": "model_call_budget"}
        prompt = runtime.context.system_prompt.format(system_time=datetime.now(tz=UTC).isoformat())
        session = environment.load_session(thread_id)
        available = (int(runtime.context.num_ctx) - int(runtime.context.num_predict) -
                     len(json.dumps(RETAIL_TOOLS)) // 4 - len(prompt) // 4 - 128)
        budget = min(int(config.get("context_budget_tokens", available)), available)
        metrics["context_budget_tokens"] = budget
        try:
            visible = build_model_context({**session, "messages": state["messages"]}, {"context_budget_tokens": budget})
        except ContextBudgetExceeded:
            return {"status": "budget_exceeded", "failure_reason": "context_budget"}
        if before_model:
            reason = before_model()
            if reason:
                return {"status": "budget_exceeded", "failure_reason": reason}
        model = model_factory(runtime.context).bind_tools(RETAIL_TOOLS)
        metrics["context_trimmed_messages"] = len(state["messages"]) - len(visible) + 1
        metrics["context_estimated_tokens"] = (len(prompt) + len(json.dumps(RETAIL_TOOLS)) +
                                                len(json.dumps(messages_to_dict(visible), ensure_ascii=False))) // 4
        kept_ids = {message.id for message in visible if message.id}
        input_record = {"budget_tokens": budget, "estimate_method": "serialized_characters_divided_by_4",
                        "estimated_total_input_tokens": metrics["context_estimated_tokens"],
                        "actual_input_tokens": None,
                        "messages": messages_to_dict([SystemMessage(content=prompt), *visible]),
                        "dropped_messages": [{"index": index, "type": message.type, "id": message.id}
                                             for index, message in enumerate(state["messages"]) if message.id not in kept_ids]}
        metrics["context_inputs"].append(input_record)
        start = time.perf_counter()
        metrics["model_calls"] += 1
        try:
            response = await model.ainvoke([SystemMessage(content=prompt), *visible], stream=False)
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
        return {"messages": [response], "model_calls": state["model_calls"] + 1,
                "status": "running" if response.tool_calls else "answered"}

    def retail_tools(state: RetailState):
        calls = state["messages"][-1].tool_calls
        if len(calls) != 1 or state["messages"][-1].content:
            metrics["protocol_errors"] += 1
            messages = [ToolMessage(content=json.dumps({"code": "PROTOCOL_ERROR", "message": "Call one tool at a time without a user-facing reply."}),
                                    name=call["name"], tool_call_id=call["id"], status="error") for call in calls]
            persist(state, messages)
            return {"messages": messages}
        call = calls[0]
        name, args = call["name"], call["args"]
        start = time.perf_counter()
        retries = 0
        if name == "ask_user":
            if (set(args) == {"fields", "question"} and isinstance(args["question"], str) and args["question"]
                    and isinstance(args["fields"], list) and args["fields"]
                    and all(isinstance(field, str) and field for field in args["fields"])
                    and len(set(args["fields"])) == len(args["fields"])):
                result = {"ok": True, "code": "NEEDS_INPUT", "data": args}
                status, request = "needs_input", args
            else:
                result = {"ok": False, "code": "INVALID_ARGUMENT", "message": "ask_user needs fields and question."}
                status, request = "running", None
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
        metrics["tool_seconds"] += time.perf_counter() - start
        metrics["tool_calls"] += 1 + retries
        message = ToolMessage(content=json.dumps(result, ensure_ascii=False), name=name, tool_call_id=call["id"],
                              status="success" if result["ok"] else "error")
        persist(state, [message])
        return {"messages": [message], "status": status, "request": request}

    def after_model(state: RetailState):
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
