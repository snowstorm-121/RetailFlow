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
from retailflow.tools import ADDRESS_TOOLS, IDENTITY_TOOLS, WRITE_TOOLS, valid_arguments
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
    if not isinstance(value, dict):
        return False
    name = value.get("action")
    if name is None and isinstance(value.get("arguments"), dict):
        name = value.get("name")
    return isinstance(name, str) and any(tool["function"]["name"] == name for tool in RETAIL_TOOLS)


def _identity_is_grounded(args, messages):
    supplied = "\n".join(m.content for m in messages if isinstance(m, HumanMessage) and isinstance(m.content, str)).casefold()
    return all(re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", supplied) for value in args.values())


def make_retail_graph(environment, thread_id, model_factory, metrics, max_model_calls, config=None,
                      persist_messages=None, before_model=None, goal_state=None, persist_goal=None):
    """Build the retail loop around the template's model → tool → model pattern."""

    from retailflow.context import ContextBudgetExceeded, build_model_context
    from retailflow.preproposal import (address_copy_target, address_order_candidates, cross_product_cancel_target,
                                        false_cancellation_claim,
                                        item_change_targets, item_order_excluded, preserved_variant_conflicts,
                                        requested_variants, wrong_item_target)
    from retailflow.address_followups import (advisory_opening, explicit_address_steps, matching_variants,
                                              processing_guarantee_fact, verified_address_reply)
    from retailflow.cancel_goals import GOAL_TOOL, GOAL_PROMPT, validate_goal, goal_progress
    from retailflow.cancel_goals import GOAL_REFERENCE_TOOL, GOAL_REFERENCE_PROMPT, goal_reference_input, resolve_goal_references

    config = config or {}
    metrics.setdefault("dialogue_repairs", 0)
    metrics.setdefault("receipt_reviews", 0)
    proposal_repair_active = False
    continuation_checked = False
    account_copy_retry_draft_id = None
    focused_step = None
    reusable_reads = IDENTITY_TOOLS | {"get_user_details", "get_order_details"}
    successful_reads = {}
    metrics.setdefault("reused_reads", 0)

    def persist(state, messages):
        if persist_messages:
            persist_messages(add_messages(state["messages"], messages))

    def active_request(messages):
        start = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].additional_kwargs.get("request_start")), 0)
        return messages[start:]

    def request_text(messages):
        return advisory_opening([m.content for m in messages
                                 if isinstance(m, HumanMessage) and isinstance(m.content, str)])

    def read_scope(session):
        last_receipt = session.get("last_receipt") or {}
        return session["user_id"], last_receipt.get("result", {}).get("proposal_id")

    def address_copy_evidence(messages, opening, session):
        if not opening or not session.get("user_id"):
            return None
        for message in messages:
            if message.additional_kwargs.get("address_copy_user_id") == session["user_id"]:
                return message.additional_kwargs["address_copy"]
            if isinstance(message, ToolMessage) and message.name == "get_user_details":
                data = json.loads(message.content).get("data")
                if isinstance(data, dict) and data.get("user_id") == session["user_id"] and data.get("address_copy"):
                    return data["address_copy"]
        return address_copy_target(opening, session["business"], session["user_id"])

    async def retail_model(state: RetailState, runtime: Runtime[Context]):
        nonlocal proposal_repair_active, continuation_checked, account_copy_retry_draft_id, focused_step
        focused_step = None
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
        active_messages = active_request(state["messages"])
        address_request = any(
            isinstance(message, HumanMessage) and isinstance(message.content, str)
            and re.search(r"\b(?:address|suite|ship(?:ping)?)\b", message.content, re.I)
            and re.search(r"\b(?:cancel|return|exchange|all|every)\b|\balso\s+(?:modify|change|update|switch)\b", message.content, re.I)
            for message in active_messages
        )
        outcomes = [message for message in active_messages
                    if message.additional_kwargs.get("application_result")]
        opening = request_text(active_messages)
        cancel_fallback = None if goal_state is not None else cross_product_cancel_target(
            opening, session["business"], session["user_id"])
        if cancel_fallback:
            reads = [json.loads(message.content)["data"] for message in active_messages
                     if isinstance(message, ToolMessage) and message.name == "get_order_details"
                     and message.status == "success"]
            if any(order["order_id"] == cancel_fallback and order["user_id"] == session["user_id"]
                   and order["status"] == "pending" for order in reads):
                marker = {"cancel_fallback_order": cancel_fallback, "cancel_fallback_user": session["user_id"]}
                reason = None
                if any(all(message.additional_kwargs.get(key) == value for key, value in marker.items())
                       for message in active_messages):
                    latest_user = next(message.content for message in reversed(active_messages)
                                       if isinstance(message, HumanMessage))
                    if isinstance(latest_user, str):
                        value = latest_user.strip().lower().rstrip(".")
                        reason = {
                            "no longer needed": "no longer needed", "i no longer need it": "no longer needed",
                            "i no longer need the order": "no longer needed", "ordered by mistake": "ordered by mistake",
                            "i ordered it by mistake": "ordered by mistake",
                            "i ordered the order by mistake": "ordered by mistake",
                        }.get(value)
                if not reason:
                    question = (f"Order {cancel_fallback} is pending. I can only change variants of the same product, "
                                "and only delivered orders can be returned. For the cancellation you requested as a "
                                "fallback, what is the reason: ordered by mistake or no longer needed?")
                    clarification = AIMessage(content=question, additional_kwargs=marker)
                    return {"messages": [clarification], "status": "needs_input",
                            "request": {"fields": ["cancel_reason"], "question": question}}
                args = {"order_id": cancel_fallback, "reason": reason}
                start = time.perf_counter()
                staged = environment.stage_cancellation(thread_id, args)
                metrics["tool_seconds"] += time.perf_counter() - start
                metrics["tool_calls"] += 1
                metrics["tool_attempts"].append({"name": "cancel_pending_order", "args": args,
                                                 "code": staged["code"], "message": staged.get("message"),
                                                 "attempt": 1, "origin": "application"})
                if staged["code"] == "CONFIRMATION_REQUIRED":
                    proposal = staged["proposal"]
                    proposal_notice = SystemMessage(
                        content=f"Application staged cancellation proposal {proposal['proposal_id']}. Separate confirmation is required.",
                        additional_kwargs={"application_proposal": True},
                    )
                    return {"messages": [proposal_notice], "status": "needs_confirmation", "request": proposal}
                return {"status": "failed", "failure_reason": staged["code"]}
            prompt += (f"\nThe customer explicitly requested an item change, then a return, then cancellation. "
                       f"Read the unique matching order {cancel_fallback} using get_order_details to verify eligibility.")
        copy = address_copy_evidence(active_messages, opening, session)
        address_request = address_request or copy is not None
        remaining_copy_step = None
        if copy:
            destination = session["business"]["orders"][copy["destination_order_id"]]
            account = session["business"]["users"][session["user_id"]]
            prompt += ("\nVerified address copy roles: " + json.dumps(copy, ensure_ascii=False)
                       + ". The source order is read-only for the copy. For the shipping proposal, use destination_order_id "
                         "as order_id and copy every address field including country. An explicitly requested account default "
                         "change requires a separate proposal with user_id. Continue other requested items after confirmation."
                       + f" Shipping address already matches: {destination['address'] == copy['address']}. "
                         f"Account address already matches: {account['address'] == copy['address']}.")
            retry_focus = state["model_calls"] == 1 and (
                state["messages"][-1].additional_kwargs.get("focused_write_mismatch")
                or state["messages"][-1].additional_kwargs.get("protocol_feedback") == "focused_prose")
            if config.get("confirmed_this_turn") and outcomes:
                if (destination["address"] != copy["address"] and destination["status"] in
                        {"pending", "pending (item modified)"}):
                    remaining_copy_step = {"tool": "modify_pending_order_address",
                                           "order_id": copy["destination_order_id"], **copy["address"]}
                elif copy["update_default"] and account["address"] != copy["address"]:
                    remaining_copy_step = {"tool": "modify_user_address", "user_id": session["user_id"],
                                           **copy["address"]}
                if remaining_copy_step and (state["model_calls"] == 0 or retry_focus
                                            or state["messages"][-1].additional_kwargs.get("protocol_feedback") == "focused_prose"):
                    focused_step = remaining_copy_step
                    prompt += ("\nThe confirmed action is complete. Propose this remaining address change separately: "
                               + json.dumps(focused_step, ensure_ascii=False))
        steps = explicit_address_steps(opening) if address_request else []
        if steps:
            committed = set()
            for message in outcomes:
                receipt = message.content.removeprefix("Application outcome: ")
                if message.additional_kwargs.get("committed_tool") and receipt != message.content:
                    order_id = json.loads(receipt).get("order_id")
                    committed.add((message.additional_kwargs["committed_tool"], order_id))
            checklist = [{**step, "committed": (step["tool"], step["order_id"]) in committed} for step in steps]
            next_step = next((step for step in checklist if not step["committed"]), None)
            if (next_step and session.get("user_id") and any(step["committed"] for step in checklist)
                    and len(checklist) > 1 and goal_state is None):
                target = session["business"]["orders"].get(next_step["order_id"], {})
                eligible_statuses = {"pending", "pending (item modified)"} if next_step["tool"] == "modify_pending_order_address" else {"pending"}
                if target.get("user_id") == session["user_id"] and target.get("status") in eligible_statuses:
                    retry_focus = (state["model_calls"] == 1 and
                                   (state["messages"][-1].additional_kwargs.get("focused_write_mismatch")
                                    or state["messages"][-1].additional_kwargs.get("protocol_feedback") == "focused_prose"))
                    if focused_step is None and config.get("confirmed_this_turn") and (state["model_calls"] == 0 or retry_focus):
                        focused_step = next_step
                    prompt += ("\nNext explicit step: " + json.dumps(next_step, ensure_ascii=False)
                               + ". Check business prerequisites and use a read tool or ask_user if needed. "
                                 "If eligible, make a native proposal for separate user confirmation.")
            prompt += ("\nExplicit request checklist (planning only; later user changes override it): "
                       + json.dumps(checklist, ensure_ascii=False)
                       + ". Continue any pending step through verified reads and a native proposal. "
                         "For pending item changes, read the old item in its order, then look up that item's product_id. "
                         "Use only available catalog variant IDs. Any catalog option hits are partial evidence: verify "
                         "all requested preferences before selecting. Never invent an ID or repeat committed steps.")
        item_targets = item_change_targets(opening, session["business"], session["user_id"])
        if item_targets:
            prompt += ("\nUse item_change_targets from authenticated reads to distinguish the original item_ids "
                       "from the new_item_ids selected from that product's available variants. "
                       "Read the selected order and product before proposing.")
        if re.search(r"\baddress\b", opening, re.I):
            prompt += ("\nIf the authenticated profile shows address_order_candidates, read the matching order before "
                       "drawing conclusions. Verify the city and requested item in that order. Address candidates are "
                       "search hints, not evidence that a shipping or account address should be changed.")
        deadline_fact = processing_guarantee_fact(opening)
        if deadline_fact:
            prompt += "\n" + deadline_fact
        has_cancel_receipt = any(message.additional_kwargs.get("committed_tool") == "cancel_pending_order"
                                 for message in outcomes)
        cancel_requested = address_request and any(
            isinstance(message, HumanMessage) and isinstance(message.content, str)
            and re.search(r"\bcancel\b", message.content, re.I)
            for message in active_messages
        )
        if address_request and outcomes:
            prompt += ("\nContinue the customer's complete request after confirmation. A confirmed write completes only "
                       "that specific target and action. Check the recorded outcomes; don't call a completed operation again. "
                       "Use a native call for the next unresolved operation, including any read needed to check a condition.")
            if cancel_requested and not has_cancel_receipt:
                prompt += (" There is no confirmed cancellation in this request. Check the customer's cancellation condition "
                           "and target separately from any address update. If eligible, call cancel_pending_order to propose; "
                           "otherwise explain why it remains unresolved. Never claim it was cancelled.")
        context_messages = state["messages"]
        focused_prose = focused_step and context_messages[-1].additional_kwargs.get("protocol_feedback") == "focused_prose"
        if focused_prose:
            context_messages = [*context_messages[:-2], context_messages[-1]]
        examples = [] if focused_prose else native_dialogue_examples()
        example_tokens = len(json.dumps(messages_to_dict(examples))) // 4
        available = (int(runtime.context.num_ctx) - int(runtime.context.num_predict) -
                     len(json.dumps(RETAIL_TOOLS)) // 4 - len(prompt) // 4 - example_tokens - 128)
        budget = min(int(config.get("context_budget_tokens", available)), available)
        metrics["context_budget_tokens"] = budget
        try:
            visible = build_model_context({**session, "messages": context_messages},
                                          {"context_budget_tokens": budget, "structured_context": config.get("structured_context", True),
                                           "required_outcome_ids": {message.id for message in outcomes} if address_request else set()})
        except ContextBudgetExceeded:
            return {"status": "budget_exceeded", "failure_reason": "context_budget"}
        latest = state["messages"][-1]
        repair = (isinstance(latest, SystemMessage) and latest.additional_kwargs.get("protocol_feedback"))
        missing_identity = (isinstance(latest, ToolMessage) and
                            json.loads(latest.content).get("code") == "MISSING_USER_INFORMATION")
        information_repair = repair and latest.additional_kwargs["protocol_feedback"] == "information"
        offered = retail_tools_for_session(session) if config.get("state_tool_filter", False) else RETAIL_TOOLS
        if cancel_fallback:
            offered = [tool for tool in offered if tool["function"]["name"] not in WRITE_TOOLS]
        if focused_step:
            offered = [tool for tool in offered if tool["function"]["name"] not in WRITE_TOOLS
                       or tool["function"]["name"] == focused_step["tool"]]
        if focused_prose:
            scope_tools = {focused_step["tool"], "get_user_details", "get_order_details", "get_product_details", "ask_user"}
            offered = [tool for tool in offered if tool["function"]["name"] in scope_tools]
        if information_repair or missing_identity:
            offered = [RETAIL_TOOLS[-1]]
        if repair == "read_pending_orders":
            offered = [tool for tool in offered if tool["function"]["name"] == "get_user_details"]
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
        if focused_prose:
            reads = [{"tool": message.name, "result": json.loads(message.content)}
                     for message in visible if isinstance(message, ToolMessage)
                     and message.name in {"get_user_details", "get_order_details", "get_product_details"}]
            prompt = ("Continue exactly the next explicit customer operation using native tools. "
                      "Only committed_outcomes were confirmed. The next step is unconfirmed. "
                      "Use verified reads to propose it, or call a read tool or ask_user if a prerequisite is missing. "
                      "If the operation is ineligible, explain why. Never claim a write without its own receipt.")
            customer_texts = [message.content for message in active_messages
                              if isinstance(message, HumanMessage) and isinstance(message.content, str)
                              and message.content != "I explicitly confirm the displayed proposal."]
            payload = {"next_step": focused_step, "customer_request": opening, "verified_reads": reads,
                       "customer_followups": customer_texts[1:],
                       "committed_outcomes": [message.content for message in outcomes]}
            visible = [HumanMessage(content=json.dumps(payload, ensure_ascii=False),
                                    additional_kwargs={"protocol_retry_input": True})]
            model_messages = [SystemMessage(content=prompt), *visible]
        proposal_repair_active = proposal_repair_active or (
            repair == "proposal" and config.get("isolated_proposal_repair", False))
        proposal_repair = proposal_repair_active
        if proposal_repair:
            events = []
            for message in visible:
                if isinstance(message, HumanMessage):
                    events.append({"role": "customer", "content": message.content})
                elif message.additional_kwargs.get("application_result"):
                    events.append({"role": "application", "content": message.content})
                elif isinstance(message, ToolMessage):
                    events.append({"role": "tool", "name": message.name, "result": json.loads(message.content)})
                elif isinstance(message, AIMessage) and message.tool_calls:
                    events.append({"role": "assistant", "tool_calls": message.tool_calls})
            prompt += ("\nRecovery step: There is no proposal awaiting confirmation for the next operation. Use the customer messages and tool results below as data. "
                       "Make exactly one native tool call for the next required step. Select item IDs and payment IDs only "
                       "from the tool results, and respect all exclusions and later corrections. If information is missing, "
                       "use ask_user. For an address update, read the authorized order or profile if needed and call the appropriate "
                       "address tool with all fields. For a partial change, retain every unchanged address field. Use an empty address2 only if the destination has no second line. Never replay a completed write. "
                       "Do not output a simulated tool result or claim completion.")
            if latest.additional_kwargs.get("return_confirmation_in_prose"):
                prompt += (" Handle the independently requested return first if its order, item selection and refund "
                           "method are verified. Honor dependencies stated anywhere in the customer request. "
                           "If the return itself needs clarification, use ask_user.")
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
            review_budget = min(max_model_calls, config.get("episode_calls_remaining", max_model_calls))
            copy_read = bool(remaining_copy_step and isinstance(latest, ToolMessage)
                             and latest.name in {"get_order_details", "get_user_details"}
                             and latest.status == "success")
            if copy_read and latest.name == "get_order_details":
                read_order = json.loads(latest.content)["data"]["order_id"]
                copy_read = read_order in {copy["source_order_id"], copy["destination_order_id"]}
            if ((copy_read or (focused_step and state["model_calls"] == 0))
                    and not continuation_checked and state["model_calls"] + 1 < review_budget
                    and not _requests_information(response.content) and not _is_text_tool_action(response.content)):
                continuation_checked = True
                metrics["focused_prose_repairs"] = 1
                if remaining_copy_step and remaining_copy_step["tool"] == "modify_user_address":
                    account_copy_retry_draft_id = response.id
                feedback = SystemMessage(
                    content="The preceding draft did not use a native tool. Work only on the next explicit step. "
                            "Use a native call to read, request missing details, or propose the eligible write. "
                            "If the step is genuinely ineligible, explain the reason using verified facts.",
                    additional_kwargs={"protocol_feedback": "focused_prose"},
                )
                persist(state, [response, feedback])
                return {"messages": [response, feedback], "model_calls": state["model_calls"] + 1, "status": "repairing"}
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
            address_confirmation = (isinstance(text, str) and re.search(r"\b(?:address|suite|shipping)\b", text, re.I)
                                    and re.search(r"\b(?:change|chang(?:ing|ed)|updat\w*|modify|proceed)\b", text, re.I))
            proposal_missing = (isinstance(text, str) and session.get("user_id")
                                and (address_confirmation or session.get("target_order_status") in {None, "pending", "delivered"})
                                and re.search(r"please confirm|(?:could|can|would) you confirm|do you (?:confirm|want)|would you like (?:me )?to (?:proceed|cancel|return|exchange|modify)|\b(?:proposal|return request|cancellation request|exchange request|modification proposal)(?: (?:has been |was |is now )?| for [^.!?\n]+? (?:has been |was |is now ))(?:successfully )?(?:created|submitted|staged)\b", text, re.I)
                                and (address_confirmation or re.search(r"cancel|return|exchange|modify|modification", text, re.I)))
            explicit_confirmation = (isinstance(text, str) and re.search(
                r"please confirm (?:if|whether) you (?:would like|want|wish)|would you like (?:me )?to (?:proceed|cancel|return|exchange|modify)|do you want me to",
                text, re.I))
            return_clause = re.search(r"(?:^|[.!?]\s+)(?:please )?return\s+([^.!?\n]+)", opening, re.I)
            return_order_ids = set(re.findall(r"#W\d+", return_clause[1].upper())) if return_clause else set()
            return_confirmation = bool(
                isinstance(text, str) and session.get("user_id") and return_clause and len(return_order_ids) <= 1
                and not re.search(r"\b(?:if|unless|until|before|after|provided|only when)\b", return_clause[1], re.I)
                and not re.match(r"[.!?]\s+(?:(?:only|do (?:this|that|it) only)\b)",
                                 opening[return_clause.end():], re.I)
                and re.search(r"\bconfirm (?:if|whether) you (?:want|wish|would like) to "
                              r"(?:proceed with returning|return)\b", text, re.I)
                and any(isinstance(message, ToolMessage) and message.name == "get_order_details"
                        and message.status == "success"
                        and (order := json.loads(message.content)["data"])["status"] == "delivered"
                        and order["user_id"] == session["user_id"]
                        and session["business"]["orders"].get(order["order_id"], {}).get("status") == "delivered"
                        and (order["order_id"] in return_order_ids if return_order_ids else
                             any(re.search(r"\b" + re.escape(item["name"].rsplit(maxsplit=1)[-1]) + r"\b",
                                           return_clause[1], re.I) for item in order["items"]))
                        for message in active_messages)
            )
            kind = ("empty" if not text else "proposal" if return_confirmation or (proposal_missing and explicit_confirmation) else
                    "information" if _requests_information(text) else "proposal" if proposal_missing else None)
            receipts = [json.loads(message.content.removeprefix("Application outcome: ")) for message in outcomes
                        if message.additional_kwargs.get("committed_tool")]
            if (kind in {None, "proposal"} and account_copy_retry_draft_id
                    and (repair == "focused_prose" or copy_read) and goal_state is None
                    and remaining_copy_step and remaining_copy_step["tool"] == "modify_user_address"
                    and destination["address"] == copy["address"]
                    and any(receipt["name"] == "modify_pending_order_address"
                            and receipt["order_id"] == copy["destination_order_id"]
                            and receipt.get("user_id") == session["user_id"]
                            and receipt.get("updated_address") == copy["address"] for receipt in receipts)):
                args = {"user_id": session["user_id"], **copy["address"]}
                start = time.perf_counter()
                staged = environment.stage_account_address(thread_id, args)
                metrics["tool_seconds"] += time.perf_counter() - start
                metrics["tool_calls"] += 1
                metrics["tool_attempts"].append({"name": "modify_user_address", "args": args,
                                                 "code": staged["code"], "message": staged.get("message"),
                                                 "attempt": 1, "origin": "application"})
                if staged["code"] == "CONFIRMATION_REQUIRED":
                    proposal = staged["proposal"]
                    marker = SystemMessage(
                        content=f"Application staged account address proposal {proposal['proposal_id']}. "
                                "Separate confirmation is required.",
                        additional_kwargs={
                            "application_proposal": True,
                            "superseded_draft_ids": [account_copy_retry_draft_id, response.id],
                        },
                    )
                    persist(state, [response, marker])
                    return {"messages": [response, marker], "model_calls": state["model_calls"] + 1,
                            "status": "needs_confirmation", "request": proposal}
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
                feedback = SystemMessage(content=feedback_text, additional_kwargs={
                    "protocol_feedback": kind,
                    **({"return_confirmation_in_prose": True} if return_confirmation else {}),
                })
                persist(state, [response, feedback])
                return {"messages": [response, feedback], "model_calls": state["model_calls"] + 1, "status": "repairing"}
            if (item_targets and isinstance(text, str) and isinstance(latest, ToolMessage)
                    and latest.name in IDENTITY_TOOLS and latest.status == "success"
                    and not metrics.get("authenticated_lookup_retries") and state["model_calls"] + 1 < review_budget
                    and re.search(r"\b(?:I will (?:now )?(?:retrieve|check|search|look up)|please wait while I (?:search|check))\b",
                                  text, re.I)):
                metrics["authenticated_lookup_retries"] = 1
                feedback = SystemMessage(
                    content=f"Authentication succeeded. Call get_user_details for {session['user_id']} "
                            "to look up the orders now. Do not tell the customer to wait.",
                    additional_kwargs={"protocol_feedback": "read_pending_orders"},
                )
                persist(state, [response, feedback])
                return {"messages": [response, feedback], "model_calls": state["model_calls"] + 1, "status": "repairing"}
            if (address_request and outcomes and config.get("confirmed_this_turn")
                    and not continuation_checked and state["model_calls"] + 1 < review_budget):
                continuation_checked = True
                metrics["address_completion_reviews"] = 1
                feedback = SystemMessage(
                    content="Before ending, compare every customer request with the committed application outcomes. "
                            "If an operation remains, make one native call to continue. Never repeat a committed action. "
                            "If all goals were processed, or the remaining steps are impossible, report each outcome accurately.",
                    additional_kwargs={"protocol_feedback": "address_completion"},
                )
                persist(state, [response, feedback])
                return {"messages": [response, feedback], "model_calls": state["model_calls"] + 1, "status": "repairing"}
            if isinstance(text, str) and session.get("user_id"):
                cancelled_orders = {
                    json.loads(message.content.removeprefix("Application outcome: ")).get("order_id")
                    for message in outcomes
                    if message.additional_kwargs.get("committed_tool") == "cancel_pending_order"
                }
                if false_cancellation_claim(text, session["business"], session["user_id"], cancelled_orders,
                                            unresolved_cancellation=bool(cancel_requested and outcomes and not cancelled_orders)):
                    return {"messages": [response], "model_calls": state["model_calls"] + 1,
                            "status": "failed", "failure_reason": "unverified_cancellation_claim"}
            if receipts and session.get("user_id"):
                order_reads = []
                source_order_id = None
                if not copy and any(receipt["name"] == "modify_user_address" for receipt in receipts):
                    for message in active_messages:
                        if isinstance(message, ToolMessage) and message.name == "get_order_details":
                            read = json.loads(message.content)
                            if read.get("ok"):
                                order_reads.append(read["data"])
                    references = set(re.findall(r"#W\d+", opening.upper()))
                    if not references and re.search(r"\baddress\b", opening, re.I):
                        owned = session["business"]["users"][session["user_id"]]["orders"]
                        candidates = address_order_candidates(opening, session["business"], session["user_id"], owned)
                        references = {row["order_id"] for row in candidates}
                    if len(references) == 1:
                        source_order_id = references.pop()
                grounded = verified_address_reply(opening, receipts, order_reads, session["user_id"], copy, source_order_id)
                if grounded is not None:
                    metrics["original_model_answer"] = response.content
                    metrics["application_rendered_receipt"] = True
                    reply = AIMessage(content=grounded, additional_kwargs={"application_rendered_receipt": True})
                    return {"messages": [response, reply], "model_calls": state["model_calls"] + 1, "status": "answered"}
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
        copy = None
        read_key = None
        read_session = None
        reused_read = False
        focused_key = "user_id" if focused_step and "user_id" in focused_step else "order_id"
        focused_mismatch = bool(focused_step and name in WRITE_TOOLS and
                                (name, args.get(focused_key)) != (focused_step["tool"], focused_step[focused_key]))
        if focused_mismatch:
            result = {"ok": False, "code": "INVALID_ARGUMENT",
                      "message": f"The next explicit step is {focused_step['tool']} for {focused_step[focused_key]}. "
                                 "This rejected call did not undo any confirmed action. Read or clarify the target if needed."}
            metrics["tool_attempts"].append({"tool_call_id": call["id"], "name": name, "args": args,
                                             "code": result["code"], "message": result["message"], "attempt": 1})
            status, request = "running", None
        elif name not in metrics["context_inputs"][-1]["tool_names"]:
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
                customer_text = " ".join(message.content for message in active_request(state["messages"])
                                         if isinstance(message, HumanMessage) and isinstance(message.content, str))
                address_text = re.sub(r"\be-?mail address\b", "email", args["question"] + " " + customer_text, flags=re.I)
                question = args["question"]
                identity_zip = not re.search(r"\bZIP\b", question, re.I) or re.search(
                    r"\b(?:full|first|last) name and (?:your )?ZIP\b", question, re.I)
                if (set(args["fields"]) == {"identity", "address"} and identity_zip
                        and re.search(r"\b(?:e-?mail|(?:full|first|last) name)\b", args["question"], re.I)
                        and not re.search(r"\b(?:address|street|suite|apartment|ship\w*|delivery|mailing|where|"
                                          r"city|state|country|province|town|postcode|postal code|house|destination)\b",
                                          address_text, re.I)):
                    args = {**args, "fields": ["identity"]}
                    metrics["clarification_fields_normalized"] = metrics.get("clarification_fields_normalized", 0) + 1
                result = {"ok": True, "code": "NEEDS_INPUT", "data": args}
                status, request = "needs_input", args
            else:
                result = {"ok": False, "code": "INVALID_ARGUMENT", "message": "ask_user needs fields and question."}
                status, request = "running", None
        else:
            if name in reusable_reads and valid_arguments(name, args):
                read_key = (name, json.dumps(args, sort_keys=True))
            cached = successful_reads.get(read_key)
            if cached:
                session = environment.load_session(thread_id)
                if (cached[0] != read_scope(session) or session.get("pending_action")
                        or (name == "get_order_details" and session["target_order_id"] != args["order_id"])):
                    cached = None
            if cached:
                result = {**cached[1],
                          "message": "This exact read already succeeded in this turn. Use its verified data. "
                                     "Take the next unresolved step or read a different target if needed."}
                reused_read = True
                metrics["reused_reads"] += 1
            elif (name == "cancel_pending_order" and goal_state is not None
                    and (goal_state.get("needs_refresh") or goal_state.get("scope") == "none"
                         or args.get("order_id") not in (goal_state.get("targets") or []))):
                result = {"ok": False, "code": "POLICY_DENIED", "message": "Cancellation is outside the recorded goal. No proposal was created."}
            elif name in IDENTITY_TOOLS and valid_arguments(name, args) and not _identity_is_grounded(args, state["messages"]):
                result = {"ok": False, "code": "MISSING_USER_INFORMATION",
                          "message": "These identity values were not supplied by the customer. Call ask_user for identity; do not guess.",
                          "fields": ["identity"]}
            elif name in ADDRESS_TOOLS or name == "modify_pending_order_items":
                active = active_request(state["messages"])
                opening = request_text(active)
                session = environment.load_session(thread_id)
                business, user_id = session["business"], session["user_id"]
                result = None
                if name in ADDRESS_TOOLS:
                    copy = address_copy_evidence(active, opening, session)
                    if copy and name == "modify_user_address" and not copy["update_default"]:
                        result = {"ok": False, "code": "INVALID_ARGUMENT",
                                  "message": "No proposal created. The customer only requested a shipping address change."}
                    elif copy:
                        target = ({"order_id": copy["destination_order_id"]} if name == "modify_pending_order_address"
                                  else {"user_id": user_id})
                        expected = {**target, **copy["address"]}
                        if args != expected:
                            result = {"ok": False, "code": "INVALID_ARGUMENT",
                                      "message": "No proposal created. Use the copy destination and every source address field.",
                                      "data": expected}
                elif valid_arguments(name, args):
                    order = business["orders"].get(args["order_id"])
                    targets = item_change_targets(opening, business, user_id)
                    mismatch = wrong_item_target(opening, targets, order, args) if order and order["user_id"] == user_id else None
                    conflicts = preserved_variant_conflicts(opening, business, user_id, args)
                    if order and order["user_id"] == user_id and item_order_excluded(opening, args["order_id"]):
                        result = {"ok": False, "code": "INVALID_ARGUMENT",
                                  "message": "No proposal created. The customer requested an item in another order. "
                                             "Read or clarify that order before proposing a change."}
                    elif mismatch:
                        result = {"ok": False, "code": "INVALID_ARGUMENT",
                                  "message": "No proposal created. The item_id below matches the requested original item; "
                                             "read its catalog and use it in item_ids, not in new_item_ids.",
                                  "data": mismatch}
                    elif conflicts:
                        result = {"ok": False, "code": "INVALID_ARGUMENT",
                                  "message": "No proposal created. The selected variant changes required unchanged options: "
                                             + ", ".join(conflicts) + ". Check the complete catalog for an available match. "
                                             "If none exists, follow the customer's stated fallback."}
                    else:
                        for old_id, new_id in zip(args["item_ids"], args["new_item_ids"]):
                            target = next((target for target in targets if target["order_id"] == args["order_id"]
                                           and target["item_id"] == old_id), None)
                            if target:
                                variants = business["products"][target["product_id"]]["variants"]
                                choices = requested_variants(variants, target["requested_variant"])
                                if choices and new_id not in {choice["item_id"] for choice in choices}:
                                    result = {"ok": False, "code": "INVALID_ARGUMENT",
                                              "message": "No proposal created. The replacement misses the requested catalog requirement.",
                                              "data": {"requested_variant": target["requested_variant"],
                                                       "available_option_hits": choices[:8]}}
                                    break
                if result is None:
                    result = environment.dispatch_tool(thread_id, name, args)
            else:
                result = environment.dispatch_tool(thread_id, name, args)
            metrics["tool_attempts"].append({"tool_call_id": call["id"], "name": name, "args": args,
                                             "code": result["code"], "message": result.get("message"), "attempt": 1,
                                             **({"reused_read": True} if reused_read else {})})
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
        if config.get("catalog_availability_facts", False) and name == "get_product_details" and result["ok"]:
            result["data"]["available_variant_count"] = sum(
                variant["available"] is True for variant in result["data"]["variants"].values())
        if name in {"get_order_details", "get_product_details", "get_user_details"} and result["ok"] and not reused_read:
            active = active_request(state["messages"])
            opening = request_text(active)
            session = read_session = environment.load_session(thread_id)
            targets = item_change_targets(opening, session["business"], session["user_id"])
            if name == "get_user_details":
                copy = address_copy_evidence(active, opening, session)
                if copy:
                    result["data"]["address_copy"] = copy
                if targets:
                    result["data"]["item_change_targets"] = targets
                candidates = address_order_candidates(opening, session["business"], session["user_id"],
                                                      result["data"]["orders"])
                if candidates:
                    result["data"]["address_order_candidates"] = candidates
            elif name == "get_order_details":
                targets = [target for target in targets if target["order_id"] == args["order_id"]]
                if targets:
                    result["data"]["item_change_targets"] = targets
            else:
                product = result["data"]
                preferences = {target["requested_variant"] for target in targets
                               if target["product_id"] == args["product_id"]}
                if len(preferences) == 1:
                    preference = preferences.pop()
                    candidates = requested_variants(product["variants"], preference)
                    if candidates:
                        product["available_option_hits"] = candidates[:8]
        if name == "get_product_details" and result["ok"]:
            active = active_request(state["messages"])
            product = result["data"]
            for step in explicit_address_steps(request_text(active)):
                if step["tool"] != "modify_pending_order_items":
                    continue
                for message in reversed(active):
                    if not isinstance(message, ToolMessage) or message.name != "get_order_details":
                        continue
                    read = json.loads(message.content)
                    if read.get("ok") and read["data"]["order_id"] == step["order_id"]:
                        if any(item["product_id"] == args["product_id"]
                               and re.search(r"(?<!\w)" + re.escape(item["name"]) + r"(?!\w)",
                                             step["item_description"], re.I)
                               for item in read["data"]["items"]):
                            candidates = matching_variants(product["variants"], step["preferences"])
                            if candidates:
                                product["available_option_hits"] = candidates
                        break
        if config.get("profile_order_summaries", False) and name == "get_user_details" and result["ok"] and not reused_read:
            session = environment.load_session(thread_id)
            result["data"]["order_summaries"] = [
                {"order_id": order_id, "status": order["status"],
                 "items": [{key: item[key] for key in ("name", "item_id", "product_id", "options")}
                           for item in order["items"]]}
                for order_id in result["data"]["orders"]
                if (order := session["business"]["orders"][order_id])["user_id"] == session["user_id"]]
        if read_key is not None and result["ok"] and not reused_read:
            if name == "get_order_details":
                for key in tuple(successful_reads):
                    if key[0] == name:
                        del successful_reads[key]
            successful_reads[read_key] = (read_scope(read_session or environment.load_session(thread_id)), result)
        metrics["tool_seconds"] += time.perf_counter() - start
        metrics["tool_calls"] += int(not reused_read) + retries
        metadata = {"focused_write_mismatch": True} if focused_mismatch else {}
        if copy and status == "needs_confirmation" and name in ADDRESS_TOOLS:
            metadata.update(address_copy=copy, address_copy_user_id=session["user_id"])
        message = ToolMessage(content=json.dumps(result, ensure_ascii=False), name=name, tool_call_id=call["id"],
                              status="success" if result["ok"] else "error", additional_kwargs=metadata)
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
