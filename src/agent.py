"""
The agent, built as an explicit LangGraph StateGraph.

I chose a hand-built graph over the prebuilt `create_react_agent` on purpose:
the two custom nodes below let me keep a readable trace, intercept the
`final_answer` tool for a clean stop condition, and enforce real safety limits —
none of which you get cleanly from the one-liner.

        ┌───────┐   tool calls    ┌───────┐
        │ agent │ ──────────────▶ │ tools │
        └───────┘                 └───────┘
            ▲   │ no tool calls        │
            │   └────────▶ END         │ final_answer / limits ─▶ END
            └──────────────────────────┘  otherwise loop back

Safety rails (see config.py to tune):
  * max_steps        — hard iteration cap
  * max_tokens_budget— cumulative-token cost cap
  * max_repeats      — loop detection: refuse an identical tool call after N tries
  * max_retries      — retry transient LLM failures with backoff
  * exec_timeout     — per-tool execution timeout (in the tool itself)
"""

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from .llm import get_chat_model, message_text
from .prompts import SYSTEM_PROMPT
from .config import settings
from .tools import TOOL_SCHEMAS, TOOL_FUNCS


# --- Graph state -----------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int
    tokens: int                 # cumulative token spend, for the cost budget
    trace: list
    answer: Optional[str]
    stopped_reason: Optional[str]
    nudges: int                 # how many times we've re-asked for final_answer


@dataclass
class AgentResult:
    answer: str
    steps_taken: int
    trace: list[dict] = field(default_factory=list)
    stopped_reason: str = "final_answer"
    tokens_used: int = 0


def _run_tool(name: str, args: dict) -> str:
    """Dispatch one tool, turning any crash into a string the model can read
    and recover from on the next turn."""
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"[error] Unknown tool: {name}"
    try:
        return func(**args)
    except TypeError as exc:  # usually a wrong argument name from the model
        return f"[error] Bad arguments for {name}: {exc}"


def _repeat_count(trace: list, name: str, args: dict) -> int:
    """How many times this exact (tool, args) call already ran — for loop
    detection. A model stuck in a rut tends to re-issue the identical call."""
    return sum(
        1 for t in trace
        if t.get("type") == "tool_call" and t.get("name") == name and t.get("args") == args
    )


# --- Nodes -----------------------------------------------------------------

def _build_nodes(model: Any):
    """Bind tools to the model (with retry/backoff) and close over it."""
    bound = model.bind_tools(TOOL_SCHEMAS)
    # LangChain Runnables ship a retry wrapper with exponential backoff — use it
    # so a rate-limit or network blip retries instead of crashing the run.
    if hasattr(bound, "with_retry"):
        bound = bound.with_retry(stop_after_attempt=settings.max_retries)

    def agent_node(state: AgentState) -> dict:
        ai_msg = bound.invoke(state["messages"])
        # Track token spend for the cost budget (not all models report it).
        usage = getattr(ai_msg, "usage_metadata", None)
        spent = usage.get("total_tokens", 0) if usage else 0
        return {
            "messages": [ai_msg],
            "steps": state["steps"] + 1,
            "tokens": state["tokens"] + spent,
        }

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_messages = []
        trace = list(state["trace"])
        answer = None
        stop = None

        for call in last.tool_calls:
            name, args, call_id = call["name"], call["args"], call["id"]

            if name == "final_answer":
                answer = str(args.get("answer", "")).strip()
                stop = "final_answer"
                trace.append({"type": "final_answer", "content": answer})
                tool_messages.append(ToolMessage(content="acknowledged", tool_call_id=call_id))
                continue

            # Loop detection: don't keep re-running an identical failing call.
            if _repeat_count(trace, name, args) >= settings.max_repeats:
                obs = (f"[loop detected] You've already called {name} with these exact "
                       f"arguments {settings.max_repeats} times without progress. "
                       "Change your approach or submit your best answer.")
                trace.append({"type": "loop_stop", "name": name, "args": args,
                              "observation": obs})
                tool_messages.append(ToolMessage(content=obs, tool_call_id=call_id))
                continue

            observation = _run_tool(name, args)
            trace.append({"type": "tool_call", "name": name, "args": args,
                          "observation": observation})
            tool_messages.append(ToolMessage(content=observation, tool_call_id=call_id))

        update: dict = {"messages": tool_messages, "trace": trace}
        if answer is not None:
            update["answer"] = answer
            update["stopped_reason"] = stop
        return update

    def nudge_node(state: AgentState) -> dict:
        """The model answered in prose instead of calling `final_answer`.
        Rather than scraping a number out of its sentence, re-ask for the tool
        call once. Reasoning models in particular often narrate the answer and
        forget to submit it."""
        said = message_text(state["messages"][-1])
        trace = list(state["trace"])
        trace.append({"type": "nudge", "content": said})
        return {
            "messages": [HumanMessage(content=(
                "You did not submit an answer. Call the `final_answer` tool now "
                "with just the value — no working, no units."))],
            "trace": trace,
            "nudges": state["nudges"] + 1,
        }

    return agent_node, tools_node, nudge_node


# --- Routing ---------------------------------------------------------------

def _over_budget(state: AgentState) -> bool:
    return settings.max_tokens_budget > 0 and state["tokens"] >= settings.max_tokens_budget


def _route_after_agent(state: AgentState) -> str:
    if state["steps"] >= settings.max_steps or _over_budget(state):
        return "stop"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    # No tool call: the model stopped talking without submitting. Ask once.
    if state["nudges"] < settings.max_nudges:
        return "nudge"
    return "stop"


def _route_after_tools(state: AgentState) -> str:
    if state.get("answer") is not None:
        return "stop"
    if state["steps"] >= settings.max_steps or _over_budget(state):
        return "stop"
    return "agent"


def build_graph(model: Any = None, checkpointer: Any = None):
    """Compile the agent graph. `model` is injectable so tests can pass a fake
    chat model and exercise the whole loop without a network call.

    `checkpointer` is LangGraph's memory hook: give it one and the graph saves
    its state per `thread_id`, so a later run on the same thread resumes with the
    earlier messages already in context. Leave it None for a one-shot solve."""
    if model is None:
        model = get_chat_model()

    agent_node, tools_node, nudge_node = _build_nodes(model)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("nudge", nudge_node)
    graph.add_edge(START, "agent")
    graph.add_edge("nudge", "agent")
    graph.add_conditional_edges(
        "agent", _route_after_agent,
        {"tools": "tools", "nudge": "nudge", "stop": END},
    )
    graph.add_conditional_edges("tools", _route_after_tools, {"agent": "agent", "stop": END})
    return graph.compile(checkpointer=checkpointer)


# --- Public entrypoint -----------------------------------------------------

# Conversation memory. One process-wide MemorySaver keeps each thread's state
# between calls to solve(); it's in-RAM only, which is the right size for a demo
# (swap in SqliteSaver for something that survives a restart).
_MEMORY = MemorySaver()


def solve(problem: str, model: Any = None, thread_id: str | None = None) -> AgentResult:
    # Input validation: reject empty / oversized problems before spending tokens.
    problem = (problem or "").strip()
    if not problem:
        return AgentResult(answer="", steps_taken=0, stopped_reason="empty_input")
    if len(problem) > settings.max_problem_chars:
        return AgentResult(
            answer="",
            steps_taken=0,
            stopped_reason="input_too_long",
            trace=[{"type": "thought",
                    "content": f"Problem exceeds {settings.max_problem_chars} chars."}],
        )

    config: dict = {"recursion_limit": settings.max_steps * 2 + 5}

    if thread_id is None:
        # One-shot: no checkpointer, no history, nothing to clean up.
        graph = build_graph(model)
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=problem)]
    else:
        # Conversational: resume this thread. The saved messages come back
        # automatically, so we only send the new question — and the system
        # prompt just once, when the thread is empty.
        graph = build_graph(model, checkpointer=_MEMORY)
        config["configurable"] = {"thread_id": thread_id}
        saved = graph.get_state(config).values
        messages = [HumanMessage(content=problem)]
        if not saved.get("messages"):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    # steps/tokens/trace have no reducer, so passing them in resets the
    # per-question counters even on a resumed thread (only `messages` accumulates).
    initial: AgentState = {
        "messages": messages,
        "steps": 0,
        "tokens": 0,
        "trace": [],
        "answer": None,
        "stopped_reason": None,
        "nudges": 0,
    }
    try:
        final = graph.invoke(initial, config=config)
    except Exception as exc:  # noqa: BLE001
        # A provider-side failure that survived the retries (malformed tool call,
        # quota, outage). One bad question shouldn't abort a 200-question eval,
        # so it becomes a scored miss with a legible reason.
        return AgentResult(
            answer="", steps_taken=0, stopped_reason="llm_error",
            trace=[{"type": "thought", "content": f"[error] {type(exc).__name__}: {exc}"}],
        )

    answer = final.get("answer")
    stopped_reason = final.get("stopped_reason")

    # No explicit final_answer — model answered in text, or we hit a limit.
    if not answer:
        # Scan back only as far as the question we just asked — on a resumed
        # thread, anything before it belongs to an earlier answer.
        history = final["messages"]
        for i in range(len(history) - 1, -1, -1):
            if isinstance(history[i], HumanMessage):
                history = history[i:]
                break
        for msg in reversed(history):
            if isinstance(msg, AIMessage) and message_text(msg):
                answer = message_text(msg)
                break
        if not stopped_reason:
            if settings.max_tokens_budget and final["tokens"] >= settings.max_tokens_budget:
                stopped_reason = "token_budget"
            elif final["steps"] >= settings.max_steps:
                stopped_reason = "step_limit"
            elif final["nudges"] >= settings.max_nudges:
                stopped_reason = "no_final_answer_call"
            else:
                stopped_reason = "plain_text_answer"
        final["trace"].append({"type": "thought", "content": answer or ""})

    return AgentResult(
        answer=answer or "",
        steps_taken=final["steps"],
        trace=final["trace"],
        stopped_reason=stopped_reason or "final_answer",
        tokens_used=final["tokens"],
    )
