from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from agent.state import GraphState
from agent.nodes.router import intent_dataset_router, route_by_query_type
from agent.nodes.planner import tool_planner, needs_clarification, user_clarifier
from agent.nodes.handlers import (
    handle_general_query,
    handle_explanation_query,
    handle_segment_query,
    handle_unsupported_query,
)


# ── Temporary stub node ────────────────────────────────────────────────────────
# Phase 7 only: stands in for the per_plan_subgraph + combiner + RAG + final_response
# that we haven't built yet. Lets us see what the planner produced without
# crashing the graph. We'll delete this in Phase 13 once the real pipeline exists.
def _stub_show_plan(state: GraphState) -> dict:
    if state.plans:
        plan_summary = "\n".join(
            f"  Step {p.step}: {p.tool}({[f'{a.key}={a.value}' for a in p.args]}) on {p.dataset}"
            for p in state.plans
        )
        response = (
            f"[STUB — execution pipeline not built yet]\n\n"
            f"Intent: {state.question_intent}\n"
            f"Datasets: {state.candidate_dataset_names}\n"
            f"Plan generated:\n{plan_summary}\n\n"
            f"Reasoning: {state.plans_intuition}"
        )
    else:
        response = "[STUB] No plan was generated."

    return {"response": response}


# ── Graph builder ───────────────────────────────────────────────────────────────
def build_graph(datasets: Dict[str, List[Any]]):
    """
    Builds the partial agent graph for Phase 7.

    Wired so far:
        user_input → intent_dataset_router → [route by query_type]
            analysis     → tool_planner → [needs_clarification?]
                                              True  → user_clarifier → END
                                              False → _stub_show_plan → END
            explanation  → handle_explanation_query → END
            general      → handle_general_query → END
            segment      → handle_segment_query → END
            unsupported  → handle_unsupported_query → END

    NOT wired yet (comes in later phases):
        per_plan_subgraph, combiner, adaptive_rag, self_corrective_rag, final_response
    """
    sg = StateGraph(GraphState)

    # ── Entry node ────────────────────────────────────────────────────────────
    def _user_input_node(state: GraphState) -> dict:
        return {"datasets": datasets}

    sg.add_node("user_input", _user_input_node)
    sg.set_entry_point("user_input")

    # ── Router ────────────────────────────────────────────────────────────────
    sg.add_node("intent_dataset_router", intent_dataset_router)
    sg.add_edge("user_input", "intent_dataset_router")

    # ── Handlers (terminal nodes) ────────────────────────────────────────────
    sg.add_node("handle_general_query", handle_general_query)
    sg.add_node("handle_explanation_query", handle_explanation_query)
    sg.add_node("handle_segment_query", handle_segment_query)
    sg.add_node("handle_unsupported_query", handle_unsupported_query)

    # ── Planner path ──────────────────────────────────────────────────────────
    sg.add_node("tool_planner", tool_planner)
    sg.add_node("user_clarifier", user_clarifier)
    sg.add_node("_stub_show_plan", _stub_show_plan)

    # ── Route from router to the right path based on query_type ─────────────
    sg.add_conditional_edges(
        "intent_dataset_router",
        route_by_query_type,
        {
            "analysis": "tool_planner",
            "explanation": "handle_explanation_query",
            "general": "handle_general_query",
            "segment": "handle_segment_query",
            "unsupported": "handle_unsupported_query",
        },
    )

    # ── From planner, route based on whether clarification is needed ────────
    sg.add_conditional_edges(
        "tool_planner",
        needs_clarification,
        {
            True: "user_clarifier",
            False: "_stub_show_plan",
        },
    )

    # ── All terminal edges ────────────────────────────────────────────────────
    sg.add_edge("handle_general_query", END)
    sg.add_edge("handle_explanation_query", END)
    sg.add_edge("handle_segment_query", END)
    sg.add_edge("handle_unsupported_query", END)
    sg.add_edge("user_clarifier", END)
    sg.add_edge("_stub_show_plan", END)

    return sg.compile()