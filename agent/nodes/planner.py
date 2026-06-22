from __future__ import annotations

from typing import List, Optional
import logging

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, ConfigDict, Field

import core.config as config
from core.llm import llm
from agent.state import GraphState, Plan
from loaders.schema_loader import build_dataset_schemas, build_dataset_catalogue
from utils.catalogues import build_tool_catalogue, build_segment_catalogue
from utils.knowledge import get_knowledge




logger = logging.getLogger(__name__)

# ── Structured output model ───────────────────────────────────────────────────
class ToolPlannerOut(BaseModel):
    plans: List[Plan]
    intuition: str = Field(description="Reasoning behind the planning decisions")
    needs_clarification: bool = Field(
        description="True if more info is needed from the user before planning"
    )
    clarification_needed: str = Field(
        description="What specific info is missing, if needs_clarification is True"
    )

    model_config = ConfigDict(json_schema_extra={"required": ["plans"]})


# ── The planner node ───────────────────────────────────────────────────────────
def tool_planner(state: GraphState) -> dict:
    """
    Builds a sequence of tool calls that will satisfy the user's intent.

    Reads: question_intent, candidate_dataset_names, available_tools, segment_registry
    Writes: plans, plans_intuition, needs_clarification, clarification_needed
    """

    logger.info(f"[PLANNER] Building plan for intent: '{state.question_intent}'")

    tools_desc = build_tool_catalogue(state.candidate_dataset_names, state.available_tools)
    segments_catalog = build_segment_catalogue(state.segment_registry, state.candidate_dataset_names)
    dataset_catalog = build_dataset_catalogue(build_dataset_schemas(), lite=False)
    planner_guidance = get_knowledge(config.knowledge_paths.TOOL_PLANNER_RULES_DIR)

    sys = SystemMessage(content=f"""
You are an expert tool planning agent. Create a precise, executable plan
that achieves the user's intent using the available tools.

A GOOD plan:
- Uses the minimum number of tools necessary
- Maps all required arguments correctly
- Only uses segments that exist in the segments catalog
- Is fully executable without further human input, unless truly ambiguous

---
User's Intent:
{state.question_intent}

Available Datasets:
{dataset_catalog}

Available Segments:
{segments_catalog}

Available Tools:
{tools_desc}

Tool Planning Guidance:
{planner_guidance}

---
## ARGUMENT MAPPING RULES

### Required arguments (no default)
1. Use explicit value from intent if present
2. Infer from context if reasonably possible
3. If truly missing → set needs_clarification = true

### Optional arguments (have defaults)
Use specified value if intent mentions it, otherwise omit (tool uses its default).

### Segment arguments
Only use segment names that exist in the segments catalog above.
Never invent a segment value. If no matching segment exists, omit the parameter.

### Date arguments
Today's date: {state.todays_date}
"last month" → previous calendar month
"YTD" → start of year to today
"Q1" → Jan 1 to Mar 31
If as_of_date not specified, default to today's date.

---
## WHEN TO ASK FOR CLARIFICATION
Set needs_clarification = true ONLY if:
- A required argument has no default and cannot be inferred
- The request is fundamentally ambiguous (e.g. "show performance" — which metric?)

Do NOT ask for clarification if you can use a default or reasonable inference.

---
Always explain your reasoning in intuition. Be specific in clarification_needed
if clarification is required — name exactly what's missing.
""")

    resp = llm.with_structured_output(ToolPlannerOut).invoke([sys])

    logger.info(
        f"[PLANNER] Generated {len(resp.plans)} step(s) | "
        f"needs_clarification={resp.needs_clarification}"
    )

    for p in resp.plans:
        logger.info(f"[PLANNER]   Step {p.step}: {p.tool} on {p.dataset}")

    if resp.needs_clarification:
        logger.info(f"[PLANNER] Clarification needed: {resp.clarification_needed}")

    return {
        "plans": resp.plans,
        "plans_intuition": resp.intuition,
        "needs_clarification": resp.needs_clarification,
        "clarification_needed": resp.clarification_needed,
    }


# ── Conditional edge function ──────────────────────────────────────────────────
def needs_clarification(state: GraphState) -> bool:
    """Reads state.needs_clarification to decide: clarifier node or proceed to execution."""
    return state.needs_clarification


# ── Clarifier node ─────────────────────────────────────────────────────────────
def user_clarifier(state: GraphState) -> dict:
    """
    When the planner can't proceed, this generates a friendly,
    natural follow-up question for the user instead of a raw error.
    """
    from langchain_core.messages import HumanMessage

    sys_msg = SystemMessage(content="""
You are helping the user clarify their request. Ask one friendly, natural
question that helps collect the missing information below.
Do not mention words like 'parameter', 'argument', or 'dataset'.
Use plain, conversational language.
""")

    human_msg = HumanMessage(content=(
        f'The user asked: "{state.user_questions}"\n\n'
        f"We still need: {state.clarification_needed}"
    ))

    followup = llm.invoke([sys_msg, human_msg])

    return {
        "clarification_needed": followup.content,
        "response": followup.content,
    }