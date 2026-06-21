from __future__ import annotations

from typing import List, Literal
import logging


from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

import core.config as config
from core.llm import llm
from agent.state import GraphState
from loaders.schema_loader import build_dataset_schemas, build_dataset_catalogue
from utils.formatters import format_chat_history_for_llm
from utils.knowledge import get_knowledge, report_template_catalogue


# ── Structured output model ───────────────────────────────────────────────────
# This is what the LLM is forced to return. with_structured_output()
# guarantees the response matches this Pydantic shape exactly.

logger = logging.getLogger(__name__)


class IntentDatasetOut(BaseModel):
    question_intent: str = Field(
        description="Standalone description of what the user wants, no pronouns"
    )
    candidate_dataset_names: List[str] = Field(
        description="Dataset names needed to satisfy the intent"
    )
    query_type: Literal["explanation", "general", "analysis", "segment", "unsupported"] = Field(
        default="analysis",
        description=(
            "'analysis' for questions needing tool use, "
            "'explanation' for clarifying previous results, "
            "'general' for questions about system capabilities, "
            "'segment' for questions about segment availability/meaning/values, "
            "'unsupported' for anything outside scope"
        ),
    )
    answer_type: str = Field(
        default="short-answer",
        description="Report format to use. Defaults to short-answer."
    )
    intuition: str = Field(
        description="Brief reasoning for the answer_type choice"
    )
    segment_suggestion: bool = Field(
        default=False,
        description="True only when user wants actual segment VALUES, not definitions"
    )


# ── The router node ───────────────────────────────────────────────────────────
def intent_dataset_router(state: GraphState) -> dict:
    """
    Classifies the user's question into a query_type and extracts
    the standalone intent + relevant datasets.

    This is the FIRST decision point in the entire graph. Every
    downstream node depends on what this node decides.
    """

    logger.info(f"[ROUTER] Processing question: '{state.user_questions[:80]}'")

    try:
        # ── Build context using Phase 4 utils ──────────────────────────────
        dataset_schemas = build_dataset_schemas()
        datasets_desc = build_dataset_catalogue(dataset_schemas)

        chat_messages = format_chat_history_for_llm(state.chat_history)

        reports_catalog, _ = report_template_catalogue()

        intent_knowledge = get_knowledge(config.knowledge_paths.INTENT_MAPPING_DIR)

        # ── Build the system prompt ────────────────────────────────────────
        system_prompt = SystemMessage(
            content=f"""
You are an expert routing assistant that classifies user queries based on conversation context.

Your task: Analyze the user's query and chat history to determine:
1. query_type — category of the query
2. question_intent — standalone description of what the user wants
3. candidate_dataset_names — datasets needed (empty for non-analysis)
4. answer_type — short-answer or a report template name
5. segment_suggestion — true ONLY if user wants actual segment values

---
## QUERY TYPE DEFINITIONS

### analysis (default)
New question requiring data processing or tool use.
Examples: "What's my churn rate?", "Show revenue by category", "Compare Q1 vs Q2"
Requirements: candidate_dataset_names must include relevant dataset(s)

### explanation
Clarification about a result ALREADY shown in chat history.
Examples: "Why was that number low?", "What does that mean?"
Requirements: candidate_dataset_names MUST be empty []

### general
Questions about system capabilities, available data, or general terms.
Examples: "What can you do?", "What datasets are available?"
Requirements: candidate_dataset_names MUST be empty []

### segment
Questions about segmentation dimensions or their valid values.
Examples: "What segments exist?", "What does churned mean?"
If user wants definition only → segment_suggestion = false, datasets = []
If user wants actual VALUES → segment_suggestion = true, include dataset

### unsupported
Completely unrelated to data analysis.
Examples: "Write me a poem", "What's the weather?"
Requirements: candidate_dataset_names MUST be empty []

---
## CREATING question_intent
Must be standalone — no "it", "that", "this" without resolution.
Resolve references from chat history.

Example: "What about January?" after "Show Q4 retention"
→ "User wants to see retention rate for January"

---
## DATASET SELECTION
Available Datasets:
{datasets_desc}

Match query keywords to dataset descriptions. If query_type requires
empty datasets (explanation/general/unsupported), return [].

---
## ANSWER TYPE SELECTION
Available Report Templates:
{reports_catalog}

Use 'short-answer' for:
- Single metric requests
- Quick factual questions
- Simple breakdowns

Use a report template for:
- Explicit "report" or "analysis" requests
- "Why" or "what caused" questions
- Multi-dimensional comparisons
- Strategic decision support

If no template matches, default to 'short-answer'.

---
## EXTRA GUIDELINES
{intent_knowledge}

---
## TODAY'S DATE
{state.todays_date}

---
Always explain your answer_type choice in the intuition field.
Default to 'analysis' unless clearly one of the other types.
"""
        )

        messages = (
            [system_prompt]
            + chat_messages
            + [{"role": "user", "content": state.user_questions}]
        )

        # ── Call the LLM with structured output ────────────────────────────
        resp = llm.with_structured_output(IntentDatasetOut).invoke(messages)

        logger.info(
            f"[ROUTER] Decision → query_type={resp.query_type} | "
            f"datasets={resp.candidate_dataset_names} | "
            f"answer_type={resp.answer_type}"
        )
        logger.debug(f"[ROUTER] Full intuition: {resp.intuition}")


        # ── Return only the fields this node updates ────────────────────────
        return {
            "question_intent": resp.question_intent,
            "candidate_dataset_names": resp.candidate_dataset_names,
            "query_type": resp.query_type,
            "answer_type": resp.answer_type,
            "answer_type_intuition": resp.intuition,
            "segment_suggestion": resp.segment_suggestion,
        }

    except Exception as e:
        logger.exception(f"[ROUTER ERROR] Failed to route query: {e}")
        raise


# ── Conditional edge function ─────────────────────────────────────────────────
def route_by_query_type(state: GraphState) -> str:
    """
    Reads state.query_type and returns the routing key.
    This function is what LangGraph's conditional edge actually calls
    to decide which node to go to next.
    """
    valid_types = {"analysis", "explanation", "general", "segment"}

    if state.query_type in valid_types:
        return state.query_type

    print(f"[ROUTER] Truly unrecognized query_type '{state.query_type}', defaulting to unsupported")
    return "unsupported"