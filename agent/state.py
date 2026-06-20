from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from utils.segment_utils import Segment


# ── Plan argument key-value pair ─────────────────────────────────────────────
# Each argument in a plan step is stored as {key, value} pair.
# This is how the planner LLM returns args in a structured way.
class ArgKV(BaseModel):
    key: str = Field(..., description="Argument name")
    value: Union[
        str,
        int,
        float,
        bool,
        None,
        List[Union[str, int, float, bool, None]],
        Dict[str, Union[str, int, float, bool, None]],
    ] = Field(..., description="Argument value")


# ── A single step in the execution plan ──────────────────────────────────────
# The planner produces a list of these. Each one maps to one tool call.
class Plan(BaseModel):
    step: int
    tool: str
    action: str
    dataset: str
    args: List[ArgKV] = Field(
        default_factory=list,
        description="Arguments to pass to the tool as key/value pairs",
    )
    depends_on: Optional[List[int]] = None


# ── Central graph state ───────────────────────────────────────────────────────
# This Pydantic model flows through every single node.
# Nodes read from it and return dicts with only the fields they update.
class GraphState(BaseModel):

    # ── Request context ───────────────────────────────────────────────────
    todays_date: str
    user_currency: Optional[str] = "₹"
    user_questions: str
    chat_history: List[dict] = []

    # ── Loaded resources (set once at graph init) ─────────────────────────
    segment_registry: Dict[str, Segment]
    available_tools: List[StructuredTool]
    available_dataset_schemas: Dict[str, Dict[str, Any]]
    datasets: Dict[str, List[Dict[str, Any]]] = {}

    # ── User segment overrides (from request) ─────────────────────────────
    # e.g. {"churned_customers": {"days_threshold": 60}}
    user_segment_default_overrides: Dict[str, Any] = {}

    # ── Router outputs ────────────────────────────────────────────────────
    question_intent: Optional[str] = None
    query_type: str = "analysis"
    candidate_dataset_names: List[str] = []
    answer_type: str = "short-answer"
    answer_type_intuition: Optional[str] = None
    segment_suggestion: bool = False

    # ── Planner outputs ───────────────────────────────────────────────────
    plans: List[Plan] = []
    plans_intuition: Optional[str] = None
    needs_clarification: bool = False
    clarification_needed: Optional[str] = None

    # ── Per-plan subgraph tracking ────────────────────────────────────────
    current_plan_idx: int = 0
    have_more_plans: bool = False

    # ── Tool execution results ────────────────────────────────────────────
    # Keyed by plan step index (0, 1, 2...)
    tool_results: Dict[int, Any] = Field(default_factory=dict)
    tool_multimedia_results: Dict[int, Optional[Dict[str, Any]]] = Field(
        default_factory=dict
    )
    tool_errors: Dict[int, Any] = Field(default_factory=dict)

    # ── Corrective RAG results ────────────────────────────────────────────
    partial_results: Dict[int, Any] = Field(default_factory=dict)
    partial_errors: Dict[int, Any] = Field(default_factory=dict)

    # ── Plan result formatting ────────────────────────────────────────────
    plan_results: Dict[int, Any] = Field(default_factory=dict)
    plan_errors: Dict[int, Any] = Field(default_factory=dict)

    # ── RAG retry counters ────────────────────────────────────────────────
    tool_corrective_rag_retry_count: Dict[int, int] = Field(default_factory=dict)
    adaptive_rag_retry_count: int = 0
    self_rag_retry_count: int = 0
    self_rag_retry_reason: str = ""

    # ── Synthesis outputs ─────────────────────────────────────────────────
    combined_answer: Optional[str] = None
    combiner_intuition: Optional[str] = None
    merged_answer: Optional[str] = None
    non_hallucinated_answer: Optional[str] = None

    # ── Final output ──────────────────────────────────────────────────────
    response: str = ""

    # ── Routing signal flags (act as edge conditions) ─────────────────────
    # These are boolean signals that conditional edges read to decide
    # which node to go to next. They are NOT business data.
    branch: Optional[bool] = None       # used by per-plan subgraph
    retry: Optional[bool] = None        # used by corrective RAG
    reroute: Optional[bool] = None      # used by adaptive RAG
    reroute_self: Optional[bool] = None # used by self corrective RAG

    class Config:
        arbitrary_types_allowed = True