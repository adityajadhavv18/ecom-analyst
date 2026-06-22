from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

import core.config as config
from core.llm import response_llm
from agent.state import GraphState
from loaders.schema_loader import build_dataset_schemas, build_dataset_catalogue
from loaders.tool_loader import load_tools
from utils.catalogues import build_tool_catalogue, build_segment_catalogue
from utils.formatters import format_chat_history_for_llm, make_sample_of_dataset_with_record
from utils.knowledge import get_knowledge


# ── General query handler ──────────────────────────────────────────────────────
def handle_general_query(state: GraphState) -> dict:
    """
    Answers questions about system capabilities, available data, or
    general terminology. No tool calls — pure conversational response.
    """
    chat_history = format_chat_history_for_llm(state.chat_history)
    general_knowledge = get_knowledge(config.knowledge_paths.PRODUCT_FAQS_AND_GUIDES_DIR)

    dataset_desc = build_dataset_catalogue(build_dataset_schemas())
    dataset_sample = make_sample_of_dataset_with_record(state)

    sys_prompt = SystemMessage(content=f"""
You are an AI business analyst assistant for e-commerce sellers. Answer the
user's question about your identity, available data, tools, or how to get started.

Available Datasets:
{dataset_desc}

Sample Records:
{dataset_sample}

System Knowledge & FAQs:
{general_knowledge}

---
Guidelines:
- Be conversational and helpful, not robotic
- Never mention internal dataset/table names — say "your order data"
- Give 2-3 concrete example questions the user could ask
- End with a question that invites them to continue exploring
- Keep response to 4-8 sentences
""")

    messages = [sys_prompt] + chat_history + [HumanMessage(content=state.user_questions)]
    response = response_llm.invoke(messages)

    return {"response": response.content}


# ── Explanation query handler ──────────────────────────────────────────────────
def handle_explanation_query(state: GraphState) -> dict:
    """
    Explains or clarifies a result already shown in chat history.
    No tool calls — works purely from existing conversation context.
    """
    chat_history = format_chat_history_for_llm(state.chat_history)
    explanation_knowledge = get_knowledge(config.knowledge_paths.EXPLANATION_KNOWLEDGE_DIR)

    if len(chat_history) < 2:
        return {
            "response": (
                "I don't see any previous results to explain yet. "
                "Try asking me to analyze something first — for example, "
                "'What's my churn rate?' or 'Show me revenue by category.'"
            )
        }

    sys_prompt = SystemMessage(content=f"""
You are explaining or clarifying a result that already exists in the
conversation history below. Use ONLY data that appears in that history —
never invent or recalculate numbers.

Explanation Knowledge:
{explanation_knowledge}

---
Guidelines:
- Quote exact numbers from the chat history, never approximate
- Distinguish facts (from data) vs interpretation (your analysis)
- If the reference is ambiguous, state what you think they mean and confirm
- End by offering to explore further
- Keep response to 4-8 sentences
""")

    messages = [sys_prompt] + chat_history + [HumanMessage(content=state.user_questions)]
    response = response_llm.invoke(messages)

    return {"response": response.content}


# ── Segment query handler ──────────────────────────────────────────────────────
def handle_segment_query(state: GraphState) -> dict:
    """
    Answers questions about segment availability, definitions, or
    (if segment_suggestion=True) data-driven value recommendations.
    """
    chat_history = format_chat_history_for_llm(state.chat_history)
    segment_knowledge = get_knowledge(config.knowledge_paths.SEGMENTS_KNOWLEDGE_DIR)

    segments_catalog = build_segment_catalogue(
        state.segment_registry,
        state.candidate_dataset_names,
        lite=not state.segment_suggestion,
    )

    sys_prompt = SystemMessage(content=f"""
You are explaining segments — dimensions for filtering and grouping order
data (customer types, behaviors, value tiers).

Available Segments:
{segments_catalog}

Segment Knowledge:
{segment_knowledge}

---
Guidelines:
- Explain what each relevant segment means in plain language
- Give 2-3 concrete example questions using these segments
- Never mention internal dataset names — say "your order data"
- Never claim you can automatically apply segment values — only suggest
- End with a question to guide the user forward
- Keep response to 5-10 sentences
""")

    messages = [sys_prompt] + chat_history + [HumanMessage(content=state.user_questions)]
    response = response_llm.invoke(messages)

    return {"response": response.content}


# ── Unsupported query handler ───────────────────────────────────────────────────
def handle_unsupported_query(state: GraphState) -> dict:
    """
    Politely declines out-of-scope queries and redirects to valid capabilities.
    """
    dataset_names = list(build_dataset_schemas().keys())

    sys_prompt = SystemMessage(content=f"""
The user asked something outside your scope. You are an e-commerce business
analyst assistant — you can ONLY help with:
- Analyzing order data: {dataset_names}
- Calculating metrics: revenue, churn, retention, AOV, repeat purchase rate
- Creating reports and exploring customer segments

You CANNOT help with: creative writing, general knowledge, personal advice,
or anything unrelated to this seller's order data.

---
Guidelines:
- Acknowledge what they asked (show you understood)
- Politely decline with a brief, specific reason
- Suggest 1-2 relevant alternatives from your actual capabilities
- End with a question to redirect them
- Tone: friendly, not apologetic. 3-5 sentences.
""")

    messages = [sys_prompt, HumanMessage(content=state.user_questions)]
    response = response_llm.invoke(messages)

    print(f"[UNSUPPORTED QUERY]: {state.user_questions[:80]}")

    return {"response": response.content}