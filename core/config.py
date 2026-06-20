import os

# ── Active plan (mutable by API layer per request) ──────────────────────────
USER_PLAN: str = "standard_plan"

# ── Base directories (derived from USER_PLAN) ───────────────────────────────
TOOLS_DIR:    str = f"{USER_PLAN}/tools"
SEGMENTS_DIR: str = f"{USER_PLAN}/segments"
SCHEMA_DIR:   str = f"{USER_PLAN}/schemas"
KNOWLEDGE_DIR: str = f"{USER_PLAN}/knowledge"

# ── File name constants ──────────────────────────────────────────────────────
TOOLS_SPEC_FILENAME:    str = "spec.json"
TOOLS_RUN_FILENAME:     str = "run.py"
SEGMENTS_SPEC_FILENAME: str = "spec.json"
SEGMENTS_RUN_FILENAME:  str = "run.py"

# ── Knowledge subdirectory paths ─────────────────────────────────────────────
# Uses @property so every access reads the current KNOWLEDGE_DIR value.
# This means when KNOWLEDGE_DIR changes (plan switch), all subpaths
# automatically reflect the new value without any extra work.
class KnowledgePaths:
    @property
    def INTENT_MAPPING_DIR(self):
        return f"{KNOWLEDGE_DIR}/intent-mapping/"

    @property
    def SEGMENTS_KNOWLEDGE_DIR(self):
        return f"{KNOWLEDGE_DIR}/segments-knowledge/"

    @property
    def PRODUCT_FAQS_AND_GUIDES_DIR(self):
        return f"{KNOWLEDGE_DIR}/product-faqs-and-guides/"

    @property
    def EXPLANATION_KNOWLEDGE_DIR(self):
        return f"{KNOWLEDGE_DIR}/explanations-and-methods/"

    @property
    def TOOL_PLANNER_RULES_DIR(self):
        return f"{KNOWLEDGE_DIR}/tool-planner-rules/"

    @property
    def TOOL_EXECUTOR_PRESETS_DIR(self):
        return f"{KNOWLEDGE_DIR}/tool-executor-presets/"

    @property
    def RESPONSE_TEMPLATES_AND_REPORTS_DIR(self):
        return f"{KNOWLEDGE_DIR}/response-templates-and-reports/"

    @property
    def RESPONSE_TEMPLATES_TEMPLATES_DIR(self):
        return f"{KNOWLEDGE_DIR}/response-templates-and-reports/templates/"

    @property
    def SHORT_RESPONSE(self):
        return f"{KNOWLEDGE_DIR}/response-templates-and-reports/short-answer/"


knowledge_paths = KnowledgePaths()