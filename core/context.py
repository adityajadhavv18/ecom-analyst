from contextvars import ContextVar

ACTIVE_PLAN: ContextVar[str] = ContextVar('active_plan', default='standard_plan')