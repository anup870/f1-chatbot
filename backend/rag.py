"""Backward-compatible entry points. The real logic lives in backend/agent.py."""

from __future__ import annotations

from typing import Optional

from backend.agent import AgentResult, F1Agent

_SERVICE: Optional[F1Agent] = None


def get_service() -> F1Agent:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = F1Agent()
    return _SERVICE


def reload_service() -> F1Agent:
    global _SERVICE
    _SERVICE = F1Agent()
    return _SERVICE


def ask(question: str, session_id: Optional[str] = None, persona: str = "default") -> AgentResult:
    return get_service().ask(question, session_id=session_id, persona=persona)
