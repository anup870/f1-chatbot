"""Per-session conversation memory and follow-up question rewriting.

Turns "How many titles did he win?" (after asking about Hamilton) into
"How many titles did Lewis Hamilton win?", and "what about Alonso?" into the
previous question with the driver swapped. The rewritten question is what the
router and retriever see.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from backend.stats_db import EntityIndex, fold
from backend.router import YEAR_RE

MAX_TURNS = 8
MAX_SESSIONS = 500
SESSION_TTL_S = 6 * 3600

FOLLOW_UP_RE = re.compile(r"^\s*(?:and|what about|how about|same for|and what about|ok and|now)\s+(?P<rest>.+?)\s*\??\s*$", re.IGNORECASE)
HE_RE = re.compile(r"\b(he|him)\b", re.IGNORECASE)
HIS_RE = re.compile(r"\bhis\b", re.IGNORECASE)
THEY_RE = re.compile(r"\b(they|them)\b", re.IGNORECASE)
THEIR_RE = re.compile(r"\btheir\b", re.IGNORECASE)
THAT_YEAR_RE = re.compile(r"\b(that|the same) (year|season)\b|\bthen\b", re.IGNORECASE)


@dataclass
class Turn:
    question: str
    resolved: str
    answer: str
    tool: str


@dataclass
class SessionState:
    turns: list[Turn] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    teams: list[str] = field(default_factory=list)
    season: Optional[int] = None
    touched: float = field(default_factory=time.time)

    def history_text(self, limit: int = 3) -> str:
        lines = []
        for turn in self.turns[-limit:]:
            lines.append(f"User: {turn.resolved}\nAssistant: {turn.answer[:300]}")
        return "\n".join(lines)


class ConversationMemory:
    def __init__(self) -> None:
        self._sessions: "OrderedDict[str, SessionState]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            now = time.time()
            for sid in [s for s, st in self._sessions.items() if now - st.touched > SESSION_TTL_S]:
                del self._sessions[sid]
            state = self._sessions.pop(session_id, None) or SessionState()
            state.touched = now
            self._sessions[session_id] = state
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
            return state

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------ rewrite

    def rewrite(self, state: SessionState, question: str, index: Optional[EntityIndex]) -> str:
        if not state.turns:
            return question
        last = state.turns[-1].resolved
        drivers = index.find_drivers(question) if index else []
        teams = index.find_constructors(question) if index else []
        years = YEAR_RE.findall(question)

        # "what about Alonso?" / "and in 2010?" -> previous question with the entity swapped.
        match = FOLLOW_UP_RE.match(question)
        if match and len(match.group("rest").split()) <= 6:
            new = self._swap_entities(last, match.group("rest"), index)
            if new:
                return new

        out = question
        if not drivers and state.drivers and index:
            name = index.driver_name(state.drivers[0])
            if HE_RE.search(out) or HIS_RE.search(out):
                out = HIS_RE.sub(f"{name}'s", out)
                out = HE_RE.sub(name, out)
            elif len(state.drivers) >= 2 and THEY_RE.search(out):
                pair = f"{index.driver_name(state.drivers[0])} and {index.driver_name(state.drivers[1])}"
                out = THEY_RE.sub(pair, out)
        if not teams and not drivers and state.teams and index and (THEY_RE.search(out) or THEIR_RE.search(out)):
            team = index.constructor_name(state.teams[0])
            out = THEIR_RE.sub(f"{team}'s", out)
            out = THEY_RE.sub(team, out)
        if not years and state.season and THAT_YEAR_RE.search(out):
            out = THAT_YEAR_RE.sub(f"in {state.season}", out)
            out = re.sub(r"\bin in\b", "in", out)
        return out

    @staticmethod
    def _swap_entities(last: str, rest: str, index: Optional[EntityIndex]) -> Optional[str]:
        new = fold(last)
        changed = False
        new_years = YEAR_RE.findall(rest)
        if new_years:
            old_years = YEAR_RE.findall(new)
            if old_years:
                new = new.replace(old_years[0], new_years[0], 1)
            else:
                new = new.rstrip(" ?") + f" in {new_years[0]}"
            changed = True
        if index:
            new_drivers = index.find_drivers(rest)
            if new_drivers:
                spans = index.driver_spans(new)
                if spans:
                    start, end, _ = spans[0]
                    new = new[:start] + fold(index.driver_name(new_drivers[0])) + new[end:]
                    changed = True
            new_teams = index.find_constructors(rest) if not new_drivers else []
            if new_teams:
                spans = index.constructor_spans(new)
                if spans:
                    start, end, _ = spans[0]
                    new = new[:start] + fold(index.constructor_name(new_teams[0])) + new[end:]
                    changed = True
        return new if changed else None

    # ------------------------------------------------------------------ update

    def record(self, state: SessionState, question: str, resolved: str, answer: str, tool: str,
               index: Optional[EntityIndex]) -> None:
        state.turns.append(Turn(question, resolved, answer, tool))
        del state.turns[:-MAX_TURNS]
        if index:
            drivers = index.find_drivers(resolved)
            teams = index.find_constructors(resolved)
            if drivers:
                state.drivers = drivers
            elif tool == "champion":
                # "Who won 2009?" -> "how many races did he win" should mean the champion.
                answer_drivers = index.find_drivers(answer)
                if answer_drivers:
                    state.drivers = answer_drivers[:1]
            if teams:
                state.teams = teams
        years = YEAR_RE.findall(resolved)
        if years:
            state.season = int(years[0])
