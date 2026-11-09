"""Decide which tool answers a question.

Deterministic rules run first: they're instant, predictable and easy to test.
When no rule matches but the question still looks like a stats question, the
local LLM is asked to pick a tool and fill its arguments as JSON (see
F1Agent._llm_route). Everything else goes to text retrieval (RAG).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.stats_db import EntityIndex, fold

YEAR_RE = re.compile(r"\b(19[5-9]\d|20\d{2})\b")

STAT_WORDS = re.compile(
    r"\b(wins?|won|victor(?:y|ies)|poles?|pole positions?|podiums?|points?|titles?|championships?|"
    r"career|stats?|statistics|record|how many|starts?|races?|finish(?:ed)?|standing|position|"
    r"season|results?|performance|numbers)\b"
)
TEAM_WORDS = re.compile(r"\b(team|teams|constructor|constructors|manufacturer)\b")
LIVE_RE = re.compile(
    r"\b(live|right now|at the moment|latest session|this session|current session|gaps?|interval|"
    r"race control|happening now|on track now|this weekend)\b"
)
WHATIF_RE = re.compile(
    r"\b(what if|what-if|would have (won|finished|been)|points system|scoring system|old points|"
    r"re-?score|medals? system|wins only|if .* points)\b"
)
COMPARE_RE = re.compile(r"\b(vs\.?|versus|compare|comparison|head[- ]to[- ]head|h2h|better than|or)\b")
LEADER_RE = re.compile(
    r"\b(most|all[- ]time|record for|greatest number|highest number|top \d+|leaderboard|rank(?:ing)?)\b"
)
STAT_NOUN = {
    "win": "wins", "wins": "wins", "won": "wins", "victories": "wins", "victory": "wins", "races won": "wins",
    "pole": "poles", "poles": "poles", "pole positions": "poles", "pole position": "poles",
    "podium": "podiums", "podiums": "podiums",
    "title": "titles", "titles": "titles", "championships": "titles", "world championships": "titles",
    "championship": "titles", "world titles": "titles",
    "starts": "starts", "races": "starts", "grand prix starts": "starts",
    "points": "points",
}
CHAMPION_RE = re.compile(r"\b(champion|championship|title)\b")
STANDINGS_RE = re.compile(r"\b(standings|table|leading the championship|leads the championship|championship order|points table)\b")
GP_RE = re.compile(r"\b(grand prix|gp)\b")
LAST_RACE_RE = re.compile(r"\b(last|latest|most recent|previous) (race|grand prix|gp)\b")
SYSTEM_YEAR_RE = re.compile(r"\b(19[5-9]\d|20\d{2})s?(?:[- ]era)? (?:points|scoring|system|rules)\b")


@dataclass
class Route:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = "rule"


def system_for_year(year: int) -> str:
    if year >= 2010:
        return "current"
    if year >= 2003:
        return "2003"
    if year >= 1991:
        return "1991"
    return "1961"


class Router:
    def __init__(self, index: Optional[EntityIndex], latest_season: int) -> None:
        self.index = index
        self.latest = latest_season

    def _season(self, q: str, years: list[int]) -> Optional[int]:
        if years:
            return years[0]
        if re.search(r"\b(this|current) (year|season)\b", q):
            return self.latest
        if re.search(r"\b(last|previous) (year|season)\b", q):
            return self.latest - 1
        return None

    def looks_like_stats(self, question: str) -> bool:
        q = fold(question)
        return bool(STAT_WORDS.search(q) or YEAR_RE.search(q) or (self.index and self.index.find_drivers(q)))

    def route(self, question: str) -> Optional[Route]:
        q = fold(question)
        years = [int(y) for y in YEAR_RE.findall(q)]
        drivers = self.index.find_drivers(q) if self.index else []
        teams = self.index.find_constructors(q) if self.index else []
        wants_team = bool(TEAM_WORDS.search(q)) or (bool(teams) and not drivers)

        # Live timing doesn't need the stats DB.
        if LIVE_RE.search(q) and not years and not STANDINGS_RE.search(q):
            return Route("live")

        if self.index is None:
            return None

        if WHATIF_RE.search(q):
            system = "current"
            sys_match = SYSTEM_YEAR_RE.search(q)
            season_years = list(years)
            if re.search(r"\b(medals?|wins only|most wins)\b", q):
                system = "wins_only"
            elif sys_match:
                sys_year = int(sys_match.group(1))
                system = system_for_year(sys_year)
                if sys_year in season_years and len(season_years) > 1:
                    season_years.remove(sys_year)
            season = self._season(q, season_years)
            if season:
                return Route("whatif", {"season": season, "system": system})

        season = self._season(q, years)

        if len(drivers) >= 2 and COMPARE_RE.search(q):
            return Route("compare", {"driver_a": drivers[0], "driver_b": drivers[1], "season": season})

        if LEADER_RE.search(q):
            stat = None
            for phrase in sorted(STAT_NOUN, key=len, reverse=True):
                if re.search(rf"\b{re.escape(phrase)}\b", q):
                    stat = STAT_NOUN[phrase]
                    break
            if stat and not (len(drivers) == 1 and re.search(r"\bdoes|did|has\b", q) and not re.search(r"\bwho\b", q)):
                return Route("leaderboard", {"stat": stat, "by": "constructor" if wants_team else "driver", "season": season})

        if STANDINGS_RE.search(q) and not drivers:
            kind = "constructors" if wants_team else "drivers"
            return Route("season_standings", {"season": season or self.latest, "kind": kind})

        if LAST_RACE_RE.search(q) and not drivers:
            return Route("race_result", {"season": season or self.latest, "race": "last"})

        if season and GP_RE.search(q) and not CHAMPION_RE.search(q):
            return Route("race_result", {"season": season, "race": question})

        if season and CHAMPION_RE.search(q) and not drivers and not (teams and not TEAM_WORDS.search(q)):
            return Route("champion", {"season": season})

        if len(drivers) == 1 and STAT_WORDS.search(q):
            if season:
                return Route("driver_season", {"driver": drivers[0], "season": season})
            return Route("driver_career", {"driver": drivers[0]})

        if teams and STAT_WORDS.search(q):
            return Route("constructor_summary", {"constructor": teams[0], "season": season})

        if season and re.search(r"\bwho won\b", q):
            return Route("champion", {"season": season})

        return None
