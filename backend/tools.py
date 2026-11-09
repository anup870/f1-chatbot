"""Deterministic F1 stats tools over the SQLite store.

Every tool returns a ToolResult whose text is computed directly from the
database, so numbers in stats answers are never invented by the language model.
Tools may also attach chart data (rendered by the frontend with Chart.js).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from backend.stats_db import EntityIndex, fold, get_meta

PODIUM = 3

# Race-finish points tables used by the what-if tool (index 0 = P1).
POINT_SYSTEMS: dict[str, dict[str, Any]] = {
    "current": {"label": "2010+ system (25-18-15-12-10-8-6-4-2-1), no fastest-lap bonus", "table": [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]},
    "2003": {"label": "2003-2009 system (10-8-6-5-4-3-2-1)", "table": [10, 8, 6, 5, 4, 3, 2, 1]},
    "1991": {"label": "1991-2002 system (10-6-4-3-2-1)", "table": [10, 6, 4, 3, 2, 1]},
    "1961": {"label": "1961-1990 system (9-6-4-3-2-1)", "table": [9, 6, 4, 3, 2, 1]},
    "wins_only": {"label": "medals system (most wins, then most 2nds, then 3rds...)", "table": None},
}


@dataclass
class ToolResult:
    text: str
    tool: str
    data: dict[str, Any] = field(default_factory=dict)
    chart: Optional[dict[str, Any]] = None
    sources: list[dict[str, str]] = field(default_factory=list)
    ok: bool = True


def _pts(value: float) -> str:
    return f"{value:g}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


class StatsTools:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.index = EntityIndex(conn)
        self.source = {
            "id": "stats",
            "file": "Jolpica F1 API (local stats DB)",
            "snippet": f"Last synced {get_meta(conn, 'last_sync') or 'unknown'}",
        }

    # ------------------------------------------------------------------ helpers

    def _q(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def latest_season(self) -> int:
        row = self._one("SELECT MAX(season) AS s FROM results")
        return int(row["s"]) if row and row["s"] else 0

    def season_complete(self, season: int) -> bool:
        latest = self.latest_season()
        if season < latest:
            return True
        total = get_meta(self.conn, f"rounds_{season}")
        if not total:
            return False
        done = self._one("SELECT MAX(round) AS r FROM results WHERE season=?", (season,))
        return bool(done and done["r"] and int(done["r"]) >= int(total))

    def _result(self, text: str, tool: str, **kwargs: Any) -> ToolResult:
        return ToolResult(text=text, tool=tool, sources=[self.source], **kwargs)

    def _fail(self, text: str, tool: str) -> ToolResult:
        return ToolResult(text=text, tool=tool, ok=False, sources=[self.source])

    def _poles_sql(self, extra_where: str = "") -> str:
        # Qualifying P1 where qualifying data exists; otherwise grid P1.
        return f"""
            SELECT season, round, driver_id, constructor_id FROM qualifying WHERE position=1 {extra_where}
            UNION
            SELECT r.season, r.round, r.driver_id, r.constructor_id FROM results r
            WHERE r.grid=1 {extra_where.replace('season', 'r.season')}
              AND NOT EXISTS (SELECT 1 FROM qualifying q WHERE q.season=r.season AND q.round=r.round)
        """

    def _titles(self, driver_id: str) -> list[int]:
        rows = self._q(
            "SELECT season FROM driver_standings WHERE driver_id=? AND position=1 ORDER BY season", (driver_id,)
        )
        return [r["season"] for r in rows if self.season_complete(r["season"])]

    def _constructor_titles(self, constructor_id: str) -> list[int]:
        rows = self._q(
            "SELECT season FROM constructor_standings WHERE constructor_id=? AND position=1 ORDER BY season",
            (constructor_id,),
        )
        return [r["season"] for r in rows if self.season_complete(r["season"])]

    def _driver_counts(self, driver_id: str, season: Optional[int] = None) -> dict[str, Any]:
        where, params = "driver_id=?", [driver_id]
        if season:
            where += " AND season=?"
            params.append(season)
        row = self._one(
            f"""SELECT COUNT(*) AS starts, SUM(position=1) AS wins, SUM(position<={PODIUM}) AS podiums,
                       SUM(points) AS points, MIN(season) AS first, MAX(season) AS last,
                       MIN(position) AS best
                FROM results WHERE {where}""",
            tuple(params),
        )
        sprint = self._one(f"SELECT COALESCE(SUM(points),0) AS p FROM sprint_results WHERE {where}", tuple(params))
        poles_where = "AND season=?" if season else ""
        pole_params = (season, season) if season else ()
        poles = self._one(
            f"SELECT COUNT(*) AS n FROM ({self._poles_sql(poles_where)}) WHERE driver_id=?",
            (*pole_params, driver_id),
        )
        return {
            "starts": row["starts"] or 0,
            "wins": row["wins"] or 0,
            "podiums": row["podiums"] or 0,
            "points": (row["points"] or 0) + (sprint["p"] or 0),
            "poles": poles["n"] or 0,
            "first": row["first"],
            "last": row["last"],
            "best": row["best"],
        }

    def _cumulative_points(self, driver_id: str, season: int) -> tuple[list[str], list[float]]:
        races = self._q("SELECT round, name FROM races WHERE season=? ORDER BY round", (season,))
        totals: list[float] = []
        labels: list[str] = []
        running = 0.0
        for race in races:
            r = self._one("SELECT points FROM results WHERE season=? AND round=? AND driver_id=?", (season, race["round"], driver_id))
            s = self._one("SELECT points FROM sprint_results WHERE season=? AND round=? AND driver_id=?", (season, race["round"], driver_id))
            if r is None and s is None and not self._one("SELECT 1 FROM results WHERE season=? AND round=?", (season, race["round"])):
                continue  # race not run yet
            running += (r["points"] if r else 0) + (s["points"] if s else 0)
            labels.append(race["name"].replace(" Grand Prix", " GP"))
            totals.append(running)
        return labels, totals

    def resolve_race(self, season: int, text: str) -> Optional[sqlite3.Row]:
        folded = fold(text)
        races = self._q("SELECT * FROM races WHERE season=? ORDER BY round", (season,))
        best, best_score = None, 0
        for race in races:
            # A race-name match ("Sakhir Grand Prix") beats a location match
            # (the Bahrain GP is also held at Sakhir).
            for field_name, weight in (("name", 1000), ("circuit_name", 0), ("locality", 0), ("country", 0)):
                value = fold(race[field_name] or "")
                if not value:
                    continue
                key = value.replace(" grand prix", "").strip()
                if key and re.search(rf"\b{re.escape(key)}\b", folded) and len(key) + weight > best_score:
                    best, best_score = race, len(key) + weight
        return best

    # -------------------------------------------------------------------- tools

    def driver_career(self, driver: str) -> ToolResult:
        name = self.index.driver_name(driver)
        c = self._driver_counts(driver)
        if not c["starts"]:
            return self._fail(f"I have no race results for {name}.", "driver_career")
        titles = self._titles(driver)
        teams = [
            self.index.constructor_name(r["constructor_id"])
            for r in self._q(
                "SELECT constructor_id, MIN(season) s FROM results WHERE driver_id=? GROUP BY constructor_id ORDER BY s",
                (driver,),
            )
        ]
        title_txt = f"{_plural(len(titles), 'world title')} ({', '.join(map(str, titles))})" if titles else "no world titles"
        text = (
            f"{name} ({c['first']}–{c['last']}): {_plural(c['starts'], 'start')}, {_plural(c['wins'], 'win')}, "
            f"{_plural(c['podiums'], 'podium')}, {_plural(c['poles'], 'pole')}, {_pts(c['points'])} career points, {title_txt}. "
            f"Teams: {', '.join(teams)}."
        )
        chart = {
            "type": "bar",
            "title": f"{name} — career",
            "labels": ["Wins", "Podiums", "Poles", "Titles"],
            "datasets": [{"label": name, "data": [c["wins"], c["podiums"], c["poles"], len(titles)]}],
        }
        return self._result(text, "driver_career", data={**c, "titles": titles, "teams": teams}, chart=chart)

    def driver_season(self, driver: str, season: int) -> ToolResult:
        name = self.index.driver_name(driver)
        c = self._driver_counts(driver, season)
        if not c["starts"]:
            return self._fail(f"{name} has no race results in {season}.", "driver_season")
        standing = self._one("SELECT position, points FROM driver_standings WHERE season=? AND driver_id=?", (season, driver))
        teams = [
            self.index.constructor_name(r["constructor_id"])
            for r in self._q("SELECT DISTINCT constructor_id FROM results WHERE season=? AND driver_id=?", (season, driver))
        ]
        verb = "finished" if self.season_complete(season) else "is currently"
        pos_txt = f"{verb} P{standing['position']} in the standings with {_pts(standing['points'])} points" if standing else f"scored {_pts(c['points'])} points"
        text = (
            f"In {season}, {name} ({', '.join(teams)}) {pos_txt}: "
            f"{_plural(c['wins'], 'win')}, {_plural(c['podiums'], 'podium')} and {_plural(c['poles'], 'pole')} from {_plural(c['starts'], 'start')}."
        )
        labels, totals = self._cumulative_points(driver, season)
        chart = {
            "type": "line",
            "title": f"{name} — {season} cumulative points",
            "labels": labels,
            "datasets": [{"label": name, "data": totals}],
        }
        return self._result(text, "driver_season", data={**c, "season": season}, chart=chart)

    def season_standings(self, season: int, kind: str = "drivers", top: int = 10) -> ToolResult:
        status = "Final" if self.season_complete(season) else "Current"
        if kind == "constructors":
            rows = self._q(
                "SELECT position, constructor_id, points, wins FROM constructor_standings WHERE season=? ORDER BY position LIMIT ?",
                (season, top),
            )
            if not rows:
                return self._fail(f"No constructors' standings for {season} (the title started in 1958).", "season_standings")
            names = [self.index.constructor_name(r["constructor_id"]) for r in rows]
        else:
            rows = self._q(
                "SELECT position, driver_id, points, wins FROM driver_standings WHERE season=? ORDER BY position LIMIT ?",
                (season, top),
            )
            if not rows:
                return self._fail(f"No drivers' standings for {season}.", "season_standings")
            names = [self.index.driver_name(r["driver_id"]) for r in rows]
        lines = [f"P{r['position']} {n} — {_pts(r['points'])} pts ({_plural(r['wins'], 'win')})" for r, n in zip(rows, names)]
        text = f"{status} {season} {kind} standings:\n" + "\n".join(lines)
        chart = {
            "type": "bar",
            "title": f"{season} {kind} standings",
            "labels": names,
            "datasets": [{"label": "Points", "data": [r["points"] for r in rows]}],
        }
        return self._result(text, "season_standings", chart=chart)

    def champion(self, season: int) -> ToolResult:
        d = self._one("SELECT driver_id, points, wins FROM driver_standings WHERE season=? AND position=1", (season,))
        if not d:
            return self._fail(f"I don't have standings for {season}.", "champion")
        c = self._one("SELECT constructor_id, points FROM constructor_standings WHERE season=? AND position=1", (season,))
        done = self.season_complete(season)
        driver = self.index.driver_name(d["driver_id"])
        if done:
            text = f"The {season} Drivers' Champion was {driver} ({_pts(d['points'])} pts, {_plural(d['wins'], 'win')})."
            if c:
                text += f" The Constructors' Champion was {self.index.constructor_name(c['constructor_id'])}."
            elif season < 1958:
                text += " There was no Constructors' Championship until 1958."
        else:
            text = f"The {season} season is still running. {driver} currently leads the Drivers' Championship on {_pts(d['points'])} points."
            if c:
                text += f" {self.index.constructor_name(c['constructor_id'])} lead the Constructors'."
        return self._result(text, "champion", data={"driver_id": d["driver_id"], "complete": done})

    def race_result(self, season: int, race_text: str = "", round_no: Optional[int] = None) -> ToolResult:
        if round_no:
            race = self._one("SELECT * FROM races WHERE season=? AND round=?", (season, round_no))
        elif race_text in ("last", "latest"):
            race = self._one(
                "SELECT ra.* FROM races ra JOIN results r ON r.season=ra.season AND r.round=ra.round "
                "WHERE ra.season=? ORDER BY ra.round DESC LIMIT 1",
                (season,),
            )
        else:
            race = self.resolve_race(season, race_text)
        if not race:
            return self._fail(f"I couldn't find that race in the {season} calendar.", "race_result")
        rows = self._q(
            "SELECT * FROM results WHERE season=? AND round=? ORDER BY position IS NULL, position LIMIT 10",
            (season, race["round"]),
        )
        if not rows:
            return self._fail(f"The {season} {race['name']} hasn't been run yet (or I have no results for it).", "race_result")
        pole = self._one(f"SELECT driver_id FROM ({self._poles_sql('AND season=? AND round=?')}) LIMIT 1",
                         (season, race["round"], season, race["round"]))
        fl = self._one("SELECT driver_id FROM results WHERE season=? AND round=? AND fastest_lap_rank=1", (season, race["round"]))
        podium = [f"P{r['position']} {self.index.driver_name(r['driver_id'])} ({self.index.constructor_name(r['constructor_id'])})"
                  for r in rows if r["position"] and r["position"] <= 3]
        rest = [f"P{r['position']} {self.index.driver_name(r['driver_id'])}" for r in rows if r["position"] and r["position"] > 3]
        text = f"{season} {race['name']} ({race['circuit_name']}, {race['date']}): " + ", ".join(podium) + "."
        if rest:
            text += " Rest of the top 10: " + ", ".join(rest) + "."
        if pole:
            text += f" Pole: {self.index.driver_name(pole['driver_id'])}."
        if fl:
            text += f" Fastest lap: {self.index.driver_name(fl['driver_id'])}."
        return self._result(text, "race_result", data={"season": season, "round": race["round"]})

    def compare(self, a: str, b: str, season: Optional[int] = None) -> ToolResult:
        na, nb = self.index.driver_name(a), self.index.driver_name(b)
        ca, cb = self._driver_counts(a, season), self._driver_counts(b, season)
        scope = f"in {season}" if season else "career"
        if not ca["starts"] or not cb["starts"]:
            missing = na if not ca["starts"] else nb
            return self._fail(f"{missing} has no results {scope if season else ''}.".replace("  ", " "), "compare")
        # Head-to-head in races both were classified in.
        where = "AND x.season=?" if season else ""
        params = (a, b, season) if season else (a, b)
        h2h = self._one(
            f"""SELECT SUM(x.position < y.position) AS a_ahead, SUM(y.position < x.position) AS b_ahead
                FROM results x JOIN results y ON x.season=y.season AND x.round=y.round
                WHERE x.driver_id=? AND y.driver_id=? AND x.position IS NOT NULL AND y.position IS NOT NULL {where}""",
            params,
        )
        parts = [
            f"{na} vs {nb} ({scope}): wins {ca['wins']}–{cb['wins']}, podiums {ca['podiums']}–{cb['podiums']}, "
            f"poles {ca['poles']}–{cb['poles']}, points {_pts(ca['points'])}–{_pts(cb['points'])}, starts {ca['starts']}–{cb['starts']}."
        ]
        if not season:
            ta, tb = self._titles(a), self._titles(b)
            parts.append(f"Titles {len(ta)}–{len(tb)}.")
        if h2h and (h2h["a_ahead"] or h2h["b_ahead"]):
            parts.append(f"When both finished the same race, {na} was ahead {_plural(h2h['a_ahead'], 'time')}, {nb} {_plural(h2h['b_ahead'], 'time')}.")
        if season:
            labels, ta_pts = self._cumulative_points(a, season)
            _, tb_pts = self._cumulative_points(b, season)
            chart = {
                "type": "line",
                "title": f"{na} vs {nb} — {season} cumulative points",
                "labels": labels,
                "datasets": [{"label": na, "data": ta_pts}, {"label": nb, "data": tb_pts}],
            }
        else:
            chart = {
                "type": "bar",
                "title": f"{na} vs {nb} — career",
                "labels": ["Wins", "Podiums", "Poles", "Titles"],
                "datasets": [
                    {"label": na, "data": [ca["wins"], ca["podiums"], ca["poles"], len(self._titles(a))]},
                    {"label": nb, "data": [cb["wins"], cb["podiums"], cb["poles"], len(self._titles(b))]},
                ],
            }
        return self._result(" ".join(parts), "compare", data={"a": ca, "b": cb}, chart=chart)

    def leaderboard(self, stat: str, by: str = "driver", season: Optional[int] = None, top: int = 10) -> ToolResult:
        key = "driver_id" if by == "driver" else "constructor_id"
        name_of: Callable[[str], str] = self.index.driver_name if by == "driver" else self.index.constructor_name
        season_where = "WHERE season=?" if season else ""
        params: tuple = (season,) if season else ()
        if stat == "titles":
            if season:
                return self.champion(season)
            table = "driver_standings" if by == "driver" else "constructor_standings"
            rows = self._q(f"SELECT {key} AS k, season FROM {table} WHERE position=1")
            counts: dict[str, int] = {}
            for r in rows:
                if self.season_complete(r["season"]):
                    counts[r["k"]] = counts.get(r["k"], 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
        else:
            expr = {
                "wins": "SUM(position=1)",
                "podiums": f"SUM(position<={PODIUM})",
                "starts": "COUNT(*)",
                "points": "SUM(points)",
            }.get(stat)
            if stat == "poles":
                pole_where = "AND season=?" if season else ""
                pole_params = (season, season) if season else ()
                rows = self._q(
                    f"SELECT {key} AS k, COUNT(*) AS v FROM ({self._poles_sql(pole_where)}) GROUP BY k ORDER BY v DESC LIMIT ?",
                    (*pole_params, top),
                )
            elif expr:
                rows = self._q(
                    f"SELECT {key} AS k, {expr} AS v FROM results {season_where} GROUP BY k ORDER BY v DESC LIMIT ?",
                    (*params, top),
                )
            else:
                return self._fail(f"I can't rank by '{stat}'.", "leaderboard")
            ranked = [(r["k"], r["v"]) for r in rows if r["v"]]
        if not ranked:
            return self._fail("No data for that leaderboard.", "leaderboard")
        scope = f"in {season}" if season else "of all time"
        who = "drivers" if by == "driver" else "constructors"
        note = " (raw race points; historical points systems differ, so eras aren't directly comparable)" if stat == "points" else ""
        lines = [f"{i}. {name_of(k)} — {_pts(v) if isinstance(v, float) else v}" for i, (k, v) in enumerate(ranked, 1)]
        text = f"Most {stat} by {who} {scope}{note}:\n" + "\n".join(lines)
        chart = {
            "type": "bar",
            "title": f"Most {stat} {scope}",
            "labels": [name_of(k) for k, _ in ranked],
            "datasets": [{"label": stat.capitalize(), "data": [v for _, v in ranked]}],
        }
        return self._result(text, "leaderboard", chart=chart)

    def constructor_summary(self, constructor: str, season: Optional[int] = None) -> ToolResult:
        name = self.index.constructor_name(constructor)
        where, params = "constructor_id=?", [constructor]
        if season:
            where += " AND season=?"
            params.append(season)
        row = self._one(
            f"SELECT COUNT(DISTINCT season||'-'||round) AS races, SUM(position=1) AS wins, SUM(position<={PODIUM}) AS podiums, "
            f"MIN(season) AS first, MAX(season) AS last FROM results WHERE {where}",
            tuple(params),
        )
        if not row or not row["races"]:
            return self._fail(f"No results for {name}{f' in {season}' if season else ''}.", "constructor_summary")
        if season:
            st = self._one("SELECT position, points FROM constructor_standings WHERE season=? AND constructor_id=?", (season, constructor))
            standing = f", P{st['position']} in the Constructors' with {_pts(st['points'])} pts" if st else ""
            text = f"{name} in {season}: {_plural(row['wins'], 'win')} and {_plural(row['podiums'], 'podium')} from {_plural(row['races'], 'race')}{standing}."
        else:
            titles = self._constructor_titles(constructor)
            title_txt = f"{_plural(len(titles), "Constructors' title")} ({', '.join(map(str, titles))})" if titles else "no Constructors' titles"
            text = (f"{name} ({row['first']}–{row['last']}): {_plural(row['races'], 'race')} entered, {_plural(row['wins'], 'win')}, "
                    f"{_plural(row['podiums'], 'podium')}, {title_txt}.")
        return self._result(text, "constructor_summary", data=dict(row))

    def whatif(self, season: int, system: str) -> ToolResult:
        if system not in POINT_SYSTEMS:
            return self._fail(f"Unknown points system '{system}'. Options: {', '.join(POINT_SYSTEMS)}.", "whatif")
        rows = self._q("SELECT driver_id, position FROM results WHERE season=? AND position IS NOT NULL", (season,))
        if not rows:
            return self._fail(f"No results for {season}.", "whatif")
        actual = {r["driver_id"]: r["position"] for r in self._q("SELECT driver_id, position FROM driver_standings WHERE season=?", (season,))}
        table = POINT_SYSTEMS[system]["table"]
        if table is None:
            finishes: dict[str, list[int]] = {}
            for r in rows:
                finishes.setdefault(r["driver_id"], [0] * 25)
                if r["position"] <= 25:
                    finishes[r["driver_id"]][r["position"] - 1] += 1
            ranked = sorted(finishes.items(), key=lambda kv: kv[1], reverse=True)
            scored = [(d, f[0]) for d, f in ranked]
            unit = "wins"
        else:
            points: dict[str, float] = {}
            for r in rows:
                pos = r["position"]
                points[r["driver_id"]] = points.get(r["driver_id"], 0) + (table[pos - 1] if pos <= len(table) else 0)
            scored = sorted(points.items(), key=lambda kv: -kv[1])
            unit = "pts"
        top = scored[:10]
        lines = []
        for i, (d, v) in enumerate(top, 1):
            was = actual.get(d)
            move = "" if was is None or was == i else f" (actually P{was})"
            amount = _plural(int(v), "win") if unit == "wins" else f"{_pts(v)} pts"
            lines.append(f"P{i} {self.index.driver_name(d)} — {amount}{move}")
        new_champ, real_champ = top[0][0], next((d for d, p in actual.items() if p == 1), None)
        if real_champ and new_champ != real_champ:
            headline = f"Under the {POINT_SYSTEMS[system]['label']}, {self.index.driver_name(new_champ)} would have won {season} instead of {self.index.driver_name(real_champ)}."
        else:
            headline = f"Under the {POINT_SYSTEMS[system]['label']}, {self.index.driver_name(new_champ)} still wins {season}."
        text = (headline + "\n" + "\n".join(lines) +
                "\n(Main races only; sprints, shared drives, half-points races and dropped-score rules are ignored.)")
        chart = {
            "type": "bar",
            "title": f"{season} re-scored: {system}",
            "labels": [self.index.driver_name(d) for d, _ in top],
            "datasets": [{"label": unit, "data": [v for _, v in top]}],
        }
        return self._result(text, "whatif", chart=chart)


# JSON-schema-ish description of the tools, used for the LLM router prompt.
TOOL_SPECS = [
    {"name": "driver_career", "args": {"driver": "driver name"}, "use": "a driver's career totals"},
    {"name": "driver_season", "args": {"driver": "driver name", "season": "year"}, "use": "a driver's stats in one season"},
    {"name": "season_standings", "args": {"season": "year", "kind": "drivers|constructors"}, "use": "championship table"},
    {"name": "champion", "args": {"season": "year"}, "use": "who won a championship"},
    {"name": "race_result", "args": {"season": "year", "race": "race or country name, or 'last'"}, "use": "result of one Grand Prix"},
    {"name": "compare", "args": {"driver_a": "name", "driver_b": "name", "season": "year or null"}, "use": "compare two drivers"},
    {"name": "leaderboard", "args": {"stat": "wins|poles|podiums|titles|starts|points", "by": "driver|constructor", "season": "year or null"}, "use": "all-time or season rankings"},
    {"name": "constructor_summary", "args": {"constructor": "team name", "season": "year or null"}, "use": "a team's record"},
    {"name": "whatif", "args": {"season": "year", "system": "current|2003|1991|1961|wins_only"}, "use": "re-score a season with another points system"},
    {"name": "live", "args": {}, "use": "what is happening in the latest/live session"},
    {"name": "rag", "args": {}, "use": "rules, regulations, flags, tyres, history, explanations"},
]
