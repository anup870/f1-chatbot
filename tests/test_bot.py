"""Offline tests: no network, no model file, no vector index needed.

Run: python -m pytest -q
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend import stats_db
from backend.agent import F1Agent
from backend.live import live_summary
from backend.tools import StatsTools
from tests.fixtures import build_fixture_db


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_fixture_db(tmp_path_factory.mktemp("db") / "f1.sqlite")


@pytest.fixture(scope="module")
def tools(db):
    return StatsTools(stats_db.connect(db, read_only=True))


@pytest.fixture()
def agent(db):
    return F1Agent(db_path=db, load_vectors=False)


# ------------------------------------------------------------------ entities

def test_entity_resolution_handles_accents_aliases_and_ambiguity(tools):
    idx = tools.index
    assert idx.find_drivers("Raikkonen") == ["raikkonen"]
    assert idx.find_drivers("compare max and lewis") == ["max_verstappen", "hamilton"]
    # Ambiguous surname -> most prominent driver; full name -> exact driver.
    assert idx.find_drivers("Schumacher")[0] in ("michael_schumacher", "ralf_schumacher")
    assert idx.find_drivers("Ralf Schumacher") == ["ralf_schumacher"]
    assert idx.find_constructors("how good was red bull") == ["red_bull"]


# --------------------------------------------------------------------- tools

def test_champion(tools):
    r = tools.champion(2021)
    assert "Max Verstappen" in r.text and "Mercedes" in r.text


def test_race_result(tools):
    r = tools.race_result(2008, "Brazilian Grand Prix")
    assert r.ok and "P1 Felipe Massa" in r.text and "Pole: Felipe Massa" in r.text


def test_compare_has_chart(tools):
    r = tools.compare("hamilton", "max_verstappen", 2021)
    assert r.chart and len(r.chart["datasets"]) == 2
    assert "poles 0–2" in r.text


def test_whatif_medals_flips_2008(tools):
    r = tools.whatif(2008, "wins_only")
    assert "Felipe Massa would have won 2008 instead of Lewis Hamilton" in r.text


def test_titles_skip_unfinished_season(db, tools):
    conn = stats_db.connect(db)
    stats_db.set_meta(conn, "rounds_2021", "23")  # pretend a round is left
    conn.commit()
    try:
        fresh = StatsTools(stats_db.connect(db, read_only=True))
        assert "still running" in fresh.champion(2021).text
        assert 2021 not in fresh.driver_career("max_verstappen").data["titles"]
    finally:
        stats_db.set_meta(conn, "rounds_2021", "22")
        conn.commit()


# -------------------------------------------------------------------- router

@pytest.mark.parametrize("question,tool", [
    ("Who won the 2021 championship?", "champion"),
    ("How many races did Verstappen win in 2021?", "driver_season"),
    ("How many titles does Hamilton have?", "driver_career"),
    ("Hamilton vs Verstappen 2021", "compare"),
    ("Who has the most pole positions?", "leaderboard"),
    ("Which team has the most wins?", "leaderboard"),
    ("Who won the 2021 Abu Dhabi GP?", "race_result"),
    ("Who won the last race?", "race_result"),
    ("2021 constructors standings", "season_standings"),
    ("What if 2021 used the 2003 points system?", "whatif"),
    ("What's happening live right now?", "live"),
    ("Ferrari stats", "constructor_summary"),
])
def test_router(agent, question, tool):
    route = agent.router.route(question)
    assert route is not None and route.tool == tool, route


def test_router_whatif_separates_season_and_system(agent):
    route = agent.router.route("What if 2021 used the 2003 points system?")
    assert route.args == {"season": 2021, "system": "2003"}


@pytest.mark.parametrize("question", ["What does a yellow flag mean?", "What is DRS?", "Explain the undercut"])
def test_router_sends_explanations_to_rag(agent, question):
    assert agent.router.route(question) is None


# -------------------------------------------------------------------- memory

def test_follow_ups(agent):
    sid = "follow"
    agent.ask("Who won the 2021 championship?", sid)
    r = agent.ask("How many races did he win that year?", sid)
    assert r.resolved_question == "How many races did Max Verstappen win in 2021?"
    assert r.tool == "driver_season"
    r = agent.ask("what about Hamilton?", sid)
    assert "lewis hamilton" in r.resolved_question and "2021" in r.resolved_question
    r = agent.ask("and in 2008?", sid)
    assert "2008" in r.resolved_question and "Lewis Hamilton" in r.answer


def test_sessions_are_isolated(agent):
    agent.ask("Who won the 2021 championship?", "a")
    r = agent.ask("How many races did he win?", "b")
    assert "Max Verstappen" not in r.resolved_question


# ----------------------------------------------------------------------- RAG

def test_rag_answers_with_citations(agent):
    r = agent.ask("What does a yellow flag mean?")
    assert not r.refusal and "overtaking" in r.answer.lower()
    assert r.sources and r.sources[0]["file"] == "f1_flags-and-penalties.txt"


@pytest.mark.parametrize("question", ["What is the capital of France?", "Write a python function to sort a list"])
def test_off_topic_refused(agent, question):
    assert agent.ask(question).refusal


# ---------------------------------------------------------------------- live

def _fake_openf1(live: bool):
    now = dt.datetime(2026, 11, 8, 18, 30, tzinfo=dt.timezone.utc)
    start = "2026-11-08T17:00:00+00:00"
    end = "2026-11-08T19:00:00+00:00" if live else "2026-11-08T18:00:00+00:00"

    def fetch(endpoint, params):
        if endpoint == "sessions" and params.get("session_key") == "latest":
            return [{"session_key": 9999, "session_name": "Race", "date_start": start, "date_end": end,
                     "country_name": "Testland", "year": 2026}]
        if endpoint == "sessions":
            return []
        if endpoint == "drivers":
            return [{"driver_number": 1, "full_name": "Driver One", "team_name": "Team A"},
                    {"driver_number": 2, "full_name": "Driver Two", "team_name": "Team B"}]
        if endpoint == "position":
            return [{"driver_number": 1, "position": 2, "date": "a"}, {"driver_number": 1, "position": 1, "date": "b"},
                    {"driver_number": 2, "position": 2, "date": "b"}]
        if endpoint == "intervals":
            return [{"driver_number": 2, "gap_to_leader": 1.234, "date": "b"}]
        if endpoint == "race_control":
            return [{"message": "GREEN LIGHT - PIT EXIT OPEN"}]
        if endpoint == "session_result":
            return [{"position": 1, "driver_number": 2, "gap_to_leader": 0}, {"position": 2, "driver_number": 1, "gap_to_leader": 5.0}]
        return []

    return fetch, now


def test_live_during_session():
    fetch, now = _fake_openf1(live=True)
    r = live_summary(fetch, now)
    assert r.ok and r.text.startswith("LIVE now")
    assert "P1 Driver One" in r.text and "P2 Driver Two (Team B) +1.234s" in r.text
    assert "PIT EXIT OPEN" in r.text


def test_live_after_session_uses_classification():
    fetch, now = _fake_openf1(live=False)
    r = live_summary(fetch, now)
    assert "Latest session (finished)" in r.text and "P1 Driver Two" in r.text


def test_live_network_failure_is_graceful():
    def boom(endpoint, params):
        raise OSError("no network")
    r = live_summary(boom)
    assert not r.ok and "Couldn't reach" in r.text


# ----------------------------------------------------------------------- API

def test_api_roundtrip(db, monkeypatch):
    from fastapi.testclient import TestClient

    from backend import rag, server

    monkeypatch.setattr(rag, "_SERVICE", F1Agent(db_path=db, load_vectors=False))
    client = TestClient(server.app)
    r = client.post("/ask", json={"question": "Hamilton vs Verstappen in 2021", "session_id": "api"}).json()
    assert r["tool"] == "compare" and r["chart"]["type"] == "line" and r["session_id"] == "api"
    assert client.get("/personas").json()["engineer"] == "Race engineer"
    assert client.get("/whatif", params={"season": 2008, "system": "wins_only"}).json()["ok"]
    assert client.post("/reset", json={"session_id": "api"}).json() == {"status": "reset"}
    assert client.get("/health").json()["stats_db"] is True
    assert "renderChart" in client.get("/").text
