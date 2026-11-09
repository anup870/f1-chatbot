"""Live / latest session data from the OpenF1 API (https://openf1.org).

Historical OpenF1 data is free. Real-time data *during* a session needs an
OpenF1 account; set OPENF1_TOKEN to use it. Without one, "latest session"
still works once the session has finished.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from backend.tools import ToolResult

OPENF1_BASE = "https://api.openf1.org/v1"
SOURCE = {"id": "openf1", "file": "OpenF1 API (live timing)", "snippet": "api.openf1.org"}

Fetcher = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


def http_fetch(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    # OpenF1 uses operators inside keys (e.g. "date_start>"), so build the query by hand.
    query = "&".join(
        f"{urllib.parse.quote(str(k), safe='<>=')}{'' if str(k)[-1] in '<>' else '='}{urllib.parse.quote(str(v))}"
        for k, v in params.items()
    )
    url = f"{OPENF1_BASE}/{endpoint}?{query}"
    headers = {"User-Agent": "f1-chatbot/1.0"}
    token = os.getenv("OPENF1_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data if isinstance(data, list) else []


def _parse_time(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _latest_by_driver(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        number = row.get("driver_number")
        if number is None:
            continue
        if number not in latest or (row.get("date") or "") >= (latest[number].get("date") or ""):
            latest[number] = row
    return latest


def _gap(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        return f"+{value:.3f}s"
    return str(value)


def live_summary(fetch: Fetcher = http_fetch, now: Optional[dt.datetime] = None) -> ToolResult:
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        sessions = fetch("sessions", {"session_key": "latest"})
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return ToolResult(
                text="A session is live right now, and OpenF1 only serves live timing to registered accounts. "
                     "Set OPENF1_TOKEN to enable it; results appear for free once the session ends.",
                tool="live", ok=False, sources=[SOURCE])
        return ToolResult(text=f"Couldn't reach OpenF1 (HTTP {exc.code}).", tool="live", ok=False, sources=[SOURCE])
    except Exception as exc:  # network down, DNS, timeout
        return ToolResult(text=f"Couldn't reach the OpenF1 live API ({exc.__class__.__name__}).", tool="live", ok=False, sources=[SOURCE])

    if not sessions:
        return ToolResult(text="OpenF1 returned no session data.", tool="live", ok=False, sources=[SOURCE])

    session = sessions[-1]
    key = session["session_key"]
    start, end = _parse_time(session.get("date_start")), _parse_time(session.get("date_end"))
    is_live = bool(start and end and start <= now <= end)
    title = f"{session.get('year', '')} {session.get('country_name') or session.get('location', '')} — {session.get('session_name', 'Session')}".strip()

    drivers = {d["driver_number"]: d for d in fetch("drivers", {"session_key": key}) if "driver_number" in d}

    def name(number: int) -> str:
        d = drivers.get(number, {})
        return d.get("full_name") or d.get("name_acronym") or f"#{number}"

    def team(number: int) -> str:
        return drivers.get(number, {}).get("team_name", "")

    order: list[tuple[int, int, str]] = []  # (position, driver_number, gap)
    if not is_live:
        try:
            for row in fetch("session_result", {"session_key": key}):
                if row.get("position"):
                    order.append((int(row["position"]), row["driver_number"], _gap(row.get("gap_to_leader"))))
        except Exception:
            order = []
    if not order:
        positions = _latest_by_driver(fetch("position", {"session_key": key}))
        intervals = _latest_by_driver(fetch("intervals", {"session_key": key})) if is_live else {}
        for number, row in positions.items():
            order.append((int(row["position"]), number, _gap(intervals.get(number, {}).get("gap_to_leader"))))
    order.sort()

    lines = []
    for pos, number, gap in order[:10]:
        team_txt = f" ({team(number)})" if team(number) else ""
        lines.append(f"P{pos} {name(number)}{team_txt}{(' ' + gap) if gap and pos > 1 else ''}")

    status = "LIVE now" if is_live else "Latest session (finished)"
    text = f"{status}: {title}.\n" + ("\n".join(lines) if lines else "No classification yet.")

    if is_live:
        try:
            messages = fetch("race_control", {"session_key": key})[-3:]
            if messages:
                text += "\nRace control: " + " | ".join(m.get("message", "") for m in messages)
        except Exception:
            pass

    try:
        upcoming = fetch("sessions", {"date_start>": now.strftime("%Y-%m-%dT%H:%M:%S")})
        if upcoming:
            nxt = min(upcoming, key=lambda s: s.get("date_start", ""))
            text += f"\nNext: {nxt.get('country_name', '')} {nxt.get('session_name', '')} at {nxt.get('date_start', '')} UTC."
    except Exception:
        pass

    return ToolResult(
        text=text,
        tool="live",
        data={"session_key": key, "live": is_live, "title": title,
              "order": [{"position": p, "driver": name(n), "team": team(n), "gap": g} for p, n, g in order[:20]]},
        sources=[SOURCE],
    )
