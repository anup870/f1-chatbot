"""Download F1 results/standings from the Jolpica (Ergast-compatible) API into SQLite.

Usage:
    python -m backend.fetch_stats                 # every season 1950 -> current
    python -m backend.fetch_stats --from 2010     # a range
    python -m backend.fetch_stats --refresh-current  # re-pull only this season

Raw responses are cached under data/stats/cache/ so an interrupted run resumes
where it stopped, and re-runs only hit the network for the current season.
Jolpica allows ~4 requests/second and ~500/hour; a full history pull is a few
hundred requests, so the first run takes a while. It backs off automatically on
HTTP 429.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from backend import stats_db

API_BASE = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR = stats_db.DEFAULT_DB_PATH.parent / "cache"
PAGE_LIMIT = 100
MIN_INTERVAL_S = 0.3
USER_AGENT = "f1-chatbot-stats-fetcher/1.0"

_last_request = 0.0


def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(url.encode()).hexdigest()}.json"


def get_json(url: str, use_cache: bool = True, retries: int = 6) -> dict[str, Any]:
    global _last_request
    cache_file = _cache_path(url)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                delay = int(exc.headers.get("Retry-After") or 60)
                print(f"  rate limited, sleeping {delay}s...", flush=True)
                time.sleep(delay)
                continue
            if exc.code >= 500 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"Gave up fetching {url}")


def fetch_paged_races(path: str, use_cache: bool) -> list[dict]:
    """Fetch every page of a race-table endpoint and merge split races."""
    merged: dict[tuple[str, str], dict] = {}
    offset = 0
    while True:
        url = f"{API_BASE}/{path}/?limit={PAGE_LIMIT}&offset={offset}"
        data = get_json(url, use_cache=use_cache)["MRData"]
        races = data.get("RaceTable", {}).get("Races", [])
        for race in races:
            key = (race["season"], race["round"])
            if key not in merged:
                merged[key] = {k: v for k, v in race.items() if k not in ("Results", "SprintResults", "QualifyingResults")}
            for list_key in ("Results", "SprintResults", "QualifyingResults"):
                if list_key in race:
                    merged[key].setdefault(list_key, []).extend(race[list_key])
        total = int(data.get("total", 0))
        offset += PAGE_LIMIT
        if offset >= total:
            break
    return list(merged.values())


def fetch_standings(season: int, kind: str, use_cache: bool) -> list[dict]:
    url = f"{API_BASE}/{season}/{kind}standings/?limit={PAGE_LIMIT}"
    data = get_json(url, use_cache=use_cache)["MRData"]
    lists = data.get("StandingsTable", {}).get("StandingsLists", [])
    if not lists:
        return []
    return lists[0].get("DriverStandings" if kind == "driver" else "ConstructorStandings", [])


def current_season() -> int:
    data = get_json(f"{API_BASE}/current/?limit=1", use_cache=False)["MRData"]
    races = data.get("RaceTable", {}).get("Races", [])
    if races:
        return int(races[0]["season"])
    return dt.date.today().year


def sync_season(conn, season: int, use_cache: bool) -> int:
    rows = 0
    rows += stats_db.load_races_payload(conn, fetch_paged_races(f"{season}/results", use_cache))
    if season >= 1994:  # qualifying data is sparse before this in the source
        rows += stats_db.load_races_payload(conn, fetch_paged_races(f"{season}/qualifying", use_cache))
    if season >= 2021:  # sprint races started in 2021
        rows += stats_db.load_races_payload(conn, fetch_paged_races(f"{season}/sprint", use_cache))
    rows += stats_db.load_driver_standings(conn, season, fetch_standings(season, "driver", use_cache))
    if season >= 1958:  # constructors' championship started in 1958
        rows += stats_db.load_constructor_standings(conn, season, fetch_standings(season, "constructor", use_cache))
    conn.commit()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=int, default=1950)
    parser.add_argument("--to", dest="end", type=int, default=None)
    parser.add_argument("--refresh-current", action="store_true", help="only re-pull the current season")
    parser.add_argument("--db", default=str(stats_db.DEFAULT_DB_PATH))
    args = parser.parse_args(argv)

    latest = current_season()
    end = args.end or latest
    seasons = [latest] if args.refresh_current else list(range(args.start, end + 1))

    conn = stats_db.connect(args.db)
    print(f"Syncing seasons {seasons[0]}-{seasons[-1]} into {args.db}")
    for season in seasons:
        # Finished seasons never change, so cached pages are reused; the live
        # season is always re-downloaded.
        use_cache = season < latest
        try:
            rows = sync_season(conn, season, use_cache)
        except Exception as exc:  # keep going; a re-run resumes from cache
            print(f"  {season}: FAILED ({exc})", file=sys.stderr)
            continue
        print(f"  {season}: {rows} rows", flush=True)

    # Total rounds on this season's calendar, so the bot knows whether the
    # standings leader is the champion or just leading.
    schedule = get_json(f"{API_BASE}/{latest}/?limit={PAGE_LIMIT}", use_cache=False)["MRData"]
    stats_db.set_meta(conn, f"rounds_{latest}", str(schedule.get("total", "0")))
    stats_db.set_meta(conn, "last_sync", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    stats_db.set_meta(conn, "latest_season", str(latest))
    conn.commit()
    conn.close()
    print("Done. Rebuild the text index too: python -m backend.ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
