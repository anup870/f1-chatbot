"""Structured F1 statistics store (SQLite) built from the Jolpica/Ergast API.

Questions like "who has the most poles?" or "how many races did Verstappen win
in 2023?" are database lookups, not text retrieval. This module owns the schema,
the loader that turns API JSON into rows, and entity resolution (fuzzy name ->
driver_id / constructor_id).
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "stats" / "f1_stats.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS drivers (
    driver_id TEXT PRIMARY KEY,
    code TEXT,
    given_name TEXT,
    family_name TEXT,
    full_name TEXT,
    nationality TEXT,
    dob TEXT
);
CREATE TABLE IF NOT EXISTS constructors (
    constructor_id TEXT PRIMARY KEY,
    name TEXT,
    nationality TEXT
);
CREATE TABLE IF NOT EXISTS races (
    season INTEGER,
    round INTEGER,
    name TEXT,
    circuit_id TEXT,
    circuit_name TEXT,
    locality TEXT,
    country TEXT,
    date TEXT,
    PRIMARY KEY (season, round)
);
CREATE TABLE IF NOT EXISTS results (
    season INTEGER,
    round INTEGER,
    driver_id TEXT,
    constructor_id TEXT,
    grid INTEGER,
    position INTEGER,           -- NULL when not classified
    position_text TEXT,
    points REAL,
    laps INTEGER,
    status TEXT,
    fastest_lap_rank INTEGER,
    PRIMARY KEY (season, round, driver_id)
);
CREATE TABLE IF NOT EXISTS sprint_results (
    season INTEGER,
    round INTEGER,
    driver_id TEXT,
    constructor_id TEXT,
    grid INTEGER,
    position INTEGER,
    position_text TEXT,
    points REAL,
    status TEXT,
    PRIMARY KEY (season, round, driver_id)
);
CREATE TABLE IF NOT EXISTS qualifying (
    season INTEGER,
    round INTEGER,
    driver_id TEXT,
    constructor_id TEXT,
    position INTEGER,
    q1 TEXT,
    q2 TEXT,
    q3 TEXT,
    PRIMARY KEY (season, round, driver_id)
);
CREATE TABLE IF NOT EXISTS driver_standings (
    season INTEGER,
    position INTEGER,
    driver_id TEXT,
    points REAL,
    wins INTEGER,
    constructor_ids TEXT,
    PRIMARY KEY (season, driver_id)
);
CREATE TABLE IF NOT EXISTS constructor_standings (
    season INTEGER,
    position INTEGER,
    constructor_id TEXT,
    points REAL,
    wins INTEGER,
    PRIMARY KEY (season, constructor_id)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_results_driver ON results(driver_id);
CREATE INDEX IF NOT EXISTS idx_results_constructor ON results(constructor_id);
CREATE INDEX IF NOT EXISTS idx_quali_driver ON qualifying(driver_id);
"""

# Common nicknames / first-name shorthands people actually type.
DRIVER_ALIASES = {
    "max": "max_verstappen",
    "checo": "perez",
    "lewis": "hamilton",
    "seb": "vettel",
    "kimi": "raikkonen",
    "nando": "alonso",
    "fernando": "alonso",
    "schumi": "michael_schumacher",
    "lando": "norris",
    "charles": "leclerc",
    "oscar": "piastri",
    "george": "russell",
    "carlos": "sainz",
    "ayrton": "senna",
    "prost": "prost",
}

CONSTRUCTOR_ALIASES = {
    "redbull": "red_bull",
    "red bull racing": "red_bull",
    "rbr": "red_bull",
    "merc": "mercedes",
    "mercs": "mercedes",
    "scuderia ferrari": "ferrari",
    "macca": "mclaren",
    "aston": "aston_martin",
    "kick sauber": "sauber",
    "alfa": "alfa",
    "toro rosso": "toro_rosso",
    "alphatauri": "alphatauri",
    "racing bulls": "rb",
    "brawn gp": "brawn",
}


def fold(text: str) -> str:
    """Lowercase and strip accents so 'Räikkönen' matches 'raikkonen'."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower().strip()


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def connect(db_path: Path | str = DEFAULT_DB_PATH, read_only: bool = False) -> sqlite3.Connection:
    path = Path(db_path)
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Loading API JSON into the schema
# ---------------------------------------------------------------------------

def _upsert_driver(conn: sqlite3.Connection, driver: dict) -> str:
    driver_id = driver["driverId"]
    given = driver.get("givenName", "")
    family = driver.get("familyName", "")
    conn.execute(
        "INSERT OR REPLACE INTO drivers VALUES (?,?,?,?,?,?,?)",
        (
            driver_id,
            driver.get("code"),
            given,
            family,
            f"{given} {family}".strip(),
            driver.get("nationality"),
            driver.get("dateOfBirth"),
        ),
    )
    return driver_id


def _upsert_constructor(conn: sqlite3.Connection, constructor: dict) -> str:
    constructor_id = constructor["constructorId"]
    conn.execute(
        "INSERT OR REPLACE INTO constructors VALUES (?,?,?)",
        (constructor_id, constructor.get("name"), constructor.get("nationality")),
    )
    return constructor_id


def _upsert_race(conn: sqlite3.Connection, race: dict) -> tuple[int, int]:
    season = int(race["season"])
    rnd = int(race["round"])
    circuit = race.get("Circuit", {})
    location = circuit.get("Location", {})
    conn.execute(
        "INSERT OR REPLACE INTO races VALUES (?,?,?,?,?,?,?,?)",
        (
            season,
            rnd,
            race.get("raceName"),
            circuit.get("circuitId"),
            circuit.get("circuitName"),
            location.get("locality"),
            location.get("country"),
            race.get("date"),
        ),
    )
    return season, rnd


def load_races_payload(conn: sqlite3.Connection, races: Iterable[dict]) -> int:
    """Load race results, sprint results and qualifying from Ergast-format Race objects.

    Paged API responses can split one race across pages; INSERT OR REPLACE on the
    (season, round, driver) key makes merging pages safe.
    """
    rows = 0
    for race in races:
        season, rnd = _upsert_race(conn, race)
        for res in race.get("Results", []):
            driver_id = _upsert_driver(conn, res["Driver"])
            constructor_id = _upsert_constructor(conn, res["Constructor"])
            fastest = res.get("FastestLap", {}) or {}
            conn.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    season,
                    rnd,
                    driver_id,
                    constructor_id,
                    _int(res.get("grid")),
                    _int(res.get("position")) if str(res.get("positionText", "")).isdigit() else None,
                    res.get("positionText"),
                    _float(res.get("points")),
                    _int(res.get("laps")),
                    res.get("status"),
                    _int(fastest.get("rank")),
                ),
            )
            rows += 1
        for res in race.get("SprintResults", []):
            driver_id = _upsert_driver(conn, res["Driver"])
            constructor_id = _upsert_constructor(conn, res["Constructor"])
            conn.execute(
                "INSERT OR REPLACE INTO sprint_results VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    season,
                    rnd,
                    driver_id,
                    constructor_id,
                    _int(res.get("grid")),
                    _int(res.get("position")) if str(res.get("positionText", "")).isdigit() else None,
                    res.get("positionText"),
                    _float(res.get("points")),
                    res.get("status"),
                ),
            )
            rows += 1
        for res in race.get("QualifyingResults", []):
            driver_id = _upsert_driver(conn, res["Driver"])
            constructor_id = _upsert_constructor(conn, res["Constructor"])
            conn.execute(
                "INSERT OR REPLACE INTO qualifying VALUES (?,?,?,?,?,?,?,?)",
                (
                    season,
                    rnd,
                    driver_id,
                    constructor_id,
                    _int(res.get("position")),
                    res.get("Q1"),
                    res.get("Q2"),
                    res.get("Q3"),
                ),
            )
            rows += 1
    return rows


def load_driver_standings(conn: sqlite3.Connection, season: int, standings: Iterable[dict]) -> int:
    rows = 0
    for entry in standings:
        driver_id = _upsert_driver(conn, entry["Driver"])
        constructor_ids = []
        for constructor in entry.get("Constructors", []):
            constructor_ids.append(_upsert_constructor(conn, constructor))
        conn.execute(
            "INSERT OR REPLACE INTO driver_standings VALUES (?,?,?,?,?,?)",
            (
                season,
                _int(entry.get("position")),
                driver_id,
                _float(entry.get("points")),
                _int(entry.get("wins")) or 0,
                ",".join(constructor_ids),
            ),
        )
        rows += 1
    return rows


def load_constructor_standings(conn: sqlite3.Connection, season: int, standings: Iterable[dict]) -> int:
    rows = 0
    for entry in standings:
        constructor_id = _upsert_constructor(conn, entry["Constructor"])
        conn.execute(
            "INSERT OR REPLACE INTO constructor_standings VALUES (?,?,?,?,?)",
            (
                season,
                _int(entry.get("position")),
                constructor_id,
                _float(entry.get("points")),
                _int(entry.get("wins")) or 0,
            ),
        )
        rows += 1
    return rows


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------

class EntityIndex:
    """Maps free-text mentions to driver/constructor ids.

    Ambiguous surnames (Schumacher, Hill, Rosberg, Verstappen) resolve to the
    driver with the most wins, then the most starts, unless a full name or
    first name is given.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.drivers: dict[str, dict] = {}
        self.surname_map: dict[str, list[str]] = {}
        self.fullname_map: dict[str, str] = {}
        self.code_map: dict[str, str] = {}
        self.constructors: dict[str, dict] = {}
        self.constructor_name_map: dict[str, str] = {}
        self._build()

    def _build(self) -> None:
        prominence = {
            row["driver_id"]: (row["wins"] or 0, row["starts"] or 0)
            for row in self.conn.execute(
                "SELECT driver_id, SUM(position=1) AS wins, COUNT(*) AS starts FROM results GROUP BY driver_id"
            )
        }
        for row in self.conn.execute("SELECT * FROM drivers"):
            info = dict(row)
            info["prominence"] = prominence.get(row["driver_id"], (0, 0))
            self.drivers[row["driver_id"]] = info
            surname = fold(row["family_name"] or "")
            if surname:
                self.surname_map.setdefault(surname, []).append(row["driver_id"])
            self.fullname_map[fold(row["full_name"] or "")] = row["driver_id"]
            if row["code"]:
                self.code_map[row["code"].lower()] = row["driver_id"]
        for ids in self.surname_map.values():
            ids.sort(key=lambda d: self.drivers[d]["prominence"], reverse=True)

        for row in self.conn.execute("SELECT * FROM constructors"):
            self.constructors[row["constructor_id"]] = dict(row)
            self.constructor_name_map[fold(row["name"] or "")] = row["constructor_id"]
            self.constructor_name_map[fold(row["constructor_id"].replace("_", " "))] = row["constructor_id"]

    def driver_name(self, driver_id: str) -> str:
        info = self.drivers.get(driver_id)
        return info["full_name"] if info else driver_id

    def constructor_name(self, constructor_id: str) -> str:
        info = self.constructors.get(constructor_id)
        return info["name"] if info else constructor_id

    def _resolve_alias(self, alias_target: str) -> Optional[str]:
        if alias_target in self.drivers:
            return alias_target
        candidates = self.surname_map.get(alias_target)
        return candidates[0] if candidates else None

    def find_drivers(self, text: str) -> list[str]:
        """Return driver ids mentioned in text, in order of appearance."""
        return [driver_id for _, _, driver_id in self.driver_spans(text)]

    def driver_spans(self, text: str) -> list[tuple[int, int, str]]:
        """(start, end, driver_id) for each mention, as offsets into fold(text)."""
        folded = fold(text)
        found: list[tuple[int, int, str]] = []
        taken_spans: list[tuple[int, int]] = []

        def add(pos: int, end: int, driver_id: Optional[str]) -> None:
            if driver_id is None:
                return
            if any(not (end <= s or pos >= e) for s, e in taken_spans):
                return
            if driver_id in [d for _, _, d in found]:
                return
            taken_spans.append((pos, end))
            found.append((pos, end, driver_id))

        # Full names first (most specific).
        for full, driver_id in sorted(self.fullname_map.items(), key=lambda kv: -len(kv[0])):
            if not full:
                continue
            for match in re.finditer(rf"\b{re.escape(full)}\b", folded):
                add(match.start(), match.end(), driver_id)
        # "Michael Schumacher" style handled above; now surnames.
        for surname, ids in sorted(self.surname_map.items(), key=lambda kv: -len(kv[0])):
            if len(surname) < 3:
                continue
            for match in re.finditer(rf"\b{re.escape(surname)}\b", folded):
                add(match.start(), match.end(), ids[0])
        for alias, target in DRIVER_ALIASES.items():
            for match in re.finditer(rf"\b{re.escape(alias)}\b", folded):
                add(match.start(), match.end(), self._resolve_alias(target))
        found.sort()
        return found

    def find_constructors(self, text: str) -> list[str]:
        return [constructor_id for _, _, constructor_id in self.constructor_spans(text)]

    def constructor_spans(self, text: str) -> list[tuple[int, int, str]]:
        folded = fold(text)
        found: list[tuple[int, int, str]] = []
        taken: list[tuple[int, int]] = []
        names = dict(self.constructor_name_map)
        for alias, target in CONSTRUCTOR_ALIASES.items():
            if target in self.constructors:
                names[alias] = target
        for name, constructor_id in sorted(names.items(), key=lambda kv: -len(kv[0])):
            if len(name) < 3:
                continue
            match = re.search(rf"\b{re.escape(name)}\b", folded)
            if not match or constructor_id in [c for _, _, c in found]:
                continue
            if any(not (match.end() <= s or match.start() >= e) for s, e in taken):
                continue
            taken.append((match.start(), match.end()))
            found.append((match.start(), match.end(), constructor_id))
        found.sort()
        return found


def database_exists(db_path: Path | str = DEFAULT_DB_PATH) -> bool:
    path = Path(db_path)
    if not path.exists():
        return False
    try:
        conn = connect(path, read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        conn.close()
        return count > 0
    except sqlite3.Error:
        return False
