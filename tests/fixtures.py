"""Tiny offline dataset in Jolpica/Ergast JSON shape, used by the tests.

Real results for three races (2008 Brazil, 2021 Bahrain, 2021 Abu Dhabi) plus
the real final standings for those seasons. Only a handful of rows, so
per-season totals in tests reflect these races, not the full seasons.
"""

from __future__ import annotations

from pathlib import Path

from backend import stats_db

D = {
    "hamilton": ("Lewis", "Hamilton", "HAM", "British"),
    "max_verstappen": ("Max", "Verstappen", "VER", "Dutch"),
    "bottas": ("Valtteri", "Bottas", "BOT", "Finnish"),
    "sainz": ("Carlos", "Sainz", "SAI", "Spanish"),
    "perez": ("Sergio", "Pérez", "PER", "Mexican"),
    "massa": ("Felipe", "Massa", "MAS", "Brazilian"),
    "alonso": ("Fernando", "Alonso", "ALO", "Spanish"),
    "raikkonen": ("Kimi", "Räikkönen", "RAI", "Finnish"),
    "michael_schumacher": ("Michael", "Schumacher", "MSC", "German"),
    "ralf_schumacher": ("Ralf", "Schumacher", None, "German"),
}
C = {
    "mercedes": "Mercedes",
    "red_bull": "Red Bull",
    "ferrari": "Ferrari",
    "mclaren": "McLaren",
    "renault": "Renault",
    "williams": "Williams",
}


def drv(driver_id: str) -> dict:
    given, family, code, nat = D[driver_id]
    out = {"driverId": driver_id, "givenName": given, "familyName": family, "nationality": nat}
    if code:
        out["code"] = code
    return out


def con(constructor_id: str) -> dict:
    return {"constructorId": constructor_id, "name": C[constructor_id], "nationality": ""}


def res(pos, driver_id, constructor_id, points, grid, fl_rank=None, status="Finished"):
    out = {
        "position": str(pos),
        "positionText": str(pos) if isinstance(pos, int) else "R",
        "points": str(points),
        "Driver": drv(driver_id),
        "Constructor": con(constructor_id),
        "grid": str(grid),
        "laps": "57",
        "status": status,
    }
    if fl_rank:
        out["FastestLap"] = {"rank": str(fl_rank)}
    return out


def race(season, rnd, name, circuit_id, circuit_name, locality, country, date, results, quali=None):
    out = {
        "season": str(season),
        "round": str(rnd),
        "raceName": name,
        "Circuit": {"circuitId": circuit_id, "circuitName": circuit_name, "Location": {"locality": locality, "country": country}},
        "date": date,
        "Results": results,
    }
    if quali:
        out["QualifyingResults"] = [
            {"position": str(i), "Driver": drv(d), "Constructor": con(c)} for i, (d, c) in enumerate(quali, 1)
        ]
    return out


RACES = [
    race(2008, 18, "Brazilian Grand Prix", "interlagos", "Autódromo José Carlos Pace", "São Paulo", "Brazil", "2008-11-02",
         [res(1, "massa", "ferrari", 10, 1, fl_rank=1), res(2, "alonso", "renault", 8, 6), res(3, "raikkonen", "ferrari", 6, 3),
          res(5, "hamilton", "mclaren", 4, 4)],
         quali=[("massa", "ferrari"), ("hamilton", "mclaren")]),
    race(2021, 1, "Bahrain Grand Prix", "bahrain", "Bahrain International Circuit", "Sakhir", "Bahrain", "2021-03-28",
         [res(1, "hamilton", "mercedes", 25, 2), res(2, "max_verstappen", "red_bull", 18, 1),
          res(3, "bottas", "mercedes", 16, 3, fl_rank=1), res(5, "perez", "red_bull", 10, 11)],
         quali=[("max_verstappen", "red_bull"), ("hamilton", "mercedes"), ("bottas", "mercedes")]),
    race(2021, 22, "Abu Dhabi Grand Prix", "yas_marina", "Yas Marina Circuit", "Abu Dhabi", "UAE", "2021-12-12",
         [res(1, "max_verstappen", "red_bull", 26, 1, fl_rank=1), res(2, "hamilton", "mercedes", 18, 2),
          res(3, "sainz", "ferrari", 15, 5), res("R", "perez", "red_bull", 0, 4, status="Retired")],
         quali=[("max_verstappen", "red_bull"), ("hamilton", "mercedes")]),
]

DRIVER_STANDINGS = {
    2008: [("1", "hamilton", 98, 5, ["mclaren"]), ("2", "massa", 97, 6, ["ferrari"]), ("3", "raikkonen", 75, 2, ["ferrari"]),
           ("5", "alonso", 61, 2, ["renault"])],
    2021: [("1", "max_verstappen", 395.5, 10, ["red_bull"]), ("2", "hamilton", 387.5, 8, ["mercedes"]),
           ("3", "bottas", 226, 1, ["mercedes"]), ("4", "perez", 190, 1, ["red_bull"]), ("5", "sainz", 164.5, 0, ["ferrari"])],
}
CONSTRUCTOR_STANDINGS = {
    2008: [("1", "ferrari", 172, 8), ("2", "mclaren", 151, 6)],
    2021: [("1", "mercedes", 613.5, 9), ("2", "red_bull", 585.5, 11), ("3", "ferrari", 323.5, 0)],
}


def build_fixture_db(path: Path) -> Path:
    if path.exists():
        path.unlink()
    conn = stats_db.connect(path)
    stats_db.load_races_payload(conn, RACES)
    # Register a second Schumacher with no results to exercise disambiguation.
    stats_db._upsert_driver(conn, drv("michael_schumacher"))
    stats_db._upsert_driver(conn, drv("ralf_schumacher"))
    for season, rows in DRIVER_STANDINGS.items():
        stats_db.load_driver_standings(conn, season, [
            {"position": p, "Driver": drv(d), "points": str(pts), "wins": str(w), "Constructors": [con(c) for c in cs]}
            for p, d, pts, w, cs in rows
        ])
    for season, rows in CONSTRUCTOR_STANDINGS.items():
        stats_db.load_constructor_standings(conn, season, [
            {"position": p, "Constructor": con(c), "points": str(pts), "wins": str(w)} for p, c, pts, w in rows
        ])
    stats_db.set_meta(conn, "rounds_2021", "22")
    stats_db.set_meta(conn, "last_sync", "test-fixture")
    conn.commit()
    conn.close()
    return path
