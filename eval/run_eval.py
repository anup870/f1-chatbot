"""Score the bot on eval/questions.jsonl.

    python -m eval.run_eval                 # full run against your stats DB
    python -m eval.run_eval --live          # also run network-dependent cases
    python -m eval.run_eval --only stats    # one category
    python -m eval.run_eval --verbose       # print every answer

A case passes when every turn (a) contains at least one expected string
(case-insensitive), (b) used the expected tool when one is given, and (c)
refused / didn't refuse as expected. Re-run after every change so you know
whether a "fix" actually helped.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from backend.agent import F1Agent

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def check_turn(turn: dict, result) -> list[str]:
    problems = []
    answer = result.answer.lower()
    if turn.get("expect_refusal") is True and not result.refusal:
        problems.append("expected a refusal")
    if turn.get("expect_refusal") is not True and result.refusal and (turn.get("expect_any") or turn.get("expect_tool")):
        problems.append("unexpected refusal")
    expected = turn.get("expect_any")
    if expected and not any(e.lower() in answer for e in expected):
        problems.append(f"missing any of {expected}")
    if turn.get("expect_tool") and result.tool != turn["expect_tool"]:
        problems.append(f"tool {result.tool} != {turn['expect_tool']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="include network-dependent cases")
    parser.add_argument("--only", help="run one category")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--db", help="stats DB path (default: data/stats/f1_stats.sqlite)")
    parser.add_argument("--no-vectors", action="store_true", help="BM25 only (faster startup)")
    args = parser.parse_args(argv)

    cases = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["category"] == args.only]
    if not args.live:
        cases = [c for c in cases if not c.get("network")]

    agent = F1Agent(db_path=args.db, load_vectors=not args.no_vectors)
    if not agent.stats:
        print("WARNING: no stats DB found; stats cases will fail. Run: python -m backend.fetch_stats\n")

    per_cat: dict[str, list[bool]] = defaultdict(list)
    failures = []
    latencies = []
    for case in cases:
        session = f"eval-{case['id']}"
        problems: list[str] = []
        for turn in case["turns"]:
            start = time.perf_counter()
            result = agent.ask(turn["q"], session_id=session)
            latencies.append(time.perf_counter() - start)
            turn_problems = check_turn(turn, result)
            if args.verbose:
                mark = "ok " if not turn_problems else "BAD"
                print(f"[{mark}] {case['id']}: {turn['q']}\n      -> ({result.tool}) {result.answer[:200]!r}")
            if turn_problems:
                problems.append(f"{turn['q']!r}: {'; '.join(turn_problems)} | got ({result.tool}) {result.answer[:160]!r}")
        per_cat[case["category"]].append(not problems)
        if problems:
            failures.append((case["id"], problems))

    total = sum(len(v) for v in per_cat.values())
    passed = sum(sum(v) for v in per_cat.values())
    print("\nCategory     Pass")
    for cat, results in sorted(per_cat.items()):
        print(f"{cat:<12} {sum(results)}/{len(results)}")
    print(f"{'TOTAL':<12} {passed}/{total} ({100 * passed / max(total, 1):.0f}%)")
    if latencies:
        lat = sorted(latencies)
        print(f"Latency: median {lat[len(lat) // 2] * 1000:.0f} ms, p90 {lat[int(len(lat) * 0.9)] * 1000:.0f} ms")
    if failures:
        print("\nFailures:")
        for case_id, problems in failures:
            for p in problems:
                print(f"  {case_id}: {p}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
