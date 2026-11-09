# f1-chatbot

A local Formula 1 assistant that runs entirely on your machine. It answers **stats questions from a real results database**, **rules and history from a text knowledge base**, and **live session questions from OpenF1**, with a small local LLM (Qwen 2.5 3B via llama.cpp) tying it together.

## What it can do

| Ask | What happens |
| --- | --- |
| "How many races did Verstappen win in 2023?" | Stats tool → exact number from the DB + cumulative-points chart |
| "Compare Hamilton and Rosberg in 2016" | Head-to-head (wins, poles, points, who finished ahead) + chart |
| "Who has the most pole positions?" | All-time leaderboard + bar chart |
| "Who won the 2020 Sakhir GP?" | Race result: podium, top 10, pole, fastest lap |
| "What if 2008 was scored with the medals system?" | Re-scores the season under another points system |
| "What's happening live right now?" | OpenF1 live / latest session classification |
| "What does a blue flag mean?" | Hybrid retrieval over the knowledge base, answer with `[n]` citations |
| "How many did he win that year?" | Follow-up rewritten using conversation memory |

Answer styles: default, race engineer, explain-like-I'm-new, TV commentator.

## How it works

```text
question
  │
  ├─ memory.py     rewrite follow-ups ("he", "that year", "what about Alonso?")
  ├─ router.py     deterministic intent rules → tool + args
  │                └─ no rule matched but looks like stats → LLM picks a tool as JSON
  │
  ├─ tools.py      SQL over data/stats/f1_stats.sqlite (Jolpica API data, 1950 → now)
  ├─ live.py       OpenF1 API
  └─ retrieval.py  BM25 + vector search fused with RRF (+ optional cross-encoder)
                   → llm.py answers from the passages with citations
```

Two design choices worth knowing:

- **Numbers never come from the LLM.** Stats answers are computed from the database. If a persona is selected, the LLM rewrites the wording, and the rewrite is discarded if any number changed.
- **Hybrid search** because embeddings are weak at exact tokens (driver names, years) and keyword search is weak at meaning. Reciprocal Rank Fusion combines both without score calibration.

```text
f1-chatbot/
├── backend/
│   ├── server.py        FastAPI app
│   ├── agent.py         orchestration: memory → router → tool / RAG
│   ├── router.py        intent rules
│   ├── tools.py         stats tools (career, season, compare, leaderboard, what-if, ...)
│   ├── stats_db.py      SQLite schema, loader, fuzzy name resolution
│   ├── fetch_stats.py   Jolpica API → SQLite (cached, resumable)
│   ├── live.py          OpenF1 live timing
│   ├── retrieval.py     BM25 + vector + RRF
│   ├── memory.py        per-session memory, follow-up rewriting
│   ├── llm.py           llama.cpp wrapper
│   └── ingest.py        builds the vector index (text files + season summaries)
├── data/*.txt           knowledge base (rules, flags, tyres, history ...)
├── eval/                50+ question eval set and runner
├── tests/               offline unit tests (no network or model needed)
└── frontend/index.html  chat UI with charts and citations
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put a GGUF model in `model/` (tested with `qwen2.5-3b-instruct-q4_k_m.gguf`) or set `F1_MODEL_PATH`. Without a model the bot still works: stats answers are unaffected and text answers fall back to extractive snippets.

### 1. Build the stats database

```bash
python -m backend.fetch_stats            # all seasons, 1950 → current
python -m backend.fetch_stats --from 2000  # quicker partial pull
python -m backend.fetch_stats --refresh-current  # after each race weekend
```

The first full pull is a few hundred API requests and takes a while because Jolpica limits clients to ~500 requests/hour; it backs off automatically and caches every response, so an interrupted run resumes where it stopped.

### 2. Build the search index

```bash
python -m backend.ingest
```

Indexes `data/*.txt` plus one generated summary per season from the stats DB. Re-run after editing text files or refreshing stats.

### 3. Run

```bash
uvicorn backend.server:app --reload
```

Open `http://127.0.0.1:8000`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `F1_MODEL_PATH` | first `model/*.gguf` | LLM file |
| `F1_MAX_TOKENS` | `300` | answer length cap |
| `F1_N_CTX` | `4096` | model context window |
| `F1_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | embedding model (re-run ingest after changing) |
| `F1_RERANKER` | off | `1` enables `cross-encoder/ms-marco-MiniLM-L-6-v2`, or pass a model name |
| `OPENF1_TOKEN` | unset | OpenF1 account token; needed only for timing *during* a live session |

## API

- `POST /ask` `{ "question": "...", "session_id": "optional", "persona": "default|engineer|newbie|commentator" }` → answer, sources, tool used, chart data, session id, rewritten question
- `POST /reset` `{ "session_id": "..." }` clears conversation memory
- `GET /live` latest / live session
- `GET /whatif?season=2008&system=wins_only` (`current`, `2003`, `1991`, `1961`, `wins_only`)
- `GET /personas`, `GET /health`, `POST /reload`

## Testing and evaluation

```bash
pip install -r requirements-dev.txt
python -m pytest -q          # offline unit tests (fixture data, no network, no model)
python -m eval.run_eval      # 50+ real questions against your built DB and index
python -m eval.run_eval --verbose --only stats
```

The eval covers stats, retrieval, refusals and multi-turn follow-ups. Run it after any change to know whether accuracy actually improved.

## Data sources

- Results and standings: [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) (successor to Ergast)
- Live timing: [OpenF1](https://openf1.org)
- Rules and background: hand-written notes in `data/`

The bot only answers Formula 1 questions and refuses anything else.
