"""F1Agent: routes each question to a stats tool, live data, or text retrieval.

Flow for one question:
  1. small talk -> canned reply
  2. memory rewrites follow-ups ("what about Alonso?") into standalone questions
  3. router picks a tool (rules first, then the LLM as a JSON tool-caller)
  4. tool answers are computed from data; the LLM may restyle them for a
     persona, but any reply that changes a number is thrown away
  5. otherwise hybrid retrieval + the LLM answer with [n] citations
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from backend import ingest, stats_db
from backend.live import live_summary
from backend.llm import LocalLLM
from backend.memory import ConversationMemory
from backend.retrieval import Chunk, HybridRetriever, tokenize
from backend.router import Route, Router
from backend.tools import TOOL_SPECS, StatsTools, ToolResult

logger = logging.getLogger(__name__)

NOT_F1 = "I only answer Formula 1 questions."
NO_CONTEXT = "I do not have enough F1 context to answer that."
MIN_DOMAIN_SIM = 0.30

F1_KEYWORDS = re.compile(
    r"\b(f1|formula ?(one|1)|grand prix|gp|fia|qualifying|constructors?|drivers?|pit ?stops?|pit lane|drs|ers|"
    r"mgu-?[kh]|tyres?|tires?|compounds?|pirelli|grid|stewards?|safety car|vsc|virtual safety car|parc ferme|"
    r"sprint|pole|podium|chequered|flags?|penalt(y|ies)|overtak\w*|undercut|overcut|downforce|diffuser|"
    r"power unit|paddock|race|races|racing|lap|laps|championships?|world champions?|titles?|grands prix|team orders|track limits|"
    r"blue flag|yellow flag|red flag|black flag|kerbs?|monza|silverstone|spa|monaco|suzuka|interlagos)\b"
)

GREETINGS = {
    "hi": "Hi! Ask me about F1 stats, race results, rules, history or what's happening live.",
    "hello": "Hello! I know F1 stats back to 1950, the rules, and live timing. What do you want to know?",
    "hey": "Hey! Ready for F1 questions.",
    "thanks": "You're welcome. Fire away with another F1 question.",
    "thank you": "You're welcome. Anything else F1?",
    "bye": "Bye! Come back for the next race.",
    "who are you": "A local Formula 1 assistant: stats from a race database, rules from a knowledge base, live timing from OpenF1, all answered by a model running on this machine.",
    "what can you do": "Career and season stats, head-to-heads with charts, race results, all-time leaderboards, 'what if' points-system rescoring, live session timing, and explanations of rules, flags, tyres and history.",
}

PERSONAS: dict[str, dict[str, str]] = {
    "default": {"label": "Default", "style": "Be concise, friendly and factual."},
    "engineer": {"label": "Race engineer", "style": "Talk like a race engineer on the pit wall: precise, technical, uses terms like delta, stint, deg, undercut. Short sentences."},
    "newbie": {"label": "Explain like I'm new", "style": "Explain for someone who has never watched F1. No jargon; if you must use a term, explain it in a few words. Use a simple analogy if it helps."},
    "commentator": {"label": "TV commentator", "style": "Answer like an excited TV commentator: energetic and vivid, but still accurate."},
}


@dataclass
class AgentResult:
    answer: str
    sources: list[dict[str, str]]
    confidence: float
    refusal: bool
    tool: str = "rag"
    chart: Optional[dict[str, Any]] = None
    session_id: str = ""
    resolved_question: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


class F1Agent:
    def __init__(self, db_path: Any = None, load_vectors: bool = True, llm: Optional[LocalLLM] = None, live_fetch: Any = None) -> None:
        self.llm = llm or LocalLLM()
        self.memory = ConversationMemory()
        self.live_fetch = live_fetch
        self.last_error: Optional[str] = None

        db_path = db_path or stats_db.DEFAULT_DB_PATH
        self.stats: Optional[StatsTools] = None
        if stats_db.database_exists(db_path):
            self.stats = StatsTools(stats_db.connect(db_path, read_only=True))
        index = self.stats.index if self.stats else None
        latest = self.stats.latest_season() if self.stats else 0
        self.router = Router(index, latest)

        self.db_path = db_path
        self.retriever = self._build_retriever(load_vectors)

    # ----------------------------------------------------------------- setup

    def _build_retriever(self, load_vectors: bool) -> HybridRetriever:
        ids, texts, metas = ingest.build_all_documents(self.db_path)
        chunks = [Chunk(i, t, m) for i, t, m in zip(ids, texts, metas)]
        collection = embedder = None
        if load_vectors:
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer

                client = chromadb.PersistentClient(path=str(ingest.EMBEDDINGS_DIR))
                names = [getattr(c, "name", c) for c in client.list_collections()]
                if ingest.COLLECTION_NAME in names:
                    collection = client.get_collection(ingest.COLLECTION_NAME)
                    embedder = SentenceTransformer(ingest.EMBEDDING_MODEL_NAME)
                else:
                    logger.warning("No vector index yet; run `python -m backend.ingest`. Using BM25 only.")
            except Exception as exc:
                self.last_error = f"vector_init_error: {exc}"
                logger.warning("Vector search unavailable (%s); using BM25 only.", exc)
        return HybridRetriever(chunks, collection, embedder)

    @property
    def index(self) -> Optional[stats_db.EntityIndex]:
        return self.stats.index if self.stats else None

    # ----------------------------------------------------------------- tools

    def run_tool(self, route: Route) -> Optional[ToolResult]:
        a = route.args
        if route.tool == "live":
            return live_summary(self.live_fetch) if self.live_fetch else live_summary()
        if not self.stats:
            return None
        s = self.stats
        try:
            season = int(a["season"]) if a.get("season") else None
        except (TypeError, ValueError):
            season = None
        if route.tool == "driver_career":
            return s.driver_career(a["driver"])
        if route.tool == "driver_season" and season:
            return s.driver_season(a["driver"], season)
        if route.tool == "season_standings":
            return s.season_standings(season or s.latest_season(), a.get("kind", "drivers"))
        if route.tool == "champion" and season:
            return s.champion(season)
        if route.tool == "race_result" and season:
            return s.race_result(season, str(a.get("race", "")))
        if route.tool == "compare":
            return s.compare(a["driver_a"], a["driver_b"], season)
        if route.tool == "leaderboard":
            return s.leaderboard(a.get("stat", "wins"), a.get("by", "driver"), season)
        if route.tool == "constructor_summary":
            return s.constructor_summary(a["constructor"], season)
        if route.tool == "whatif" and season:
            return s.whatif(season, a.get("system", "current"))
        return None

    def _llm_route(self, question: str) -> Optional[Route]:
        """Let the model pick a tool when the rules didn't. Arguments are
        re-validated against the entity index, so a hallucinated name can't
        reach the database."""
        if not self.stats or not self.llm.available():
            return None
        spec = "\n".join(f"- {t['name']}({', '.join(f'{k}: {v}' for k, v in t['args'].items())}): {t['use']}" for t in TOOL_SPECS)
        system = (
            "You route Formula 1 questions to a tool. Reply with JSON only: "
            '{"tool": "<name>", "args": {...}}. Tools:\n' + spec +
            f"\nThe latest season in the database is {self.stats.latest_season()}."
        )
        out = self.llm.chat_json(system, question)
        if not out or out.get("tool") in (None, "rag"):
            return None
        tool, args = str(out["tool"]), dict(out.get("args") or {})
        idx = self.stats.index
        for key in ("driver", "driver_a", "driver_b"):
            if key in args:
                found = idx.find_drivers(str(args[key]))
                if not found:
                    return None
                args[key] = found[0]
        if "constructor" in args:
            found = idx.find_constructors(str(args["constructor"]))
            if not found:
                return None
            args["constructor"] = found[0]
        if tool == "race_result" and "race" in args:
            args["race"] = str(args["race"])
        return Route(tool, args, reason="llm")

    def _stylize(self, text: str, persona: str, question: str) -> str:
        if persona == "default" or not self.llm.available():
            return text
        system = (
            f"{PERSONAS[persona]['style']} Rewrite the FACTS as an answer to the question. "
            "Keep every name and number exactly as given. Do not add facts."
        )
        out = self.llm.chat(system, f"Question: {question}\nFACTS:\n{text}", max_tokens=220)
        # Guard: if the model dropped or invented a number, keep the exact facts.
        if not out or _numbers(out) - _numbers(text) or not _numbers(text) <= _numbers(out) | {"0"}:
            return text
        return out

    # ------------------------------------------------------------------- RAG

    def _is_f1(self, question: str) -> bool:
        q = question.lower()
        if F1_KEYWORDS.search(q):
            return True
        if self.index and (self.index.find_drivers(question) or self.index.find_constructors(question)):
            return True
        return False

    @staticmethod
    def _extractive(question: str, hits: list) -> str:
        q = set(tokenize(question))
        scored = []
        for rank, hit in enumerate(hits):
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", hit.chunk.text) if s.strip()]
            for i, sent in enumerate(sents):
                overlap = len(q & set(tokenize(sent)))
                if overlap:
                    # Take the following sentence too: headings like "VSC: Used for
                    # minor incidents." usually continue with the actual detail.
                    span = sent + (" " + sents[i + 1] if i + 1 < len(sents) else "")
                    scored.append((overlap - rank * 0.1, span))
        if not scored:
            return NO_CONTEXT
        return sorted(scored, key=lambda x: -x[0])[0][1]

    def _rag(self, question: str, persona: str, history: str) -> AgentResult:
        hits = self.retriever.search(question)
        is_f1 = self._is_f1(question)
        top_sim = max((h.vector_sim or 0.0) for h in hits) if hits else 0.0
        # Strong lexical match with an F1 passage also counts (covers the
        # BM25-only mode before the vector index is built).
        q_tokens = set(tokenize(question))
        lexical = bool(hits) and len(q_tokens) >= 2 and len(q_tokens & set(tokenize(hits[0].chunk.text))) >= max(2, int(0.6 * len(q_tokens)))
        if not is_f1 and not lexical and not (self.retriever.has_vectors and top_sim >= MIN_DOMAIN_SIM):
            return AgentResult(NOT_F1, [], 0.0, True)
        if not hits:
            return AgentResult(NO_CONTEXT, [], 0.0, True)

        sources = [
            {"id": str(i), "file": h.chunk.metadata.get("source_file", "unknown"), "snippet": h.chunk.text[:240]}
            for i, h in enumerate(hits, 1)
        ]
        context = "\n\n".join(f"[{s['id']}] ({s['file']}) {h.chunk.text}" for s, h in zip(sources, hits))
        system = (
            "You are an F1 assistant. Answer ONLY from the numbered context. "
            "Cite the passages you used like [1] or [2]. "
            f"If the context doesn't contain the answer, reply exactly: {NO_CONTEXT} "
            f"If the question is not about Formula 1, reply exactly: {NOT_F1} "
            f"{PERSONAS.get(persona, PERSONAS['default'])['style']} Use at most 5 sentences."
        )
        user = (f"Conversation so far:\n{history}\n\n" if history else "") + f"Context:\n{context}\n\nQuestion: {question}"
        answer = self.llm.chat(system, user)
        if answer is None:
            answer = self._extractive(question, hits)
            used = sources[:2]
        else:
            cited = set(re.findall(r"\[(\d+)\]", answer))
            used = [s for s in sources if s["id"] in cited] or sources
        refusal = answer.strip() in (NO_CONTEXT, NOT_F1)
        confidence = round(min(1.0, hits[0].score / 1.5), 3)
        return AgentResult(answer.strip(), [] if refusal else used, confidence, refusal)

    # ------------------------------------------------------------------- ask

    def ask(self, question: str, session_id: Optional[str] = None, persona: str = "default") -> AgentResult:
        self.last_error = None
        session_id = session_id or uuid.uuid4().hex
        persona = persona if persona in PERSONAS else "default"
        question = question.strip()
        state = self.memory.get(session_id)

        greeting = GREETINGS.get(re.sub(r"[^\w\s]", "", question.lower()).strip())
        if greeting:
            return AgentResult(greeting, [], 1.0, False, tool="small_talk", session_id=session_id, resolved_question=question)

        resolved = self.memory.rewrite(state, question, self.index)
        route = self.router.route(resolved)
        if route is None and self.router.looks_like_stats(resolved):
            route = self._llm_route(resolved)

        result: Optional[AgentResult] = None
        if route:
            tool_result = self.run_tool(route)
            if tool_result and tool_result.ok:
                answer = self._stylize(tool_result.text, persona, resolved)
                result = AgentResult(answer, tool_result.sources, 1.0, False, tool_result.tool, tool_result.chart, data=tool_result.data)
            elif tool_result:
                # The tool understood the question but the data says no (e.g. race
                # not run yet). Try the text knowledge base before giving that answer.
                rag = self._rag(resolved, persona, state.history_text())
                result = rag if not rag.refusal else AgentResult(tool_result.text, tool_result.sources, 0.5, False, tool_result.tool)
        if result is None:
            result = self._rag(resolved, persona, state.history_text())

        result.session_id = session_id
        result.resolved_question = resolved
        if not result.refusal:
            self.memory.record(state, question, resolved, result.answer, result.tool, self.index)
        if self.llm.last_error:
            self.last_error = self.llm.last_error
        return result

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "stats_db": bool(self.stats),
            "stats_latest_season": self.stats.latest_season() if self.stats else None,
            "stats_last_sync": stats_db.get_meta(self.stats.conn, "last_sync") if self.stats else None,
            "vector_search": self.retriever.has_vectors,
            "reranker": self.retriever.reranker is not None,
            "chunks_indexed": len(self.retriever.chunks),
            "model_present": bool(self.llm.path and self.llm.path.exists()),
            "last_error": self.last_error,
        }
