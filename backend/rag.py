
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / "chroma"
MODEL_DIR = BASE_DIR / "model"
COLLECTION_NAME = "f1_knowledge_v1"

EMBEDDING_MODEL_NAME = os.getenv("F1_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("F1_TOP_K", "6"))
FINAL_CONTEXT_CHUNKS = int(os.getenv("F1_FINAL_CONTEXT_CHUNKS", "4"))
MIN_CONTEXT_SCORE = float(os.getenv("F1_MIN_CONTEXT_SCORE", "0.35"))
MAX_CONTEXT_CHARS = int(os.getenv("F1_MAX_CONTEXT_CHARS", "2500"))
MAX_TOKENS = int(os.getenv("F1_MAX_TOKENS", "96"))
TEMPERATURE = float(os.getenv("F1_TEMPERATURE", "0.1"))
MIN_DOMAIN_SCORE = float(os.getenv("F1_MIN_DOMAIN_SCORE", "0.18"))

logger = logging.getLogger(__name__)

F1_KEYWORDS = {
    "f1",
    "formula",
    "grand prix",
    "gp",
    "fia",
    "qualifying",
    "constructor",
    "constructors",
    "driver",
    "drivers",
    "pit",
    "pit stop",
    "drs",
    "ers",
    "tyre",
    "tyres",
    "tire",
    "tires",
    "grid",
    "steward",
    "stewards",
    "safety car",
    "vsc",
    "verstappen",
    "hamilton",
    "leclerc",
    "norris",
    "ferrari",
    "mercedes",
    "red bull",
    "mclaren",
    "alpine",
    "aston martin",
    "williams",
    "haas",
    "sauber",
}

GREETING_RESPONSES = {
    "hi": "Hi. I can help with Formula 1 history, rules, teams, drivers, and race formats.",
    "hello": "Hello. Ask me any Formula 1 question and I will answer from the local F1 knowledge base.",
    "hey": "Hey. I am ready for Formula 1 questions.",
    "good morning": "Good morning. I am ready to help with Formula 1 questions.",
    "good afternoon": "Good afternoon. Ask me anything about Formula 1.",
    "good evening": "Good evening. I can help with Formula 1 questions.",
    "thanks": "You are welcome. If you want, ask me another Formula 1 question.",
    "thank you": "You are welcome. I am here for Formula 1 questions.",
    "bye": "Bye. Come back anytime with a Formula 1 question.",
    "goodbye": "Goodbye. I will be here when you want more Formula 1 help.",
    "how are you": "I am doing well and ready to talk Formula 1.",
    "who are you": "I am a local Formula 1 chatbot powered by retrieval and a local language model.",
}

QUESTION_WORDS = {
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "tell",
    "explain",
    "list",
    "give",
}


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    score: float
    metadata: dict[str, Any]


@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    confidence: float
    refusal: bool


class RAGService:
    def __init__(self) -> None:
        self._embedder = None
        self._collection = None
        self._llm = None
        self._last_error: Optional[str] = None

    def _lazy_init_retrieval(self) -> None:
        if self._embedder is not None and self._collection is not None:
            return

        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for retrieval. Install requirements.txt before running the bot."
            ) from exc

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
        self._collection = client.get_or_create_collection(name=COLLECTION_NAME)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Retrieval initialized with collection %s.", COLLECTION_NAME)

    def _lazy_init_llm(self) -> None:
        if self._llm is not None:
            return

        model_path = os.getenv("F1_MODEL_PATH", "").strip()
        if model_path:
            gguf_path = Path(model_path)
        else:
            gguf_files = sorted(MODEL_DIR.glob("*.gguf"))
            gguf_path = gguf_files[0] if gguf_files else Path()

        if not gguf_path or not gguf_path.exists():
            self._llm = None
            return

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "Missing llama-cpp-python. Install requirements.txt before running generation."
            ) from exc

        self._llm = Llama(
            model_path=str(gguf_path),
            n_ctx=2048,
            n_threads=max((os.cpu_count() or 2) - 1, 1),
            verbose=False,
        )
        logger.info("Loaded GGUF model from %s.", gguf_path)

    @staticmethod
    def _has_f1_keyword(question: str) -> bool:
        q = question.lower().strip()
        if not q:
            return False
        return any(token in q for token in F1_KEYWORDS)

    @staticmethod
    def _normalize_small_talk(question: str) -> str:
        return re.sub(r"[^\w\s]", "", question.lower()).strip()

    def _small_talk_response(self, question: str) -> Optional[str]:
        normalized = self._normalize_small_talk(question)
        return GREETING_RESPONSES.get(normalized)

    def _embed_query(self, question: str) -> list[float]:
        assert self._embedder is not None
        embedding = self._embedder.encode([question], normalize_embeddings=True)[0]
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    @staticmethod
    def _extract_query_signals(question: str) -> dict[str, str]:
        lowered = question.lower().strip()
        season_match = re.findall(r"\b(19[5-9]\d|20\d{2})\b", question)

        category = ""
        if any(term in lowered for term in ("driver", "drivers", "champion", "champions", "won title")):
            category = "drivers"
        elif any(term in lowered for term in ("team", "teams", "constructor", "constructors", "grid")):
            category = "teams"
        elif any(term in lowered for term in ("rule", "rules", "regulation", "regulations", "parc ferme", "track limits")):
            category = "rules"
        elif any(term in lowered for term in ("flag", "flags", "penalty", "penalties", "yellow flag", "red flag")):
            category = "flags"
        elif any(term in lowered for term in ("tyre", "tyres", "tire", "tires", "compound")):
            category = "tyres"
        elif any(term in lowered for term in ("weekend", "qualifying", "sprint", "practice")):
            category = "weekend_format"
        elif any(term in lowered for term in ("history", "historical", "when did", "began")):
            category = "history"

        entity = ""
        title_words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", question)
        filtered_title_words = [
            word for word in title_words if word.lower() not in QUESTION_WORDS
        ]
        if filtered_title_words:
            entity = max(filtered_title_words, key=len)

        season = season_match[0] if season_match else ""
        return {
            "category": category,
            "entity": entity.lower(),
            "season": season,
            "wants_current": "current" if "current" in lowered else "",
            "wants_driver_list": "drivers" if "current drivers" in lowered else "",
        }

    @staticmethod
    def _rerank_score(
        base_score: float,
        question: str,
        chunk_text: str,
        metadata: dict[str, Any],
        signals: dict[str, str],
    ) -> float:
        score = base_score
        lowered = question.lower()
        chunk_lowered = chunk_text.lower()
        category = str(metadata.get("category", "")).lower()
        entity = str(metadata.get("entity", "")).lower()
        heading = str(metadata.get("heading", "")).lower()
        season = str(metadata.get("season", "")).lower()
        era = str(metadata.get("era", "")).lower()

        if signals["category"] and category == signals["category"]:
            score += 0.18
        if signals["entity"] and signals["entity"] in entity:
            score += 0.25
        if signals["entity"] and signals["entity"] in heading:
            score += 0.2
        if signals["season"] and signals["season"] in season:
            score += 0.2
        if "current" in lowered and ("current" in season or "current" in heading):
            score += 0.12
        if signals["wants_driver_list"] and "drivers:" in chunk_lowered:
            score += 0.2
        if signals["wants_driver_list"] and category == "teams":
            score += 0.08
        if "modern" in lowered and era == "modern":
            score += 0.1
        if "hybrid" in lowered and era == "hybrid":
            score += 0.1
        if "ground effect" in lowered and era == "ground_effect":
            score += 0.1

        return score

    def retrieve_context(self, question: str) -> list[RetrievedChunk]:
        self._lazy_init_retrieval()
        assert self._collection is not None

        query_vector = self._embed_query(question)
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=max(TOP_K, FINAL_CONTEXT_CHUNKS),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            self._last_error = f"retrieval_error: {exc}"
            logger.exception("Retrieval query failed.")
            return []

        docs = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        signals = self._extract_query_signals(question)

        ranked: list[RetrievedChunk] = []
        for doc, metadata, distance in zip(docs, metadatas, distances):
            if not doc:
                continue
            metadata_dict = dict(metadata or {})
            base_score = max(0.0, 1.0 - float(distance))
            score = self._rerank_score(base_score, question, doc, metadata_dict, signals)
            source_file = str(metadata_dict.get("source_file", "unknown"))
            ranked.append(
                RetrievedChunk(
                    text=doc,
                    source_file=source_file,
                    score=score,
                    metadata=metadata_dict,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)

        filtered = [item for item in ranked if item.score >= MIN_CONTEXT_SCORE]
        if filtered:
            return filtered[:FINAL_CONTEXT_CHUNKS]

        # Avoid hard failure from a slightly aggressive threshold. If we retrieved
        # something meaningful, pass the best chunks through and let downstream
        # domain checks decide whether to refuse.
        return ranked[:FINAL_CONTEXT_CHUNKS]

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        total_chars = 0
        for index, chunk in enumerate(chunks, start=1):
            block = f"[{index}] Source={chunk.source_file}\n{chunk.text.strip()}\n"
            if total_chars + len(block) > MAX_CONTEXT_CHARS:
                break
            lines.append(block)
            total_chars += len(block)
        return "\n".join(lines).strip()

    @staticmethod
    def _fallback_answer(question: str, context: str) -> str:
        del question
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        content_lines = [line for line in lines if not re.match(r"^\[\d+\]\s+Source=", line)]
        merged = " ".join(content_lines).strip()
        if not merged:
            return "I do not have enough F1 context to answer that."

        first_sentence = re.split(r"(?<=[.!?])\s+", merged)[0].strip()
        return first_sentence or "I do not have enough F1 context to answer that."

    @staticmethod
    def _sanitize_answer(text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return ""

        clean = re.sub(r"\[\d+\]\s*Source=[^.\n]+\.?\s*", "", clean).strip()

        refusal_markers = [
            "I only answer Formula 1 questions.",
            "I do not have enough F1 context to answer that.",
        ]
        for marker in refusal_markers:
            if marker in clean:
                clean = clean.split(marker, 1)[0].strip()

        # Keep at most two sentences to avoid run-on responses on small local models.
        parts = re.split(r"(?<=[.!?])\s+", clean)
        parts = [part.strip() for part in parts if part.strip()]
        if not parts:
            return ""

        deduped: list[str] = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
            if len(deduped) >= 2:
                break
        return " ".join(deduped).strip()

    def generate_answer(self, question: str, context: str) -> str:
        try:
            self._lazy_init_llm()
        except Exception as exc:
            self._last_error = f"model_load_error: {exc}"
            logger.exception("Model initialization failed.")
            return self._fallback_answer(question, context)

        if self._llm is None:
            return self._fallback_answer(question, context)

        prompt = (
            "You are an F1-only assistant.\n"
            "Rules:\n"
            "1) Answer ONLY using the provided context.\n"
            "2) If the question is not Formula 1 related, reply exactly: "
            "'I only answer Formula 1 questions.'\n"
            "3) If context is insufficient, reply exactly: "
            "'I do not have enough F1 context to answer that.'\n"
            "4) Keep answers concise and factual.\n"
            "5) Output at most 2 short sentences.\n"
            "6) Do not repeat any sentence.\n\n"
            f"Question:\n{question.strip()}\n\n"
            f"Context:\n{context.strip() or 'No context.'}\n\n"
            "Answer:\n"
        )

        try:
            response = self._llm.create_completion(
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.9,
                repeat_penalty=1.15,
                stop=[
                    "\n\nQuestion:",
                    "\n\nContext:",
                    "I only answer Formula 1 questions.",
                    "I do not have enough F1 context to answer that.",
                ],
            )
        except Exception as exc:
            self._last_error = f"generation_error: {exc}"
            logger.exception("Model generation failed.")
            return self._fallback_answer(question, context)

        text = (response.get("choices") or [{}])[0].get("text", "")
        clean = self._sanitize_answer(text)
        return clean or "I do not have enough F1 context to answer that."

    def ask(self, question: str) -> RAGResult:
        self._last_error = None
        small_talk = self._small_talk_response(question)
        if small_talk:
            return RAGResult(
                answer=small_talk,
                sources=[],
                confidence=1.0,
                refusal=False,
            )

        chunks = self.retrieve_context(question)
        top_score = chunks[0].score if chunks else 0.0
        has_keyword = self._has_f1_keyword(question)

        if not chunks:
            return RAGResult(
                answer=(
                    "I do not have enough F1 context to answer that."
                    if has_keyword
                    else "I only answer Formula 1 questions."
                ),
                sources=[],
                confidence=0.0,
                refusal=True,
            )

        if not has_keyword and top_score < MIN_DOMAIN_SCORE:
            return RAGResult(
                answer="I only answer Formula 1 questions.",
                sources=[],
                confidence=round(top_score, 3),
                refusal=True,
            )

        context = self._format_context(chunks)
        answer = self.generate_answer(question, context)
        confidence = round(sum(chunk.score for chunk in chunks) / len(chunks), 3)

        refusal = answer in {
            "I only answer Formula 1 questions.",
            "I do not have enough F1 context to answer that.",
        }
        sources = sorted({chunk.source_file for chunk in chunks})

        return RAGResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            refusal=refusal,
        )

    def health(self) -> dict[str, Any]:
        model_loaded = self._llm is not None
        retrieval_ready = self._embedder is not None and self._collection is not None

        model_exists = bool(os.getenv("F1_MODEL_PATH", "").strip()) or any(
            MODEL_DIR.glob("*.gguf")
        )

        return {
            "status": "ok",
            "retrieval_ready": retrieval_ready,
            "model_loaded": model_loaded,   
            "model_present": model_exists,
            "last_error": self._last_error,
            "collection": COLLECTION_NAME,
            "embeddings_path": str(EMBEDDINGS_DIR),
        }


_SERVICE: Optional[RAGService] = None


def get_service() -> RAGService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = RAGService()
    return _SERVICE


def reload_service() -> RAGService:
    global _SERVICE
    _SERVICE = RAGService()
    return _SERVICE


def load_model() -> None:
    get_service()._lazy_init_llm()


def retrieve_context(question: str) -> str:
    chunks = get_service().retrieve_context(question)
    return get_service()._format_context(chunks)


def generate_answer(question: str, context: str) -> str:
    return get_service().generate_answer(question, context)
