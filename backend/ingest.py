from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings" / "chroma"
COLLECTION_NAME = "f1_knowledge_v2"
EMBEDDING_MODEL_NAME = os.getenv("F1_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 550
CHUNK_OVERLAP = 90

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS = {
    "drivers": {"driver", "drivers", "champion", "champions"},
    "teams": {"team", "teams", "grid", "constructor", "constructors"},
    "rules": {"rule", "rules", "regulation", "regulations", "parc ferme", "track limits"},
    "flags": {"flag", "flags", "penalty", "penalties", "steward"},
    "tyres": {"tyre", "tyres", "tire", "tires", "pirelli", "compound"},
    "history": {"history", "historical", "began", "1950", "era"},
    "weekend_format": {"weekend", "qualifying", "sprint", "practice", "race"},
    "champions": {"champion", "champions", "title", "titles", "world champion"},
}

ERA_PATTERNS = (
    ("ground_effect", re.compile(r"\b2022\+\b|\bground effect\b", re.IGNORECASE)),
    ("hybrid", re.compile(r"\bhybrid\b|\b2014\b|\b2015\b|\b2016\b|\b2017\b|\b2018\b|\b2019\b|\b2020\b", re.IGNORECASE)),
    ("modern", re.compile(r"\bmodern\b|\bcurrent\b|\b202[0-9]\b", re.IGNORECASE)),
    ("1950s", re.compile(r"\b195[0-9]\b|\b1950s\b", re.IGNORECASE)),
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def infer_topic(file_path: Path) -> str:
    return file_path.stem.replace("f1_", "").replace("-", "_")


def split_paragraphs(text: str) -> list[str]:
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks = [normalize_text(block) for block in raw_blocks]
    return [block for block in blocks if block]


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.endswith(":"):
        return True
    if len(stripped) <= 80 and re.match(r"^[A-Z][A-Za-z0-9'()\/+,&\-\s]+$", stripped):
        return True
    return False


def split_structured_blocks(text: str) -> list[tuple[Optional[str], str]]:
    lines = [line.rstrip() for line in text.splitlines()]
    blocks: list[tuple[Optional[str], str]] = []
    current_heading: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        block = normalize_text("\n".join(current_lines))
        if block:
            blocks.append((current_heading, block))
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        if is_heading(stripped):
            flush()
            current_heading = stripped.rstrip(":")
            continue

        current_lines.append(stripped)

    flush()
    return blocks


def infer_category(text: str, topic: str) -> str:
    normalized = f"{topic} {text}".lower()
    best_category = topic
    best_hits = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in normalized)
        if hits > best_hits:
            best_hits = hits
            best_category = category
    return best_category


def infer_entity(heading: Optional[str], block: str) -> str:
    if heading:
        return heading

    first_line = block.split(".")[0].strip()
    if ":" in first_line:
        return first_line.split(":", 1)[0].strip()
    if len(first_line.split()) <= 6:
        return first_line
    return "general"


def infer_season(text: str) -> str:
    lowered = text.lower()
    years = re.findall(r"\b(19[5-9]\d|20\d{2})\b", text)
    if not years:
        if "current" in lowered or "modern" in lowered:
            return "current"    
        return "historical"

    unique_years = sorted(set(years))
    if "current" in lowered:
        return f"current,{unique_years[-1]}"
    if len(unique_years) == 1:
        return unique_years[0]
    if len(unique_years) <= 3:
        return ",".join(unique_years)
    return f"{unique_years[0]}-{unique_years[-1]}"


def infer_era(text: str) -> str:
    for label, pattern in ERA_PATTERNS:
        if pattern.search(text):
            return label
    return "general"


def chunk_block(block: str, chunk_size: int, overlap: int, heading: Optional[str] = None) -> list[str]:
    prefix = f"{heading}: " if heading else ""
    if len(prefix) + len(block) <= chunk_size:
        return [f"{prefix}{block}".strip()]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(block):
        end = min(len(block), start + chunk_size)
        segment = block[start:end].strip()
        if segment:
            chunks.append(f"{prefix}{segment}".strip())
        if end >= len(block):
            break
        start += step

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[Optional[str], str, str]]:
    chunks: list[tuple[Optional[str], str, str]] = []
    for heading, block in split_structured_blocks(text):
        entity = infer_entity(heading, block)
        for chunk in chunk_block(block, chunk_size, overlap, heading=heading):
            chunks.append((heading, entity, chunk))
    return chunks


def build_documents(data_dir: Path) -> tuple[list[str], list[str], list[dict]]:
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    txt_files = sorted(data_dir.glob("*.txt"))

    for file_path in txt_files:
        raw_text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(raw_text)
        topic = infer_topic(file_path)
        category = infer_category(raw_text, topic)

        version_date = datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=timezone.utc
        ).date().isoformat()

        for idx, (heading, entity, chunk) in enumerate(chunks):
            short_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{file_path.stem}:{idx}:{short_hash}"

            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append(
                {
                    "source_file": file_path.name,
                    "topic": topic,
                    "category": infer_category(chunk, category),
                    "entity": entity,
                    "heading": heading or "",
                    "season": infer_season(chunk),
                    "era": infer_era(chunk),
                    "chunk_index": idx,
                    "version_date": version_date,
                }
            )

    return ids, texts, metadatas


STATS_SOURCE = "season_summaries (stats DB)"


def build_stats_documents(db_path=None) -> tuple[list[str], list[str], list[dict]]:
    """One readable summary per season from the stats DB, so narrative questions
    ("who won races for Brawn in 2009?") can be answered by retrieval too."""
    from backend import stats_db

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    db_path = db_path or stats_db.DEFAULT_DB_PATH
    if not stats_db.database_exists(db_path):
        return ids, texts, metadatas

    conn = stats_db.connect(db_path, read_only=True)
    index = stats_db.EntityIndex(conn)
    seasons = [r["season"] for r in conn.execute("SELECT DISTINCT season FROM races ORDER BY season")]
    for season in seasons:
        lines: list[str] = []
        champ = conn.execute(
            "SELECT driver_id, points, wins FROM driver_standings WHERE season=? AND position=1", (season,)
        ).fetchone()
        if champ:
            lines.append(
                f"{season} Drivers' Championship leader/champion: {index.driver_name(champ['driver_id'])} "
                f"with {champ['points']:g} points and {champ['wins']} wins."
            )
        cchamp = conn.execute(
            "SELECT constructor_id, points FROM constructor_standings WHERE season=? AND position=1", (season,)
        ).fetchone()
        if cchamp:
            lines.append(f"{season} Constructors' leader/champion: {index.constructor_name(cchamp['constructor_id'])}.")
        winners = conn.execute(
            "SELECT ra.round, ra.name, r.driver_id, r.constructor_id FROM races ra "
            "JOIN results r ON r.season=ra.season AND r.round=ra.round AND r.position=1 "
            "WHERE ra.season=? ORDER BY ra.round",
            (season,),
        ).fetchall()
        for w in winners:
            lines.append(
                f"Round {w['round']} {w['name']} {season} won by {index.driver_name(w['driver_id'])} "
                f"({index.constructor_name(w['constructor_id'])})."
            )
        if not lines:
            continue
        # Several races per chunk keeps chunks retrievable but focused.
        for part, start in enumerate(range(0, len(lines), 6)):
            text = f"{season} season: " + " ".join(lines[start:start + 6])
            ids.append(f"season_{season}:{part}")
            texts.append(text)
            metadatas.append({
                "source_file": STATS_SOURCE,
                "topic": "season_summary",
                "category": "history",
                "entity": f"{season} season",
                "heading": f"{season} season",
                "season": str(season),
                "era": infer_era(text),
                "chunk_index": part,
                "version_date": stats_db.get_meta(conn, "last_sync") or "",
            })
    conn.close()
    return ids, texts, metadatas


def build_all_documents(db_path=None) -> tuple[list[str], list[str], list[dict]]:
    ids, texts, metadatas = build_documents(DATA_DIR)
    s_ids, s_texts, s_meta = build_stats_documents(db_path)
    return ids + s_ids, texts + s_texts, metadatas + s_meta


def ingest() -> None:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for ingestion. Install requirements.txt first."
        ) from exc

    ids, texts, metadatas = build_all_documents()
    if not ids:
        print("No .txt files found in /data. Nothing to ingest.")
        return

    print(f"Preparing {len(texts)} chunks ({len(sorted(DATA_DIR.glob('*.txt')))} text files + season summaries)...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))

    # Rebuild the collection on each ingest to avoid stale chunks surviving
    # file edits, renames, or deletions in data/.
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection %s before rebuild.", COLLECTION_NAME)
    except Exception:
        logger.info("No existing collection %s to delete before rebuild.", COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 64
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=vectors[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    print("Ingestion complete.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embeddings path: {EMBEDDINGS_DIR}")
    print(f"Total chunks indexed: {len(ids)}")


if __name__ == "__main__":
    ingest()
