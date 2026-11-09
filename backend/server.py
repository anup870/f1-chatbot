from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.agent import PERSONAS
from backend.live import live_summary
from backend.rag import get_service, reload_service
from backend.tools import POINT_SYSTEMS

app = FastAPI(title="F1 Chatbot API")
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    persona: str = "default"


class Source(BaseModel):
    id: str
    file: str
    snippet: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    refusal: bool
    tool: str
    chart: Optional[dict[str, Any]] = None
    session_id: str
    resolved_question: str


class SessionRequest(BaseModel):
    session_id: str


@app.get("/")
def read_root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return JSONResponse({"message": "F1 Chatbot API is running. Use POST /ask."})


@app.get("/health")
def health() -> dict:
    return get_service().health()


@app.post("/reload")
def reload_runtime() -> dict:
    service = reload_service()
    return {"status": "reloaded", "health": service.health()}


@app.get("/personas")
def personas() -> dict:
    return {key: value["label"] for key, value in PERSONAS.items()}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        return AskResponse(
            answer="Please provide a Formula 1 question.", sources=[], confidence=0.0, refusal=True,
            tool="none", session_id=payload.session_id or "", resolved_question="",
        )
    result = get_service().ask(question, session_id=payload.session_id, persona=payload.persona)
    return AskResponse(
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
        confidence=result.confidence,
        refusal=result.refusal,
        tool=result.tool,
        chart=result.chart,
        session_id=result.session_id,
        resolved_question=result.resolved_question,
    )


@app.post("/reset")
def reset_session(payload: SessionRequest) -> dict:
    get_service().memory.reset(payload.session_id)
    return {"status": "reset"}


@app.get("/live")
def live() -> dict:
    result = live_summary()
    return {"ok": result.ok, "text": result.text, **result.data}


@app.get("/whatif")
def whatif(season: int, system: str = "current") -> dict:
    service = get_service()
    if not service.stats:
        return {"ok": False, "text": "Stats database not built yet. Run: python -m backend.fetch_stats"}
    result = service.stats.whatif(season, system)
    return {"ok": result.ok, "text": result.text, "chart": result.chart, "systems": {k: v["label"] for k, v in POINT_SYSTEMS.items()}}
