from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.rag import get_service, reload_service

app = FastAPI(title="F1 Chatbot API")
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    refusal: bool


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
    return {
        "status": "reloaded",
        "health": service.health(),
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        return AskResponse(
            answer="Please provide a Formula 1 question.",
            sources=[],
            confidence=0.0,
            refusal=True,
        )

    result = get_service().ask(question)
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        refusal=result.refusal,
    )
