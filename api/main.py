
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.db import Base, engine, get_db
from api.models import ChatHistory
from agents.orchestrator import AgentOrchestrator
from rag.vector_store import VectorStore


# Ensure database tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Personal Knowledge Assistant API",
    description="Multi-agent RAG API with PostgreSQL chat history",
    version="1.0.0",
)

# Shared vector index, separate conversation memory per session
_vector_store = None
_orchestrators: dict[str, AgentOrchestrator] = {}


def get_session_orchestrator(session_id: str) -> AgentOrchestrator:
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    if session_id not in _orchestrators:
        _orchestrators[session_id] = AgentOrchestrator(_vector_store)

    return _orchestrators[session_id]


class ChatCreate(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=10000)
    answer: str = Field(min_length=1)


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=10000)
    filters: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    id: int
    session_id: str
    question: str
    answer: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    citations: list[dict[str, Any]]
    history_id: int


@app.get("/")
def root():
    return {
        "message": "Personal Knowledge Assistant API",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db),
):
    try:
        orchestrator = get_session_orchestrator(request.session_id)

        result = orchestrator.query(
            request.question,
            filters=request.filters,
        )

        if result.get("error"):
            if result["error"] == "No documents uploaded":
                raise HTTPException(
                    status_code=400,
                    detail="Please upload documents before asking questions.",
                )

            raise HTTPException(
                status_code=502,
                detail="The RAG pipeline could not process this question.",
            )

        answer = result["answer"]
        citations = result.get("citations", [])

        record = ChatHistory(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return AskResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            citations=citations,
            history_id=record.id,
        )

    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the request.",
        )


@app.post("/history", response_model=ChatResponse, status_code=201)
def save_chat(
    chat: ChatCreate,
    db: Session = Depends(get_db),
):
    record = ChatHistory(
        session_id=chat.session_id,
        question=chat.question,
        answer=chat.answer,
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/history/{session_id}", response_model=list[ChatResponse])
def get_history(
    session_id: str,
    db: Session = Depends(get_db),
):
    return (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.created_at.asc(), ChatHistory.id.asc())
        .all()
    )
