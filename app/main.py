"""FastAPI application: routes, startup wiring and SSE streaming."""

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.config import get_settings
from app.memory import ChatMemory, MemoryUnavailableError, build_memory
from app.rag import RAGService
from app.schemas import ChatRequest, ChatResponse, HealthResponse, HistoryResponse

logger = logging.getLogger("medical_rag")

FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        # pid identifies the worker when running with --workers / WEB_CONCURRENCY > 1.
        format="%(asctime)s %(levelname)s [pid %(process)d] %(name)s: %(message)s",
    )
    # Tests may pre-populate app.state with fakes; only build real services if missing.
    # Memory first: the Redis check is instant, the RAG load (model + Pinecone) is slow.
    if getattr(app.state, "memory", None) is None:
        app.state.memory = build_memory(settings)
    memory: ChatMemory = app.state.memory
    await memory.ping()  # fail fast: refuse to start if Redis is unreachable
    logger.info("Chat memory backend: %s", memory.backend)

    if getattr(app.state, "rag", None) is None:
        logger.info(
            "Loading RAG service (provider=%s, model=%s, index=%s)",
            settings.llm_provider,
            settings.llm_model_name,
            settings.pinecone_index_name,
        )
        app.state.rag = RAGService(settings)

    workers = int(os.environ.get("WEB_CONCURRENCY", "1") or 1)
    if memory.backend == "memory" and workers > 1:
        logger.warning(
            "WEB_CONCURRENCY=%d with MEMORY_BACKEND=memory: each worker keeps its own chat history, "
            "so follow-ups may lose context. Set MEMORY_BACKEND=redis to share it.",
            workers,
        )
    logger.info("Startup complete")
    yield
    await memory.close()


def get_rag(request: Request) -> RAGService:
    return request.app.state.rag


def get_memory(request: Request) -> ChatMemory:
    return request.app.state.memory


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Medical RAG Chatbot",
        version="1.1.0",
        description="Question answering over medical documents with citations, streaming and chat memory.",
        lifespan=lifespan,
    )
    @app.exception_handler(MemoryUnavailableError)
    async def memory_unavailable(request: Request, exc: MemoryUnavailableError) -> JSONResponse:
        logger.error("Chat memory unavailable: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Chat memory is temporarily unavailable."},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_INDEX)

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["meta"],
        responses={503: {"description": "Chat memory backend unreachable"}},
    )
    async def health(
        rag: RAGService = Depends(get_rag),
        memory: ChatMemory = Depends(get_memory),
    ) -> HealthResponse:
        # A 503 here takes this instance out of a load balancer / Docker healthcheck rotation.
        await memory.ping()
        s = rag.settings
        return HealthResponse(
            llm_provider=s.llm_provider,
            llm_model=s.llm_model_name,
            index=s.pinecone_index_name,
            memory_backend=memory.backend,
        )

    @app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(
        body: ChatRequest,
        rag: RAGService = Depends(get_rag),
        memory: ChatMemory = Depends(get_memory),
    ) -> ChatResponse:
        session_id = body.session_id or str(uuid.uuid4())
        history = await memory.get(session_id)
        try:
            answer, citations = await rag.answer(body.message, history)
        except Exception:
            logger.exception("Chat request failed")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The assistant is temporarily unavailable.")
        await memory.append(session_id, body.message, answer)
        return ChatResponse(session_id=session_id, answer=answer, citations=citations)

    @app.post("/api/chat/stream", tags=["chat"], response_class=StreamingResponse)
    async def chat_stream(
        body: ChatRequest,
        request: Request,
        rag: RAGService = Depends(get_rag),
        memory: ChatMemory = Depends(get_memory),
    ) -> StreamingResponse:
        """Server-Sent Events: `session`, `citations`, `token`*, then `done` (or `error`)."""
        session_id = body.session_id or str(uuid.uuid4())
        history = await memory.get(session_id)  # before streaming starts, so failures are a clean 503

        async def events() -> AsyncIterator[str]:
            yield _sse("session", {"session_id": session_id})
            try:
                async for event in rag.stream(body.message, history):
                    if await request.is_disconnected():
                        logger.info("Client disconnected; dropping turn for session %s", session_id)
                        return
                    if event.type == "citations":
                        yield _sse("citations", [c.model_dump() for c in event.data])
                    elif event.type == "token":
                        yield _sse("token", {"text": event.data})
                    elif event.type == "done":
                        answer = event.data["answer"]
                        await memory.append(session_id, body.message, answer)
                        yield _sse(
                            "done",
                            {
                                "session_id": session_id,
                                "answer": answer,
                                "citations": [c.model_dump() for c in event.data["citations"]],
                            },
                        )
            except Exception:
                logger.exception("Streaming chat failed")
                yield _sse("error", {"detail": "The assistant is temporarily unavailable."})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions/{session_id}/history", response_model=HistoryResponse, tags=["sessions"])
    async def get_history(session_id: str, memory: ChatMemory = Depends(get_memory)) -> HistoryResponse:
        if not await memory.exists(session_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        return HistoryResponse(session_id=session_id, messages=await memory.get(session_id))

    @app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["sessions"])
    async def delete_session(session_id: str, memory: ChatMemory = Depends(get_memory)) -> Response:
        await memory.clear(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
