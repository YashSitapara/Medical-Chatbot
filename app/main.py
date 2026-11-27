"""FastAPI application: routes, startup wiring and SSE streaming."""

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi import Path as PathParam
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog
from typing_extensions import Annotated

from app import __version__
from app.config import Settings, get_settings
from app.log_config import configure_logging
from app.errors import DependencyUnavailableError, RequestTimeoutError
from app.memory import ChatMemory, build_memory
from app.rag import RAGService
from app.rate_limit import FixedWindowRateLimiter, build_rate_limiter
from app.readiness import check_readiness
from app.request_context import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    current_request_id,
    mark_sensitive,
    set_outcome,
    set_session,
)
from app.schemas import (
    SESSION_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    HistoryResponse,
    ReadyResponse,
)
from app.tracing import NoopTracer, RequestTrace, Tracer, build_tracer

logger = structlog.get_logger("medical_rag")

FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
RATE_LIMIT_HEADERS = ["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
DEPENDENCY_RETRY_AFTER_SECONDS = 5
_HTTP_CODES = {404: "not_found", 405: "method_not_allowed", 429: "rate_limited"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.log_content:
        logger.warning(
            "log_content_enabled",
            hint="LOG_CONTENT=true writes user questions to the logs. Local development only.",
        )
    # Tests may pre-populate app.state with fakes; only build real services if missing.
    # Memory first: the Redis check is instant, the RAG load (model + Pinecone) is slow.
    if getattr(app.state, "memory", None) is None:
        app.state.memory = build_memory(settings)
    memory: ChatMemory = app.state.memory
    await memory.ping()  # fail fast: refuse to start if Redis is unreachable
    logger.info("memory_ready", memory_backend=memory.backend)

    if getattr(app.state, "rag", None) is None:
        logger.info(
            "rag_loading",
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model_name,
            index=settings.pinecone_index_name,
        )
        app.state.rag = RAGService(settings)

    if not hasattr(app.state, "rate_limiter"):  # tests may inject one (or None)
        app.state.rate_limiter = build_rate_limiter(settings)
    limiter: FixedWindowRateLimiter | None = app.state.rate_limiter
    logger.info(
        "rate_limit_ready",
        enabled=limiter is not None,
        backend=type(limiter).__name__ if limiter else None,
        per_session=settings.rate_limit_per_minute,
        per_ip=settings.rate_limit_per_ip_per_minute,
    )

    if not hasattr(app.state, "tracer"):  # tests may inject one
        app.state.tracer = build_tracer(settings)
    tracer: Tracer = app.state.tracer
    logger.info(
        "tracing_ready",
        enabled=tracer.enabled,
        host=settings.langfuse_base_url if tracer.enabled else None,
        environment=settings.langfuse_environment if tracer.enabled else None,
        sample_rate=settings.langfuse_sample_rate if tracer.enabled else None,
    )

    workers = int(os.environ.get("WEB_CONCURRENCY", "1") or 1)
    if memory.backend == "memory" and workers > 1:
        logger.warning(
            "inmemory_with_multiple_workers",
            workers=workers,
            hint="Each worker keeps its own chat history, so follow-ups may lose context. "
            "Set MEMORY_BACKEND=redis to share it.",
        )
    logger.info("startup_complete", version=__version__)
    yield
    await memory.close()
    if limiter is not None:
        await limiter.close()
    await asyncio.to_thread(tracer.shutdown)  # flush buffered spans and feedback scores


def get_rag(request: Request) -> RAGService:
    return request.app.state.rag


def get_memory(request: Request) -> ChatMemory:
    return request.app.state.memory


def get_tracer(request: Request) -> Tracer:
    return getattr(request.app.state, "tracer", None) or NoopTracer()


def _error_body(detail, code: str) -> dict:
    """Every error response has the same shape, and always carries the request ID."""
    return {"detail": detail, "code": code, "request_id": current_request_id()}


async def enforce_rate_limit(request: Request, body: ChatRequest, response: Response) -> None:
    """Dependency for the chat routes only: per-session and per-IP fixed-window limits.

    The IP is `request.client.host`. uvicorn already replaces it with the X-Forwarded-For
    address, but ONLY when the direct peer is a trusted proxy (FORWARDED_ALLOW_IPS), so a
    client connecting directly can't fake its IP with that header.
    """
    limiter: FixedWindowRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    if body.session_id:
        set_session(body.session_id)
    decision = await limiter.check(body.session_id, _client_ip(request))
    _apply_rate_limit(request, response, decision)


async def enforce_feedback_rate_limit(request: Request, response: Response) -> None:
    """POST /api/feedback: its own per-IP counter, so rating answers doesn't use up chat quota."""
    limiter: FixedWindowRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    limit = get_rag(request).settings.feedback_rate_limit_per_ip_per_minute
    decision = await limiter.check_scoped("feedback", _client_ip(request), limit)
    _apply_rate_limit(request, response, decision)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _apply_rate_limit(request: Request, response: Response, decision) -> None:
    headers = decision.headers()
    if not decision.allowed:
        set_outcome("rate_limited")
        # Neither the IP nor the counter key is logged: an IP address is personal data.
        logger.info("rate_limited", limited_by=decision.limited_by, limit=decision.limit, retry_after=decision.retry_after)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit exceeded. Try again in {decision.retry_after} seconds.",
            headers=headers,
        )
    response.headers.update(headers)  # merged into /api/chat's JSON response
    request.state.rate_limit_headers = headers  # StreamingResponse is returned directly: copied there


def _outcome_or_ok() -> str:
    from app.request_context import current

    ctx = current()
    return (ctx.outcome if ctx and ctx.outcome else None) or "ok"


def _request_started(request: Request, settings: Settings, body: ChatRequest, session_id: str, stream: bool) -> None:
    """Bind the session to this request's logs and emit `request_started` (metadata only)."""
    set_session(session_id)
    mark_sensitive(body.message)  # scrubbed from every later log line, e.g. exception messages
    extra = {"question": body.message} if settings.log_content else {}
    logger.info(
        "request_started",
        route=request.scope["route"].path,
        method=request.method,
        stream=stream,
        new_session=body.session_id is None,
        question_chars=len(body.message),
        **extra,
    )


def _trace_output(answer: str, citations) -> dict:
    return {"answer": answer, "cited": [c.id for c in citations if c.cited]}


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# Path parameter version of ChatRequest.session_id: anything that isn't a UUID4 is a 422.
SessionId = Annotated[str, PathParam(pattern=SESSION_ID_PATTERN, description="Server-issued session id (UUID4).")]


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    app = FastAPI(
        title="Medical RAG Chatbot",
        version=__version__,
        description="Question answering over medical documents with citations, streaming and chat memory.",
        lifespan=lifespan,
    )

    # --- Error mapping: 503 dependency, 504 overall cap, 500 bug (in the middleware) ---

    @app.exception_handler(DependencyUnavailableError)
    async def dependency_unavailable(request: Request, exc: DependencyUnavailableError) -> JSONResponse:
        cause = exc.__cause__
        logger.warning(
            "dependency_unavailable",
            error=type(exc).__name__,
            cause=type(cause).__name__ if cause else None,
            detail=str(exc),
        )
        set_outcome("dependency_unavailable")
        return JSONResponse(
            _error_body(exc.public_detail, exc.code),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": str(DEPENDENCY_RETRY_AFTER_SECONDS)},
        )

    @app.exception_handler(RequestTimeoutError)
    async def request_timeout(request: Request, exc: RequestTimeoutError) -> JSONResponse:
        logger.warning("request_timeout", limit_seconds=get_settings().request_timeout_seconds)
        set_outcome("timeout")
        return JSONResponse(_error_body(exc.public_detail, exc.code), status_code=status.HTTP_504_GATEWAY_TIMEOUT)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return JSONResponse(_error_body(exc.detail, code), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            _error_body(jsonable_encoder(exc.errors()), "validation_error"),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    if settings.cors_origins:  # default: none; the bundled UI is same-origin and needs no CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[REQUEST_ID_HEADER, *RATE_LIMIT_HEADERS],  # lets browser JS read them
        )
    # Added last = outermost: wraps CORS too, so every response (even preflight) gets an ID.
    app.add_middleware(RequestContextMiddleware)

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

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["meta"],
        responses={503: {"model": ReadyResponse, "description": "A dependency failed or timed out"}},
    )
    async def ready(
        rag: RAGService = Depends(get_rag),
        memory: ChatMemory = Depends(get_memory),
    ) -> JSONResponse:
        """Readiness: Redis ping + Pinecone index stats, each with a time limit. Used by the
        Docker healthcheck. 200 if both pass, else 503 with the failing check marked."""
        result = await check_readiness(memory, rag, rag.settings.ready_timeout_seconds)
        code = status.HTTP_200_OK if result.status == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(result.model_dump(), status_code=code)

    error_responses = {
        429: {"description": "Rate limit exceeded (see Retry-After)"},
        503: {"description": "A dependency (LLM, Pinecone, Redis) is down or timed out"},
        504: {"description": "The request exceeded REQUEST_TIMEOUT_SECONDS"},
    }

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
        tags=["chat"],
        dependencies=[Depends(enforce_rate_limit)],
        responses=error_responses,
    )
    async def chat(
        body: ChatRequest,
        request: Request,
        rag: RAGService = Depends(get_rag),
        memory: ChatMemory = Depends(get_memory),
    ) -> ChatResponse:
        session_id = body.session_id or str(uuid.uuid4())
        _request_started(request, rag.settings, body, session_id, stream=False)
        # One Langfuse trace per request (a no-op when TRACING_ENABLED=false). An exception
        # leaving this block is recorded on the trace by class name, then re-raised.
        with get_tracer(request).request_trace("chat", session_id, current_request_id(), body.message) as trace:
            history = await memory.get(session_id)
            # Dependency errors -> 503 and bugs -> 500 propagate to the handlers / middleware.
            try:
                async with asyncio.timeout(rag.settings.request_timeout_seconds) as cap:
                    answer, citations = await rag.answer(body.message, history)
            except TimeoutError:
                if cap.expired():  # our overall cap, not some inner timeout
                    raise RequestTimeoutError() from None
                raise
            await memory.append(session_id, body.message, answer, trace_id=trace.trace_id)
            trace.finish(output=_trace_output(answer, citations), outcome=_outcome_or_ok())
        return ChatResponse(session_id=session_id, answer=answer, citations=citations, trace_id=trace.trace_id)

    @app.post(
        "/api/chat/stream",
        tags=["chat"],
        response_class=StreamingResponse,
        dependencies=[Depends(enforce_rate_limit)],
        responses=error_responses,
    )
    async def chat_stream(
        body: ChatRequest,
        request: Request,
        rag: RAGService = Depends(get_rag),
        memory: ChatMemory = Depends(get_memory),
    ) -> StreamingResponse:
        """Server-Sent Events: `session`, `citations`, `token`*, then `done` (or `error`)."""
        session_id = body.session_id or str(uuid.uuid4())
        _request_started(request, rag.settings, body, session_id, stream=True)
        history = await memory.get(session_id)  # before streaming starts, so failures are a clean 503

        cap_seconds = rag.settings.request_timeout_seconds
        tracer = get_tracer(request)
        request_id = current_request_id()

        async def events() -> AsyncIterator[str]:
            # Entered inside the generator: Starlette iterates it in ONE task, so the root
            # span's context is attached and detached in the same place.
            with tracer.request_trace("chat-stream", session_id, request_id, body.message) as trace:
                async for chunk in _stream_events(trace):
                    yield chunk

        async def _stream_events(trace: RequestTrace) -> AsyncIterator[str]:
            def error_event(detail: str, code: str, exc: BaseException) -> str:
                trace.finish(outcome=code if code != "internal_error" else "error", error=exc)
                return _sse("error", {**_error_body(detail, code), "trace_id": trace.trace_id})

            yield _sse("session", {"session_id": session_id})
            loop = asyncio.get_running_loop()
            deadline = loop.time() + cap_seconds
            stream = rag.stream(body.message, history)
            # A deadline instead of `async with asyncio.timeout()`: a timeout context can't
            # safely span this generator's `yield`s. Each step gets the time that is left.
            try:
                while True:
                    remaining = deadline - loop.time()
                    try:
                        if remaining <= 0:
                            raise TimeoutError
                        event = await asyncio.wait_for(anext(stream), remaining)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        if loop.time() >= deadline:
                            raise RequestTimeoutError() from None
                        raise
                    if await request.is_disconnected():
                        logger.info("client_disconnected", detail="turn not saved")
                        set_outcome("client_disconnected")
                        trace.finish(outcome="client_disconnected")
                        return
                    if event.type == "citations":
                        yield _sse("citations", [c.model_dump() for c in event.data])
                    elif event.type == "token":
                        yield _sse("token", {"text": event.data})
                    elif event.type == "done":
                        answer = event.data["answer"]
                        await memory.append(session_id, body.message, answer, trace_id=trace.trace_id)
                        trace.finish(output=_trace_output(answer, event.data["citations"]), outcome=_outcome_or_ok())
                        yield _sse(
                            "done",
                            {
                                "session_id": session_id,
                                "answer": answer,
                                "citations": [c.model_dump() for c in event.data["citations"]],
                                "trace_id": trace.trace_id,
                            },
                        )
            # HTTP 200 was already sent: the error event + outcome carry the failure instead.
            # The partial answer is not saved to memory.
            except RequestTimeoutError as exc:
                logger.warning("request_timeout", limit_seconds=cap_seconds)
                set_outcome("timeout")
                yield error_event(exc.public_detail, exc.code, exc)
            except DependencyUnavailableError as exc:
                cause = exc.__cause__
                logger.warning("dependency_unavailable", error=type(exc).__name__,
                               cause=type(cause).__name__ if cause else None, detail=str(exc))
                set_outcome("dependency_unavailable")
                yield error_event(exc.public_detail, exc.code, exc)
            except Exception as exc:
                logger.exception("chat_stream_failed")
                set_outcome("error")
                yield error_event("Something went wrong on our side.", "internal_error", exc)
            finally:
                await stream.aclose()  # stops the LLM stream on timeout/error/disconnect

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        headers.update(getattr(request.state, "rate_limit_headers", {}))
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    @app.post(
        "/api/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["feedback"],
        dependencies=[Depends(enforce_feedback_rate_limit)],
        responses={
            429: {"description": "Rate limit exceeded (see Retry-After)"},
            503: {"description": "Tracing is disabled, or the score could not be queued"},
        },
    )
    async def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse | JSONResponse:
        """👍/👎 on an answer: stored as the `user_feedback` score (1 or -1) on its Langfuse trace.

        Sent in the background (202). Sending again for the same trace replaces the score.
        The trace_id acts as the permission: 128 random bits, only given to the client that asked.
        """
        tracer = get_tracer(request)
        if not tracer.enabled:
            return JSONResponse(
                _error_body("Feedback is unavailable because tracing is disabled.", "tracing_disabled"),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        try:
            tracer.score(body.trace_id, body.rating, body.comment)
        except Exception as exc:
            logger.warning("feedback_failed", error=type(exc).__name__)
            return JSONResponse(
                _error_body("Feedback could not be recorded. Please try again.", "feedback_unavailable"),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        logger.info("feedback_received", feedback_trace_id=body.trace_id, rating=body.rating,
                    has_comment=body.comment is not None)
        return FeedbackResponse()

    @app.get("/api/sessions/{session_id}/history", response_model=HistoryResponse, tags=["sessions"])
    async def get_history(session_id: SessionId, memory: ChatMemory = Depends(get_memory)) -> HistoryResponse:
        set_session(session_id)
        if not await memory.exists(session_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        return HistoryResponse(session_id=session_id, messages=await memory.get(session_id))

    @app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["sessions"])
    async def delete_session(session_id: SessionId, memory: ChatMemory = Depends(get_memory)) -> Response:
        set_session(session_id)
        await memory.clear(session_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
