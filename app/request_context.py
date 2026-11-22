"""Per-request context (request ID, session ID, outcome) and the middleware that sets it.

The context is a small MUTABLE object stored in a ContextVar. A plain
`structlog.contextvars.bind_contextvars` would not work for SSE: StreamingResponse runs
the generator in a child task, which gets a *copy* of the context, so anything bound
there never reaches the middleware that logs `request_finished`. Mutating one shared
object works in both directions.
"""

import json
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = structlog.get_logger("medical_rag.request")

REQUEST_ID_HEADER = "X-Request-ID"
# Reuse a client's ID only if it is short and plain: no newlines/spaces that could forge log lines.
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
# Probes hit these every few seconds; successful ones are logged at DEBUG only.
QUIET_ROUTES = frozenset({"/health", "/ready"})
_MIN_SENSITIVE_CHARS = 3


@dataclass
class RequestContext:
    request_id: str
    session_id: str | None = None
    outcome: str | None = None
    trace_id: str | None = None  # Langfuse trace (app/tracing.py), logged on every later line
    trace: Any = field(default=None, repr=False)  # the RequestTrace the RAG layer adds spans to
    # Exact texts (question, rewritten question) scrubbed from every log line. Never logged.
    sensitive: set[str] = field(default_factory=set, repr=False)


_current: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def current() -> RequestContext | None:
    return _current.get()


def current_request_id() -> str | None:
    ctx = current()
    return ctx.request_id if ctx else None


def set_session(session_id: str) -> None:
    if ctx := current():
        ctx.session_id = session_id


def set_outcome(outcome: str) -> None:
    if ctx := current():
        ctx.outcome = outcome


def mark_sensitive(*texts: str | None) -> None:
    """Register user-derived text so the log redactor replaces it wherever it appears."""
    if ctx := current():
        for text in texts:
            if text and len(text.strip()) >= _MIN_SENSITIVE_CHARS:
                ctx.sensitive.add(text)
                ctx.sensitive.add(text.strip())


def resolve_request_id(incoming: str | None) -> str:
    if incoming and _VALID_REQUEST_ID.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Pure ASGI middleware: request ID in/out, and one `request_finished` line per request.

    Not BaseHTTPMiddleware: its call_next returns when the headers are sent, so an SSE
    response would be "finished" before the first token. Here `request_finished` is logged
    when the LAST body chunk has been sent, so latency covers the whole stream.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        ctx = RequestContext(request_id=request_id)
        token = _current.set(ctx)
        start = time.perf_counter()
        status_code: int | None = None
        response_started = False
        logged = False

        def finish(code: int, outcome: str | None = None, exc_info: bool = False) -> None:
            nonlocal logged
            if logged:
                return
            logged = True
            self._log_finished(scope, ctx, code, start, outcome, exc_info)

        async def send_with_id(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finish(status_code or 500)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            if response_started:  # too late to answer; just record it
                finish(status_code or 500, "exception", exc_info=True)
                raise
            # Answer the 500 ourselves so the client still gets its X-Request-ID.
            finish(500, "exception", exc_info=True)
            body = json.dumps(
                {"detail": "Internal Server Error", "code": "internal_error", "request_id": request_id}
            ).encode()
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (REQUEST_ID_HEADER.lower().encode(), request_id.encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})
        finally:
            # No final body chunk was sent: the client went away mid-response (e.g. SSE).
            finish(status_code or 499, ctx.outcome or "client_disconnected")
            _current.reset(token)

    @staticmethod
    def _log_finished(scope, ctx, status_code, start, outcome, exc_info) -> None:
        route = scope.get("route")
        route_path = getattr(route, "path", None) or "<unmatched>"  # template, never the raw path
        if not (outcome or ctx.outcome):
            outcome = "ok" if status_code < 400 else "client_error" if status_code < 500 else "error"
        outcome = outcome or ctx.outcome
        fields = {
            "route": route_path,
            "method": scope["method"],
            "status_code": status_code,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "outcome": outcome,
        }
        if exc_info:
            logger.error("request_finished", exc_info=True, **fields)
        elif status_code >= 500 or outcome in ("error", "exception"):
            logger.warning("request_finished", **fields)
        elif route_path in QUIET_ROUTES and status_code < 400:
            logger.debug("request_finished", **fields)
        else:
            logger.info("request_finished", **fields)
