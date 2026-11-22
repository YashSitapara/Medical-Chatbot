"""Langfuse tracing: one trace per chat request, plus user feedback scores.

    chat (root span)          question in, answer out, outcome
    ├── condense              transcript + question in, rewritten question out
    ├── retrieve              query in, chunk ids / scores / pages out
    │   ├── retrieve.plain    (condense fallback only)
    │   └── retrieve.combined
    └── generate              full prompt in, answer + token usage out

Only the ROOT is attached to the context (in the route). Children are created with
`start_observation` + `.end()`: the streaming route runs each step in its own task, and a
context-attached span would be attached and detached in different tasks.

Everything sent passes through `mask_pii` first (phone numbers, emails, Aadhaar, PAN).
Tracing must never break a request: every SDK call is wrapped, failures are logged once.
Export happens in a background thread, so a slow or down Langfuse adds no latency.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

from app.config import Settings
from app.request_context import current

if TYPE_CHECKING:
    from langfuse import Langfuse

logger = structlog.get_logger(__name__)

FEEDBACK_SCORE_NAME = "user_feedback"
_ERROR_LOG_INTERVAL = 60.0

# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

# Order matters: Aadhaar (12 digits) before phone numbers, or it would look like one.
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    # Aadhaar: 12 digits, first digit 2-9, optionally grouped 4-4-4.
    (re.compile(r"(?<![\w.])[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\w|\.\d)"), "[AADHAAR]"),
    # PAN: 5 letters, 4 digits, 1 letter (users may type it in lower case).
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE), "[PAN]"),
    # Indian mobile: optional +91 / 91 / 0, then 10 digits starting 6-9 (optionally 5-5 grouped).
    (re.compile(r"(?<![\w.+])(?:\+91[ -]?|91[ -]|0)?[6-9]\d{4}[ -]?\d{5}(?!\w|\.\d)"), "[PHONE]"),
    # Other international numbers: must start with "+" (so doses, years, pages never match).
    (re.compile(r"\+\d{1,3}[ -]?(?:\(\d{1,4}\)[ -]?)?\d{2,4}(?:[ -]?\d{2,4}){1,4}(?!\w|\.\d)"), "[PHONE]"),
]


def mask_text(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def mask_pii(*, data: Any, **_: Any) -> Any:
    """Langfuse `mask` hook: called on every observation's input / output / metadata.

    Strings are masked, containers walked; numbers, booleans and None pass through (scores,
    page numbers). If this raises, the SDK replaces the value with its own fallback mask,
    so a bug here fails closed (nothing unmasked is sent).
    """
    if isinstance(data, str):
        return mask_text(data)
    if isinstance(data, dict):
        return {key: mask_pii(data=value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_pii(data=value) for value in data]
    return data


# ---------------------------------------------------------------------------
# Observation handles (real + no-op), safe to call even if the SDK misbehaves
# ---------------------------------------------------------------------------

_last_error_log = -float("inf")


def _safe(fn, *args, **kwargs):
    global _last_error_log
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # tracing must never break a request
        now = time.monotonic()
        if now - _last_error_log >= _ERROR_LOG_INTERVAL:
            _last_error_log = now
            logger.warning("tracing_error", error=type(exc).__name__)
        return None


class Span:
    """One Langfuse observation. With `obs=None` every method is a no-op."""

    def __init__(self, obs: Any = None) -> None:
        self._obs = obs
        self._ended = False

    def update(self, **fields: Any) -> None:
        fields = {k: v for k, v in fields.items() if v is not None}
        if self._obs is not None and fields:
            _safe(self._obs.update, **fields)

    def end(self, **fields: Any) -> None:
        if self._obs is None or self._ended:
            return
        self._ended = True
        self.update(**fields)
        _safe(self._obs.end)

    def child(self, name: str, as_type: str = "span", **fields: Any) -> Span:
        if self._obs is None:
            return Span()
        return Span(_safe(self._obs.start_observation, name=name, as_type=as_type, **fields))


class RequestTrace:
    """The trace of one chat request. The no-op version has trace_id None."""

    def __init__(self, client: Langfuse | None = None, root: Any = None, trace_id: str | None = None) -> None:
        self._client = client
        self._root = root
        self.trace_id = trace_id
        self._root_metadata: dict[str, Any] = {}
        self.finished = False

    def span(self, name: str, as_type: str = "span", **fields: Any) -> Span:
        """A child of the root (taken from the current context: the root is attached there)."""
        if self._client is None:
            return Span()
        return Span(_safe(self._client.start_observation, name=name, as_type=as_type, **fields))

    def annotate(self, **metadata: Any) -> None:
        """Root metadata, written when the trace finishes (e.g. condense_strategy)."""
        self._root_metadata.update(metadata)

    def finish(self, output: Any = None, outcome: str = "ok", error: BaseException | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        if self._root is None:
            return
        fields: dict[str, Any] = {"metadata": {"outcome": outcome, **self._root_metadata}}
        if output is not None:
            fields["output"] = output
        if error is not None:
            # Class name only: an exception message could contain user text.
            fields.update(level="ERROR", status_message=type(error).__name__)
        elif outcome not in ("ok", "no_context"):
            fields.update(level="WARNING", status_message=outcome)
        _safe(self._root.update, **fields)


NOOP_TRACE = RequestTrace()


def current_trace() -> RequestTrace:
    """The trace of the request being handled, or a no-op (tests, scripts, tracing off)."""
    ctx = current()
    trace = getattr(ctx, "trace", None) if ctx else None
    return trace if trace is not None else NOOP_TRACE


# ---------------------------------------------------------------------------
# Tracers
# ---------------------------------------------------------------------------


class NoopTracer:
    enabled = False

    @contextmanager
    def request_trace(self, name: str, session_id: str, request_id: str | None, question: str) -> Iterator[RequestTrace]:
        yield NOOP_TRACE

    def score(self, trace_id: str, rating: int, comment: str | None) -> None:
        raise RuntimeError("tracing is disabled")

    def shutdown(self) -> None:
        pass


class LangfuseTracer:
    enabled = True

    def __init__(self, client: Langfuse, settings: Settings, trace_metadata: dict[str, Any]) -> None:
        self.client = client
        self.settings = settings
        self._metadata = trace_metadata

    @contextmanager
    def request_trace(self, name: str, session_id: str, request_id: str | None, question: str) -> Iterator[RequestTrace]:
        from langfuse import propagate_attributes
        from opentelemetry import context as otel_context

        ctx = current()
        error: BaseException | None = None
        trace = NOOP_TRACE
        # Start from an EMPTY context: FastAPI creates its own OpenTelemetry span per request
        # ("fastapi.endpoint"), which Langfuse does not export. As its child, our root would
        # point at a missing parent and the trace tree would break in the Langfuse UI.
        empty_token = otel_context.attach(otel_context.Context())
        try:
            attrs = propagate_attributes(
                session_id=session_id,
                user_id=None,  # set this once the app has users
                trace_name=name,
                version=self._metadata["prompt_version"],
                tags=[self.settings.llm_provider],
                metadata={**self._metadata, "request_id": request_id or ""},
            )
            attrs.__enter__()
        except Exception as exc:  # e.g. an invalid attribute: answer the request untraced
            logger.warning("tracing_error", error=type(exc).__name__, stage="propagate_attributes")
            otel_context.detach(empty_token)
            yield NOOP_TRACE
            return
        try:
            try:
                root_cm = self.client.start_as_current_observation(name=name, as_type="span", input={"question": question})
                root = root_cm.__enter__()
            except Exception as exc:
                logger.warning("tracing_error", error=type(exc).__name__, stage="root_span")
                yield NOOP_TRACE
                return
            # An unsampled trace (LANGFUSE_SAMPLE_RATE < 1) is never sent: no trace_id to hand out.
            sampled = bool(getattr(root, "_otel_span", None) and root._otel_span.is_recording())
            trace = RequestTrace(self.client, root, root.trace_id if sampled else None)
            if ctx is not None:
                ctx.trace = trace
                ctx.trace_id = trace.trace_id
            try:
                yield trace
            except BaseException as exc:  # recorded by class name, re-raised below
                error = exc
            if not trace.finished:
                if error is not None:  # dependency_unavailable / timeout, else a plain "error"
                    outcome = getattr(error, "code", None) or "error"
                else:
                    outcome = ctx.outcome if ctx and ctx.outcome else "ok"
                trace.finish(outcome=outcome, error=error)
            # Never pass the exception into the SDK: OpenTelemetry would record its message.
            _safe(root_cm.__exit__, None, None, None)
        finally:
            _safe(attrs.__exit__, None, None, None)
            _safe(otel_context.detach, empty_token)
            if ctx is not None:
                ctx.trace = None
        if error is not None:
            raise error

    def score(self, trace_id: str, rating: int, comment: str | None) -> None:
        self.client.create_score(
            name=FEEDBACK_SCORE_NAME,
            value=float(rating),
            data_type="NUMERIC",
            trace_id=trace_id,
            # Same id for the same trace: changing 👍 to 👎, or adding a comment, replaces the score.
            score_id=f"{trace_id}-{FEEDBACK_SCORE_NAME}",
            comment=mask_text(comment) if comment else None,  # scores bypass the SDK mask
        )

    def shutdown(self) -> None:
        self.client.shutdown()  # flushes buffered spans and scores


Tracer = NoopTracer | LangfuseTracer


def trace_metadata(settings: Settings) -> dict[str, Any]:
    """Trace-level metadata. Each value must stay under 200 characters (SDK limit)."""
    from app.rag import PROMPT_SHA, PROMPT_VERSION

    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha": PROMPT_SHA,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model_name,
        "embedding_model": settings.embedding_model,
        "index": settings.pinecone_index_name,
        "k": settings.retriever_k,
        "min_score": settings.min_relevance_score,
    }


def build_tracer(settings: Settings, span_exporter: Any = None) -> Tracer:
    if not settings.tracing_enabled:
        return NoopTracer()
    from langfuse import Langfuse

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value() if settings.langfuse_secret_key else None,
        base_url=settings.langfuse_base_url,
        tracing_enabled=True,
        environment=settings.langfuse_environment,
        sample_rate=settings.langfuse_sample_rate,
        mask=mask_pii,
        span_exporter=span_exporter,  # tests: InMemorySpanExporter (no network)
    )
    return LangfuseTracer(client, settings, trace_metadata(settings))
