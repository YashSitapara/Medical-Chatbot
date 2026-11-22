"""JSON logging via structlog, with request context and redaction.

Every log line - ours and third-party (uvicorn, redis, langchain...) - goes through the
same processors, so request IDs are attached and secrets/user text are redacted
everywhere, including in exception tracebacks.
"""

import logging
import os
import re
import sys
from typing import Any, TextIO

import structlog

from app.config import Settings
from app.request_context import current

# Fields that may only be logged when LOG_CONTENT=true (local development).
CONTENT_FIELDS = frozenset({"question", "condensed_question", "rewrite_raw"})
# Never logged in any mode.
ALWAYS_DROPPED_FIELDS = frozenset({"answer"})
# Whole key names whose values are secrets ("input_tokens" is not matched: it is a count).
_SECRET_KEY = re.compile(r"(api_?key|apikey|password|passwd|secret|token|authorization|auth)", re.IGNORECASE)
_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"pcsk_[A-Za-z0-9_\-]+"), "[REDACTED]"),                      # Pinecone keys
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "[REDACTED]"),                  # OpenAI / Langfuse secret keys
    (re.compile(r"\bpk-lf-[A-Za-z0-9_\-]+"), "[REDACTED]"),                    # Langfuse public keys
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]+=*"), "Bearer [REDACTED]"),
    (re.compile(r"\b([a-z][a-z0-9+.\-]*://[^:/@\s]*:)[^@\s]+@", re.IGNORECASE), r"\1***@"),  # URL passwords
    (re.compile(r"(?i)\b(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\s\"',;}]+"), r"\1[REDACTED]"),
]
_HANDLER_MARK = "_medical_rag_handler"


def _merge_request_context(_, __, event_dict: dict) -> dict:
    if ctx := current():
        event_dict.setdefault("request_id", ctx.request_id)
        if ctx.session_id:
            event_dict.setdefault("session_id", ctx.session_id)
        if ctx.trace_id:
            event_dict.setdefault("trace_id", ctx.trace_id)  # links this log line to Langfuse
    return event_dict


def _add_pid(_, __, event_dict: dict) -> dict:
    event_dict["pid"] = os.getpid()  # identifies the worker with WEB_CONCURRENCY > 1
    return event_dict


class Redactor:
    """structlog processor: removes secrets always, and user text unless log_content."""

    def __init__(self, log_content: bool, secret_values: list[str]) -> None:
        self.log_content = log_content
        self.secret_values = sorted({s for s in secret_values if s and len(s) >= 8}, key=len, reverse=True)

    def _clean_str(self, text: str, sensitive: list[str]) -> str:
        for value in self.secret_values:
            text = text.replace(value, "[REDACTED]")
        for pattern, repl in _SECRET_PATTERNS:
            text = pattern.sub(repl, text)
        for value in sensitive:
            text = text.replace(value, "[question]")
        return text

    def _clean(self, value: Any, sensitive: list[str]) -> Any:
        if isinstance(value, str):
            return self._clean_str(value, sensitive)
        if isinstance(value, dict):
            return {k: self._clean(v, sensitive) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(v, sensitive) for v in value]
        return value

    def __call__(self, _, __, event_dict: dict) -> dict:
        ctx = current()
        # Longest first so a question isn't half-replaced by a shorter substring of it.
        sensitive = [] if self.log_content or ctx is None else sorted(ctx.sensitive, key=len, reverse=True)
        out = {}
        for key, value in event_dict.items():
            if key in ALWAYS_DROPPED_FIELDS:
                continue
            if key in CONTENT_FIELDS and not self.log_content:
                continue
            if _SECRET_KEY.fullmatch(key):
                out[key] = "[REDACTED]"
                continue
            out[key] = value if key in CONTENT_FIELDS else self._clean(value, sensitive)
        return out


def configure_logging(settings: Settings, stream: TextIO | None = None) -> None:
    """Idempotent: replaces only the handler this function installed (pytest's caplog survives)."""
    shared: list = [
        _merge_request_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_pid,
    ]
    structlog.configure(
        processors=[structlog.stdlib.filter_by_level, *shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    secrets = [settings.pinecone_api_key.get_secret_value()]
    if settings.openai_api_key:
        secrets.append(settings.openai_api_key.get_secret_value())
    if settings.langfuse_secret_key:
        secrets.append(settings.langfuse_secret_key.get_secret_value())
    renderer = (
        structlog.dev.ConsoleRenderer()
        if settings.log_format == "console"
        else structlog.processors.JSONRenderer(ensure_ascii=False)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,  # stdlib records (uvicorn, redis, ...) get the same fields
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,  # traceback -> "exception" field, then redacted
            Redactor(settings.log_content, secrets),
            renderer,
        ],
    )

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)
    setattr(handler, _HANDLER_MARK, True)

    root = logging.getLogger()
    for old in [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]:
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # uvicorn installs its own plain-text handlers; route its logs through ours instead.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    access = logging.getLogger("uvicorn.access")  # replaced by our `request_finished` event
    access.handlers.clear()
    access.propagate = False
    access.disabled = True
    # One line per HTTP call to Ollama/Pinecone otherwise.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
