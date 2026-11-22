"""Pydantic request/response models shared by the API, the RAG layer and tests."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator
from typing_extensions import Annotated

MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChatRequest(BaseModel):
    message: MessageText = Field(description="The user's question.")
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Conversation id. Omit to start a new conversation.",
    )


class Citation(BaseModel):
    id: int = Field(description="Number used for the inline [n] marker in the answer.")
    source: str = Field(description="Source document file name.")
    page: int | None = Field(default=None, description="1-based page number.")
    snippet: str = Field(description="Excerpt of the retrieved chunk.")
    score: float | None = Field(default=None, description="Similarity score from the vector store.")
    cited: bool = Field(default=False, description="Whether the answer referenced this source.")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation]
    trace_id: str | None = Field(
        default=None,
        description="Langfuse trace of this answer (for POST /api/feedback). Null when tracing is off.",
    )


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=_now)
    # Assistant turns only: the Langfuse trace, so restored history can still be rated.
    trace_id: str | None = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[Message]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm_provider: str
    llm_model: str
    index: str
    memory_backend: str


class CheckResult(BaseModel):
    ok: bool
    latency_ms: float
    error: str | None = Field(
        default=None,
        description="'timed out after Ns' or the exception class name. Details are only logged.",
    )


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckResult] = Field(description="Keyed by dependency: redis (or memory), pinecone.")


class FeedbackRequest(BaseModel):
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$", description="trace_id from a chat response.")
    rating: Literal[1, -1] = Field(description="1 = helpful (thumbs up), -1 = not helpful.")
    comment: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] | None = None

    @field_validator("comment")
    @classmethod
    def _empty_is_none(cls, value: str | None) -> str | None:
        return value or None


class FeedbackResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
