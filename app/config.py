"""Application settings, loaded from environment variables / `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Vector store ---
    pinecone_api_key: SecretStr
    pinecone_index_name: str = "medical-chatbot"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Embeddings (dimension is measured from the model, see app/embeddings.py) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM ---
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_temperature: float = 0.0
    ollama_model: str = "gemma3:4b"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"

    # --- Retrieval ---
    retriever_k: int = Field(default=4, ge=1, le=20)
    # Chunks scoring below this cosine similarity are discarded; if none remain the
    # question is treated as out of scope and the LLM is not called. Model-specific:
    # for all-MiniLM-L6-v2 on the medical book, on-topic hits score ~0.48-0.76 and
    # off-topic questions ~0.18-0.32. Set to 0 to disable.
    min_relevance_score: float = Field(default=0.40, ge=0.0, le=1.0)

    # --- Chat memory ---
    memory_max_turns: int = Field(default=10, ge=1)
    memory_ttl_seconds: int = Field(default=3600, ge=60)
    # "memory": per-process dict (single worker only). "redis": shared by every worker/container.
    memory_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "medical-rag:chat:"

    # --- Timeouts and retries ---
    # LLM: an httpx *read* timeout (no bytes for N seconds), so long streamed answers are fine.
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)  # OpenAI only; Ollama is local
    condense_timeout_seconds: float = Field(default=10.0, gt=0)  # on timeout: dual-retrieval fallback
    retrieval_timeout_seconds: float = Field(default=5.0, gt=0)  # per Pinecone attempt
    retrieval_max_attempts: int = Field(default=3, ge=1, le=10)
    request_timeout_seconds: float = Field(default=45.0, gt=0)  # overall cap: 504 / SSE timeout event

    # --- Rate limiting (/api/chat, /api/chat/stream) ---
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = Field(default=10, ge=1)  # per session
    # Per client IP. Higher than the per-session limit so several people behind one NAT can
    # chat, but it still stops a client that sends a new session_id with every request.
    rate_limit_per_ip_per_minute: int = Field(default=30, ge=1)

    # --- Readiness (/ready) ---
    # Time limit for EACH dependency check (Redis ping, Pinecone stats); they run concurrently.
    ready_timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    # --- Tracing (Langfuse) ---
    # Off by default so tests and a fresh clone run without keys. When on, each chat request
    # becomes one Langfuse trace (question, rewrite, chunks, full prompt, answer), with phone
    # numbers / emails / Aadhaar / PAN masked first (app/tracing.py: mask_pii).
    tracing_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    # EU region. US: https://us.cloud.langfuse.com
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = Field(default="development", pattern=r"^[a-z0-9_-]{1,40}$")
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    langfuse_retention_days: int = Field(default=30, ge=1)  # scripts/langfuse_retention.py
    feedback_rate_limit_per_ip_per_minute: int = Field(default=30, ge=1)

    # --- Server ---
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"
    # "json" (one JSON object per line, for log collectors) or "console" (readable, for a terminal).
    log_format: Literal["json", "console"] = "json"
    # LOCAL DEVELOPMENT ONLY. Adds the user's question and the rewritten question to the logs.
    # Off: logs carry metadata only (lengths, scores, latencies), never question text.
    log_content: bool = False

    @model_validator(mode="after")
    def _check_provider_credentials(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.tracing_enabled and not (self.langfuse_public_key and self.langfuse_secret_key):
            raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required when TRACING_ENABLED=true")
        return self

    @property
    def llm_model_name(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.openai_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
