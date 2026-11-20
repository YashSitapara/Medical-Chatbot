"""Error types and how they map to HTTP.

* DependencyUnavailableError -> 503: a dependency is down, timed out or returned 5xx
  (after any retries). The client may try again shortly.
* RequestTimeoutError        -> 504: the overall per-request time cap was hit.
* anything else              -> 500: a bug in our code (or our configuration, e.g. a
  rejected API key). Handled by RequestContextMiddleware.

Every error body has the same shape: {"detail", "code", "request_id"}.
"""


class DependencyUnavailableError(RuntimeError):
    """A dependency failed in a way that may be temporary."""

    code = "dependency_unavailable"
    public_detail = "The assistant is temporarily unavailable. Please try again shortly."


class LLMUnavailableError(DependencyUnavailableError):
    pass


class RetrievalUnavailableError(DependencyUnavailableError):
    pass


class RequestTimeoutError(RuntimeError):
    """The request exceeded REQUEST_TIMEOUT_SECONDS."""

    code = "timeout"
    public_detail = "The request took too long. Please try again."


def is_transient_llm_error(exc: BaseException) -> bool:
    """Timeouts, connection failures and 5xx from the LLM provider (OpenAI or Ollama).

    OpenAI's RateLimitError (429) counts too: it means *their* quota/overload, not a bad
    request from us. Other provider errors (401 bad key, 400 bad request) are not transient.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import httpx

        if isinstance(exc, httpx.TransportError):  # timeouts, connect/read errors (Ollama)
            return True
    except ImportError:  # pragma: no cover
        pass
    try:
        import openai

        if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError)):
            return True  # APITimeoutError is a subclass of APIConnectionError
    except ImportError:  # pragma: no cover
        pass
    try:
        import ollama

        if isinstance(exc, ollama.ResponseError) and exc.status_code >= 500:
            return True
    except ImportError:  # pragma: no cover
        pass
    return False


def is_transient_pinecone_error(exc: BaseException) -> bool:
    """Retry timeouts, connection errors and 5xx. Never 4xx: the request itself is wrong."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    from urllib3 import exceptions as u3

    if isinstance(exc, (u3.TimeoutError, u3.ProtocolError, u3.MaxRetryError, u3.NewConnectionError)):
        return True
    from pinecone.exceptions import PineconeApiException, PineconeProtocolError

    if isinstance(exc, PineconeProtocolError):
        return True
    if isinstance(exc, PineconeApiException):  # includes ServiceException (5xx), Unauthorized (401)...
        return (exc.status or 0) >= 500
    return False
