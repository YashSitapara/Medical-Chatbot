"""Readiness checks for GET /ready: can this instance serve a chat request right now?

`/health` says the process is up and its chat memory answers. `/ready` additionally checks
Pinecone, and gives every dependency a hard time limit so a hung dependency turns into a
quick 503 instead of a hung healthcheck.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

import structlog

from app.memory import ChatMemory
from app.rag import RAGService
from app.schemas import CheckResult, ReadyResponse

logger = structlog.get_logger(__name__)


async def run_check(name: str, check: Callable[[], Awaitable[object]], timeout: float) -> CheckResult:
    """Run one check with a time limit. Never raises: failures become ok=False.

    The response only carries a short reason (timeout or exception class name); the full
    error, which may include hostnames, goes to the log. /ready is publicly reachable.
    """
    start = time.perf_counter()
    try:
        await asyncio.wait_for(check(), timeout)
        error = None
    except TimeoutError:
        error = f"timed out after {timeout:g}s"
        logger.warning("readiness_check_failed", check=name, error=error)
    except Exception as exc:
        error = type(exc).__name__
        logger.warning("readiness_check_failed", check=name, error=error, detail=repr(exc))
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return CheckResult(ok=error is None, latency_ms=latency_ms, error=error)


async def check_readiness(memory: ChatMemory, rag: RAGService, timeout: float) -> ReadyResponse:
    """Check chat memory and Pinecone concurrently, so the worst case is ~timeout, not 2x."""
    names = [memory.backend, "pinecone"]  # "redis" or "memory"
    results = await asyncio.gather(
        run_check(names[0], memory.ping, timeout),
        # The Pinecone SDK is blocking: run it in a thread. It also gets its own HTTP
        # timeout, because a thread can't be cancelled when wait_for gives up on it.
        run_check(names[1], lambda: asyncio.to_thread(rag.ping_vectorstore, timeout), timeout),
    )
    checks = dict(zip(names, results))
    status = "ready" if all(c.ok for c in checks.values()) else "not_ready"
    return ReadyResponse(status=status, checks=checks)
