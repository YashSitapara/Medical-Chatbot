FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /srv

# Dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the embedding model into the image so containers start without downloading it.
# Keep this in sync with the runtime EMBEDDING_MODEL (.env); a different model still works
# but is downloaded at startup (and the index must have been built with it).
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY app ./app
COPY scripts ./scripts
COPY frontend ./frontend

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /srv /opt/hf-cache
USER appuser

EXPOSE 8000
# /ready pings Redis and Pinecone (each limited to READY_TIMEOUT_SECONDS, default 5s, run
# concurrently). --timeout must stay above that plus Python startup, or Docker kills the
# probe before /ready can answer. A 503 makes urlopen raise, so the probe exits non-zero.
HEALTHCHECK --interval=30s --timeout=12s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=10).status == 200 else 1)"

# Workers per container = $WEB_CONCURRENCY (uvicorn reads it; default 1).
# Use >1 (or several containers) only with MEMORY_BACKEND=redis, otherwise each worker
# has its own chat history. Each worker loads its own embedding model (~0.5 GB RAM with torch).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
