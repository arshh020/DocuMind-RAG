# Serves the retrieval + generation API. The Streamlit UI runs from the same
# image with a different command (see docker-compose.yml).
FROM python:3.11-slim

# Keeps logs unbuffered so `docker logs` shows tracebacks immediately, and stops
# .pyc files being written into the image layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /srv

# Dependency metadata first so the layer cache survives source edits.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[api,ui]"

COPY app ./app
COPY scripts ./scripts

# Belt and braces: the package is installed above, and PYTHONPATH also covers
# running the scripts directly inside the container.
ENV PYTHONPATH=/srv/src

# The index is DATA, not code. Building it needs the corpus and an embedding
# model, so mount it at runtime rather than baking a stale copy into the image:
#   docker run -v "$PWD/artifacts:/srv/artifacts" ...
# The API returns 503 on /retrieve when the index is missing, which is the
# correct behavior for a container that is up but not yet fed.
VOLUME ["/srv/artifacts"]

EXPOSE 8000

# A slim image has no curl, so probe with Python instead of adding a package.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=4).status==200 else 1)"

# Shell form so ${PORT} is expanded by the platform (Fly.io, Render, Spaces all
# inject their own PORT).
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT}
