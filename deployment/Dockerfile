# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
# ChronoGrid low-level model diagnostics and explicitly authorized legacy replay.
# The preferred GUI/journal deployment uses Dockerfile.operations.
FROM python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1 AS locked-wheels
COPY deployment/requirements-service.lock /requirements.lock
RUN python -m pip download --no-cache-dir --only-binary=:all: --require-hashes -r /requirements.lock --dest /wheels
FROM python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    DATABASE_URL=sqlite:////state/registry.sqlite \
    DDIR_STATE_FILE=/state/ddir_state.json DQCP_STATE_FILE=/state/dqcp_aci_state.json
WORKDIR /app
COPY deployment/requirements-service.lock /app/requirements.lock
RUN --mount=type=bind,from=locked-wheels,source=/wheels,target=/wheels,readonly \
    pip install --no-index --find-links=/wheels --require-hashes -r requirements.lock && pip check
COPY src/ /app/src/
COPY artifacts/models/ /app/artifacts/models/
COPY artifacts/calibration/ /app/artifacts/calibration/
RUN useradd -u 10001 -m chronogrid && mkdir /state && chown chronogrid:chronogrid /state
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=15s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"
CMD ["uvicorn","src.serving.app:app","--host","0.0.0.0","--port","8000","--workers","1"]
